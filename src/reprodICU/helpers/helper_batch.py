import glob
import os
import time
from typing import Optional

import polars as pl


def batch_process_timeseries(
    timeseries: Optional[str] = None,
    input_file: Optional[str] = None,
    output_file: Optional[str] = None,
    save_path: Optional[str] = None,
    tempfiles_path: str = None,
    operation: str = None,
    method: callable = None,
    id_col: str = "Global ICU Stay ID",
    batch_size: int = 5000,
    delete_after: bool = False,
    **kwargs,
) -> None:
    """
    Generic batch processing for timeseries operations (impute or resample).

    Processes large datasets in batches of 5000 patients to avoid memory issues.
    Saves intermediate results to tempfiles_path and concatenates at the end.

    Arguments
    ---------
        timeseries : str, optional
            Type of timeseries to process (e.g., "vitals", "labs"). If None and input_file/output_file
            are provided, a name will be derived from the input file path.
        input_file : str, optional
            Complete path to input file. If None, uses save_path + f"timeseries_{timeseries}.parquet".
        output_file : str, optional
            Complete path to output file. If None, uses save_path + f"timeseries_{timeseries}_{operation}d.parquet".
        save_path : str
            Base path for input and output files (used when input_file/output_file are None).
        tempfiles_path : str
            Path to directory for temporary files.
        operation : str
            Operation type ("impute" or "resample").
        method : callable
            Method to use for the data.
        id_col : str
            Column name for the ID column (default: "Global ICU Stay ID").
        batch_size : int
            Number of patients to process per batch (default: 5000).
        delete_after : bool
            Whether to delete temporary files after processing.
        **kwargs
            Additional keyword arguments passed to the method (e.g., resolution_in_seconds).

    Returns
    -------
        None
            Saves the processed data to output_file (or save_path + output_file if output_file is just a filename)
    """
    # Validate parameters
    if timeseries is None and (input_file is None or output_file is None):
        raise ValueError(
            "Either 'timeseries' must be provided, or both 'input_file' and 'output_file' must be specified."
        )

    # Determine batch identifier for temporary files
    if timeseries is not None:
        batch_id = f"timeseries_{timeseries}"
    else:
        batch_id = os.path.splitext(os.path.basename(input_file))[0]

    # add e to operation for file naming
    if not operation.endswith("e"):
        operation += "e"

    input_path = input_file or save_path + f"{batch_id}.parquet"
    output_path = output_file or save_path + f"{batch_id}_{operation}d.parquet"

    # Get unique ICU stay IDs
    unique_ids = (
        pl.scan_parquet(input_path)
        .select(id_col)
        .unique()
        .collect()
        .to_series()
        .sort()
        .to_list()
    )

    # Process in batches of batch_size patients
    total_batches = (len(unique_ids) + batch_size - 1) // batch_size
    times = []
    for i in range(0, len(unique_ids), batch_size):
        start = time.time()
        batch_ids = unique_ids[i : i + batch_size]
        
        # Filter to current batch and process
        index = str(i // batch_size).zfill(4)
        (
            pl.scan_parquet(input_path)
            .filter(pl.col(id_col).is_in(batch_ids))
            .pipe(method, **kwargs)
            .sort(id_col, "Time Relative to Admission (seconds)")
            .sink_parquet(
                tempfiles_path + f"{batch_id}_{operation}d_{index}.parquet"
            )
        )
        
        # Update timing information
        elapsed = time.time() - start
        times.append(elapsed)
        avg = sum(times) / len(times)
        eta_min = int(avg * (total_batches - (i // batch_size) - 1) / 60 + 0.5)
        
        print(
            f"Processing batch {i//batch_size + 1:3.0f} of {total_batches} "
            f"with {len(batch_ids):4.0f} patients "
            f"(last: {elapsed:.2f}s, avg: {avg:.2f}s, ETA: {eta_min:d} min)",
            end="\r",
        )
        
    print("\nBatch processing complete. Concatenating results...")

    # Concatenate all processed frames and sink
    files = sorted(
        glob.glob(tempfiles_path + f"{batch_id}_{operation}d_*.parquet")
    )
    
    # Read all batch files and concatenate them, handling schema differences
    batch_frames = []
    for file in files:
        batch_frames.append(pl.scan_parquet(file))
    
    pl.concat(batch_frames, how="diagonal_relaxed").sink_parquet(output_path)

    # Optionally delete temporary files
    if delete_after:
        for file in glob.glob(
            tempfiles_path + f"{batch_id}_{operation}d_*.parquet"
        ):
            os.remove(file)
