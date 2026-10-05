# Author: Finn Fassbender
# Last modified: 2025-10-30

# Description: This script processes the HiRID data and stores it in a structured format for further
# processing and harmonization.


import glob
import os
import sys
from pathlib import Path

import polars as pl
import time

from ..A_extract.A_extract_hirid import HiRIDExtractor
from ..helper import GlobalHelpers
from ..helper_conversions import UnitConverter


class HiRIDProcessor(HiRIDExtractor):
    def __init__(self, paths):
        """
        Initializes the HiRIDProcessor instance.

        Args:
            paths: Object containing various source and destination paths.

        Sets:
            path: Source path for HiRID data ({hirid_source_path}).
            helpers: Instance of GlobalHelpers.
            convert: Instance of HiRIDConverter.
            index_cols: List containing {icu_stay_id_col} and {timeseries_time_col}.
        """
        super().__init__(paths)
        self.path = paths.hirid_source_path
        self.helpers = GlobalHelpers()
        self.convert = HiRIDConverter()
        self.index_cols = [self.icu_stay_id_col, self.timeseries_time_col]
        self.admissiontime = (
            self._extract_admissions()
            .select(self.icu_stay_id_col, "admissiontime")
            .cast({"admissiontime": str})
        )
        self.length_of_stay = self._extract_length_of_stay()

    def _get_labname_components(self) -> pl.LazyFrame:
        """
        Retrieves the lab names from the HiRID dataset.

        Returns:
            pl.LazyFrame: LazyFrame containing the LOINC data.
        """
        data = (
            pl.scan_parquet(self.timeseries_path + "*.parquet")
            .select("variableid")
            .unique()
            .join(self._get_observation_variables(), on="variableid")
            .select("variable")
        )

        # Remove duplicates and sort the LOINC data
        labnames = data.collect().to_series().to_list()
        return data.with_columns(
            pl.col("variable")
            .replace_strict(
                self.omop.get_lab_component_from_name(labnames),
                default=None,
            )
            .alias("LOINC_component")
        ).unique()

    # region time series
    def process_timeseries(self) -> pl.LazyFrame:
        """
        Process non-laboratory time series data.

        Steps:
            1. Check for preprocessed non-lab timeseries file; load if available.
            2. For each raw timeseries file: filter to non-lab variables, extract and pivot measurements.
            3. Combine all files into single wide-format dataset.
            4. Sort by index columns and save.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time offset (seconds from ICU admission).
                - Non-laboratory measurement columns (pivoted from variable).
        """
        ts_path = self.precalc_path + "HiRID_timeseries.parquet"

        if os.path.isfile(ts_path):
            # Load the preprocessed data
            return pl.scan_parquet(ts_path).select(
                pl.col(self.index_cols).set_sorted(),
                pl.exclude(self.index_cols),
            )

        print("HiRID   - Processing timeseries data...")

        # Since each case has its data in only one file, iterating over the files specifically
        # allows for a more efficient processing of the data.
        files = sorted(list(Path(self.timeseries_path).glob("*.parquet")))
        total_files = len(files)
        batch_size = 10
        total_batches = (total_files + batch_size - 1) // batch_size

        times, cases = [], 0
        batch_id = "HiRID_ts_batch"
        for i in range(0, total_files, batch_size):
            start = time.time()
            index = str(i // batch_size).zfill(4)

            batch_paths = files[i : i + batch_size]

            # Process timeseries data
            timeseries = (
                # Drop the lab values from the timeseries data
                pl.scan_parquet(batch_paths)
                .filter(~pl.col("variableid").is_between(20000000, 25000000))
                .pipe(
                    self._extract_timeseries_helper,
                    self.admissiontime,
                    self.length_of_stay,
                )
            )

            (
                # Pivot the timeseries data
                self.pivot_numeric_or_string(
                    timeseries,
                    dataset="HiRID_ts",
                    on_col="variable",
                    index_cols=self.index_cols,
                    numeric_col="value",
                ).sort(self.index_cols)
                # Sink each intermittent batch result to a parquet
                .sink_parquet(self.precalc_path + f"{batch_id}_{index}.parquet")
            )

            # Update timing information
            elapsed = time.time() - start
            times.append(elapsed)
            avg = sum(times) / len(times)
            eta_min = int(avg * (total_batches - (i // batch_size) - 1) / 60 + 0.5) # fmt: skip

            cases += (
                timeseries.select(self.icu_stay_id_col)
                .unique()
                .collect()
                .shape[0]
            )

            sys.stdout.write("\033[K")  # Clear to the end of line
            print(
                f"Processing batch {i//batch_size + 1:3.0f} of {total_batches} "
                f"with {len(batch_paths):4.0f} files ({cases:5.0f} cases) "
                f"(last: {elapsed:.2f}s, avg: {avg:.2f}s, ETA: {eta_min:d} min)",
                end="\r",
            )

        print("\nBatch processing complete. Concatenating results...")

        # Concatenate and sink all processed frames
        temp_files = sorted(
            glob.glob(self.precalc_path + f"{batch_id}_*.parquet")
        )
        batch_frames = [pl.scan_parquet(file) for file in temp_files]
        pl.concat(batch_frames, how="diagonal_relaxed").sink_parquet(ts_path)

        # Delete the temporary files
        for file in temp_files:
            os.remove(file)

        # Load the preprocessed data
        return pl.scan_parquet(ts_path).select(
            pl.col(self.index_cols).set_sorted(),
            pl.exclude(self.index_cols),
        )

    # endregion

    # region timeseries labs
    def process_timeseries_labs(self) -> pl.LazyFrame:
        """
        Process laboratory time series data.

        Steps:
            1. Check for preprocessed labs file; load if available.
            2. For each raw timeseries file: filter to lab variables, extract measurements.
            3. Combine all files, convert lab values to canonical units.
            4. Apply LOINC component mapping and JSON encode structured fields.
            5. Save extracted data to temporary file before batch pivoting.
            6. Batch pivot data on variable name to create wide-format dataset.
            7. Apply post-pivot unit conversions for derived measurements.
            8. Save, sort by index columns, and clean temporary files.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time offset (seconds from ICU admission).
                - Laboratory measurement columns (pivoted from variable, JSON-encoded).
        """
        ts_labs_path = self.precalc_path + "HiRID_timeseries_labs.parquet"
        ts_labs_path_unsorted = self.precalc_path + "HiRID_ts_labs.parquet"

        if os.path.isfile(ts_labs_path):
            # Load the preprocessed data
            return pl.scan_parquet(ts_labs_path).select(
                pl.col(self.index_cols).set_sorted(),
                pl.exclude(self.index_cols),
            )

        print("HiRID   - Collecting lab time series data...")
        labname_components = self._get_labname_components()

        # Process timeseries data
        (
            pl.scan_parquet(self.timeseries_path + "*.parquet")
            # Keep the lab values from the timeseries data
            .filter(pl.col("variableid").is_between(20000000, 25000000))
            .pipe(
                self._extract_timeseries_helper,
                self.admissiontime,
                self.length_of_stay,
            )
            # Drop the non-lab values from the timeseries data
            .join(labname_components, on="variable")
            .filter(pl.col("LOINC_component").is_not_null())
            # Extract and transform lab data
            .pipe(self._extract_timeseries_labs_helper)
            # Convert the lab values to the correct units
            .pipe(
                self.convert._convert_lab_values,
                labelcol="variable",
                valuecol="labstruct",
            )
            # Replace the LOINC codes
            .pipe(
                self.convert._assign_LOINC_codes,
                self.omop,
                self.index_cols,
                struct_cols=["labstruct"],
                component_col="variable",
            )
            # JSON encode the structs for pivoting
            .with_columns(pl.col("labstruct").struct.json_encode())
            # Pivot the lab data
            .pipe(
                self.pivot_numeric_or_string,
                dataset="HiRID_labs",
                on_col="variable",
                index_cols=self.index_cols,
                string_col="labstruct",
            )
            # Save extracted data before pivoting
            .sink_parquet(ts_labs_path_unsorted)
        )

        # Sort the data
        (
            pl.scan_parquet(ts_labs_path_unsorted)
            .sort(self.index_cols)
            .sink_parquet(ts_labs_path)
        )
        os.remove(ts_labs_path_unsorted)

        # Load the preprocessed data
        return pl.scan_parquet(ts_labs_path).select(
            pl.col(self.index_cols).set_sorted(),
            pl.exclude(self.index_cols),
        )

    # endregion

    # region medications
    def process_medications(self) -> pl.LazyFrame:
        """
        Process the medication administrations.

        Steps:
            1. Check for preprocessed medications file; load if available.
            2. Extract the fluids (observation tables) and pharma records in batches of files.
            3. Derive infusion rates, start and end times of the pharma records.
            4. Sink each batch, concatenate them and clean temporary files.

        Returns:
            pl.LazyFrame: Medication administrations in the common medication format.
        """
        medications_path = self.precalc_path + "HiRID_medications.parquet"

        if os.path.isfile(medications_path):
            # Load the preprocessed data
            return pl.scan_parquet(medications_path)

        print("HiRID   - Processing medications...")

        pharmaid_to_ingredient = self._extract_pharmaid_to_ingredient()
        sources = {
            "observation": (
                self.timeseries_path,
                lambda data: data.pipe(
                    self._extract_medications_observation,
                    self.admissiontime,
                    self.length_of_stay,
                ),
            ),
            "pharma": (
                self.pharma_path,
                lambda data: data.pipe(
                    self._extract_medications_pharma,
                    self.admissiontime,
                    self.length_of_stay,
                    pharmaid_to_ingredient,
                ).pipe(self._process_pharma_infusions),
            ),
        }

        # Since each case has its data in only one file, iterating over the files
        # in batches allows for a more efficient processing of the data.
        batch_size = 10
        for name, (folder, method) in sources.items():
            files = sorted(Path(folder).glob("*.parquet"))
            total_batches = (len(files) + batch_size - 1) // batch_size
            for i in range(0, len(files), batch_size):
                start = time.time()
                index = str(i // batch_size).zfill(4)

                # Sink each intermittent batch result to a parquet
                pl.scan_parquet(files[i : i + batch_size]).pipe(method).sink_parquet(
                    self.precalc_path
                    + f"HiRID_medications_{name}_batch_{index}.parquet"
                )

                sys.stdout.write("\033[K")  # Clear to the end of line
                print(
                    f"Processing {name} batch {i//batch_size + 1:3.0f} of "
                    f"{total_batches} (last: {time.time() - start:.2f}s)",
                    end="\r",
                )

        print("\nBatch processing complete. Concatenating results...")

        # Concatenate and sink all processed batches
        batch_files = sorted(
            glob.glob(self.precalc_path + "HiRID_medications_*_batch_*.parquet")
        )
        pl.concat(
            [pl.scan_parquet(file) for file in batch_files],
            how="diagonal_relaxed",
        ).sink_parquet(medications_path)

        # Delete the temporary files
        for file in batch_files:
            os.remove(file)

        return pl.scan_parquet(medications_path)

    def _process_pharma_infusions(self, pharma: pl.LazyFrame) -> pl.LazyFrame:
        """
        Derive infusion rates, start and end times from the pharma log entries
        of complete cases.

        Arguments:
            pharma: Pharma records with all log entries of each included case.

        Returns:
            pl.LazyFrame: One row per infusion or administration, with rates,
                start and end time.
        """
        # Get infusion duration where possible, by checking whether the drugname reappears
        # on next log entry (as determined by a different offset)
        # 1. Get list of log entry offsets for each patient
        pharma_offsets = (
            pharma.select(
                self.icu_stay_id_col,
                self.drug_end_col,
                self.drug_mixture_id_col,
            )
            .unique()
            .sort(
                self.icu_stay_id_col,
                self.drug_mixture_id_col,
                self.drug_end_col,
            )
            .with_columns(
                pl.col(self.drug_end_col)
                .shift(1)
                .over(self.icu_stay_id_col, self.drug_mixture_id_col)
                .alias("prev_drug_end"),
                pl.col(self.drug_end_col)
                .shift(-1)
                .over(self.icu_stay_id_col, self.drug_mixture_id_col)
                .alias("next_drug_end"),
            )
        )

        pharma = (
            pharma.join(
                pharma_offsets,
                on=[
                    self.icu_stay_id_col,
                    self.drug_end_col,
                    self.drug_mixture_id_col,
                ],
                how="left",
            )
            # Sort by patient ID, drug name and drug start time
            .sort(
                self.icu_stay_id_col,
                self.drug_name_col,
                self.drug_mixture_id_col,
                self.drug_end_col,
                "prev_drug_end",  # sometimes, there is the same drug given twice at the same time
                # all other columns as tie-breakers for a deterministic order
                pl.exclude(
                    self.icu_stay_id_col,
                    self.drug_name_col,
                    self.drug_mixture_id_col,
                    self.drug_end_col,
                    "prev_drug_end",
                ),
            )
            # NOTE: Convert drug_amount to drug_rates, fluid_amount to fluid_rates
            .with_columns(
                (
                    pl.col(self.drug_amount_col)
                    / (pl.col(self.drug_end_col) - pl.col("prev_drug_end"))
                    * 3600
                )
                .round_sig_figs(2)
                .alias(self.drug_rate_col),
                pl.col(self.drug_amount_unit_col).str.replace("µ", "mc"),
                (pl.col(self.drug_amount_unit_col) + pl.lit("/hr"))
                .str.replace("µ", "mc")
                .alias(self.drug_rate_unit_col),
                (
                    pl.col(self.fluid_amount_col)
                    / (pl.col(self.drug_end_col) - pl.col("prev_drug_end"))
                    * 3600
                )
                .round_sig_figs(2)
                .alias(self.fluid_rate_col),
            )
            # 2. Check if drug is continued from the previous log entry
            #    and if it is continued in the next log entry
            .with_columns(
                # Check if drug is continued from the previous log entry
                pl.when(pl.col("prev_drug_end").is_not_null())
                .then(
                    pl.when(
                        # Check if the previous drug is the same as the current drug
                        pl.col(self.drug_name_col)
                        == pl.col(self.drug_name_col).shift(1),
                        # Check if the previous drug end time is the previous log entry time
                        pl.col("prev_drug_end")
                        == pl.col(self.drug_end_col).shift(1),
                        # Check if the drug rate is the same as the previous drug rate
                        pl.col(self.drug_rate_col)
                        == pl.col(self.drug_rate_col).shift(1),
                    )
                    .then(pl.lit("continued"))
                    .otherwise(pl.lit("started"))
                )
                .otherwise(None)
                .alias("drug_status_prev"),
                # Check if drug is continued in the next log entry
                pl.when(
                    # Check if the next drug is the same as the current drug
                    pl.col(self.drug_name_col)
                    == pl.col(self.drug_name_col).shift(-1),
                    # Check if the next drug end time is the next log entry time
                    pl.col("next_drug_end")
                    == pl.col(self.drug_end_col).shift(-1),
                    # Check if the drug rate is the same as the next drug rate
                    pl.col(self.drug_rate_col)
                    == pl.col(self.drug_rate_col).shift(-1),
                )
                .then(pl.lit("continued"))
                .otherwise(pl.lit("discontinued"))
                .alias("drug_status_next"),
            )
            # Filter for rows where the drug status changes
            .filter(pl.col("drug_status_prev") != pl.col("drug_status_next"))
            # 3. Get the end time of the drug if it is discontinued
            .with_columns(
                pl.when(pl.col("drug_status_next") == "discontinued")
                .then("prev_drug_end")
                .otherwise(None)
                .alias(self.drug_start_col)
            )
            # 4. Combine rows where the drug is started, continued, then discontinued in the next row
            .with_columns(
                pl.when(
                    pl.col("drug_status_prev").shift(1) == "started",
                    pl.col("drug_status_next").shift(1) == "continued",
                    pl.col("drug_status_prev") == "continued",
                    pl.col("drug_status_next") == "discontinued",
                    # Check if the previous drug is the same as the current drug
                    pl.col(self.drug_name_col)
                    == pl.col(self.drug_name_col).shift(1),
                    # Check if the drug amount is the same as the previous drug amount
                    pl.col(self.drug_rate_col)
                    == pl.col(self.drug_rate_col).shift(1),
                )
                .then(pl.col("prev_drug_end").shift(1))
                .otherwise(pl.col("prev_drug_end"))
                .alias(self.drug_start_col)
            )
            # 5. filter out duplicate rows (same drug, same start time, same rate, different end time)
            .filter(pl.col(self.drug_start_col).is_not_null())
            .sort(
                self.icu_stay_id_col,
                self.drug_name_col,
                self.drug_start_col,
                self.drug_rate_col,
                self.drug_end_col,
                pl.exclude(
                    self.icu_stay_id_col,
                    self.drug_name_col,
                    self.drug_start_col,
                    self.drug_rate_col,
                    self.drug_end_col,
                ),
            )
            .group_by(
                self.icu_stay_id_col,
                self.drug_name_col,
                self.drug_start_col,
                self.drug_rate_col,
                self.drug_admin_type_col,
                maintain_order=True,
            )
            .last()
            # 6. Remove the helper columns
            .drop(
                "prev_drug_end",
                "next_drug_end",
                "drug_status_prev",
                "drug_status_next",
            )
        )

        return pharma

    # endregion


# region convert
class HiRIDConverter(UnitConverter):
    def __init__(self):
        super().__init__()

    def _convert_lab_values(
        self,
        data: pl.LazyFrame,
        labelcol: str = "variableid",
        valuecol: str = "value_struct",
        structfield: str = "value",
    ) -> pl.LazyFrame:
        """Convert raw lab values to canonical units.

        Applies sequential unit conversions for multiple lab tests including:
        bilirubin, creatinine, cortisol, fibrinogen, glucose, hemoglobin, and urea.

        Expected columns:
            - {labelcol}: Lab test identifier.
            - {valuecol}: Lab measurement value.

        Returns:
            pl.LazyFrame: Lab data with unit-converted values.
        """
        return (
            data.pipe(
                self.convert_bilirubin_umol_L_to_mg_dL,
                itemid="Bilirubin.direct",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_bilirubin_umol_L_to_mg_dL,
                itemid="Bilirubin",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_creatinine_umol_L_to_mg_dL,
                itemid="Creatinine",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_cortisol_nmol_L_to_ug_dL,
                itemid="Cortisol",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_g_L_to_mg_dL,
                itemid="Fibrinogen",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_glucose_mmol_L_to_mg_dL,
                itemid="Glucose",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_g_L_to_g_dL,
                itemid="Hemoglobin",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                # same conversion due to definition of MCHC
                self.convert_g_L_to_g_dL,
                itemid="Erythrocyte mean corpuscular hemoglobin concentration",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_urea_nitrogen_from_urea,
                itemid_urea="Urea",
                itemid_BUN="Urea nitrogen",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_blood_urea_nitrogen_mmol_L_to_mg_dL,
                itemid="Urea nitrogen",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
        )


# endregion
