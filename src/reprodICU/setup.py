"""
Database setup and data preparation functions.

Handles encoding fixes, unpacking, and header normalization for raw source data.
"""

import csv
import gzip
import io
import os
import struct
import zipfile
from pathlib import Path

from config import get_config_manager, reprodICUPaths


def setup_umcdb(config_manager=None, paths=None):
    """
    Fix UMCdb CSV encoding issues.

    UMCdb raw data files come in unicode-escape encoding and need to be
    converted to UTF-8. This function:
    1. Checks if CSV files exist in the umcdb_source folder
    2. Verifies they are in unicode-escape encoding
    3. Re-encodes them to UTF-8 in an "ENCODINGFIX" subfolder
    4. Updates the local config path to point to the ENCODINGFIX folder

    Args:
        config_manager: Optional ConfigManager instance (uses get_config_manager if None)
        paths: Optional reprodICUPaths instance

    Returns:
        Path to the ENCODINGFIX folder

    Raises:
        FileNotFoundError: If umcdb_source folder or CSV files not found
        RuntimeError: If encoding conversion fails
    """
    if config_manager is None:
        config_manager = get_config_manager()
    if paths is None:
        paths = reprodICUPaths(config_manager)

    # Get source path
    source_path = Path(
        config_manager.load_config("PATHS.yaml", user_override=False)[
            "umcdb_source_path"
        ]
    )

    if not source_path.exists():
        raise FileNotFoundError(f"UMCdb source folder not found: {source_path}")

    # Create ENCODINGFIX folder
    encoding_fix_path = source_path / "ENCODINGFIX"
    encoding_fix_path.mkdir(parents=True, exist_ok=True)

    print("UMCdb - Converting encodings from unicode-escape to UTF-8")
    print(f"  Source: {source_path}")
    print(f"  Target: {encoding_fix_path}")

    # Files to convert
    csv_files = [
        "admissions.csv",
        "drugitems.csv",
        "freetextitems.csv",
        "listitems.csv",
        "procedureorderitems.csv",
        "processitems.csv",
    ]

    for filename in csv_files:
        input_file = source_path / filename
        output_file = encoding_fix_path / filename

        if not input_file.exists():
            print(f"!! Skipping {filename} (not found)")
            continue

        try:
            with (
                open(
                    input_file, "r", encoding="unicode-escape", errors="ignore"
                ) as infile,
                open(output_file, "w", encoding="utf-8") as outfile,
            ):
                print(f"  Converting {filename}...")
                reader = csv.reader(infile)
                writer = csv.writer(outfile)
                for row in reader:
                    writer.writerow(row)
        except Exception as e:
            raise RuntimeError(f"Failed to convert {filename}: {e}")

    # Handle zipped numericitems (stream directly from the .zip without extracting)
    zip_file = source_path / "numericitems.zip"
    if zip_file.exists():
        output_gz_file = encoding_fix_path / "numericitems.csv.gz"
        try:

            with zipfile.ZipFile(zip_file, "r") as zf:
                # Find the numericitems.csv entry (allow for possible subfolders)
                members = [name for name in zf.namelist() if name.endswith("numericitems.csv")]
                if not members:
                    raise FileNotFoundError(f"numericitems.csv not found inside {zip_file}")
                member = members[0]

                with zf.open(member) as binary_in, gzip.open(output_gz_file, "wt", encoding="utf-8") as outfile:
                    # Wrap the binary stream to read text with the original encoding
                    with io.TextIOWrapper(binary_in, encoding="unicode-escape", errors="ignore") as infile:
                        print("  Converting numericitems.csv from numericitems.zip...")
                        reader = csv.reader(infile)
                        writer = csv.writer(outfile)
                        for row in reader:
                            writer.writerow(row)
        except Exception as e:
            raise RuntimeError(f"Failed to convert numericitems.csv from {zip_file}: {e}")

    print("-> UMCdb encoding conversion complete")
    
    # Update configuration to point to ENCODINGFIX folder
    config_manager.update_config(
        "PATHS.yaml",
        {"umcdb_source_path": str(encoding_fix_path)},
        user_override=True
    )
    print(f"  Updated config: umcdb_source_path -> {encoding_fix_path}")
    
    return str(encoding_fix_path)


def setup_sicdb(config_manager=None, paths=None):
    """
    Unpack SICdb raw data.

    SICdb raw data contains packed float values that need to be unpacked.
    This function reads the packed data from data_float_h.csv.gz and
    unpacks it into individual float values in data_float_m.csv.gz.

    Slighly modified from source:
    https://github.com/nrodemund/sicdb/blob/ea9210169777c13a4732629d4e2979de0d1d9c37/Scripts/Unpack%20raw%20data/unpack.py

    Args:
        config_manager: Optional ConfigManager instance (uses get_config_manager if None)
        paths: Optional reprodICUPaths instance

    Returns:
        Path to the unpacked data folder

    Raises:
        FileNotFoundError: If sicdb_source folder or required files not found
        RuntimeError: If unpacking fails
    """
    if config_manager is None:
        config_manager = get_config_manager()
    if paths is None:
        paths = reprodICUPaths(config_manager)

    # Get source path
    source_path = Path(
        config_manager.load_config("PATHS.yaml", user_override=False)[
            "sicdb_source_path"
        ]
    )

    if not source_path.exists():
        raise FileNotFoundError(f"SICdb source folder not found: {source_path}")

    print("SICdb - Unpacking raw float data")
    print(f"  Source: {source_path}")

    # Save current directory and change to source path
    original_dir = os.getcwd()
    try:
        os.chdir(source_path)

        input_file = "data_float_h.csv.gz"
        output_file = "data_float_m.csv.gz"

        if not Path(input_file).exists():
            raise FileNotFoundError(f"Required file not found: {input_file}")

        def set_raw_values(row, dictwriter, n):
            """Unpack raw float values from hex data."""
            t = int(row["Offset"])
            data = bytes.fromhex(row["rawdata"][2:])
            for i in range(int(len(data) / 4)):
                if (
                    data[i * 4] == 0
                    and data[i * 4 + 1] == 0
                    and data[i * 4 + 2] == 0
                    and data[i * 4 + 3] == 0
                ):
                    continue  # Skip null values
                n = n + 1
                newrow = row.copy()
                del newrow["rawdata"]
                del newrow["cnt"]
                newrow["id"] = n
                newrow["Val"] = struct.unpack("<f", data[i * 4 : i * 4 + 4])[0]
                newrow["Offset"] = t + i * 60
                dictwriter.writerow(newrow)
            return n

        n = 0
        N = 2.6e9  # Approximately 2.6 billion entries
        print(f"  Unpacking {N/1e9:.1f}B entries from {input_file}...")

        with gzip.open(output_file, "wt") as csvfile:
            dict_writer = csv.DictWriter(
                csvfile, ["id", "CaseID", "DataID", "Offset", "Val"]
            )
            dict_writer.writeheader()
            with gzip.open(input_file, "rt", encoding="utf-8") as gzf:
                for row in csv.DictReader(gzf):
                    n = set_raw_values(row, dict_writer, n)
                    if n % 1e6 == 0:
                        print(
                            f"    Processing entry {n:_.0f} ({n/N:6.1%})",
                            end="\r",
                        )

        print("\n-> SICdb unpacking complete")
        return str(source_path)

    finally:
        os.chdir(original_dir)


def setup_mimic3_demo(config_manager=None, paths=None):
    """
    Fix MIMIC-III demo data headers.

    MIMIC-III demo data comes with lowercase column headers that need to be
    converted to uppercase. This function:
    1. Checks if CSV files exist in the mimic3_demo_source folder
    2. Converts the first line (header) to uppercase
    3. Saves files to a "HEADERFIX" subfolder
    4. Updates the local config path to point to the HEADERFIX folder

    Args:
        config_manager: Optional ConfigManager instance (uses get_config_manager if None)
        paths: Optional reprodICUPaths instance

    Returns:
        Path to the HEADERFIX folder

    Raises:
        FileNotFoundError: If mimic3_demo_source folder or CSV files not found
        RuntimeError: If header fixing fails
    """
    if config_manager is None:
        config_manager = get_config_manager()
    if paths is None:
        paths = reprodICUPaths(config_manager)

    # Get source path
    source_path = Path(
        config_manager.load_config("PATHS.yaml", user_override=False)[
            "mimic3_demo_source_path"
        ]
    )

    if not source_path.exists():
        raise FileNotFoundError(
            f"MIMIC-III demo source folder not found: {source_path}"
        )

    # Create HEADERFIX folder
    headerfix_path = source_path.parent / "HEADERFIX"
    headerfix_path.mkdir(parents=True, exist_ok=True)

    print("MIMIC-III demo - Converting headers to uppercase")
    print(f"  Source: {source_path}")
    print(f"  Target: {headerfix_path}")

    # Process all CSV files in the source directory
    csv_count = 0
    for input_file in source_path.glob("*.csv*"):
        if input_file.is_file():
            output_file = headerfix_path / input_file.name

            try:
                if input_file.suffix == ".gz":
                    # Handle gzipped CSV files
                    with (
                        gzip.open(input_file, "rt") as infile,
                        gzip.open(output_file, "wt") as outfile,
                    ):
                        for line_num, line in enumerate(infile, 1):
                            if line_num == 1:
                                # Convert header to uppercase
                                outfile.write(line.upper())
                            else:
                                outfile.write(line)
                else:
                    # Handle regular CSV files
                    with (
                        open(input_file, "r") as infile,
                        open(output_file, "w") as outfile,
                    ):
                        for line_num, line in enumerate(infile, 1):
                            if line_num == 1:
                                # Convert header to uppercase
                                outfile.write(line.upper())
                            else:
                                outfile.write(line)

                print(f"  Fixed {input_file.name}")
                csv_count += 1
            except Exception as e:
                raise RuntimeError(f"Failed to process {input_file.name}: {e}")

    if csv_count == 0:
        raise FileNotFoundError(f"No CSV files found in {source_path}")

    print(f"-> MIMIC-III demo header fixing complete ({csv_count} files)")
    
    # Update configuration to point to HEADERFIX folder
    config_manager.update_config(
        "PATHS.yaml",
        {"mimic3_demo_source_path": str(headerfix_path)},
        user_override=True
    )
    print(f"  Updated config: mimic3_demo_source_path -> {headerfix_path}")
    
    return str(headerfix_path)
