# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script extracts the data from the source files and stores it in a structured
# format for further processing and harmonization.
# It can be called with command line arguments to specify the source datasets to be extracted.

import argparse
import yaml

# import harmonizing functions
from helpers.C_harmonize.C_harmonize_patient_information import (
    PatientInformationHarmonizer,
)
from helpers.C_harmonize.C_harmonize_timeseries import TimeseriesHarmonizer
from helpers.C_harmonize.C_harmonize_medications import MedicationHarmonizer
from helpers.C_harmonize.C_harmonize_procedures import ProceduresHarmonizer
from helpers.C_harmonize.C_harmonize_diagnoses import DiagnosesHarmonizer


def load_mapping(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class reprodICUPaths:
    def __init__(self) -> None:
        config = load_mapping("configs/paths_local.yaml")
        for key, value in config.items():
            setattr(self, key, str(value))


# region main
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Select datasets to extract.")
    parser.add_argument(
        "-d",
        "--datasets",
        type=str,
        nargs="+",
        default=["all"],
        help="Datasets to extract.",
    )
    parser.add_argument(
        "-t",
        "--tables",
        type=str,
        nargs="+",
        default=["all"],
        help="Tables to build.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force recomputation of precalculated data. This will delete existing files.",
    )
    parser.add_argument(
        "-b",
        "--build",
        type=str,
        nargs="+",
        default=["all"],
        help="What parts of the datasets to extract.",
    )
    args = parser.parse_args()

    # Initialize paths
    paths = reprodICUPaths()
    column_names = load_mapping("configs/COLUMN_NAMES.yaml")

    # Select datasets to extract
    if "all" in args.datasets:
        datasets = ["eICU", "HiRID", "MIMIC3", "MIMIC4", "SICdb", "UMCdb"]
    else:
        datasets = args.datasets

    # Select tables to build
    if "all" in args.tables:
        tables = [
            "patient_information",
            "diagnoses",
            "procedures",
            "medications",
            "timeseries",
        ]
    else:
        tables = args.tables

    # Run harmonizing
    if "patient_information" in tables:
        print("reprodICU - Combining patient information...")
        patient_info_harmonizer = PatientInformationHarmonizer(
            paths=paths, datasets=datasets
        )

        # Winsorize the patient information
        columns_to_winsorize = [
            column_names["weight_col"],
            column_names["height_col"],
        ]
        patient_info_harmonizer.harmonize_patient_information().sink_parquet(
            paths.reprodICU_files_path + "patient_information.parquet"
        )

    if "diagnoses" in tables:
        print("reprodICU - Combining diagnoses...")
        diagnoses_harmonizer = DiagnosesHarmonizer(
            paths=paths, datasets=datasets
        )
        diagnoses_harmonizer.harmonize_diagnoses().collect().write_parquet(
            paths.reprodICU_files_path + "diagnoses.parquet"
        )

    if "procedures" in tables:
        print("reprodICU - Combining procedures...")
        procedures_harmonizer = ProceduresHarmonizer(
            paths=paths, datasets=datasets
        )
        procedures_harmonizer.harmonize_procedures().collect().write_parquet(
            paths.reprodICU_files_path + "procedures.parquet"
        )

    if "medications" in tables:
        print("reprodICU - Combining medications...")
        medication_harmonizer = MedicationHarmonizer(
            paths=paths, datasets=datasets
        )
        medication_harmonizer.harmonize_medications().sink_parquet(
            paths.reprodICU_files_path + "medications.parquet"
        )

    if "timeseries" in tables:
        print("reprodICU - Combining timeseries...")
        timeseries_harmonizer = TimeseriesHarmonizer(
            paths=paths, datasets=datasets
        )
        # timeseries_harmonizer.harmonize_timeseries().sink_parquet("tempfiles/reprodICU_timeseries.parquet")
        vitals, labs, resp, inout = timeseries_harmonizer.split_timeseries(
            paths.reprodICU_files_path
            + "_tempfiles/reprodICU_timeseries.parquet",
            save_to_default=False,
        )
        vitals.sink_parquet(
            paths.reprodICU_files_path + "timeseries_vitals.parquet"
        )
        labs.sink_parquet(
            paths.reprodICU_files_path + "timeseries_labs.parquet"
        )
        resp.sink_parquet(
            paths.reprodICU_files_path + "timeseries_respiratory.parquet"
        )
        inout.sink_parquet(
            paths.reprodICU_files_path + "timeseries_intakeoutput.parquet"
        )

    else:
        print("reprodICU - No tables selected.")
        print("reprodICU - Make sure to select at least one table to build.")
        print("reprodICU - Must be one of:")
        print(
            "reprodICU - patient_information, diagnoses, procedures, medications, timeseries."
        )

    print("reprodICU - Done.")

# endregion
