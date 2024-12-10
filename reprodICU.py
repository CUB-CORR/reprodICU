# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script extracts the data from the source files and stores it in a structured
# format for further processing and harmonization.
# It can be called with command line arguments to specify the source datasets to be extracted.

import argparse
import os
import polars as pl
import yaml

# import harmonizing functions
from helpers.C_harmonize.C_harmonize_patient_information import (
    PatientInformationHarmonizer,
)
from helpers.C_harmonize.C_harmonize_timeseries import TimeseriesHarmonizer
from helpers.C_harmonize.C_harmonize_medications import MedicationHarmonizer
from helpers.C_harmonize.C_harmonize_procedures import ProceduresHarmonizer
from helpers.C_harmonize.C_harmonize_diagnoses import DiagnosesHarmonizer

# import extra functions for cleaning, winsorizing, etc.
from helpers.X1_clean.X1_clean_patient_information import (
    PatientInformationCleaner,
)
from helpers.X2_winsorize.X2_winsorize import X2_Winsorizer
from helpers.X3_impute.X3_impute_diagnoses import DiagnosesImputer
from helpers.X3_impute.X3_impute_patient_information import (
    PatientInformationImputer,
)
from helpers.X3_impute.X3_impute_timeseries import TimeseriesImputer
from helpers.X3_impute.X3_impute_medications import MedicationImputer


def load_mapping(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def create_overview(save_path: str) -> None:
    """Create an overview of the data extracted and harmonized."""
    # Create DataFrame to store the overview, initialize columns for each dataset
    overview = pl.scan_parquet(
        save_path + "patient_information.parquet"
    ).select(["Global ICU Stay ID", "Source Dataset"])

    # Add columns for each table
    tables = [
        "diagnoses_imputed",
        # "procedures",
        "medications_imputed",
        "timeseries_vitals",
        "timeseries_labs",
        "timeseries_respiratory",
        "timeseries_intakeoutput",
    ]

    for table in tables:
        # print(f"Adding {table} to overview...")
        overview = (
            overview.join(
                pl.scan_parquet(save_path + table + ".parquet")
                .select("Global ICU Stay ID", pl.nth(1))
                .group_by("Global ICU Stay ID")
                .len()
                .rename({"len": table}),
                on="Global ICU Stay ID",
                how="left",
            )
            .collect()
            .lazy()
        )

    # Save the overview to a parquet file
    overview.sink_parquet(save_path + "overview.parquet")


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
        nargs="*",
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
    parser.add_argument(
        "--DEMO",
        action="store_true",
        help="Create a demo dataset with a subset of the data.",
    )
    args = parser.parse_args()

    # Initialize paths
    paths = reprodICUPaths()
    column_names = load_mapping("configs/COLUMN_NAMES.yaml")
    save_path = (
        paths.reprodICU_files_path
        if not args.DEMO
        else paths.reprodICU_demo_files_path
    )

    # Select datasets to extract
    if "all" in args.datasets:
        datasets = ["eICU", "HiRID", "MIMIC3", "MIMIC4", "SICdb", "UMCdb"]
        if args.DEMO:
            datasets = ["eICU", "MIMIC3", "MIMIC4"]
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
    # region info
    if "patient_information" in tables:
        print("reprodICU - Combining patient information...")
        patient_info_harmonizer = PatientInformationHarmonizer(
            paths=paths, datasets=datasets, DEMO=args.DEMO
        )
        patient_info_cleaner = PatientInformationCleaner(paths=paths)
        patient_info_imputer = PatientInformationImputer(paths=paths)

        # Winsorize the patient information
        columns_to_winsorize = [
            column_names["weight_col"],
            column_names["height_col"],
        ]
        (
            patient_info_harmonizer.harmonize_patient_information()
            .pipe(patient_info_cleaner.clean_patient_information)
            .pipe(patient_info_cleaner.remove_bad_patient_information)
            .pipe(patient_info_cleaner.add_good_patient_information)
            .pipe(
                X2_Winsorizer.winsorize_clip_lower_0_quantiles,
                columns=columns_to_winsorize,
                alpha=0.9995,
            )
            .collect(streaming=True)
            .write_parquet(save_path + "patient_information.parquet")
        )

    # region diags
    if "diagnoses" in tables:
        print("reprodICU - Combining diagnoses...")
        diagnoses_harmonizer = DiagnosesHarmonizer(
            paths=paths, datasets=datasets, DEMO=args.DEMO
        )
        diagnoses_imputer = DiagnosesImputer(
            paths=paths,
            patient_info_location=save_path + "patient_information.parquet",
        )

        (
            diagnoses_harmonizer.harmonize_diagnoses()
            .pipe(diagnoses_imputer.impute_diagnoses)
            .collect(streaming=True)
            .write_parquet(save_path + "diagnoses_imputed.parquet")
        )

    # region procs
    if "procedures" in tables:
        print("reprodICU - Combining procedures...")
        procedures_harmonizer = ProceduresHarmonizer(
            paths=paths, datasets=datasets, DEMO=args.DEMO
        )
        (
            procedures_harmonizer.harmonize_procedures()
            .collect()
            .write_parquet(save_path + "procedures.parquet")
        )

    # region meds
    if "medications" in tables:
        print("reprodICU - Combining medications...")
        medication_harmonizer = MedicationHarmonizer(
            paths=paths, datasets=datasets, DEMO=args.DEMO
        )
        medication_imputer = MedicationImputer(
            paths=paths,
            patient_info_location=save_path + "patient_information.parquet",
        )
        (
            medication_harmonizer.harmonize_medications()
            # .pipe(medication_imputer.add_common_rate)
            .collect(streaming=True)
            .write_parquet(save_path + "medications_imputed.parquet")
            # .sink_parquet(save_path + "medications.parquet")
        )

    # region timeseries
    if "timeseries" in tables:
        print("reprodICU - Combining timeseries...")
        timeseries_harmonizer = TimeseriesHarmonizer(
            paths=paths, datasets=datasets, DEMO=args.DEMO
        )
        timeseries_imputer = TimeseriesImputer(paths=paths, DEMO=args.DEMO)
        print("reprodICU - Splitting timeseries...")
        # Default paths are used for saving the timeseries data
        # vitals -> timeseries_vitals.parquet
        # labs -> timeseries_labs.parquet
        # resp -> timeseries_respiratory.parquet
        # inout -> timeseries_intakeoutput.parquet
        timeseries_harmonizer.harmonize_split_timeseries(save_to_default=True)

        # Remove the lab data metadata
        print("reprodICU - Removing lab data metadata...")
        labs = (
            pl.scan_parquet(save_path + "timeseries_labs.parquet")
            .pipe(timeseries_harmonizer.remove_metadata)
            .collect(streaming=True)
            .write_parquet(save_path + "timeseries_labs_no_meta.parquet")
        )

        # Winsorize the lab data
        print("reprodICU - Winsorizing lab data...")
        columns_to_exclude = [
            column_names["global_icu_stay_id_col"],
            column_names["timeseries_time_col"],
            "Base excess",
        ]
        labs = pl.scan_parquet(save_path + "timeseries_labs_no_meta.parquet")
        labs_cols = labs.collect_schema().names()
        columns_to_winsorize = list(set(labs_cols) - set(columns_to_exclude))
        (
            labs.pipe(
                X2_Winsorizer.winsorize_clip_lower_0_quantiles,
                columns=columns_to_winsorize,
                alpha=0.99,
            )
            .pipe(
                X2_Winsorizer.winsorize_quantiles,
                columns=["Base excess"],
                alpha=0.99,
            )
            .collect(streaming=True)
            .write_parquet(
                save_path + "timeseries_labs_no_meta_winsorized.parquet"
            )
        )

        # # Impute the timeseries data
        # print("reprodICU - Imputing timeseries data...")
        # # Impute the vitals data
        # vitals = pl.scan_parquet(save_path + "timeseries_vitals.parquet")
        # (
        #     vitals.pipe(
        #         timeseries_imputer.impute_timeseries,
        #         resolution_in_seconds=300,
        #         keep_preadmission_data=True,
        #     )
        # )

    # region info 2
    if "patient_information" in tables:
        # Add availability information to the patient information
        print("reprodICU - Adding data availability to patient information...")
        (
            pl.scan_parquet(save_path + "patient_information.parquet")
            .pipe(
                patient_info_cleaner.add_data_availability_information,
                diagnoses=save_path + "diagnoses_imputed.parquet",
                medications=save_path + "medications_imputed.parquet",
                procedures=save_path + "procedures.parquet",
                timeseries_labs=save_path + "timeseries_labs.parquet",
                timeseries_vitals=save_path + "timeseries_vitals.parquet",
                timeseries_resp=save_path + "timeseries_respiratory.parquet",
                timeseries_inout=save_path + "timeseries_intakeoutput.parquet",
            )
            .sink_parquet(
                save_path + "patient_information_with_data_availability.parquet"
            )
        )
        os.remove(save_path + "patient_information.parquet")
        os.rename(
            save_path + "patient_information_with_data_availability.parquet",
            save_path + "patient_information.parquet",
        )

        # # Impute the patient information
        # (
        #     pl.scan_parquet(save_path + "patient_information.parquet")
        #     .pipe(patient_info_imputer.impute_patient_IDs)
        #     .pipe(
        #         patient_info_imputer.impute_patient_anthropometrics,
        #         n_neighbors=5,
        #     )
        #     .collect(streaming=True)
        #     .write_parquet(save_path + "patient_information_imputed.parquet")
        # )

    # region overview
    elif len(tables) == 0:
        print("reprodICU - No tables selected.")
        print("reprodICU - Make sure to select at least one table to build.")
        print("reprodICU - Must be one of:")
        print(
            "reprodICU - patient_information, diagnoses, procedures, medications, timeseries."
        )

    # Create an overview of the data extracted and harmonized
    print("reprodICU - Creating overview...")
    create_overview(save_path)

    print("reprodICU - Done.")

# endregion
