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
from helpers.C_harmonize.C_harmonize_diagnoses import DiagnosesHarmonizer
from helpers.C_harmonize.C_harmonize_medications import MedicationHarmonizer
from helpers.C_harmonize.C_harmonize_microbiology import MicrobiologyHarmonizer
from helpers.C_harmonize.C_harmonize_patient_information import \
    PatientInformationHarmonizer
from helpers.C_harmonize.C_harmonize_procedures import ProceduresHarmonizer
from helpers.C_harmonize.C_harmonize_timeseries import TimeseriesHarmonizer

# import overview functions
from helpers.helper_overview import Overview

# import extra functions for cleaning, winsorizing, etc.
from helpers.X1_clean.X1_clean_patient_information import \
    PatientInformationCleaner
from helpers.X1_clean.X1_improve_timeseries import IntakeOutputImprover
from helpers.X2_winsorize.X2_winsorize import X2_Winsorizer
from helpers.X3_impute.X3_impute_diagnoses import DiagnosesImputer
from helpers.X3_impute.X3_impute_medications import MedicationImputer
from helpers.X3_impute.X3_impute_patient_information import \
    PatientInformationImputer
from helpers.X3_impute.X3_impute_timeseries import TimeseriesImputer
from helpers.X4_resample.X4_resample_timeseries import TimeseriesResampler


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
        help="Which datasets to extract.",
    )
    parser.add_argument(
        "-t",
        "--tables",
        type=str,
        nargs="*",
        default=["all"],
        help="Which tables to build.",
    )
    parser.add_argument(
        "-s",
        "--timeseries",
        type=str,
        nargs="*",
        default=["all"],
        help="Which timeseries to extract specifically.",
    )
    parser.add_argument(
        "--FORCE",
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
    parser.add_argument(
        "--IMPUTE",
        action="store_true",
        help="Impute missing values in the data.",
    )
    parser.add_argument(
        "--RESAMPLE",
        type=int,
        nargs="?",
        const=300,
        help="Resample the timeseries data to a specified resolution in seconds.",
    )
    parser.add_argument(
        "--NO-OVERVIEW",
        action="store_true",
        help="Do not create an overview of the data extracted and harmonized.",
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
    # check that the tempfiles path exists, if not create it
    if not os.path.exists(save_path + "_tempfiles/"):
        os.makedirs(save_path + "_tempfiles/")

    # Select datasets to extract
    if "all" in args.datasets:
        DATASETS = [
            "eICU",
            "HiRID",
            "MIMIC3",
            "MIMIC4",
            "NWICU",
            "SICdb",
            "UMCdb",
        ]
        if args.DEMO:
            DATASETS = ["eICU", "MIMIC3", "MIMIC4"]
    else:
        DATASETS = args.datasets

    # Select tables to build
    if "all" in args.tables:
        TABLES = [
            "patient_information",
            "diagnoses",
            "procedures",
            "medications",
            "timeseries",
        ]
    else:
        TABLES = args.tables

    # Select timeseries to extract
    if "all" in args.timeseries:
        TIMESERIES = ["vitals", "labs", "respiratory", "inout"]
    else:
        TIMESERIES = args.timeseries

    # Setup the winsorizer
    winsorizer = X2_Winsorizer()

    # Run harmonizing
    # region info
    if "patient_information" in TABLES:
        print("reprodICU - Combining patient information...")
        patient_info_harmonizer = PatientInformationHarmonizer(
            paths=paths, datasets=DATASETS, DEMO=args.DEMO
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
            .pipe(patient_info_cleaner.add_good_patient_information)
            .pipe(
                winsorizer.winsorize_clip_lower_0_quantiles,
                columns=columns_to_winsorize,
                alpha=0.9995,
            )
            .pipe(patient_info_imputer.impute_patient_IDs)
            .collect()
            .write_parquet(save_path + "patient_information.parquet")
        )

    # region diags
    if "diagnoses" in TABLES:
        print("reprodICU - Combining diagnoses...")
        diagnoses_harmonizer = DiagnosesHarmonizer(
            paths=paths, datasets=DATASETS, DEMO=args.DEMO
        )
        diagnoses_imputer = DiagnosesImputer(
            paths=paths,
            patient_info_location=save_path + "patient_information.parquet",
        )

        (
            diagnoses_harmonizer.harmonize_diagnoses()
            .pipe(diagnoses_imputer.impute_diagnoses)
            .collect()
            .write_parquet(save_path + "diagnoses_imputed.parquet")
        )

    # region procs
    if "procedures" in TABLES:
        print("reprodICU - Combining procedures...")
        procedures_harmonizer = ProceduresHarmonizer(
            paths=paths, datasets=DATASETS, DEMO=args.DEMO
        )
        (
            procedures_harmonizer.harmonize_procedures()
            .collect()
            .write_parquet(save_path + "procedures.parquet")
        )

    # region meds
    if "medications" in TABLES:
        print("reprodICU - Combining medications...")
        medication_harmonizer = MedicationHarmonizer(
            paths=paths, datasets=DATASETS, DEMO=args.DEMO
        )
        medication_imputer = MedicationImputer(
            paths=paths,
            patient_info_location=save_path + "patient_information.parquet",
        )
        (
            medication_harmonizer.harmonize_medications()
            .collect()
            .write_parquet(save_path + "medications.parquet")
        )

    # region micro
    if "microbiology" in TABLES:
        print("reprodICU - Combining microbiology data...")
        microbiology_harmonizer = MicrobiologyHarmonizer(
            paths=paths, datasets=DATASETS, DEMO=args.DEMO
        )
        (
            microbiology_harmonizer.harmonize_microbiology()
            .collect()
            .write_parquet(save_path + "microbiology.parquet")
        )

    # region timeseries
    if "timeseries" in TABLES:
        print("reprodICU - Combining timeseries...")
        timeseries_harmonizer = TimeseriesHarmonizer(
            paths=paths, datasets=DATASETS, DEMO=args.DEMO
        )
        timeseries_imputer = TimeseriesImputer(paths=paths, DEMO=args.DEMO)
        timeseries_resampler = TimeseriesResampler(paths=paths, DEMO=args.DEMO)
        print("reprodICU - Splitting timeseries...")
        # Default paths are used for saving the timeseries data
        # vitals -> timeseries_vitals.parquet
        # labs -> timeseries_labs.parquet
        # resp -> timeseries_respiratory.parquet
        # inout -> timeseries_intakeoutput.parquet
        timeseries_harmonizer.harmonize_split_timeseries(
            timeseries=TIMESERIES, save_to_default=True
        )

        print("reprodICU - Improving intake/output data...")
        timeseries_inout_improver = IntakeOutputImprover(paths=paths)
        (
            pl.scan_parquet(save_path + "timeseries_intakeoutput.parquet")
            .pipe(
                timeseries_inout_improver.add_infusion_volumes,
                medications=pl.scan_parquet(save_path + "medications.parquet"),
            )
            .pipe(timeseries_inout_improver.improve_intake_output)
            .collect()
            .write_parquet(save_path + "timeseries_intakeoutput_balanced.parquet")
        )

        if "labs" in TIMESERIES:
            # Winsorize the lab data
            print("reprodICU - Winsorizing lab data...")
            labs = pl.scan_parquet(save_path + "timeseries_labs.parquet")
            columns_to_exclude = [
                column_names["global_icu_stay_id_col"],
                column_names["timeseries_time_col"],
                "Base excess",
            ]
            labs_cols = labs.collect_schema().names()
            columns_to_winsorize = list(
                set(labs_cols) - set(columns_to_exclude)
            )
            (
                labs.pipe(
                    winsorizer.winsorize_structs,
                    winsorization_columns=columns_to_winsorize,
                    winsorization_methods=[
                        "quantiles" for _ in columns_to_winsorize
                    ],
                )
                .collect()
                .write_parquet(save_path + "timeseries_labs_winsorized.parquet")
            )

        if args.IMPUTE and "vitals" in TIMESERIES:
            # Impute the timeseries data
            print("reprodICU - Imputing timeseries data...")
            # Impute the vitals data
            (
                pl.scan_parquet(save_path + "timeseries_vitals.parquet")
                .pipe(timeseries_imputer.impute_timeseries_vitals)
                .collect(streaming=True)
                .write_parquet(save_path + "timeseries_vitals_imputed.parquet")
            )

        if args.RESAMPLE and "vitals" in TIMESERIES:
            # Resample the timeseries data
            print("reprodICU - Resampling timeseries data...")
            (
                pl.scan_parquet(save_path + "timeseries_vitals.parquet")
                .pipe(
                    timeseries_resampler.resample_timeseries_vitals,
                    resolution_in_seconds=args.RESAMPLE,
                )
                .collect(streaming=True)
                .write_parquet(
                    save_path + "timeseries_vitals_resampled.parquet"
                )
            )

    # region info 2
    if "patient_information" in TABLES:
        # Add availability information to the patient information
        print("reprodICU - Adding data availability to patient information...")
        (
            pl.scan_parquet(save_path + "patient_information.parquet")
            .pipe(
                patient_info_cleaner.add_primary_diagnoses,
                diagnoses=save_path + "diagnoses_imputed.parquet",
            )
            .pipe(
                patient_info_cleaner.add_data_availability_information,
                diagnoses=save_path + "diagnoses_imputed.parquet",
                medications=save_path + "medications.parquet",
                procedures=save_path + "procedures.parquet",
                timeseries_labs=save_path + "timeseries_labs.parquet",
                timeseries_vitals=save_path + "timeseries_vitals.parquet",
                timeseries_resp=save_path + "timeseries_respiratory.parquet",
                timeseries_inout=save_path + "timeseries_intakeoutput.parquet",
            )
            .pipe(patient_info_cleaner.remove_bad_patient_information)
            .pipe(patient_info_cleaner.sort_columns)
            .collect()
            .write_parquet(
                save_path + "patient_information_with_data_availability.parquet"
            )
        )
        os.remove(save_path + "patient_information.parquet")
        os.rename(
            save_path + "patient_information_with_data_availability.parquet",
            save_path + "patient_information.parquet",
        )

        if args.IMPUTE:
            # Impute the patient information
            (
                pl.scan_parquet(save_path + "patient_information.parquet")
                .pipe(
                    patient_info_imputer.impute_patient_anthropometrics,
                    n_neighbors=5,
                )
                .collect(streaming=True)
                .write_parquet(
                    save_path + "patient_information_imputed.parquet"
                )
            )

    # region overview
    elif len(TABLES) == 0:
        print("reprodICU - No tables selected.")
        print("reprodICU - Make sure to select at least one table to build.")
        print("reprodICU - Must be one of:")
        print(
            "reprodICU - patient_information, diagnoses, procedures, medications, timeseries."
        )

    if not args.NO_OVERVIEW:
        # Create an overview of the data extracted and harmonized
        overview = Overview(save_path=save_path)
        print("reprodICU - Creating overview...")
        overview.create_overview()
        print("reprodICU - Creating database variable overview...")
        overview.create_database_variable_overview()

    print("reprodICU - Done.")

# endregion
