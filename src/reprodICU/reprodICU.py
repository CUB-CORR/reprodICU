# Author: Finn Fassbender
# Last modified: 2024-10-28

# Description: This module provides functions to extract, harmonize, and process
# data from source files and store it in a structured format.
# All functions are designed to be called programmatically from Python code.

import os
from typing import Dict, List, Optional

import polars as pl
import yaml
from config import get_config_manager, reprodICUPaths

# import harmonizing functions
from helpers.C_harmonize.C_harmonize_diagnoses import DiagnosesHarmonizer
from helpers.C_harmonize.C_harmonize_medications import MedicationHarmonizer
from helpers.C_harmonize.C_harmonize_microbiology import MicrobiologyHarmonizer
from helpers.C_harmonize.C_harmonize_notes import NotesHarmonizer
from helpers.C_harmonize.C_harmonize_patient_information import PatientInformationHarmonizer
from helpers.C_harmonize.C_harmonize_procedures import ProceduresHarmonizer
from helpers.C_harmonize.C_harmonize_timeseries import TimeseriesHarmonizer

# import overview functions
from helpers.helper_overview import Overview

# import extra functions for cleaning, winsorizing, etc.
from helpers.X1_clean.X1_clean_patient_information import PatientInformationCleaner
from helpers.X1_clean.X1_improve_timeseries import IntakeOutputImprover
from helpers.X1_clean.X1_map_diagnoses import DiagnosesMapper
from helpers.X2_winsorize.X2_winsorize import X2_Winsorizer
from helpers.X3_impute.X3_impute_patient_information import PatientInformationImputer
from helpers.X3_impute.X3_impute_timeseries import TimeseriesImputer
from helpers.X4_resample.X4_resample_timeseries import TimeseriesResampler


def load_mapping(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# Helper functions for parameter processing
def _normalize_datasets(
    datasets: Optional[List[str]], demo: bool = False
) -> List[str]:
    """Normalize dataset selection.

    Args:
        datasets: None for "all", or list of specific datasets
        demo: If True, restrict to demo-compatible datasets

    Returns:
        List of dataset names to process
    """
    if datasets is None or (isinstance(datasets, list) and "all" in datasets):
        all_datasets = [
            "eICU",
            "HiRID",
            "MIMIC3",
            "MIMIC4",
            "NWICU",
            "SICdb",
            "UMCdb",
        ]
        if demo:
            return ["eICU", "MIMIC3", "MIMIC4"]
        return all_datasets
    return datasets


def _normalize_tables(tables: Optional[List[str]]) -> List[str]:
    """Normalize table selection.

    Args:
        tables: None for "all", or list of specific tables

    Returns:
        List of table names to build
    """
    if tables is None or (isinstance(tables, list) and "all" in tables):
        return [
            "patient_information",
            "diagnoses",
            "procedures",
            "medications",
            "timeseries",
        ]
    return tables


def _normalize_timeseries(timeseries: Optional[List[str]]) -> List[str]:
    """Normalize timeseries selection.

    Args:
        timeseries: None for "all", or list of specific timeseries types

    Returns:
        List of timeseries types to extract
    """
    if timeseries is None or (
        isinstance(timeseries, list) and "all" in timeseries
    ):
        return ["vitals", "labs", "respiratory", "inout"]
    return timeseries


def _get_save_path(paths: reprodICUPaths, demo: bool = False) -> str:
    """Get the appropriate save path.

    Args:
        paths: reprodICUPaths instance
        demo: If True, use demo path

    Returns:
        Path string for saving files
    """
    save_path = (
        paths.reprodICU_files_path
        if not demo
        else paths.reprodICU_demo_files_path
    )

    # Ensure tempfiles directory exists
    tempfiles_path = save_path + "_tempfiles/"
    if not os.path.exists(tempfiles_path):
        os.makedirs(tempfiles_path)

    return save_path


# Building functions


def build_patient_information(
    paths: Optional[reprodICUPaths] = None,
    datasets: Optional[List[str]] = None,
    demo: bool = False,
    impute: bool = False,
    create_overview: bool = True,
) -> List[str]:
    """Build patient information table.

    Args:
        paths: Optional paths object (uses ConfigManager if None)
        datasets: Datasets to process (uses "all" if None)
        demo: Use demo data if True
        impute: Impute missing values if True
        create_overview: Create overview if True

    Returns:
        Dict mapping dataset names to output file paths

    Raises:
        FileNotFoundError: If dataset files not found
        ValueError: If invalid dataset selection
        RuntimeError: If processing fails
    """
    if paths is None:
        config_manager = get_config_manager()
        paths = reprodICUPaths(config_manager)

    config_manager = get_config_manager()
    column_names = config_manager.load_config(
        "COLUMN_NAMES.yaml", user_override=False
    )

    DATASETS = _normalize_datasets(datasets, demo=demo)
    save_path = _get_save_path(paths, demo=demo)

    print("reprodICU - Combining patient information...")
    patient_info_harmonizer = PatientInformationHarmonizer(
        paths=paths, datasets=DATASETS, DEMO=demo
    )
    patient_info_cleaner = PatientInformationCleaner(paths=paths)
    patient_info_imputer = PatientInformationImputer(paths=paths)

    # Winsorize the patient information
    winsorizer = X2_Winsorizer()
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

    return [save_path + "patient_information.parquet"]


def build_diagnoses(
    paths: Optional[reprodICUPaths] = None,
    datasets: Optional[List[str]] = None,
    demo: bool = False,
) -> List[str]:
    """Build diagnoses table.

    Args:
        paths: Optional paths object (uses ConfigManager if None)
        datasets: Datasets to process (uses "all" if None)
        demo: Use demo data if True

    Returns:
        Dict mapping dataset names to output file paths

    Raises:
        FileNotFoundError: If dataset files not found
        ValueError: If invalid dataset selection
        RuntimeError: If processing fails
    """
    if paths is None:
        config_manager = get_config_manager()
        paths = reprodICUPaths(config_manager)

    DATASETS = _normalize_datasets(datasets, demo=demo)
    save_path = _get_save_path(paths, demo=demo)

    print("reprodICU - Combining diagnoses...")
    diagnoses_harmonizer = DiagnosesHarmonizer(
        paths=paths, datasets=DATASETS, DEMO=demo
    )
    diagnoses_mapper = DiagnosesMapper(
        paths=paths,
        patient_info_path=save_path + "patient_information.parquet",
    )

    (
        diagnoses_harmonizer.harmonize_diagnoses()
        .pipe(diagnoses_mapper.map_diagnoses)
        .collect()
        .write_parquet(save_path + "diagnoses.parquet")
    )

    return [save_path + "diagnoses.parquet"]


def build_procedures(
    paths: Optional[reprodICUPaths] = None,
    datasets: Optional[List[str]] = None,
    demo: bool = False,
) -> List[str]:
    """Build procedures table.

    Args:
        paths: Optional paths object (uses ConfigManager if None)
        datasets: Datasets to process (uses "all" if None)
        demo: Use demo data if True

    Returns:
        Dict mapping dataset names to output file paths

    Raises:
        FileNotFoundError: If dataset files not found
        ValueError: If invalid dataset selection
        RuntimeError: If processing fails
    """
    if paths is None:
        config_manager = get_config_manager()
        paths = reprodICUPaths(config_manager)

    DATASETS = _normalize_datasets(datasets, demo=demo)
    save_path = _get_save_path(paths, demo=demo)

    print("reprodICU - Combining procedures...")
    procedures_harmonizer = ProceduresHarmonizer(
        paths=paths, datasets=DATASETS, DEMO=demo
    )
    (
        procedures_harmonizer.harmonize_procedures()
        .collect()
        .write_parquet(save_path + "procedures.parquet")
    )

    return [save_path + "procedures.parquet"]


def build_medications(
    paths: Optional[reprodICUPaths] = None,
    datasets: Optional[List[str]] = None,
    demo: bool = False,
) -> List[str]:
    """Build medications table.

    Args:
        paths: Optional paths object (uses ConfigManager if None)
        datasets: Datasets to process (uses "all" if None)
        demo: Use demo data if True

    Returns:
        Dict mapping dataset names to output file paths

    Raises:
        FileNotFoundError: If dataset files not found
        ValueError: If invalid dataset selection
        RuntimeError: If processing fails
    """
    if paths is None:
        config_manager = get_config_manager()
        paths = reprodICUPaths(config_manager)

    DATASETS = _normalize_datasets(datasets, demo=demo)
    save_path = _get_save_path(paths, demo=demo)

    print("reprodICU - Combining medications...")
    medication_harmonizer = MedicationHarmonizer(
        paths=paths, datasets=DATASETS, DEMO=demo
    )
    (
        medication_harmonizer.harmonize_split_medications(
            "administered"
        ).sink_parquet(save_path + "medications.parquet")
    )
    (
        medication_harmonizer.harmonize_split_medications(
            "prescribed"
        ).sink_parquet(save_path + "medications_prescribed.parquet")
    )

    return [
        save_path + "medications.parquet",
        save_path + "medications_prescribed.parquet",
    ]


def build_microbiology(
    paths: Optional[reprodICUPaths] = None,
    datasets: Optional[List[str]] = None,
    demo: bool = False,
) -> List[str]:
    """Build microbiology table.

    Args:
        paths: Optional paths object (uses ConfigManager if None)
        datasets: Datasets to process (uses "all" if None)
        demo: Use demo data if True

    Returns:
        Dict mapping dataset names to output file paths

    Raises:
        FileNotFoundError: If dataset files not found
        ValueError: If invalid dataset selection
        RuntimeError: If processing fails
    """
    if paths is None:
        config_manager = get_config_manager()
        paths = reprodICUPaths(config_manager)

    DATASETS = _normalize_datasets(datasets, demo=demo)
    save_path = _get_save_path(paths, demo=demo)

    print("reprodICU - Combining microbiology data...")
    microbiology_harmonizer = MicrobiologyHarmonizer(
        paths=paths, datasets=DATASETS, DEMO=demo
    )
    (
        microbiology_harmonizer.harmonize_microbiology()
        .collect()
        .write_parquet(save_path + "microbiology.parquet")
    )

    return [save_path + "microbiology.parquet"]


def build_notes(
    paths: Optional[reprodICUPaths] = None,
    datasets: Optional[List[str]] = None,
    demo: bool = False,
) -> List[str]:
    """Build notes table.

    Args:
        paths: Optional paths object (uses ConfigManager if None)
        datasets: Datasets to process (uses "all" if None)
        demo: Use demo data if True

    Returns:
        Dict mapping dataset names to output file paths

    Raises:
        FileNotFoundError: If dataset files not found
        ValueError: If invalid dataset selection
        RuntimeError: If processing fails
    """
    if paths is None:
        config_manager = get_config_manager()
        paths = reprodICUPaths(config_manager)

    DATASETS = _normalize_datasets(datasets, demo=demo)
    save_path = _get_save_path(paths, demo=demo)

    print("reprodICU - Combining notes...")
    notes_harmonizer = NotesHarmonizer(
        paths=paths, datasets=DATASETS, DEMO=demo
    )
    (
        notes_harmonizer.harmonize_notes().sink_parquet(
            save_path + "notes.parquet"
        )
    )

    return [save_path + "notes.parquet"]


def build_timeseries(
    paths: Optional[reprodICUPaths] = None,
    datasets: Optional[List[str]] = None,
    timeseries: Optional[List[str]] = None,
    demo: bool = False,
    impute: bool = False,
    resample: Optional[int] = None,
) -> List[str]:
    """Build timeseries data.

    Args:
        paths: Optional paths object (uses ConfigManager if None)
        datasets: Datasets to process (uses "all" if None)
        timeseries: Timeseries types to extract (uses "all" if None)
        demo: Use demo data if True
        impute: Impute missing values if True
        resample: Resample to specified resolution in seconds, or None

    Returns:
        Dict mapping dataset names to output file paths

    Raises:
        FileNotFoundError: If dataset files not found
        ValueError: If invalid selection
        RuntimeError: If processing fails
    """
    if paths is None:
        config_manager = get_config_manager()
        paths = reprodICUPaths(config_manager)

    config_manager = get_config_manager()
    column_names = config_manager.load_config(
        "COLUMN_NAMES.yaml", user_override=False
    )

    DATASETS = _normalize_datasets(datasets, demo=demo)
    TIMESERIES = _normalize_timeseries(timeseries)
    save_path = _get_save_path(paths, demo=demo)

    print("reprodICU - Combining timeseries...")
    timeseries_harmonizer = TimeseriesHarmonizer(
        paths=paths, datasets=DATASETS, DEMO=demo
    )
    timeseries_imputer = TimeseriesImputer(paths=paths, DEMO=demo)
    timeseries_resampler = TimeseriesResampler(paths=paths, DEMO=demo)
    print("reprodICU - Splitting timeseries...")

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
        winsorizer = X2_Winsorizer()
        labs = pl.scan_parquet(save_path + "timeseries_labs.parquet")
        columns_to_exclude = [
            column_names["global_icu_stay_id_col"],
            column_names["timeseries_time_col"],
            "Base excess",
        ]
        labs_cols = labs.collect_schema().names()
        columns_to_winsorize = list(set(labs_cols) - set(columns_to_exclude))
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

    if impute and "vitals" in TIMESERIES:
        # Impute the timeseries data
        print("reprodICU - Imputing timeseries data...")
        (
            pl.scan_parquet(save_path + "timeseries_vitals.parquet")
            .pipe(timeseries_imputer.impute_timeseries_vitals)
            .collect()
            .write_parquet(save_path + "timeseries_vitals_imputed.parquet")
        )

    if resample and "vitals" in TIMESERIES:
        # Resample the timeseries data
        print("reprodICU - Resampling timeseries data...")
        (
            pl.scan_parquet(save_path + "timeseries_vitals.parquet")
            .pipe(
                timeseries_resampler.resample_timeseries_vitals,
                resolution_in_seconds=resample,
            )
            .collect()
            .write_parquet(save_path + "timeseries_vitals_resampled.parquet")
        )

    # Collect output files
    return [
        save_path + f"timeseries_{ts_type}.parquet" for ts_type in TIMESERIES
    ]


def build_all(
    paths: Optional[reprodICUPaths] = None,
    datasets: Optional[List[str]] = None,
    demo: bool = False,
    impute: bool = False,
    resample: Optional[int] = None,
    create_overview: bool = True,
) -> List[str]:
    """Build all tables and timeseries.

    Args:
        paths: Optional paths object (uses ConfigManager if None)
        datasets: Datasets to process (uses "all" if None)
        demo: Use demo data if True
        impute: Impute missing values if True
        resample: Resample to specified resolution in seconds, or None
        create_overview: Create overview if True

    Returns:
        Dict mapping dataset names to list of all output file paths

    Raises:
        FileNotFoundError: If dataset files not found
        ValueError: If invalid selection
        RuntimeError: If processing fails
    """
    if paths is None:
        config_manager = get_config_manager()
        paths = reprodICUPaths(config_manager)

    config_manager = get_config_manager()
    column_names = config_manager.load_config(
        "COLUMN_NAMES.yaml", user_override=False
    )

    DATASETS = _normalize_datasets(datasets, demo=demo)
    TABLES = _normalize_tables(None)  # Always build all tables
    TIMESERIES = _normalize_timeseries(None)  # Always build all timeseries
    save_path = _get_save_path(paths, demo=demo)

    all_output_files = []

    # Build patient information
    if "patient_information" in TABLES:
        print("reprodICU - Combining patient information...")
        patient_info_harmonizer = PatientInformationHarmonizer(
            paths=paths, datasets=DATASETS, DEMO=demo
        )
        patient_info_cleaner = PatientInformationCleaner(paths=paths)
        patient_info_imputer = PatientInformationImputer(paths=paths)

        winsorizer = X2_Winsorizer()
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
        all_output_files.append(save_path + "patient_information.parquet")

    # Build diagnoses
    if "diagnoses" in TABLES:
        print("reprodICU - Combining diagnoses...")
        diagnoses_harmonizer = DiagnosesHarmonizer(
            paths=paths, datasets=DATASETS, DEMO=demo
        )
        diagnoses_mapper = DiagnosesMapper(
            paths=paths,
            patient_info_path=save_path + "patient_information.parquet",
        )

        (
            diagnoses_harmonizer.harmonize_diagnoses()
            .pipe(diagnoses_mapper.map_diagnoses)
            .collect()
            .write_parquet(save_path + "diagnoses.parquet")
        )
        all_output_files.append(save_path + "diagnoses.parquet")

    # Build procedures
    if "procedures" in TABLES:
        print("reprodICU - Combining procedures...")
        procedures_harmonizer = ProceduresHarmonizer(
            paths=paths, datasets=DATASETS, DEMO=demo
        )
        (
            procedures_harmonizer.harmonize_procedures()
            .collect()
            .write_parquet(save_path + "procedures.parquet")
        )
        all_output_files.append(save_path + "procedures.parquet")

    # Build medications
    if "medications" in TABLES:
        print("reprodICU - Combining medications...")
        medication_harmonizer = MedicationHarmonizer(
            paths=paths, datasets=DATASETS, DEMO=demo
        )
        (
            medication_harmonizer.harmonize_split_medications(
                "administered"
            ).sink_parquet(save_path + "medications.parquet")
        )
        (
            medication_harmonizer.harmonize_split_medications(
                "prescribed"
            ).sink_parquet(save_path + "medications_prescribed.parquet")
        )
        all_output_files.extend(
            [
                save_path + "medications.parquet",
                save_path + "medications_prescribed.parquet",
            ]
        )

    # Build microbiology
    if "microbiology" in TABLES:
        print("reprodICU - Combining microbiology data...")
        microbiology_harmonizer = MicrobiologyHarmonizer(
            paths=paths, datasets=DATASETS, DEMO=demo
        )
        (
            microbiology_harmonizer.harmonize_microbiology()
            .collect()
            .write_parquet(save_path + "microbiology.parquet")
        )
        all_output_files.append(save_path + "microbiology.parquet")

    # Build notes
    if "notes" in TABLES:
        print("reprodICU - Combining notes...")
        notes_harmonizer = NotesHarmonizer(
            paths=paths, datasets=DATASETS, DEMO=demo
        )
        (
            notes_harmonizer.harmonize_notes().sink_parquet(
                save_path + "notes.parquet"
            )
        )
        all_output_files.append(save_path + "notes.parquet")

    # Build timeseries
    if "timeseries" in TABLES:
        print("reprodICU - Combining timeseries...")
        timeseries_harmonizer = TimeseriesHarmonizer(
            paths=paths, datasets=DATASETS, DEMO=demo
        )
        timeseries_imputer = TimeseriesImputer(paths=paths, DEMO=demo)
        timeseries_resampler = TimeseriesResampler(paths=paths, DEMO=demo)
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
            .write_parquet(
                save_path + "timeseries_intakeoutput_balanced.parquet"
            )
        )

        if "labs" in TIMESERIES:
            print("reprodICU - Winsorizing lab data...")
            winsorizer = X2_Winsorizer()
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

        if impute and "vitals" in TIMESERIES:
            print("reprodICU - Imputing timeseries data...")
            (
                pl.scan_parquet(save_path + "timeseries_vitals.parquet")
                .pipe(timeseries_imputer.impute_timeseries_vitals)
                .collect()
                .write_parquet(save_path + "timeseries_vitals_imputed.parquet")
            )

        if resample and "vitals" in TIMESERIES:
            print("reprodICU - Resampling timeseries data...")
            (
                pl.scan_parquet(save_path + "timeseries_vitals.parquet")
                .pipe(
                    timeseries_resampler.resample_timeseries_vitals,
                    resolution_in_seconds=resample,
                )
                .collect()
                .write_parquet(
                    save_path + "timeseries_vitals_resampled.parquet"
                )
            )

        all_output_files.extend(
            [
                save_path + f"timeseries_{ts_type}.parquet"
                for ts_type in TIMESERIES
            ]
        )

    # Add patient information availability
    if "patient_information" in TABLES:
        print("reprodICU - Adding data availability to patient information...")
        patient_info_cleaner = PatientInformationCleaner(paths=paths)
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

        if impute:
            patient_info_imputer = PatientInformationImputer(paths=paths)
            (
                pl.scan_parquet(save_path + "patient_information.parquet")
                .pipe(
                    patient_info_imputer.impute_patient_anthropometrics,
                    n_neighbors=5,
                )
                .collect()
                .write_parquet(
                    save_path + "patient_information_imputed.parquet"
                )
            )

    # Create overview if requested
    if create_overview:
        overview = Overview(save_path=save_path)
        print("reprodICU - Creating overview...")
        overview.create_overview()
        print("reprodICU - Creating database variable overview...")
        overview.create_database_variable_overview()

    print("reprodICU - Done.")
    return all_output_files
