import os
from typing import List, Optional

import polars as pl
import yaml

from .config import get_config_manager, reprodICUPaths

# import harmonizing functions
from .helpers.C_harmonize.C_harmonize_diagnoses import DiagnosesHarmonizer
from .helpers.C_harmonize.C_harmonize_medications import MedicationHarmonizer
from .helpers.C_harmonize.C_harmonize_microbiology import MicrobiologyHarmonizer
from .helpers.C_harmonize.C_harmonize_notes import NotesHarmonizer
from .helpers.C_harmonize.C_harmonize_patient_information import (
    PatientInformationHarmonizer,
)
from .helpers.C_harmonize.C_harmonize_procedures import ProceduresHarmonizer
from .helpers.C_harmonize.C_harmonize_timeseries import TimeseriesHarmonizer

# import overview functions
from .helpers.helper_overview import Overview

# import extra functions for cleaning, winsorizing, etc.
from .helpers.X1_clean.X1_clean_patient_information import (
    PatientInformationCleaner,
)
from .helpers.X1_clean.X1_improve_timeseries import IntakeOutputImprover
from .helpers.X1_clean.X1_map_diagnoses import DiagnosesMapper
from .helpers.X2_winsorize.X2_winsorize import X2_Winsorizer
from .helpers.X3_impute.X3_impute_patient_information import (
    PatientInformationImputer,
)
from .helpers.X3_impute.X3_impute_timeseries import TimeseriesImputer
from .helpers.X4_resample.X4_resample_timeseries import TimeseriesResampler


def load_mapping(path: str) -> dict:
    """Load a YAML mapping file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


# region helper functions
def _normalize_datasets(
    datasets: Optional[List[str]], demo: bool = False
) -> List[str]:
    """
    Normalize dataset selection.

    Arguments
    ---------
        datasets : list or None
            None for "all", or list of specific datasets to process
        demo : bool
            If True, restrict to demo-compatible datasets

    Returns
    -------
        list
            Dataset names to process
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
    """
    Normalize table selection.

    Arguments
    ---------
        tables : list or None
            None for "all", or list of specific tables

    Returns
    -------
        list
            Table names to build
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
    """
    Normalize timeseries selection.

    Arguments
    ---------
        timeseries : list or None
            None for "all", or list of specific timeseries types

    Returns
    -------
        list
            Timeseries types to extract
    """
    if timeseries is None or (
        isinstance(timeseries, list) and "all" in timeseries
    ):
        return ["vitals", "labs", "respiratory", "inout"]
    return timeseries


def _get_save_path(paths: reprodICUPaths, demo: bool = False) -> str:
    """
    Get the appropriate save path for output files.

    Arguments
    ---------
        paths : reprodICUPaths
            Paths configuration object
        demo : bool
            If True, use demo path; otherwise use full data path

    Returns
    -------
        str
            Path string for saving files

    Raises
    ------
        OSError
            If unable to create tempfiles directory
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


# endregion


# region patient information
def build_patient_information(
    paths: Optional[reprodICUPaths] = None,
    datasets: Optional[List[str]] = None,
    demo: bool = False,
) -> List[str]:
    """
    Build patient information table from raw data sources.

    Harmonizes patient demographics, anthropometrics, and admission metadata
    from configured source datasets. Applies data cleaning, validation, and
    winsorization for clinical plausibility.

    Arguments
    ---------
        paths : reprodICUPaths, optional
            Paths configuration object. Uses default ConfigManager if None.
        datasets : list, optional
            Datasets to process. Uses all datasets if None or contains "all".
        demo : bool
            If True, use demo-sized datasets instead of full data.

    Returns
    -------
        list
            List of output file paths created

    Raises
    ------
        FileNotFoundError
            If dataset files not found
        ValueError
            If invalid dataset selection
        RuntimeError
            If processing fails
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


# endregion


# region diagnoses
def build_diagnoses(
    paths: Optional[reprodICUPaths] = None,
    datasets: Optional[List[str]] = None,
    demo: bool = False,
) -> List[str]:
    """
    Build diagnoses table from raw data sources.

    Harmonizes diagnosis codes (ICD-9 and ICD-10) across datasets and maps
    them to standard coding systems. Preserves source information and temporal
    relationships to ICU admission.

    Arguments
    ---------
        paths : reprodICUPaths, optional
            Paths configuration object. Uses default ConfigManager if None.
        datasets : list, optional
            Datasets to process. Uses all datasets if None or contains "all".
        demo : bool
            If True, use demo-sized datasets instead of full data.

    Returns
    -------
        list
            List of output file paths created

    Raises
    ------
        FileNotFoundError
            If dataset files not found
        ValueError
            If invalid dataset selection
        RuntimeError
            If processing fails
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


# endregion


# region procedures
def build_procedures(
    paths: Optional[reprodICUPaths] = None,
    datasets: Optional[List[str]] = None,
    demo: bool = False,
) -> List[str]:
    """
    Build procedures table from raw data sources.

    Harmonizes procedure codes (ICD-9 and ICD-10) across datasets. Standardizes
    coding systems while preserving temporal and source information.

    Arguments
    ---------
        paths : reprodICUPaths, optional
            Paths configuration object. Uses default ConfigManager if None.
        datasets : list, optional
            Datasets to process. Uses all datasets if None or contains "all".
        demo : bool
            If True, use demo-sized datasets instead of full data.

    Returns
    -------
        list
            List of output file paths created

    Raises
    ------
        FileNotFoundError
            If dataset files not found
        ValueError
            If invalid dataset selection
        RuntimeError
            If processing fails
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


# endregion


# region medications
def build_medications(
    paths: Optional[reprodICUPaths] = None,
    datasets: Optional[List[str]] = None,
    demo: bool = False,
) -> List[str]:
    """
    Build medications table from raw data sources.

    Harmonizes medication administration data across datasets, standardizing
    drug codes, doses, routes, and administration times. Handles both
    administered and prescribed medication records.

    Arguments
    ---------
        paths : reprodICUPaths, optional
            Paths configuration object. Uses default ConfigManager if None.
        datasets : list, optional
            Datasets to process. Uses all datasets if None or contains "all".
        demo : bool
            If True, use demo-sized datasets instead of full data.

    Returns
    -------
        list
            List of output file paths created

    Raises
    ------
        FileNotFoundError
            If dataset files not found
        ValueError
            If invalid dataset selection
        RuntimeError
            If processing fails
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


# endregion


# region microbiology
def build_microbiology(
    paths: Optional[reprodICUPaths] = None,
    datasets: Optional[List[str]] = None,
    demo: bool = False,
) -> List[str]:
    """
    Build microbiology table from raw data sources.

    Harmonizes culture results, organisms identified, and susceptibility data
    across datasets. Standardizes organism nomenclature and test reporting.

    Arguments
    ---------
        paths : reprodICUPaths, optional
            Paths configuration object. Uses default ConfigManager if None.
        datasets : list, optional
            Datasets to process. Uses all datasets if None or contains "all".
        demo : bool
            If True, use demo-sized datasets instead of full data.

    Returns
    -------
        list
            List of output file paths created

    Raises
    ------
        FileNotFoundError
            If dataset files not found
        ValueError
            If invalid dataset selection
        RuntimeError
            If processing fails
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


# endregion


# region notes
def build_notes(
    paths: Optional[reprodICUPaths] = None,
    datasets: Optional[List[str]] = None,
    demo: bool = False,
) -> List[str]:
    """
    Build notes table from raw data sources.

    Harmonizes clinical notes (free text documentation) across datasets,
    standardizing timestamps and preserving note content and metadata.

    Arguments
    ---------
        paths : reprodICUPaths, optional
            Paths configuration object. Uses default ConfigManager if None.
        datasets : list, optional
            Datasets to process. Uses all datasets if None or contains "all".
        demo : bool
            If True, use demo-sized datasets instead of full data.

    Returns
    -------
        list
            List of output file paths created

    Raises
    ------
        FileNotFoundError
            If dataset files not found
        ValueError
            If invalid dataset selection
        RuntimeError
            If processing fails
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


# endregion


# region timeseries
def build_timeseries(
    paths: Optional[reprodICUPaths] = None,
    datasets: Optional[List[str]] = None,
    timeseries: Optional[List[str]] = None,
    demo: bool = False,
    impute: bool = False,
    resample: Optional[int] = None,
) -> List[str]:
    """
    Build timeseries data from raw data sources.

    Harmonizes vital signs, laboratory values, respiratory parameters, and
    intake/output records across datasets. Applies optional imputation and
    resampling to standardize temporal resolution.

    Arguments
    ---------
        paths : reprodICUPaths, optional
            Paths configuration object. Uses default ConfigManager if None.
        datasets : list, optional
            Datasets to process. Uses all datasets if None or contains "all".
        timeseries : list, optional
            Types to extract: "vitals", "labs", "respiratory", "inout".
            Uses all types if None or contains "all".
        demo : bool
            If True, use demo-sized datasets instead of full data.
        impute : bool
            If True, impute missing values in vital signs.
        resample : int, optional
            Resample to specified resolution in seconds. None for no resampling.

    Returns
    -------
        list
            List of output file paths created

    Raises
    ------
        FileNotFoundError
            If dataset files not found
        ValueError
            If invalid selection
        RuntimeError
            If processing fails
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


# endregion


# region overview
def build_overview(
    paths: Optional[reprodICUPaths] = None,
    demo: bool = False,
) -> List[str]:
    """
    Build data availability overview and summaries.

    Generates comprehensive overview of extracted data including:
    - By ICU stay: count of records for each data type
    - By database variable: aggregated counts per source dataset

    Arguments
    ---------
        paths : reprodICUPaths, optional
            Paths configuration object. Uses default ConfigManager if None.
        demo : bool
            If True, use demo-sized datasets instead of full data.

    Returns
    -------
        list
            List of overview file paths created

    Raises
    ------
        OSError
            If unable to access or write overview files
    """
    if paths is None:
        config_manager = get_config_manager()
        paths = reprodICUPaths(config_manager)

    save_path = _get_save_path(paths, demo=demo)

    overview = Overview(save_path=save_path)
    print("reprodICU - Creating overview...")
    overview.create_overview()
    print("reprodICU - Creating database variable overview...")
    overview.create_database_variable_overview()

    return [
        save_path + "overview.parquet",
        save_path + "overview_database_variable.parquet",
    ]


# endregion


# region build all
def build_all(
    paths: Optional[reprodICUPaths] = None,
    datasets: Optional[List[str]] = None,
    demo: bool = False,
    impute: bool = False,
    resample: Optional[int] = None,
    create_overview: bool = True,
) -> List[str]:
    """
    Build all data tables and timeseries from raw data sources.

    Orchestrates the complete data pipeline: harmonizes all clinical data
    (patient information, diagnoses, procedures, medications, microbiology,
    notes), timeseries data (vitals, labs, respiratory, intake/output), and
    generates comprehensive data availability overview.

    Arguments
    ---------
        paths : reprodICUPaths, optional
            Paths configuration object. Uses default ConfigManager if None.
        datasets : list, optional
            Datasets to process. Uses all datasets if None or contains "all".
        demo : bool
            If True, use demo-sized datasets instead of full data.
        impute : bool
            If True, impute missing values in vital signs.
        resample : int, optional
            Resample vitals to specified resolution in seconds.
            None for no resampling.
        create_overview : bool
            If True, generate data availability overview and summaries.

    Returns
    -------
        list
            List of all output file paths created

    Raises
    ------
        FileNotFoundError
            If dataset files not found
        ValueError
            If invalid selection
        RuntimeError
            If processing fails
    """
    if paths is None:
        config_manager = get_config_manager()
        paths = reprodICUPaths(config_manager)

    TABLES = _normalize_tables(None)  # Always build all tables
    save_path = _get_save_path(paths, demo=demo)

    all_output_files = []

    # Build all individual tables
    if "patient_information" in TABLES:
        all_output_files.extend(
            build_patient_information(paths=paths, datasets=datasets, demo=demo)
        )

    if "diagnoses" in TABLES:
        all_output_files.extend(
            build_diagnoses(paths=paths, datasets=datasets, demo=demo)
        )

    if "procedures" in TABLES:
        all_output_files.extend(
            build_procedures(paths=paths, datasets=datasets, demo=demo)
        )

    if "medications" in TABLES:
        all_output_files.extend(
            build_medications(paths=paths, datasets=datasets, demo=demo)
        )

    if "microbiology" in TABLES:
        all_output_files.extend(
            build_microbiology(paths=paths, datasets=datasets, demo=demo)
        )

    if "notes" in TABLES:
        all_output_files.extend(
            build_notes(paths=paths, datasets=datasets, demo=demo)
        )

    # Build timeseries
    if "timeseries" in TABLES:
        all_output_files.extend(
            build_timeseries(
                paths=paths,
                datasets=datasets,
                timeseries=None,
                demo=demo,
                impute=impute,
                resample=resample,
            )
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
        all_output_files.extend(build_overview(paths=paths, demo=demo))

    print("reprodICU - Done.")
    return all_output_files


# endregion
