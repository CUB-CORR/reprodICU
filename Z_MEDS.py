# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: Converts the reprodICU structure to the Medical Event Data Standard (MEDS).

# Input: reprodICU structure
# Output: MEDS structure

# Usage: python Z_MEDS.py

# Importing necessary libraries
import argparse
import json
import os
import warnings

from datetime import datetime

import polars as pl
import yaml
from helpers.C_harmonize.C_harmonize_diagnoses import DiagnosesHarmonizer
from helpers.helper_OMOP import Vocabulary
from tqdm import tqdm

warnings.filterwarnings("ignore")

SEED = 42

SECONDS_IN_DAY = 86400
DAYS_IN_YEAR = 365.25
DAY_ZERO = pl.datetime(year=2000, month=1, day=1, hour=0, minute=0, second=0)

PATIENT_INFO_CODES = {
    "Source Dataset": "SOURCE_DB",
    "Admission Age (years)": "AGE",
    "Gender": "GENDER",
    "Admission Height (cm)": "HEIGHT",
    "Admission Weight (kg)": "WEIGHT",
    "Ethnicity": "ETHNICITY",
    "Admission Type": "ADMISSION_TYPE",
    "Admission Urgency": "ADMISSION_URGENCY",
    "Admission Origin": "ADMISSION_ORIGIN",
    "Specialty": "SPECIALTY",
    "Care Site": "CARE_SITE",
    "Unit Type": "UNIT_TYPE",
    "Discharge Location": "DISCHARGE_LOCATION",
    "Hospital Length of Stay (days)": "HOSPITAL_LOS",
    "Pre-ICU Length of Stay (days)": "PRE_ICU_LOS",
    "ICU Length of Stay (days)": "ICU_LOS",
    "Mortality in ICU": "MORTALITY_ICU",
    "Mortality in Hospital": "MORTALITY_HOSPITAL",
    "Mortality after ICU discharge (days)": "MORTALITY_AFTER_ICU",
    "Mortality after ICU discharge cutoff (days)": "MORTALITY_AFTER_ICU_CUTOFF",
}


def load_mapping(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class reprodICUPaths:
    def __init__(self) -> None:
        config = load_mapping("configs/paths_local.yaml")
        for key, value in config.items():
            setattr(self, key, str(value))


# region helpers
# The _ID_ICUOFFSET function creates a table with the patient ID and the ICU admission time
def _ID_ICUOFFSET(patient_information: pl.LazyFrame) -> pl.LazyFrame:
    return (
        patient_information.select(
            "Global ICU Stay ID",
            "Global Hospital Stay ID",
            "Global Person ID",
            "Admission Time (24h)",
            "Pre-ICU Length of Stay (days)",
        )
        .with_columns(
            (
                DAY_ZERO.dt.combine(
                    # get hospital admission time
                    (
                        DAY_ZERO.dt.combine(pl.col("Admission Time (24h)"))
                        - pl.duration(
                            days=pl.col("Pre-ICU Length of Stay (days)")
                        )
                    ).dt.time()
                )
                + pl.duration(days=pl.col("Pre-ICU Length of Stay (days)"))
            ).alias("icu_admission_dttm"),
        )
        .select(
            "Global ICU Stay ID",
            "icu_admission_dttm",
        )
    )


# The create_dataset_json function creates a JSON file with metadata about the
# dataset.


# dataset.json
# This file contains metadata about the dataset itself, including the
# following:
# - dataset_name: The name of the dataset, of type string.
# - dataset_version: The version of the dataset, of type string. Ensuring the
#   version numbers used are meaningful and unique is important for
#   reproducibility, but is ultimately not enforced by the MEDS schema and is
#   left to the dataset creator.
# - etl_name: The name of the ETL process used to generate the dataset, of type
#   string.
# – etl_version: The version of the ETL process used to generate the dataset,
#   of type string.
# - meds_version: The version of the MEDS standard used to generate the dataset,
#   of type string.
# - created_at: The timestamp at which the dataset was created, of type string
#   in ISO 8601 format (note that this is not an official timestamp type, but
#   rather a string representation of a timestamp as this is a JSON file).
def create_dataset_json(OUTPATH: str) -> None:
    dataset = {
        "dataset_name": "reprodICU_MEDS",
        "dataset_version": "0.1",
        "etl_name": "Z_MEDS.py",
        "etl_version": "0.1",
        "meds_version": "0.1",
        "created_at": datetime.now().isoformat(),
    }
    with open(f"{OUTPATH}metadata/dataset.json", "w") as f:
        json.dump(dataset, f)


#######################################
# MEDS TABLES
#######################################
# region info_to_MEDS
def patient_information_to_MEDS(
    patient_information: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Convert the patient_information table to the MEDS format.

    Args:
        patient_information (pl.LazyFrame): reprodICU patient_information table

    Returns:
        pl.LazyFrame: MEDS table
    """

    return (
        patient_information.select(
            "Global ICU Stay ID",
            "Source Dataset",
            "Admission Age (years)",
            "Gender",
            "Admission Height (cm)",
            "Admission Weight (kg)",
            "Ethnicity",
            "Admission Type",
            "Admission Urgency",
            "Admission Origin",
            "Specialty",
            "Care Site",
            "Unit Type",
            "Discharge Location",
            "Hospital Length of Stay (days)",
            "Pre-ICU Length of Stay (days)",
            "ICU Length of Stay (days)",
            "Mortality in ICU",
            "Mortality in Hospital",
            "Mortality After ICU Discharge (days)",
            "Mortality After ICU Discharge Censor Cutoff (days)",
        )
        .unpivot(
            index="Global ICU Stay ID", variable_name="code", value_name="value"
        )
        .with_columns(
            # create an empty time column
            pl.lit(None).cast(pl.Date).alias("time"),
            # split the value column into a numeric and a string column
            pl.when(pl.col("value").cast(float, strict=False).is_not_null())
            .then(pl.col("value").cast(float, strict=False))
            .otherwise(pl.lit(None))
            .alias("numeric_value"),
            pl.when(pl.col("value").cast(float, strict=False).is_null())
            .then(pl.col("value"))
            .otherwise(pl.lit(None))
            .alias("string_value"),
            # replace the code colum with better codes
            pl.col("code").replace(PATIENT_INFO_CODES),
        )
        .with_columns(
            # add the string_value column to the code column
            pl.concat_str(
                [
                    pl.col("code"),
                    pl.lit("//"),
                    pl.col("string_value")
                    .str.to_uppercase()
                    .str.replace_all(" ", "_"),
                ],
                ignore_nulls=True,
            ).alias("code"),
        )
        .select("Global ICU Stay ID", "time", "code", "numeric_value")
    )


# region vitals_resp_inout
def vitals_resp_inout_to_MEDS(
    patient_information: pl.LazyFrame,
    timeseries_vitals: pl.LazyFrame,
    timeseries_resp: pl.LazyFrame,
    timeseries_inout: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Convert the timeseries_vitals, timeseries_resp, and timeseries_inout tables to the MEDS format.

    Args:
        patient_information (pl.LazyFrame): reprodICU patient_information table
        timeseries_vitals (pl.LazyFrame): reprodICU timeseries_vitals table
        timeseries_resp (pl.LazyFrame): reprodICU timeseries_resp table
        timeseries_inout (pl.LazyFrame): reprodICU timeseries_inout table

    Returns:
        pl.LazyFrame: MEDS table
    """

    ID_ICUOFFSET = _ID_ICUOFFSET(patient_information)

    return (
        pl.concat(
            [
                timeseries_vitals.collect(streaming=True),
                timeseries_resp.collect(streaming=True),
                timeseries_inout.collect(streaming=True),
            ],
            how="diagonal_relaxed",
        )
        .lazy()
        .join(ID_ICUOFFSET, on="Global ICU Stay ID", how="left")
        .with_columns(
            # calculate the time column
            (
                pl.col("icu_admission_dttm")
                + pl.duration(
                    seconds=pl.col("Time Relative to Admission (seconds)")
                )
            ).alias("time"),
        )
        .drop(
            "Time Relative to Admission (seconds)",
            "icu_admission_dttm",
        )
        .unpivot(
            index=["Global ICU Stay ID", "time"],
            variable_name="code",
            value_name="value",
        )
        .drop_nulls(["code", "value"])
        .with_columns(
            # split the value column into a numeric and a string column
            pl.when(pl.col("value").cast(float, strict=False).is_not_null())
            .then(pl.col("value").cast(float, strict=False))
            .otherwise(pl.lit(None))
            .alias("numeric_value"),
            pl.when(pl.col("value").cast(float, strict=False).is_null())
            .then(pl.col("value"))
            .otherwise(pl.lit(None))
            .alias("string_value"),
        )
        .with_columns(
            # add the string_value column to the code column
            pl.concat_str(
                [pl.col("code"), pl.lit("//"), pl.col("string_value")],
                ignore_nulls=True,
            ).alias("code"),
        )
        .select("Global ICU Stay ID", "time", "code", "numeric_value")
    )


# region labs_to_MEDS
def labs_to_MEDS(
    patient_information: pl.LazyFrame,
    timeseries_labs: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Convert the timeseries_labs table to the MEDS format.

    Args:
        patient_information (pl.LazyFrame): reprodICU patient_information table
        timeseries_labs (pl.LazyFrame): reprodICU timeseries_labs table

    Returns:
        pl.LazyFrame: MEDS table
    """

    ID_ICUOFFSET = _ID_ICUOFFSET(patient_information)

    return (
        timeseries_labs.join(ID_ICUOFFSET, on="Global ICU Stay ID", how="left")
        .with_columns(
            # calculate the time column
            (
                pl.col("icu_admission_dttm")
                + pl.duration(
                    seconds=pl.col("Time Relative to Admission (seconds)")
                )
            ).alias("time"),
        )
        .drop(
            "Time Relative to Admission (seconds)",
            "icu_admission_dttm",
        )
        .unpivot(
            index=["Global ICU Stay ID", "time"],
            variable_name="code",
            value_name="struct_value",
        )
        .drop_nulls(["code", "struct_value"])
        .with_columns(
            # rename the time field in the struct_value column
            pl.col("struct_value").struct.rename_fields(
                ["value", "system", "method", "_time_", "LOINC"]
            )
        )
        .unnest("struct_value")
        .with_columns(
            # use the LOINC code as the code column
            pl.col("LOINC").alias("code"),
            # use the numeric value as the numeric_value column
            pl.col("value").alias("numeric_value"),
        )
        .select("Global ICU Stay ID", "time", "code", "numeric_value")
    )


#######################################
# GENERAL MEDS FUNCTIONS
#######################################
# region to_MEDS
def to_MEDS(
    patient_information: pl.LazyFrame,
    # diagnoses: pl.LazyFrame,
    # medications: pl.LazyFrame,
    # procedures: pl.LazyFrame,
    timeseries_vitals: pl.LazyFrame,
    timeseries_labs: pl.LazyFrame,
    timeseries_resp: pl.LazyFrame,
    timeseries_inout: pl.LazyFrame,
    omop: Vocabulary,
) -> pl.LazyFrame:
    """_summary_

    Args:
        patient_information (pl.LazyFrame): reprodICU patient_information table
        diagnoses (pl.LazyFrame): reprodICU diagnoses table
        medications (pl.LazyFrame): reprodICU medications table
        procedures (pl.LazyFrame): reprodICU procedures table
        timeseries_vitals (pl.LazyFrame): reprodICU timeseries_vitals table
        timeseries_labs (pl.LazyFrame): reprodICU timeseries_labs table
        timeseries_resp (pl.LazyFrame): reprodICU timeseries_resp table
        timeseries_inout (pl.LazyFrame): reprodICU timeseries_inout table
        omop (Vocabulary): OMOP vocabulary helper

    Returns:
        pl.LazyFrame: MEDS table
    """

    return pl.concat(
        [
            patient_information_to_MEDS(patient_information).collect(),
            vitals_resp_inout_to_MEDS(
                patient_information,
                timeseries_vitals,
                timeseries_resp,
                timeseries_inout,
            ).collect(streaming=True),
            labs_to_MEDS(patient_information, timeseries_labs).collect(),
        ],
        how="vertical_relaxed",
    ).lazy()


# MEDS Data File Specification
##############################
# As is shown above, data files are stored in any nested (potentially
# multi-level) parquet files within the data/ folder (and all such parquet
# files must be data files). Each of these individual data files is a
# single shard of the dataset, and must follow the following specifications:
# - It must be compliant with the MEDS data schema
# - All data for a given subject must be stored in the same shard.
# - Shards must be sorted by subject_id and time within the shard - ordering
#   within these groups is unspecified.

# The MEDS data schema is an Apache Arrow schema that specifies the required
# columns and data types for MEDS data files. It currently includes the
# following columns:
# - subject_id: A unique identifier for each subject in the dataset, of type
#               int64.
# - time: The time at which the measurement corresponding to this row occurred,
#         of type timestamp[us].
# - code: A code representing the measurement that occurred (e.g., a diagnosis
#         or medication code), of type string.
# - numeric_value: If the measurement has a numeric value associated with it
#                  (e.g., a lab result), this column contains that value, of type
#                  float32.
# All columns except subject_id and code may contain nulls. If the time column
# is null it indicates a static measurement, and such rows should be sorted to
# the beginning of their associated subject's data. If the numeric_value column
# is null, it indicates that the measurement does not have an associated
# numeric value.

# The path from the MEDS data folder ($MEDS_ROOT/data/) to the shard file,
# "/" separated and without the .parquet extension, is the shard name.


# region MEDS
def MEDS(
    patient_information: pl.LazyFrame,
    # diagnoses: pl.LazyFrame,
    # medications: pl.LazyFrame,
    # procedures: pl.LazyFrame,
    timeseries_vitals: pl.LazyFrame,
    timeseries_labs: pl.LazyFrame,
    timeseries_resp: pl.LazyFrame,
    timeseries_inout: pl.LazyFrame,
    omop: Vocabulary,
) -> pl.LazyFrame:
    """_summary_

    Args:
        patient_information (pl.LazyFrame)
        diagnoses (pl.LazyFrame)
        medications (pl.LazyFrame)
        procedures (pl.LazyFrame)
        timeseries_vitals (pl.LazyFrame)
        timeseries_labs (pl.LazyFrame)
        timeseries_resp (pl.LazyFrame)
        timeseries_inout (pl.LazyFrame)
        omop (Vocabulary): OMOP vocabulary helper

    Returns:
        pl.LazyFrame: MEDS table
    """

    # Split into shards of 1000 patients each
    # (random shuffling to ensure approximately equal distribution time series data)
    ID_SHARDS = (
        patient_information.select("Global ICU Stay ID")
        .collect()
        .to_series()
        .shuffle(seed=SEED)
        .to_list()
    )
    ID_SHARDS = [
        ID_SHARDS[i : i + 1000] for i in range(0, len(ID_SHARDS), 1000)
    ]
    SUBJECT_IDS = patient_information.select("Global ICU Stay ID", "subject_id")

    # Iterate over the shards
    for i, ID_SHARD in tqdm(
        enumerate(ID_SHARDS),
        total=len(ID_SHARDS),
        desc="Processing MEDS shards",
    ):
        # print(f"MEDS - processing shard {i + 1}/{len(ID_SHARDS)}")
        patient_information_shard = patient_information.filter(
            pl.col("Global ICU Stay ID").is_in(ID_SHARD)
        )
        timeseries_vitals_shard = timeseries_vitals.filter(
            pl.col("Global ICU Stay ID").is_in(ID_SHARD)
        )
        timeseries_labs_shard = timeseries_labs.filter(
            pl.col("Global ICU Stay ID").is_in(ID_SHARD)
        )
        timeseries_resp_shard = timeseries_resp.filter(
            pl.col("Global ICU Stay ID").is_in(ID_SHARD)
        )
        timeseries_inout_shard = timeseries_inout.filter(
            pl.col("Global ICU Stay ID").is_in(ID_SHARD)
        )

        # Convert the reprodICU structure to the Medical Event Data Standard (MEDS)
        (
            to_MEDS(
                patient_information_shard,
                # diagnoses,
                # medications,
                # procedures,
                timeseries_vitals_shard,
                timeseries_labs_shard,
                timeseries_resp_shard,
                timeseries_inout_shard,
                omop,
            )
            .join(SUBJECT_IDS, on="Global ICU Stay ID", how="left")
            .drop("Global ICU Stay ID")
            .cast(
                {
                    "subject_id": pl.Int64,
                    "time": pl.Datetime,
                    "code": pl.String,
                    "numeric_value": pl.Float32,
                }
            )
            .collect(streaming=True)
            .sort("subject_id", "time")
            .select("subject_id", "time", "code", "numeric_value")
            .write_parquet(f"{OUTPATH}data/shard_{i}.parquet")
        )


# region main
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        help="Path to the reprodICU data",
        default="../reprodICU_files/",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        help="Path to the OMOP vocabulary files",
        default="../OMOP_vocabulary/",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to the output directory",
        default="../reprodICU_files_MEDS/",
    )
    args = parser.parse_args()

    # Create Output subdirectories
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.output + "data/", exist_ok=True)
    os.makedirs(args.output + "metadata/", exist_ok=True)

    # Initialize paths
    paths = reprodICUPaths()
    omop = Vocabulary(paths)
    os.makedirs(args.output, exist_ok=True)

    # Load the reprodICU data
    INPATH = args.input
    OUTPATH = args.output
    VOCABPATH = args.vocab
    diagnoses = pl.scan_parquet(INPATH + "diagnoses_imputed.parquet")
    medications = pl.scan_parquet(INPATH + "medications.parquet")
    patient_information = pl.scan_parquet(
        INPATH + "patient_information.parquet"
    )
    procedures = pl.scan_parquet(INPATH + "procedures.parquet")
    timeseries_vitals = pl.scan_parquet(INPATH + "timeseries_vitals.parquet")
    timeseries_labs = pl.scan_parquet(INPATH + "timeseries_labs.parquet")
    timeseries_resp = pl.scan_parquet(INPATH + "timeseries_respiratory.parquet")
    timeseries_inout = pl.scan_parquet(
        INPATH + "timeseries_intakeoutput_balanced.parquet"
    )

    CODE_STATUS = pl.scan_parquet(INPATH + "MAGIC_CONCEPTS/CODE_STATUS.parquet")

    # Setup some helpers instead of using the generated files
    diagnoses_harmonizer = DiagnosesHarmonizer(
        paths,
        datasets=[
            "eICU",
            "HiRID",
            "MIMIC3",
            "MIMIC4",
            "NWICU",
            "SICdb",
            "UMCdb",
        ],
    )

    #########
    # LOADING
    # Load the OMOP vocabulary files
    CONCEPT = pl.scan_parquet(VOCABPATH + "CONCEPT.parquet")
    CONCEPT_RELATIONSHIP = pl.scan_parquet(
        VOCABPATH + "CONCEPT_RELATIONSHIP.parquet"
    )
    CONCEPT_ANCESTOR = pl.scan_parquet(VOCABPATH + "CONCEPT_ANCESTOR.parquet")
    CONCEPT_CLASS = pl.scan_parquet(VOCABPATH + "CONCEPT_CLASS.parquet")
    CONCEPT_SYNONYM = pl.scan_parquet(VOCABPATH + "CONCEPT_SYNONYM.parquet")
    DOMAIN = pl.scan_parquet(VOCABPATH + "DOMAIN.parquet")
    RELATIONSHIP = pl.scan_parquet(VOCABPATH + "RELATIONSHIP.parquet")
    VOCABULARY = pl.scan_parquet(VOCABPATH + "VOCABULARY.parquet")

    ###########
    # FILTERING
    # Filter for first ICU stay
    patient_information = (
        patient_information.filter(
            pl.col("ICU Stay Sequential Number (per Person ID)").eq(1)
            | pl.col("ICU Stay Sequential Number (per Person ID)").is_null()
        )
        .with_row_index(name="subject_id")
        .cast({"subject_id": pl.Int64})
    )

    ############
    # CONVERTING
    # Convert the reprodICU structure to the Medical Event Data Standard (MEDS)
    create_dataset_json(OUTPATH)
    MEDS(
        patient_information,
        # diagnoses,
        # medications,
        # procedures,
        timeseries_vitals,
        timeseries_labs,
        timeseries_resp,
        timeseries_inout,
        omop,
    )

    print("MEDS - done")
