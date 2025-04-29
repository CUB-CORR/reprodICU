# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: Converts the reprodICU structure to the OMOP Common Data Model (CDM) structure.
# The script is based on the OMOP CDM version 5.4

# Input: reprodICU structure
# Output: OMOP CDM structure

# Usage: python reprOMOPIZE.py

# Importing necessary libraries
import argparse
import glob
import os
import warnings

import polars as pl
import yaml
from helpers.helper import GlobalVars
from helpers.helper_OMOP import Vocabulary
import tempfile
import atexit

warnings.filterwarnings("ignore")

SECONDS_IN_DAY = 86400
DAY_ZERO = pl.datetime(year=2000, month=1, day=1, hour=0, minute=0, second=0)


def load_mapping(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class reprodICUPaths:
    def __init__(self) -> None:
        config = load_mapping("configs/paths_local.yaml")
        for key, value in config.items():
            setattr(self, key, str(value))


# region helpers
# The FIELD_LEVEL table contains a list of fields that are used in the
# observational data tables. Each field is uniquely identified by a field
# concept ID and a field name.
def _field_level(table_name: str, return_required: bool = False) -> list:
    """
    return a list of fields for the table in the OMOP CDM in order
    """
    field_level_ = pl.read_csv("mappings/OMOP_CDMv5.4_Field_Level.csv").filter(
        pl.col("cdmTableName") == table_name
    )
    fields = field_level_.select("cdmFieldName").to_series().to_list()
    if not return_required:
        return fields

    required = field_level_.select("isRequired").to_series().to_list()
    return fields, required


# The _add_missing_fields function adds missing fields to the data
# The function checks if the fields are required for the table
# If the field is required and missing, the function raises a ValueError
# If the field is missing and not required, the function adds the field with a NULL value
def _add_missing_fields(
    data: pl.LazyFrame, table_name: str, check_required: bool = True
) -> pl.LazyFrame:
    """
    add missing fields to the data
    """
    fields, required = _field_level(table_name, return_required=True)
    columns = data.collect_schema().names()

    for field, req in zip(fields, required):
        if req == "Yes" and check_required:
            if not data.filter(pl.col(field).is_null()).collect().is_empty():
                # count the number of nulls
                count = data.collect().height
                null_count = (
                    data.filter(pl.col(field).is_null()).collect().height
                )
                print(
                    f"Field {field} is required for the {table_name} table, dropping {null_count} nulls ({count} total)"
                )
            data = data.drop_nulls(field)

        if field not in columns:
            if req == "Yes" and check_required:
                raise ValueError(
                    f"Field {field} is required for the {table_name} table"
                )
            data = data.with_columns(pl.lit(None).alias(field))

    return data.select(fields)


# Approximate the time relative to the first ICU admission in eICU-CRD.
# i.e. distribute the hospital stays equally over the data collection time
# period of two years.
# ICU stays are kept relative to each other within the same hospital stay.
def _approximate_time_relative_to_first_icu_admission(
    patient_information: pl.LazyFrame,
) -> pl.LazyFrame:
    patient_information_eicu = patient_information.filter(
        pl.col("Source Dataset") == "eICU-CRD"
    )

    eicu_avg_time_out_of_hospital_per_patient = (
        patient_information_eicu.group_by("Global Hospital Stay ID")
        .first()
        .group_by("Global Person ID")
        .agg(
            # Calculate the average time out of hospital per patient
            (730 - pl.col("Hospital Length of Stay (days)").sum())
            .truediv(pl.col("Global Hospital Stay ID").unique().len())
            .alias("Average Time Between Hospital Stays (days)"),
        )
    )

    eicu_cum_time_in_hospital_per_patient = (
        patient_information_eicu.group_by("Global Hospital Stay ID")
        .first()
        .with_columns(
            pl.col("Hospital Length of Stay (days)")
            .cum_sum()
            .over(
                "Global Person ID",
                order_by="ICU Stay Sequential Number (per Person ID)",
            )
            .alias("Cumulative Time in Hospital (days)"),
            pl.col("Global Hospital Stay ID")
            .shift(-1)
            .over(
                "Global Person ID",
                order_by="ICU Stay Sequential Number (per Person ID)",
            ),
        )
        .select("Global Hospital Stay ID", "Cumulative Time in Hospital (days)")
    )

    # Calculate the time relative to the first ICU admission this hospital stay
    eicu_time_since_previous_icu_during_hospital_stay = (
        patient_information_eicu.with_columns(
            (
                pl.col("Pre-ICU Length of Stay (days)")
                - pl.col("Pre-ICU Length of Stay (days)")
                .min()
                .over("Global Hospital Stay ID")
            ).alias(
                "Time Relative to First ICU Admission this Hospital Stay (days)"
            ),
        ).select(
            "Global ICU Stay ID",
            "Time Relative to First ICU Admission this Hospital Stay (days)",
        )
    )

    patient_information_eicu = (
        patient_information_eicu.with_columns(
            # Number the Hospital Stays per Person ID
            pl.col("Global Hospital Stay ID")
            .shift(1)
            .over(
                "Global Person ID",
                order_by="ICU Stay Sequential Number (per Person ID)",
            )
            .alias("shiftHID")
        )
        .with_columns(
            pl.col("shiftHID")
            .ne(pl.col("Global Hospital Stay ID"))
            .cum_sum()
            .over(
                "Global Person ID",
                order_by="ICU Stay Sequential Number (per Person ID)",
            )
            .fill_null(0)
            .add(1)
            .alias("Hospital Stay Sequential Number (per Person ID)"),
        )
        .drop("shiftHID")
        .join(
            eicu_avg_time_out_of_hospital_per_patient,
            on="Global Person ID",
            how="left",
            coalesce=True,
        )
        .join(
            eicu_cum_time_in_hospital_per_patient,
            on="Global Hospital Stay ID",
            how="left",
            coalesce=True,
        )
        .join(
            eicu_time_since_previous_icu_during_hospital_stay,
            on="Global ICU Stay ID",
            how="left",
            coalesce=True,
        )
        .with_columns(
            # If the ICU stay is the first ICU stay, the time relative to the
            # first ICU admission is 0
            pl.when(pl.col("ICU Stay Sequential Number (per Person ID)") == 1)
            .then(pl.lit(0))
            # If the ICU stay is not the first ICU stay,
            # but if the ICU is during the same hospital stay, the time relative
            # to the first ICU admission is the time relative to the first
            # ICU admission this hospital stay
            .when(
                pl.col("Hospital Stay Sequential Number (per Person ID)") == 1
            )
            .then(
                pl.col(
                    "Time Relative to First ICU Admission this Hospital Stay (days)"
                )
                * SECONDS_IN_DAY
            )
            # Otherwise it is the time in hospital after the first ICU
            # admission this hospital stay,
            # plus the sum of all previous times in hospital,
            # plus the average time out of hospital per patient times the
            # number of hospital stays before this one
            .otherwise(
                (
                    pl.col("Cumulative Time in Hospital (days)").fill_null(0)
                    + pl.col(
                        "Time Relative to First ICU Admission this Hospital Stay (days)"
                    )
                    + pl.col("Average Time Between Hospital Stays (days)")
                    * pl.col(
                        "Hospital Stay Sequential Number (per Person ID)"
                    ).sub(1)
                )
                * SECONDS_IN_DAY
            )
            .cast(int)
            .alias("Time Relative to First ICU Admission (seconds)"),
        )
    )

    return pl.concat(
        [
            patient_information_eicu,
            patient_information.filter(pl.col("Source Dataset") != "eICU-CRD"),
        ],
        how="diagonal",
    )


# The _ID function creates the person_id column by hashing the Global Person ID
# to ensure that the person_id is unique
def _ID(
    patient_information: pl.LazyFrame, additional_columns: list = []
) -> pl.LazyFrame:

    if "Time Relative to First ICU Admission (seconds)" in additional_columns:
        patient_information = _approximate_time_relative_to_first_icu_admission(
            patient_information
        )

    return patient_information.with_columns(
        ###########
        # PERSON_ID
        # Create the person_id column with a hash of the Global Person ID
        # NOTE: same as in the PERSON table
        pl.col("Global Person ID")
        .hash()
        .alias("person_id"),
    ).select(
        "Global ICU Stay ID",
        "person_id",
        *additional_columns,
    )


# endregion


# region DRUG_EXPOSURE
# This table captures records about the exposure to a Drug ingested or otherwise
# introduced into the body. A Drug is a biochemical substance formulated in such
# a way that when administered to a Person it will exert a certain biochemical
# effect on the metabolism. Drugs include prescription and over-the-counter
# medicines, vaccines, and large-molecule biologic therapies. Radiological
# devices ingested or applied locally do not count as Drugs.
def drug_exposure(
    CONCEPT: pl.LazyFrame,
    medications: pl.LazyFrame,
    patient_information: pl.LazyFrame,
) -> pl.LazyFrame:
    print("reprOMOPIZE - drug_exposure")

    ID = _ID(
        patient_information,
        [
            "Time Relative to First ICU Admission (seconds)",
            "Admission Time (24h)",
        ],
    )
    CONCEPTS = CONCEPT.filter(
        pl.col("domain_id") == "Drug",
        pl.col("concept_class_id") == "Ingredient",
    ).select("concept_id", "concept_name")

    # Extract the drug exposure information
    return (
        medications.join(ID, on="Global ICU Stay ID", how="right")
        # Create the drug_concept_id column with the concept_id of the Drug Ingredient
        .join(
            CONCEPTS,
            left_on="Drug Ingredient",
            right_on="concept_name",
            how="left",
        )
        .drop("Drug Ingredient")
        .rename({"concept_id": "drug_concept_id"})
        .with_columns(
            #####################
            # VISIT_OCCURRENCE_ID
            # Create the visit_occurrence_id column with a hash of the Global ICU Stay ID
            pl.col("Global ICU Stay ID").hash().alias("visit_occurrence_id"),
            ##############################
            # DRUG_EXPOSURE_START_DATETIME
            # Create the drug_exposure_start_datetime column with the datetime of the drug exposure
            (
                DAY_ZERO.dt.combine(pl.col("Admission Time (24h)"))
                + pl.duration(
                    seconds=pl.col("Drug Start Relative to Admission (seconds)")
                )
                + pl.duration(
                    seconds=pl.col(
                        "Time Relative to First ICU Admission (seconds)"
                    )
                )
            ).alias("drug_exposure_start_datetime"),
            ############################
            # DRUG_EXPOSURE_END_DATETIME
            # Create the drug_exposure_end_datetime column with the datetime of the drug exposure
            (
                DAY_ZERO.dt.combine(pl.col("Admission Time (24h)"))
                + pl.duration(
                    seconds=pl.col("Drug End Relative to Admission (seconds)")
                )
                + pl.duration(
                    seconds=pl.col(
                        "Time Relative to First ICU Admission (seconds)"
                    )
                )
            ).alias("drug_exposure_end_datetime"),
            ######################
            # DRUG_TYPE_CONCEPT_ID
            # 32817 = EHR
            pl.lit(32817).alias("drug_type_concept_id"),
        )
        .with_columns(
            ############################
            # DRUG_EXPOSURE_END_DATETIME
            # Fill the drug_exposure_end_datetime column with the drug_exposure_start_datetime
            # if the drug_exposure_end_datetime is missing
            pl.when(pl.col("drug_exposure_end_datetime").is_null())
            .then(pl.col("drug_exposure_start_datetime"))
            .otherwise(pl.col("drug_exposure_end_datetime"))
            .alias("drug_exposure_end_datetime"),
        )
        .with_columns(
            ##########################
            # DRUG_EXPOSURE_START_DATE
            # Create the drug_exposure_start_date column with the date of the drug exposure
            pl.col("drug_exposure_start_datetime")
            .dt.date()
            .alias("drug_exposure_start_date"),
            ########################
            # DRUG_EXPOSURE_END_DATE
            # Create the drug_exposure_end_datetime column with the date of the drug exposure
            pl.col("drug_exposure_end_datetime")
            .dt.date()
            .alias("drug_exposure_end_date"),
        )
        .rename(
            {
                "Drug Name": "drug_source_value",
                "Drug Administration Route": "route_source_value",
                "Drug Amount": "quantity",
            }
        )
        .drop_nulls("drug_concept_id")
        .unique()
        .with_row_index("drug_exposure_id")
        .pipe(_add_missing_fields, "drug_exposure")
    )


# region CARE_SITE
# The CARE_SITE table contains a list of uniquely identified institutional
# (physical or organizational) units where healthcare delivery is practiced
# (offices, wards, hospitals, clinics, etc.
def care_site(patient_information: pl.LazyFrame) -> pl.LazyFrame:
    print("reprOMOPIZE - care_site")

    # Extract the care site information
    care_site = (
        patient_information.select("Care Site")
        .with_columns(
            ##############
            # CARE_SITE_ID
            # Create the care_site_id column with a hash of the Care Site
            # NOTE: same as in the LOCATION table
            pl.col("Care Site").hash().alias("care_site_id"),
            # Create the care_site_source_value column with the Care Site for backreference
            pl.col("Care Site")
            .str.slice(0, 50)
            .alias("care_site_source_value"),
            # Create the care_site_name column with the Care Site for backreference
            pl.col("Care Site").str.slice(0, 255).alias("care_site_name"),
            #############
            # LOCATION_ID
            # Create the location_id column with a hash of the Care Site + "Location"
            # NOTE: same as in the LOCATION table
            pl.concat_str(pl.col("Care Site"), pl.lit("_Location"))
            .hash()
            .alias("location_id"),
        )
        .pipe(_add_missing_fields, "care_site")
        .unique()
    )

    return care_site


# endregion


# region CONDITION_OCCURRENCE
# This table contains records of Events of a Person suggesting the presence of
# a disease or medical condition stated as a diagnosis, a sign, or a symptom,
# which is either observed by a Provider or reported by the patient.
def condition_occurrence(
    CONCEPT: pl.LazyFrame,
    patient_information: pl.LazyFrame,
    diagnoses: pl.LazyFrame,
) -> pl.LazyFrame:
    print("reprOMOPIZE - condition_occurrence")

    ID = _ID(
        patient_information,
        [
            "Time Relative to First ICU Admission (seconds)",
            "Admission Time (24h)",
            "Pre-ICU Length of Stay (days)",
            "Hospital Length of Stay (days)",
            "Source Dataset",
        ],
    )
    CONCEPTS = CONCEPT.filter(
        pl.col("vocabulary_id").str.starts_with("ICD"),
    ).select(
        "concept_id",
        pl.col("concept_code").str.replace_all(".", "", literal=True),
    )

    # prefer ICD-10 over ICD-9, except for MIMIC-III (which only has ICD-9)
    diagnoses_ICD10 = (
        diagnoses.filter(
            pl.col("Global ICU Stay ID").str.starts_with("mimic3-").not_(),
            pl.col("Diagnosis ICD-10 Code").is_not_null(),
        )
        .join(ID, on="Global ICU Stay ID", how="right")
        .join(
            CONCEPTS,
            left_on="Diagnosis ICD-10 Code",
            right_on="concept_code",
            how="left",
        )
        .rename({"concept_id": "condition_concept_id"})
    )
    diagnoses_ICD9 = (
        diagnoses.filter(
            pl.col("Global ICU Stay ID").str.starts_with("mimic3-")
            | pl.col("Diagnosis ICD-10 Code").is_null()
        )
        .join(ID, on="Global ICU Stay ID", how="right")
        .join(
            CONCEPTS,
            left_on="Diagnosis ICD-9 Code",
            right_on="concept_code",
            how="left",
        )
        .rename({"concept_id": "condition_concept_id"})
    )

    # relevant ETL Conventions for discharge diagnoses in MIMIC:
    # Most often data sources do not have the idea of a start date for a
    # condition. Rather, if a source only has one date associated with a
    # condition record it is acceptable to use that date for both the
    # CONDITION_START_DATE and the CONDITION_END_DATE.

    return (
        pl.concat([diagnoses_ICD10, diagnoses_ICD9], how="vertical")
        .with_columns(
            ##########################
            # CONDITION_START_DATETIME
            # Create the condition_start_datetime column with the datetime of the diagnosis
            (
                DAY_ZERO.dt.combine(pl.col("Admission Time (24h)"))
                + pl.duration(
                    seconds=pl.col(
                        "Time Relative to First ICU Admission (seconds)"
                    )
                )
                + pl.when(pl.col("Source Dataset").str.starts_with("MIMIC"))
                .then(
                    pl.duration(
                        days=pl.col("Hospital Length of Stay (days)")
                        - pl.col("Pre-ICU Length of Stay (days)")
                    )
                )
                .otherwise(
                    pl.duration(
                        seconds=pl.col(
                            "Diagnosis Start Relative to Admission (seconds)"
                        )
                    )
                )
            ).alias("condition_start_datetime"),
            #######################
            # CONDITION_END_DATETIME
            # Create the condition_end_datetime column with the datetime of the diagnosis
            (
                DAY_ZERO.dt.combine(pl.col("Admission Time (24h)"))
                + pl.duration(
                    seconds=pl.col(
                        "Time Relative to First ICU Admission (seconds)"
                    )
                )
                + pl.when(pl.col("Source Dataset").str.starts_with("MIMIC"))
                .then(
                    pl.duration(
                        days=pl.col("Hospital Length of Stay (days)")
                        - pl.col("Pre-ICU Length of Stay (days)")
                    )
                )
                .otherwise(
                    pl.duration(
                        seconds=pl.col(
                            "Diagnosis Start Relative to Admission (seconds)"
                        )
                    )
                )
            ).alias("condition_end_datetime"),
            ###########################
            # CONDITION_TYPE_CONCEPT_ID
            # 32817 = EHR
            pl.lit(32817).alias("condition_type_concept_id"),
            #############################
            # CONDITION_STATUS_CONCEPT_ID
            # 32890 = Admission diagnosis
            # 32896 = Discharge diagnosis
            # 32901 = Primary admission diagnosis
            # 32902 = Primary diagnosis
            # 32903 = Primary discharge diagnosis
            # 32908 = Secondary diagnosis
            # 32909 = Secondary discharge diagnosis
            # SICdb only includes primary admission diagnoses
            pl.when(pl.col("Source Dataset") == "SICdb").then(pl.lit(32901))
            # MIMIC-III and MIMIC-IV only include discharge diagnoses
            .when(pl.col("Source Dataset").str.starts_with("MIMIC"))
            .then(
                pl.when(pl.col("Diagnosis Priority") == 1)
                .then(pl.lit(32903))
                .when(pl.col("Diagnosis Priority") == 2)
                .then(pl.lit(32909))
                .otherwise(32896)
            )
            .when(pl.col("Diagnosis Priority") == 1)
            .then(pl.lit(32902))
            .when(pl.col("Diagnosis Priority") == 2)
            .then(pl.lit(32908))
            .otherwise(None)
            .alias("condition_status_concept_id"),
            #####################
            # VISIT_OCCURRENCE_ID
            # Create the visit_occurrence_id column with a hash of the Global ICU Stay ID
            # NOTE: same as in the VISIT_OCCURRENCE table
            pl.col("Global ICU Stay ID").hash().alias("visit_occurrence_id"),
            ########################
            # CONDITION_SOURCE_VALUE
            # Create the condition_source_value column with the Diagnosis for backreference
            pl.col("Diagnosis Description").alias("condition_source_value"),
        )
        .with_columns(
            ######################
            # CONDITION_START_DATE
            # Create the condition_start_date column with the date of the diagnosis
            pl.col("condition_start_datetime")
            .dt.date()
            .alias("condition_start_date"),
            ####################
            # CONDITION_END_DATE
            # Create the condition_end_date column with the date of the diagnosis
            pl.col("condition_end_datetime")
            .dt.date()
            .alias("condition_end_date"),
        )
        .unique()
        .with_row_index("condition_occurrence_id")
        .pipe(_add_missing_fields, "condition_occurrence")
    )


# endregion


# region DEATH
# The death domain contains the clinical event for how and when a Person dies.
# A person can have up to one record if the source system contains evidence
# about the Death, such as: Condition in an administrative claim, status of
# enrollment into a health plan, or explicit record in EHR data.
def death(patient_information: pl.LazyFrame) -> pl.LazyFrame:
    print("reprOMOPIZE - death")

    ID = _ID(patient_information)

    return (
        patient_information.join(ID, on="Global ICU Stay ID", how="right")
        .with_columns(
            ############
            # DEATH_DATE
            # Create the death_date column with the date of the death
            (
                DAY_ZERO.dt.combine(pl.col("Admission Time (24h)"))
                + pl.duration(days=pl.col("ICU Length of Stay (days)"))
                + pl.duration(
                    seconds=pl.col(
                        "Time Relative to First ICU Admission (seconds)"
                    )
                )
                # add a day if the patient died in the ICU
                + pl.when(pl.col("Mortality in ICU"))
                .then(pl.duration(days=1))
                .otherwise(
                    pl.duration(
                        days=pl.col("Mortality After ICU Discharge (days)")
                    )
                )
            )
            .dt.date()
            .alias("death_date"),
            #######################
            # DEATH_TYPE_CONCEPT_ID
            # 32817 = EHR
            pl.lit(32817).alias("death_type_concept_id"),
        )
        # select latest death date per person
        .group_by("person_id")
        .max()
        .drop_nulls("death_date")
        .pipe(_add_missing_fields, "death")
        .unique()
    )


# region DEVICE_EXPOSURE
# The Device domain captures information about a person’s exposure to a foreign
# physical object or instrument which is used for diagnostic or therapeutic
# purposes through a mechanism beyond chemical action. Devices include
# implantable objects (e.g. pacemakers, stents, artificial joints), medical
# equipment and supplies (e.g. bandages, crutches, syringes), other instruments
# used in medical procedures (e.g. sutures, defibrillators) and material used
# in clinical care (e.g. adhesives, body material, dental material, surgical
# material).
def device_exposure(
    CONCEPT: pl.LazyFrame,
    patient_information: pl.LazyFrame,
    procedures: pl.LazyFrame,
) -> pl.LazyFrame:
    print("reprOMOPIZE - device_exposure")

    ID = _ID(
        patient_information,
        [
            "Global Person ID",
            "Time Relative to First ICU Admission (seconds)",
            "Admission Time (24h)",
        ],
    )
    CONCEPTS = CONCEPT.filter(
        pl.col("domain_id") == "Device",
        pl.col("concept_class_id") == "Physical Object",
        pl.col("standard_concept") == "S",
    ).select("concept_id", "concept_name")

    procedures = procedures.with_columns(
        pl.col("Procedure Description").replace(
            {
                "Insertion of catheter into peripheral vein": (
                    "Peripheral venous cannula"
                ),
                "Peripheral venous cannula insertion": (
                    "Peripheral venous cannula"
                ),
                "Insertion of catheter into artery": "Arterial catheter",
                "Change of dressing": "Wound dressing",
                "Open insertion of central venous catheter": (
                    "Central venous catheter"
                ),
                "Insertion of peripherally inserted central catheter": (
                    "Peripherally inserted central catheter"
                ),
                "Invasive ventilation": "Endotracheal tube",
                "Central venous cannula insertion": "Central venous catheter",
                "Pulmonary - Ventilation and Oxygenation - Mechanical Ventilation": (
                    "Endotracheal tube"
                ),  # eICU
                "Introduction of catheter into pulmonary artery": (
                    "Pulmonary artery catheter"
                ),
                "Insertion of endotracheal tube": "Endotracheal tube",
                "Pulmonary - Ventilation and Oxygenation - Oxygen Therapy (< 40%) - Nasal Cannula": (
                    "Oxygen nasal cannula"
                ),  # eICU
                "Insertion of hemodialysis catheter": "Hemodialysis catheter",
                "Pulmonary - Ventilation and Oxygenation - Non-Invasive Ventilation": (
                    "Continuous positive airway pressure/Bilevel positive airway pressure mask"
                ),  # eICU
                "Pulmonary catheterization with Swan-Ganz catheter": (
                    "Pulmonary artery catheter"
                ),
                "Bronchoscopy": "Bronchoscope",
                "Pulmonary - Ventilation and Oxygenation - Oxygen Therapy (40% to 60%)": (
                    "Oxygen mask"
                ),  # eICU
                "Cardiovascular - Vascular Disorders - VTE Prophylaxis - Compression Stockings": (
                    "Compression stockings"
                ),  # eICU
                "Renal - Urinary Catheters - Foley Catheter": (
                    "Foley catheter"
                ),  # eICU
                "Pulmonary - Ventilation and Oxygenation - Oxygen Therapy (< 40%)": (
                    "Oxygen nasal cannula"
                ),  # eICU
                "Continuous renal replacement therapy": "Hemodialysis catheter",
                "Hemodialysis": "Hemodialysis catheter",
                "Insertion of cannula for hemodialysis, vein to vein": (
                    "Hemodialysis catheter"
                ),
                "Cardiovascular - Vascular Disorders - VTE Prophylaxis - Compression Boots": (
                    "Compression stockings"
                ),  # eICU
                "Pulmonary - Ventilation and Oxygenation - CPAP/PEEP Therapy": (
                    "Continuous positive airway pressure/Bilevel positive airway pressure mask"
                ),  # eICU
                "Pulmonary - Ventilation and Oxygenation - Ventilator Weaning": (
                    "Endotracheal tube"
                ),  # eICU
                "Renal - Dialysis - Hemodialysis": (
                    "Hemodialysis catheter"
                ),  # eICU
                "Pulmonary - Ventilation and Oxygenation - Tracheal Suctioning": (
                    "Tracheal suction catheter"
                ),  # eICU
                "Noninvasive ventilation": (
                    "Continuous positive airway pressure/Bilevel positive airway pressure mask"
                ),
                "Introduction of intracranial pressure catheter": (
                    "Intracranial pressure catheter"
                ),
                "Pulmonary - Ventilation and Oxygenation - Oxygen Therapy (> 60%)": (
                    "Oxygen mask"
                ),  # eICU
                "Introduction of urinary catheter": "Foley catheter",
                "Surgery - Tubes and Catheters - Foley Catheter": (
                    "Foley catheter"
                ),  # eICU
                "Pulmonary - Radiologic Procedures / Bronchoscopy - Endotracheal Tube": (
                    "Endotracheal tube"
                ),  # eICU
                "Endoscopy": "Endoscope",
                "Pulmonary - Radiologic Procedures / Bronchoscopy - Endotracheal Tube - Insertion": (
                    "Endotracheal tube"
                ),  # eICU
                "Ultrasonography guided insertion of midline intravenous catheter": (
                    "Midline catheter"
                ),
                "Pulmonary - Ventilation and Oxygenation - Mechanical Ventilation - Assist Controlled": (
                    "Endotracheal tube"
                ),  # eICU
                "Surgery - Tubes and Catheters - Chest Tube": (
                    "Chest drain"
                ),  # eICU
                "Percutaneous insertion of intra-aortic balloon catheter": (
                    "Intra-aortic balloon catheter"
                ),
                "Pulmonary - Ventilation and Oxygenation - Non-Invasive Ventilation - Face Mask": (
                    "Oxygen mask"
                ),  # eICU
                "Pulmonary - Vascular Disorders - VTE Prophylaxis - Compression Boots": (
                    "Compression stockings"
                ),  # eICU
            }
        )
    )

    return (
        procedures.join(ID, on="Global Person ID", how="right")
        .join(
            CONCEPTS,
            left_on="Procedure Description",
            right_on="concept_name",
            how="left",
        )
        .rename({"concept_id": "device_concept_id"})
        .with_columns(
            #####################
            # VISIT_OCCURRENCE_ID
            # Create the visit_occurrence_id column with a hash of the Global ICU Stay ID
            pl.col("Global ICU Stay ID").hash().alias("visit_occurrence_id"),
            ################################
            # DEVICE_EXPOSURE_START_DATETIME
            # Create the device_exposure_start_datetime column with the datetime of the device exposure
            (
                DAY_ZERO.dt.combine(pl.col("Admission Time (24h)"))
                + pl.duration(
                    seconds=pl.col(
                        "Procedure Start Relative to Admission (seconds)"
                    )
                )
                + pl.duration(
                    seconds=pl.col(
                        "Time Relative to First ICU Admission (seconds)"
                    )
                )
            ).alias("device_exposure_start_datetime"),
            ##############################
            # DEVICE_EXPOSURE_END_DATETIME
            # Create the device_exposure_end_datetime column with the datetime of the device exposure
            (
                DAY_ZERO.dt.combine(pl.col("Admission Time (24h)"))
                + pl.duration(
                    seconds=pl.col(
                        "Procedure End Relative to Admission (seconds)"
                    )
                )
                + pl.duration(
                    seconds=pl.col(
                        "Time Relative to First ICU Admission (seconds)"
                    )
                )
            ).alias("device_exposure_end_datetime"),
            ########################
            # DEVICE_TYPE_CONCEPT_ID
            # 32817 = EHR
            pl.lit(32817).alias("device_type_concept_id"),
            #####################
            # DEVICE_SOURCE_VALUE
            # Create the device_source_value column with the Procedure for backreference
            pl.col("Procedure Description").alias("device_source_value"),
        )
        .with_columns(
            ############################
            # DEVICE_EXPOSURE_START_DATE
            # Create the device_exposure_start_date column with the date of the device exposure
            pl.col("device_exposure_start_datetime")
            .dt.date()
            .alias("device_exposure_start_date"),
            ##########################
            # DEVICE_EXPOSURE_END_DATE
            # Create the device_exposure_end_date column with the date of the device exposure
            pl.col("device_exposure_end_datetime")
            .dt.date()
            .alias("device_exposure_end_date"),
        )
        .drop_nulls("device_concept_id")
        .with_row_index("device_exposure_id")
        .pipe(_add_missing_fields, "device_exposure")
        .unique()
    )


# endregion


# region LOCATION
# The LOCATION table represents a generic way to capture physical location or
# address information of Persons and Care Sites.
def location(patient_information: pl.LazyFrame) -> pl.LazyFrame:
    print("reprOMOPIZE - location")

    # Adresses of the known institutions
    # HiRID:
    # Universitätsspital Bern
    # Freiburgstrasse 20, 3010 Bern, Schweiz

    # MIMIC:
    # Beth Israel Deaconess Medical Center
    # 330 Brookline Ave, Boston, MA 02215, USA

    # SICdb:
    # Landeskrankenhaus Salzburg
    # Müllner Hauptstraße 48, 5020 Salzburg, Österreich

    # UMCdb:
    # Amsterdam Universitair Medische Centra
    # Meibergdreef 9, 1105 AZ Amsterdam, Nederlands

    # Extract the location information
    return (
        patient_information.select("Care Site")
        .with_columns(
            #############
            # LOCATION_ID
            # Create the location_id column with a hash of the Care Site + "Location"
            # NOTE: same as in the CARE_SITE table
            pl.concat_str(pl.col("Care Site"), pl.lit("_Location"))
            .hash()
            .alias("location_id"),
            # Create the location_source_value column with the Care Site for backreference
            pl.col("Care Site").alias("location_source_value"),
            #########
            # ADDRESS
            # Create the address_1 column
            pl.when(pl.col("Care Site") == "Universitätsspital Bern")
            .then(pl.lit("Freiburgstrasse 20"))
            .when(pl.col("Care Site") == "Beth Israel Deaconess Medical Center")
            .then(pl.lit("330 Brookline Ave"))
            .when(pl.col("Care Site") == "Landeskrankenhaus Salzburg")
            .then(pl.lit("Müllner Hauptstraße 48"))
            .when(
                pl.col("Care Site") == "Amsterdam Universitair Medische Centra"
            )
            .then(pl.lit("Meibergdreef 9"))
            .otherwise(None)
            .alias("address_1"),
            # Create the city column
            pl.when(pl.col("Care Site") == "Universitätsspital Bern")
            .then(pl.lit("Bern"))
            .when(pl.col("Care Site") == "Beth Israel Deaconess Medical Center")
            .then(pl.lit("Boston"))
            .when(pl.col("Care Site") == "Landeskrankenhaus Salzburg")
            .then(pl.lit("Salzburg"))
            .when(
                pl.col("Care Site") == "Amsterdam Universitair Medische Centra"
            )
            .then(pl.lit("Amsterdam"))
            .otherwise(None)
            .alias("city"),
            # Create the state column
            pl.when(
                pl.col("Care Site") == "Beth Israel Deaconess Medical Center"
            )
            .then(pl.lit("MA"))
            .otherwise(None)
            .alias("state"),
            #########
            # COUNTRY
            # Create the country_source_value column
            pl.when(pl.col("Care Site") == "Universitätsspital Bern")
            .then(pl.lit("Switzerland"))  # 4330427
            .when(pl.col("Care Site") == "Beth Israel Deaconess Medical Center")
            .then(pl.lit("United States of America"))  # 4330442
            .when(pl.col("Care Site") == "Landeskrankenhaus Salzburg")
            .then(pl.lit("Austria"))  # 4329596
            .when(
                pl.col("Care Site") == "Amsterdam Universitair Medische Centra"
            )
            .then(pl.lit("Netherlands"))  # 4320169
            .otherwise(
                pl.lit("United States of America")
            )  # 4330442 -> eICU default
            .alias("country_source_value"),
            # Create the country_concept_id column
            pl.when(pl.col("Care Site") == "Universitätsspital Bern")
            .then(pl.lit(4330427))
            .when(pl.col("Care Site") == "Beth Israel Deaconess Medical Center")
            .then(pl.lit(4330442))
            .when(pl.col("Care Site") == "Landeskrankenhaus Salzburg")
            .then(pl.lit(4329596))
            .when(
                pl.col("Care Site") == "Amsterdam Universitair Medische Centra"
            )
            .then(pl.lit(4320169))
            .otherwise(pl.lit(4330442))
            .alias("country_concept_id"),
        )
        .pipe(_add_missing_fields, "location")
        .unique()
    )


# endregion


# region MEASUREMENT
# The MEASUREMENT table contains records of Measurements, i.e. structured
# values (numerical or categorical) obtained through systematic and
# standardized examination or testing of a Person or Person’s sample. The
# MEASUREMENT table contains both orders and results of such Measurements as
# laboratory tests, vital signs, quantitative findings from pathology reports,
# etc. Measurements are stored as attribute value pairs, with the attribute as
# the Measurement Concept and the value representing the result. The value can
# be a Concept (stored in VALUE_AS_CONCEPT), or a numerical value
# (VALUE_AS_NUMBER) with a Unit (UNIT_CONCEPT_ID). The Procedure for obtaining
# the sample is housed in the PROCEDURE_OCCURRENCE table, though it is
# unnecessary to create a PROCEDURE_OCCURRENCE record for each measurement if
# one does not exist in the source data. Measurements differ from Observations
# in that they require a standardized test or some other activity to generate a
# quantitative or qualitative result. If there is no result, it is assumed that
# the lab test was conducted but the result was not captured.
def measurement(
    CONCEPT: pl.LazyFrame,
    patient_information: pl.LazyFrame,
    timeseries_vitals: pl.LazyFrame,
    timeseries_labs: pl.LazyFrame,
    timeseries_resp: pl.LazyFrame,
    OUTPATH: str,
) -> pl.LazyFrame:
    print("reprOMOPIZE - measurement")

    ID = _ID(
        patient_information,
        [
            "Time Relative to First ICU Admission (seconds)",
            "Admission Time (24h)",
        ],
    )
    CONCEPTS = (
        CONCEPT.filter(
            pl.col("domain_id").is_in(["Measurement", "Observation"]),
            pl.col("concept_class_id").is_in(
                [
                    "Clinical Observation",
                    "Observable Entity",
                    "Staging / Scales",
                    "Lab Test",
                ]
            ),
            pl.col("standard_concept") == "S",
        )
        .select("concept_id", "concept_name")
        .group_by("concept_name")
        .first()
    )
    UNITS = vars.MEASUREMENT_UNIT_CONCEPT_IDS

    def _make_datetime(data: pl.LazyFrame) -> pl.LazyFrame:
        """
        make time columns to datetime
        """
        print("reprOMOPIZE - measurement - making datetime")
        return (
            data.join(ID, on="Global ICU Stay ID", how="right")
            .with_columns(
                #####################
                # VISIT_OCCURRENCE_ID
                # Create the visit_occurrence_id column with a hash of the Global ICU Stay ID
                # NOTE: same as in the VISIT_OCCURRENCE table
                pl.col("Global ICU Stay ID")
                .hash()
                .alias("visit_occurrence_id"),
                ######################
                # MEASUREMENT_DATETIME
                # Create the measurement_datetime column with the datetime of the measurement
                (
                    DAY_ZERO.dt.combine(pl.col("Admission Time (24h)"))
                    + pl.duration(
                        seconds=pl.col("Time Relative to Admission (seconds)")
                    )
                    + pl.duration(
                        seconds=pl.col(
                            "Time Relative to First ICU Admission (seconds)"
                        )
                    )
                ).alias("measurement_datetime"),
            )
            .drop(
                "Global ICU Stay ID",
                "Time Relative to First ICU Admission (seconds)",
                "Admission Time (24h)",
                "Time Relative to Admission (seconds)",
            )
        )

    def _unpivot(data: pl.LazyFrame) -> pl.LazyFrame:
        """
        unpivot the data
        """
        print("reprOMOPIZE - measurement - unpivoting")
        return data.unpivot(
            index=["person_id", "visit_occurrence_id", "measurement_datetime"],
            variable_name="variable_name",
            value_name="value_as_number",
        )

    def _destruct_after_unpivot(data: pl.LazyFrame) -> pl.LazyFrame:
        """
        de-struct the data after the unpivot operation
        """
        print("reprOMOPIZE - measurement - de-structing after unpivot")

        LOINC_code_list = (
            data.select("value_as_number")
            .unnest("value_as_number")
            .select("LOINC")
            .unique()
            .drop_nulls()
            .collect(streaming=True)
            .to_series()
            .to_list()
        )
        LOINC_code_frame = omop.get_concept_names_from_codes(
            LOINC_code_list, return_dict=False
        ).lazy()

        return (
            data.unnest("value_as_number")
            .rename({"value": "value_as_number", "LOINC": "variable_code"})
            .drop("system", "method", "time")
            .join(
                LOINC_code_frame,
                left_on="variable_code",
                right_on="concept_code",
            )
            .rename({"concept_name": "variable_name"})
            .drop("variable_code")
        )

    def _add_units(data: pl.LazyFrame) -> pl.LazyFrame:
        """
        add the units to the data
        """
        print("reprOMOPIZE - measurement - adding units")
        return data.join(
            UNITS.lazy(),
            left_on="variable_name",
            right_on="measurement",
            how="left",
            coalesce=True,
        ).rename({"unit": "unit_source_value"})

    def _conceptualize(data: pl.LazyFrame) -> pl.LazyFrame:
        """
        add the concept_id to the data
        """
        print("reprOMOPIZE - measurement - adding concept_id")

        return (
            data
            # Create the measurement_concept_id column with the concept_id of the Measurement
            .join(
                CONCEPTS,
                left_on="variable_name",
                right_on="concept_name",
                how="left",
            )
            .rename(
                {
                    "variable_name": "measurement_source_value",
                    "concept_id": "measurement_concept_id",
                }
            )
            .drop_nulls("value_as_number")
        )

    # Extract the measurement information
    # Create a unique prefix for temp files
    # Create a subdirectory in the temp directory for output files
    output_subdir = os.path.join(OUTPATH, "reprOMOPIZE_output")
    os.makedirs(output_subdir, exist_ok=True)

    # Set up pattern for intermediate temp files
    temp_prefix = "measurement_"
    temp_pattern = os.path.join(output_subdir, f"{temp_prefix}*.parquet")

    # Register cleanup function to run at script exit
    def cleanup_temp_files():
        for f in glob.glob(temp_pattern):
            try:
                os.remove(f)
            except Exception:
                pass  # If we can't delete, just continue

    atexit.register(cleanup_temp_files)

    # Save each dataset to temp file
    temp_file1 = os.path.join(
        output_subdir, f"{temp_prefix}heights_weights.parquet"
    )
    temp_file2 = os.path.join(output_subdir, f"{temp_prefix}vitals.parquet")
    temp_file3 = os.path.join(output_subdir, f"{temp_prefix}labs.parquet")

    # Process heights and weights
    (
        patient_information.select(
            "Global ICU Stay ID",
            pl.lit(0).alias("Time Relative to Admission (seconds)"),
            pl.col("Admission Height (cm)").alias("Patient height"),
            pl.col("Admission Weight (kg)").alias("Body weight"),
        )
        .pipe(_make_datetime)
        .pipe(_unpivot)
        .pipe(_add_units)
        .cast({"value_as_number": float})
        .sink_parquet(temp_file1)
    )

    # Process vitals data
    (
        timeseries_vitals.drop("Heart rate rhythm")
        .pipe(_make_datetime)
        .pipe(_unpivot)
        .pipe(_add_units)
        .cast({"value_as_number": float})
        .sink_parquet(temp_file2)
    )

    # Process labs data
    (
        timeseries_labs
        .pipe(_make_datetime)
        .pipe(_unpivot)
        .pipe(_add_units)
        .pipe(_destruct_after_unpivot)
        .cast({"value_as_number": float})
        .sink_parquet(temp_file3)
    )

    # Now scan and concat the temporary files
    return (
        pl.concat(
            [
                pl.scan_parquet(temp_file1),
                pl.scan_parquet(temp_file2),
                pl.scan_parquet(temp_file3),
            ],
            how="vertical",
        )
        .pipe(_conceptualize)
        .with_columns(
            pl.col("measurement_datetime").dt.date().alias("measurement_date"),
            pl.lit(32817).alias("measurement_type_concept_id"),
        )
        .with_row_index("measurement_id")
        .pipe(_add_missing_fields, "measurement")
    )


# endregion


# region PERSON
# This table serves as the central identity management for all Persons in the
# database. It contains records that uniquely identify each person or patient,
# and some demographic information.
def person(
    CONCEPT: pl.LazyFrame, patient_information: pl.LazyFrame
) -> pl.LazyFrame:
    print("reprOMOPIZE - person")

    # Dates of the databases
    # eICU: 2014 to 2015
    # HiRID: 2008-01 to 2016-06
    # MIMIC-III: 2001 to 2012
    # MIMIC-IV: 2008 to 2022
    # SICdb: 2013 to 2021
    # UMCdb: 2003 to 2016

    RACE_CONCEPTS = CONCEPT.filter(pl.col("domain_id") == "Race").select(
        "concept_id", "concept_name"
    )
    ETHNICITY_CONCEPTS = CONCEPT.filter(
        pl.col("domain_id") == "Ethnicity"
    ).select("concept_id", "concept_name")

    # Extract the person information
    return (
        patient_information.filter(
            pl.col("ICU Stay Sequential Number (per Person ID)").is_in(
                [1, None]
            )
        )
        .select(
            "Global Person ID",
            "Gender",
            pl.col("Ethnicity").cast(str),
            "Admission Age (years)",
            "Care Site",
            "Source Dataset",
        )
        .with_columns(
            ###########
            # PERSON_ID
            # Create the person_id column with a hash of the Global Person ID
            pl.col("Global Person ID").hash().alias("person_id"),
            # Create the person_source_value column with the Global Person ID for backreference
            pl.col("Global Person ID").alias("person_source_value"),
            ###################
            # GENDER_CONCEPT_ID
            # Create gender_concept_id column based on the Gender column
            pl.when(pl.col("Gender") == "Male")
            .then(pl.lit(8507))
            .when(pl.col("Gender") == "Female")
            .then(pl.lit(8532))
            .otherwise(None)
            .alias("gender_concept_id"),
            # Create gender_source_value column with the Gender column for backreference
            pl.col("Gender").alias("gender_source_value"),
            ###################
            # YEAR_OF_BIRTH
            # Create the year_of_birth column based on the source database timeframe and the admission age
            (2000 - pl.col("Admission Age (years)")).alias("year_of_birth"),
            # pl.when(pl.col("Source Dataset") == "eICU-CRD")
            # .then(pl.lit(2015) - pl.col("Admission Age (years)"))
            # .when(pl.col("Source Dataset") == "HiRID")
            # .then(pl.lit(2016) - pl.col("Admission Age (years)"))
            # .when(pl.col("Source Dataset") == "MIMIC-III")
            # .then(pl.lit(2012) - pl.col("Admission Age (years)"))
            # .when(pl.col("Source Dataset") == "MIMIC-IV")
            # .then(pl.lit(2022) - pl.col("Admission Age (years)"))
            # .when(pl.col("Source Dataset") == "SICdb")
            # .then(pl.lit(2021) - pl.col("Admission Age (years)"))
            # .when(pl.col("Source Dataset") == "UMCdb")
            # .then(pl.lit(2016) - pl.col("Admission Age (years)"))
            # .alias("year_of_birth"),
            ##############
            # CARE_SITE_ID
            # Create the care_site_id column with a hash of the Care Site
            # NOTE: same as in the CARE_SITE table
            pl.col("Care Site").hash().alias("care_site_id"),
        )
        ###################
        # RACE_CONCEPT_ID
        # Create the race_concept_id column based on the Ethnicity column
        # relevant ETL Conventions:
        # Mixed races are not supported.
        # If a clear race or ethnic background cannot be established, use Concept_Id 0.
        .join(
            RACE_CONCEPTS,
            left_on="Ethnicity",
            right_on="concept_name",
            how="left",
        )
        .rename({"concept_id": "race_concept_id"})
        .with_columns(pl.col("race_concept_id").fill_null(0))
        ###################
        # ETHNICITY_CONCEPT_ID
        # Create the ethnicity_concept_id column based on the Ethnicity column
        .join(
            ETHNICITY_CONCEPTS,
            left_on="Ethnicity",
            right_on="concept_name",
            how="left",
        )
        .rename({"concept_id": "ethnicity_concept_id"})
        .with_columns(
            pl.col("ethnicity_concept_id").fill_null(
                pl.lit(38003564)
            ),  # Not Hispanic or Latino
        )
        .pipe(_add_missing_fields, "person")
        .unique()
    )


# endregion


# region PROCEDURE_OCCURRENCE
# This table contains records of activities or processes ordered by, or carried
# out by, a healthcare provider on the patient with a diagnostic or therapeutic
# purpose.
def procedure_occurrence(
    CONCEPT: pl.LazyFrame,
    patient_information: pl.LazyFrame,
    procedures: pl.LazyFrame,
) -> pl.LazyFrame:
    print("reprOMOPIZE - procedure_occurrence")

    ID = _ID(
        patient_information,
        [
            "Global Person ID",
            "Time Relative to First ICU Admission (seconds)",
            "Admission Time (24h)",
        ],
    )
    CONCEPTS = CONCEPT.filter(
        pl.col("domain_id") == "Procedure",
        pl.col("concept_class_id") == "Procedure",
        pl.col("standard_concept") == "S",
    ).select("concept_id", "concept_name")

    return (
        procedures.join(ID, on="Global Person ID", how="right")
        .join(
            CONCEPTS,
            left_on="Procedure Description",
            right_on="concept_name",
            how="left",
        )
        .rename({"concept_id": "procedure_concept_id"})
        .with_columns(
            #####################
            # VISIT_OCCURRENCE_ID
            # Create the visit_occurrence_id column with a hash of the Global ICU Stay ID
            # NOTE: same as in the VISIT_OCCURRENCE table
            pl.col("Global ICU Stay ID").hash().alias("visit_occurrence_id"),
            ################################
            # PROCEDURE_DATETIME
            # Create the procedure_datetime column with the datetime of the device exposure
            (
                DAY_ZERO.dt.combine(pl.col("Admission Time (24h)"))
                + pl.duration(
                    seconds=pl.col(
                        "Procedure Start Relative to Admission (seconds)"
                    )
                )
                + pl.duration(
                    seconds=pl.col(
                        "Time Relative to First ICU Admission (seconds)"
                    )
                )
            ).alias("procedure_datetime"),
            ##############################
            # PROCEDURE_END_DATETIME
            # Create the procedure_end_datetime column with the datetime of the device exposure
            (
                DAY_ZERO.dt.combine(pl.col("Admission Time (24h)"))
                + pl.duration(
                    seconds=pl.col(
                        "Procedure End Relative to Admission (seconds)"
                    )
                )
                + pl.duration(
                    seconds=pl.col(
                        "Time Relative to First ICU Admission (seconds)"
                    )
                )
            ).alias("procedure_end_datetime"),
            ###########################
            # PROCEDURE_TYPE_CONCEPT_ID
            # 32817 = EHR
            pl.lit(32817).alias("procedure_type_concept_id"),
            ########################
            # PROCEDURE_SOURCE_VALUE
            # Create the procedure_source_value column with the Procedure for backreference
            pl.col("Procedure Description").alias("procedure_source_value"),
        )
        .with_columns(
            ######################
            # PROCEDURE_DATE
            # Create the procedure_date column with the date of the device exposure
            pl.col("procedure_datetime").dt.date().alias("procedure_date"),
            ####################
            # PROCEDURE_END_DATE
            # Create the procedure_end_date column with the date of the device exposure
            pl.col("procedure_end_datetime")
            .dt.date()
            .alias("procedure_end_date"),
        )
        .drop_nulls("procedure_concept_id")
        .with_row_index("procedure_occurrence_id")
        .pipe(_add_missing_fields, "procedure_occurrence")
        .unique()
    )


# endregion


# region VISIT_OCCURRENCE
# This table contains Events where Persons engage with the healthcare system
# for a duration of time. They are often also called “Encounters”. Visits are
# defined by a configuration of circumstances under which they occur, such as
# (i) whether the patient comes to a healthcare institution, the other way
# around, or the interaction is remote, (ii) whether and what kind of trained
# medical staff is delivering the service during the Visit, and (iii) whether
# the Visit is transient or for a longer period involving a stay in bed.
def visit_occurrence(patient_information: pl.LazyFrame) -> pl.LazyFrame:
    print("reprOMOPIZE - visit_occurrence")

    ID = _ID(
        patient_information, ["Time Relative to First ICU Admission (seconds)"]
    )

    # Extract the visit occurrence information
    return (
        patient_information.drop(
            "Time Relative to First ICU Admission (seconds)"
        )
        .join(ID, on="Global ICU Stay ID", how="right")
        .with_columns(
            #####################
            # VISIT_OCCURRENCE_ID
            # Create the visit_occurrence_id column with a hash of the Global ICU Stay ID
            pl.col("Global ICU Stay ID").hash().alias("visit_occurrence_id"),
            ###########
            # PERSON_ID
            # Create the person_id column with a hash of the Global Person ID
            # NOTE: same as in the PERSON table
            pl.col("Global Person ID").hash().alias("person_id"),
            ##################
            # VISIT_CONCEPT_ID
            # 32037 = Intensive Care
            pl.lit(32037).alias("visit_concept_id"),
            #######################
            # VISIT_TYPE_CONCEPT_ID
            # 44818518 = Visit derived from EHR record
            pl.lit(44818518).alias("visit_type_concept_id"),
            ######################
            # VISIT_START_DATETIME
            # Create the visit_start_datetime column with the start datetime of the ICU stay
            (
                DAY_ZERO.dt.combine(pl.col("Admission Time (24h)"))
                + pl.duration(
                    seconds=pl.col(
                        "Time Relative to First ICU Admission (seconds)"
                    )
                )
            ).alias("visit_start_datetime"),
            ####################
            # VISIT_END_DATETIME
            # Create the visit_end_datetime column with the end datetime of the ICU stay
            (
                DAY_ZERO.dt.combine(pl.col("Admission Time (24h)"))
                + pl.duration(
                    seconds=pl.col(
                        "Time Relative to First ICU Admission (seconds)"
                    )
                )
                + pl.duration(
                    seconds=pl.col("ICU Length of Stay (days)") * SECONDS_IN_DAY
                )
            ).alias("visit_end_datetime"),
            ##############
            # CARE_SITE_ID
            # Create the care_site_id column with a hash of the Care Site
            # NOTE: same as in the CARE_SITE table
            pl.col("Care Site").hash().alias("care_site_id"),
            ##########################
            # ADMITTED_FROM_CONCEPT_ID
            # Create the admitted_from_concept_id column with the concept_id of the admission location
            ###
            # ADMITTED_FROM_SOURCE_VALUE
            # Create the admitted_from_source_value column admission location for backreference
            # pl.col("Admission Location").alias("admitted_from_source_value"),
            ##########################
            # DISCHARGED_TO_CONCEPT_ID
            # Create the discharged_to_concept_id column with the concept_id of the discharge location
            ###
            # DISCHARGED_TO_SOURCE_VALUE
            # Create the discharged_to_source_value column with the discharge location for backreference
            # pl.col("Discharge Location").alias("discharged_to_source_value"),
        )
        .with_columns(
            ##################
            # VISIT_START_DATE
            # Create the visit_start_date column with the start date of the ICU stay
            pl.col("visit_start_datetime").dt.date().alias("visit_start_date"),
            ################
            # VISIT_END_DATE
            # Create the visit_end_date column with the end date of the ICU stay
            pl.col("visit_end_datetime").dt.date().alias("visit_end_date"),
        )
        # ###############################
        # # PRECEDING_VISIT_OCCURRENCE_ID
        # .join(
        #     patient_information.select(
        #         "Global Person ID",
        #         "Global ICU Stay ID",
        #         "ICU Stay Sequential Number (per Person ID)",
        #     )
        #     .with_columns(
        #         (pl.col("ICU Stay Sequential Number (per Person ID)") - 1),
        #         pl.col("Global ICU Stay ID")
        #         .hash()
        #         .alias("preceding_visit_occurrence_id"),
        #     )
        #     .drop("Global ICU Stay ID"),
        #     on=[
        #         "Global Person ID",
        #         "ICU Stay Sequential Number (per Person ID)",
        #     ],
        #     how="left",
        # )
        .pipe(_add_missing_fields, "visit_occurrence")
        .unique()
    )


################################################################################
# region VOCABULARIES
################################################################################
def VOCABULARIES(
    outpath: str,
    CONCEPT: pl.LazyFrame,
    CONCEPT_RELATIONSHIP: pl.LazyFrame,
    CONCEPT_ANCESTOR: pl.LazyFrame,
    CONCEPT_CLASS: pl.LazyFrame,
    CONCEPT_SYNONYM: pl.LazyFrame,
    DOMAIN: pl.LazyFrame,
    RELATIONSHIP: pl.LazyFrame,
    VOCABULARY: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Initializes the vocabulary tables for the OMOP CDM, including only the
    concepts that are used in the reprodICU data.

    Args:
        CONCEPT (pl.LazyFrame): _description_
        CONCEPT_RELATIONSHIP (pl.LazyFrame): _description_
        CONCEPT_ANCESTOR (pl.LazyFrame): _description_
        CONCEPT_CLASS (pl.LazyFrame): _description_
        CONCEPT_SYNONYM (pl.LazyFrame): _description_
        DOMAIN (pl.LazyFrame): _description_
        RELATIONSHIP (pl.LazyFrame): _description_
        VOCABULARY (pl.LazyFrame): _description_
    """

    print("reprOMOPIZE - VOCABULARIES")

    # Iterate over output directory, scan all files, select all columns
    # containing "concept_id" and put the unique values in a list
    concept_ids = []
    for table_path in [
        f
        for f in glob.glob(os.path.join(outpath, "*.parquet"))
        if os.path.basename(f).islower()
    ]:
        table_df = pl.scan_parquet(table_path)
        concept_id_cols = [
            col for col in table_df.columns if "concept_id" in col
        ]
        for col in concept_id_cols:
            concept_ids += (
                table_df.select(pl.col(col))
                .unique()
                .collect()
                .to_series()
                .to_list()
            )

    concept_ids = list(set(concept_ids))
    print(f"reprOMOPIZE - found {len(concept_ids)} unique concept_ids")

    # Filter the vocabulary tables to only include the concept_ids that are in
    # the reprodICU data.
    CONCEPT = CONCEPT.filter(pl.col("concept_id").is_in(concept_ids))
    CONCEPT_RELATIONSHIP = CONCEPT_RELATIONSHIP.filter(
        pl.col("concept_id_1").is_in(concept_ids)
        | pl.col("concept_id_2").is_in(concept_ids)
    )
    CONCEPT_ANCESTOR = CONCEPT_ANCESTOR.filter(
        pl.col("ancestor_concept_id").is_in(concept_ids)
        | pl.col("descendant_concept_id").is_in(concept_ids)
    )
    CONCEPT_SYNONYM = CONCEPT_SYNONYM.filter(
        pl.col("concept_id").is_in(concept_ids)
    )

    # Get the relevant relating IDs from the other tables
    # -> CONCEPT_CLASS: concept_class_id
    # -> DOMAIN: domain_id
    # -> RELATIONSHIP: relationship_id (from CONCEPT_RELATIONSHIP)
    # -> VOCABULARY: vocabulary_id

    CONCEPT_CLASS = CONCEPT_CLASS.filter(
        pl.col("concept_class_id").is_in(
            CONCEPT.select("concept_class_id").collect().to_series().to_list()
        )
    )
    DOMAIN = DOMAIN.filter(
        pl.col("domain_id").is_in(
            CONCEPT.select("domain_id").collect().to_series().to_list()
        )
    )
    RELATIONSHIP = RELATIONSHIP.filter(
        pl.col("relationship_id").is_in(
            CONCEPT_RELATIONSHIP.select("relationship_id")
            .collect()
            .to_series()
            .to_list()
        )
    )
    VOCABULARY = VOCABULARY.filter(
        pl.col("vocabulary_id").is_in(
            CONCEPT.select("vocabulary_id").collect().to_series().to_list()
        )
    )

    # Write the filtered tables to parquet files
    CONCEPT.sink_parquet(outpath + "CONCEPT.parquet")
    CONCEPT_RELATIONSHIP.sink_parquet(outpath + "CONCEPT_RELATIONSHIP.parquet")
    CONCEPT_ANCESTOR.sink_parquet(outpath + "CONCEPT_ANCESTOR.parquet")
    CONCEPT_CLASS.sink_parquet(outpath + "CONCEPT_CLASS.parquet")
    CONCEPT_SYNONYM.sink_parquet(outpath + "CONCEPT_SYNONYM.parquet")
    DOMAIN.sink_parquet(outpath + "DOMAIN.parquet")
    RELATIONSHIP.sink_parquet(outpath + "RELATIONSHIP.parquet")
    VOCABULARY.sink_parquet(outpath + "VOCABULARY.parquet")

    return


################################################################################
# region OTHER
################################################################################
def other():
    """
    add missing tables to the output directory
    """
    tables = (
        pl.read_csv("mappings/OMOP_CDMv5.4_Field_Level.csv")
        .select("cdmTableName")
        .unique()
        .to_series()
        .to_list()
    )

    print(os.listdir(OUTPATH))

    for table in tables:
        if ((table + ".parquet") not in os.listdir(OUTPATH)) and (
            (table.upper() + ".parquet") not in os.listdir(OUTPATH)
        ):
            print(f"reprOMOPIZE - adding missing table: {table}")
            pl.DataFrame().pipe(_add_missing_fields, table).write_parquet(
                OUTPATH + table + ".parquet"
            )


################################################################################
# region MAIN
################################################################################
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
        default="../reprodICU_files_OMOP/",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        help="List of specific datasets to convert, e.g. eICU, HiRID",
        default=["eICU", "HiRID", "MIMIC-III", "MIMIC-IV", "SICdb", "UMCdb"],
    )
    args = parser.parse_args()

    # Initialize paths
    paths = reprodICUPaths()
    omop = Vocabulary(paths)
    vars = GlobalVars(paths)

    INPATH = args.input
    OUTPATH = args.output
    VOCABPATH = args.vocab
    os.makedirs(args.output, exist_ok=True)

    # Load the reprodICU data
    diagnoses = pl.scan_parquet(INPATH + "diagnoses_imputed.parquet")
    medications = pl.scan_parquet(INPATH + "medications.parquet")
    patient_information = pl.scan_parquet(
        INPATH + "patient_information.parquet"
    )
    procedures = pl.scan_parquet(INPATH + "procedures.parquet")
    timeseries_vitals = pl.scan_parquet(INPATH + "timeseries_vitals.parquet")
    timeseries_labs = pl.scan_parquet(INPATH + "timeseries_labs.parquet")
    timeseries_resp = pl.scan_parquet(INPATH + "timeseries_resp.parquet")

    ########
    # FILTER
    # filter patient_information to only include the datasets specified in the
    # command line arguments
    patient_information = patient_information.filter(
        pl.col("Source Dataset").str.contains_any(args.datasets)
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

    ############
    # CONVERTING
    # Convert the reprodICU structure to the OMOP CDM structure
    # Tables with transformed IDs
    care_site(patient_information).sink_parquet(OUTPATH + "care_site.parquet")
    (
        condition_occurrence(CONCEPT, patient_information, diagnoses)
        .collect()
        .write_parquet(OUTPATH + "condition_occurrence.parquet")
    )
    (
        death(patient_information)
        .collect()
        .write_parquet(OUTPATH + "death.parquet")
    )
    (
        device_exposure(CONCEPT, patient_information, procedures)
        .collect()
        .write_parquet(OUTPATH + "device_exposure.parquet")
    )
    location(patient_information).sink_parquet(OUTPATH + "location.parquet")
    (
        person(CONCEPT, patient_information)
        .collect()
        .write_parquet(OUTPATH + "person.parquet")
    )
    (
        procedure_occurrence(CONCEPT, patient_information, procedures)
        .collect()
        .write_parquet(OUTPATH + "procedure_occurrence.parquet")
    )
    (
        visit_occurrence(patient_information).sink_parquet(
            OUTPATH + "visit_occurrence.parquet"
        )
    )

    # Tables with row indices
    (
        drug_exposure(CONCEPT, medications, patient_information)
        .collect(streaming=True)
        .write_parquet(OUTPATH + "drug_exposure.parquet")
    )
    (
        measurement(
            CONCEPT,
            patient_information,
            timeseries_vitals,
            timeseries_labs,
            timeseries_resp,
        )
        .collect(streaming=True)
        .write_parquet(OUTPATH + "measurement.parquet")
    )

    ##################
    # ADD VOCABULARIES
    VOCABULARIES(
        OUTPATH,
        CONCEPT,
        CONCEPT_RELATIONSHIP,
        CONCEPT_ANCESTOR,
        CONCEPT_CLASS,
        CONCEPT_SYNONYM,
        DOMAIN,
        RELATIONSHIP,
        VOCABULARY,
    )

    ####################
    # ADD MISSING TABLES
    # other()

    print("reprOMOPIZE - done")
