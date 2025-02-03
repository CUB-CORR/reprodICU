# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: Converts the reprodICU structure to the Common Longitudinal ICU Format (CLIF) structure.
# The script is based on the CLIF-2.0

# Input: reprodICU structure
# Output: CLIF structure

# Usage: python Z_reprodiCLIF.py

# Importing necessary libraries
import argparse
import os
import sys

import polars as pl
import yaml
from helpers.helper_OMOP import Vocabulary
from helpers.C_harmonize.C_harmonize_diagnoses import DiagnosesHarmonizer

SECONDS_IN_DAY = 86400
DAYS_IN_YEAR = 365.25
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
# The CLIF schema table contains a list of fields that are used in the
# observational data tables. Each field is uniquely identified by a field name.
def _field_level(table_name: str, return_required: bool = False) -> list:
    """
    return a list of fields for the table in the CLIF schema in order
    """
    field_level_ = pl.read_csv("mappings/CLIF_DataDictionary.csv").filter(
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
    data: pl.LazyFrame, table_name: str, check_required: bool = False
) -> pl.LazyFrame:
    """
    add missing fields to the data
    """
    fields, required = _field_level(table_name, return_required=True)
    columns = data.collect_schema().names()

    for field, req in zip(fields, required):
        if field not in columns:
            if req == "Yes" and check_required:
                raise ValueError(
                    f"Field {field} is required for the {table_name} table"
                )

            data = data.with_columns(pl.lit(None).cast(pl.Int8).alias(field))

    return data.select(fields)


# The _ID function creates a table with the patient ID and additional columns
def _ID(
    patient_information: pl.LazyFrame, additional_columns: list = []
) -> pl.LazyFrame:
    return patient_information.select(
        "Global ICU Stay ID",
        "Global Hospital Stay ID",
        "Global Person ID",
        *additional_columns,
    )


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
            "Global Hospital Stay ID",
            "icu_admission_dttm",
        )
    )


# endregion


#######################################
# GENERAL INPATIENT TABLES
#######################################


# region Patient
# This table contains demographic information about the patient that does not
# vary between hospitalizations.
def Patient(patient_information: pl.LazyFrame) -> pl.LazyFrame:
    print("reprodiCLIF - Patient")
    return (
        patient_information.with_columns(
            # patient_id
            # Unique identifier for each patient. This is presumed to be a distinct individual.
            pl.col("Global Person ID").alias("patient_id"),
            # race_name
            # Patient race string from source data
            # (N/A)
            # race_category
            # A standardized CDE description of patient’s race per the US Census permissible values.
            # The source data may contain different strings for race.
            pl.col("Ethnicity")
            .cast(str)
            .replace(
                {
                    "Black / African American": "Black or African American",
                    "Hispanic or Latino": "Other",
                    "Native American": "American Indian or Alaska Native",
                }
            )
            .alias("race_category"),
            # ethnicity_name
            # Patient ethnicity string from source data
            # (N/A)
            # ethnicity_category
            # Description of patient’s ethnicity per the US census definition.
            # The source data may contain different strings for ethnicity.
            pl.col("Ethnicity")
            .cast(str)
            .replace_strict(
                {"Hispanic or Latino": "Hispanic", "Unknown": "Unknown"},
                default="Non-Hispanic",
            )
            .alias("ethnicity_category"),
            # sex_name
            # Patient’s biological sex as given in the source data.
            # (N/A)
            # sex_category
            # Patient’s biological sex.
            pl.col("Gender")
            .replace({"Other": "Unknown"})
            .alias("sex_category"),
            # birth_date
            # Patient’s date of birth.
            (
                DAY_ZERO
                - pl.duration(
                    days=pl.col("Admission Age (years)") * DAYS_IN_YEAR
                )
            ).alias("birth_date"),
            # death_dttm
            # Patient’s death date, including time.
            (
                DAY_ZERO
                + pl.duration(
                    days=pl.when(pl.col("Mortality in ICU"))
                    .then(pl.col("ICU Length of Stay (days)"))
                    .otherwise(
                        pl.col("ICU Length of Stay (days)")
                        + pl.col("Mortality after ICU discharge (days)")
                    )
                )
            ).alias("death_dttm"),
            # language_name
            # Patient’s preferred language.
            # (N/A)
            # language_category
            # Maps language_name to a standardized list of spoken languages.
            # (N/A)
        )
        .unique()
        .pipe(_add_missing_fields, "Patient")
    )


# endregion


# region Hospitalization
# The hospitalization table contains information about each hospitalization
# event. Each row in this table represents a unique hospitalization event for a
# patient. This table is inspired by the visit_occurance OMOP table but is
# specific to inpatient hospitalizations (including those that begin in the
# emergency room).
def Hospitalization(patient_information: pl.LazyFrame) -> pl.LazyFrame:
    print("reprodiCLIF - Hospitalization")
    return (
        patient_information
        # Select only the first ICU stay for each patient
        .filter(
            (pl.col("ICU Stay Sequential Number (per Person ID)") == 1)
            | (pl.col("ICU Stay Sequential Number (per Person ID)").is_null())
        )
        .with_columns(
            # patient_id
            # Unique identifier for each patient. This is presumed to be a distinct individual.
            pl.col("Global Person ID").alias("patient_id"),
            # hospitalization_id
            # Unique identifier for each hospitalization event.
            pl.col("Global Hospital Stay ID").alias("hospitalization_id"),
            # hospitalization_joined_id
            # Unique identifier for each continuous inpatient stay in a health system which may span different hospitals (Optional).
            # (N/A)
            # admission_dttm
            # Date and time the patient is admitted to the hospital. Datetime format should be %Y-%m-%d %H:%M:%S.
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
            ).alias("admission_dttm"),
            # discharge_dttm
            # Date and time the patient is discharged from the hospital. Datetime format should be %Y-%m-%d %H:%M:%S.
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
                + pl.duration(days=pl.col("Hospital Length of Stay (days)"))
            ).alias("discharge_dttm"),
            # age_at_admission
            # Age of the patient at the time of admission, in years.
            pl.col("Admission Age (years)").alias("age_at_admission"),
            # admission_type_name
            # Type of inpatient admission. Original string from the source data.
            # (N/A)
            # admission_type_category
            # Admission disposition mapped to mCIDE categories.
            # (N/A)
            # discharge_name
            # Original discharge disposition name string recorded in the raw data.
            # (N/A)
            # discharge_category
            # Maps discharge_name to a standardized list of discharge categories.
            pl.col("Discharge Location")
            .cast(str)
            .replace(
                {
                    "Hospital": "Still Admitted",
                    "Death": "Expired",
                    "Other ICU": "Still Admitted",
                    "Operating Room": "Still Admitted",
                    "Rehabilitation": "Acute Inpatient Rehab Facility",
                    "Nursing Facility": "Skilled Nursing Facility (SNF)",
                    "Psychiatric Facility": "Psychiatric Hospital",
                    "High-Dependency Unit": "Still Admitted",
                    "Against Medical Advice": "Against Medical Advice (AMA)",
                    "Unknown": "Missing",
                }
            )
            .alias("discharge_category"),
            # zipcode_nine_digit
            # Patient’s 9 digit zip code, used to link with other indices such as ADI and SVI.
            # (N/A)
            # zipcode_five_digit
            # Patient’s 5 digit zip code, used to link with other indices such as ADI and SVI.
            # (N/A)
            # census_block_code
            # 15 digit FIPS code.
            # (N/A)
            # census_block_group_code
            # 12 digit FIPS code.
            # (N/A)
            # census_tract
            # 11 digit FIPS code.
            # (N/A)
            # state_code
            # 2 digit FIPS code.
            # (N/A)
            # county_code
            # 5 digit FIPS code.
            # (N/A)
        )
        .unique()
        .pipe(_add_missing_fields, "Hospitalization")
    )


# endregion


# region ADT
# The admission, discharge, and transfer (ADT) table is a start-stop
# longitudinal dataset that contains information about each patient’s movement
# within the hospital. It also has a hospital_id field to distinguish between
# different hospitals within a health system.
def ADT(patient_information: pl.LazyFrame) -> pl.LazyFrame:
    print("reprodiCLIF - ADT")
    return (
        patient_information.with_columns(
            # hospitalization_id
            # ID variable for each patient encounter.
            pl.col("Global Hospital Stay ID").alias("hospitalization_id"),
            # hospital_id
            # Assign a unique ID to each hospital within a health system.
            pl.col("Care Site").alias("hospital_id"),
            # hospital_type
            # Maps hospital_id to a standardized list of hospital types.
            # (N/A)
            # in_dttm
            # Start date and time at a particular location. Datetime format should be %Y-%m-%d %H:%M:%S.
            # TODO
            # out_dttm
            # End date and time at a particular location. Datetime format should be %Y-%m-%d %H:%M:%S.
            # TODO
            # location_name
            # Location of the patient inside the hospital. This field is used to store the patient location from the source data. It is not used for analysis.
            # (N/A)
            # location_category
            # Maps location_name to a standardized list of ADT location categories.
            pl.lit("icu").alias("location_category"),
            # location_type
            # Maps location_name to a standardized list of ADT location types.
            pl.col("Unit Type")
            .cast(str)
            .replace_strict(
                {
                    "Cardiac": "mixed_cardiac_icu",
                    "Neurological": "mixed_neuro_icu",
                    "Neonatal": "general_icu",
                    "Medical": "medical_icu",
                    "Medical-Surgical": "general_icu",
                    "Pediatric": "general_icu",
                    "Surgical": "surgical_icu",
                    "Trauma": "surgical_icu",
                },
                default="general_icu",
            )
            .alias("location_type"),
        )
        .unique()
        .pipe(_add_missing_fields, "ADT")
    )


# endregion


# region Vitals
# The vitals table is a long-form (one vital sign per row) longitudinal table.
def Vitals(
    patient_information: pl.LazyFrame, timeseries_vitals: pl.LazyFrame
) -> pl.LazyFrame:
    print("reprodiCLIF - Vitals")
    return (
        timeseries_vitals.select(
            "Global ICU Stay ID",
            "Time Relative to Admission (seconds)",
            "Temperature",
            "Heart rate",
            "Invasive systolic arterial pressure",
            "Invasive mean arterial pressure",
            "Invasive diastolic arterial pressure",
            "Non-invasive systolic arterial pressure",
            "Non-invasive mean arterial pressure",
            "Non-invasive diastolic arterial pressure",
            "Peripheral oxygen saturation",
            "Respiratory rate",
        )
        # Rename columns
        .rename(
            {
                "Temperature": "temp_c",
                "Heart rate": "heart_rate",
                "Peripheral oxygen saturation": "spo2",
                "Respiratory rate": "respiratory_rate",
            }
        )
        # Combine invasive and non-invasive blood pressure measurements
        # -> prefer invasive measurements
        .with_columns(
            pl.coalesce(
                "Invasive systolic arterial pressure",
                "Non-invasive systolic arterial pressure",
            ).alias("sbp"),
            pl.coalesce(
                "Invasive mean arterial pressure",
                "Non-invasive mean arterial pressure",
            ).alias("map"),
            pl.coalesce(
                "Invasive diastolic arterial pressure",
                "Non-invasive diastolic arterial pressure",
            ).alias("dbp"),
        )
        # Join with patient_information to get the ICU admission time
        .join(
            _ID_ICUOFFSET(patient_information),
            on="Global ICU Stay ID",
            how="left",
        )
        # Unpivot the table
        .unpivot(
            on=[
                "temp_c",
                "heart_rate",
                "sbp",
                "map",
                "dbp",
                "spo2",
                "respiratory_rate",
            ],
            index=[
                "Global Hospital Stay ID",
                "Time Relative to Admission (seconds)",
                "icu_admission_dttm",
            ],
            variable_name="vital_category",
            value_name="vital_value",
        )
        # Add missing fields
        .with_columns(
            # hospitalization_id
            # ID variable for each patient encounter.
            pl.col("Global Hospital Stay ID").alias("hospitalization_id"),
            # recorded_dttm
            # Date and time when the vital is recorded. Datetime format should be %Y-%m-%d %H:%M:%S.
            (
                pl.col("icu_admission_dttm")
                + pl.duration(
                    seconds=pl.col("Time Relative to Admission (seconds)")
                )
            ).alias("recorded_dttm"),
            # vital_name
            # This field is used to store the description of the flowsheet measure from the source data. This field is not used for analysis.
            # (N/A)
            # vital_category
            # Maps vital_name to a list standard vital sign categories.
            pl.col("vital_category"),
            # vital_value
            # Recorded value of the vital. Ensure that the measurement unit is aligned with the permissible units of measurements.
            pl.col("vital_value"),
            # meas_site_name
            # Site where the vital is recorded. No CDE corresponding to this variable (Optional field).
            # (N/A)
        )
        .drop_nulls("vital_value")
        .unique()
        .pipe(_add_missing_fields, "Vitals")
    )


# endregion

# region Labs
# The labs table is a long form (one lab result per row) longitudinal table.
# Each lab result is associated with a hospitalization event.


# endregion


# region Patient Assessments
# The patient_assessments table captures various assessments performed on
# patients across different domains, including neurological status, sedation
# levels, pain, and withdrawal. The table is designed to provide detailed
# information about the assessments, such as the name of the assessment, the
# category, and the recorded values.
def PatientAssessments(
    patient_information: pl.LazyFrame, timeseries_vitals: pl.LazyFrame
) -> pl.LazyFrame:
    print("reprodiCLIF - Patient Assessments")
    return (
        timeseries_vitals.select(
            "Global ICU Stay ID",
            "Time Relative to Admission (seconds)",
            "Glasgow Coma Score total",
            "Glasgow Coma Score eye opening",
            "Glasgow Coma Score motor",
            "Glasgow Coma Score verbal",
            "Richmond agitation-sedation scale",
            "Numeric Pain Rating Scale",
        )
        # Rename columns
        .rename(
            {
                "Glasgow Coma Score total": "gcs_total",
                "Glasgow Coma Score eye opening": "gcs_eye",
                "Glasgow Coma Score motor": "gcs_motor",
                "Glasgow Coma Score verbal": "gcs_verbal",
                "Richmond agitation-sedation scale": "RASS",
                "Numeric Pain Rating Scale": "NRS",
            }
        )
        .join(
            _ID_ICUOFFSET(patient_information),
            on="Global ICU Stay ID",
            how="left",
        )
        .unpivot(
            on=[
                "gcs_total",
                "gcs_eye",
                "gcs_motor",
                "gcs_verbal",
                "RASS",
                "NRS",
            ],
            index=[
                "Global Hospital Stay ID",
                "Time Relative to Admission (seconds)",
                "icu_admission_dttm",
            ],
            variable_name="assessment_category",
            value_name="numerical_value",
        )
        .with_columns(
            # hospitalization_id
            # ID variable for each patient encounter.
            pl.col("Global Hospital Stay ID").alias("hospitalization_id"),
            # recorded_dttm
            # Date and time when the vital is recorded. Datetime format should be %Y-%m-%d %H:%M:%S.
            (
                pl.col("icu_admission_dttm")
                + pl.duration(
                    seconds=pl.col("Time Relative to Admission (seconds)")
                )
            ).alias("recorded_dttm"),
            # assessment_name
            # This field is used to store the description of the flowsheet measure from the source data. This field is not used for analysis.
            # (N/A)
            # assessment_category
            # Maps assessment_name to a standardized list of patient assessments.
            pl.col("assessment_category"),
            # assessment_group
            # Broader Assessment Group. This groups the assessments into categories such as 'Sedation', 'Neurologic', 'Pain', etc.
            pl.col("assessment_category")
            .replace(
                {
                    "gcs_total": "Neurological",
                    "gcs_eye": "Neurological",
                    "gcs_motor": "Neurological",
                    "gcs_verbal": "Neurological",
                    "RASS": "Sedation/Agitation",
                    "NRS": "Pain",
                }
            )
            .alias("assessment_group"),
            # numerical_value
            # Numerical Assessment Result. The numerical result or score from the assessment component.
            pl.col("numerical_value"),
            # categorical_value
            # Categorical Assessment Result. The categorical outcome from the assessment component.
            # (N/A)
            # text_value
            # Textual Assessment Result. The textual explanation or notes from the assessment component.
            # (N/A)
        )
        .drop_nulls("numerical_value")
        .unique()
        .pipe(_add_missing_fields, "Vitals")
    )


# endregion


# region Admission Diagnosis
# Record of all diagnoses associated with the hospitalization. Expect breaking
# changes to this table as we seek to align it with existing diagnosis
# ontologies.
def AdmissionDiagnosis(
    patient_information: pl.LazyFrame,
    diagnoses_harmonizer: DiagnosesHarmonizer,
) -> pl.LazyFrame:
    print("reprodiCLIF - Admission Diagnosis")
    return (
        diagnoses_harmonizer.harmonize_diagnoses()
        .join(
            _ID_ICUOFFSET(patient_information),
            on="Global Hospital Stay ID",
            how="left",
        )
        .with_columns(
            # patient_id
            # Unique identifier for each patient.
            pl.col("Global Person ID").alias("patient_id"),
            # diagnostic_code
            # numeric diagnosis code
            pl.col("Diagnosis ICD Code").alias("diagnostic_code"),
            # diagnostic_code_format
            # description of the diagnostic code format
            pl.col("Diagnosis ICD Code Version").alias(
                "diagnostic_code_format"
            ),
            # start_dttm
            # date time the diagnosis was recorded
            (
                pl.col("icu_admission_dttm")
                + pl.duration(
                    seconds=pl.col(
                        "Diagnosis Start Relative to Admission (seconds)"
                    )
                )
            ).alias("start_dttm"),
            # end_dttm
            # date time the diagnosis was noted as resolved (if resolved)
            (
                pl.col("icu_admission_dttm")
                + pl.duration(
                    seconds=pl.col(
                        "Diagnosis End Relative to Admission (seconds)"
                    )
                )
            ).alias("end_dttm"),
        )
        .drop_nulls("diagnostic_code")
        .unique()
        .pipe(_add_missing_fields, "AdmissionDiagnosis")
    )


# endregion


# region Medication Admin Intermittent
# This table has exactly the same schema as medication_admin_continuous
# described below. The consortium decided to separate the medications that are
# administered intermittently from the continuously administered medications.
# However, the CDE for medication_category remains undefined for
# medication_admin_intermittent.

# endregion

# region Medication Orders
# This table records the ordering (not administration) of medications. The table
# is in long form (one medication order per row) longitudinal table. Linkage to
# the medication_admin_continuous and medication_admin_intermittent tables is
# through the med_order_id field.

# endregion


#######################################
# CRITICAL ILLNESS SPECIFIC TABLES
#######################################


# region Respiratory Support
# The respiratory support table is a wider longitudinal table that captures
# simultaneously recorded ventilator settings and observed ventilator parameters.
# The table is designed to capture the most common respiratory support devices
# and modes used in the ICU. It will be sparse for patients who are not on
# mechanical ventilation.
def RespiratorySupport(
    patient_information: pl.LazyFrame, timeseries_resp: pl.LazyFrame
) -> pl.LazyFrame:
    print("reprodiCLIF - Respiratory Support")
    return (
        timeseries_resp.join(
            _ID_ICUOFFSET(patient_information),
            on="Global ICU Stay ID",
            how="left",
        )
        .with_columns(
            # hospitalization_id
            # ID variable for each patient encounter.
            pl.col("Global Hospital Stay ID").alias("hospitalization_id"),
            # recorded_dttm
            # Date and time when the device settings and/or measurement was recorded.
            # Datetime format should be %Y-%m-%d %H:%M:%S.
            (
                pl.col("icu_admission_dttm")
                + pl.duration(
                    seconds=pl.col("Time Relative to Admission (seconds)")
                )
            ).alias("recorded_dttm"),
            # device_name
            # Includes raw string of the devices. Not used for analysis.
            # (N/A)
            # device_category
            # Maps device_name to a standardized list of respiratory support device categories.
            pl.col("Oxygen delivery system").alias("device_category"),
            # vent_brand_name
            # Ventilator model name when device_category is IMV or NIPPV.
            # (N/A)
            # mode_name
            # Includes raw string of the modes.
            pl.col("Ventilation mode Ventilator").alias("mode_name"),
            # mode_category
            # Maps mode_name to a standardized list of modes of mechanical ventilation.
            # (N/A)
            # tracheostomy
            # Indicates if tracheostomy is present.
            # (N/A)
            # fio2_set
            # Fraction of inspired oxygen set in decimals.
            pl.col(
                "Oxygen/Total gas setting [Volume Fraction] Ventilator"
            ).alias("fio2_set"),
            # lpm_set
            # Liters per minute set.
            pl.col("Oxygen gas flow Oxygen delivery system").alias("lpm_set"),
            # tidal_volume_set
            # Tidal volume set in mL.
            pl.col("Tidal volume setting Ventilator").alias("tidal_volume_set"),
            # resp_rate_set
            # Respiratory rate set in bpm.
            pl.col("Breath rate setting Ventilator").alias("resp_rate_set"),
            # pressure_control_set
            # Pressure control set in cmH2O.
            # (N/A)
            # pressure_support_set
            # Pressure support set in cmH2O.
            pl.col("Pressure support setting Ventilator").alias(
                "pressure_support_set"
            ),
            # flow_rate_set
            # Flow rate set.
            # (N/A)
            # peak_inspiratory_pressure_set
            # Peak inspiratory pressure set in cmH2O.
            # (N/A)
            # inspiratory_time_set
            # Inspiratory time set in seconds.
            pl.col("Inspiratory time setting Ventilator").alias(
                "inspiratory_time_set"
            ),
            # peep_set
            # Positive-end-expiratory pressure set in cmH2O.
            pl.col("Positive end expiratory pressure setting Ventilator").alias(
                "peep_set"
            ),
            # tidal_volume_obs
            # Observed tidal volume in mL.
            pl.col("Tidal volume.spontaneous+mechanical --on ventilator").alias(
                "tidal_volume_obs"
            ),
            # resp_rate_obs
            # Observed respiratory rate in bpm.
            pl.col(
                "Breath rate spontaneous and mechanical --on ventilator"
            ).alias("resp_rate_obs"),
            # plateau_pressure_obs
            # Observed plateau pressure in cmH2O.
            pl.col(
                "Pressure.plateau Respiratory system airway --on ventilator"
            ).alias("plateau_pressure_obs"),
            # peak_inspiratory_pressure_obs
            # Observed peak inspiratory pressure in cmH2O.
            pl.col(
                "Pressure.max Respiratory system airway --on ventilator"
            ).alias("peak_inspiratory_pressure_obs"),
            # peep_obs
            # Observed positive-end-expiratory pressure in cmH2O.
            pl.col("PEEP Respiratory system").alias("peep_obs"),
            # minute_vent_obs
            # Observed minute ventilation in liters.
            # (N/A)
            # mean_airway_pressure_obs
            # Observed mean airway pressure.
            pl.col("Mean airway pressure").alias("mean_airway_pressure_obs"),
        )
        .unique()
        .pipe(_add_missing_fields, "RespiratorySupport")
    )


# endregion

# region Medication Admin Continuous
# The medication admin continuous table is a long-form (one medication
# administration record per) longitudinal table designed for continuous
# infusions of common ICU medications such as vasopressors and sedation
# (Boluses of these drugs should be recorded in med_admin_intermittent).
# Note that it only reflects dose changes of the continuous medication and does
# not have a specific “end_time” variable to indicate the medication being
# stopped. The end of a continuous infusion should be recorded as a new row
# with med_dose = 0 and an appropriate mar_action_name (e.g. “stopped” or
# “paused”).

# endregion

# region Position
# The position table is a long form (one position per row) longitudinal table
# that captures all documented position changes of the patient. The table is
# designed for the explicit purpose of constructing the position_category CDE
# and identifying patients in prone position.

# endregion

# region Dialysis
# The dialysis table is a wider longitudinal table that captures the start and
# stop times of dialysis sessions, the type of dialysis performed, and the
# amount of dialysate flow and ultrafiltration.

# endregion

# region ECMO/MCS
# The ECMO/MCS table is a wider longitudinal table that captures the start and
# stop times of ECMO/MCS support, the type of device used, and the work rate of
# the device.

# endregion

# region Intake/Output
# The intake_output table is long form table that captures the times intake and
# output events were recorded, the type of fluid administered or recorded as
# “out”, and the amount of fluid.

# endregion

# region Therapy Details
# The therapy_details table is a wide longitudinal table that captures the
# details of therapy sessions. The table is designed to capture and categorize
# the most common therapy elements used in the ICU.

# endregion

# region Microbiology Culture
# The microbiology culture table is a wide longitudinal table that captures the
# order and result times of microbiology culture tests, the type of fluid
# collected, the component of the test, and the organism identified.

# endregion

# region Sensitivity
# a.k.a. (Microbiology Susceptibility)
# This table is used to store the susceptibility results of the organisms
# identified in the Microbiology Culture table and may be renamed to
# Microbiology_Susceptibility.

# endregion

# region Microbiology Nonculture
# The microbiology non-culture table is a wide longitudinal table that captures
# the order and result times of non-culture microbiology tests, the type of
# fluid collected, the component of the test, and the result of the test.

# endregion

# region Procedures
# A longitudinal record of each bedside ICU procedure performed on the patient
# (e.g. central line placement, chest tube placement). Note that this table is
# not intended to capture the full set of procedures performed on inpatients.

# endregion

# region Transfusion
# This table provides detailed information about transfusion events linked to
# specific hospitalizations.

# endregion


# region Code Status
# This table provides a longitudinal record of changes in a patient’s code
# status during their hospitalization. It tracks the timeline and
# categorization of code status updates, facilitating the analysis of care
# preferences and decisions.
def CodeStatus(
    patient_information: pl.LazyFrame, CODE_STATUS: pl.LazyFrame
) -> pl.LazyFrame:
    return (
        CODE_STATUS.join(
            _ID_ICUOFFSET(patient_information),
            on="Global Hospital Stay ID",
            how="left",
        )
        .with_columns(
            # hospitalization_id
            # ID variable for each patient encounter.
            pl.col("Global Hospital Stay ID").alias("hospitalization_id"),
            # recorded_dttm
            # Date and time when the code status was recorded. Datetime format should be %Y-%m-%d %H:%M:%S.
            (
                pl.col("icu_admission_dttm")
                + pl.duration(
                    seconds=pl.col("Time Relative to Admission (seconds)")
                )
            ).alias("recorded_dttm"),
            # code_status_name
            # Original code status string from the source data.
            pl.col("CODE_STATUS").alias("code_status_name"),
            # code_status_category
            # Maps code_status_name to a standardized list of code status categories.
            pl.col("CODE_STATUS")
            .replace(
                {
                    "full code": "Full",
                    "DNCPR": "DNR",
                    "DNI": "DNI/DNR",
                    "CMO": "Other",
                }
            )
            .alias("code_status_category"),
        )
        .unique()
        .pipe(_add_missing_fields, "CodeStatus")
    )


# endregion

# region Invasive Hemodynamics
# The invasive_hemodynamics table records invasive hemodynamic measurements
# during a patient’s hospitalization. These measurements represent pressures
# recorded via invasive monitoring and are expressed in millimeters of
# mercury (mmHg).

# endregion

# region Key ICU orders
# The key_icu_orders table captures key orders related to physical therapy (PT)
# and occupational therapy (OT) during ICU stays. It includes details about the
# hospitalization, the timing of the order, the specific name of the order, its
# category, and the status of the order (completed or sent).

# endregion


# region OTHER
def other():
    """
    add missing tables to the output directory
    """
    tables = (
        pl.read_csv("mappings/CLIF_DataDictionary.csv")
        .select("cdmTableName")
        .unique(maintain_order=True)
        .to_series()
        .to_list()
    )

    print(os.listdir(OUTPATH))

    for table in tables:
        if ((table + ".parquet") not in os.listdir(OUTPATH)) and (
            (table.upper() + ".parquet") not in os.listdir(OUTPATH)
        ):
            print(f"reprodiCLIF - adding missing table: {table}")
            pl.DataFrame().pipe(_add_missing_fields, table).write_parquet(
                OUTPATH + "clif_" + table.lower() + ".parquet"
            )


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
        default="../reprodICU_files_CLIF/",
    )
    args = parser.parse_args()

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

    ############
    # CONVERTING
    # Convert the reprodICU structure to the Common Longitudinal ICU Format (CLIF) structure
    # General inpatient tables
    (
        Patient(patient_information)
        .collect()
        .write_parquet(OUTPATH + "clif_patient.parquet")
    )
    (
        Hospitalization(patient_information)
        .collect()
        .write_parquet(OUTPATH + "clif_hospitalization.parquet")
    )
    (
        ADT(patient_information)
        .collect()
        .write_parquet(OUTPATH + "clif_adt.parquet")
    )
    (
        Vitals(patient_information, timeseries_vitals).sink_parquet(
            OUTPATH + "clif_vitals.parquet"
        )
    )
    # (
    #     Labs(patient_information, timeseries_labs)
    #     .collect()
    #     .write_parquet(OUTPATH + "clif_labs.parquet")
    # )
    (
        PatientAssessments(patient_information, timeseries_vitals).sink_parquet(
            OUTPATH + "clif_patient_assessments.parquet"
        )
    )
    (
        AdmissionDiagnosis(patient_information, diagnoses_harmonizer)
        .collect()
        .write_parquet(OUTPATH + "clif_admission_diagnosis.parquet")
    )
    # Medication Admin Intermittent
    # Medication Orders
    (
        RespiratorySupport(patient_information, timeseries_resp).sink_parquet(
            OUTPATH + "clif_respiratory_support.parquet"
        )
    )
    # Medication Admin Continuous
    # Position
    # Dialysis
    # ECMO/MCS
    # Intake/Output
    # Therapy Details
    # Microbiology Culture
    # Sensitivity
    # Microbiology Nonculture
    # Procedures
    # Transfusion
    (
        CodeStatus(patient_information, CODE_STATUS)
        .collect()
        .write_parquet(OUTPATH + "clif_code_status.parquet")
    )
    # Invasive Hemodynamics
    # Key ICU orders

    ####################
    # ADD MISSING TABLES
    # other()

    print("reprodiCLIF - done")
