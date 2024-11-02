# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: Converts the reprodICU structure to the OMOP Common Data Model (CDM) structure.
# The script is based on the OMOP CDM version 5.4

# Input: reprodICU structure
# Output: OMOP CDM structure

# Usage: python reprOMOPIZE.py

# Importing necessary libraries
import argparse
import os

import polars as pl

SECONDS_IN_DAY = 86400


# region helpers
# The FIELD_LEVEL table contains a list of fields that are used in the
# observational data tables. Each field is uniquely identified by a field
# concept ID and a field name.
def field_level(table_name: str, return_required: bool = False) -> list:
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


def add_missing_fields(
    data: pl.LazyFrame, table_name: str, check_required: bool = False
) -> pl.LazyFrame:
    """
    add missing fields to the data
    """
    fields, required = field_level(table_name, return_required=True)
    columns = data.collect_schema().names()

    for field, req in zip(fields, required):
        if field not in columns:
            if req == "Yes" and check_required:
                raise ValueError(
                    f"Field {field} is required for the {table_name} table"
                )

            data = data.with_columns(pl.lit(None).cast(pl.Int8).alias(field))

    return data.select(fields)


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

    ID = (
        patient_information.select(
            "Global ICU Stay ID",
            "Global Person ID",
            "Pre-ICU Length of Stay (days)",
        )
        .with_columns(
            ###########
            # PERSON_ID
            # Create the person_id column with a hash of the Global Person ID
            pl.col("Global Person ID")
            .hash()
            .alias("person_id")
        )
        .select(
            "Global ICU Stay ID", "person_id", "Pre-ICU Length of Stay (days)"
        )
    )
    CONCEPTS = CONCEPT.filter(
        pl.col("domain_id") == "Drug",
        pl.col("concept_class_id") == "Ingredient",
    ).select("concept_id", "concept_name")

    # Extract the drug exposure information
    return (
        medications.join(ID, on="Global ICU Stay ID", how="left")
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
                pl.datetime(
                    year=2000, month=1, day=1, hour=0, minute=0, second=0
                )
                + pl.duration(
                    seconds=pl.col("Drug Start Relative to Admission (seconds)")
                )
                + pl.when(pl.col("Pre-ICU Length of Stay (days)").is_not_null())
                .then(
                    pl.duration(
                        seconds=pl.col("Pre-ICU Length of Stay (days)")
                        * SECONDS_IN_DAY
                    )
                )
                .otherwise(pl.duration(days=0))
            ).alias("drug_exposure_start_datetime"),
            ############################
            # DRUG_EXPOSURE_END_DATETIME
            # Create the drug_exposure_end_datetime column with the datetime of the drug exposure
            (
                pl.datetime(
                    year=2000, month=1, day=1, hour=0, minute=0, second=0
                )
                + pl.duration(
                    seconds=pl.col("Drug End Relative to Admission (seconds)")
                )
                + pl.when(pl.col("Pre-ICU Length of Stay (days)").is_not_null())
                .then(
                    pl.duration(
                        seconds=pl.col("Pre-ICU Length of Stay (days)")
                        * SECONDS_IN_DAY
                    )
                )
                .otherwise(pl.duration(days=0))
            ).alias("drug_exposure_end_datetime"),
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
        .pipe(add_missing_fields, "drug_exposure")
    )


# region CARE_SITE
# The CARE_SITE table contains a list of uniquely identified institutional
# (physical or organizational) units where healthcare delivery is practiced
# (offices, wards, hospitals, clinics, etc.
def care_site(patient_information: pl.LazyFrame) -> pl.LazyFrame:
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
        .pipe(add_missing_fields, "care_site")
        .unique()
    )

    return care_site


# endregion


# region CONDITION_OCCURRENCE
# This table contains records of Events of a Person suggesting the presence of
# a disease or medical condition stated as a diagnosis, a sign, or a symptom,
# which is either observed by a Provider or reported by the patient.
def condition_occurrence(
    CONCEPT: pl.LazyFrame, diagnoses: pl.LazyFrame
) -> pl.LazyFrame:

    ID = patient_information.with_columns(
        ###########
        # PERSON_ID
        # Create the person_id column with a hash of the Global Person ID
        pl.col("Global Person ID")
        .hash()
        .alias("person_id")
    ).select(
        "Global ICU Stay ID",
        "person_id",
        "Pre-ICU Length of Stay (days)",
        "Source Database",
    )
    CONCEPTS = CONCEPT.filter(
        pl.col("domain_id") == "Condition",
    ).select("concept_id", "concept_name")

    return (
        diagnoses.join(ID, on="Global ICU Stay ID", how="left")
        .join(
            CONCEPTS,
            left_on="Diagnosis Description",
            right_on="concept_name",
            how="left",
        )
        .rename({"concept_id": "condition_concept_id"})
        .with_columns(
            ##########################
            # CONDITION_START_DATETIME
            # Create the condition_start_datetime column with the datetime of the diagnosis
            (
                pl.datetime(
                    year=2000, month=1, day=1, hour=0, minute=0, second=0
                )
                + pl.duration(
                    seconds=pl.col(
                        "Diagnosis Start Relative to Admission (seconds)"
                    )
                )
                + pl.when(pl.col("Pre-ICU Length of Stay (days)").is_not_null())
                .then(
                    pl.duration(
                        seconds=pl.col("Pre-ICU Length of Stay (days)")
                        * SECONDS_IN_DAY
                    )
                )
                .otherwise(pl.duration(days=0))
            ).alias("condition_start_datetime"),
            #######################
            # CONDITION_END_DATETIME
            # Create the condition_end_datetime column with the datetime of the diagnosis
            (
                pl.datetime(
                    year=2000, month=1, day=1, hour=0, minute=0, second=0
                )
                + pl.duration(
                    seconds=pl.col(
                        "Diagnosis End Relative to Admission (seconds)"
                    )
                )
                + pl.when(pl.col("Pre-ICU Length of Stay (days)").is_not_null())
                .then(
                    pl.duration(
                        seconds=pl.col("Pre-ICU Length of Stay (days)")
                        * SECONDS_IN_DAY
                    )
                )
                .otherwise(pl.duration(days=0))
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
            pl.when(pl.col("Source Database") == "SICdb").then(pl.lit(32901))
            # MIMIC-III and MIMIC-IV only include discharge diagnoses
            .when(pl.col("Source Database").str.starts_with("MIMIC"))
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
        .pipe(add_missing_fields, "condition_occurrence")
    )


# endregion


# region LOCATION
# The LOCATION table represents a generic way to capture physical location or
# address information of Persons and Care Sites.
def location(patient_information: pl.LazyFrame) -> pl.LazyFrame:
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
        .pipe(add_missing_fields, "location")
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
) -> pl.LazyFrame:

    ID = (
        patient_information.select(
            "Global ICU Stay ID",
            "Global Person ID",
            "Pre-ICU Length of Stay (days)",
        )
        .with_columns(
            ###########
            # PERSON_ID
            # Create the person_id column with a hash of the Global Person ID
            pl.col("Global Person ID")
            .hash()
            .alias("person_id")
        )
        .select(
            "Global ICU Stay ID", "person_id", "Pre-ICU Length of Stay (days)"
        )
    )
    CONCEPTS = CONCEPT.filter(
        pl.col("domain_id") == "Measurement",
        pl.col("concept_class_id") == "Clinical Observation",
    ).select("concept_id", "concept_name")

    def _unpivot(data: pl.LazyFrame) -> pl.LazyFrame:
        """
        unpivot the data
        """
        return (
            data.join(ID, on="Global ICU Stay ID", how="left")
            .drop("Global ICU Stay ID")
            .with_columns(
                ##################
                # MEASUREMENT_DATE
                # Create the measurement_datetime column with the datetime of the measurement
                (
                    pl.datetime(
                        year=2000, month=1, day=1, hour=0, minute=0, second=0
                    )
                    + pl.duration(
                        seconds=pl.col("Time Relative to Admission (seconds)")
                    )
                    + pl.duration(
                        seconds=pl.col("Pre-ICU Length of Stay (days)")
                        * SECONDS_IN_DAY
                    )
                ).alias("measurement_datetime")
            )
            .drop("Time Relative to Admission (seconds)")
            .with_columns(
                ######################
                # MEASUREMENT_DATETIME
                # Create the measurement_date column with the date of the measurement
                pl.col("measurement_datetime")
                .dt.date()
                .alias("measurement_date"),
            )
            .unpivot(
                index=["person_id", "measurement_date", "measurement_datetime"],
                variable_name="variable_name",
                value_name="value_as_number",
            )
            # Create the measurement_concept_id column with the concept_id of the Measurement
            .join(
                CONCEPTS,
                left_on="variable_name",
                right_on="concept_name",
                how="left",
            )
            .drop("variable_name")
            .rename({"concept_id": "measurement_concept_id"})
            .drop_nulls("value_as_number")
        )

    # Extract the measurement information
    return (
        pl.concat(
            [timeseries_vitals.pipe(_unpivot), timeseries_labs.pipe(_unpivot)],
            how="vertical",
        )
        .with_row_index("measurement_id")
        .pipe(add_missing_fields, "measurement")
    )


# endregion


# region PERSON
# This table serves as the central identity management for all Persons in the
# database. It contains records that uniquely identify each person or patient,
# and some demographic information.
def person(patient_information: pl.LazyFrame) -> pl.LazyFrame:
    # Dates of the databases
    # eICU: 2014 to 2015
    # HiRID: 2008-01 to 2016-06
    # MIMIC-III: 2001 to 2012
    # MIMIC-IV: 2008 to 2022
    # SICdb: 2013 to 2021
    # UMCdb: 2003 to 2016

    # Extract the person information
    return (
        patient_information.select(
            "Global Person ID",
            "Gender",
            "Ethnicity",
            "Admission Age (years)",
            "Care Site",
            "Source Database",
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
            # pl.when(pl.col("Source Database") == "eICU-CRD")
            # .then(pl.lit(2015) - pl.col("Admission Age (years)"))
            # .when(pl.col("Source Database") == "HiRID")
            # .then(pl.lit(2016) - pl.col("Admission Age (years)"))
            # .when(pl.col("Source Database") == "MIMIC-III")
            # .then(pl.lit(2012) - pl.col("Admission Age (years)"))
            # .when(pl.col("Source Database") == "MIMIC-IV")
            # .then(pl.lit(2022) - pl.col("Admission Age (years)"))
            # .when(pl.col("Source Database") == "SICdb")
            # .then(pl.lit(2021) - pl.col("Admission Age (years)"))
            # .when(pl.col("Source Database") == "UMCdb")
            # .then(pl.lit(2016) - pl.col("Admission Age (years)"))
            # .alias("year_of_birth"),
            ###################
            # RACE_CONCEPT_ID
            # Create the race_concept_id column based on the Ethnicity column
            # TODO: Edit Ethnicity mapping to OMOP CDM Concepts
            ##############
            # CARE_SITE_ID
            # Create the care_site_id column with a hash of the Care Site
            # NOTE: same as in the CARE_SITE table
            pl.col("Care Site").hash().alias("care_site_id"),
        )
        .pipe(add_missing_fields, "person")
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
    # Extract the visit occurrence information
    return (
        patient_information.with_columns(
            #####################
            # VISIT_OCCURRENCE_ID
            # Create the visit_occurrence_id column with a hash of the Global ICU Stay ID
            pl.col("Global ICU Stay ID").hash().alias("visit_occurrence_id"),
            ###########
            # PERSON_ID
            # Create the person_id column with a hash of the Global Person ID
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
                pl.datetime(
                    year=2000, month=1, day=1, hour=0, minute=0, second=0
                )
                + pl.when(pl.col("Pre-ICU Length of Stay (days)").is_not_null())
                .then(
                    pl.duration(
                        seconds=pl.col("Pre-ICU Length of Stay (days)")
                        * SECONDS_IN_DAY
                    )
                )
                .otherwise(pl.duration(days=0))
            ).alias("visit_start_datetime"),
            ####################
            # VISIT_END_DATETIME
            # Create the visit_end_datetime column with the end datetime of the ICU stay
            (
                pl.datetime(
                    year=2000, month=1, day=1, hour=0, minute=0, second=0
                )
                + pl.when(pl.col("Pre-ICU Length of Stay (days)").is_not_null())
                .then(
                    pl.duration(
                        seconds=pl.col("Pre-ICU Length of Stay (days)")
                        * SECONDS_IN_DAY
                    )
                )
                .otherwise(pl.duration(days=0))
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
        .pipe(add_missing_fields, "visit_occurrence")
        .unique()
    )


# region OTHER
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
            print(f"Adding missing table: {table}")
            pl.DataFrame().pipe(add_missing_fields, table).write_parquet(
                OUTPATH + table + ".parquet"
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
        "--output",
        type=str,
        help="Path to the output directory",
        default="../reprodICU_files_OMOP/",
    )
    args = parser.parse_args()

    # Load the reprodICU data
    INPATH = args.input
    OUTPATH = args.output
    diagnoses = pl.scan_parquet(INPATH + "diagnoses_imputed.parquet")
    patient_information = pl.scan_parquet(
        INPATH + "patient_information_imputed.parquet"
    )
    medications = pl.scan_parquet(INPATH + "medications.parquet")
    timeseries_vitals = pl.scan_parquet(INPATH + "timeseries_vitals.parquet")
    timeseries_labs = pl.scan_parquet(
        INPATH + "timeseries_labs_winsorized.parquet"
    )
    timeseries_resp = pl.scan_parquet(INPATH + "timeseries_resp.parquet")

    # Parquetize the OMOP vocabulary files
    for file in os.listdir(OUTPATH + "OMOP_vocabulary/"):
        # Check if the file is already parquetized
        if os.path.isfile(OUTPATH + file[:-4] + ".parquet"):
            continue

        pl.scan_csv(
            OUTPATH + "OMOP_vocabulary/" + file,
            separator="\t",
            infer_schema_length=10000,
            quote_char=None,
        ).sink_parquet(OUTPATH + file[:-4] + ".parquet")

    # Load the OMOP vocabulary files
    CONCEPT = pl.scan_parquet(OUTPATH + "CONCEPT.parquet")
    CONCEPT_RELATIONSHIP = pl.scan_parquet(
        OUTPATH + "CONCEPT_RELATIONSHIP.parquet"
    )
    CONCEPT_ANCESTOR = pl.scan_parquet(OUTPATH + "CONCEPT_ANCESTOR.parquet")
    CONCEPT_CLASS = pl.scan_parquet(OUTPATH + "CONCEPT_CLASS.parquet")
    CONCEPT_SYNONYM = pl.scan_parquet(OUTPATH + "CONCEPT_SYNONYM.parquet")
    DOMAIN = pl.scan_parquet(OUTPATH + "DOMAIN.parquet")
    RELATIONSHIP = pl.scan_parquet(OUTPATH + "RELATIONSHIP.parquet")
    VOCABULARY = pl.scan_parquet(OUTPATH + "VOCABULARY.parquet")

    # Convert the reprodICU structure to the OMOP CDM structure
    # Tables with transformed IDs
    care_site(patient_information).sink_parquet(OUTPATH + "care_site.parquet")
    condition_occurrence(CONCEPT, diagnoses).collect().write_parquet(
        OUTPATH + "condition_occurrence.parquet"
    )
    location(patient_information).sink_parquet(OUTPATH + "location.parquet")
    person(patient_information).collect().write_parquet(
        OUTPATH + "person.parquet"
    )
    visit_occurrence(patient_information).sink_parquet(
        OUTPATH + "visit_occurrence.parquet"
    )

    # Tables with row indices
    drug_exposure(CONCEPT, medications, patient_information).collect(
        streaming=True
    ).write_parquet(OUTPATH + "drug_exposure.parquet")
    # measurement(
    #     CONCEPT,
    #     patient_information,
    #     timeseries_vitals,
    #     timeseries_labs,
    #     timeseries_resp,
    # ).collect(streaming=True).write_parquet(OUTPATH + "measurement.parquet")

    # Add missing tables
    other()
