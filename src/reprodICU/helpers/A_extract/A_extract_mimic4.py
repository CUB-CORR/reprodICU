# Author: Finn Fassbender
# Last modified: 2025-10-30

# Description: This script extracts the data from the source files and provides the extracted data
# in a structured format for further processing and harmonization.

import os.path
import re

import numpy as np
import polars as pl

from ..helper import GlobalHelpers
from ..helper_filepaths import MIMIC4Paths
from ..helper_OMOP import Vocabulary


class MIMIC4Extractor(MIMIC4Paths):
    def __init__(self, paths, DEMO=False):
        super().__init__(paths, DEMO)
        self.path = paths.mimic4_source_path
        self.helpers = GlobalHelpers()
        self.omop = Vocabulary(paths)
        self.icu_stay_id = self.extract_patient_information().select(
            self.icu_stay_id_col,
            self.hospital_stay_id_col,
            self.person_id_col,
        )
        self.icu_length_of_stay = self.extract_patient_information().select(
            self.icu_stay_id_col, self.icu_length_of_stay_col
        )

        self.lab_specimen_map = {
            "ART.": "Blood arterial",
            "CENTRAL VENOUS.": "Blood central venous",
            "MIX.": "Blood mixed venous",
            "VEN.": "Blood venous",
        }

    # region ID mapping table
    def extract_patient_IDs(self) -> pl.LazyFrame:
        """
        Extract patient IDs from ICU stays.

        Steps:
            1. Read ICU stays CSV and rename columns to standardized names.
            2. Remove duplicates and cast ID columns to integer.
            3. Select required columns.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {hospital_stay_id_col}: Hospital admission identifier.
                - {person_id_col}: Patient identifier.
                - {icu_length_of_stay_col}: ICU length of stay (days).
                - intime: ICU admission datetime.
        """
        return (
            pl.scan_csv(self.icustays_path)
            .rename(
                {
                    "stay_id": self.icu_stay_id_col,
                    "hadm_id": self.hospital_stay_id_col,
                    "subject_id": self.person_id_col,
                    "los": self.icu_length_of_stay_col,
                }
            )
            .unique()
            .cast(
                {
                    self.icu_stay_id_col: int,
                    self.hospital_stay_id_col: int,
                    self.person_id_col: int,
                }
            )
            .select(
                self.icu_stay_id_col,
                self.hospital_stay_id_col,
                self.person_id_col,
                self.icu_length_of_stay_col,
                "intime",
            )
        )

    # region patient
    # Extract patient information from the patient.csv file
    def extract_patient_information(self) -> pl.LazyFrame:
        """
        Extract and derive comprehensive patient demographics and clinical information.

        Steps:
            1. Read ICU stays, admissions, and patients CSV files.
            2. Join on hospital and patient identifiers.
            3. Merge height, weight, and specialty information.
            4. Parse datetime strings and calculate derived fields: age, LOS variants, mortality flags.
            5. Compute ICU stay sequence and time offset from first admission.
            6. Standardize categorical columns (gender, ethnicity, admission type, etc.).

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {hospital_stay_id_col}: Hospital admission identifier.
                - {person_id_col}: Patient identifier.
                - {icu_stay_seq_num_col}: ICU stay sequence number.
                - {icu_time_rel_to_first_col}: Time relative to first ICU admission (seconds).
                - {gender_col}: Patient gender.
                - {age_col}: Patient age (years).
                - {height_col}: Patient height (cm).
                - {weight_col}: Patient weight (kg).
                - {ethnicity_col}: Patient ethnicity.
                - {pre_icu_length_of_stay_col}: Pre-ICU length of stay (days).
                - {icu_length_of_stay_col}: ICU length of stay (days).
                - {hospital_length_of_stay_col}: Hospital length of stay (days).
                - {mortality_hosp_col}: Hospital mortality flag.
                - {mortality_icu_col}: ICU mortality flag.
                - {mortality_after_col}: Days between discharge and death (null if alive).
                - {admission_type_col}: Admission type based on specialty.
                - {admission_urgency_col}: Admission urgency.
                - {admission_time_col}: ICU admission time of day.
                - {admission_year_col}: Admission year.
                - {admission_loc_col}: Admission location.
                - {specialty_col}: Treating specialty.
                - {unit_type_col}: ICU unit type.
                - {care_site_col}: Care site identifier.
                - {discharge_loc_col}: Discharge location.
        """
        # scanning csv files to build labels DataFrame
        icustays = pl.scan_csv(self.icustays_path).rename(
            {
                "stay_id": self.icu_stay_id_col,
                "hadm_id": self.hospital_stay_id_col,
                "subject_id": self.person_id_col,
                "los": self.icu_length_of_stay_col,
                "first_careunit": self.unit_type_col,
            }
        )

        admissions = (
            pl.scan_csv(self.admissions_path)
            .rename(
                {
                    "hadm_id": self.hospital_stay_id_col,
                    "race": self.ethnicity_col,  # "race" is the choice of the dataset creators # fmt: skip
                    "admission_location": self.admission_loc_col,
                    "discharge_location": self.discharge_loc_col,
                    "admission_type": self.admission_urgency_col,
                    "hospital_expire_flag": self.mortality_hosp_col,
                }
            )
            .select(
                self.hospital_stay_id_col,
                self.ethnicity_col,
                self.admission_loc_col,
                self.discharge_loc_col,
                self.admission_urgency_col,
                self.mortality_hosp_col,
                "admittime",
                "dischtime",
                "deathtime",
            )
        )

        patients = (
            pl.scan_csv(self.patients_path)
            .rename(
                {
                    "subject_id": self.person_id_col,
                    "gender": self.gender_col,
                }
            )
            .select(
                self.person_id_col,
                self.gender_col,
                "anchor_age",
                "anchor_year",
                "anchor_year_group",
                "dod",
            )
        )

        # calculate mortality after discharge censor cutoff (1 year after last hospital discharge)
        # Dates of death are censored at one-year from the patient’s last hospital discharge.
        # As a result, null dates of death indicate the patient was alive at least up to that time point.
        MORTALITY_AFTER_CENSOR_CUTOFF = (
            pl.scan_csv(self.admissions_path)
            .select("subject_id", "dischtime")
            .rename({"subject_id": self.person_id_col})
            .with_columns(
                pl.col("dischtime").str.to_datetime("%Y-%m-%d %H:%M:%S")
            )
            .group_by(self.person_id_col)
            .agg(pl.col("dischtime").max().alias("last_dischtime"))
            .with_columns(
                pl.col("last_dischtime").dt.offset_by("1y").alias("censortime")
            )
            .select(self.person_id_col, "censortime")
        )

        return (
            icustays.join(
                admissions,
                on=self.hospital_stay_id_col,
                how="left",
                coalesce=True,
            )
            .join(patients, on=self.person_id_col, how="left", coalesce=True)
            .join(
                self._extract_patient_height_weight(icustays),
                on=self.icu_stay_id_col,
                how="left",
                coalesce=True,
            )
            .join(
                self._extract_specialties(),
                on=self.icu_stay_id_col,
                how="left",
                coalesce=True,
            )
            .join(
                self._extract_mimic4_version(),
                on=self.icu_stay_id_col,
                how="left",
                coalesce=True,
            )
            .join(
                MORTALITY_AFTER_CENSOR_CUTOFF,
                on=self.person_id_col,
                how="left",
                coalesce=True,
            )
            .with_columns(
                pl.col("intime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("outtime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("dod").str.to_datetime("%Y-%m-%d"),
                pl.col("admittime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("dischtime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("deathtime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col(self.icu_stay_id_col).cast(int),
                pl.col(self.hospital_stay_id_col).cast(int),
                pl.col(self.icu_length_of_stay_col).cast(float),
                pl.lit("Beth Israel Deaconess Medical Center").alias(
                    self.care_site_col
                ),
            )
            .with_columns(
                # Calculate age in years at ICU admission similar to demographics/age.sql
                pl.col("admittime")
                .dt.date()
                .dt.year()
                .sub(pl.col("anchor_year"))
                .add(pl.col("anchor_age"))
                .cast(int)
                .alias(self.age_col),
                # For admission year, assume average of the group
                (
                    pl.col("anchor_year_group")
                    .str.split(" - ")
                    .map_elements(
                        lambda s: np.mean([int(i) for i in s if i]),
                        return_dtype=float,
                    )
                    .cast(int)
                    + (pl.col("intime").dt.year() - pl.col("anchor_year"))
                ).alias(self.admission_year_col),
                # Convert categorical gender to enum
                pl.col(self.gender_col)
                .replace({"M": "Male", "F": "Female"})
                .cast(self.gender_dtype),
                # Convert categorical ethnicity to enum
                pl.col(self.ethnicity_col)
                .replace(self.ETHNICITY_MAP)
                .cast(self.ethnicity_dtype),
                # Calculate pre ICU length of stay
                (pl.col("intime") - pl.col("admittime"))
                .truediv(pl.duration(days=1))
                .cast(float)
                .alias(self.pre_icu_length_of_stay_col),
                # Calculate hospital length of stay
                (pl.col("dischtime") - pl.col("admittime"))
                .truediv(pl.duration(days=1))
                .cast(float)
                .alias(self.hospital_length_of_stay_col),
                # Calculate admission time
                pl.col("intime").dt.time().alias(self.admission_time_col),
                # Calculate ICU mortality
                (pl.col("deathtime") - pl.col("outtime"))
                .truediv(pl.duration(hours=1))
                .le(pl.duration(hours=self.ICU_DISCHARGE_MORTALITY_CUTOFF))
                .cast(bool)
                # .fill_null(False)
                .alias(self.mortality_icu_col),
                # Calculate hospital mortality
                pl.col(self.mortality_hosp_col).cast(bool),
                # Calculate mortality after discharge
                (pl.col("dod") - pl.col("outtime"))
                .truediv(pl.duration(days=1))
                .cast(int)
                .alias(self.mortality_after_col),
                # Calculate mortality after discharge cutoff
                (pl.col("censortime") - pl.col("outtime"))
                .truediv(pl.duration(days=1))
                .cast(int)
                .alias(self.mortality_after_cutoff_col),
                # Convert categorical admission location to enum
                pl.col(self.admission_loc_col)
                .replace(self.ADMISSION_LOCATIONS_MAP)
                .cast(self.admission_locations_dtype),
                # Convert categorical unit type to enum
                pl.col(self.unit_type_col)
                .replace(self.UNIT_TYPES_MAP)
                .cast(self.unit_types_dtype),
                # Convert categorical discharge location to enum
                pl.col(self.discharge_loc_col)
                .replace(self.DISCHARGE_LOCATIONS_MAP)
                .cast(self.discharge_locations_dtype),
                # Determine Admission Type based on treating specialty
                pl.col(self.specialty_col)
                .replace_strict(self.ADMISSION_TYPES_MAP, default=None)
                .cast(self.admission_types_dtype)
                .alias(self.admission_type_col),
                # Convert categorical admission urgency to enum
                pl.col(self.admission_urgency_col)
                .replace_strict(self.ADMISSION_URGENCY_MAP, default=None)
                .cast(self.admission_urgency_dtype),
                # Convert categorical specialty to enum
                pl.col(self.specialty_col)
                .replace(self.SPECIALTIES_MAP)
                .cast(self.specialties_dtype),
            )
            # Calculate ICU stay sequence number
            .sort(self.person_id_col, "intime")
            .with_columns(
                (pl.int_range(pl.len()).over(self.person_id_col) + 1).alias(
                    self.icu_stay_seq_num_col
                ),
                # Calculate time relative to first ICU admission
                (
                    pl.col("intime")
                    - pl.col("intime").min().over(self.person_id_col)
                )
                .dt.total_seconds()
                .alias(self.icu_time_rel_to_first_col),
            )
            # Fill missing ICU mortality values with False if patient was
            # discharged from hospital alive
            .with_columns(
                pl.when(
                    pl.col(self.mortality_icu_col).is_null()
                    & pl.col(self.mortality_hosp_col).eq(False)
                )
                .then(False)
                .otherwise(pl.col(self.mortality_icu_col))
                .alias(self.mortality_icu_col)
            )
        )

    # endregion

    # region specialties
    # Extract specialties from the services.csv file
    def _extract_specialties(self) -> pl.LazyFrame:
        """
        Extract specialties from the services CSV file and merge with ICU stay data.

        Steps:
            1. Extract necessary IDs and intime from patient data.
            2. Scan and rename the services CSV:
               - {hospital_stay_id_col}: Hospital stay identifier.
               - {specialty_col}: Specialty service.
            3. Filter and group to retrieve the most recent specialty before ICU admission.

        Returns:
            pl.LazyFrame: A lazy frame containing:
                - {icu_stay_id_col}: ICU stay identifier.
                - {specialty_col}: Specialty at ICU admission.
        """
        IDs = self.extract_patient_IDs().select(
            self.hospital_stay_id_col, self.icu_stay_id_col, "intime"
        )

        services = pl.scan_csv(self.services_path).rename(
            {
                "hadm_id": self.hospital_stay_id_col,
                "curr_service": self.specialty_col,
            }
        )

        return (
            services.select(
                self.hospital_stay_id_col, "transfertime", self.specialty_col
            )
            .join(IDs, on=self.hospital_stay_id_col)
            .with_columns(
                pl.col("transfertime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("intime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
            # Get the most recent specialty
            .filter(
                pl.col("transfertime")
                < (pl.col("intime") + pl.duration(hours=2))
            )
            # Get the most recent specialty on ICU admission
            .sort("transfertime")
            .group_by(self.icu_stay_id_col)
            .last()
            .select(self.icu_stay_id_col, self.specialty_col)
        )

    # endregion

    # region versions
    def _extract_mimic4_version(self) -> pl.LazyFrame:
        """
        Extracts the MIMIC-IV version when a patient was first introduced.
        """

        versions = pl.scan_csv(self.icustays_path).with_columns(
            pl.lit(None).alias(self.dataset_version_col)
        )
        for version, path in self.icustays_version_paths.items():
            if path is None or not os.path.isfile(path):
                continue

            if version == "current":
                # mimic4_source_path: ".../mimiciv/3.1/"
                m = re.search(r"(\d+\.\d+)[\/\\]*$", self.path)
                version = m.group(1) if m else None

            versions = (
                versions.join(
                    pl.scan_csv(path)
                    .select("stay_id", "hadm_id", "subject_id")
                    .with_columns(pl.lit(version).alias("version")),
                    on=["stay_id", "hadm_id", "subject_id"],
                    how="left",
                    coalesce=True,
                )
                .with_columns(
                    pl.coalesce(self.dataset_version_col, "version").alias(
                        self.dataset_version_col
                    )
                )
                .drop("version")
            )

        return versions.rename({"stay_id": self.icu_stay_id_col}).select(
            self.icu_stay_id_col, self.dataset_version_col
        )

    # region (h/w)eight
    # Extract patient height and weight from the chartevents.csv file
    # NOTE: This function is used in the extract_patient_information function
    # NOTE: Pre-calculated data is stored in a parquet file to speed up the process
    #       Rerun the function with the force parameter set to True to recalculate the data
    #       and overwrite the parquet file
    #       Runtime: ~ 7 min
    def _extract_patient_height_weight(
        self, icustays: pl.LazyFrame, force=False
    ) -> pl.DataFrame:
        """
        Extract patient height and weight from the chartevents CSV file, using cached data if available.

        Height IDs taken from:
        https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/measurement/height.sql
        Weight IDs taken from:
        https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/demographics/weight_durations.sql

        Steps:
            1. Check if pre-calculated parquet file exists; if so, load unless force=True.
            2. Read chartevents CSV and filter rows based on ITEMID values corresponding to height and weight.
            3. Join with ICU stays data to constrain measurements within a given time cutoff.
            4. Convert CHARTTIME and INTIME to datetime and perform unit conversions:
               - Convert height from inches to centimeters.
               - Convert weight from pounds to kilograms.
            5. Pivot the data so each ICU stay has separate columns for weight and height.
            6. Save the resulting DataFrame as a parquet file for caching.

        Returns:
            pl.LazyFrame: A lazy frame containing:
                - {icu_stay_id_col}: ICU stay identifier.
                - {weight_col}: Patient weight in kilograms.
                - {height_col}: Patient height in centimeters.
        """
        # check if precalculated data is available
        if (
            os.path.isfile(self.precalc_path + "MIMIC4_height_weight.parquet")
            and not force
        ):
            return pl.scan_parquet(
                self.precalc_path + "MIMIC4_height_weight.parquet"
            )

        print("MIMIC4  - Extracting patient height and weight...")

        if "parquet" in self.chartevents_path:
            chartevents = pl.scan_parquet(self.chartevents_path)
        else:
            chartevents = pl.scan_csv(
                self.chartevents_path,
                schema_overrides={"VALUE": str, "VALUENUM": float},
            )

        WEIGHT_ITEMIDS = {
            224639: self.weight_col,  # Daily Weight [metavision]
            226512: self.weight_col,  # Admission Weight (Kg) [metavision]
            226531: "weight_lbs",  # Admission Weight (lbs.) [metavision]
        }
        HEIGHT_ITEMIDS = {
            226707: "height_inch",  # Height [metavision]
            226730: self.height_col,  # Height (cm) [metavision]
        }

        KEEPIDS = [*(WEIGHT_ITEMIDS | HEIGHT_ITEMIDS).keys()]
        ADMIT_WEIGHT_IDS = [226512, 226531]
        DAILY_WEIGHT_IDS = [224639]

        height_weight = (
            chartevents.select("stay_id", "itemid", "valuenum", "charttime")
            # Rename columns for consistency
            .rename({"stay_id": self.icu_stay_id_col})
            .filter(pl.col("itemid").is_in(KEEPIDS))
            .join(
                icustays.select(self.icu_stay_id_col, "intime"),
                on=self.icu_stay_id_col,
                how="left",
            )
            .with_columns(
                pl.col("charttime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("intime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("itemid")
                .replace_strict(
                    WEIGHT_ITEMIDS | HEIGHT_ITEMIDS,  # "|" merges dictionaries
                    default=None,
                )
                .alias("label"),
            )
            .drop_nulls("label")
            .with_columns(
                # Convert height in in to cm, weight in lbs to kg
                pl.when(pl.col("label") == "height_inch")
                .then(pl.col("valuenum").mul(self.INCH_TO_CM))
                .when(pl.col("label") == "weight_lbs")
                .then(pl.col("valuenum").mul(self.LBS_TO_KG))
                .otherwise(pl.col("valuenum"))
                .alias("valuenum"),
                # Rename ITEMID to height_cm / weight_kg
                pl.when(pl.col("label") == "height_inch")
                .then(pl.lit(self.height_col))
                .when(pl.col("label") == "weight_lbs")
                .then(pl.lit(self.weight_col))
                .otherwise(pl.col("label"))
                .alias("label"),
            )
        )

        # Backfill weights as in:
        # https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/demographics/weight_durations.sql
        # Select the first admission weight
        first_admit_weight = (
            height_weight.filter(pl.col("itemid").is_in(ADMIT_WEIGHT_IDS))
            .collect()
            .with_columns(
                pl.col("valuenum")
                .first()
                .over(partition_by=self.icu_stay_id_col, order_by="charttime")
                .alias("first_admit_weight")
            )
        )
        # Select the first daily weight measurement
        first_daily_weight = (
            height_weight.filter(pl.col("itemid").is_in(DAILY_WEIGHT_IDS))
            .collect()
            .with_columns(
                pl.col("valuenum")
                .first()
                .over(partition_by=self.icu_stay_id_col, order_by="charttime")
                .alias("first_daily_weight")
            )
        )
        # Combine the first admit and first daily weight measurements
        weight = (
            pl.concat([first_admit_weight, first_daily_weight], how="diagonal")
            .drop("valuenum")
            .with_columns(
                pl.coalesce(
                    pl.col("first_admit_weight"), pl.col("first_daily_weight")
                ).alias("valuenum")
            )
            .drop("first_admit_weight", "first_daily_weight")
        )

        # Height measurements from the first 24 hours of the ICU stay since it's unlikely to change
        height = (
            height_weight.filter(pl.col("itemid").is_in(HEIGHT_ITEMIDS.keys()))
            .collect()
            .filter(
                (pl.col("charttime") - pl.col("intime")).le(
                    pl.duration(hours=24)
                ),
            )
        )

        height_weight = (
            pl.concat([weight, height], how="diagonal")
            .drop("itemid", "intime", "charttime")
            .pivot(
                index=self.icu_stay_id_col,
                on="label",
                values="valuenum",
                aggregate_function="median",
            )
            .select(self.icu_stay_id_col, self.weight_col, self.height_col)
            .with_columns(
                pl.col(self.weight_col).cast(float).round(1),
                pl.col(self.height_col).cast(float).round(0),
            )
        )

        # Save precalculated data
        height_weight.write_parquet(
            self.precalc_path + "MIMIC4_height_weight.parquet"
        )

        return height_weight.lazy()

    # endregion

    # region TS helper
    # make available the common processing steps for the MIMIC-IV timeseries
    def extract_timeseries_helper(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """
        Process timeseries event data to calculate time offsets from ICU admission.

        Steps:
            1. Join input data with patient IDs (including {intime}).
            2. Convert string timestamps to datetime.
            3. Calculate the offset in seconds from ICU admission.
            4. Filter events to those within ICU stay and a pre-ICU cutoff.

        Returns:
            pl.LazyFrame: A lazy frame with these columns:
                - {timeseries_time_col}: Time offset in seconds from ICU admission.
                - Other original measurement/value columns.
        """
        IDs = self.extract_patient_IDs()

        return (
            data.join(IDs, on=self.hospital_stay_id_col, how="left")
            .with_columns(pl.col("intime").str.to_datetime("%Y-%m-%d %H:%M:%S"))
            .with_columns(
                (pl.col("charttime") - pl.col("intime")).alias("offset")
            )
            .drop("charttime", "intime")
            # Keep only data within timeframe of ICU stay + PRE_ICU_TIMESERIES_DAYS_CUTOFF
            .filter(
                pl.col("offset")
                < pl.duration(days=1) * pl.col(self.icu_length_of_stay_col),
                pl.col("offset")
                > pl.duration(days=-self.PRE_ICU_TIMESERIES_DAYS_CUTOFF),
            )
            .with_columns(
                pl.col("offset")
                .dt.total_seconds()
                .cast(float)
                .alias(self.timeseries_time_col)
            )
            .drop(self.icu_length_of_stay_col)
        )

    # region vitals
    # Extract measurements from the chartevents.csv file
    def extract_chartevents(self) -> pl.LazyFrame:
        """
        Extract vital measurement events from chartevents.

        Steps:
            1. Load vital sign mappings and merge from two data sources.
            2. Filter to keep only relevant vital, respiratory, and intake-output signals.
            3. Read chartevents CSV and standardize column names.
            4. Parse timestamps and apply timeseries helper for time offset calculation.
            5. Join with vital sign mappings and apply categorical replacements for rhythm/device types.
            6. Remove null/duplicate rows.

        Returns:
            pl.LazyFrame: Contains columns:
                - {hospital_stay_id_col}: Hospital admission identifier.
                - label: Vital sign measurement name.
                - VALUENUM: Measurement value.
                - {timeseries_time_col}: Time offset (seconds from ICU admission).
        """
        meas_chartevents_main_original_data = (
            pl.scan_csv(self.meas_chartevents_main_path)
            .select("itemid (omop_source_code)", "label", "omop_concept_name")
            .with_columns(
                pl.when(pl.col("label") == "Temperature Celsius")
                .then(pl.lit("Temperature"))
                .otherwise(pl.col("omop_concept_name"))
                .alias("omop_concept_name")
            )
            .drop("label")
            .rename(
                {
                    "itemid (omop_source_code)": "itemid",
                    "omop_concept_name": "label",
                }
            )
        )
        meas_chartevents_main_additional_data = (
            pl.scan_csv(self.meas_chartevents_main_additional_path)
            .select("itemid (omop_source_code)", "omop_concept_name")
            .rename(
                {
                    "itemid (omop_source_code)": "itemid",
                    "omop_concept_name": "label",
                }
            )
        )
        meas_chartevents_main_data = (
            pl.concat(
                [
                    meas_chartevents_main_original_data,
                    meas_chartevents_main_additional_data,
                ],
                how="vertical",
            ).with_columns(
                pl.col("label").replace(
                    {
                        **self.timeseries_vitals_mapping,
                        **self.timeseries_intakeoutput_mapping,
                        **self.timeseries_respiratory_mapping,
                        **self.timeseries_extracorporeal_mapping,
                    }
                )
            )
            # Filter for names of interest
            .filter(
                pl.col("label").is_not_null(),
                # lab values are stored in the labevents.csv file and just
                # duplicated to chartevents.csv
                pl.col("label").is_in(self.all_relevant_values),
            )
        )

        if "parquet" in self.chartevents_path:
            chartevents = pl.scan_parquet(self.chartevents_path)
        else:
            chartevents = pl.scan_csv(
                self.chartevents_path,
                schema_overrides={"VALUE": str, "VALUENUM": float},
            )

        return (
            chartevents
            # Select relevant columns
            .select("hadm_id", "itemid", "charttime", "value", "valuenum")
            # Rename columns for consistency
            .rename({"hadm_id": self.hospital_stay_id_col})
            .with_columns(
                pl.col("charttime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col(self.hospital_stay_id_col).cast(int),
            )
            .pipe(self.extract_timeseries_helper)
            .pipe(self._compute_cam_icu)
            .join(meas_chartevents_main_data, on="itemid", how="left")
            .with_columns(
                pl.when(pl.col("label") == "Heart rate rhythm")
                .then(
                    pl.col("value").replace_strict(
                        self.HEART_RHYTHM_MAP, default=None
                    )
                )
                .when(pl.col("label") == "Oxygen delivery system")
                .then(
                    pl.col("value").replace_strict(
                        self.OXYGEN_DELIVERY_SYSTEM_MAP, default=None
                    )
                )
                .when(pl.col("label") == "Ventilation mode Ventilator")
                .then(
                    pl.col("value").replace_strict(
                        self.VENTILATOR_MODE_MAP, default=None
                    )
                )
                .when(
                    pl.col("label")
                    == "Continuous renal replacement therapy mode Renal replacement therapy circuit"
                )
                .then(
                    pl.col("value").replace_strict(
                        self.RRT_MODE_MAP, default=None
                    )
                )
                .when(pl.col("label") == "Confusion Assessment Method")
                .then(
                    pl.col("value").replace_strict(
                        self.DELIRIUM_MAP, default=None
                    )
                )
                .otherwise(None)
                .alias("value"),
            )
            .drop("itemid")
            # Remove rows with empty names
            .filter(pl.col("label").is_not_null(), pl.col("label") != "")
            # Remove rows with empty values (numeric or categorical string)
            .filter(
                pl.col("valuenum").is_not_null()
                | pl.col("value").is_not_null(),
            )
            # Remove duplicate rows
            .unique()
        )

    # endregion

    # region CAM-ICU
    # compute delirium status based on the CAM-ICU criteria using chartevents data
    def _compute_cam_icu(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """
        Compute CAM-ICU delirium status based on chartevents data.

        Steps:
            1. Check for precomputed parquet file; load if available.
            2. Otherwise, filter data for each CAM-ICU component.
            4. Join component scores and compute overall delirium status.
            5. Group by patient/time and select first valid score.
            6. Save result as parquet file for future reuse.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time offset (seconds) from ICU admission.
                - Acute change or fluctuating course of mental status
                - Inattention
                - Altered level of consciousness
                - Disorganized thinking
                - CAM-ICU delirium status (positive/negative)
        """

        if os.path.isfile(self.precalc_path + "MIMIC4_cam_icu.parquet"):
            return pl.concat(
                [
                    pl.scan_parquet(
                        self.precalc_path + "MIMIC4_cam_icu.parquet"
                    )
                    # custom itemid to identify precomputed CAM-ICU data
                    .select(
                        self.icu_stay_id_col,
                        self.timeseries_time_col,
                        pl.lit(999999).alias("itemid"),
                        pl.col("CAM-ICU delirium status").alias("value"),
                    ),
                    data,
                ],
                how="diagonal_relaxed",
            )

        COMPONENTS = [
            "Acute change or fluctuating course of mental status",
            "Inattention",
            "Altered level of consciousness",
            "Disorganized thinking",
        ]

        cam_icu = (
            data.select(
                self.icu_stay_id_col,
                self.timeseries_time_col,
                "value",
                "itemid",
            )
            .filter(
                pl.col("itemid").is_in(
                    [228300, 228337, 229326]    # CAM-ICU MS Change
                    + [228301, 228336, 229325]  # CAM-ICU Inattention
                    + [228302, 228334]          # CAM-ICU Altered LOC
                    + [228303, 228335, 229324]  # CAM-ICU Disorganized Thinking
                )
            )
            .with_columns(
                pl.when(pl.col("value").str.contains("Yes"))
                .then(pl.lit("Yes"))
                .when(pl.col("value").str.contains("No"))
                .then(pl.lit("No"))
                .otherwise(pl.lit("Unable to Assess"))
                .alias("value"),
                pl.when(pl.col("itemid").is_in([228300, 228337, 229326]))
                .then(pl.lit("Acute change or fluctuating course of mental status"))
                .when(pl.col("itemid").is_in([228301, 228336, 229325]))
                .then(pl.lit("Inattention"))
                .when(pl.col("itemid").is_in([228302, 228334]))
                .then(pl.lit("Altered level of consciousness"))
                .when(pl.col("itemid").is_in([228303, 228335, 229324]))
                .then(pl.lit("Disorganized thinking"))
                .alias("label"),
            )
            .collect()
            .pivot(
                index=[self.icu_stay_id_col, self.timeseries_time_col],
                on="label",
                values="value",
                aggregate_function="first",
            )
            .with_columns(
                pl.when(pl.any_horizontal(pl.col(COMPONENTS) == "Unable to Assess"))
                .then(pl.lit("Unable to Assess"))
                .when(pl.any_horizontal(pl.col(COMPONENTS) == "No"))
                .then(pl.lit("CAM-ICU negative"))
                .otherwise(pl.lit("CAM-ICU positive"))
                .alias("CAM-ICU delirium status")
            )
            .select(
                self.icu_stay_id_col,
                self.timeseries_time_col,
                *COMPONENTS,
                "CAM-ICU delirium status",
            )
            .lazy()
        ) # fmt: skip

        cam_icu.sink_parquet(self.precalc_path + "MIMIC4_cam_icu.parquet")

        return pl.concat(
            [
                # custom itemid to identify precomputed CAM-ICU data
                cam_icu.select(
                    self.icu_stay_id_col,
                    self.timeseries_time_col,
                    pl.lit(999999).alias("itemid"),
                    pl.col("CAM-ICU delirium status").alias("value"),
                ),
                data,
            ],
            how="diagonal_relaxed",
        )

    # endregion

    # region lab
    # Extract lab measurements from the labevents.csv file
    def extract_lab_measurements(self) -> pl.LazyFrame:
        """
        Extract laboratory measurement events from labevents.

        Steps:
            1. Load LOINC mapping data and filter to keep relevant lab components and systems.
            2. Read labevents CSV and standardize column names.
            3. Parse timestamps and calculate time offset via timeseries helper.
            4. Join with LOINC mappings and filter for valid lab names and result values.
            5. Extract specimen information for blood gas samples.
            6. Create struct column with lab value, LOINC system, method, time, and code.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time offset (seconds from ICU admission).
                - label: Lab component name.
                - labstruct: Struct containing value, system, method, time, LOINC code.
        """
        d_labitems_to_loinc_data = (
            pl.scan_csv(self.d_labitems_to_loinc_path)
            .select("itemid (omop_source_code)", "omop_concept_name", "category")
            .rename(
                {
                    "itemid (omop_source_code)": "itemid",
                    "omop_concept_name": "label",
                }
            )
            .with_columns(
                pl.col("label")
                # "/100 leukocytes" obselete in v20250827
                # -> now without "/100", kept for compatibility and conversion
                .str.replace("/100 leukocytes", "/Leukocytes")
                .str.replace("/100 erythrocytes", "/Erythrocytes")
            )
        ) # fmt: skip
        labnames = (
            d_labitems_to_loinc_data.select("label")
            .unique()
            .collect()
            .to_series()
            .to_list()
        )

        d_labitems_to_loinc_data = (
            d_labitems_to_loinc_data
            # Add columns for LOINC components and systems
            .with_columns(
                pl.col("label")
                .replace_strict(
                    self.omop.get_lab_component_from_name(labnames),
                    default=None,
                )
                .alias("LOINC_component"),
                pl.col("label")
                .replace_strict(
                    self.omop.get_lab_system_from_name(labnames),
                    default=None,
                )
                .alias("LOINC_system"),
                pl.col("label")
                .replace_strict(
                    self.omop.get_lab_method_from_name(labnames),
                    default=None,
                )
                .alias("LOINC_method"),
                pl.col("label").replace_strict(
                    self.omop.get_lab_time_aspect_from_name(labnames),
                    default=None,
                )
                # remove "Point in time (spot)" values
                .replace({"Point in time (spot)": None}).alias("LOINC_time"),
                pl.col("label")
                .replace_strict(
                    self.omop.get_concept_codes_from_names(labnames),
                    default=None,
                )
                .alias("LOINC_code"),
            )
            # Filter for lab names of interest
            .filter(
                pl.col("LOINC_component").is_in(
                    self.relevant_lab_LOINC_components
                )
            )
            # Filter for systems of interest
            .filter(
                pl.col("LOINC_system").is_in(
                    pl.col("LOINC_component").replace_strict(
                        self.relevant_lab_LOINC_systems,
                        return_dtype=pl.List(str),
                        default=None,
                    )
                )
            )
        )

        if "parquet" in self.labevents_path:
            labevents = pl.scan_parquet(self.labevents_path)
        else:
            labevents = pl.scan_csv(self.labevents_path)

        SPECIMEN_ID = 52033
        SPECIMENS = (
            labevents.filter(pl.col("itemid") == SPECIMEN_ID)
            .select("specimen_id", "value")
            .with_columns(
                pl.col("value").replace(self.lab_specimen_map, default=None)
            )
        )

        return (
            labevents.select(
                "hadm_id", "specimen_id", "itemid", "charttime", "valuenum"
            )
            # Rename columns for consistency
            .rename({"hadm_id": self.hospital_stay_id_col})
            .with_columns(
                pl.col("charttime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col(self.hospital_stay_id_col).cast(int),
            )
            .pipe(self.extract_timeseries_helper)
            .join(d_labitems_to_loinc_data, on="itemid", how="left")
            # Remove rows with empty lab names
            .filter(pl.col("label").is_not_null() & (pl.col("label") != ""))
            # Remove rows with empty lab results
            .filter(pl.col("valuenum").is_not_null())
            # Remove duplicate rows
            .unique()
            # Cast valuenum to float
            .cast({"valuenum": float})
            # Replace the systems as determined by the specimen
            .join(SPECIMENS, on="specimen_id", how="left", coalesce=True)
            .with_columns(
                pl.col("LOINC_component").alias("label"),
                pl.coalesce(
                    pl.col("value"),
                    pl.col("LOINC_system"),
                ).alias("LOINC_system"),
            )
            # MAKE STRUCT
            .with_columns(
                pl.struct(
                    value=pl.col("valuenum"),
                    system=pl.col("LOINC_system"),
                    method=pl.col("LOINC_method"),
                    time=pl.col("LOINC_time"),
                    LOINC=pl.col("LOINC_code"),
                ).alias("labstruct")
            )
            .select(
                self.icu_stay_id_col,
                self.timeseries_time_col,
                "itemid",
                "label",
                "labstruct",
            )
        )

    # endregion

    # region output
    # Extract output measurements from the outputevents.csv file
    def extract_output_measurements(self) -> pl.LazyFrame:
        """
        Extract input and output (fluid balance) measurements.

        Steps:
            1. Load mappings for input/output event labels from multiple sources.
            2. Read input events and output events CSV files.
            3. Standardize column names and concat all sources.
            4. Parse timestamps and calculate time offset via timeseries helper.
            5. Join with label mappings and apply categorical replacement.
            6. Remove null/duplicate rows and filter for relevant measurements.

        Returns:
            pl.LazyFrame: Contains columns:
                - {hospital_stay_id_col}: Hospital admission identifier.
                - {timeseries_time_col}: Time offset (seconds from ICU admission).
                - label: Input/output measurement name.
                - VALUENUM: Measurement value.
        """
        # NOTE: ASSUMPTION: These are the lab values of interest
        # TODO: Confer with medical experts to confirm these are the correct values
        outputevents_to_loinc_data = (
            pl.scan_csv(self.outputevents_to_loinc_path)
            .select("itemid (omop_source_code)", "omop_concept_name")
            .rename(
                {
                    "itemid (omop_source_code)": "itemid",
                    "omop_concept_name": "label",
                }
            )
            .cast({"itemid": str})
            # Harmonize names of interest
            .with_columns(
                pl.col("label").replace_strict(
                    self.timeseries_intakeoutput_mapping, default=None
                )
            )
            # Filter for names of interest
            .filter(pl.col("label").is_in(self.relevant_intakeoutput_values))
        )
        input_mappings = self.helpers.load_mapping(self.inputs_mapping_path)

        # Load correct inputevents file
        if "parquet" in self.inputevents_path:
            inputevents = pl.scan_parquet(self.inputevents_path)
        else:
            inputevents = pl.scan_csv(
                self.inputevents_path,
                schema_overrides={"amount": float},
            )

        inputevents = (
            inputevents.select(
                "hadm_id",
                "storetime",
                "ordercategoryname",
                "amount",
                "amountuom",
            )
            # rename columns for consistency
            .rename(
                {
                    "hadm_id": self.hospital_stay_id_col,
                    "storetime": "charttime",
                    "amount": "valuenum",
                    "ordercategoryname": "itemid",
                }
            )
            .filter(pl.col("amountuom") == "mL")
            .drop("amountuom")
        )

        outputevents = (
            pl.scan_csv(
                self.outputevents_path, infer_schema_length=100000
            ).select("hadm_id", "itemid", "charttime", "value")
            # Rename columns for consistency
            .rename({"hadm_id": self.hospital_stay_id_col, "value": "valuenum"})
        )

        return (
            pl.concat([inputevents, outputevents], how="diagonal_relaxed")
            .with_columns(
                pl.col("charttime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col(self.hospital_stay_id_col).cast(int),
            )
            .pipe(self.extract_timeseries_helper)
            .join(outputevents_to_loinc_data, on="itemid", how="left")
            .with_columns(
                pl.when(pl.col("label").is_null())
                .then(
                    pl.col("itemid").replace_strict(
                        input_mappings, default=None
                    )
                )
                .otherwise(pl.col("label"))
                .alias("label")
            )
            .drop("itemid")
            # Remove rows with empty names
            .filter(pl.col("label").is_not_null() & (pl.col("label") != ""))
            # Remove rows with empty values
            .filter(pl.col("valuenum").is_not_null())
            # Remove duplicate rows
            .unique()
        )

    # endregion

    # region microbiology
    # Extract microbiology data from the microbiologyevents.csv file
    def extract_microbiology(self) -> pl.LazyFrame:
        """
        Extract microbiology test results from microbiologyevents CSV and compute time offsets.

        Steps:
            1. Join microbiology events with patient ICU stay timings.
            2. Convert charttime and intime to datetime.
            3. Compute offset from ICU admission.
            4. Concatenate dilution comparison and value into {micro_dilution_col}.
            5. Filter events within the designated time window.

        Returns:
            pl.LazyFrame: A lazy frame containing:
                - {hospital_stay_id_col}: Hospital stay identifier.
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time offset in seconds.
                - {micro_specimen_col}: Specimen type.
                - {micro_organism_col}: Identified microorganism.
                - {micro_antibiotic_col}: Antibiotic tested.
                - {micro_dilution_col}: Combined dilution comparison and value.
        """
        print("MIMIC4  - Extracting microbiology...")

        intimes = self.extract_patient_IDs().select(
            self.icu_stay_id_col, self.icu_length_of_stay_col, "intime"
        )

        def _create_mapping(
            data: pl.LazyFrame, column_in: str, column_out: str
        ) -> pl.LazyFrame:
            return data.with_columns(
                pl.col(column_in)
                .replace(
                    self.omop.get_concept_names_from_ids(
                        data.select(column_in).collect().to_series().to_list()
                    ),
                    return_dtype=pl.String,
                    default=None,
                )
                .alias(column_out)
            )

        micro_specimen_mapping = pl.scan_csv(self.micro_specimen_path)
        micro_specimen_mapping = (
            micro_specimen_mapping.rename({"concept_name": "spec_type_desc"})
            .pipe(_create_mapping, "target_concept_id", self.micro_specimen_col)
            .select("spec_type_desc", self.micro_specimen_col)
        )
        micro_test_mapping = pl.scan_csv(self.micro_microtest_path)
        micro_test_mapping = (
            micro_test_mapping.rename({"concept_name": "test_name"})
            .pipe(_create_mapping, "target_concept_id", self.micro_test_col)
            .select("test_name", self.micro_test_col)
        )
        micro_organism_mapping = pl.scan_csv(self.micro_organism_path)
        micro_organism_mapping = (
            micro_organism_mapping.rename({"concept_name": "org_name"})
            .pipe(_create_mapping, "target_concept_id", self.micro_organism_col)
            .select("org_name", self.micro_organism_col)
        )
        micro_antibiotic_mapping = pl.scan_csv(self.micro_antibiotic_path)
        micro_antibiotic_mapping = (
            micro_antibiotic_mapping.rename({"concept_name": "ab_name"})
            .pipe(
                _create_mapping, "target_concept_id", self.micro_antibiotic_col
            )
            .select("ab_name", self.micro_antibiotic_col)
        )

        return (
            pl.scan_csv(self.microbiologyevents_path)
            .select(
                "subject_id",
                "charttime",
                "spec_type_desc",
                "test_name",
                "org_name",
                "ab_name",
                "dilution_comparison",
                "dilution_value",
                "interpretation",
            )
            # rename columns for consistency
            .rename(
                {
                    "subject_id": self.person_id_col,
                    # "spec_type_desc": self.micro_specimen_col,
                    # "test_name": self.micro_test_col,
                    # "org_name": self.micro_organism_col,
                    # "ab_name": self.micro_antibiotic_col,
                    "interpretation": self.micro_sensitivity_col,
                }
            )
            .join(self.icu_stay_id, on=self.person_id_col, how="left")
            # include only ICU patients
            .filter(pl.col(self.icu_stay_id_col).is_not_null())
            .join(intimes, on=self.icu_stay_id_col)
            # Add mappings
            .join(micro_specimen_mapping, on="spec_type_desc", how="left")
            .join(micro_test_mapping, on="test_name", how="left")
            .join(micro_organism_mapping, on="org_name", how="left")
            .join(micro_antibiotic_mapping, on="ab_name", how="left")
            # Convert timestamps to datetime
            .with_columns(
                pl.col("intime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("charttime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
            .with_columns(
                (pl.col("charttime") - pl.col("intime")).alias("offset"),
                pl.concat_str(
                    pl.when(pl.col("dilution_comparison") == "=")
                    .then(pl.lit("=="))
                    .otherwise(pl.col("dilution_comparison")),
                    pl.lit(" "),
                    pl.col("dilution_value"),
                ).alias(self.micro_dilution_col),
            )
            .drop("charttime", "intime")
            # keep only microbiology within timeframe of icu stay + PRE_ICU_TIMESERIES_DAYS_CUTOFF
            .filter(
                pl.col("offset")
                < pl.duration(days=1) * pl.col(self.icu_length_of_stay_col),
                pl.col("offset")
                > pl.duration(days=-self.PRE_ICU_TIMESERIES_DAYS_CUTOFF),
            )
            .with_columns(
                (pl.col("offset").dt.total_seconds())
                .cast(float)
                .alias(self.timeseries_time_col)
            )
            .drop(self.icu_length_of_stay_col)
            # remove rows with empty values
            .filter(
                pl.col(self.timeseries_time_col).is_not_null(),
                pl.col(self.micro_specimen_col).is_not_null(),
                pl.col(self.micro_test_col).is_not_null(),
            )
            # remove duplicate rows
            .unique()
        )

    # endregion

    # region medications
    # Extract medications from the inputevents.csv file
    def extract_medications(self) -> pl.LazyFrame:
        """
        Extract medication administration events from input events.

        Steps:
            1. Read input events from CSV file.
            2. Load medication mappings, routes, and drug class lookups.
            3. Standardize columns and convert timestamps to datetime.
            4. Calculate relative start/end times from ICU admission.
            5. Apply medication ingredient mappings.
            6. Filter to relevant timeframe and remove null/duplicate records.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {drug_mixture_id_col}: Mixture identifier.
                - {drug_mixture_admin_id_col}: Mixture administration identifier.
                - {drug_name_col}: Drug name.
                - {drug_ingredient_col}: Active drug ingredient (mapped).
                - {drug_amount_col}: Drug amount.
                - {drug_amount_unit_col}: Amount unit.
                - {drug_rate_col}: Administration rate.
                - {drug_rate_unit_col}: Rate unit.
                - {drug_start_col}: Relative start time (seconds).
                - {drug_end_col}: Relative end time (seconds).
                - {drug_patient_weight_col}: Patient weight used for dosing.
        """
        print("MIMIC4  - Extracting medications...")

        intimes = self.extract_patient_IDs().select(
            self.icu_stay_id_col, "intime", self.icu_length_of_stay_col
        )

        # Load additional mappings
        mimic4_drug_administration_route_mapping = self.helpers.load_mapping(
            self.drug_administration_route_mapping_path
        )
        mimic4_drug_class_mapping = self.helpers.load_mapping(
            self.drug_class_mapping_path
        )

        d_items = (
            pl.read_csv(self.d_items_path, infer_schema_length=10000)
            .select("itemid", "label")
            .lazy()
        )

        # region INPUTEVENTS
        ########################################################################
        inputevents_to_rxnorm_data = (
            pl.scan_csv(self.inputevents_to_rxnorm_path)
            .select("itemid (omop_source_code)", "omop_concept_name")
            .rename(
                {
                    "itemid (omop_source_code)": "itemid",
                    "omop_concept_name": "label_OMOP",
                }
            )
        )

        # Extract medication mappings by building a chain of references
        # 1. Get drug name references from our mapping files
        references = (
            pl.read_csv(self.inputevents_to_rxnorm_path)
            .filter(pl.col("omop_concept_name") != "Unmapped")
            .select("itemid (omop_source_code)", "omop_concept_id")
            .drop_nulls("itemid (omop_source_code)")
            .unique()
        )

        drug_references = dict(
            zip(
                references["itemid (omop_source_code)"].to_numpy(),
                references["omop_concept_id"].to_numpy(),
            )
        )
        concept_ids = drug_references.values()

        # 2. Retrieve active ingredients for these concept IDs
        ingredients = self.omop.get_ingredient(concept_ids, return_dict=False)

        # 3. Create a mapping from drug names to their active ingredients
        # Convert drug_references dictionary to DataFrame
        drug_references_df = pl.from_dict(
            {
                "itemid": list(drug_references.keys()),
                "drug_concept_id": list(drug_references.values()),
            }
        )

        # Join drug references with ingredients to get all drug-ingredient mappings
        # This preserves one-to-many relationships (one drug to multiple ingredients)
        itemid_to_ingredient = (
            drug_references_df.join(
                ingredients, on="drug_concept_id", how="inner"
            )
            .rename({"ingredient_name": self.drug_ingredient_col})
            .select("itemid", self.drug_ingredient_col)
            .lazy()
        )

        # Load correct inputevents file
        if "parquet" in self.inputevents_path:
            inputevents = pl.scan_parquet(self.inputevents_path)
        else:
            inputevents = pl.scan_csv(
                self.inputevents_path,
                schema_overrides={
                    "amount": float,
                    "totalamount": float,
                    "patientweight": float,
                },
            )

        inputevents = (
            inputevents.select(
                "hadm_id",
                "stay_id",
                "starttime",
                "endtime",
                "itemid",
                "amount",
                "amountuom",
                "rate",
                "rateuom",
                "orderid",
                "linkorderid",
                "ordercategoryname",
                "secondaryordercategoryname",
                "ordercomponenttypedescription",
                "ordercategorydescription",
                "patientweight",
            )
            .rename(
                {
                    "hadm_id": self.hospital_stay_id_col,
                    "stay_id": self.icu_stay_id_col,
                    "amount": self.drug_amount_col,
                    "amountuom": self.drug_amount_unit_col,
                    "rate": self.drug_rate_col,
                    "rateuom": self.drug_rate_unit_col,
                    "linkorderid": self.drug_mixture_id_col,
                    "orderid": self.drug_mixture_admin_id_col,
                    "patientweight": self.drug_patient_weight_col,
                }
            )
            .with_columns(
                pl.concat_str(
                    pl.col(self.icu_stay_id_col),
                    pl.col(self.drug_mixture_id_col),
                    separator="-",
                ).alias(self.drug_mixture_id_col),
                pl.concat_str(
                    pl.col(self.icu_stay_id_col),
                    pl.col(self.drug_mixture_admin_id_col),
                    separator="-",
                ).alias(self.drug_mixture_admin_id_col),
            )
            .with_columns(
                pl.col("ordercategoryname")
                .replace(mimic4_drug_administration_route_mapping, default=None)
                .alias(self.drug_admin_route_col),
                pl.col("ordercategoryname")
                .replace(mimic4_drug_class_mapping, default=None)
                .alias(self.drug_class_col),
                # Rename units
                pl.col(self.drug_rate_unit_col)
                .str.replace("grams", "g")
                .str.replace("hour", "hr")
                .str.replace("mL", "ml")
                .str.replace("mEq\.", "mEq")
                .str.replace("units", "U")
                .str.replace("µ", "mc"),
                # Add a column to indicate if the drug is continuous
                pl.col("ordercategorydescription")
                .str.contains("Continuous")
                .alias(self.drug_continuous_col),
                # Add a column to indicate the administration type
                pl.lit("given").alias(self.drug_admin_type_col),
            )
            .drop("ordercategorydescription")
        )

        # specifically handle certain medications differently
        # https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/concepts/medication
        # -> only norepinephrine has incorrect units

        inputevents = inputevents.with_columns(
            # norepinephrine
            # two rows in mg/kg/min... rest in mcg/kg/min
            # -> the rows in mg/kg/min are documented incorrectly
            # -> all rows converted into mcg/kg/min
            pl.when(pl.col("itemid") == 221906)
            .then(
                pl.when(pl.col(self.drug_rate_unit_col) == "mg/kg/min")
                .then(pl.col(self.drug_rate_col).mul(1000))
                .otherwise(pl.col(self.drug_rate_col))
            )
            .alias(self.drug_rate_col),
            pl.when(pl.col("itemid") == 221906)
            .then(
                pl.when(pl.col(self.drug_rate_unit_col) == "mg/kg/min")
                .then(pl.lit("mcg/kg/min"))
                .otherwise(pl.col(self.drug_rate_unit_col))
            )
            .alias(self.drug_rate_unit_col),
        )

        # select all inputevents that have no secondary associated order
        inputevents_no_secondary = inputevents.filter(
            pl.col("secondaryordercategoryname").is_null(),
            pl.col("ordercategoryname") != "03-IV Fluid Bolus",
        ).drop("secondaryordercategoryname", "ordercomponenttypedescription")

        # select all inputevents that are fluids only
        inputevents_fluids_only = (
            inputevents.filter(
                pl.col("secondaryordercategoryname").is_null(),
                pl.col("ordercategoryname") == "03-IV Fluid Bolus",
            )
            .rename(
                {
                    "itemid": "itemid_fluid",
                    self.drug_amount_col: self.fluid_amount_col,
                    self.drug_rate_col: self.fluid_rate_col,
                }
            )
            .drop("secondaryordercategoryname", "ordercomponenttypedescription")
        )

        # select all input events that are drips (drugs in a continuous infusion)
        inputevents_drips = (
            inputevents.filter(
                pl.col("secondaryordercategoryname").is_not_null(),
                pl.col("secondaryordercategoryname")
                .str.contains("Additive")
                .not_(),
                pl.col("ordercomponenttypedescription")
                == "Main order parameter",
            )
            .join(
                # with drips the main order parameter is the drug itself,
                # the fluid is the mixture solution
                inputevents.filter(
                    pl.col("secondaryordercategoryname").is_not_null(),
                    pl.col("secondaryordercategoryname")
                    .str.contains("Additive")
                    .not_(),
                    pl.col("ordercomponenttypedescription") == "Mixed solution",
                )
                .rename(
                    {
                        "itemid": "itemid_fluid",
                        self.drug_amount_col: self.fluid_amount_col,
                        self.drug_rate_col: self.fluid_rate_col,
                    }
                )
                .select(
                    self.drug_mixture_admin_id_col,
                    "itemid_fluid",
                    self.fluid_amount_col,
                    self.fluid_rate_col,
                ),
                on=self.drug_mixture_admin_id_col,
                how="left",
            )
            .drop("secondaryordercategoryname", "ordercomponenttypedescription")
        )

        # select all input events that are additives (drugs added to a continuous infusion)
        inputevents_additives = (
            inputevents.filter(
                pl.col("secondaryordercategoryname").is_not_null(),
                pl.col("secondaryordercategoryname").str.contains("Additive"),
                pl.col("ordercomponenttypedescription").str.contains(
                    "Additive"
                ),
            )
            .join(
                # with additives the main order parameter is the fluid
                inputevents.filter(
                    pl.col("secondaryordercategoryname").is_not_null(),
                    pl.col("secondaryordercategoryname")
                    .str.contains("Additive")
                    .not_(),
                    pl.col("ordercomponenttypedescription")
                    == "Main order parameter",
                )
                .rename(
                    {
                        "itemid": "itemid_fluid",
                        self.drug_amount_col: self.fluid_amount_col,
                        self.drug_rate_col: self.fluid_rate_col,
                    }
                )
                .select(
                    self.drug_mixture_admin_id_col,
                    "itemid_fluid",
                    self.fluid_amount_col,
                    self.fluid_rate_col,
                ),
                on=self.drug_mixture_admin_id_col,
                how="left",
            )
            .drop("secondaryordercategoryname", "ordercomponenttypedescription")
        )

        inputevents = (
            pl.concat(
                [
                    inputevents_no_secondary,
                    inputevents_fluids_only,
                    inputevents_drips,
                    inputevents_additives,
                ],
                how="diagonal_relaxed",
            )
            .join(d_items, on="itemid", how="left")
            .join(
                d_items,
                left_on="itemid_fluid",
                right_on="itemid",
                how="left",
                suffix="_fluid",
            )
            .join(inputevents_to_rxnorm_data, on="itemid", how="left")
            .join(itemid_to_ingredient, on="itemid", how="left")
            .drop("itemid", "itemid_fluid")
            # Rename columns for consistency
            .rename(
                {
                    "label": self.drug_name_col,
                    "label_OMOP": self.drug_name_OMOP_col,
                    "label_fluid": self.fluid_name_col,
                }
            )
            # Replace drug names with mapped names
            .with_columns(
                pl.col(self.fluid_name_col)
                .replace_strict(self.SOLUTION_FLUIDS_MAP, default=None)
                .alias(self.fluid_group_col),
            )
        )

        # region PRESCRIPTIONS
        ########################################################################
        # Load medication mappings from MIMIC-IV OMOP files
        # These mappings connect medication names to standard concepts and ingredients
        print("MIMIC4  - Loading medication mapping files...")

        if "parquet" in self.prescriptions_path:
            prescriptions = pl.scan_parquet(self.prescriptions_path)
        else:
            prescriptions = pl.scan_csv(
                self.prescriptions_path,
                schema_overrides={
                    "dose_val_rx": str,
                    "doses_per_24_hrs": float,
                },
                infer_schema_length=10000,
            )

        # 1. Load route and administration mappings
        route_to_concept = (
            pl.read_csv(self.drug_route_path)
            .rename({"concept_name": "route"})
            .with_columns(
                # Map administration route concept IDs to human-readable names
                pl.col("target_concept_id")
                .replace_strict(
                    self.omop.get_concept_names_from_ids(
                        pl.read_csv(self.drug_route_path)[
                            "target_concept_id"
                        ].to_list()
                    ),
                    default=None,
                )
                .str.to_lowercase()
                .alias(self.drug_admin_route_col)
            )
            .select("route", self.drug_admin_route_col)
            .lazy()
        )

        # 2. Create NDC to RxNorm concept mappings
        # Extract unique NDC codes from prescriptions
        ndc_codes = (
            prescriptions.select("ndc").unique().collect().to_series().to_list()
        )

        # Map NDCs to RxNorm concept IDs (standardize to 11 digits with leading zeros)
        ndc_to_rxnorm = self.omop.get_rxnorm_concept_id_from_ndc(
            [str(x).zfill(11) for x in ndc_codes]
        )

        # 3. Get active ingredients for all medication concept IDs
        ingredients = self.omop.get_ingredient(list(ndc_to_rxnorm.values()))
        rxnorm_names = self.omop.get_concept_names_from_ids(
            list(ndc_to_rxnorm.values())
        )

        # 5. Create final mappings from codes to ingredients and names
        # Map NDC codes to active ingredients
        ndc_to_ingredient = {
            ndc: ingredients[rxnorm_id]
            for ndc, rxnorm_id in ndc_to_rxnorm.items()
            if rxnorm_id in ingredients
        }

        # Map NDC codes to standardized drug names
        ndc_to_drugname = {
            ndc: rxnorm_names[rxnorm_id]
            for ndc, rxnorm_id in ndc_to_rxnorm.items()
            if rxnorm_id in rxnorm_names
        }

        prescriptions = (
            prescriptions.select(
                "hadm_id",
                "pharmacy_id",
                "starttime",
                "stoptime",
                "drug",
                "ndc",
                "dose_val_rx",
                "dose_unit_rx",
                "doses_per_24_hrs",
                "route",
            )
            .rename(
                {
                    "hadm_id": self.hospital_stay_id_col,
                    "pharmacy_id": self.drug_prescription_id_col,
                    "drug": self.drug_name_col,
                    "dose_val_rx": self.drug_amount_col,
                    "dose_unit_rx": self.drug_amount_unit_col,
                }
            )
            .join(route_to_concept, on="route", how="left")
            .with_columns(
                pl.col("ndc").cast(int).alias(self.drug_code_col),
                pl.lit("prescribed")
                .cast(self.drug_admin_type_dtype)
                .alias(self.drug_admin_type_col),
                # Map NDC codes to ingredients and drug names
                pl.col("ndc")
                .replace_strict(ndc_to_ingredient, default=None)
                .alias(self.drug_ingredient_col),
                pl.col("ndc")
                .replace_strict(ndc_to_drugname, default=None)
                .alias(self.drug_name_OMOP_col),
                # Add a column to indicate if the drug is continuous
                pl.lit(False).alias(self.drug_continuous_col),
                # Calculate total doses that should have been given in period
                # -> how often was the threshold of next administration crossed within the start and stop time?
                (
                    pl.col(self.drug_amount_col).cast(float, strict=False)
                    * (
                        pl.col("stoptime").str.to_datetime("%Y-%m-%d %H:%M:%S")
                        - pl.col("starttime").str.to_datetime("%Y-%m-%d %H:%M:%S")
                    ).dt.total_hours()
                    // (24 / pl.col("doses_per_24_hrs"))
                ).alias(self.drug_amount_col),
            )
            .rename({"stoptime": "endtime"})
            .join(
                self.icu_stay_id.drop(self.person_id_col),
                on=self.hospital_stay_id_col,
                how="right",
            )
        )

        # region EMAR
        ########################################################################
        if "parquet" in self.emar_path:
            emar = pl.scan_parquet(self.emar_path)
        else:
            emar = pl.scan_csv(self.emar_path)

        if "parquet" in self.emar_detail_path:
            emar_detail = pl.scan_parquet(self.emar_detail_path)
        else:
            emar_detail = pl.scan_csv(self.emar_detail_path)

        emar_detail = (
            emar_detail.select(
                "emar_id",
                "pharmacy_id",
                "dose_given",
                "dose_given_unit",
                pl.coalesce(
                    "product_description", "product_description_other"
                ).alias("product_description"),
            )
            .with_columns(
                # avoid '-' entries in dose_given; these edge cases are:
                # 10678311-96, 11509567-159, 11781252-27, 11820032-50,
                # 11992178-58, 12242144-77, 12619068-444, 13677643-11,
                # 14347713-32, 14756839-33, 14879136-820, 15242698-52,
                # 15679519-210, 16142906-6, 16203494-60, 19204653-118,
                # 19417622-8
                pl.when(pl.col("dose_given").str.contains("-"))
                .then(None)
                .otherwise(
                    pl.col("dose_given")
                    .str.replace_all("_", "")
                    .str.strip_chars()
                    .replace("", None)
                )
                .cast(float)
                .alias("dose_given")
            )
            .rename(
                {
                    "pharmacy_id": self.drug_prescription_id_col,
                    "dose_given": self.drug_amount_col,
                    "dose_given_unit": self.drug_amount_unit_col,
                    "product_description": self.drug_name_col,
                }
            )
            .group_by("emar_id")
            .agg(
                pl.col(self.drug_amount_col).sum(),
                pl.col(
                    self.drug_prescription_id_col,
                    self.drug_amount_unit_col,
                    self.drug_name_col,
                ).max(),
            )
        )

        emar = (
            emar.select(
                "hadm_id",
                "emar_id",
                "pharmacy_id",
                "charttime",
                "event_txt",
            )
            .rename(
                {
                    "hadm_id": self.hospital_stay_id_col,
                    "pharmacy_id": self.drug_prescription_id_col,
                }
            )
            .with_columns(
                # TODO: incomplete mapping; extend as needed
                pl.col("event_txt")
                .replace(
                    {
                        "Administered": "given",  # 30389320
                        "Flushed": "given",  # 2948067
                        "Started": "given",  # 948840
                        "Applied": "given",  # 242679
                        "Delayed Administered": "given",  # 164946
                        "Administered Bolus from IV Drip": "given",  # 154997
                        "Delayed Started": "given",  # 18594
                        "Administered in Other Location": "given",  # 13432
                        "Partial Administered": "given",  # 11228
                    }
                )
                .alias(self.drug_admin_type_col)
            )
            .join(emar_detail, on="emar_id", how="left")
            .join(
                prescriptions.select(
                    self.drug_prescription_id_col,
                    self.drug_ingredient_col,
                ),
                on=self.drug_prescription_id_col,
                how="left",
            )
            .join(
                self.icu_stay_id.drop(self.person_id_col),
                on=self.hospital_stay_id_col,
                how="right",
            )
            .with_columns(
                pl.col("charttime").alias("starttime"),
                pl.col("charttime")
                .str.to_datetime("%Y-%m-%d %H:%M:%S")
                .add(pl.duration(minutes=1))
                .dt.to_string("%Y-%m-%d %H:%M:%S")
                .alias("endtime"),
            )
        )

        # region COMBINED
        ########################################################################
        return (
            pl.concat(
                [inputevents, prescriptions, emar], how="diagonal_relaxed"
            )
            .join(intimes, on=self.icu_stay_id_col)
            # Change times to relative times
            .with_columns(
                pl.col("intime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("starttime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("endtime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
            .with_columns(
                (pl.col("starttime") - pl.col("intime"))
                .dt.total_seconds()
                .alias(self.drug_start_col),
                (pl.col("endtime") - pl.col("intime"))
                .dt.total_seconds()
                .alias(self.drug_end_col),
            )
            # Keep only drugs within timeframe of ICU stay + PRE_ICU_TIMESERIES_DAYS_CUTOFF
            .filter(
                pl.col(self.drug_start_col)
                < pl.duration(days=1).dt.total_seconds()
                * pl.col(self.icu_length_of_stay_col),
                pl.col(self.drug_start_col)
                > pl.duration(
                    days=-self.PRE_ICU_TIMESERIES_DAYS_CUTOFF
                ).dt.total_seconds(),
            )
            .drop(
                self.hospital_stay_id_col,
                "starttime",
                "endtime",
                "intime",
                self.icu_length_of_stay_col,
            )
        )

    # endregion

    # region diagnoses
    # Extract diagnoses from the diagnoses_icd.csv file
    def extract_diagnoses(self) -> pl.LazyFrame:
        """
        Extract hospital discharge diagnoses.

        Steps:
            1. Read diagnosis events and join with patient identifiers.
            2. Load ICD diagnosis descriptions.
            3. Cast identifiers to appropriate types and add ICD version info.
            4. Mark all diagnoses as discharge diagnoses (billing diagnoses).
            5. Join with descriptions and filter out null ICD codes.

        Returns:
            pl.LazyFrame: Contains columns:
                - {person_id_col}: Patient identifier.
                - {hospital_stay_id_col}: Hospital admission identifier.
                - {diagnosis_icd_code_col}: ICD code.
                - {diagnosis_icd_version_col}: ICD version (9 or 10).
                - {diagnosis_priority_col}: Diagnosis priority/sequence.
                - {diagnosis_description_col}: Diagnosis description.
                - {diagnosis_discharge_col}: Discharge diagnosis flag.
        """
        print("MIMIC4  - Extracting diagnoses...")
        diagnoses = pl.scan_csv(
            self.diagnoses_icd_path, schema_overrides={"icd_code": str}
        ).rename(
            {
                "subject_id": self.person_id_col,
                "hadm_id": self.hospital_stay_id_col,
            }
        )
        d_diagnoses = pl.scan_csv(
            self.d_icd_diagnoses_path, schema_overrides={"icd_code": str}
        )

        return (
            diagnoses.select(
                self.person_id_col,
                self.hospital_stay_id_col,
                "icd_code",
                "icd_version",
                "seq_num",
            )
            # include only ICU patients
            .filter(
                pl.col(self.hospital_stay_id_col).is_in(
                    self.extract_patient_IDs()
                    .select(self.hospital_stay_id_col)
                    .collect()
                    .to_series()
                )
            )
            .with_columns(
                pl.col(self.hospital_stay_id_col).cast(int),
                # NOTE: all diagnoses in MIMIC are discharge diagnoses for billing purposes
                pl.lit(True).alias(self.diagnosis_discharge_col),
            )
            .join(
                d_diagnoses.select("icd_code", "icd_version", "long_title"),
                on="icd_code",
                how="left",
            )
            .rename(
                {
                    "icd_code": self.diagnosis_icd_code_col,
                    "icd_version": self.diagnosis_icd_version_col,
                    "seq_num": self.diagnosis_priority_col,
                    "long_title": self.diagnosis_description_col,
                }
            )
            # drop rows with empty ICD codes
            .filter(pl.col(self.diagnosis_icd_code_col).is_not_null())
            # drop duplicates
            .unique()
        )

    # endregion

    # region procedures
    # Extract procedures from the procedureevents.csv and procedures_icd.csv file
    def extract_procedures(self) -> pl.LazyFrame:
        """
        Extract procedure data from procedureevents and procedures_icd.

        Steps:
            1. Load procedure mappings from ITEMID to concept names.
            2. Extract procedureevents and join with ICU stay times.
            3. Calculate relative procedure start/end times from ICU admission.
            4. Extract ICD procedures and add procedure-specific metadata.
            5. Extract datetime-based procedures and concat all sources.
            6. Remove null descriptions and duplicates.

        Returns:
            pl.LazyFrame: Contains columns:
                - {person_id_col}: Patient identifier.
                - {hospital_stay_id_col}: Hospital admission identifier.
                - {icu_stay_id_col}: ICU stay identifier (if available).
                - {procedure_start_col}: Relative start time (seconds).
                - {procedure_end_col}: Relative end time (seconds).
                - {procedure_category_col}: Procedure category.
                - {procedure_description_col}: Procedure description.
                - {procedure_icd_code_col}: ICD procedure code (if from ICD source).
                - {procedure_icd_version_col}: ICD version (9 or 10).
                - {procedure_priority_col}: Procedure priority.
                - {procedure_discharge_col}: Discharge procedure flag.
        """
        print("MIMIC4  - Extracting procedures...")

        intimes = self.extract_patient_IDs().select(
            self.icu_stay_id_col, "intime"
        )

        d_icd_procedures = pl.scan_csv(
            self.d_icd_procedures_path, schema_overrides={"icd_code": str}
        )
        proc_itemid_data = (
            pl.scan_csv(self.proc_itemid_path)
            .select("itemid (omop_source_code)", "omop_concept_name")
            .rename(
                {
                    "itemid (omop_source_code)": "itemid",
                    "omop_concept_name": "label",
                }
            )
        )
        proc_datetimeevents_data = (
            pl.scan_csv(self.proc_datetimeevents_path)
            .filter(pl.col("omop_domain_id") == "Procedure")
            .select("itemid (omop_source_code)", "omop_concept_name")
            .rename(
                {
                    "itemid (omop_source_code)": "itemid",
                    "omop_concept_name": "label",
                }
            )
        )

        procedureevents = (
            pl.scan_csv(self.procedureevents_path)
            .rename(
                {
                    "subject_id": self.person_id_col,
                    "hadm_id": self.hospital_stay_id_col,
                    "stay_id": self.icu_stay_id_col,
                }
            )
            .select(
                self.person_id_col,
                self.hospital_stay_id_col,
                self.icu_stay_id_col,
                "ordercategoryname",
                "starttime",
                "endtime",
                "itemid",
            )
            .join(intimes, on=self.icu_stay_id_col, how="left")
            .join(proc_itemid_data, on="itemid", how="left")
            .with_columns(
                pl.col("starttime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("endtime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("intime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
            .with_columns(
                (pl.col("starttime") - pl.col("intime"))
                .dt.total_seconds()
                .alias(self.procedure_start_col),
                (pl.col("endtime") - pl.col("intime"))
                .dt.total_seconds()
                .alias(self.procedure_end_col),
            )
            .drop("itemid", "starttime", "endtime", "intime")
            .rename(
                {
                    "ordercategoryname": self.procedure_category_col,
                    "label": self.procedure_description_col,
                }
            )
            .drop_nulls(self.procedure_description_col)
            .unique()
        )

        procedures_icd = (
            pl.scan_csv(
                self.procedures_icd_path, schema_overrides={"icd_code": str}
            )
            .rename(
                {
                    "subject_id": self.person_id_col,
                    "hadm_id": self.hospital_stay_id_col,
                }
            )
            .select(
                self.person_id_col,
                self.hospital_stay_id_col,
                "icd_code",
                "icd_version",
                "seq_num",
            )
            # include only ICU patients
            .filter(
                pl.col(self.hospital_stay_id_col).is_in(
                    self.extract_patient_IDs()
                    .select(self.hospital_stay_id_col)
                    .collect()
                    .to_series()
                )
            )
            .with_columns(
                # NOTE: all ICD procedures in MIMIC are on discharge for billing purposes
                pl.lit(True).alias(self.procedure_discharge_col),
            )
            .join(
                d_icd_procedures.select(
                    "icd_code", "icd_version", "long_title"
                ),
                on="icd_code",
            )
            .rename(
                {
                    "icd_code": self.procedure_icd_code_col,
                    "icd_version": self.procedure_icd_version_col,
                    "long_title": self.procedure_description_col,
                    "seq_num": self.procedure_priority_col,
                }
            )
            # drop rows with empty ICD codes
            .filter(pl.col(self.procedure_icd_code_col).is_not_null())
            .unique()
        )

        datetimeevents = (
            pl.scan_csv(self.datetimeevents_path)
            .rename(
                {
                    "subject_id": self.person_id_col,
                    "hadm_id": self.hospital_stay_id_col,
                    "stay_id": self.icu_stay_id_col,
                }
            )
            .select(
                self.person_id_col,
                self.hospital_stay_id_col,
                self.icu_stay_id_col,
                "itemid",
                "value",
            )
            .join(intimes, on=self.icu_stay_id_col, how="left")
            .join(proc_datetimeevents_data, on="itemid", how="left")
            .with_columns(
                pl.col("value").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("intime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
            .with_columns(
                (pl.col("value") - pl.col("intime"))
                .dt.total_seconds()
                .alias(self.procedure_start_col)
            )
            .drop("intime", "value")
            .rename({"label": self.procedure_description_col})
            .drop_nulls(self.procedure_description_col)
            .unique()
        )

        return pl.concat(
            [procedureevents, procedures_icd, datetimeevents],
            how="diagonal_relaxed",
        )

    # endregion

    # region notes
    # Extract clinical notes from the noteevents.csv file
    def extract_notes(self) -> pl.LazyFrame:
        """
        Extract and process clinical notes from noteevents.csv.

        Steps:
            1. Read clinical notes and rename columns for merging.
            2. Filter to include only ICU stay notes.
            3. Standardize and convert timestamps, computing {note_time_col} as relative time.
            4. Select relevant columns for the final output.
        Returns:
            pl.LazyFrame: A lazy frame with the columns:
                - {person_id_col}: Patient identifier.
                - {hospital_stay_id_col}: Hospital admission identifier.
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Relative time of the note (in seconds).
                - {note_category_col}: Category of the note.
                - {note_text_col}: Full text of the clinical note.
        """
        print("MIMIC4  - Extracting notes...")

        # DISCHARGE SUMMARIES
        # ----------------------------------------------------------------------
        discharge_summaries = (
            pl.scan_csv(self.discharge_summaries_path)
            .rename(
                {
                    "subject_id": self.person_id_col,
                    "hadm_id": self.hospital_stay_id_col,
                    "text": self.note_text_col,
                }
            )
            .select(
                self.person_id_col,
                self.hospital_stay_id_col,
                # use "storetime" for actual entry as charttime refers to discharge date only
                pl.col("storetime").alias("charttime"),
                pl.lit("Discharge summary").alias(self.note_category_col),
                pl.lit(None).alias(self.note_description_col),
                self.note_text_col,
            )
        )

        # RADIOLOGY REPORTS
        # ----------------------------------------------------------------------
        radiology_exam_name = (
            pl.scan_csv(self.radiology_reports_detail_path)
            .filter(pl.col("field_name") == "exam_name")
            .select("note_id", "field_value")
            .rename({"field_value": "description"})
        )
        radiology_addendum_note = (
            pl.scan_csv(self.radiology_reports_detail_path)
            .filter(pl.col("field_name") == "addendum_note_id")
            .select("note_id", "field_value")
            .rename({"field_value": "addendum_note_id"})
        )
        radiology = pl.scan_csv(self.radiology_reports_path)
        radiology = (
            radiology.filter(pl.col("note_type") == "RR")
            .join(radiology_exam_name, on="note_id", how="left")
            .join(radiology_addendum_note, on="note_id", how="left")
            # NOTE: this drops the time of the addendum note
            .join(
                radiology.filter(pl.col("note_type") == "AR").select(
                    pl.col("note_id").alias("addendum_note_id"),
                    pl.col("text").alias("addendum_text"),
                ),
                on="addendum_note_id",
                how="left",
            )
            .with_columns(
                pl.lit("Radiology").alias("category"),
                pl.concat_str(
                    pl.col("text"),
                    pl.when(pl.col("addendum_text").is_not_null())
                    .then(
                        pl.concat_str(
                            pl.lit("\n\n"),
                            pl.col("addendum_text"),
                        )
                    )
                    .otherwise(pl.lit("")),
                    separator="",
                    ignore_nulls=True,
                ).alias("text"),
            )
            .rename(
                {
                    "subject_id": self.person_id_col,
                    "hadm_id": self.hospital_stay_id_col,
                    "category": self.note_category_col,
                    "description": self.note_description_col,
                    "text": self.note_text_col,
                }
            )
            .select(
                self.person_id_col,
                self.hospital_stay_id_col,
                "charttime",
                self.note_category_col,
                self.note_description_col,
                self.note_text_col,
            )
        )

        return (
            pl.concat([discharge_summaries, radiology], how="vertical_relaxed")
            .with_columns(
                pl.col("charttime").str.to_datetime("%Y-%m-%d %H:%M:%S")
            )
            .pipe(self.extract_timeseries_helper)
            # Remove rows with empty lab results
            .filter(pl.col(self.note_text_col).is_not_null())
            # Remove duplicate rows
            .unique()
            .select(
                self.person_id_col,
                self.hospital_stay_id_col,
                self.icu_stay_id_col,
                pl.col(self.timeseries_time_col).alias(self.note_time_col),
                self.note_category_col,
                self.note_description_col,
                self.note_text_col,
            )
        )

    # endregion
