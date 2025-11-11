# Author: Finn Fassbender
# Last modified: 2025-10-30

# Description: This script extracts the data from the source files and provides the extracted data
# in a structured format for further processing and harmonization.

import os.path

import polars as pl

from ..helper import GlobalHelpers
from ..helper_filepaths import MIMIC3Paths
from ..helper_OMOP import Vocabulary


class MIMIC3Extractor(MIMIC3Paths):
    def __init__(self, paths, DEMO=False):
        super().__init__(paths, DEMO)
        self.path = paths.mimic3_source_path
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
            "ART": "Blood arterial",
            "CENTRAL VENOUS": "Blood central venous",
            "MIX": "Blood mixed venous",
            "VEN": "Blood venous",
        }

    # region IDs
    # Extract the patient IDs that are used in the MIMIC-III dataset
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
                - INTIME: ICU admission datetime.
        """
        return (
            pl.scan_csv(self.icustays_path)
            .rename(
                {
                    "ICUSTAY_ID": self.icu_stay_id_col,
                    "HADM_ID": self.hospital_stay_id_col,
                    "SUBJECT_ID": self.person_id_col,
                    "LOS": self.icu_length_of_stay_col,
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
                "INTIME",
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
                - {admission_loc_col}: Admission location.
                - {specialty_col}: Treating specialty.
                - {unit_type_col}: ICU unit type.
                - {care_site_col}: Care site identifier.
                - {discharge_loc_col}: Discharge location.
        """
        # scanning csv files to build labels DataFrame
        icustays = pl.scan_csv(self.icustays_path).rename(
            {
                "ICUSTAY_ID": self.icu_stay_id_col,
                "HADM_ID": self.hospital_stay_id_col,
                "SUBJECT_ID": self.person_id_col,
                "LOS": self.icu_length_of_stay_col,
                "FIRST_CAREUNIT": self.unit_type_col,
            }
        )

        admissions = (
            pl.scan_csv(self.admissions_path)
            .rename(
                {
                    "HADM_ID": self.hospital_stay_id_col,
                    "ETHNICITY": self.ethnicity_col,
                    "ADMISSION_LOCATION": self.admission_loc_col,
                    "DISCHARGE_LOCATION": self.discharge_loc_col,
                    "ADMISSION_TYPE": self.admission_urgency_col,
                    "HOSPITAL_EXPIRE_FLAG": self.mortality_hosp_col,
                }
            )
            .select(
                self.hospital_stay_id_col,
                self.ethnicity_col,
                self.admission_loc_col,
                self.discharge_loc_col,
                self.admission_urgency_col,
                self.mortality_hosp_col,
                "ADMITTIME",
                "DISCHTIME",
                "DEATHTIME",
            )
        )

        patients = (
            pl.scan_csv(self.patients_path)
            .rename(
                {
                    "SUBJECT_ID": self.person_id_col,
                    "GENDER": self.gender_col,
                }
            )
            .select(self.person_id_col, self.gender_col, "DOB", "DOD")
        )

        # calculate mortality after discharge censor cutoff (150 days after last hospital admission)
        MORTALITY_AFTER_CENSOR_CUTOFF = (
            pl.scan_csv(self.admissions_path)
            .select("SUBJECT_ID", "ADMITTIME")
            .rename({"SUBJECT_ID": self.person_id_col})
            .with_columns(
                pl.col("ADMITTIME").str.to_datetime("%Y-%m-%d %H:%M:%S")
            )
            .group_by(self.person_id_col)
            .agg(pl.col("ADMITTIME").max().alias("LAST_ADMITTIME"))
            .with_columns(
                pl.col("LAST_ADMITTIME")
                .dt.offset_by("150d")
                .alias("CENSORTIME")
            )
            .select(self.person_id_col, "CENSORTIME")
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
                MORTALITY_AFTER_CENSOR_CUTOFF,
                on=self.person_id_col,
                how="left",
                coalesce=True,
            )
            .with_columns(
                pl.col("INTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("OUTTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("DOB").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("DOD").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("ADMITTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("DISCHTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("DEATHTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col(self.icu_stay_id_col).cast(int),
                pl.col(self.hospital_stay_id_col).cast(int),
                pl.col(self.icu_length_of_stay_col).cast(float),
                pl.lit("Beth Israel Deaconess Medical Center").alias(
                    self.care_site_col
                ),
            )
            .with_columns(
                # Calculate age
                (
                    pl.col("INTIME").dt.date().dt.year()
                    - pl.col("DOB").dt.date().dt.year()
                    - (  # -1 if birthday not yet reached this year
                        (pl.col("INTIME").dt.month() < pl.col("DOB").dt.month())
                        & (pl.col("INTIME").dt.day() < pl.col("DOB").dt.day())
                    ).cast(int)
                ).alias(self.age_col),
            )
            .with_columns(
                # Convert categorical gender to enum
                pl.col(self.gender_col)
                .replace({"M": "Male", "F": "Female"})
                .cast(self.gender_dtype),
                # Convert categorical ethnicity to enum
                pl.col(self.ethnicity_col)
                .replace(self.ETHNICITY_MAP)
                .cast(self.ethnicity_dtype),
                # Fix age
                # NOTE: ASSUMPTION: Replace age values of 300 with 90 and convert to int
                # cf. https://github.com/MIT-LCP/mimic-code/issues/637
                pl.when((pl.col(self.age_col) >= 299))
                .then(90)
                .otherwise(pl.col(self.age_col))
                .cast(int)
                .alias(self.age_col),
                # Calculate pre ICU length of stay
                (pl.col("INTIME") - pl.col("ADMITTIME"))
                .truediv(pl.duration(days=1))
                .cast(float)
                .alias(self.pre_icu_length_of_stay_col),
                # Calculate hospital length of stay
                (pl.col("DISCHTIME") - pl.col("ADMITTIME"))
                .truediv(pl.duration(days=1))
                .cast(float)
                .alias(self.hospital_length_of_stay_col),
                # Calculate admission time
                pl.col("INTIME").dt.time().alias(self.admission_time_col),
                # Calculate ICU mortality
                (pl.col("DEATHTIME") - pl.col("OUTTIME"))
                .truediv(pl.duration(hours=1))
                .le(pl.duration(hours=self.ICU_DISCHARGE_MORTALITY_CUTOFF))
                .cast(bool)
                # .fill_null(False)
                .alias(self.mortality_icu_col),
                # Calculate hospital mortality
                pl.col(self.mortality_hosp_col).cast(bool),
                # Calculate mortality after discharge
                (pl.col("DOD") - pl.col("OUTTIME"))
                .truediv(pl.duration(days=1))
                .cast(int)
                .alias(self.mortality_after_col),
                # Calculate mortality after discharge cutoff
                (pl.col("CENSORTIME") - pl.col("OUTTIME"))
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
                # Add dataset source
                pl.when(pl.col("DBSOURCE") == "carevue")
                .then(pl.lit("v1.4 (CareVue)"))
                .when(pl.col("DBSOURCE") == "metavision")
                .then(pl.lit("v1.4 (MetaVision)"))
                .otherwise(pl.lit("v1.4 (Mixed)"))
                .alias(self.dataset_version_col),
            )
            # Calculate ICU stay sequence number
            .sort(self.person_id_col, "INTIME")
            .with_columns(
                (pl.int_range(pl.len()).over(self.person_id_col) + 1).alias(
                    self.icu_stay_seq_num_col
                ),
                # Calculate time relative to first ICU admission
                (
                    pl.col("INTIME")
                    - pl.col("INTIME").min().over(self.person_id_col)
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
        Extract treating specialty for each ICU stay.

        Steps:
            1. Get ICU admission times from patient IDs.
            2. Read services CSV and join with admission information.
            3. Filter for specialties assigned within 2 hours before ICU admission.
            4. Select the most recent specialty for each ICU stay.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {specialty_col}: Treating specialty.
        """
        IDs = self.extract_patient_IDs().select(
            self.hospital_stay_id_col, self.icu_stay_id_col, "INTIME"
        )

        services = pl.scan_csv(self.services_path).rename(
            {
                "HADM_ID": self.hospital_stay_id_col,
                "CURR_SERVICE": self.specialty_col,
            }
        )

        return (
            services.select(
                self.hospital_stay_id_col, "TRANSFERTIME", self.specialty_col
            )
            .join(IDs, on=self.hospital_stay_id_col, how="outer")
            .with_columns(
                pl.col("TRANSFERTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("INTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
            # Get the most recent specialty
            .filter(
                pl.col("TRANSFERTIME")
                < (pl.col("INTIME") + pl.duration(hours=2))
            )
            # Get the most recent specialty on ICU admission
            .sort("TRANSFERTIME")
            .group_by(self.icu_stay_id_col)
            .last()
            .select(self.icu_stay_id_col, self.specialty_col)
        )

    # endregion

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
        Extract and compute patient height and weight from chartevents.

        Height IDs taken from:
        https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/concepts/demographics/heightweight.sql
        Weight IDs taken from:
        https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/concepts/durations/weight_durations.sql

        Steps:
            1. Check cache for pre-calculated parquet file (skip if force=True).
            2. Read chartevents and filter for height/weight ITEMID codes.
            3. Join with ICU stays to constrain measurements to relevant time window.
            4. Convert height from inches to cm, weight from lbs to kg.
            5. Pivot data so each ICU stay has separate height/weight columns.
            6. Cache result to parquet file.

        Returns:
            pl.DataFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {weight_col}: Patient weight (kg).
                - {height_col}: Patient height (cm).
        """
        # check if precalculated data is available
        if (
            os.path.isfile(self.precalc_path + "MIMIC3_height_weight.parquet")
            and not force
        ):
            return pl.scan_parquet(
                self.precalc_path + "MIMIC3_height_weight.parquet"
            )

        print("MIMIC3  - Extracting patient height and weight...")

        if "parquet" in self.chartevents_path:
            chartevents = pl.scan_parquet(
                self.chartevents_path, parallel="prefiltered"
            )
        else:
            chartevents = pl.scan_csv(
                self.chartevents_path,
                schema_overrides={"VALUE": str, "VALUENUM": float},
            )

        WEIGHT_ITEMIDS = {
            762: self.weight_col,  # Admit Wt [carevue]
            763: self.weight_col,  # Daily Weight [carevue]
            3580: self.weight_col,  # Present Weight  (kg) [carevue]
            3581: "weight_lbs",  # Present Weight  (lb) [carevue]
            3693: self.weight_col,  # Weight Kg [carevue]
            3723: self.weight_col,  # Birth Weight    (kg) [carevue]
            4183: self.weight_col,  # Birthweight (kg) [carevue]
            224639: self.weight_col,  # Daily Weight [metavision]
            226512: self.weight_col,  # Admission Weight (Kg) [metavision]
            226531: "weight_lbs",  # Admission Weight (lbs.) [metavision]
        }
        HEIGHT_ITEMIDS = {
            920: "height_inch",  # Admit Ht [carevue]
            1394: "height_inch",  # Height Inches [carevue]
            3485: self.height_col,  # Length   Calc   (cm) [carevue]
            3486: "height_inch",  # Length in Inches [carevue]
            4187: "height_inch",  # Length Calc Inches [carevue]
            4188: self.height_col,  # Length in cm [carevue]
            226707: "height_inch",  # Height [metavision]
            226730: self.height_col,  # Height (cm) [metavision]
        }

        KEEPIDS = [*(WEIGHT_ITEMIDS | HEIGHT_ITEMIDS).keys()]
        ADMIT_WEIGHT_IDS = [762, 3723, 4183, 226512, 226531]
        DAILY_WEIGHT_IDS = [763, 3580, 3581, 3693, 224639]

        height_weight = (
            chartevents.select("ICUSTAY_ID", "ITEMID", "VALUENUM", "CHARTTIME")
            # Rename columns for consistency
            .rename({"ICUSTAY_ID": self.icu_stay_id_col})
            .filter(pl.col("ITEMID").is_in(KEEPIDS))
            .join(
                icustays.select(self.icu_stay_id_col, "INTIME"),
                on=self.icu_stay_id_col,
                how="left",
            )
            .with_columns(
                pl.col("CHARTTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("INTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("ITEMID")
                .replace_strict(
                    WEIGHT_ITEMIDS | HEIGHT_ITEMIDS,  # "|" merges dictionaries
                    default=None,
                )
                .alias("LABEL"),
            )
            .with_columns(
                # Convert height in in to cm, weight in lbs to kg
                pl.when(pl.col("LABEL") == "height_inch")
                .then(pl.col("VALUENUM").mul(self.INCH_TO_CM))
                .when(pl.col("LABEL") == "weight_lbs")
                .then(pl.col("VALUENUM").mul(self.LBS_TO_KG))
                .otherwise(pl.col("VALUENUM"))
                .alias("VALUENUM"),
                # Rename LABEL to height_cm / weight_kg
                pl.when(pl.col("LABEL") == "height_inch")
                .then(pl.lit(self.height_col))
                .when(pl.col("LABEL") == "weight_lbs")
                .then(pl.lit(self.weight_col))
                .otherwise(pl.col("LABEL"))
                .alias("LABEL"),
            )
        )

        # Backfill weights as in:
        # https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/concepts/durations/weight_durations.sql
        # Select the first admission weight
        first_admit_weight = (
            height_weight.filter(pl.col("ITEMID").is_in(ADMIT_WEIGHT_IDS))
            .collect()
            .with_columns(
                pl.col("VALUENUM")
                .drop_nulls()
                .first()
                .over(partition_by=self.icu_stay_id_col, order_by="CHARTTIME")
                .alias("FIRST_ADMIT_WEIGHT")
            )
        )
        # Select the first daily weight measurement
        first_daily_weight = (
            height_weight.filter(pl.col("ITEMID").is_in(DAILY_WEIGHT_IDS))
            .collect()
            .with_columns(
                pl.col("VALUENUM")
                .drop_nulls()
                .first()
                .over(partition_by=self.icu_stay_id_col, order_by="CHARTTIME")
                .alias("FIRST_DAILY_WEIGHT")
            )
        )
        # Combine the first admit and first daily weight measurements
        weight = (
            pl.concat([first_admit_weight, first_daily_weight], how="diagonal")
            .drop("VALUENUM")
            .with_columns(
                pl.coalesce(
                    pl.col("FIRST_ADMIT_WEIGHT"), pl.col("FIRST_DAILY_WEIGHT")
                ).alias("VALUENUM")
            )
            .drop("FIRST_ADMIT_WEIGHT", "FIRST_DAILY_WEIGHT")
        )

        # Height measurements from the first 24 hours of the ICU stay since it's unlikely to change
        height = (
            height_weight.filter(pl.col("ITEMID").is_in(HEIGHT_ITEMIDS.keys()))
            .collect()
            .filter(
                (pl.col("CHARTTIME") - pl.col("INTIME")).le(
                    pl.duration(hours=24)
                )
            )
        )

        height_weight = (
            pl.concat([weight, height], how="diagonal")
            .drop("ITEMID", "INTIME", "CHARTTIME")
            .pivot(
                index=self.icu_stay_id_col,
                on="LABEL",
                values="VALUENUM",
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
            self.precalc_path + "MIMIC3_height_weight.parquet"
        )

        return height_weight.lazy()

    # endregion

    # region TS helper
    # make available the common processing steps for the MIMIC-III timeseries
    def extract_timeseries_helper(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """
        Process raw timeseries data by aligning measurements with the ICU admission time.

        Steps:
            1. Join raw timeseries data with patient IDs to include {hospital_stay_id_col} and "INTIME".
            2. Convert "INTIME" and "CHARTTIME" to datetime.
            3. Calculate the time offset ("OFFSET") as the difference between CHARTTIME and INTIME.
            4. Filter rows so that the offset falls within pre-defined ICU stay and pre-ICU cutoff.
            5. Compute the {timeseries_time_col} by converting offset to total seconds.
            6. Cast "VALUENUM" as float and remove rows with nulls in "VALUENUM".

        Returns:
            pl.LazyFrame: A lazy frame with the columns:
                - {timeseries_time_col}: Time offset (in seconds) relative to ICU admission.
                - Other columns from the input data (with "VALUENUM" as float).
        """
        IDs = self.extract_patient_IDs()

        return (
            data.join(IDs, on=self.hospital_stay_id_col, how="left")
            .with_columns(pl.col("INTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"))
            .with_columns(
                (pl.col("CHARTTIME") - pl.col("INTIME")).alias("OFFSET")
            )
            .drop("CHARTTIME", "INTIME")
            .filter(
                (
                    pl.col("OFFSET")
                    < pl.duration(days=1) * pl.col(self.icu_length_of_stay_col)
                )
                & (
                    pl.col("OFFSET")
                    > pl.duration(days=-self.PRE_ICU_TIMESERIES_DAYS_CUTOFF)
                )
            )
            .with_columns(
                (pl.col("OFFSET").dt.total_seconds())
                .cast(float)
                .alias(self.timeseries_time_col)
            )
            .drop(self.icu_length_of_stay_col)
        )

    # region vitals TS
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
                - LABEL: Vital sign measurement name.
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
                    "itemid (omop_source_code)": "ITEMID",
                    "omop_concept_name": "LABEL",
                }
            )
        )
        meas_chartevents_main_additional_data = (
            pl.scan_csv(self.meas_chartevents_main_additional_path)
            .select("itemid (omop_source_code)", "omop_concept_name")
            .rename(
                {
                    "itemid (omop_source_code)": "ITEMID",
                    "omop_concept_name": "LABEL",
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
                pl.col("LABEL").replace(
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
                pl.col("LABEL").is_not_null(),
                # lab values are stored in the labevents.csv file and just
                # duplicated to chartevents.csv
                pl.col("LABEL").is_in(self.all_relevant_values),
            )
        )

        if "parquet" in self.chartevents_path:
            chartevents = pl.scan_parquet(
                self.chartevents_path, parallel="prefiltered"
            )
        else:
            chartevents = pl.scan_csv(
                self.chartevents_path,
                schema_overrides={"VALUE": str, "VALUENUM": float},
            )

        return (
            chartevents
            # Select relevant columns
            .select("HADM_ID", "ITEMID", "CHARTTIME", "VALUE", "VALUENUM")
            # Rename columns for consistency
            .rename({"HADM_ID": self.hospital_stay_id_col})
            .with_columns(
                pl.col("CHARTTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col(self.hospital_stay_id_col).cast(int),
            )
            .pipe(self.extract_timeseries_helper)
            .join(meas_chartevents_main_data, on="ITEMID", how="left")
            .with_columns(
                pl.when(pl.col("LABEL") == "Heart rate rhythm")
                .then(
                    pl.col("VALUE")
                    .replace_strict(self.HEART_RHYTHM_MAP, default=None)
                    .replace(self.heart_rhythm_enum_map)
                )
                .when(pl.col("LABEL") == "Oxygen delivery system")
                .then(
                    pl.col("VALUE")
                    .replace_strict(
                        self.OXYGEN_DELIVERY_SYSTEM_MAP, default=None
                    )
                    .replace(self.oxygen_delivery_system_enum_map)
                )
                .when(pl.col("LABEL") == "Ventilation mode Ventilator")
                .then(
                    pl.col("VALUE")
                    .replace_strict(self.VENTILATOR_MODE_MAP, default=None)
                    .replace(self.ventilator_mode_enum_map)
                )
                .when(
                    pl.col("LABEL")
                    == "Continuous renal replacement therapy mode Renal replacement therapy circuit"
                )
                .then(
                    pl.col("VALUE")
                    .replace_strict(self.RRT_MODE_MAP, default=None)
                    .replace(self.rrt_mode_enum_map)
                )
                .otherwise(pl.col("VALUENUM"))
                .cast(float)
                .alias("VALUENUM"),
            )
            .drop("ITEMID")
            # Remove rows with empty names
            .filter(pl.col("LABEL").is_not_null())  # & (pl.col("LABEL") != ""))
            # Remove rows with empty values
            .filter(pl.col("VALUENUM").is_not_null())
            # Remove duplicate rows
            .unique()
        )

    # endregion

    # region lab TS
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
                - LABEL: Lab component name.
                - labstruct: Struct containing value, system, method, time, LOINC code.
        """
        d_labitems_to_loinc_data = (
            pl.scan_csv(self.d_labitems_to_loinc_path)
            .select("ITEMID", "COALESCED_CONCEPT_NAME", "CATEGORY")
            .rename({"COALESCED_CONCEPT_NAME": "LABEL"})
        )
        labnames = (
            d_labitems_to_loinc_data.select("LABEL")
            .unique()
            .collect()
            .to_series()
            .to_list()
        )

        d_labitems_to_loinc_data = (
            d_labitems_to_loinc_data
            # Add columns for LOINC components and systems
            .with_columns(
                pl.col("LABEL")
                .replace_strict(
                    self.omop.get_lab_component_from_name(labnames),
                    default=None,
                )
                .alias("LOINC_component"),
                pl.col("LABEL")
                .replace_strict(
                    self.omop.get_lab_system_from_name(labnames), default=None
                )
                .alias("LOINC_system"),
                pl.col("LABEL")
                .replace_strict(
                    self.omop.get_lab_method_from_name(labnames), default=None
                )
                .alias("LOINC_method"),
                pl.col("LABEL").replace_strict(
                    self.omop.get_lab_time_aspect_from_name(labnames),
                    default=None,
                )
                # remove "Point in time (spot)" values
                .replace({"Point in time (spot)": None}).alias("LOINC_time"),
                pl.col("LABEL")
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
            labevents = pl.scan_parquet(
                self.labevents_path, parallel="prefiltered"
            )
        else:
            labevents = pl.scan_csv(
                self.labevents_path, infer_schema_length=10000
            )

        SPECIMEN_ID = 50800
        SPECIMENS = (
            labevents.filter(pl.col("ITEMID") == SPECIMEN_ID)
            .select("HADM_ID", "CHARTTIME", "VALUE")
            .rename({"HADM_ID": self.hospital_stay_id_col})
            .with_columns(
                pl.col("CHARTTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("VALUE").replace(self.lab_specimen_map, default=None),
            )
            .pipe(self.extract_timeseries_helper)
        )
        BLOOD_GAS_ITEMIDS = (
            d_labitems_to_loinc_data.filter(pl.col("CATEGORY") == "Blood Gas")
            .select("ITEMID")
            .unique()
            .collect()
            .to_series()
            .to_list()
        )

        return (
            labevents.select(
                "HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM", "VALUEUOM"
            )
            # Rename columns for consistency
            .rename({"HADM_ID": self.hospital_stay_id_col})
            .with_columns(
                pl.col("CHARTTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col(self.hospital_stay_id_col).cast(int),
            )
            .pipe(self.extract_timeseries_helper)
            .join(d_labitems_to_loinc_data, on="ITEMID", how="left")
            # Remove rows with empty lab names
            .filter(pl.col("LABEL").is_not_null() & (pl.col("LABEL") != ""))
            # Remove rows with empty lab results
            .filter(pl.col("VALUENUM").is_not_null())
            # Remove duplicate rows
            .unique()
            # Cast valuenum to float
            .cast({"VALUENUM": float})
            # Replace the systems as determined by the specimen
            .join(
                SPECIMENS,
                on=[self.hospital_stay_id_col, self.timeseries_time_col],
                how="left",
                coalesce=True,
            )
            .with_columns(
                pl.col("LOINC_component").alias("LABEL"),
                # Replace only Blood Gas systems with the specimen system
                pl.when(pl.col("ITEMID").is_in(BLOOD_GAS_ITEMIDS))
                .then(
                    pl.coalesce(
                        pl.col("VALUE"),
                        pl.col("LOINC_system"),
                    ).alias("LOINC_system")
                )
                .otherwise(pl.col("LOINC_system"))
                .alias("LOINC_system"),
            )
            # MAKE STRUCT
            .with_columns(
                pl.struct(
                    value=pl.col("VALUENUM"),
                    system=pl.col("LOINC_system"),
                    method=pl.col("LOINC_method"),
                    time=pl.col("LOINC_time"),
                    LOINC=pl.col("LOINC_code"),
                ).alias("labstruct")
            )
            .select(
                self.icu_stay_id_col,
                self.timeseries_time_col,
                "ITEMID",
                "LABEL",
                "labstruct",
            )
        )

    # endregion

    # region output TS
    # Extract output measurements from the outputevents.csv file
    def extract_output_measurements(self) -> pl.LazyFrame:
        """
        Extract input and output (fluid balance) measurements.

        Steps:
            1. Load mappings for input/output event labels from multiple sources.
            2. Read input events (CareVue and MetaVision) and output events CSV files.
            3. Standardize column names and concat all sources.
            4. Parse timestamps and calculate time offset via timeseries helper.
            5. Join with label mappings and apply categorical replacement.
            6. Remove null/duplicate rows and filter for relevant measurements.

        Returns:
            pl.LazyFrame: Contains columns:
                - {hospital_stay_id_col}: Hospital admission identifier.
                - {timeseries_time_col}: Time offset (seconds from ICU admission).
                - LABEL: Input/output measurement name.
                - VALUENUM: Measurement value.
        """
        cv_input_label_to_concept = pl.scan_csv(
            self.cv_input_label_to_concept_path
        )
        cv_input_label_to_concept = (
            cv_input_label_to_concept.rename({"item_id": "ITEMID"})
            .pipe(self._mapping_from_ids, "concept_id", "LABEL")
            .select("ITEMID", "LABEL")
        )
        mv_input_label_to_concept = pl.scan_csv(
            self.mv_input_label_to_concept_path
        )
        mv_input_label_to_concept = (
            mv_input_label_to_concept.rename({"item_id": "ITEMID"})
            .pipe(self._mapping_from_ids, "concept_id", "LABEL")
            .select("ITEMID", "LABEL")
        )
        output_label_to_concept = pl.scan_csv(self.output_label_to_concept_path)
        output_label_to_concept = (
            output_label_to_concept.rename({"item_id": "ITEMID"})
            .pipe(self._mapping_from_ids, "concept_id", "LABEL")
            .select("ITEMID", "LABEL")
        )

        label_to_concept = (
            pl.concat(
                [
                    cv_input_label_to_concept,
                    mv_input_label_to_concept,
                    output_label_to_concept,
                ],
                how="vertical",
            )
            # Harmonize names of interest
            .with_columns(
                pl.col("LABEL").replace_strict(
                    self.timeseries_intakeoutput_mapping, default=None
                )
            )
            # Filter for names of interest
            .filter(pl.col("LABEL").is_in(self.relevant_intakeoutput_values))
        )
        input_mappings = self.helpers.load_mapping(self.inputs_mapping_path)

        # Load correct inputevents_cv file
        if "parquet" in self.inputevents_cv_path:
            inputevents_cv = pl.scan_parquet(
                self.inputevents_cv_path, parallel="prefiltered"
            )
        else:
            inputevents_cv = pl.scan_csv(
                self.inputevents_cv_path,
                schema_overrides={"AMOUNT": float},
            )

        inputevents_cv = (
            inputevents_cv.select(
                "HADM_ID",
                "ITEMID",
                "CHARTTIME",
                "AMOUNT",
                "AMOUNTUOM",
                "ORIGINALROUTE",
            )
            # Rename columns for consistency
            .rename(
                {
                    "HADM_ID": self.hospital_stay_id_col,
                    "AMOUNT": "VALUENUM",
                }
            )
            .filter(pl.col("AMOUNTUOM").is_in(["ml", "cc"]))
            .drop("AMOUNTUOM")
        )

        # Load correct inputevents_mv file
        if "parquet" in self.inputevents_mv_path:
            inputevents_mv = pl.scan_parquet(
                self.inputevents_mv_path, parallel="prefiltered"
            )
        else:
            inputevents_mv = pl.scan_csv(
                self.inputevents_mv_path,
                schema_overrides={"AMOUNT": float},
            )

        inputevents_mv = (
            inputevents_mv.select(
                "HADM_ID",
                "ITEMID",
                "STORETIME",
                "ORDERCATEGORYNAME",
                "AMOUNT",
                "AMOUNTUOM",
            )
            # Rename columns for consistency
            .rename(
                {
                    "HADM_ID": self.hospital_stay_id_col,
                    "STORETIME": "CHARTTIME",
                    "AMOUNT": "VALUENUM",
                    "ORDERCATEGORYNAME": "ORIGINALROUTE",
                }
            )
            .filter(pl.col("AMOUNTUOM").is_in(["ml", "cc"]))
            .drop("AMOUNTUOM")
        )

        outputevents = (
            pl.scan_csv(
                self.outputevents_path, infer_schema_length=100000
            ).select("HADM_ID", "ITEMID", "CHARTTIME", "VALUE")
            # Rename columns for consistency
            .rename({"HADM_ID": self.hospital_stay_id_col, "VALUE": "VALUENUM"})
        )

        return (
            pl.concat(
                [inputevents_cv, inputevents_mv, outputevents],
                how="diagonal_relaxed",
            )
            .with_columns(
                pl.col("CHARTTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col(self.hospital_stay_id_col).cast(int),
            )
            .pipe(self.extract_timeseries_helper)
            .join(label_to_concept, on="ITEMID", how="left")
            .with_columns(
                pl.when(pl.col("LABEL").is_null())
                .then(
                    pl.col("ORIGINALROUTE").replace_strict(
                        input_mappings, default=None
                    )
                )
                .otherwise(pl.col("LABEL"))
                .alias("LABEL")
            )
            .drop("ITEMID", "ORIGINALROUTE")
            # Remove rows with empty names
            .filter(pl.col("LABEL").is_not_null() & (pl.col("LABEL") != ""))
            # Remove rows with empty values
            .filter(pl.col("VALUENUM").is_not_null())
            # Remove duplicate rows
            .unique()
        )

    # endregion

    # region microbiology
    # Extract microbiology data from the microbiologyevents.csv file
    def extract_microbiology(self) -> pl.LazyFrame:
        """
        Extract microbiology culture results and susceptibility testing.

        Steps:
            1. Load specimen, organism, and antibiotic concept mappings from CSV files.
            2. Read microbiologyevents CSV and join with ICU stay data.
            3. Parse timestamps and compute time offset from ICU admission.
            4. Apply concept mappings to standardize specimen, organism, and antibiotic names.
            5. Combine dilution comparison and value into single column.
            6. Filter events to ICU timeframe window and remove nulls/duplicates.

        Returns:
            pl.LazyFrame: Contains columns:
                - {hospital_stay_id_col}: Hospital admission identifier.
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time offset (seconds from ICU admission).
                - {micro_specimen_col}: Specimen type.
                - {micro_organism_col}: Microorganism identified.
                - {micro_antibiotic_col}: Antibiotic tested.
                - {micro_sensitivity_col}: Sensitivity result.
                - {micro_dilution_col}: Dilution comparison with value.
        """
        print("MIMIC3  - Extracting microbiology...")

        intimes = self.extract_patient_IDs().select(
            self.icu_stay_id_col, self.icu_length_of_stay_col, "INTIME"
        )

        microbiology_specimen_to_concept_mapping = (
            pl.scan_csv(self.microbiology_specimen_to_concept_path)
            .rename(
                {
                    "label": "SPEC_TYPE_DESC",
                    "concept_name": self.micro_specimen_col,
                }
            )
            .select("SPEC_TYPE_DESC", self.micro_specimen_col)
        )
        org_name_to_concept_mapping = pl.scan_csv(self.org_name_to_concept_path)
        org_name_to_concept_mapping = (
            org_name_to_concept_mapping.rename({"org_name": "ORG_NAME"})
            .pipe(self._mapping_from_codes, "snomed", self.micro_organism_col)
            .select("ORG_NAME", self.micro_organism_col)
        )
        atb_to_concept_mapping = pl.scan_csv(self.atb_to_concept_path)
        atb_to_concept_mapping = (
            atb_to_concept_mapping.rename({"ab_name": "AB_NAME"})
            .pipe(
                self._mapping_from_codes,
                "concept_code",
                self.micro_antibiotic_col,
            )
            .select("AB_NAME", self.micro_antibiotic_col)
        )

        return (
            pl.scan_csv(self.microbiologyevents_path)
            .select(
                "HADM_ID",
                "CHARTTIME",
                "SPEC_TYPE_DESC",
                "ORG_NAME",
                "AB_NAME",
                "DILUTION_COMPARISON",
                "DILUTION_VALUE",
                "INTERPRETATION",
            )
            # Rename columns for consistency
            .rename(
                {
                    "HADM_ID": self.hospital_stay_id_col,
                    # "SPEC_TYPE_DESC": self.micro_specimen_col,
                    # "ORG_NAME": self.micro_organism_col,
                    # "AB_NAME": self.micro_antibiotic_col,
                    "INTERPRETATION": self.micro_sensitivity_col,
                }
            )
            .join(self.icu_stay_id, on=self.hospital_stay_id_col)
            .drop(self.person_id_col)
            .join(intimes, on=self.icu_stay_id_col)
            # Add mappings
            .join(
                microbiology_specimen_to_concept_mapping,
                on="SPEC_TYPE_DESC",
                how="left",
            )
            .join(org_name_to_concept_mapping, on="ORG_NAME", how="left")
            .join(atb_to_concept_mapping, on="AB_NAME", how="left")
            # Convert timestamps to datetime
            .with_columns(
                pl.col("INTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("CHARTTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
            .with_columns(
                (pl.col("CHARTTIME") - pl.col("INTIME")).alias("OFFSET"),
                pl.concat_str(
                    pl.when(pl.col("DILUTION_COMPARISON") == "=")
                    .then(pl.lit("=="))
                    .otherwise(pl.col("DILUTION_COMPARISON")),
                    pl.lit(" "),
                    pl.col("DILUTION_VALUE"),
                ).alias(self.micro_dilution_col),
            )
            .drop("CHARTTIME", "INTIME")
            # Keep only microbiology within timeframe of ICU stay + PRE_ICU_TIMESERIES_DAYS_CUTOFF
            .filter(
                pl.col("OFFSET")
                < pl.duration(days=1) * pl.col(self.icu_length_of_stay_col),
                pl.col("OFFSET")
                > pl.duration(days=-self.PRE_ICU_TIMESERIES_DAYS_CUTOFF),
            )
            .with_columns(
                (pl.col("OFFSET").dt.total_seconds())
                .cast(float)
                .alias(self.timeseries_time_col)
            )
            .drop(self.icu_length_of_stay_col)
            # Remove rows with empty values
            .filter(
                pl.col(self.timeseries_time_col).is_not_null(),
                pl.col(self.micro_specimen_col).is_not_null(),
            )
            # Remove duplicate rows
            .unique()
        )

    # endregion

    # region medications
    # Extract medications from the inputevents.csv file
    def extract_medications(self) -> pl.LazyFrame:
        """
        Extract medication administration events from input events.

        Steps:
            1. Read input events (MetaVision and CareVue) from CSV files.
            2. Load medication mappings, routes, and drug class lookups.
            3. Standardize columns and convert timestamps to datetime.
            4. Calculate relative start/end times from ICU admission.
            5. Apply medication ingredient mappings.
            6. Filter to relevant timeframe and remove null/duplicate records.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
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
        print("MIMIC3  - Extracting medications...")

        intimes = self.extract_patient_IDs().select(
            self.icu_stay_id_col, self.icu_length_of_stay_col, "INTIME"
        )

        # Map order categories to administration routes
        map_route_to_concept = (
            pl.scan_csv(self.map_route_to_concept_path)
            .with_columns(
                pl.col("concept_name")
                .str.to_lowercase()
                .alias(self.drug_admin_route_col)
            )
            .select("ordercategoryname", self.drug_admin_route_col)
        )

        # Load additional mappings
        mimic3_medication_mapping = (
            self.helpers.load_many_to_many_to_one_mapping(
                self.mapping_path + "MEDICATIONS.yaml", "mimic3"
            )
        )
        mimic3_drug_class_mapping = self.helpers.load_mapping(
            self.drug_class_mapping_path
        )

        d_items = pl.scan_csv(self.d_items_path).select("ITEMID", "LABEL")

        # region INPUTEVENTS_MV
        #######################################################################
        # Load correct inputevents_mv file
        if "parquet" in self.inputevents_mv_path:
            inputevents_mv = pl.scan_parquet(
                self.inputevents_mv_path, parallel="prefiltered"
            )
        else:
            inputevents_mv = pl.scan_csv(
                self.inputevents_mv_path,
                schema_overrides={
                    "AMOUNT": float,
                    "TOTALAMOUNT": float,
                    "PATIENTWEIGHT": float,
                },
            )

        inputevents_mv = (
            inputevents_mv.select(
                "ICUSTAY_ID",
                "STARTTIME",
                "ENDTIME",
                "ITEMID",
                "AMOUNT",
                "AMOUNTUOM",
                "RATE",
                "RATEUOM",
                "ORDERID",
                "LINKORDERID",
                "ORDERCATEGORYNAME",
                "SECONDARYORDERCATEGORYNAME",
                "ORDERCOMPONENTTYPEDESCRIPTION",
                "ORDERCATEGORYDESCRIPTION",
                "PATIENTWEIGHT",
                "CANCELREASON",
            )
            .rename(
                {
                    "ICUSTAY_ID": self.icu_stay_id_col,
                    "AMOUNT": self.drug_amount_col,
                    "AMOUNTUOM": self.drug_amount_unit_col,
                    "RATE": self.drug_rate_col,
                    "RATEUOM": self.drug_rate_unit_col,
                    "LINKORDERID": self.drug_mixture_id_col,
                    "ORDERID": self.drug_mixture_admin_id_col,
                    "PATIENTWEIGHT": self.drug_patient_weight_col,
                    "CANCELREASON": self.drug_admin_type_col,
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
            .join(
                map_route_to_concept,
                left_on="ORDERCATEGORYNAME",
                right_on="ordercategoryname",
                how="left",
            )
            .with_columns(
                pl.col("ORDERCATEGORYNAME")
                .replace(mimic3_drug_class_mapping, default=None)
                .alias(self.drug_class_col),
                # Rename units
                pl.col(self.drug_rate_unit_col)
                .str.replace("grams", "g")
                .str.replace("hour", "hr")
                .str.replace("mL", "ml")
                .str.replace("mEq\.", "mEq")
                .str.replace("units", "U"),
                # Add a column to indicate if the drug is continuous
                pl.col("ORDERCATEGORYDESCRIPTION")
                .str.contains("Continuous")
                .alias(self.drug_continuous_col),
                # Add a column to indicate the administration type
                pl.when(pl.col(self.drug_admin_type_col) == 1)
                .then(pl.lit("cancelled"))
                .when(pl.col(self.drug_admin_type_col) == 2)
                .then(pl.lit("rewritten"))
                .otherwise(pl.lit("given"))
                .alias(self.drug_admin_type_col),
            )
        )

        # select all inputevents that have no secondary associated order
        inputevents_mv_no_secondary = inputevents_mv.filter(
            pl.col("SECONDARYORDERCATEGORYNAME").is_null(),
            pl.col("ORDERCATEGORYNAME") != "03-IV Fluid Bolus",
        ).drop("SECONDARYORDERCATEGORYNAME", "ORDERCOMPONENTTYPEDESCRIPTION")

        # select all inputevents that are fluids only
        inputevents_mv_fluids_only = (
            inputevents_mv.filter(
                pl.col("SECONDARYORDERCATEGORYNAME").is_null(),
                pl.col("ORDERCATEGORYNAME") == "03-IV Fluid Bolus",
            )
            .rename(
                {
                    "ITEMID": "ITEMID_FLUID",
                    self.drug_amount_col: self.fluid_amount_col,
                    self.drug_rate_col: self.fluid_rate_col,
                }
            )
            .drop("SECONDARYORDERCATEGORYNAME", "ORDERCOMPONENTTYPEDESCRIPTION")
        )

        # select all input events that are drips (drugs in a continuous infusion)
        inputevents_mv_drips = (
            inputevents_mv.filter(
                pl.col("SECONDARYORDERCATEGORYNAME").is_not_null(),
                pl.col("SECONDARYORDERCATEGORYNAME")
                .str.contains("Additive")
                .not_(),
                pl.col("ORDERCOMPONENTTYPEDESCRIPTION")
                == "Main order parameter",
            )
            .join(
                # with drips the main order parameter is the drug itself,
                # the fluid is the mixture solution
                inputevents_mv.filter(
                    pl.col("SECONDARYORDERCATEGORYNAME").is_not_null(),
                    pl.col("SECONDARYORDERCATEGORYNAME")
                    .str.contains("Additive")
                    .not_(),
                    pl.col("ORDERCOMPONENTTYPEDESCRIPTION") == "Mixed solution",
                )
                .rename(
                    {
                        "ITEMID": "ITEMID_FLUID",
                        self.drug_amount_col: self.fluid_amount_col,
                        self.drug_rate_col: self.fluid_rate_col,
                    }
                )
                .select(
                    self.drug_mixture_admin_id_col,
                    "ITEMID_FLUID",
                    self.fluid_amount_col,
                    self.fluid_rate_col,
                ),
                on=self.drug_mixture_admin_id_col,
                how="left",
            )
            .drop("SECONDARYORDERCATEGORYNAME", "ORDERCOMPONENTTYPEDESCRIPTION")
        )

        # select all input events that are additives (drugs added to a continuous infusion)
        inputevents_mv_additives = (
            inputevents_mv.filter(
                pl.col("SECONDARYORDERCATEGORYNAME").is_not_null(),
                pl.col("SECONDARYORDERCATEGORYNAME").str.contains("Additive"),
                pl.col("ORDERCOMPONENTTYPEDESCRIPTION").str.contains(
                    "Additive"
                ),
            )
            .join(
                # with additives the main order parameter is the fluid
                inputevents_mv.filter(
                    pl.col("SECONDARYORDERCATEGORYNAME").is_not_null(),
                    pl.col("SECONDARYORDERCATEGORYNAME")
                    .str.contains("Additive")
                    .not_(),
                    pl.col("ORDERCOMPONENTTYPEDESCRIPTION")
                    == "Main order parameter",
                )
                .rename(
                    {
                        "ITEMID": "ITEMID_FLUID",
                        self.drug_amount_col: self.fluid_amount_col,
                        self.drug_rate_col: self.fluid_rate_col,
                    }
                )
                .select(
                    self.drug_mixture_admin_id_col,
                    "ITEMID_FLUID",
                    self.fluid_amount_col,
                    self.fluid_rate_col,
                ),
                on=self.drug_mixture_admin_id_col,
                how="left",
            )
            .drop("SECONDARYORDERCATEGORYNAME", "ORDERCOMPONENTTYPEDESCRIPTION")
        )

        # region INPUTEVENTS_CV
        #######################################################################
        # Load correct inputevents_cv file
        if "parquet" in self.inputevents_cv_path:
            inputevents_cv = pl.scan_parquet(
                self.inputevents_cv_path, parallel="prefiltered"
            )
        else:
            inputevents_cv = pl.scan_csv(
                self.inputevents_cv_path,
                schema_overrides={
                    "AMOUNT": float,
                    "RATE": float,
                    "TOTALAMOUNT": float,
                },
            )

        inputevents_cv = (
            inputevents_cv.select(
                "ICUSTAY_ID",
                "CHARTTIME",
                "ITEMID",
                "AMOUNT",
                "AMOUNTUOM",
                "RATE",
                "RATEUOM",
                "ORDERID",
                "LINKORDERID",
                "ORIGINALROUTE",
            )
            .rename(
                {
                    "ICUSTAY_ID": self.icu_stay_id_col,
                    "AMOUNT": self.drug_amount_col,
                    "AMOUNTUOM": self.drug_amount_unit_col,
                    "RATE": self.drug_rate_col,
                    "RATEUOM": self.drug_rate_unit_col,
                    "LINKORDERID": self.drug_mixture_id_col,
                    "ORDERID": self.drug_mixture_admin_id_col,
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
            .join(
                map_route_to_concept,
                left_on="ORIGINALROUTE",
                right_on="ordercategoryname",
                how="left",
            )
            .with_columns(
                # Rename units
                pl.col(self.drug_rate_unit_col)
                .str.replace("hr", "/hr")
                .str.replace("min", "/min")
                .str.replace("kg", "/kg"),
                # Add a column to indicate if the drug is continuous
                pl.lit(True).alias(self.drug_continuous_col),
                # Add a column to indicate the administration type
                pl.lit("given")
                .cast(self.drug_admin_type_dtype)
                .alias(self.drug_admin_type_col),
            )
        )

        # select all inputevents that only represent fluids within the same LINKORDERID
        inputevents_cv_fluids_only = (
            inputevents_cv.with_columns(
                (pl.col(self.drug_amount_unit_col) == "ml")
                .all()
                .over(self.drug_mixture_id_col)
                .alias("is_fluid_only"),
                pl.col("ITEMID").alias("ITEMID_FLUID"),
                pl.col("CHARTTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
            .filter(pl.col("is_fluid_only"))
            .drop("is_fluid_only")
            .with_columns(
                pl.col("CHARTTIME")
                .shift(1)
                .over(self.drug_mixture_id_col, order_by="CHARTTIME")
                # fill empty first with nearest previous full hour
                # -> "For volumes, the CHARTTIME will correspond to an end time."
                .fill_null(
                    pl.col("CHARTTIME")
                    .min()
                    .over(self.drug_mixture_id_col)
                    .sub(pl.duration(minutes=1))
                    .dt.truncate("1h")
                )
                .alias("PREV_CHARTTIME")
            )
            # recalculate the rate for fluids in ml/hr
            .with_columns(
                pl.col(self.drug_amount_col)
                .truediv(
                    pl.col("CHARTTIME")
                    .sub(pl.col("PREV_CHARTTIME"))
                    .dt.total_seconds()
                    .truediv(3600)
                )
                .alias(self.fluid_rate_col)
            )
            .group_by(self.drug_mixture_id_col, self.drug_mixture_admin_id_col)
            .agg(
                *[
                    pl.col(column).sort_by("CHARTTIME").drop_nulls().first()
                    for column in [
                        self.icu_stay_id_col,
                        self.fluid_rate_col,
                        self.drug_continuous_col,
                        self.drug_admin_type_col,
                        self.drug_admin_route_col,
                        "ITEMID_FLUID",
                    ]
                ],
                pl.col("PREV_CHARTTIME").min().alias("STARTTIME"),
                pl.col("CHARTTIME").max().alias("ENDTIME"),
                # sum but return None if empty
                pl.when(pl.col(self.drug_amount_col).count() > 0)
                .then(pl.col(self.drug_amount_col).sum())
                .alias(self.fluid_amount_col),
            )
        )

        # select all other inputevents
        # Split into two dataframes for drug and fluid components
        inputevents_cv_mixtures = inputevents_cv.with_columns(
            pl.col(self.drug_rate_col, self.drug_rate_unit_col)
            .forward_fill()
            .backward_fill()
            .over(self.drug_mixture_admin_id_col),
        ).filter(
            pl.col(self.drug_amount_unit_col)
            .ne_missing("ml")
            .any()
            .over(self.drug_mixture_id_col)
        )

        # Group by mixture ID and rate continuity
        inputevents_cv_mixtures = (
            inputevents_cv_mixtures.group_by(
                self.drug_mixture_id_col, self.drug_mixture_admin_id_col
            )
            .agg(
                *[
                    pl.col(column).sort_by("CHARTTIME").drop_nulls().first()
                    for column in [
                        self.icu_stay_id_col,
                        self.drug_continuous_col,
                        self.drug_admin_type_col,
                        self.drug_admin_route_col,
                    ]
                ],
                pl.col("CHARTTIME").min().alias("STARTTIME"),
                pl.col("CHARTTIME").max().alias("ENDTIME"),
            )
            .join(
                # Process drug components (not ml)
                inputevents_cv_mixtures.filter(
                    pl.col(self.drug_amount_unit_col).ne_missing("ml")
                )
                .group_by(
                    self.drug_mixture_id_col, self.drug_mixture_admin_id_col
                )
                .agg(
                    *[
                        pl.col(column).sort_by("CHARTTIME").drop_nulls().first()
                        for column in [
                            "ITEMID",
                            self.drug_rate_col,
                            self.drug_rate_unit_col,
                            self.drug_amount_unit_col,
                        ]
                    ],
                    # sum but return None if empty
                    pl.when(pl.col(self.drug_amount_col).count() > 0)
                    .then(pl.col(self.drug_amount_col).sum())
                    .alias(self.drug_amount_col),
                ),
                on=[self.drug_mixture_id_col, self.drug_mixture_admin_id_col],
                how="left",
            )
            .join(
                # process fluid components (ml)
                inputevents_cv_mixtures.filter(
                    pl.col(self.drug_amount_unit_col).eq_missing("ml")
                )
                .group_by(
                    self.drug_mixture_id_col, self.drug_mixture_admin_id_col
                )
                .agg(
                    pl.col("ITEMID").drop_nulls().first().alias("ITEMID_FLUID"),
                    pl.col(self.drug_rate_col)
                    .drop_nulls()
                    .first()
                    .alias(self.fluid_rate_col),
                    # sum but return None if empty
                    pl.when(pl.col(self.drug_amount_col).count() > 0)
                    .then(pl.col(self.drug_amount_col).sum())
                    .alias(self.fluid_amount_col),
                ),
                on=[self.drug_mixture_id_col, self.drug_mixture_admin_id_col],
                how="left",
            )
        )

        inputevents_cv_combined = (
            pl.concat(
                [inputevents_cv_fluids_only, inputevents_cv_mixtures],
                how="diagonal_relaxed",
            )
            # Drop the 0 drug amount with a non-zero rate -> non-zero rates trump zero amounts
            .with_columns(
                pl.when(
                    pl.col(self.drug_amount_col).eq(0),
                    pl.col(self.drug_rate_col).gt(0),
                )
                .then(None)
                .otherwise(pl.col(self.drug_amount_col))
                .alias(self.drug_amount_col),
            )
            # Group by rate continuity and mixture ID
            .with_columns(
                # NOTE: Sometimes there is a singular hour for which only a volume, but no rate is given.
                # Visual inspection of the data shows that these are continuous infusions with the constant rate as before and after.
                # -> forward fill and backward fill the drug rate
                pl.col(self.drug_rate_col)
                .forward_fill()
                .backward_fill()
                .over(self.drug_mixture_id_col, order_by="STARTTIME"),
            )
            .with_columns(
                pl.struct([self.drug_rate_col, self.fluid_rate_col])
                .rle_id()
                .over(self.drug_mixture_id_col, order_by="STARTTIME")
                .alias("has_same_rate")
            )
            .group_by(self.drug_mixture_id_col, "has_same_rate")
            .agg(
                *[
                    pl.col(column).sort_by("STARTTIME").drop_nulls().first()
                    for column in [
                        self.icu_stay_id_col,
                        self.drug_mixture_admin_id_col,
                        self.drug_continuous_col,
                        self.drug_admin_type_col,
                        self.drug_admin_route_col,
                        self.drug_amount_unit_col,
                        self.drug_rate_col,
                        self.drug_rate_unit_col,
                        self.fluid_rate_col,
                        "ITEMID",
                        "ITEMID_FLUID",
                    ]
                ],
                pl.col("STARTTIME").min().alias("STARTTIME"),
                pl.col("ENDTIME").max().alias("ENDTIME"),
                # sum but return None if empty
                pl.when(pl.col(self.drug_amount_col).count() > 0)
                .then(pl.col(self.drug_amount_col).sum())
                .alias(self.drug_amount_col),
                pl.when(pl.col(self.fluid_amount_col).count() > 0)
                .then(pl.col(self.fluid_amount_col).sum())
                .alias(self.fluid_amount_col),
            )
        )

        # region INPUTEVENTS
        #######################################################################
        inputevents_to_rxnorm_data = (
            pl.scan_csv(self.inputevents_to_rxnorm_path)
            .select("itemid (omop_source_code)", "omop_concept_name")
            .rename(
                {
                    "itemid (omop_source_code)": "ITEMID",
                    "omop_concept_name": "LABEL_OMOP",
                }
            )
        )

        inputevents = (
            pl.concat(
                [
                    inputevents_mv_no_secondary,
                    inputevents_mv_fluids_only,
                    inputevents_mv_drips,
                    inputevents_mv_additives,
                    inputevents_cv_combined,
                ],
                how="diagonal_relaxed",
            )
            .join(d_items, on="ITEMID", how="left")
            .join(
                d_items,
                left_on="ITEMID_FLUID",
                right_on="ITEMID",
                how="left",
                suffix="_FLUID",
            )
            .join(inputevents_to_rxnorm_data, on="ITEMID", how="left")
            .drop("ITEMID", "ITEMID_FLUID")
            # Rename columns for consistency
            .rename(
                {
                    "LABEL": self.drug_name_col,
                    "LABEL_OMOP": self.drug_name_OMOP_col,
                    "LABEL_FLUID": self.fluid_name_col,
                }
            )
            # Replace drug names with mapped names
            .with_columns(
                pl.col(self.drug_name_col)
                .replace_strict(mimic3_medication_mapping, default=None)
                .alias(self.drug_ingredient_col),
                pl.col(self.fluid_name_col)
                .replace_strict(self.SOLUTION_FLUIDS_MAP, default=None)
                .alias(self.fluid_group_col),
            )
        )

        # region PRESCRIPTIONS
        #######################################################################
        # Load medication mappings from MIMIC-III OMOP files
        # These mappings connect medication names to standard concepts and ingredients
        print("MIMIC3  - Loading medication mapping files...")

        if "parquet" in self.prescriptions_path:
            prescriptions = pl.scan_parquet(
                self.prescriptions_path, parallel="prefiltered"
            )
        else:
            prescriptions = pl.scan_csv(self.prescriptions_path)

        # 1. Load route and administration mappings
        route_to_concept = (
            pl.read_csv(self.route_to_concept_path)
            .with_columns(
                # Map administration route concept IDs to human-readable names
                pl.col("concept_id")
                .replace_strict(
                    self.omop.get_concept_names_from_ids(
                        pl.read_csv(self.route_to_concept_path)[
                            "concept_id"
                        ].to_list()
                    ),
                    default=None,
                )
                .str.to_lowercase()
                .alias(self.drug_admin_route_col)
            )
            .select("ROUTE", self.drug_admin_route_col)
            .lazy()
        )

        # Load prescriptions mapping for entries without NDC codes
        prescriptions_ndcisnullzero_to_concept = pl.read_csv(
            self.prescriptions_ndcisnullzero_to_concept_path
        )

        # 2. Create NDC to RxNorm concept mappings
        # Extract unique NDC codes from prescriptions
        ndc_codes = (
            prescriptions.select("NDC").unique().collect().to_series().to_list()
        )

        # Map NDCs to RxNorm concept IDs (standardize to 11 digits with leading zeros)
        ndc_to_rxnorm = self.omop.get_rxnorm_concept_id_from_ndc(
            [str(x).zfill(11) for x in ndc_codes]
        )

        # 3. Create mappings for prescriptions without valid NDCs
        # Map drug labels to concept IDs and names
        prescriptions_ndcisnullzero_concept_ids = dict(
            zip(
                prescriptions_ndcisnullzero_to_concept["label"],
                prescriptions_ndcisnullzero_to_concept["concept_id"],
            )
        )
        prescriptions_ndcisnullzero_concept_names = dict(
            zip(
                prescriptions_ndcisnullzero_to_concept["label"],
                prescriptions_ndcisnullzero_to_concept["concept_name"],
            )
        )

        # 4. Get active ingredients for all medication concept IDs
        all_concept_ids = list(ndc_to_rxnorm.values()) + list(
            prescriptions_ndcisnullzero_concept_ids.values()
        )
        ingredients = self.omop.get_ingredient(all_concept_ids)
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

        # Map prescription labels to active ingredients for entries without NDCs
        prescriptions_ndcisnullzero_to_ingredient = {
            label: ingredients[concept_id]
            for label, concept_id in prescriptions_ndcisnullzero_concept_ids.items()
            if concept_id in ingredients
        }

        prescriptions = (
            prescriptions.select(
                "ICUSTAY_ID",
                "STARTDATE",
                "ENDDATE",
                "DRUG",
                "NDC",
                "DOSE_VAL_RX",
                "DOSE_UNIT_RX",
                "ROUTE",
            )
            .rename(
                {
                    "ICUSTAY_ID": self.icu_stay_id_col,
                    "DRUG": self.drug_name_col,
                    "DOSE_VAL_RX": self.drug_amount_col,
                    "DOSE_UNIT_RX": self.drug_amount_unit_col,
                }
            )
            .join(route_to_concept, on="ROUTE", how="left")
            # NOTE: dirty, but necessary to join with inputevents
            .rename({"STARTDATE": "STARTTIME", "ENDDATE": "ENDTIME"})
            .with_columns(
                pl.lit("prescribed")
                .cast(self.drug_admin_type_dtype)
                .alias(self.drug_admin_type_col),
                pl.when(pl.col("NDC") != 0)
                .then(
                    pl.col("NDC").replace_strict(
                        ndc_to_ingredient, default=None
                    )
                )
                .otherwise(
                    pl.col(self.drug_name_col).replace_strict(
                        prescriptions_ndcisnullzero_to_ingredient,
                        default=None,
                    )
                )
                .alias(self.drug_ingredient_col),
                pl.when(pl.col("NDC") != 0)
                .then(
                    pl.col("NDC").replace_strict(ndc_to_drugname, default=None)
                )
                .otherwise(
                    pl.col(self.drug_name_col).replace_strict(
                        prescriptions_ndcisnullzero_concept_names,
                        default=None,
                    )
                )
                .alias(self.drug_name_OMOP_col),
                # Add a column to indicate if the drug is continuous
                pl.lit(False).alias(self.drug_continuous_col),
            )
        )

        # region RETURN
        #######################################################################
        return (
            pl.concat([inputevents, prescriptions], how="diagonal_relaxed")
            .join(intimes, on=self.icu_stay_id_col)
            # Change times to relative times
            .with_columns(
                pl.col("INTIME").str.to_datetime("%Y-%m-%d %H:%M:%S%.9f"),
                pl.col("STARTTIME").str.to_datetime("%Y-%m-%d %H:%M:%S%.9f"),
                pl.col("ENDTIME").str.to_datetime("%Y-%m-%d %H:%M:%S%.9f"),
            )
            .with_columns(
                (pl.col("STARTTIME") - pl.col("INTIME"))
                .dt.total_seconds()
                .alias(self.drug_start_col),
                (pl.col("ENDTIME") - pl.col("INTIME"))
                .dt.total_seconds()
                .alias(self.drug_end_col),
            )
            # Keep only drugs within timeframe of ICU stay + PRE_ICU_TIMESERIES_DAYS_CUTOFF
            .filter(
                (
                    pl.col(self.drug_start_col)
                    < pl.duration(days=1).dt.total_seconds()
                    * pl.col(self.icu_length_of_stay_col)
                )
                & (
                    pl.col(self.drug_start_col)
                    > pl.duration(
                        days=-self.PRE_ICU_TIMESERIES_DAYS_CUTOFF
                    ).dt.total_seconds()
                )
            )
            .drop("STARTTIME", "ENDTIME", "INTIME", self.icu_length_of_stay_col)
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
                - {diagnosis_icd_version_col}: ICD version (9 for MIMIC-III).
                - {diagnosis_priority_col}: Diagnosis priority/sequence.
                - {diagnosis_description_col}: Diagnosis description.
                - {diagnosis_discharge_col}: Discharge diagnosis flag.
        """
        print("MIMIC3  - Extracting diagnoses...")
        diagnoses = pl.scan_csv(
            self.diagnoses_icd_path, schema_overrides={"ICD9_CODE": str}
        ).rename(
            {
                "SUBJECT_ID": self.person_id_col,
                "HADM_ID": self.hospital_stay_id_col,
            }
        )
        d_diagnoses = pl.scan_csv(
            self.d_icd_diagnoses_path, schema_overrides={"ICD9_CODE": str}
        )

        return (
            diagnoses.select(
                self.person_id_col,
                self.hospital_stay_id_col,
                "ICD9_CODE",
                "SEQ_NUM",
            )
            .with_columns(
                pl.col(self.hospital_stay_id_col).cast(int),
                pl.lit(9).alias(self.diagnosis_icd_version_col),
                # NOTE: all diagnoses in MIMIC are discharge diagnoses for billing purposes
                pl.lit(True).alias(self.diagnosis_discharge_col),
            )
            .join(
                d_diagnoses.select("ICD9_CODE", "LONG_TITLE"),
                on="ICD9_CODE",
                how="left",
            )
            .rename(
                {
                    "ICD9_CODE": self.diagnosis_icd_code_col,
                    "SEQ_NUM": self.diagnosis_priority_col,
                    "LONG_TITLE": self.diagnosis_description_col,
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
            2. Extract procedureevents_mv and join with ICU stay times.
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
                - {procedure_icd_version_col}: ICD version (9 for MIMIC-III).
                - {procedure_priority_col}: Procedure priority.
                - {procedure_discharge_col}: Discharge procedure flag.
        """
        print("MIMIC3  - Extracting procedures...")

        intimes = self.extract_patient_IDs().select(
            self.icu_stay_id_col, "INTIME"
        )

        # d_items = pl.scan_csv(self.d_items_path).select("ITEMID", "LABEL")
        d_icd_procedures = pl.scan_csv(
            self.d_icd_procedures_path, schema_overrides={"ICD9_CODE": str}
        )
        proc_itemid_data = (
            pl.scan_csv(self.proc_itemid_path)
            .select("itemid (omop_source_code)", "omop_concept_name")
            .rename(
                {
                    "itemid (omop_source_code)": "ITEMID",
                    "omop_concept_name": "LABEL",
                }
            )
        )
        proc_datetimeevents_data = (
            pl.scan_csv(self.proc_datetimeevents_path)
            .filter(pl.col("omop_domain_id") == "Procedure")
            .select("itemid (omop_source_code)", "omop_concept_name")
            .rename(
                {
                    "itemid (omop_source_code)": "ITEMID",
                    "omop_concept_name": "LABEL",
                }
            )
        )

        procedureevents_mv = (
            pl.scan_csv(self.procedureevents_mv_path)
            .rename(
                {
                    "SUBJECT_ID": self.person_id_col,
                    "HADM_ID": self.hospital_stay_id_col,
                    "ICUSTAY_ID": self.icu_stay_id_col,
                }
            )
            .select(
                self.person_id_col,
                self.hospital_stay_id_col,
                self.icu_stay_id_col,
                "ORDERCATEGORYNAME",
                "STARTTIME",
                "ENDTIME",
                "ITEMID",
            )
            .join(intimes, on=self.icu_stay_id_col, how="left")
            # .join(d_items, on="ITEMID")
            .join(proc_itemid_data, on="ITEMID", how="left")
            .with_columns(
                pl.col("STARTTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("ENDTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("INTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
            .with_columns(
                (pl.col("STARTTIME") - pl.col("INTIME"))
                .dt.total_seconds()
                .alias(self.procedure_start_col),
                (pl.col("ENDTIME") - pl.col("INTIME"))
                .dt.total_seconds()
                .alias(self.procedure_end_col),
            )
            .drop("ITEMID", "STARTTIME", "ENDTIME", "INTIME")
            .rename(
                {
                    "ORDERCATEGORYNAME": self.procedure_category_col,
                    "LABEL": self.procedure_description_col,
                }
            )
            .drop_nulls(self.procedure_description_col)
            .unique()
        )

        procedures_icd = (
            pl.scan_csv(
                self.procedures_icd_path, schema_overrides={"ICD9_CODE": str}
            )
            .rename(
                {
                    "SUBJECT_ID": self.person_id_col,
                    "HADM_ID": self.hospital_stay_id_col,
                }
            )
            .select(
                self.person_id_col,
                self.hospital_stay_id_col,
                "ICD9_CODE",
                "SEQ_NUM",
            )
            .with_columns(
                pl.lit(9).alias(self.procedure_icd_version_col),
                # NOTE: all ICD procedures in MIMIC are on discharge for billing purposes
                pl.lit(True).alias(self.procedure_discharge_col),
            )
            .join(
                d_icd_procedures.select("ICD9_CODE", "LONG_TITLE"),
                on="ICD9_CODE",
            )
            .rename(
                {
                    "ICD9_CODE": self.procedure_icd_code_col,
                    "LONG_TITLE": self.procedure_description_col,
                    "SEQ_NUM": self.procedure_priority_col,
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
                    "SUBJECT_ID": self.person_id_col,
                    "HADM_ID": self.hospital_stay_id_col,
                    "ICUSTAY_ID": self.icu_stay_id_col,
                }
            )
            .select(
                self.person_id_col,
                self.hospital_stay_id_col,
                self.icu_stay_id_col,
                "ITEMID",
                "VALUE",
            )
            .join(intimes, on=self.icu_stay_id_col, how="left")
            .join(proc_datetimeevents_data, on="ITEMID", how="left")
            .with_columns(
                pl.col("VALUE").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("INTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
            .with_columns(
                (pl.col("VALUE") - pl.col("INTIME"))
                .dt.total_seconds()
                .alias(self.procedure_start_col)
            )
            .drop("INTIME", "VALUE")
            .rename({"LABEL": self.procedure_description_col})
            .drop_nulls(self.procedure_description_col)
            .unique()
        )

        return pl.concat(
            [procedureevents_mv, procedures_icd, datetimeevents],
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
        print("MIMIC3  - Extracting notes...")

        if "parquet" in self.noteevents_path:
            noteevents = pl.scan_parquet(
                self.noteevents_path, parallel="prefiltered"
            )
        else:
            noteevents = pl.scan_csv(self.noteevents_path)

        return (
            noteevents.filter(pl.col("ISERROR").is_null())
            .rename(
                {
                    "SUBJECT_ID": self.person_id_col,
                    "HADM_ID": self.hospital_stay_id_col,
                    "CATEGORY": self.note_category_col,
                    "DESCRIPTION": self.note_description_col,
                    "TEXT": self.note_text_col,
                }
            )
            .select(
                self.person_id_col,
                self.hospital_stay_id_col,
                "CHARTDATE",
                "CHARTTIME",
                self.note_category_col,
                self.note_description_col,
                self.note_text_col,
            )
            .with_columns(
                # use 00:00:00 at date if CHARTTIME is missing
                pl.coalesce(
                    pl.col("CHARTTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                    pl.col("CHARTDATE").str.to_datetime("%Y-%m-%d"),
                ).alias("CHARTTIME"),
                # alias "Nursing/other" to "Nursing"
                pl.col(self.note_category_col).replace(
                    "Nursing/other", "Nursing"
                ),
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

    # region helper functions
    def _mapping_from_codes(
        self, data: pl.LazyFrame, column_in: str, column_out: str
    ) -> pl.LazyFrame:
        return data.with_columns(
            pl.col(column_in)
            .replace_strict(
                self.omop.get_concept_names_from_codes(
                    data.select(column_in).collect().to_series().to_list()
                ),
                return_dtype=pl.String,
                default=None,
            )
            .alias(column_out)
        )

    def _mapping_from_ids(
        self, data: pl.LazyFrame, column_in: str, column_out: str
    ) -> pl.LazyFrame:
        return data.with_columns(
            pl.col(column_in)
            .replace_strict(
                self.omop.get_concept_names_from_ids(
                    data.select(column_in).collect().to_series().to_list()
                ),
                return_dtype=pl.String,
                default=None,
            )
            .alias(column_out)
        )
