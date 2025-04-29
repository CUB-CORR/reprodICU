# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script extracts the data from the source files and provides the extracted data
# in a structured format for further processing and harmonization.

import os.path

import polars as pl
from helpers.helper import GlobalHelpers
from helpers.helper_filepaths import MIMIC4Paths
from helpers.helper_OMOP import Vocabulary


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

        self.other_lab_values = [
            "Anion gap 4",
            "Bilirubin.direct [Mass/volume]",
            "Bilirubin.indirect [Mass/volume]",
            "Bilirubin.total [Mass/volume]",
            "Calcium [Mass/volume]",
            "Calcium.ionized [Mass/volume]",
            "Creatine kinase.MB [Mass/volume]",
            "Iron [Mass/volume]",
            "Iron binding capacity [Mass/volume]",
            "Magnesium [Mass/volume]",
            "Phosphate [Mass/volume]",
            "Triiodothyronine (T3) [Mass/volume]",
            "Thyroxine (T4) [Mass/volume]",
            "Thyroxine (T4) free [Mass/volume]",
            "Cobalamin (Vitamin B12) [Mass/volume]",
            "Basophils [#/volume]",
            "Eosinophils [#/volume]",
            "Lymphocytes [#/volume]",
            "Monocytes [#/volume]",
            "Neutrophils [#/volume]",
            "Reticulocytes [#/volume]",
        ]

    # region ID mapping table
    def extract_patient_IDs(self) -> pl.LazyFrame:
        """
        Extract patient IDs from the MIMIC-IV ICU stays CSV file.

        Scans the CSV file, renames columns to variable names, removes duplicates, casts
        patient ID columns, and selects only the required columns.

        Steps:
            1. Read the ICU stays CSV file.
            2. Rename columns:
               - "ICUSTAY_ID" → {icu_stay_id_col}: Unique ICU stay identifier.
               - "HADM_ID" → {hospital_stay_id_col}: Hospital admission identifier.
               - "SUBJECT_ID" → {person_id_col}: Patient identifier.
               - "LOS" → {icu_length_of_stay_col}: ICU length of stay in days.
            3. Remove duplicate rows.
            4. Cast appropriate columns to integer.
            5. Select and return the columns:
               - {icu_stay_id_col}, {hospital_stay_id_col}, {person_id_col}, {icu_length_of_stay_col}, and "INTIME".

        Returns:
            pl.LazyFrame: A lazy frame containing the columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {hospital_stay_id_col}: Hospital admission identifier.
                - {person_id_col}: Patient ID.
                - {icu_length_of_stay_col}: ICU length of stay as a float.
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
        Extract and aggregate detailed patient information from multiple MIMIC-IV CSV files.

        Joins data from ICU stays, admissions, and patients files while applying several transformations
        including date conversions, derived column computations, and unit casts.

        Steps:
            1. Scan and rename ICU stays CSV using:
               - {icu_stay_id_col}: ICU stay identifier.
               - {hospital_stay_id_col}: Hospital admission identifier.
               - {person_id_col}: Patient ID.
               - {icu_length_of_stay_col}: ICU length of stay.
               - {unit_type_col}: ICU unit type.
            2. Scan admissions CSV and rename columns:
               - {hospital_stay_id_col}: Hospital admission identifier.
               - {ethnicity_col}: Patient ethnicity.
               - {admission_loc_col}: Admission location.
               - {discharge_loc_col}: Discharge location.
               - {admission_urgency_col}: Admission urgency.
               - {mortality_hosp_col}: Hospital mortality flag.
            3. Scan patients CSV and rename columns:
               - {person_id_col}: Patient ID.
               - {gender_col}: Patient gender.
               - {age_col}: Patient age.
            4. Join ICU stays, admissions, patients, height/weight data, and specialties.
            5. Convert timestamp strings to datetime objects and cast numeric columns.
            6. Compute derived columns:
               - {pre_icu_length_of_stay_col}: Days from admit time to ICU intime.
               - {hospital_length_of_stay_col}: Days from admit time to dischtime.
               - {icu_stay_seq_num_col}: ICU stay sequence number.
               - {mortality_icu_col}: ICU mortality flag.
            7. Apply categorical conversions for {gender_col}, {ethnicity_col}, {admission_loc_col},
               {unit_type_col}, {discharge_loc_col}, {admission_type_col}, {admission_urgency_col}, {specialty_col}.
            8. Fill missing ICU mortality values.

        Returns:
            pl.LazyFrame: A lazy frame containing the following columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {hospital_stay_id_col}: Hospital admission identifier.
                - {person_id_col}: Patient ID.
                - {icu_stay_seq_num_col}: ICU stay sequence number.
                - {icu_time_rel_to_first_col}: Time relative to first ICU admission.
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
                - {mortality_after_col}: Post-discharge mortality indicator.
                - {admission_type_col}: Admission type based on specialty.
                - {admission_urgency_col}: Admission urgency.
                - {admission_time_col}: ICU admission time.
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
                    "race": (
                        self.ethnicity_col
                    ),  # "race" is the choice of the dataset creators
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
                    "anchor_age": self.age_col,
                }
            )
            .select(self.person_id_col, self.gender_col, self.age_col, "dod")
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
            icustays.join(admissions, on=self.hospital_stay_id_col, how="left")
            .join(patients, on=self.person_id_col, how="left")
            .join(
                self._extract_patient_height_weight(icustays),
                on=self.icu_stay_id_col,
                how="left",
            )
            .join(
                self._extract_specialties(), on=self.icu_stay_id_col, how="left"
            )
            .join(
                MORTALITY_AFTER_CENSOR_CUTOFF, on=self.person_id_col, how="left"
            )
            .with_columns(
                pl.col("intime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("outtime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("dod").str.to_datetime(
                    "%Y-%m-%d"
                ),  # hour and minute are not provided
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
            # Get the most recent specialty
            .filter(pl.col("transfertime") < pl.col("intime"))
            # Get the most recent specialty on ICU admission
            .group_by(self.icu_stay_id_col)
            .first()
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
        Extract patient height and weight from the chartevents CSV file, using cached data if available.

        Steps:
            1. Check for existence of a precalculated parquet file at {precalc_path} + "MIMIC4_height_weight.parquet".
            2. If available and force is False, load and return the cached data.
            3. If not available or force=True, then:
               a. Scan chartevents CSV and select columns: {icu_stay_id_col}, ITEMID, VALUENUM, CHARTTIME.
               b. Filter for ITEMIDs of interest.
               c. Join with {intime} from ICU stays.
               d. Convert date columns to datetime.
               e. Apply unit conversions: inches to cm for height, lbs to kg for weight.
               f. Pivot data so each {icu_stay_id_col} has separate columns for {weight_col} and {height_col}.
               g. Cast resulting columns to float.
               h. Save the resulting DataFrame to a parquet file.

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

        ITEMIDS = {
            762: self.weight_col,  # Admit Wt [carevue]
            763: self.weight_col,  # Daily Weight [carevue]
            3580: self.weight_col,  # Present Weight  (kg) [carevue]
            3693: self.weight_col,  # Weight Kg [carevue]
            224639: self.weight_col,  # Daily Weight [metavision]
            226512: self.weight_col,  # Admission Weight (Kg) [metavision]
            3581: "weight_lbs",  # Present Weight  (lb) [carevue]
            226531: "weight_lbs",  # Admission Weight (lbs.) [metavision]
            920: "height_inch",  # Admit Ht [carevue]
            1394: "height_inch",  # Height Inches [carevue]
            226707: "height_inch",  # Height [metavision]
            226730: self.height_col,  # Height (cm) [metavision]
        }

        KEEPIDS = [*ITEMIDS.keys()]

        height_weight = (
            pl.scan_csv(self.chartevents_path)
            .select("stay_id", "itemid", "valuenum", "charttime")
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
                pl.col("itemid").replace_strict(ITEMIDS, default=None),
            )
            .drop_nulls("itemid")
            .filter(
                (pl.col("charttime") - pl.col("intime")).le(
                    pl.duration(hours=self.ADMISSION_WEIGHT_HEIGHT_CUTOFF)
                )
            )
            .drop("intime", "charttime")
            .with_columns(
                # Convert height in in to cm, weight in lbs to kg
                pl.when(pl.col("itemid") == "height_inch")
                .then(pl.col("valuenum").mul(self.INCH_TO_CM))
                .when(pl.col("itemid") == "weight_lbs")
                .then(pl.col("valuenum").mul(self.LBS_TO_KG))
                .otherwise(pl.col("valuenum"))
                .alias("valuenum"),
                # Rename ITEMID to height_cm / weight_kg
                pl.when(pl.col("itemid") == "height_inch")
                .then(pl.lit(self.height_col))
                .when(pl.col("itemid") == "weight_lbs")
                .then(pl.lit(self.weight_col))
                .otherwise(pl.col("itemid"))
                .alias("itemid"),
            )
            .collect(streaming=True)
            .pivot(
                index=self.icu_stay_id_col,
                on="itemid",
                values="valuenum",
                aggregate_function="mean",  # NOTE: -> or mean?
            )
            .select(self.icu_stay_id_col, self.weight_col, self.height_col)
            .cast({self.weight_col: float, self.height_col: float})
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
            .filter(
                (
                    pl.col("offset")
                    < pl.duration(days=pl.col(self.icu_length_of_stay_col))
                )
                & (
                    pl.col("offset")
                    > pl.duration(days=-self.PRE_ICU_TIMESERIES_DAYS_CUTOFF)
                )
            )
            .with_columns(
                pl.col("offset")
                .dt.total_seconds()
                .cast(float)
                .alias(self.timeseries_time_col)
            )
            .drop(self.icu_length_of_stay_col)
            .cast({"valuenum": float})
        )

    # region vitals
    # Extract measurements from the chartevents.csv file
    def extract_chartevents(self) -> pl.LazyFrame:
        """
        Extract vital measurement events from chartevents CSV and align them with ICU admission.

        Merges original and additional measurement mappings and replaces values with appropriate enumerations.

        Returns:
            pl.LazyFrame: A lazy frame containing:
                - {hospital_stay_id_col}: Hospital stay identifier.
                - label: Mapped vital sign name.
                - VALUENUM: Measurement value (float).
                - {timeseries_time_col}: Time offset in seconds from ICU admission.
        """
        meas_chartevents_main_original_data = (
            pl.scan_csv(self.meas_chartevents_main_path)
            .select("itemid (omop_source_code)", "label", "omop_concept_name")
            .with_columns(
                pl.when(pl.col("label") == "Temperature Fahrenheit")
                .then(pl.lit("Temperature Fahrenheit"))
                .when(pl.col("label") == "Temperature Celsius")
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
                    }
                )
            )
            # Filter for names of interest
            .filter(
                pl.col("label").is_not_null(),
                # lab values are stored in the labevents.csv file and just
                # duplicated to chartevents.csv
                pl.col("label").is_in(
                    self.relevant_vital_values
                    + self.relevant_respiratory_values
                    + self.relevant_intakeoutput_values
                ),
            )
        )

        return (
            pl.scan_csv(
                self.chartevents_path,
                schema_overrides={"value": str, "valuenum": float},
            )
            .select("hadm_id", "itemid", "charttime", "value", "valuenum")
            # Rename columns for consistency
            .rename({"hadm_id": self.hospital_stay_id_col})
            .with_columns(
                pl.col("charttime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col(self.hospital_stay_id_col).cast(int),
            )
            .pipe(self.extract_timeseries_helper)
            .join(meas_chartevents_main_data, on="itemid", how="left")
            .with_columns(
                pl.when(pl.col("label") == "Heart rate rhythm")
                .then(
                    pl.col("value")
                    .replace_strict(self.HEART_RHYTHM_MAP, default=None)
                    .replace(self.heart_rhythm_enum_map)
                )
                .when(pl.col("label") == "Oxygen delivery system")
                .then(
                    pl.col("value")
                    .replace_strict(
                        self.OXYGEN_DELIVERY_SYSTEM_MAP, default=None
                    )
                    .replace(self.oxygen_delivery_system_enum_map)
                )
                .when(pl.col("label") == "Ventilation mode Ventilator")
                .then(
                    pl.col("value")
                    .replace_strict(self.VENTILATOR_MODE_MAP, default=None)
                    .replace(self.ventilator_mode_enum_map)
                )
                .otherwise(pl.col("valuenum"))
                .cast(float)
                .alias("valuenum"),
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

    # region lab
    # Extract lab measurements from the labevents.csv file
    def extract_lab_measurements(self) -> pl.LazyFrame:
        """
        Extract laboratory measurements from labevents CSV and structure LOINC information.

        Joins lab measurements with LOINC mapping and creates a structured column containing lab details.

        Returns:
            pl.LazyFrame: A lazy frame containing:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time offset in seconds from ICU admission.
                - label: Lab test name.
                - labstruct: A struct with:
                    • value: Lab measurement value (float).
                    • system: LOINC system.
                    • method: LOINC method.
                    • time: LOINC time aspect.
                    • LOINC: LOINC code.
        """
        # NOTE: ASSUMPTION: These are the lab values of interest
        # TODO: Confer with medical experts to confirm these are the correct values
        d_labitems_to_loinc_data = (
            pl.scan_csv(self.d_labitems_to_loinc_path)
            .select("itemid (omop_source_code)", "omop_concept_name")
            .rename(
                {
                    "itemid (omop_source_code)": "itemid",
                    "omop_concept_name": "label",
                }
            )
        )
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
                    self.omop.get_lab_system_from_name(labnames), default=None
                )
                .alias("LOINC_system"),
                pl.col("label")
                .replace_strict(
                    self.omop.get_lab_method_from_name(labnames), default=None
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

        return (
            pl.scan_csv(self.labevents_path)
            .select("hadm_id", "itemid", "charttime", "valuenum")
            # Rename columns for consistency
            .rename({"hadm_id": self.hospital_stay_id_col})
            # BUG: .drop_nulls() drops all rows with any(!) null values
            # .drop_nulls()  # NOTE: CLEARLY THINK ABOUT THIS (-> are these baselines?)
            .with_columns(
                pl.col("charttime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col(self.hospital_stay_id_col).cast(int),
            )
            .pipe(self.extract_timeseries_helper)
            .join(d_labitems_to_loinc_data, on="itemid", how="left")
            .drop("itemid")
            # Remove rows with empty lab names
            .filter(pl.col("label").is_not_null() & (pl.col("label") != ""))
            # Remove rows with empty lab results
            .filter(pl.col("valuenum").is_not_null())
            # Remove duplicate rows
            .unique()
            # Cast valuenum to float
            .cast({"valuenum": float})
            # MAKE STRUCT
            .with_columns(pl.col("LOINC_component").alias("label"))
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
                "label",
                "labstruct",
            )
        )

    # endregion

    # region output
    # Extract output measurements from the outputevents.csv file
    def extract_output_measurements(self) -> pl.LazyFrame:
        """
        Extract output (fluid balance) measurements from input and output events.

        Combines data from inputevents and outputevents CSVs, applies concept mappings, and converts timestamps.

        Returns:
            pl.LazyFrame: A lazy frame containing:
                - {hospital_stay_id_col}: Hospital stay identifier.
                - {timeseries_time_col}: Time offset in seconds from ICU admission.
                - label: Output measurement name.
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

        inputevents = (
            pl.scan_csv(
                self.inputevents_path, schema_overrides={"amount": float}
            )
            .select(
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
                "hadm_id",
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
                    "hadm_id": self.hospital_stay_id_col,
                    # "spec_type_desc": self.micro_specimen_col,
                    # "test_name": self.micro_test_col,
                    # "org_name": self.micro_organism_col,
                    # "ab_name": self.micro_antibiotic_col,
                    "interpretation": self.micro_sensitivity_col,
                }
            )
            .join(self.icu_stay_id, on=self.hospital_stay_id_col)
            .drop(self.person_id_col)
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
                < pl.duration(days=pl.col(self.icu_length_of_stay_col)),
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
        Extract medication administration data from input events CSVs.

        Aggregates data from multiple medication sources, normalizes drug names/units, and calculates
        relative start and end times.

        Returns:
            pl.LazyFrame: A lazy frame containing:
                - {icu_stay_id_col}: ICU stay identifier.
                - {drug_mixture_id_col}: Mixture identifier.
                - {drug_mixture_admin_id_col}: Mixture administration identifier.
                - {drug_name_col}: Original drug name.
                - {drug_ingredient_col}: Mapped active drug ingredient.
                - {drug_amount_col}: Drug amount.
                - {drug_amount_unit_col}: Unit of drug amount.
                - {drug_rate_col}: Drug rate.
                - {drug_rate_unit_col}: Unit for drug rate.
                - {drug_start_col}: Drug administration start time (seconds offset from ICU admission).
                - {drug_end_col}: Drug administration end time (seconds offset from ICU admission).
                - {drug_patient_weight_col}: Patient weight used in dosing.
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
        #######################################################################
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

        inputevents = (
            pl.scan_csv(
                self.inputevents_path,
                schema_overrides={
                    "amount": float,
                    "totalamount": float,
                    "patientweight": float,
                },
            )
            .select(
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
                .alias(self.drug_continous_col),
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
        #######################################################################
        # Load medication mappings from MIMIC-IV OMOP files
        # These mappings connect medication names to standard concepts and ingredients
        print("MIMIC4  - Loading medication mapping files...")

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
                .alias(self.drug_admin_route_col)
            )
            .select("route", self.drug_admin_route_col)
            .lazy()
        )

        # 2. Create NDC to RxNorm concept mappings
        # Extract unique NDC codes from prescriptions
        ndc_codes = (
            pl.scan_csv(self.prescriptions_path)
            .select("ndc")
            .unique()
            .collect()
            .to_series()
            .to_list()
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
            pl.scan_csv(
                self.prescriptions_path, schema_overrides={"dose_val_rx": str}
            )
            .select(
                "hadm_id",
                "starttime",
                "stoptime",
                "drug",
                "ndc",
                "dose_val_rx",
                "dose_unit_rx",
                "route",
            )
            .rename(
                {
                    "hadm_id": self.hospital_stay_id_col,
                    "drug": self.drug_name_col,
                    "dose_val_rx": self.drug_amount_col,
                    "dose_unit_rx": self.drug_amount_unit_col,
                }
            )
            .join(route_to_concept, on="route", how="left")
            .with_columns(
                pl.col("ndc")
                .replace_strict(ndc_to_ingredient, default=None)
                .alias(self.drug_ingredient_col),
                pl.col("ndc")
                .replace_strict(ndc_to_drugname, default=None)
                .alias(self.drug_name_OMOP_col),
                # Add a column to indicate if the drug is continuous
                pl.lit(False).alias(self.drug_continous_col),
            )
            .rename({"stoptime": "endtime"})
            .join(
                self.icu_stay_id.drop(self.person_id_col),
                on=self.hospital_stay_id_col,
                how="left",
            )
        )

        # region COMBINED
        #######################################################################
        return (
            pl.concat([inputevents, prescriptions], how="diagonal_relaxed")
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
                (
                    pl.col(self.drug_start_col)
                    < pl.duration(
                        days=pl.col(self.icu_length_of_stay_col)
                    ).truediv(pl.duration(seconds=1))
                )
                & (
                    pl.col(self.drug_start_col)
                    > pl.duration(
                        days=-self.PRE_ICU_TIMESERIES_DAYS_CUTOFF
                    ).truediv(pl.duration(seconds=1))
                )
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
        Extract diagnoses from diagnoses_icd CSV and join with ICD description details.

        Returns:
            pl.LazyFrame: A lazy frame containing:
                - {person_id_col}: Patient identifier.
                - {hospital_stay_id_col}: Hospital stay identifier.
                - {diagnosis_icd_code_col}: ICD diagnosis code.
                - {diagnosis_icd_version_col}: ICD version number.
                - {diagnosis_priority_col}: Diagnosis priority/order.
                - {diagnosis_description_col}: Detailed diagnosis description.
                - {diagnosis_discharge_col}: Flag indicating a discharge diagnosis.
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
        Extract procedures from procedure events and ICD procedure CSVs.

        Aggregates sources of procedure data including event timings and ICD codes.

        Returns:
            pl.LazyFrame: A lazy frame containing:
                - {person_id_col}: Patient identifier.
                - {hospital_stay_id_col}: Hospital stay identifier.
                - {icu_stay_id_col}: ICU stay identifier.
                - {procedure_start_col}: Procedure start time (seconds offset from ICU admission).
                - {procedure_end_col}: Procedure end time (seconds offset from ICU admission).
                - {procedure_category_col}: Procedure category.
                - {procedure_description_col}: Detailed procedure description.
                (May also include {procedure_icd_code_col}, {procedure_icd_version_col}, and {procedure_priority_col} if available.)
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
