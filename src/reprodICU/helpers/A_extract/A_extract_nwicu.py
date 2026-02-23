# Author: Finn Fassbender
# Last modified: 2025-10-30

# Description: This script extracts the data from the source files and provides the extracted data
# in a structured format for further processing and harmonization.

import os.path

import numpy as np
import polars as pl

from ..helper import GlobalHelpers
from ..helper_filepaths import NWICUPaths
from ..helper_OMOP import Vocabulary


class NWICUExtractor(NWICUPaths):
    def __init__(self, paths):
        super().__init__(paths)
        self.path = paths.nwicu_source_path
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

    # region ID mapping table
    # Extract the patient IDs that are used in the NWICU dataset
    def extract_patient_IDs(self) -> pl.LazyFrame:
        """
        Extract patient IDs and basic ICU stay information.

        Steps:
            1. Scan ICU stays CSV file.
            2. Rename columns to standardized names.
            3. Remove duplicates and cast ID columns to integer.
            4. Select required columns.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {hospital_stay_id_col}: Hospital admission identifier.
                - {person_id_col}: Patient identifier.
                - {icu_length_of_stay_col}: ICU length of stay (days).
                - intime: ICU admission timestamp.
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
        Extract and transform patient demographics and clinical information.

        Steps:
            1. Scan and rename ICU stays, admissions, and patients CSV files.
            2. Join datasets on key identifiers.
            3. Join height and weight data.
            4. Convert timestamp columns to datetime and cast numeric columns.
            5. Compute derived columns (age, mortality flags, lengths of stay).
            6. Sort by patient ID and compute ICU stay sequence number.
            7. Fill missing ICU mortality values based on hospital mortality status.

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
                - {admission_time_col}: ICU admission time of day.
                - {admission_year_col}: Admission year.
                - {admission_loc_col}: Admission location.
                - {admission_urgency_col}: Admission urgency.
                - {pre_icu_length_of_stay_col}: Pre-ICU length of stay (days).
                - {icu_length_of_stay_col}: ICU length of stay (days).
                - {hospital_length_of_stay_col}: Hospital length of stay (days).
                - {mortality_icu_col}: ICU mortality flag.
                - {mortality_hosp_col}: Hospital mortality flag.
                - {mortality_after_col}: Days between discharge and death.
                - {unit_type_col}: ICU unit type.
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
                # create dummy date of birth from anchor year and age
                pl.datetime(
                    year=pl.col("anchor_year") - pl.col("anchor_age"),
                    month=6,
                    day=1,
                ).alias("dob"),
            )
            .with_columns(
                # Calculate age in years at ICU admission
                (pl.col("intime") - pl.col("dob"))
                .dt.total_days()
                .floordiv(365.25)
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
                .replace({"M": "Male", "F": "Female", "U": "Unknown"})
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
                # NOTE: no deathtime for deaths in hospital -> use discharge time
                (
                    pl.when(pl.col(self.mortality_hosp_col).cast(bool))
                    .then(pl.col("dischtime"))
                    .otherwise(pl.col("deathtime"))
                    - pl.col("outtime")
                )
                .truediv(pl.duration(hours=1))
                .le(pl.duration(hours=self.ICU_DISCHARGE_MORTALITY_CUTOFF))
                .cast(bool)
                .fill_null(False)
                .alias(self.mortality_icu_col),
                # Calculate hospital mortality
                # NOTE: hospital_expire_flag is not reliable
                pl.when(pl.col(self.mortality_hosp_col).cast(bool))
                .then(pl.lit(True))
                .otherwise(
                    (pl.col("deathtime") - pl.col("dischtime"))
                    .truediv(pl.duration(hours=1))
                    .le(pl.duration(hours=self.ICU_DISCHARGE_MORTALITY_CUTOFF))
                )
                .cast(bool)
                .fill_null(False)
                .alias(self.mortality_hosp_col),
                # Calculate mortality after discharge
                (  # Prefer deathtime over dod if available
                    pl.when(pl.col("deathtime").is_not_null())
                    .then(pl.col("deathtime"))
                    .otherwise(pl.col("dod"))
                    - pl.col("outtime")
                )
                .truediv(pl.duration(days=1))
                .cast(int)
                .alias(self.mortality_after_col),
                # Convert categorical admission location to enum
                pl.col(self.admission_loc_col)
                .replace(self.ADMISSION_LOCATIONS_MAP)
                .cast(self.admission_locations_dtype),
                # Convert categorical unit type to enum
                pl.col(self.unit_type_col)
                .replace_strict(self.UNIT_TYPES_MAP, default=None)
                .cast(self.unit_types_dtype),
                # Convert categorical discharge location to enum
                pl.col(self.discharge_loc_col)
                .replace(self.DISCHARGE_LOCATIONS_MAP)
                .cast(self.discharge_locations_dtype),
                # # Determine Admission Type based on treating specialty
                # pl.col(self.specialty_col)
                # .replace_strict(self.ADMISSION_TYPES_MAP, default=None)
                # .cast(self.admission_types_dtype),
                # Convert categorical admission urgency to enum
                pl.col(self.admission_urgency_col)
                .replace_strict(self.ADMISSION_URGENCY_MAP, default=None)
                .cast(self.admission_urgency_dtype),
                # # Convert categorical specialty to enum
                # pl.col(self.specialty_col)
                # .replace(self.SPECIALTIES_MAP)
                # .cast(self.specialties_dtype),
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
        Extract patient height and weight from chartevents.

        Steps:
            1. Check for precomputed parquet file; load if available.
            2. Otherwise, scan chartevents CSV and filter for height/weight item IDs.
            3. Join with ICU stay intime, convert units (inches→cm, oz→kg).
            4. Select first admission weight and first 24-hour height measurements.
            5. Pivot data to separate weight and height columns.
            6. Save result as parquet file for future reuse.

        Returns:
            pl.DataFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {weight_col}: Patient weight (kg).
                - {height_col}: Patient height (cm).
        """
        # check if precalculated data is available
        if (
            os.path.isfile(self.precalc_path + "NWICU_height_weight.parquet")
            and not force
        ):
            return pl.scan_parquet(
                self.precalc_path + "NWICU_height_weight.parquet"
            )

        print("NWICU   - Extracting patient height and weight...")

        WEIGHT_ITEMID = {
            326531: "weight_oz",  # WEIGHT/SCALE (Admission Weight) [in oz]
        }
        HEIGHT_ITEMID = {
            326707: "height_in",  # HEIGHT (Height) [in inches]
        }

        KEEPIDS = [*(WEIGHT_ITEMID | HEIGHT_ITEMID).keys()]

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
                pl.col("itemid")
                .replace_strict(
                    WEIGHT_ITEMID | HEIGHT_ITEMID,  # "|" merges dictionaries
                    default=None,
                )
                .alias("label"),
            )
            .drop_nulls("itemid")
            .with_columns(
                # Convert height in in to cm, weight in oz to kg
                pl.when(pl.col("label") == "height_in")
                .then(pl.col("valuenum").mul(self.INCH_TO_CM))
                .when(pl.col("label") == "weight_oz")
                .then(pl.col("valuenum").mul(self.OZ_TO_KG))
                .otherwise(pl.col("valuenum"))
                .alias("valuenum"),
                # Rename ITEMID to height_cm / weight_kg
                pl.when(pl.col("label") == "height_in")
                .then(pl.lit(self.height_col))
                .when(pl.col("label") == "weight_oz")
                .then(pl.lit(self.weight_col))
                .otherwise(pl.col("label"))
                .alias("label"),
            )
        )

        # Backfill weights similar to:
        # https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/concepts/durations/weight_durations.sql
        # https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/demographics/weight_durations.sql
        # Select the first admission weight
        weight = (
            height_weight.filter(pl.col("itemid").is_in(WEIGHT_ITEMID.keys()))
            .collect(engine="streaming")
            .with_columns(
                pl.col("valuenum")
                .first()
                .over(partition_by=self.icu_stay_id_col, order_by="charttime")
                .alias("valuenum")
            )
        )

        # Height measurements from the first 24 hours of the ICU stay since it's unlikely to change
        height = (
            height_weight.filter(pl.col("itemid").is_in(HEIGHT_ITEMID.keys()))
            .collect(engine="streaming")
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
            self.precalc_path + "NWICU_height_weight.parquet"
        )

        return height_weight.lazy()

    # endregion

    # region TS helper
    # make available the common processing steps for the NWICU timeseries
    def extract_timeseries_helper(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """
        Align timeseries data with ICU admission time and compute time offsets.

        Steps:
            1. Join data with patient IDs (including intime).
            2. Convert intime from string to datetime.
            3. Compute time offset by subtracting intime from charttime.
            4. Filter out rows outside ICU stay window.
            5. Convert offset to total seconds.

        Returns:
            pl.LazyFrame: Contains columns:
                - {timeseries_time_col}: Time offset (seconds) from ICU admission.
                - Other original measurement columns (e.g., valuenum).
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
        Extract vital measurements from chartevents.

        Steps:
            1. Load vital names mapping.
            2. Scan chartevents CSV, select relevant columns.
            3. Map item IDs to standardized vital sign names.
            4. Align data with ICU admission time using timeseries helper.
            5. Filter for valid measurements and remove duplicates.

        Returns:
            pl.LazyFrame: Contains columns:
                - {hospital_stay_id_col}: Hospital admission identifier.
                - label: Mapped vital sign name.
                - valuenum: Measurement value (float).
                - {timeseries_time_col}: Time offset (seconds) from ICU admission.
        """
        # NOTE: ASSUMPTION: These are the lab values of interest
        # TODO: Confer with medical experts to confirm these are the correct values
        vital_names_mapping = self.helpers.load_mapping(
            self.vitals_mapping_path
        )

        return (
            pl.scan_csv(
                self.chartevents_path,
                schema_overrides={"value": str, "valuenum": float},
            )
            .select("hadm_id", "itemid", "charttime", "valuenum")
            # Rename columns for consistency
            .rename({"hadm_id": self.hospital_stay_id_col})
            .with_columns(
                pl.col("charttime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col(self.hospital_stay_id_col).cast(int),
                pl.col("itemid")
                .replace_strict(vital_names_mapping, default=None)
                .alias("label"),
            )
            .pipe(self.extract_timeseries_helper)
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
        Extract laboratory measurements and map to LOINC concepts.

        Steps:
            1. Load LOINC mapping from d_labitems_to_loinc.
            2. Scan labevents CSV, select relevant columns.
            3. Align data with ICU admission time using timeseries helper.
            4. Join with LOINC mappings.
            5. Filter for relevant lab components and systems.
            6. Create struct column with LOINC details.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time offset (seconds) from ICU admission.
                - label: Lab test name.
                - labstruct: Struct with value, system, method, time, LOINC code.
        """
        d_labitems_to_loinc_data = (
            pl.scan_csv(self.d_labitems_to_loinc_path)
            .select("itemid", "mapped_concept_name")
            .rename({"mapped_concept_name": "label"})
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
            # Remove rows with bad lab results
            # either less than values, or string values
            # -> TODO: handle these cases
            .filter(pl.col("valuenum").ne_missing(9999999.0))
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

    # region medications
    # Extract medications from the prescriptions.csv and emar.csv file
    def extract_medications(self) -> pl.LazyFrame:
        """
        Extract medication administration data from prescriptions and EMAR.

        Steps:
            1. Load medication ingredient mapping.
            2. Read prescriptions CSV, normalize units and rates.
            3. Map drug names to standardized ingredients.
            4. Read EMAR CSV and map event types to administration types.
            5. Join EMAR with prescriptions to get ingredient info.
            6. Combine prescriptions and EMAR data.
            7. Compute relative start and end times from ICU admission.
            8. Filter for events within ICU stay window.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {drug_name_col}: Medication name.
                - {drug_ingredient_col}: Active drug ingredient.
                - {drug_amount_col}: Drug amount.
                - {drug_amount_unit_col}: Amount unit.
                - {drug_rate_col}: Administration rate.
                - {drug_rate_unit_col}: Rate unit.
                - {drug_start_col}: Relative start time (seconds).
                - {drug_end_col}: Relative end time (seconds).
                - {drug_admin_type_col}: Administration type.
        """
        print("NWICU   - Extracting medications...")

        intimes = self.extract_patient_IDs().select(
            self.icu_stay_id_col, "intime", self.icu_length_of_stay_col
        )

        NWICU_medication_mapping = (
            self.helpers.load_many_to_many_to_one_mapping(
                self.mapping_path + "MEDICATIONS.yaml", "nwicu"
            )
        )
        nwicu_drug_administration_route_mapping = self.helpers.load_mapping(
            self.drug_administration_route_mapping_path
        )

        # region PRESCRIPTIONS
        ########################################################################
        prescriptions = (
            pl.scan_csv(self.prescriptions_path, infer_schema_length=10000)
            .select(
                "hadm_id",
                "pharmacy_id",
                "starttime",
                "stoptime",
                "drug",
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
            .with_columns(
                pl.lit("prescribed")
                .cast(self.drug_admin_type_dtype)
                .alias(self.drug_admin_type_col),
                pl.col("route")
                .replace(nwicu_drug_administration_route_mapping, default=None)
                .alias(self.drug_admin_route_col),
                # Rename units
                pl.col(self.drug_amount_unit_col)
                .str.replace("grams", "g")
                .str.replace("hour", "hr")
                .str.replace("mL", "ml")
                .str.replace("mEq\.", "mEq")
                .str.replace("units", "U")
                .str.replace("µ", "mc"),
                # Mark rows with rates, not amounts
                pl.col(self.drug_amount_unit_col)
                .str.contains_any(["min", "hr", "day"])
                .alias("is_rate"),
                # Calculate total doses that should have been given in period
                # -> how often was the threshold of next administration crossed within the start and stop time?
                pl.col(self.drug_amount_col)
                * (
                    pl.col("stoptime").str.to_datetime("%Y-%m-%d %H:%M:%S")
                    - pl.col("starttime").str.to_datetime("%Y-%m-%d %H:%M:%S")
                ).dt.total_hours()
                // (24 / pl.col("doses_per_24_hrs")),
            )
            .with_columns(
                # select rates
                pl.when(pl.col("is_rate"))
                .then(pl.col(self.drug_amount_col))
                .alias(self.drug_rate_col),
                pl.when(pl.col("is_rate"))
                .then(pl.col(self.drug_amount_unit_col))
                .alias(self.drug_rate_unit_col),
                # select amounts
                pl.when(pl.col("is_rate"))
                .then(None)
                .otherwise(pl.col(self.drug_amount_col))
                .alias(self.drug_amount_col),
                pl.when(pl.col("is_rate"))
                .then(None)
                .otherwise(pl.col(self.drug_amount_unit_col))
                .alias(self.drug_amount_unit_col),
            )
            .drop("is_rate", "route")
            # Replace drug names with mapped names
            .with_columns(
                pl.col(self.drug_name_col)
                .replace_strict(NWICU_medication_mapping, default=None)
                .alias(self.drug_ingredient_col),
            )
            .join(
                self.icu_stay_id.drop(self.person_id_col),
                on=self.hospital_stay_id_col,
                how="right",
            )
        )

        # region EMAR
        ########################################################################
        emar = (
            pl.scan_csv(self.emar_path)
            .select(
                "hadm_id",
                "emar_id",
                "pharmacy_id",
                "charttime",
                "medication",
                "event_txt",
            )
            .rename(
                {
                    "hadm_id": self.hospital_stay_id_col,
                    "medication": self.drug_name_col,
                    "pharmacy_id": self.drug_prescription_id_col,
                }
            )
            .with_columns(
                # Map event types to administration types
                pl.col("event_txt")
                .replace(
                    {
                        "Applied": "given",
                        "Not Given": "not given",
                        "Confirmed": "confirmed",
                    }
                )
                .alias(self.drug_admin_type_col)
            )
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
                .alias("stoptime"),
            )
        )

        # region COMBINED
        ########################################################################
        return (
            pl.concat([prescriptions, emar], how="diagonal_relaxed")
            .join(intimes, on=self.icu_stay_id_col)
            # Change times to relative times
            .with_columns(
                pl.col("intime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("starttime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("stoptime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
            .with_columns(
                (pl.col("starttime") - pl.col("intime"))
                .dt.total_seconds()
                .alias(self.drug_start_col),
                (pl.col("stoptime") - pl.col("intime"))
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
                "stoptime",
                "intime",
                self.icu_length_of_stay_col,
            )
        )

    # endregion

    # region diagnoses
    # Extract diagnoses from the diagnoses_icd.csv file
    def extract_diagnoses(self) -> pl.LazyFrame:
        """
        Extract diagnosis codes from ICD diagnoses files.

        Steps:
            1. Read diagnoses and d_icd_diagnoses CSV files.
            2. Filter for ICU patients only.
            3. Join diagnoses with ICD descriptions.
            4. Rename columns to standardized names.
            5. Mark all as discharge diagnoses (billing purpose).
            6. Remove duplicates and null codes.

        Returns:
            pl.LazyFrame: Contains columns:
                - {person_id_col}: Patient identifier.
                - {hospital_stay_id_col}: Hospital admission identifier.
                - {diagnosis_icd_code_col}: ICD diagnosis code.
                - {diagnosis_icd_version_col}: ICD version (9 or 10).
                - {diagnosis_priority_col}: Diagnosis priority (1-indexed).
                - {diagnosis_description_col}: Diagnosis description.
                - {diagnosis_discharge_col}: Discharge diagnosis flag (always True).
        """
        print("NWICU   - Extracting diagnoses...")
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
                    self.icu_stay_id.select(self.hospital_stay_id_col)
                    .collect()
                    .to_series()
                )
            )
            .with_columns(
                pl.col(self.hospital_stay_id_col).cast(int),
                # NOTE: all diagnoses in NWICU are discharge diagnoses for billing purposes
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
            .with_columns(
                pl.col(self.diagnosis_priority_col) + 1  # Priority is 1-indexed
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
        Extract procedure events from procedureevents CSV.

        Steps:
            1. Read procedureevents CSV and rename columns.
            2. Join with ICU stay intime for time alignment.
            3. Join with item descriptions from d_items.
            4. Convert timestamp columns to datetime.
            5. Compute relative start and end times from ICU admission.

        Returns:
            pl.LazyFrame: Contains columns:
                - {person_id_col}: Patient identifier.
                - {hospital_stay_id_col}: Hospital admission identifier.
                - {icu_stay_id_col}: ICU stay identifier.
                - {procedure_start_col}: Procedure start time (seconds) from admission.
                - {procedure_end_col}: Procedure end time (seconds) from admission.
                - {procedure_category_col}: Procedure category.
                - {procedure_description_col}: Procedure description.
        """
        print("NWICU   - Extracting procedures...")

        intimes = self.extract_patient_IDs().select(
            self.icu_stay_id_col, "intime"
        )

        d_items = pl.scan_csv(self.d_items_path).select("itemid", "label")

        return (
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
            .join(d_items, on="itemid")
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

    # endregion
