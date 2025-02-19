# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script extracts the data from the source files and provides the extracted data
# in a structured format for further processing and harmonization.

import os.path

import polars as pl
from helpers.helper import GlobalHelpers
from helpers.helper_filepaths import NWICUPaths
from helpers.helper_OMOP import Vocabulary


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

        self.other_lab_values = [
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
    # Extract the patient IDs that are used in the NWICU dataset
    def extract_patient_IDs(self) -> pl.LazyFrame:
        """
        Extract patient IDs from the NWICU dataset.

        Reads the ICU stays CSV file and returns a LazyFrame with the following columns:
          - {icu_stay_id_col}: Unique ICU stay identifier.
          - {hospital_stay_id_col}: Hospital stay identifier.
          - {person_id_col}: Patient identifier.
          - {icu_length_of_stay_col}: Length of ICU stay.
          - intime: ICU admission time.

        Returns:
            pl.LazyFrame: Extracted patient IDs with specified columns.
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
        Extract comprehensive patient information.

        Joins ICU stays, admissions, and patient demographics, converts timestamps and ensures proper casting.
        The final DataFrame includes the following columns:
          - {icu_stay_id_col}
          - {hospital_stay_id_col}
          - {person_id_col}
          - {icu_stay_seq_num_col}: ICU stay sequence number.
          - {gender_col}: Patient gender (as enum).
          - {age_col}: Age at admission.
          - {height_col} and {weight_col}: Patient height and weight.
          - {ethnicity_col}: Ethnicity (as enum).
          - {pre_icu_length_of_stay_col}: Pre-ICU length of stay.
          - {icu_length_of_stay_col}: ICU length of stay.
          - {hospital_length_of_stay_col}: Hospital length of stay.
          - {mortality_hosp_col}: Hospital mortality flag.
          - {mortality_icu_col}: ICU mortality flag.
          - {mortality_after_col}: Post-discharge mortality.
          - {admission_urgency_col}: Admission urgency.
          - {admission_time_col}: Admission time (extracted from ICU intime).
          - {admission_loc_col}: Admission location.
          - {unit_type_col}: ICU unit type.
          - {discharge_loc_col}: Discharge location.

        Returns:
            pl.LazyFrame: Patient information with detailed demographics and stay details.
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
                    "race": self.ethnicity_col,  # "race" is the choice of the dataset creators
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

        return (
            icustays.join(admissions, on=self.hospital_stay_id_col, how="left")
            .join(patients, on=self.person_id_col, how="left")
            .join(
                self._extract_patient_height_weight(icustays),
                on=self.icu_stay_id_col,
                how="left",
            )
            # .join(
            #     self._extract_specialties(), on=self.icu_stay_id_col, how="left"
            # )
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
            )
            .with_columns(
                # Convert categorical gender to enum
                pl.col(self.gender_col)
                .replace({"M": "Male", "F": "Female", "U": "Unknown"})
                .cast(self.gender_dtype),
                # Convert categorical ethnicity to enum
                pl.col(self.ethnicity_col)
                .replace(self.ETHNICITY_MAP)
                .cast(self.ethnicity_dtype),
                # Calculate pre ICU length of stay
                (
                    (pl.col("intime") - pl.col("admittime")).truediv(
                        pl.duration(days=1)
                    )
                )
                .cast(float)
                .alias(self.pre_icu_length_of_stay_col),
                # Calculate hospital length of stay
                (
                    (pl.col("dischtime") - pl.col("admittime")).truediv(
                        pl.duration(days=1)
                    )
                )
                .cast(float)
                .alias(self.hospital_length_of_stay_col),
                # Calculate admission time
                pl.col("intime").dt.time().alias(self.admission_time_col),
                # Calculate ICU mortality
                (
                    (pl.col("deathtime") - pl.col("outtime")).truediv(
                        pl.duration(hours=1)
                    )
                )
                .le(pl.duration(hours=self.ICU_DISCHARGE_MORTALITY_CUTOFF))
                .cast(bool)
                # .fill_null(False)
                .alias(self.mortality_icu_col),
                # Calculate hospital mortality
                pl.col(self.mortality_hosp_col).cast(bool),
                # Calculate mortality after discharge
                (
                    (pl.col("dod") - pl.col("outtime")).truediv(
                        pl.duration(days=1)
                    )
                )
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
                )
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
            .select(
                self.icu_stay_id_col,
                self.hospital_stay_id_col,
                self.person_id_col,
                self.icu_stay_seq_num_col,
                self.gender_col,
                self.age_col,
                self.height_col,
                self.weight_col,
                self.ethnicity_col,
                self.pre_icu_length_of_stay_col,
                self.icu_length_of_stay_col,
                self.hospital_length_of_stay_col,
                self.mortality_hosp_col,
                self.mortality_icu_col,
                self.mortality_after_col,
                # self.admission_type_col,
                self.admission_urgency_col,
                self.admission_time_col,
                self.admission_loc_col,
                # self.specialty_col,
                self.unit_type_col,
                self.discharge_loc_col,
            )
        )

    # endregion

    # # region specialties
    # # Extract specialties from the services.csv file
    # def _extract_specialties(self) -> pl.LazyFrame:
    #     IDs = self.extract_patient_IDs().select(
    #         self.hospital_stay_id_col, self.icu_stay_id_col, "intime"
    #     )

    #     services = pl.scan_csv(self.services_path).rename(
    #         {
    #             "hadm_id": self.hospital_stay_id_col,
    #             "curr_service": self.specialty_col,
    #         }
    #     )

    #     return (
    #         services.select(
    #             [self.hospital_stay_id_col, "transfertime", self.specialty_col]
    #         )
    #         .join(IDs, on=self.hospital_stay_id_col)
    #         # Get the most recent specialty
    #         .filter(pl.col("transfertime") < pl.col("intime"))
    #         # Get the most recent specialty on ICU admission
    #         .group_by(self.icu_stay_id_col)
    #         .first()
    #         .select(self.icu_stay_id_col, self.specialty_col)
    #     )

    # # endregion

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
        Extract patient height and weight.

        Retrieves data from the chartevents.csv file (or a cached parquet file if available) and computes:
          - {height_col}: Patient height in centimeters.
          - {weight_col}: Patient weight in kilograms.
          - {icu_stay_id_col}: ICU stay identifier.

        Args:
            icustays (pl.LazyFrame): ICU stays data.
            force (bool, optional): If True, force recalculation even if cached data exists. Defaults to False.

        Returns:
            pl.DataFrame: DataFrame with columns {icu_stay_id_col}, {weight_col}, and {height_col}.
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

        ITEMIDS = {
            326531: "weight_oz",  # WEIGHT/SCALE (Admission Weight) [in oz]
            326707: "height_in",  # HEIGHT (Height) [in inches]
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
                # Convert height in in to cm, weight in oz to kg
                pl.when(pl.col("itemid") == "height_in")
                .then(pl.col("valuenum").mul(self.INCH_TO_CM))
                .when(pl.col("itemid") == "weight_oz")
                .then(pl.col("valuenum").mul(self.OZ_TO_KG))
                .otherwise(pl.col("valuenum"))
                .alias("valuenum"),
                # Rename ITEMID to height_cm / weight_kg
                pl.when(pl.col("itemid") == "height_in")
                .then(pl.lit(self.height_col))
                .when(pl.col("itemid") == "weight_oz")
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
            self.precalc_path + "NWICU_height_weight.parquet"
        )

        return height_weight.lazy()

    # endregion

    # region TS helper
    # make available the common processing steps for the NWICU timeseries
    def extract_timeseries_helper(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """
        Process timeseries data by aligning with ICU admission time.

        Joins the given data with patient IDs, computes the time offset relative to ICU admission,
        and filters based on ICU length of stay and pre-ICU cutoff. Adds the column:
          - {timeseries_time_col}: Offset in seconds from ICU admission time.

        Args:
            data (pl.LazyFrame): Raw timeseries data.

        Returns:
            pl.LazyFrame: Processed timeseries data containing the {timeseries_time_col} and other relevant fields.
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
            .drop_nulls("valuenum")
        )

    # region vitals
    # Extract measurements from the chartevents.csv file
    def extract_chartevents(self) -> pl.LazyFrame:
        """
        Extract vital measurements from chartevents.csv.

        Processes vital sign data and returns a LazyFrame with:
          - {hospital_stay_id_col}: Hospital stay identifier.
          - label: Mapped vital name.
          - valuenum: Measurement value.
          - {timeseries_time_col}: Time offset in seconds relative to ICU admission.

        Returns:
            pl.LazyFrame: Vital measurements with detailed column mapping.
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
        Extract lab measurements from labevents.csv.

        Processes lab data by joining with LOINC mappings and returns a LazyFrame with:
          - {icu_stay_id_col}: ICU stay identifier.
          - {timeseries_time_col}: Time offset in seconds.
          - label: Lab test name.
          - labstruct: A structured column containing:
              • value: Lab value.
              • system: LOINC system.
              • method: LOINC method.
              • time: LOINC time aspect.
              • LOINC: LOINC code.

        Returns:
            pl.LazyFrame: Lab measurement data with the specified structured columns.
        """
        # NOTE: ASSUMPTION: These are the lab values of interest
        # TODO: Confer with medical experts to confirm these are the correct values
        d_labitems_to_loinc_data = (
            pl.scan_csv(self.d_labitems_to_loinc_path)
            .select("itemid", "mapped_concept_name")
            .rename({"mapped_concept_name": "label"})
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

    # # region output
    # # Extract output measurements from the outputevents.csv file
    # def extract_output_measurements(self) -> pl.LazyFrame:
    #     # NOTE: ASSUMPTION: These are the lab values of interest
    #     # TODO: Confer with medical experts to confirm these are the correct values
    #     outputevents_to_loinc_data = (
    #         pl.scan_csv(self.outputevents_to_loinc_path)
    #         .select("itemid (omop_source_code)", "omop_concept_name")
    #         .rename(
    #             {
    #                 "itemid (omop_source_code)": "itemid",
    #                 "omop_concept_name": "label",
    #             }
    #         )
    #         # Harmonize names of interest
    #         .with_columns(
    #             pl.col("label").replace_strict(
    #                 self.timeseries_intakeoutput_mapping, default=None
    #             )
    #         )
    #         # Filter for names of interest
    #         .filter(pl.col("label").is_in(self.relevant_intakeoutput_values))
    #     )

    #     return (
    #         pl.scan_csv(self.outputevents_path, infer_schema_length=100000)
    #         .select("hadm_id", "itemid", "charttime", "value")
    #         # Rename columns for consistency
    #         .rename({"hadm_id": self.hospital_stay_id_col, "value": "valuenum"})
    #         # BUG: .drop_nulls() drops all rows with any(!) null values
    #         # .drop_nulls()  # NOTE: CLEARLY THINK ABOUT THIS (-> are these baselines?)
    #         .with_columns(
    #             pl.col("charttime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
    #             pl.col(self.hospital_stay_id_col).cast(int),
    #         )
    #         .pipe(self.extract_timeseries_helper)
    #         .join(outputevents_to_loinc_data, on="itemid", how="left")
    #         .drop("itemid")
    #         # Remove rows with empty names
    #         .filter(pl.col("label").is_not_null() & (pl.col("label") != ""))
    #         # Remove rows with empty values
    #         .filter(pl.col("valuenum").is_not_null())
    #         # Remove duplicate rows
    #         .unique()
    #     )

    # # endregion

    # region medications
    # Extract medications from the prescriptions.csv and emar.csv file
    # TODO: check with NWICU documentation about the medication data
    def extract_medications(self) -> pl.LazyFrame:
        """
        Extract medication administration data.

        Retrieves data from prescriptions.csv (and emar.csv if applicable), normalizes drug names, units, and times.
        The resulting LazyFrame contains:
          - {hospital_stay_id_col}: Hospital stay identifier.
          - {drug_name_col}: Original drug name mapped to {drug_ingredient_col}.
          - {drug_amount_col} and {drug_amount_unit_col}: Drug amount and unit (if not a rate).
          - {drug_rate_col} and {drug_rate_unit_col}: Drug rate and rate unit (if applicable).
          - {drug_start_col} and {drug_end_col}: Relative start and end times (in seconds) from ICU admission.

        Returns:
            pl.LazyFrame: Medication administration data with timing and dosing details.
        """
        print("NWICU   - Extracting medications...")

        intimes = self.extract_patient_IDs().select(
            self.hospital_stay_id_col,
            self.icu_stay_id_col,
            "intime",
            self.icu_length_of_stay_col,
        )
        NWICU_medication_mapping = (
            self.helpers.load_many_to_many_to_one_mapping(
                self.mapping_path + "MEDICATIONS.yaml", "nwicu"
            )
        )
        nwicu_drug_administration_route_mapping = self.helpers.load_mapping(
            self.drug_administration_route_mapping_path
        )

        prescriptions = (
            pl.scan_csv(self.prescriptions_path)
            .select(
                "hadm_id",
                # "stay_id",
                "starttime",
                "stoptime",
                "drug",
                "dose_val_rx",
                "dose_unit_rx",
                "route",
            )
            .rename(
                {
                    "hadm_id": self.hospital_stay_id_col,
                    # "stay_id": self.icu_stay_id_col,
                    "drug": self.drug_name_col,
                    "dose_val_rx": self.drug_amount_col,
                    "dose_unit_rx": self.drug_amount_unit_col,
                }
            )
            .with_columns(
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
            .drop("is_rate")
        )

        return (
            prescriptions.join(
                intimes, on=self.hospital_stay_id_col, how="left"
            )
            # Replace drug names with mapped names
            .with_columns(
                pl.col(self.drug_name_col)
                .replace_strict(NWICU_medication_mapping, default=None)
                .alias(self.drug_ingredient_col),
            )
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
                "starttime", "stoptime", "intime", self.icu_length_of_stay_col
            )
        )

    # endregion

    # region diagnoses
    # Extract diagnoses from the diagnoses_icd.csv file
    def extract_diagnoses(self) -> pl.LazyFrame:
        """
        Extract diagnoses from diagnoses_icd.csv.

        Retrieves diagnosis data and joins with detailed ICD diagnosis descriptions.
        The final LazyFrame includes:
          - {person_id_col}: Patient identifier.
          - {hospital_stay_id_col}: Hospital stay identifier.
          - {diagnosis_icd_code_col}: Diagnosis ICD code.
          - {diagnosis_icd_version_col}: ICD version.
          - {diagnosis_priority_col}: Diagnosis order (priority).
          - {diagnosis_description_col}: Long diagnosis description.
          - 'diagnosis_discharge_col': Flag indicating a discharge diagnosis.

        Returns:
            pl.LazyFrame: Diagnosis data with complete ICD and priority details.
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
        Extract procedures from procedureevents.csv and procedures_icd.csv.

        Processes procedure events by aligning with ICU admission time and joins additional item details.
        The output LazyFrame comprises:
          - {person_id_col}: Patient identifier.
          - {hospital_stay_id_col}: Hospital stay identifier.
          - {icu_stay_id_col}: ICU stay identifier.
          - {procedure_start_col}: Procedure start time offset in seconds.
          - {procedure_end_col}: Procedure end time offset in seconds.
          - {procedure_category_col}: Procedure category.
          - {procedure_description_col}: Detailed procedure description.

        Returns:
            pl.LazyFrame: Procedure events data with timing and descriptive details.
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
