# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script extracts the data from the source files and provides the extracted data
# in a structured format for further processing and harmonization.

import numpy as np
import pandas as pd
import polars as pl
import os.path

from helpers.helper_filepaths import MIMIC3Paths
from helpers.helper import GlobalHelpers


class MIMIC3Extractor(MIMIC3Paths):
    def __init__(self, paths, DEMO=False):
        super().__init__(paths, DEMO)
        self.path = paths.mimic3_source_path
        self.helpers = GlobalHelpers()
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
            "Iron [Mass/volume]",
            "Iron binding capacity [Mass/volume]",
            "Magnesium [Mass/volume]",
            "Phosphate [Mass/volume]",
            "Triiodothyronine (T3) [Mass/volume]",
            "Thyroxine (T4) [Mass/volume]",
            "Thyroxine (T4) free [Mass/volume]",
            "Cobalamin (Vitamin B12) [Mass/volume]",
            # "Basophils [#/volume]",
            "Eosinophils [#/volume]",
            "Lymphocytes [#/volume]",
            # "Monocytes [#/volume]",
            # "Neutrophils [#/volume]",
            "Reticulocytes [#/volume]",
        ]

    # region IDs
    # Extract the patient IDs that are used in the MIMIC-III dataset
    def extract_patient_IDs(self) -> pl.LazyFrame:
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
                    (pl.col("INTIME") - pl.col("DOB")).truediv(
                        pl.duration(days=self.DAYS_IN_YEAR)
                    )
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
                (
                    (pl.col("INTIME") - pl.col("ADMITTIME")).truediv(
                        pl.duration(days=1)
                    )
                )
                .cast(float)
                .alias(self.pre_icu_length_of_stay_col),
                # Calculate hospital length of stay
                (
                    (pl.col("DISCHTIME") - pl.col("ADMITTIME")).truediv(
                        pl.duration(days=1)
                    )
                )
                .cast(float)
                .alias(self.hospital_length_of_stay_col),
                # Calculate admission time
                pl.col("INTIME").dt.time().alias(self.admission_time_col),
                # Calculate ICU mortality
                (
                    (pl.col("DEATHTIME") - pl.col("OUTTIME")).truediv(
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
                    (pl.col("DOD") - pl.col("OUTTIME")).truediv(
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
            .sort(self.person_id_col, "INTIME")
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
                self.admission_type_col,
                self.admission_urgency_col,
                self.admission_time_col,
                self.admission_loc_col,
                self.specialty_col,
                self.unit_type_col,
                self.care_site_col,
                self.discharge_loc_col,
            )
        )

    # endregion

    # region specialties
    # Extract specialties from the services.csv file
    def _extract_specialties(self) -> pl.LazyFrame:
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
            # Get the most recent specialty
            .filter(pl.col("TRANSFERTIME") < pl.col("INTIME"))
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
        # check if precalculated data is available
        if (
            os.path.isfile(self.precalc_path + "MIMIC3_height_weight.parquet")
            and not force
        ):
            return pl.scan_parquet(
                self.precalc_path + "MIMIC3_height_weight.parquet"
            )

        print("MIMIC3  - Extracting patient height and weight...")

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
            .select("ICUSTAY_ID", "ITEMID", "VALUENUM", "CHARTTIME")
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
                pl.col("ITEMID").replace_strict(ITEMIDS, default=None),
            )
            .filter(
                (pl.col("CHARTTIME") - pl.col("INTIME")).le(
                    pl.duration(hours=self.ADMISSION_WEIGHT_HEIGHT_CUTOFF)
                )
            )
            .drop("INTIME", "CHARTTIME")
            .with_columns(
                # Convert height in in to cm, weight in lbs to kg
                pl.when(pl.col("ITEMID") == "height_inch")
                .then(pl.col("VALUENUM").mul(self.INCH_TO_CM))
                .when(pl.col("ITEMID") == "weight_lbs")
                .then(pl.col("VALUENUM").mul(self.LBS_TO_KG))
                .otherwise(pl.col("VALUENUM"))
                .alias("VALUENUM"),
                # Rename ITEMID to height_cm / weight_kg
                pl.when(pl.col("ITEMID") == "height_inch")
                .then(pl.lit(self.height_col))
                .when(pl.col("ITEMID") == "weight_lbs")
                .then(pl.lit(self.weight_col))
                .otherwise(pl.col("ITEMID"))
                .alias("ITEMID"),
            )
            .collect(streaming=True)
            .pivot(
                index=self.icu_stay_id_col,
                on="ITEMID",
                values="VALUENUM",
                aggregate_function="mean",  # NOTE: -> or mean?
            )
            .select(self.icu_stay_id_col, self.weight_col, self.height_col)
            .cast({self.weight_col: float, self.height_col: float})
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
                    < pl.duration(days=pl.col(self.icu_length_of_stay_col))
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
            .cast({"VALUENUM": float})
            .drop_nulls("VALUENUM")
        )

    # region vitals TS
    # Extract measurements from the chartevents.csv file
    def extract_chartevents(self) -> pl.LazyFrame:
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
                        **self.relevant_vital_values_mapping,
                        **self.relevant_lab_values_mapping,
                        **self.relevant_intakeoutput_values_mapping,
                        **self.relevant_respiratory_values_mapping,
                    }
                )
            )
            # Filter for names of interest
            .filter(
                pl.col("LABEL").is_not_null(),
                # lab values are stored in the labevents.csv file and just
                # duplicated to chartevents.csv
                pl.col("LABEL").is_in(
                    self.relevant_vital_values
                    + self.relevant_respiratory_values
                    + self.relevant_intakeoutput_values
                ),
            )
        )

        return (
            pl.scan_csv(
                self.chartevents_path,
                schema_overrides={"VALUE": str, "VALUENUM": float},
            )
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
                .when(pl.col("LABEL") == "Ventilation mode Ventilator")
                .then(
                    pl.col("VALUE")
                    .replace_strict(self.VENTILATOR_MODE_MAP, default=None)
                    .replace(self.ventilator_mode_enum_map)
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

    # region helpers
    # Print the number of unique cases in the timeseries data
    def _print_unique_cases(
        self, data: pl.LazyFrame, name: str
    ) -> pl.LazyFrame:
        unique_count = (
            data.select(self.icu_stay_id_col)
            .unique()
            .count()
            .collect(streaming=True)
            .to_numpy()[0][0]
        )
        print(
            f"reprodICU - {unique_count:6.0f} unique cases with timeseries data in {name}."
        )

        return data

    # region lab TS
    # Extract lab measurements from the labevents.csv file
    def extract_lab_measurements(self) -> pl.LazyFrame:
        # NOTE: ASSUMPTION: These are the lab values of interest
        # TODO: Confer with medical experts to confirm these are the correct values
        d_labitems_to_loinc_data = (
            pl.scan_csv(self.d_labitems_to_loinc_path)
            .select("ITEMID", "COALESCED_CONCEPT_NAME")
            .rename({"COALESCED_CONCEPT_NAME": "LABEL"})
            # Filter for lab names of interest
            .filter(
                pl.col("LABEL")
                .str.replace("in HDL", "inHDL")
                .str.replace("in LDL", "inLDL")
                .str.replace(" (in|of) ", " INOF ")
                .str.split_exact(by=" INOF ", n=1)
                .struct.rename_fields(["variable", "_"])
                .struct.field("variable")
                .is_in(self.relevant_lab_values + self.other_lab_values)
            )
        )

        return (
            pl.scan_csv(self.labevents_path)
            .select("HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM")
            # Rename columns for consistency
            .rename({"HADM_ID": self.hospital_stay_id_col})
            # BUG: .drop_nulls() drops all rows with any(!) null values
            # .drop_nulls()  # NOTE: CLEARLY THINK ABOUT THIS (-> are these baselines?)
            .with_columns(
                pl.col("CHARTTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col(self.hospital_stay_id_col).cast(int),
            )
            .pipe(self.extract_timeseries_helper)
            .join(d_labitems_to_loinc_data, on="ITEMID", how="left")
            .drop("ITEMID")
            # Remove rows with empty lab names
            .filter(pl.col("LABEL").is_not_null() & (pl.col("LABEL") != ""))
            # Remove rows with empty lab results
            .filter(pl.col("VALUENUM").is_not_null())
            # Remove duplicate rows
            .unique()
            # Cast valuenum to float
            .cast({"VALUENUM": float})
            # MAKE STRUCT
            .with_columns(
                pl.col("LABEL")
                .str.split_exact(by=" by ", n=1)
                .struct.rename_fields(["variable_source", "method"])
                .alias("fields1")
            )
            .unnest("fields1")
            .with_columns(
                pl.col("variable_source")
                .str.replace("in HDL", "inHDL")
                .str.replace("in LDL", "inLDL")
                .str.replace(" (in|of) ", " INOF ")
                .str.split_exact(by=" INOF ", n=1)
                .struct.rename_fields(["variable", "source"])
                .alias("fields2")
            )
            .unnest("fields2")
            .select(
                self.icu_stay_id_col,
                self.timeseries_time_col,
                pl.col("variable")
                .str.replace("inHDL", "in HDL")
                .str.replace("inLDL", "in LDL")
                .alias("LABEL"),
                pl.struct(
                    value="VALUENUM", source="source", method="method"
                ).alias("value_struct"),
            )
        )

    # endregion

    # region output TS
    # Extract output measurements from the outputevents.csv file
    def extract_output_measurements(self) -> pl.LazyFrame:
        # NOTE: ASSUMPTION: These are the lab values of interest
        # TODO: Confer with medical experts to confirm these are the correct values
        outputevents_to_loinc_data = (
            pl.scan_csv(self.outputevents_to_loinc_path)
            .select("itemid (omop_source_code)", "omop_concept_name")
            .rename(
                {
                    "itemid (omop_source_code)": "ITEMID",
                    "omop_concept_name": "LABEL",
                }
            )
            .cast({"ITEMID": str})
            # Harmonize names of interest
            .with_columns(
                pl.col("LABEL").replace_strict(
                    self.relevant_intakeoutput_values_mapping, default=None
                )
            )
            # Filter for names of interest
            .filter(pl.col("LABEL").is_in(self.relevant_intakeoutput_values))
        )
        input_mappings = self.helpers.load_mapping(self.inputs_mapping_path)

        inputevents_cv = (
            pl.scan_csv(
                self.inputevents_cv_path, schema_overrides={"AMOUNT": float}
            )
            .select(
                "HADM_ID", "CHARTTIME", "AMOUNT", "AMOUNTUOM", "ORIGINALROUTE"
            )
            # Rename columns for consistency
            .rename(
                {
                    "HADM_ID": self.hospital_stay_id_col,
                    "AMOUNT": "VALUENUM",
                    "ORIGINALROUTE": "ITEMID",
                }
            )
            .filter(pl.col("AMOUNTUOM").is_in(["ml", "cc"]))
            .drop("AMOUNTUOM")
        )
        inputevents_mv = (
            pl.scan_csv(
                self.inputevents_mv_path, schema_overrides={"AMOUNT": float}
            )
            .select(
                "HADM_ID",
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
                    "ORDERCATEGORYNAME": "ITEMID",
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
            # BUG: .drop_nulls() drops all rows with any(!) null values
            # .drop_nulls()  # NOTE: CLEARLY THINK ABOUT THIS (-> are these baselines?)
            .with_columns(
                pl.col("CHARTTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col(self.hospital_stay_id_col).cast(int),
            )
            .pipe(self.extract_timeseries_helper)
            .join(outputevents_to_loinc_data, on="ITEMID", how="left")
            .with_columns(
                pl.when(pl.col("LABEL").is_null())
                .then(
                    pl.col("ITEMID").replace_strict(
                        input_mappings, default=None
                    )
                )
                .otherwise(pl.col("LABEL"))
                .alias("LABEL")
            )
            .drop("ITEMID")
            # Remove rows with empty names
            .filter(pl.col("LABEL").is_not_null() & (pl.col("LABEL") != ""))
            # Remove rows with empty values
            .filter(pl.col("VALUENUM").is_not_null())
            # Remove duplicate rows
            .unique()
        )

    # endregion

    # region medications
    # Extract medications from the inputevents.csv file
    def extract_medications(self) -> pl.LazyFrame:
        print("MIMIC3  - Extracting medications...")

        intimes = self.extract_patient_IDs().select(
            self.icu_stay_id_col, self.icu_length_of_stay_col, "INTIME"
        )
        mimic3_medication_mapping = (
            self.helpers.load_many_to_many_to_one_mapping(
                self.mapping_path + "MEDICATIONS.yaml", "mimic3"
            )
        )
        mimic3_drug_administration_route_mapping = self.helpers.load_mapping(
            self.drug_administration_route_mapping_path
        )
        mimic3_drug_class_mapping = self.helpers.load_mapping(
            self.drug_class_mapping_path
        )
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

        d_items = pl.scan_csv(self.d_items_path).select("ITEMID", "LABEL")
        inputevents_mv = (
            pl.scan_csv(
                self.inputevents_mv_path, schema_overrides={"AMOUNT": float}
            )
            .select(
                "HADM_ID",
                "ICUSTAY_ID",
                "STARTTIME",
                "ENDTIME",
                "ITEMID",
                "AMOUNT",
                "AMOUNTUOM",
                "RATE",
                "RATEUOM",
                "ORDERCATEGORYNAME",
            )
            .with_columns(
                pl.col("ORDERCATEGORYNAME")
                .replace(mimic3_drug_administration_route_mapping, default=None)
                .alias(self.drug_admin_route_col),
                pl.col("ORDERCATEGORYNAME")
                .replace(mimic3_drug_class_mapping, default=None)
                .alias(self.drug_class_col),
                # Rename units
                pl.col("RATEUOM")
                .str.replace("grams", "g")
                .str.replace("hour", "hr")
                .str.replace("mL", "ml")
                .str.replace("mEq\.", "mEq")
                .str.replace("units", "U"),
            )
        )
        inputevents_cv = (
            pl.scan_csv(
                self.inputevents_cv_path, schema_overrides={"AMOUNT": float}
            )
            .select(
                "HADM_ID",
                "ICUSTAY_ID",
                "CHARTTIME",
                "ITEMID",
                "AMOUNT",
                "AMOUNTUOM",
                "RATE",
                "RATEUOM",
                "ORIGINALROUTE",
            )
            # NOTE: dirty, but necessary to join with inputevents_mv
            .rename({"CHARTTIME": "STARTTIME"})
            .with_columns(
                pl.col("ORIGINALROUTE")
                .replace(mimic3_drug_administration_route_mapping, default=None)
                .alias(self.drug_admin_route_col),
                # Rename units
                pl.col("RATEUOM")
                .str.replace("hr", "/hr")
                .str.replace("min", "/min")
                .str.replace("kg", "/kg"),
            )
        )

        return (
            pl.concat([inputevents_mv, inputevents_cv], how="diagonal_relaxed")
            .rename(
                {
                    "HADM_ID": self.hospital_stay_id_col,
                    "ICUSTAY_ID": self.icu_stay_id_col,
                    "AMOUNT": self.drug_amount_col,
                    "AMOUNTUOM": self.drug_amount_unit_col,
                    "RATE": self.drug_rate_col,
                    "RATEUOM": self.drug_rate_unit_col,
                }
            )
            .join(d_items, on="ITEMID")
            .join(inputevents_to_rxnorm_data, on="ITEMID", how="left")
            .drop(self.hospital_stay_id_col, "ITEMID")
            .join(intimes, on=self.icu_stay_id_col)
            # Rename columns for consistency
            .rename(
                {
                    "LABEL": self.drug_name_col,
                    "LABEL_OMOP": self.drug_name_OMOP_col,
                }
            )
            # Replace drug names with mapped names
            .with_columns(
                pl.col(self.drug_name_col)
                .replace_strict(mimic3_medication_mapping, default=None)
                .alias(self.drug_ingredient_col),
            )
            # Change times to relative times
            .with_columns(
                pl.col("INTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("STARTTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("ENDTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
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
            .drop("STARTTIME", "ENDTIME", "INTIME", self.icu_length_of_stay_col)
        )

    # endregion

    # region diagnoses
    # Extract diagnoses from the diagnoses_icd.csv file
    def extract_diagnoses(self) -> pl.LazyFrame:
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
            # include only ICU patients
            .filter(
                pl.col(self.hospital_stay_id_col).is_in(
                    self.icu_stay_id.select(self.hospital_stay_id_col).collect(
                        streaming=True
                    )
                )
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
