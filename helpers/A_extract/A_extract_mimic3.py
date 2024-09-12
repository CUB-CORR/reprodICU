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
            [self.icu_stay_id_col, self.hospital_stay_id_col, self.person_id_col]
        )
        self.icu_length_of_stay = self.extract_patient_information().select(
            [self.icu_stay_id_col, self.icu_length_of_stay_col]
        )

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
                [
                    self.icu_stay_id_col,
                    self.hospital_stay_id_col,
                    self.person_id_col,
                    self.icu_length_of_stay_col,
                    "INTIME",
                ]
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
                    "HOSPITAL_EXPIRE_FLAG": self.mortality_hosp_col,
                }
            )
            .select(
                [
                    self.hospital_stay_id_col,
                    self.ethnicity_col,
                    self.admission_loc_col,
                    self.discharge_loc_col,
                    self.mortality_hosp_col,
                    "ADMITTIME",
                    "DISCHTIME",
                    "DEATHTIME",
                ]
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
            .select([self.person_id_col, self.gender_col, "DOB", "DOD"])
        )

        return (
            icustays.join(admissions, on=self.hospital_stay_id_col, how="left")
            .join(patients, on=self.person_id_col, how="left")
            .join(self._extract_patient_height_weight(icustays), on=self.icu_stay_id_col)
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
                pl.lit("Beth Israel Deaconess Medical Center").alias(self.care_site_col),
            )
            .with_columns(
                # Calculate age
                ((pl.col("INTIME") - pl.col("DOB")) / pl.duration(days=self.DAYS_IN_YEAR)).alias(
                    self.age_col
                ),
            )
            .with_columns(
                # Convert categorical gender to enum
                pl.col(self.gender_col)
                .replace({"M": "Male", "F": "Female"})
                .cast(self.gender_dtype),
                # Convert categorical ethnicity to enum
                pl.col(self.ethnicity_col).replace(self.ETHNICITY_MAP).cast(self.ethnicity_dtype),
                # Fix age
                # NOTE: ASSUMPTION: Replace age values of 300 with 90 and convert to int
                # cf. https://github.com/MIT-LCP/mimic-code/issues/637
                pl.when((pl.col(self.age_col) >= 299))
                .then(90)
                .otherwise(pl.col(self.age_col))
                .cast(int)
                .alias(self.age_col),
                # Calculate pre ICU length of stay
                ((pl.col("INTIME") - pl.col("ADMITTIME")) / pl.duration(days=1))
                .cast(float)
                .alias(self.pre_icu_length_of_stay_col),
                # Calculate ICU mortality
                ((pl.col("DEATHTIME") - pl.col("OUTTIME")) / pl.duration(hours=1))
                .le(pl.duration(hours=self.ICU_DISCHARGE_MORTALITY_CUTOFF))
                .cast(bool)
                .fill_null(False)
                .alias(self.mortality_icu_col),
                # Calculate hospital mortality
                pl.col(self.mortality_hosp_col).cast(bool),
                # Calculate mortality after discharge
                ((pl.col("DOD") - pl.col("OUTTIME")) / pl.duration(days=1))
                .cast(int)
                .alias(self.mortality_after_col),
                # Convert categorical admission location to enum
                pl.col(self.admission_loc_col)
                .replace(self.ADMISSION_LOCATIONS_MAP)
                .cast(self.admission_locations_dtype),
                # Convert categorical unit type to enum
                pl.col(self.unit_type_col).replace(self.UNIT_TYPES_MAP).cast(self.unit_types_dtype),
                # Convert categorical discharge location to enum
                pl.col(self.discharge_loc_col)
                .replace(self.DISCHARGE_LOCATIONS_MAP)
                .cast(self.discharge_locations_dtype),
            )
            .select(
                [
                    self.icu_stay_id_col,
                    self.hospital_stay_id_col,
                    self.person_id_col,
                    self.gender_col,
                    self.age_col,
                    self.height_col,
                    self.weight_col,
                    self.ethnicity_col,
                    self.pre_icu_length_of_stay_col,
                    self.icu_length_of_stay_col,
                    self.mortality_hosp_col,
                    self.mortality_icu_col,
                    self.mortality_after_col,
                    self.admission_loc_col,
                    self.unit_type_col,
                    self.care_site_col,
                    self.discharge_loc_col,
                ]
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
    def _extract_patient_height_weight(self, icustays: pl.LazyFrame, force=False) -> pl.DataFrame:
        # check if precalculated data is available
        if os.path.isfile(self.precalc_path + "MIMIC3_height_weight.parquet") and not force:
            return pl.scan_parquet(self.precalc_path + "MIMIC3_height_weight.parquet")

        print("MIMIC3 - Extracting patient height and weight...")

        # ITEMIDS = {
        #     "weight_kg_2": 224639,
        #     "weight_kg": 226512,
        #     "weight_lbs": 226531,
        #     "height_inch": 226707,
        #     "height_cm": 226730,
        # }

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
        KEEPITEMS = [*ITEMIDS.values()]

        height_weight = (
            pl.scan_csv(self.chartevents_path)
            .select("ICUSTAY_ID", "ITEMID", "VALUENUM", "CHARTTIME")
            # Rename columns for consistency
            .rename({"ICUSTAY_ID": self.icu_stay_id_col})
            .join(icustays.select(self.icu_stay_id_col, "INTIME"), on=self.icu_stay_id_col)
            .with_columns(
                pl.col("CHARTTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("INTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("ITEMID").replace_strict(ITEMIDS, default=None),
            )
            .drop_nulls("ITEMID")
            .filter(
                (pl.col("CHARTTIME") - pl.col("INTIME")).le(
                    pl.duration(hours=self.ADMISSION_WEIGHT_HEIGHT_CUTOFF)
                ),
                pl.col("ITEMID").is_in(KEEPITEMS),
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
                aggregate_function="max",  # NOTE: -> or mean?
            )
            .select([self.icu_stay_id_col, self.weight_col, self.height_col])
            .cast({self.weight_col: float, self.height_col: float})
        )

        # Save precalculated data
        height_weight.write_parquet(self.precalc_path + "MIMIC3_height_weight.parquet")

        return height_weight.lazy()

    # endregion

    # region TS helper
    # make available the common processing steps for the MIMIC-III timeseries
    def extract_timeseries_helper(self, data) -> pl.LazyFrame:
        IDs = self.extract_patient_IDs()

        return (
            data.join(IDs, on=self.hospital_stay_id_col, how="left")
            .with_columns(pl.col("INTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"))
            .with_columns((pl.col("CHARTTIME") - pl.col("INTIME")).alias("OFFSET"))
            .drop("CHARTTIME", "INTIME")
            .with_columns(
                pl.duration(days=pl.col(self.icu_length_of_stay_col)).alias(
                    self.icu_length_of_stay_col
                ),
            )
            .filter(
                (pl.col("OFFSET") < pl.col(self.icu_length_of_stay_col))
                & (pl.col("OFFSET") > pl.duration(days=-self.PRE_ICU_TIMESERIES_DAYS_CUTOFF))
            )
            .with_columns(
                (pl.col("OFFSET") / pl.duration(seconds=1))
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
        # NOTE: ASSUMPTION: These are the lab values of interest
        # TODO: Confer with medical experts to confirm these are the correct values
        vital_names_mapping = self.helpers.load_mapping(self.vitals_mapping_path)

        d_items = pl.scan_csv(self.d_items_path)
        chartevents = (
            pl.scan_csv(self.chartevents_path)
            # Rename columns for consistency
            .rename(
                {
                    "HADM_ID": self.hospital_stay_id_col,
                }
            )
        )

        return (
            chartevents.select([self.hospital_stay_id_col, "ITEMID", "CHARTTIME", "VALUENUM"])
            # BUG: .drop_nulls() drops all rows with any(!) null values
            # .drop_nulls()  # NOTE: CLEARLY THINK ABOUT THIS (-> are these baselines?)
            .with_columns(
                pl.col("CHARTTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col(self.hospital_stay_id_col).cast(int),
            )
            .pipe(self.extract_timeseries_helper)
            .join(d_items.select("ITEMID", "LABEL"), on="ITEMID")
            .drop("ITEMID")
            # Keep only relevant columns
            .with_columns(
                # Replace lab names with mapped names
                pl.col("LABEL")
                .replace_strict(vital_names_mapping, default=None)
                .alias("LABEL")
            )
            # Filter for lab names of interest
            .filter(pl.col("LABEL").is_in(self.all_relevant_values))
            # Remove duplicate rows
            .unique()
            # Remove rows with empty lab names
            .filter(pl.col("VALUENUM").is_not_null())
            # Remove rows with empty lab results
            .filter(pl.col("LABEL").is_not_null() & (pl.col("LABEL") != ""))
        )

    # endregion

    # region lab TS
    # Extract lab measurements from the labevents.csv file
    def extract_lab_measurements(self) -> pl.LazyFrame:
        # NOTE: ASSUMPTION: These are the lab values of interest
        # TODO: Confer with medical experts to confirm these are the correct values
        lab_names_mapping = self.helpers.load_mapping(self.labs_mapping_path)

        d_labitems = pl.scan_csv(self.d_labitems_path)
        labevents = (
            pl.scan_csv(self.labevents_path)
            # Rename columns for consistency
            .rename(
                {
                    "HADM_ID": self.hospital_stay_id_col,
                }
            )
        )

        return (
            labevents.select([self.hospital_stay_id_col, "ITEMID", "CHARTTIME", "VALUENUM"])
            # BUG: .drop_nulls() drops all rows with any(!) null values
            # .drop_nulls()  # NOTE: CLEARLY THINK ABOUT THIS (-> are these baselines?)
            .with_columns(
                pl.col("CHARTTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col(self.hospital_stay_id_col).cast(int),
            )
            .pipe(self.extract_timeseries_helper)
            .join(d_labitems.select("ITEMID", "LABEL"), on="ITEMID")
            .drop("ITEMID")
            # Keep only relevant columns
            .with_columns(
                # Replace lab names with mapped names
                pl.col("LABEL")
                .replace_strict(lab_names_mapping, default=None)
                .alias("LABEL")
            )
            # Filter for lab names of interest
            .filter(pl.col("LABEL").is_in(self.all_relevant_values))
            # Remove duplicate rows
            .unique()
            # Remove rows with empty lab names
            .filter(pl.col("VALUENUM").is_not_null())
            # Remove rows with empty lab results
            .filter(pl.col("LABEL").is_not_null() & (pl.col("LABEL") != ""))
        )

    # endregion

    # region output TS
    # Extract output measurements from the outputevents.csv file
    def extract_output_measurements(self) -> pl.LazyFrame:
        # NOTE: ASSUMPTION: These are the lab values of interest
        # TODO: Confer with medical experts to confirm these are the correct values
        output_names_mapping = self.helpers.load_mapping(self.outputs_mapping_path)

        d_items = pl.scan_csv(self.d_items_path)
        outputevents = (
            pl.scan_csv(self.outputevents_path, infer_schema_length=100000)
            # Rename columns for consistency
            .rename(
                {
                    "HADM_ID": self.hospital_stay_id_col,
                    "VALUE": "VALUENUM",
                }
            )
        )

        return (
            outputevents.select([self.hospital_stay_id_col, "ITEMID", "CHARTTIME", "VALUENUM"])
            # BUG: .drop_nulls() drops all rows with any(!) null values
            # .drop_nulls()  # NOTE: CLEARLY THINK ABOUT THIS (-> are these baselines?)
            .with_columns(
                pl.col("CHARTTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col(self.hospital_stay_id_col).cast(int),
            )
            .pipe(self.extract_timeseries_helper)
            .join(d_items.select("ITEMID", "LABEL"), on="ITEMID")
            .drop("ITEMID")
            # Keep only relevant columns
            .with_columns(
                # Replace lab names with mapped names
                pl.col("LABEL")
                .replace_strict(output_names_mapping, default=None)
                .alias("LABEL")
            )
            # Filter for lab names of interest
            .filter(pl.col("LABEL").is_in(self.all_relevant_values))
            # Remove duplicate rows
            .unique()
            # Remove rows with empty lab names
            .filter(pl.col("VALUENUM").is_not_null())
            # Remove rows with empty lab results
            .filter(pl.col("LABEL").is_not_null() & (pl.col("LABEL") != ""))
        )

    # endregion

    # region medications
    # Extract medications from the inputevents.csv file
    def extract_medications(self) -> pl.LazyFrame:
        intimes = self.extract_patient_IDs().select(
            self.icu_stay_id_col, self.icu_length_of_stay_col, "INTIME"
        )
        mimic3_medication_mapping = self.helpers.load_many_to_many_to_one_mapping(
            self.mapping_path + "MEDICATIONS.yaml", "mimic3"
        )

        d_items = pl.scan_csv(self.d_items_path).select("ITEMID", "LABEL")
        inputevents_mv = (
            pl.scan_csv(self.inputevents_mv_path, dtypes={"AMOUNT": float})
            .rename(
                {
                    "HADM_ID": self.hospital_stay_id_col,
                    "ICUSTAY_ID": self.icu_stay_id_col,
                }
            )
            .select(
                [
                    self.hospital_stay_id_col,
                    self.icu_stay_id_col,
                    "STARTTIME",
                    "ENDTIME",
                    "ITEMID",
                    "AMOUNT",
                    "AMOUNTUOM",
                ]
            )
        )
        inputevents_cv = (
            pl.scan_csv(self.inputevents_cv_path, dtypes={"AMOUNT": float})
            .rename(
                {
                    "HADM_ID": self.hospital_stay_id_col,
                    "ICUSTAY_ID": self.icu_stay_id_col,
                    # NOTE: dirty, but necessary to join with inputevents_mv
                    "CHARTTIME": "STARTTIME",
                }
            )
            .select(
                [
                    self.hospital_stay_id_col,
                    self.icu_stay_id_col,
                    "STARTTIME",
                    "ITEMID",
                    "AMOUNT",
                    "AMOUNTUOM",
                ]
            )
        )

        return (
            pl.concat([inputevents_mv, inputevents_cv], how="diagonal_relaxed")
            .join(d_items, on="ITEMID")
            .drop(self.hospital_stay_id_col, "ITEMID")
            .join(intimes, on=self.icu_stay_id_col)
            # Rename columns for consistency
            .rename(
                {
                    "LABEL": self.drug_name_col,
                    "AMOUNT": self.drug_amount_col,
                    "AMOUNTUOM": self.drug_unit_col,
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
                .truediv(pl.duration(seconds=1))
                .alias(self.drug_start_col),
                (pl.col("ENDTIME") - pl.col("INTIME"))
                .truediv(pl.duration(seconds=1))
                .alias(self.drug_end_col),
            )
            # Keep only drugs within timeframe of ICU stay + PRE_ICU_TIMESERIES_DAYS_CUTOFF
            .filter(
                (
                    pl.col(self.drug_start_col)
                    < pl.duration(days=pl.col(self.icu_length_of_stay_col)).truediv(
                        pl.duration(seconds=1)
                    )
                )
                & (
                    pl.col(self.drug_start_col)
                    > pl.duration(days=-self.PRE_ICU_TIMESERIES_DAYS_CUTOFF).truediv(
                        pl.duration(seconds=1)
                    )
                )
            )
            .drop("STARTTIME", "ENDTIME", "INTIME", self.icu_length_of_stay_col)
        )

    # endregion

    # region diagnoses
    # Extract diagnoses from the diagnoses_icd.csv file
    def extract_diagnoses(self) -> pl.LazyFrame:
        print("MIMIC3 - Extracting diagnoses...")
        diagnoses = pl.scan_csv(self.diagnoses_icd_path, dtypes={"ICD9_CODE": str}).rename(
            {
                "SUBJECT_ID": self.person_id_col,
                "HADM_ID": self.hospital_stay_id_col,
            }
        )
        d_diagnoses = pl.scan_csv(self.d_icd_diagnoses_path, dtypes={"ICD9_CODE": str})

        return (
            diagnoses.select(self.person_id_col, self.hospital_stay_id_col, "ICD9_CODE", "SEQ_NUM")
            # include only ICU patients
            .filter(
                pl.col(self.hospital_stay_id_col).is_in(self.icu_stay_id.select(self.hospital_stay_id_col).collect(streaming=True))
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
        intimes = self.extract_patient_IDs().select(self.icu_stay_id_col, "INTIME")
        procedureevents_mv = pl.scan_csv(self.procedureevents_mv_path).rename(
            {
                "SUBJECT_ID": self.person_id_col,
                "HADM_ID": self.hospital_stay_id_col,
                "ICUSTAY_ID": self.icu_stay_id_col,
            }
        )
        d_items = pl.scan_csv(self.d_items_path).select("ITEMID", "LABEL")

        procedures_icd = pl.scan_csv(self.procedures_icd_path, dtypes={"ICD9_CODE": str}).rename(
            {
                "SUBJECT_ID": self.person_id_col,
                "HADM_ID": self.hospital_stay_id_col,
            }
        )
        d_icd_procedures = pl.scan_csv(self.d_icd_procedures_path, dtypes={"ICD9_CODE": str})

        procedureevents_mv = (
            procedureevents_mv.select([self.icu_stay_id_col, "STARTTIME", "ENDTIME", "ITEMID"])
            .join(intimes, on=self.icu_stay_id_col)
            .join(d_items, on="ITEMID")
            .with_columns(
                pl.col("STARTTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("ENDTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("INTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
            .with_columns(
                (pl.col("STARTTIME") - pl.col("INTIME"))
                .truediv(pl.duration(seconds=1))
                .alias(self.procedure_start_col),
                (pl.col("ENDTIME") - pl.col("INTIME"))
                .truediv(pl.duration(seconds=1))
                .alias(self.procedure_end_col),
            )
            .drop("ITEMID", "STARTTIME", "ENDTIME", "INTIME")
            .rename(
                {
                    "LABEL": self.procedure_description_col,
                }
            )
            .drop_nulls(self.procedure_description_col)
            .unique()
        )

        procedures_icd = (
            procedures_icd.select(
                [self.person_id_col, self.hospital_stay_id_col, "ICD9_CODE", "SEQ_NUM"]
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

        return pl.concat([procedureevents_mv, procedures_icd], how="diagonal_relaxed")

    # endregion
