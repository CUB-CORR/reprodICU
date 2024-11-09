# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script extracts the data from the source files and provides the extracted data
# in a structured format for further processing and harmonization.

import numpy as np
import pandas as pd
import polars as pl
import os.path

from helpers.helper_filepaths import MIMIC4Paths
from helpers.helper import GlobalHelpers


class MIMIC4Extractor(MIMIC4Paths):
    def __init__(self, paths, DEMO=False):
        super().__init__(paths, DEMO)
        self.path = paths.mimic3_source_path
        self.helpers = GlobalHelpers()
        self.icu_stay_id = self.extract_patient_information().select(
            [
                self.icu_stay_id_col,
                self.hospital_stay_id_col,
                self.person_id_col,
            ]
        )
        self.icu_length_of_stay = self.extract_patient_information().select(
            [self.icu_stay_id_col, self.icu_length_of_stay_col]
        )

        self.other_lab_values = [
            "Bilirubin.direct [Mass/volume] in Serum or Plasma",
            "Bilirubin.indirect [Mass/volume] in Serum or Plasma",
            "Bilirubin.total [Mass/volume] in Serum or Plasma",
            "Calcium [Mass/volume] in Blood",
            "Calcium.ionized [Mass/volume] in Blood",
            "Creatine kinase.MB [Mass/volume] in Serum or Plasma",
            "Iron [Mass/volume] in Serum or Plasma",
            "Iron binding capacity [Mass/volume] in Serum or Plasma",
            "Magnesium [Mass/volume] in Serum or Plasma",
            "Phosphate [Mass/volume] in Serum or Plasma",
            "Triiodothyronine (T3) [Mass/volume] in Serum or Plasma",
            "Thyroxine (T4) [Mass/volume] in Serum or Plasma",
            "Thyroxine (T4) free [Mass/volume] in Serum or Plasma",
            "Cobalamin (Vitamin B12) [Mass/volume] in Serum or Plasma",
            "Ammonia [Moles/volume] in Plasma",
            "Bicarbonate [Moles/volume] in Blood",
            "Carboxyhemoglobin/Hemoglobin.total in Blood",
            "Methehemoglobin/Hemoglobin.total in Blood",
            "Oxyhemoglobin/Hemoglobin.total in Blood",
            "Leukocytes [#/volume] in Blood by Automated count",
            "Basophils/100 leukocytes in Blood by Automated count",
            "Eosinophils/100 leukocytes in Blood by Automated count",
            "Lymphocytes/100 leukocytes in Blood by Automated count",
            "Monocytes/100 leukocytes in Blood by Automated count",
            "Neutrophils/100 leukocytes in Blood by Automated count",
            "Erythrocyte distribution width [Ratio] by Automated count",
            "Erythrocytes [#/volume] in Blood by Automated count",
            "Platelets [#/volume] in Blood by Automated count",
            "MCH [Entitic mass] by Automated count",
            "MCHC [Mass/volume] by Automated count",
            "MCV [Entitic volume] by Automated count",
            "Troponin T.cardiac [Mass/volume] in Serum or Plasma",
            "Basophils [#/volume] in Blood by Automated count",
            "Eosinophils [#/volume] in Blood by Automated count",
            "Lymphocytes [#/volume] in Blood by Automated count",
            "Monocytes [#/volume] in Blood by Automated count",
            "Neutrophils [#/volume] in Blood by Automated count",
            "Reticulocytes [#/volume] in Blood",
            "Reticulocytes [#/volume] in Blood by Automated count",
            "Reticulocytes [#/volume] in Blood by Manual count",
        ]

    # region ID mapping table
    # Extract the patient IDs that are used in the MIMIC-IV dataset
    def extract_patient_IDs(self) -> pl.LazyFrame:
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
                [
                    self.icu_stay_id_col,
                    self.hospital_stay_id_col,
                    self.person_id_col,
                    self.icu_length_of_stay_col,
                    "intime",
                ]
            )
        )

    # region patient
    # Extract patient information from the patient.csv file
    def extract_patient_information(self) -> pl.LazyFrame:
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
                    "admission_type": self.admission_type_col,
                    "hospital_expire_flag": self.mortality_hosp_col,
                }
            )
            .select(
                [
                    self.hospital_stay_id_col,
                    self.ethnicity_col,
                    self.admission_loc_col,
                    self.discharge_loc_col,
                    self.admission_type_col,
                    self.mortality_hosp_col,
                    "admittime",
                    "dischtime",
                    "deathtime",
                ]
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
            .select([self.person_id_col, self.gender_col, self.age_col, "dod"])
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
                .replace(self.UNIT_TYPES_MAP)
                .cast(self.unit_types_dtype),
                # Convert categorical discharge location to enum
                pl.col(self.discharge_loc_col)
                .replace(self.DISCHARGE_LOCATIONS_MAP)
                .cast(self.discharge_locations_dtype),
                # Convert categorical admission type to enum
                pl.col(self.admission_type_col)
                .replace(self.ADMISSION_TYPES_MAP)
                .cast(self.admission_types_dtype),
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
                [self.hospital_stay_id_col, "transfertime", self.specialty_col]
            )
            .join(IDs, on=self.hospital_stay_id_col)
            # Get the most recent specialty
            .filter(pl.col("transfertime") < pl.col("intime"))
            # Get the most recent specialty on ICU admission
            .group_by(self.icu_stay_id_col)
            .first()
            .select([self.icu_stay_id_col, self.specialty_col])
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
            .select([self.icu_stay_id_col, self.weight_col, self.height_col])
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
        # NOTE: ASSUMPTION: These are the lab values of interest
        # TODO: Confer with medical experts to confirm these are the correct values
        vital_names_mapping = self.helpers.load_mapping(
            self.vitals_mapping_path
        )
        d_items = (
            pl.scan_csv(self.d_items_path)
            .select("itemid", "label")
            .with_columns(
                pl.col("label").replace(vital_names_mapping, default=None)
            )
            # Filter for names of interest
            .filter(
                pl.col("label").is_not_null(),
                pl.col("label").is_in(self.all_values + self.other_lab_values),
            )
        )

        meas_chartevents_main_data = (
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
            .with_columns(
                pl.col("label").replace(
                    {
                        "Systolic blood pressure": "Invasive systolic arterial pressure",
                        "Diastolic blood pressure": "Invasive diastolic arterial pressure",
                        "Mean blood pressure": "Invasive mean arterial pressure",
                        "Systolic blood pressure by Noninvasive": "Non-invasive systolic arterial pressure",
                        "Diastolic blood pressure by Noninvasive": "Non-invasive diastolic arterial pressure",
                        "Mean blood pressure by Noninvasive": "Non-invasive mean arterial pressure",
                        "Glasgow coma score verbal": "Glasgow Coma Score verbal",
                        "Glasgow coma score motor": "Glasgow Coma Score motor",
                        "Glasgow coma score eye opening": "Glasgow Coma Score eye opening",
                        "Body temperature": "Temperature",
                        "Intracranial pressure (ICP)": "Intracranial pressure",
                        "Central venous pressure (CVP)": "Central venous pressure",
                    }
                )
            )
            # Filter for names of interest
            .filter(
                pl.col("label").is_in(self.all_values + self.other_lab_values)
            )
        )

        return (
            pl.scan_csv(self.chartevents_path)
            .select("hadm_id", "itemid", "charttime", "valuenum")
            # Rename columns for consistency
            .rename({"hadm_id": self.hospital_stay_id_col})
            .with_columns(
                pl.col("charttime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col(self.hospital_stay_id_col).cast(int),
            )
            .pipe(self.extract_timeseries_helper)
            .join(meas_chartevents_main_data, on="itemid", how="left")
            .join(d_items, on="itemid", how="left", suffix="_d_items")
            .with_columns(
                pl.when(pl.col("label").is_null())
                .then(pl.col("label_d_items"))
                .otherwise(pl.col("label"))
                .alias("label")
            )
            .drop("itemid", "label_d_items")
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
            # Filter for lab names of interest
            # .filter(pl.col("label").is_in(self.all_values + self.other_lab_values))
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
            .with_columns(
                pl.col("label")
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
                .alias("label"),
                pl.struct(
                    value="valuenum", source="source", method="method"
                ).alias("value_struct"),
            )
        )

    # endregion

    # region output
    # Extract output measurements from the outputevents.csv file
    def extract_output_measurements(self) -> pl.LazyFrame:
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
            # Harmonize names of interest
            .with_columns(
                pl.col("label").replace_strict(
                    self.relevant_intakeoutput_values_mapping, default=None
                )
            )
            # Filter for names of interest
            .filter(pl.col("label").is_in(self.relevant_intakeoutput_values))
        )

        return (
            pl.scan_csv(self.outputevents_path, infer_schema_length=100000)
            .select("hadm_id", "itemid", "charttime", "value")
            # Rename columns for consistency
            .rename({"hadm_id": self.hospital_stay_id_col, "value": "valuenum"})
            # BUG: .drop_nulls() drops all rows with any(!) null values
            # .drop_nulls()  # NOTE: CLEARLY THINK ABOUT THIS (-> are these baselines?)
            .with_columns(
                pl.col("charttime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col(self.hospital_stay_id_col).cast(int),
            )
            .pipe(self.extract_timeseries_helper)
            .join(outputevents_to_loinc_data, on="itemid", how="left")
            .drop("itemid")
            # Remove rows with empty names
            .filter(pl.col("label").is_not_null() & (pl.col("label") != ""))
            # Remove rows with empty values
            .filter(pl.col("valuenum").is_not_null())
            # Remove duplicate rows
            .unique()
        )

    # endregion

    # region medications
    # Extract medications from the inputevents.csv file
    def extract_medications(self) -> pl.LazyFrame:
        print("MIMIC4  - Extracting medications...")

        intimes = self.extract_patient_IDs().select(
            self.icu_stay_id_col, "intime", self.icu_length_of_stay_col
        )
        mimic4_medication_mapping = (
            self.helpers.load_many_to_many_to_one_mapping(
                self.mapping_path + "MEDICATIONS.yaml", "mimic4"
            )
        )
        mimic4_drug_administration_route_mapping = self.helpers.load_mapping(
            self.drug_administration_route_mapping_path
        )
        mimic4_drug_class_mapping = self.helpers.load_mapping(
            self.drug_class_mapping_path
        )
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

        d_items = pl.scan_csv(self.d_items_path).select("itemid", "label")
        inputevents = (
            pl.scan_csv(self.inputevents_path)
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
                "ordercategoryname",
            )
            .rename(
                {
                    "hadm_id": self.hospital_stay_id_col,
                    "stay_id": self.icu_stay_id_col,
                    "amount": self.drug_amount_col,
                    "amountuom": self.drug_amount_unit_col,
                    "rate": self.drug_rate_col,
                    "rateuom": self.drug_rate_unit_col,
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
            )
        )

        return (
            inputevents.join(d_items, on="itemid")
            .join(inputevents_to_rxnorm_data, on="itemid", how="left")
            .drop(self.hospital_stay_id_col, "itemid")
            .join(intimes, on=self.icu_stay_id_col)
            # Rename columns for consistency
            .rename(
                {
                    "label": self.drug_name_col,
                    "label_OMOP": self.drug_name_OMOP_col,
                }
            )
            # Replace drug names with mapped names
            .with_columns(
                pl.col(self.drug_name_col)
                .replace_strict(mimic4_medication_mapping, default=None)
                .alias(self.drug_ingredient_col),
            )
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
            .drop("starttime", "endtime", "intime", self.icu_length_of_stay_col)
        )

    # endregion

    # region diagnoses
    # Extract diagnoses from the diagnoses_icd.csv file
    def extract_diagnoses(self) -> pl.LazyFrame:
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
                    self.icu_stay_id.select(self.hospital_stay_id_col).collect(
                        streaming=True
                    )
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
        print("MIMIC4  - Extracting procedures...")

        intimes = self.extract_patient_IDs().select(
            self.icu_stay_id_col, "intime"
        )

        # d_items = pl.scan_csv(self.d_items_path).select("ITEMID", "LABEL"
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
            # .join(d_items, on="itemid")
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
