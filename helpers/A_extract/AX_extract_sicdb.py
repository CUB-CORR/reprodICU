# Author: Finn Fassbender
# Last modified: 2024-09-10

# Description: This script extracts the data from the source files and provides the extracted data
# in a structured format for further processing and harmonization.

import numpy as np
import pandas as pd
import polars as pl
import os.path

from helpers.helper_filepaths import SICdbPaths
from helpers.helper import GlobalHelpers


class SICdbExtractor(SICdbPaths):
    def __init__(self, paths):
        super().__init__(paths)
        self.path = paths.sicdb_source_path
        self.helpers = GlobalHelpers()

    # region patient
    # Extract patient information from the patient.csv file
    def extract_patient_information(self) -> pl.LazyFrame:
        return (
            pl.scan_csv(self.cases_path)
            .rename(
                {
                    "CaseID": self.icu_stay_id_col,
                    "PatientID": self.person_id_col,
                    "AgeOnAdmission": self.age_col,
                    "HeightOnAdmission": self.height_col,
                    "WeightOnAdmission": self.weight_col,
                }
            )
            .with_columns(
                # Convert weight to kg from g
                pl.col(self.weight_col).truediv(1000).cast(float).alias(self.weight_col),
                # Convert length of stay to days
                pl.duration(seconds=(pl.col("TimeOfStay") - pl.col("ICUOffset")))
                .truediv(pl.duration(days=1))
                .alias(self.icu_length_of_stay_col),
                # Convert gender to established dtype
                pl.col("Sex")
                .replace_strict({735: "Male", 736: "Female"}, default="Unknown")
                .cast(self.gender_dtype),
                # Convert admission origin to established dtype
                pl.col("ReferringUnit")
                .replace_strict(self._extract_references("ReferringUnit"))
                .replace_strict(self.ADMISSION_LOCATIONS_MAP, default=None)
                .cast(self.admission_locations_dtype)
                .alias(self.admission_loc_col),
                # Convert specialty to established dtype
                pl.col("ReferringUnit")
                .replace_strict(self._extract_references("ReferringUnit"))
                .replace_strict(self.SPECIALTIES_MAP, default=None)
                .cast(self.specialties_dtype)
                .alias(self.specialty_col),
                # Convert unit type to established dtype
                pl.col("HospitalUnit")
                .replace_strict(self._extract_references("HospitalUnit"))
                .replace_strict(self.UNIT_TYPES_MAP, default=None)
                .cast(self.unit_types_dtype)
                .alias(self.unit_type_col),
                # Convert discharge destination to established dtype
                pl.col("DischargeUnit")
                .replace_strict(self._extract_references("DischargeUnit"))
                .replace_strict(self.DISCHARGE_LOCATIONS_MAP, default=None)
                .cast(self.discharge_locations_dtype)
                .alias(self.discharge_loc_col),
                # Convert mortality to established dtype
                (pl.col("DischargeState") == 2202)  # "lebend"
                .cast(bool)
                .alias(self.mortality_icu_col),
                (pl.col("HospitalDischargeType") == 2026)  # "Survived"
                .cast(bool)
                .alias(self.mortality_hosp_col),
                # Convert post ICU discharge mortality to days
                pl.duration(seconds=pl.col("OffsetOfDeath"))
                .truediv(pl.duration(days=1))
                .alias(self.mortality_after_col),
                # Set care site
                pl.lit("Landeskrankenhaus Salzburg").alias(self.care_site_col),
                # Create empty HospitalStayID column
                pl.lit(None).alias(self.hospital_stay_id_col),
            )
        )

    # endregion

    # region timeseries
    # Extract timeseries information from the data_float_h.csv file
    def extract_timeseries(self) -> pl.LazyFrame:
        timeseries_mapping = self.load_mapping(self.timeseries_mapping_path)
        offsets = (
            pl.scan_csv(self.cases_path)
            .rename({"CaseID": self.icu_stay_id_col})
            .select([self.icu_stay_id_col, "ICUOffset"])
        )
        timeseries = (
            pl.scan_csv(self.data_float_h_path)
            .select(["CaseID", "Offset", "DataID", "Val"])
            .rename({"CaseID": self.icu_stay_id_col})
        )

        return (
            timeseries.join(offsets, on=self.icu_stay_id_col).with_columns(
                # Fix time offset
                (pl.col("Offset") - pl.col("ICUOffset"))
                .cast(float)
                .alias(self.timeseries_time_col),
                # Convert parameter IDs to names, then map them
                pl.col("DataID").replace_strict(timeseries_mapping, default=None).alias("DataID"),
            )
            # Keep only timepoints within timeframe of ICU stay + PRE_ICU_TIMESERIES_DAYS_CUTOFF
            # NOTE: seems not to be necessary, as the data is already filtered
            # Filter only relevant timeseries values
            .filter(pl.col("DataID").is_in(self.all_relevant_values))
            # Remove duplicate rows
            .unique()
            # Remove rows with empty parameter names
            .filter(pl.col(self.timeseries_time_col).is_not_null())
            # Remove rows with empty parameter results
            .filter(pl.col("Val").is_not_null())
            # Drop columns
            .drop(["ICUOffset", "Offset"])
        )

    # region laboratory
    # Extract laboratory information from the laboratory.csv file
    def extract_laboratory_timeseries(self) -> pl.LazyFrame:
        laboratory_mapping = self.load_mapping(self.laboratory_mapping_path)
        offsets = self._get_offsets()

        return (
            pl.scan_csv(self.laboratory_path)
            .rename({"CaseID": self.icu_stay_id_col})
            .join(offsets, on=self.icu_stay_id_col)
            .with_columns(
                # Fix lab time offset
                (pl.col("Offset") - pl.col("CaseOffset"))
                .cast(float)
                .alias(self.timeseries_time_col),
                # Convert lab IDs to names, then map them
                pl.col("LaboratoryID")
                .replace_strict(self._extract_references("Laboratory"), default=None)
                .replace_strict(laboratory_mapping, default=None),
            )
            # Keep only timepoints within timeframe of ICU stay + PRE_ICU_TIMESERIES_DAYS_CUTOFF
            # NOTE: seems not to be necessary, as the data is already filtered
            # Filter only relevant lab values
            .filter(pl.col("LaboratoryID").is_in(self.all_relevant_values))
            # Remove duplicate rows
            .unique()
            # Remove rows with empty lab names
            .filter(pl.col("LaboratoryID").is_not_null())
            # Remove rows with empty lab results
            .filter(pl.col("LaboratoryValue").is_not_null() & (pl.col("LaboratoryID") != ""))
            # Drop columns
            .drop(["CaseOffset", "LaboratoryType"])
        )

    # endregion

    # region medication
    # Extract medication information from the medication.csv file
    def extract_medications(self) -> pl.LazyFrame:
        print("SICdb   - Extracting medications...")

        # sicdb_medication_mapping = self.helpers.load_many_to_many_to_one_mapping(
        #     self.mapping_path + "MEDICATIONS.yaml", "sicdb"
        # )
        offsets = self._get_offsets()

        return (
            pl.scan_csv(self.medication_path)
            .select(["CaseID", "DrugID", "Offset", "OffsetDrugEnd", "Amount"])
            .rename({"CaseID": self.icu_stay_id_col, "Amount": self.drug_amount_col})
            .join(offsets, on=self.icu_stay_id_col)
            .with_columns(
                # Fix medication time offset
                (pl.col("Offset") - pl.col("CaseOffset")).cast(float).alias(self.drug_start_col),
                (pl.col("OffsetDrugEnd") - pl.col("CaseOffset"))
                .cast(float)
                .alias(self.drug_end_col),
                # Convert medication IDs to names, then map them
                pl.col("DrugID")
                .replace_strict(self._extract_references("Drug"), default=None)
                .alias(self.drug_name_col),
            )
            .with_columns(
                # Get drug units
                pl.col(self.drug_name_col)
                .replace_strict(self._extract_drug_units(), default=None)
                .alias(self.drug_unit_col),
            )
            # .with_columns(
            #     # Map medication names to harmonized names
            #     pl.col(self.drug_name_col)
            #     .replace_strict(sicdb_medication_mapping, default=None)
            #     .alias(self.drug_ingredient_col),
            # )
            # Keep only timepoints within timeframe of ICU stay + PRE_ICU_TIMESERIES_DAYS_CUTOFF
            # NOTE: seems not to be necessary, as the data is already filtered
            # Remove duplicate rows
            .unique()
            # Remove rows with empty medication names
            .filter(pl.col(self.drug_name_col).is_not_null())
            # Remove rows with empty medication results
            .filter(pl.col(self.drug_amount_col).is_not_null())
            # Drop columns
            .drop(["CaseOffset", "Offset", "OffsetDrugEnd"])
        )

    # endregion

    # region diagnosis
    # Extract diagnosis information from the cases.csv file
    def extract_diagnoses(self) -> pl.LazyFrame:
        print("SICdb   - Extracting diagnoses...")

        return (
            pl.scan_csv(self.cases_path)
            .select(["CaseID", "PatientID", "ICD10Main"])
            .rename(
                {
                    "CaseID": self.icu_stay_id_col,
                    "PatientID": self.person_id_col,
                    "ICD10Main": self.diagnosis_icd_code_col,
                }
            )
            .with_columns(
                # Remove dot from ICD code
                pl.col(self.diagnosis_icd_code_col).str.replace(".", ""),
                # Diagnoses are admission diagnoses
                pl.lit(0).alias(self.diagnosis_start_col),
                pl.lit(1).alias(self.diagnosis_priority_col),
                pl.lit(10).alias(self.diagnosis_icd_version_col),
                # Diagnosis descriptions are available, but only in German
            )
        )

    # region mappers
    # Extract the information from the d_references.csv file
    def _extract_references(self, ReferenceName: str) -> dict:
        references = (
            pl.read_csv(self.d_references_path)
            .filter(pl.col("ReferenceName") == ReferenceName)
            .select(["ReferenceGlobalID", "ReferenceValue"])
        )

        return dict(
            zip(references["ReferenceGlobalID"].to_numpy(), references["ReferenceValue"].to_numpy())
        )

    def _extract_drug_units(self) -> pl.LazyFrame:
        drug_units = (
            pl.read_csv(self.d_references_path)
            .filter(pl.col("ReferenceName") == "Drug")
            .select(["ReferenceValue", "ReferenceUnit"])
        )

        return dict(
            zip(drug_units["ReferenceValue"].to_numpy(), drug_units["ReferenceUnit"].to_numpy())
        )

    # endregion

    # region timehelper
    def _get_offsets(self) -> float:
        return (
            pl.scan_csv(self.cases_path)
            .select(["CaseID", "ICUOffset", "OffsetAfterFirstAdmission"])
            .rename({"CaseID": self.icu_stay_id_col})
            .with_columns(
                (pl.col("OffsetAfterFirstAdmission") + pl.col("ICUOffset"))
                .cast(float)
                .alias("CaseOffset")
            )
            .drop(["ICUOffset", "OffsetAfterFirstAdmission"])
        )
