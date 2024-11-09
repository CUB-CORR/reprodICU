# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script contains helper functions and classes that are used across multiple scripts.
# It contains the GlobalVars class that stores globally configured variables and the GlobalHelpers class
# that contains helper functions that are used across multiple scripts.

import polars as pl
import yaml

from typing import Sequence, Optional, Union


class GlobalHelpers:
    def __init__(self):
        pass

    def load_mapping(self, path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def load_mapping_keys(self, path: str) -> list:
        mapping = self.load_mapping(path)
        return list(mapping.keys())

    def load_mapping_true_keys(self, path: str) -> list:
        mapping = self.load_mapping(path)
        return list(k for k, v in mapping.items() if v)

    def load_many_to_one_mapping(self, path: str) -> dict:
        mapping = self.load_mapping(path)
        return {v: k for k, vs in mapping.items() for v in vs}

    def load_many_to_one_mapping_incl_keys(self, path: str) -> dict:
        mapping1 = {
            v: k
            for k, vs in self.load_mapping(path).items()
            if isinstance(vs, list)
            for v in vs
        }
        mapping2 = {k: k for k in self.load_mapping_keys(path)}
        return {**mapping1, **mapping2}

    def load_many_to_many_to_one_mapping(
        self, path: str, database: str
    ) -> dict:
        mapping = self.load_mapping(path)
        return_dict = {}
        for key, value in mapping.items():
            return_dict.update({v: key for v in value[database]})
        return return_dict

    def _convert_time_to_days_float(
        self, data: pl.LazyFrame, col_name: str, base_unit: str = "minutes"
    ):
        assert base_unit in ["hours", "minutes", "seconds"]
        if base_unit == "hours":
            divided_by = 24
        if base_unit == "minutes":
            divided_by = 24 * 60
        if base_unit == "seconds":
            divided_by = 24 * 60 * 60

        return data.with_columns(
            (pl.col(col_name) / divided_by).cast(float).alias(col_name)
        )

    def _convert_time_to_seconds_float(
        self, data: pl.LazyFrame, col_name: str, base_unit: str = "minutes"
    ) -> pl.LazyFrame:
        assert base_unit in ["hours", "minutes", "seconds"]
        if base_unit == "hours":
            multplicator = 60 * 60
        if base_unit == "minutes":
            multplicator = 60
        if base_unit == "seconds":
            multplicator = 1

        return data.with_columns(
            (pl.col(col_name) * multplicator).cast(float).alias(col_name)
        )

    def dropna(
        self,
        data: pl.LazyFrame,
        how: str = "any",
        subset_cols: Optional[Union[str, Sequence[str]]] = None,
        verbose: bool = True,
    ) -> pl.LazyFrame:
        """
        Remove null and NaN values from polars DataFrame.
        Modified from https://stackoverflow.com/a/73978691
        """

        if verbose:
            print(
                "Dropping null, NaN and empty string values from DataFrame"
                + f" in columns {subset_cols}"
                if not subset_cols is None
                else "" + "..."
            )

        subset = pl.all() if subset_cols is None else pl.col(subset_cols)
        subset_is_na = (
            subset.is_null()
            | (subset.cast(str) == "NaN")
            | (subset.cast(str) == "")
        )

        if how == "any":
            result = data.filter(~pl.any_horizontal(subset_is_na))
        elif how == "all":
            result = data.filter(~pl.all_horizontal(subset_is_na))
        elif how == "onlynull":
            result = data.filter(subset.is_not_null())
        else:
            raise ValueError(f"how must be either 'any' or 'all', got {how}")

        return result


class GlobalVars(GlobalHelpers):
    def __init__(self, paths, DEMO=False) -> None:
        config_path = "configs/"
        mapping_path = "mappings/"
        reprodICU_files_path = (
            paths.reprodICU_files_path
            if not DEMO
            else paths.reprodICU_demo_files_path
        )
        tempfiles_path = reprodICU_files_path + "_tempfiles/"

        # append globally configured variables as class attributes
        for key, value in self.load_mapping(
            config_path + "GLOBAL_CONFIG.yaml"
        ).items():
            setattr(self, key, value)

        for key, value in self.load_mapping(
            config_path + "COLUMN_NAMES.yaml"
        ).items():
            setattr(self, key, value)

        # append globally configured mappings as class attributes
        self.ETHNICITY_MAP = self.load_many_to_one_mapping(
            mapping_path + "ETHNICITY.yaml"
        )
        self.ADMISSION_LOCATIONS_MAP = self.load_many_to_one_mapping(
            mapping_path + "ADMISSION_LOCATIONS.yaml"
        )
        self.ADMISSION_TYPES_MAP = self.load_many_to_one_mapping(
            mapping_path + "ADMISSION_TYPES.yaml"
        )
        self.DISCHARGE_LOCATIONS_MAP = self.load_many_to_one_mapping(
            mapping_path + "DISCHARGE_LOCATIONS.yaml"
        )
        self.SPECIALTIES_MAP = self.load_many_to_one_mapping(
            mapping_path + "SPECIALTIES.yaml"
        )
        self.UNIT_TYPES_MAP = self.load_many_to_one_mapping(
            mapping_path + "UNIT_TYPES.yaml"
        )

        # append globally configured paths as class attributes
        self.config_path = config_path
        self.relevant_values_path = config_path + "RELEVANT_VALUES/"
        self.relevant_OMOP_values_path = config_path + "RELEVANT_VALUES_OMOP/"
        self.mapping_path = mapping_path
        self.precalc_path = tempfiles_path

        # Define constants
        self.DAYS_IN_YEAR = 365.25
        self.INCH_TO_CM = 2.54  # 1 inch = 2.54 cm
        self.LBS_TO_KG = 0.454  # 1 lb = 0.454 kg

        def F_TO_C(F: float) -> float:
            return (F - 32) * 5 / 9

        # Define custom data types
        self.gender_dtype = pl.Enum(["Male", "Female", "Other", "Unknown"])
        self.mortality_dtype = pl.Enum(["Alive", "Dead", "Unknown"])
        self.ethnicity_dtype = pl.Enum(
            self.load_mapping_keys(mapping_path + "ETHNICITY.yaml")
        )
        self.admission_locations_dtype = pl.Enum(
            self.load_mapping_keys(mapping_path + "ADMISSION_LOCATIONS.yaml")
        )
        self.admission_types_dtype = pl.Enum(
            self.load_mapping_keys(mapping_path + "ADMISSION_TYPES.yaml")
        )
        self.discharge_locations_dtype = pl.Enum(
            self.load_mapping_keys(mapping_path + "DISCHARGE_LOCATIONS.yaml")
        )
        self.specialties_dtype = pl.Enum(
            self.load_mapping_keys(mapping_path + "SPECIALTIES.yaml")
        )
        self.unit_types_dtype = pl.Enum(
            self.load_mapping_keys(mapping_path + "UNIT_TYPES.yaml")
        )

        # Define global mappings (ICD diagnoses)
        self.ICD9_TO_ICD10_DIAGS = pl.read_csv(
            mapping_path + "_icd_codes/icd9_diagnoses.csv",
            infer_schema_length=25000,
        )
        self.ICD9_TO_ICD10_PROCS = pl.read_csv(
            mapping_path + "_icd_codes/icd9_procedures.csv",
            infer_schema_length=25000,
        )
        self.ICD10_TO_ICD9_DIAGS = pl.read_csv(
            mapping_path + "_icd_codes/icd10_diagnoses.csv",
            infer_schema_length=25000,
        )
        self.ICD10_TO_ICD9_PROCS = pl.read_csv(
            mapping_path + "_icd_codes/icd10_procedures.csv",
            infer_schema_length=25000,
        )

        # Select relevant variables
        self.relevant_respiratory_values_mapping = (
            self.load_many_to_one_mapping_incl_keys(
                self.relevant_values_path + "RELEVANT_RESPIRATORY_VALUES.yaml"
            )
        )
        self.relevant_intakeoutput_values_mapping = (
            self.load_many_to_one_mapping_incl_keys(
                self.relevant_values_path + "RELEVANT_INTAKE_OUTPUT_VALUES.yaml"
            )
        )

        self.relevant_vital_values = self.load_mapping_true_keys(
            self.relevant_values_path + "RELEVANT_VITALS.yaml"
        )
        self.relevant_lab_values = self.load_mapping_true_keys(
            self.relevant_values_path + "RELEVANT_LABS.yaml"
        )
        self.relevant_respiratory_values = list(
            set(self.relevant_respiratory_values_mapping.keys())
        )
        self.relevant_intakeoutput_values = list(
            set(self.relevant_intakeoutput_values_mapping.keys())
        )

        self.all_relevant_values = (
            self.relevant_lab_values
            + self.relevant_vital_values
            + self.relevant_respiratory_values
            + self.relevant_intakeoutput_values
        )

        self.relevant_lab_values_pre_conversion = [
            "base_excess",  # for base_excess conversion in eICU
            "base_deficit",  # for base_excess conversion in eICU
            "Temperature Fahrenheit",  # for temperature conversion in MIMIC
            "Temperature Celsius",  # for temperature conversion in MIMIC
            "Bilirubin.direct [Mass/volume] in Serum or Plasma",
            "Bilirubin.indirect [Mass/volume] in Serum or Plasma",
            "Bilirubin.total [Mass/volume] in Serum or Plasma",
            "Calcium [Mass/volume] in Blood",
            "Calcium.ionized [Mass/volume] in Blood",
            "Cholesterol [Moles/volume] in Serum or Plasma",
            "Cholesterol in HDL [Moles/volume] in Serum or Plasma",
            "Cholesterol in LDL [Moles/volume] in Serum or Plasma",
            "Cobalamin (Vitamin B12) [Mass/volume] in Serum or Plasma",
            "Cortisol [Moles/volume] in Serum or Plasma",
            "Creatine kinase.MB [Mass/volume] in Serum or Plasma",
            "Creatinine [Moles/volume] in Blood",
            "Creatinine [Moles/volume] in Serum or Plasma",
            "Creatinine [Moles/volume] in Urine",
            "Glucose [Moles/volume] in Blood",
            "Glucose [Moles/volume] in Serum or Plasma",
            "Hemoglobin [Moles/volume] in Blood",
            "Iron [Mass/volume] in Serum or Plasma",
            "Iron binding capacity [Mass/volume] in Serum or Plasma",
            "Lymphocytes [#/volume] in Blood",
            "Magnesium [Mass/volume] in Serum or Plasma",
            "MCHC [Moles/volume]",
            "Phosphate [Mass/volume] in Serum or Plasma",
            "Thyroxine (T4) [Mass/volume] in Serum or Plasma",
            "Thyroxine (T4) free [Mass/volume] in Serum or Plasma",
            "Triglyceride [Moles/volume] in Blood",
            "Triiodothyronine (T3) [Mass/volume] in Serum or Plasma",
            "Urea [Mass/volume] in Serum or Plasma",
            "Urea nitrogen [Mass/volume] in Serum or Plasma",
        ]

        self.all_values = (
            self.relevant_lab_values_pre_conversion + self.all_relevant_values
        )
