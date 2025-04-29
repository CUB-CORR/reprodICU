# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script contains helper functions and classes that are used across multiple scripts.
# It contains the GlobalVars class that stores globally configured variables and the GlobalHelpers class
# that contains helper functions that are used across multiple scripts.

from typing import Optional, Sequence, Union

import polars as pl
import yaml


# region GlobalHelpers
class GlobalHelpers:
    def __init__(self):
        pass

    # region mapping helpers
    def load_mapping(self, path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def load_mapping_keys(self, path: str) -> list:
        mapping = self.load_mapping(path)
        return list(mapping.keys())

    def load_mapping_true_keys(self, path: str) -> list:
        mapping = self.load_mapping(path)
        return list(k for k, v in mapping.items() if v)

    def load_mapping_subkeys(self, path: str, key: str) -> dict:
        mapping = self.load_mapping(path)
        return {
            k: v.get(key) for k, v in mapping.items() if v.get(key) is not None
        }

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
        self, path: str, database: str, DEBUG: bool = False
    ) -> dict:
        mapping = self.load_mapping(path)
        return_dict = {}
        for key, value in mapping.items():
            if DEBUG:
                print(key, value)
            return_dict.update({v: key for v in value[database]})
        return return_dict

    # region time conversion
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

    # region dropna
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
                if subset_cols is not None
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


# region GlobalVars
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

        # region GLOBAL PATHS
        # append globally configured paths as class attributes
        self.config_path = config_path
        self.relevant_values_path = config_path + "RELEVANT_VALUES/"
        self.mapping_path = mapping_path
        self.precalc_path = tempfiles_path

        # region CONSTANTS
        # Define constants
        self.DAYS_IN_YEAR = 365.25
        self.INCH_TO_CM = 2.54  # 1 inch = 2.54 cm
        self.LBS_TO_KG = 0.454  # 1 lb = 0.454 kg
        self.OZ_TO_KG = 0.0283495  # 1 oz = 0.0283495 kg

        # region GLOBAL MAPS
        self.MEDICATION_MAPPING_PATH = mapping_path + "MEDICATIONS/"

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
        self.ADMISSION_URGENCY_MAP = self.load_many_to_one_mapping(
            mapping_path + "ADMISSION_URGENCY.yaml"
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

        self.HEART_RHYTHM_MAP = self.load_many_to_one_mapping(
            mapping_path + "ADDITIONAL_MAPPINGS/heart_rhythm_mapping.yaml"
        )
        self.OXYGEN_DELIVERY_SYSTEM_MAP = self.load_many_to_one_mapping(
            mapping_path
            + "ADDITIONAL_MAPPINGS/oxygen_delivery_device_mapping.yaml"
        )
        self.VENTILATOR_MODE_MAP = self.load_many_to_one_mapping(
            mapping_path + "ADDITIONAL_MAPPINGS/ventilator_mode_mapping.yaml"
        )
        self.SOLUTION_FLUIDS_MAP = self.load_many_to_one_mapping(
            mapping_path + "ADDITIONAL_MAPPINGS/solution_fluids_mapping.yaml"
        )

        # region DATA TYPES
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
        self.admission_urgency_dtype = pl.Enum(
            self.load_mapping_keys(mapping_path + "ADMISSION_URGENCY.yaml")
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

        # region ENUM MAPS
        # Define custom enum maps
        self.heart_rhythm_enum_map = {
            v: i
            for i, v in enumerate(
                self.load_mapping_keys(
                    mapping_path
                    + "ADDITIONAL_MAPPINGS/heart_rhythm_mapping.yaml"
                )
            )
        }
        self.heart_rhythm_enum_map_inverted = {
            i: v for v, i in self.heart_rhythm_enum_map.items()
        }
        self.oxygen_delivery_system_enum_map = {
            v: i
            for i, v in enumerate(
                self.load_mapping_keys(
                    mapping_path
                    + "ADDITIONAL_MAPPINGS/oxygen_delivery_device_mapping.yaml"
                )
            )
        }
        self.oxygen_delivery_system_enum_map_inverted = {
            i: v for v, i in self.oxygen_delivery_system_enum_map.items()
        }
        self.ventilator_mode_enum_map = {
            v: i
            for i, v in enumerate(
                self.load_mapping_keys(
                    mapping_path
                    + "ADDITIONAL_MAPPINGS/ventilator_mode_mapping.yaml"
                )
            )
        }
        self.ventilator_mode_enum_map_inverted = {
            i: v for v, i in self.ventilator_mode_enum_map.items()
        }

        # region ICD
        # Define global mappings (ICD diagnoses & procedures and more)
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
        self.ICD_TO_ICDSUBCHAPTER_DF = (
            pl.read_csv(
                mapping_path + "_icd_codes/icd_subchapters.csv",
                separator=";",
            )
            .with_columns(
                pl.col("Subchapter").str.split("|").alias("ICD Codes"),
                pl.concat_str(
                    pl.col("Chapter"),
                    # pl.lit(" ("),
                    # pl.col("Chapter Title"),
                    # pl.lit(") - "),
                    pl.lit(" - "),
                    pl.col("Subchapter Title"),
                ).alias("Title"),
            )
            .explode("ICD Codes")
            .select("ICD Codes", "Title")
        )
        self.ICD_TO_ICDSUBCHAPTER_DICT = dict(
            zip(
                self.ICD_TO_ICDSUBCHAPTER_DF["ICD Codes"],
                self.ICD_TO_ICDSUBCHAPTER_DF["Title"],
            )
        )

        self.MEASUREMENT_UNIT_CONCEPT_IDS = pl.read_csv(
            mapping_path
            + "ADDITIONAL_MAPPINGS/measurement_unit_concept_ids.csv",
            infer_schema_length=25000,
        )

        # region GLOBAL MAPPINGS
        self.timeseries_vitals_mapping = (
            self.load_many_to_one_mapping_incl_keys(
                mapping_path + "TIMESERIES_VITALS.yaml"
            )
        )
        self.timeseries_respiratory_mapping = (
            self.load_many_to_one_mapping_incl_keys(
                mapping_path + "TIMESERIES_RESPIRATORY.yaml"
            )
        )
        self.timeseries_intakeoutput_mapping = (
            self.load_many_to_one_mapping_incl_keys(
                mapping_path + "TIMESERIES_INTAKEOUTPUT.yaml"
            )
        )

        # region RELEVANT
        # Select relevant variables
        self.relevant_lab_LOINC_components = self.load_mapping_keys(
            self.relevant_values_path + "RELEVANT_LABS_LOINC.yaml"
        )
        self.relevant_lab_LOINC_systems = self.load_mapping_subkeys(
            self.relevant_values_path + "RELEVANT_LABS_LOINC.yaml", "systems"
        )
        self.conversion_lab_LOINC_components = self.load_mapping_subkeys(
            self.relevant_values_path + "RELEVANT_LABS_LOINC.yaml",
            "for_conversion",
        )

        self.relevant_vital_values = list(
            set(
                self.load_mapping_true_keys(
                    self.relevant_values_path + "RELEVANT_VITALS.yaml"
                )
            )
        )
        self.relevant_lab_values = list(
            set(
                self.load_mapping_true_keys(
                    self.relevant_values_path + "RELEVANT_LABS_LOINC.yaml"
                )
            )
        )
        self.relevant_respiratory_values = list(
            set(
                self.load_mapping_true_keys(
                    self.relevant_values_path + "RELEVANT_RESPIRATORY.yaml"
                )
            )
        )
        self.relevant_intakeoutput_values = list(
            set(
                self.load_mapping_true_keys(
                    self.relevant_values_path + "RELEVANT_INTAKEOUTPUT.yaml"
                )
            )
        )

        self.all_relevant_values = (
            self.relevant_vital_values
            + self.relevant_respiratory_values
            + self.relevant_intakeoutput_values
        )

    def ICD_TO_ICDSUBCHAPTER(self, data: pl.LazyFrame):
        return data.with_columns(
            pl.coalesce(
                pl.when(pl.col("ICD").str.starts_with(icd_code))
                .then(pl.lit(icd_code))
                .otherwise(None)
                .alias(icd_code)
                for icd_code in self.ICD_TO_ICDSUBCHAPTER_DF["ICD Codes"]
            )
            .replace_strict(self.ICD_TO_ICDSUBCHAPTER_DICT, default=None)
            .alias(self.admission_diagnosis_icd_col)
        )
