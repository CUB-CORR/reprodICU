# Author: Finn Fassbender
# Last modified: 2024-09-10

# Description: This script extracts the data from the source files and provides the extracted data
# in a structured format for further processing and harmonization.

import numpy as np
import pandas as pd
import polars as pl
import os.path

from helpers.helper_filepaths import UMCdbPaths
from helpers.helper import GlobalHelpers


class UMCdbExtractor(UMCdbPaths):
    def __init__(self, paths):
        super().__init__(paths)
        self.path = paths.umcdb_source_path
        self.helpers = GlobalHelpers()
        self.index_cols = [self.icu_stay_id_col, self.timeseries_time_col]

    # region patient
    # Extract patient information from the patient.csv file
    def extract_patient_information(self) -> pl.LazyFrame:
        return (
            pl.scan_csv(self.admissions_path)
            .select(
                [
                    "patientid",
                    "admissionid",
                    "agegroup",
                    "weightgroup",
                    "heightgroup",
                    "gender",
                    "origin",
                    "destination",
                    "specialty",
                    "admittedat",
                    "lengthofstay",
                    "dischargedat",
                    "dateofdeath",
                ]
            )
            # Rename columns for consistency
            .rename({"patientid": self.person_id_col, "admissionid": self.icu_stay_id_col})
            .with_columns(
                # for age, weight and height, assume average of the group
                pl.col("agegroup")
                .str.replace("-|\+", "–")
                .str.split("–")
                .map_elements(lambda s: np.mean([int(i) for i in s if i]), return_dtype=float)
                .cast(int)
                .alias(self.age_col),
                pl.col("weightgroup")
                .str.replace("-|\+", "–")
                .str.split("–")
                .map_elements(lambda s: np.mean([int(i) for i in s if i]), return_dtype=float)
                .cast(int)
                .alias(self.weight_col),
                pl.col("heightgroup")
                .str.replace("-|\+", "–")
                .str.split("–")
                .map_elements(lambda s: np.mean([int(i) for i in s if i]), return_dtype=float)
                .cast(int)
                .alias(self.height_col),
                # Convert categorical mortality to binary
                (pl.col("destination") == "Overleden").cast(bool).alias(self.mortality_icu_col),
                # NOTE: pre-ICU length of stay is not available in the UMCdb dataset,
                # as there is no known hospital admission / discharge data
                # # Calculate pre-ICU length of stay in days
                # pl.duration(milliseconds=pl.col("admittedat"))
                # .truediv(pl.duration(days=1))
                # .cast(float)
                # .alias(self.pre_icu_length_of_stay_col),
                # Calculate ICU length of stay in days
                pl.duration(hours=pl.col("lengthofstay"))
                .truediv(pl.duration(days=1))
                .cast(float)
                .alias(self.icu_length_of_stay_col),
                # Calculate mortality after discharge
                pl.duration(milliseconds=(pl.col("dateofdeath") - pl.col("dischargedat")))
                .truediv(pl.duration(days=1))
                .cast(float)
                .alias(self.mortality_after_col),
                # Convert categorical gender to enum
                pl.col("gender")
                .replace_strict({"Man": "Male", "Vrouw": "Female"}, default="Unknown")
                .cast(self.gender_dtype)
                .alias(self.gender_col),
                # Convert categorical admission location to enum
                pl.col("origin")
                .replace_strict(self.ADMISSION_LOCATIONS_MAP, default="Unknown")
                .cast(self.admission_locations_dtype)
                .alias(self.admission_loc_col),
                # Convert categorical discharge location to enum
                pl.col("destination")
                .replace_strict(self.DISCHARGE_LOCATIONS_MAP, default="Unknown")
                .cast(self.discharge_locations_dtype)
                .alias(self.discharge_loc_col),
                # Convert categorical unit type to enum
                pl.col("specialty")
                .replace_strict(self.UNIT_TYPES_MAP, default="Unknown")
                .cast(self.unit_types_dtype)
                .alias(self.unit_type_col),
                # Set hospital stay ID to none
                pl.lit(None).alias(self.hospital_stay_id_col),
                # Set care site to the hospital name
                pl.lit("Amsterdam Universitair Medische Centra").alias(self.care_site_col),
            )
            .drop(
                [
                    "agegroup",
                    "weightgroup",
                    "heightgroup",
                    # "gender",
                    "origin",
                    "destination",
                    "specialty",
                    "dateofdeath",
                    "dischargedat",
                    "admittedat",
                ]
            )
        )

    # endregion

    # region timeseries
    # Extract timeseries information from the listitems.csv file
    def extract_timeseries_listitems(self) -> pl.LazyFrame:
        listitems_mapping = self.helpers.load_mapping(self.listitems_mapping_path)

        listitems = (
            pl.scan_csv(self.listitems_path, dtypes={"value": str})
            .select(["admissionid", "item", "itemid", "value", "valueid", "measuredat"])
            .rename({"admissionid": self.icu_stay_id_col})
            .with_columns(
                # Replace item names with standardized names
                pl.col("item")
                .replace_strict(listitems_mapping, default=None)
                .alias("item"),
            )
            .pipe(self._extract_timeseries_helper)
        )

        gcs = self._compute_gcs(listitems)

        return listitems.drop(["valueid", "itemid"]).join(gcs, on=self.index_cols)

    # Extract timeseries information from the numericitems.csv file
    def extract_timeseries_numericitems(self) -> pl.LazyFrame:
        numericitems_mapping = self.helpers.load_mapping(self.numeric_mapping_path)

        return (
            pl.scan_csv(self.numericitems_path, dtypes={"value": str})
            .select(["admissionid", "item", "value", "measuredat"])
            .rename({"admissionid": self.icu_stay_id_col})
            .with_columns(
                # Replace item names with standardized names
                pl.col("item")
                .replace_strict(numericitems_mapping, default=None)
                .alias("item"),
            )
            .pipe(self._extract_timeseries_helper)
        )

    # filter and rename columns for timeseries data
    def _extract_timeseries_helper(self, data: pl.LazyFrame) -> pl.LazyFrame:
        intimes = (
            pl.scan_csv(self.admissions_path)
            .select(["admissionid", "admittedat", "dischargedat"])
            .rename(
                {
                    "admissionid": self.icu_stay_id_col,
                    "admittedat": "intime",
                    "dischargedat": "outtime",
                }
            )
        )

        return (
            data.join(intimes, on=self.icu_stay_id_col)
            # Keep only timepoints within timeframe of ICU stay + PRE_ICU_TIMESERIES_DAYS_CUTOFF
            .filter(
                (pl.col("measuredat") < pl.col("outtime"))
                & (
                    pl.col("measuredat")
                    > (
                        pl.col("intime")
                        - pl.duration(days=self.PRE_ICU_TIMESERIES_DAYS_CUTOFF).truediv(
                            pl.duration(milliseconds=1)
                        )
                    )
                )
            )
            .with_columns(
                pl.duration(milliseconds=(pl.col("measuredat") - pl.col("intime")))
                .dt.total_seconds()
                .cast(float)
                .alias(self.timeseries_time_col),
            )
            # Filter only relevant timeseries values
            .filter(pl.col("item").is_in(self.all_relevant_values))
            .drop(["measuredat", "intime", "outtime"])
            # Convert values to numbers, if possible, ignore if not
            .cast({"value": float}, strict=False)
        )

    # compute Glasgow Coma Scale (GCS) from listitems data
    # Implementation using item IDs as in BlendedICU
    # https://github.com/USM-CHU-FGuyon/BlendedICU/blob/master/amsterdam_preprocessing/AmsterdamPreparator.py#L131
    def _compute_gcs(self, data: pl.LazyFrame) -> pl.LazyFrame:
        if os.path.isfile(self.precalc_path + "UMCdb_A_gcs.parquet"):
            return pl.scan_parquet(self.precalc_path + "UMCdb_A_gcs.parquet")

        data = data.sort(self.index_cols).select(
            [self.icu_stay_id_col, self.timeseries_time_col, "valueid", "itemid"]
        )

        data_eye = (
            data.filter(pl.col("itemid").is_in([6732, 13077, 14470, 16628, 19635, 19638]))
            .with_columns(
                pl.when(pl.col("itemid") == 6732)
                .then(5 - pl.col("valueid"))
                .when(pl.col("itemid").is_in([14470, 16628, 19635]))
                .then(pl.col("valueid") - 4)
                .when(pl.col("itemid") == 19638)
                .then(pl.col("valueid") - 8)
                .alias("eyes_score"),
            )
            .drop(["itemid", "valueid"])
        )

        data_motor = (
            data.filter(pl.col("itemid").is_in([6734, 13072, 14476, 16634, 19636, 19639]))
            .with_columns(
                "valueid",
                pl.when(pl.col("itemid") == 6734)
                .then(5 - pl.col("valueid"))
                .when(pl.col("itemid").is_in([14476, 16634, 19636]))
                .then(pl.col("valueid") - 6)
                .when(pl.col("itemid") == 19639)
                .then(pl.col("valueid") - 12)
                .alias("motor_score"),
            )
            .drop(["itemid", "valueid"])
        )

        data_verbal = (
            data.filter(pl.col("itemid").is_in([6735, 13066, 14482, 16640, 19637, 19640]))
            .with_columns(
                "valueid",
                pl.when(pl.col("itemid") == 6735)
                .then(6 - pl.col("valueid"))
                .when(pl.col("itemid").is_in([14482, 16640, 19637]))
                .then(pl.col("valueid") - 5)
                .when(pl.col("itemid") == 19640)
                .then(pl.col("valueid") - 15)
                .alias("verbal_score"),
            )
            .drop(["itemid", "valueid"])
        )

        data_gcs = (
            data_eye.join(data_motor, on=self.index_cols)
            .join(data_verbal, on=self.index_cols)
            .collect(streaming=True)
        )
        data_gcs = data_gcs.with_columns(
            (
                data_gcs.select(["eyes_score", "motor_score", "verbal_score"]).sum_horizontal(
                    ignore_nulls=False
                )
            ).alias("gcs_score"),
        )

        data_gcs.write_parquet(self.precalc_path + "UMCdb_A_gcs.parquet")

        return data_gcs.lazy()

    # endregion

    # region medication
    # Extract medication information from the drugitems.csv file
    def extract_medications(self) -> pl.LazyFrame:
        print("UMCdb   - Extracting medications...")

        umcdb_medication_mapping = self.helpers.load_many_to_many_to_one_mapping(
            self.mapping_path + "MEDICATIONS.yaml", "amsterdam"
        )
        intimes = (
            pl.scan_csv(self.admissions_path)
            .select(["admissionid", "admittedat", "dischargedat"])
            .rename(
                {
                    "admissionid": self.icu_stay_id_col,
                    "admittedat": "intime",
                    "dischargedat": "outtime",
                }
            )
        )

        return (
            pl.scan_csv(self.drugitems_path)
            .select(["admissionid", "item", "start", "stop", "administered", "administeredunit"])
            .rename(
                {
                    "admissionid": self.icu_stay_id_col,
                    "item": self.drug_name_col,
                    "start": self.drug_start_col,
                    "stop": self.drug_end_col,
                }
            )
            .join(intimes, on=self.icu_stay_id_col)
            # Keep only timepoints within timeframe of ICU stay + PRE_ICU_TIMESERIES_DAYS_CUTOFF
            .filter(
                (pl.col(self.drug_start_col) < pl.col("outtime"))
                & (
                    pl.col(self.drug_end_col)
                    > (
                        pl.col("intime")
                        - pl.duration(days=self.PRE_ICU_TIMESERIES_DAYS_CUTOFF).truediv(
                            pl.duration(milliseconds=1)
                        )
                    )
                )
            )
            .with_columns(
                pl.duration(milliseconds=(pl.col(self.drug_start_col) - pl.col("intime")))
                .dt.total_seconds()
                .cast(float)
                .alias(self.drug_start_col),
                pl.duration(milliseconds=(pl.col(self.drug_end_col) - pl.col("intime")))
                .dt.total_seconds()
                .cast(float)
                .alias(self.drug_end_col),
                # Replace drug names with standardized ingredient names
                pl.col(self.drug_name_col)
                .replace(umcdb_medication_mapping, default=None)
                .alias(self.drug_ingredient_col),
                # Format drug amount
                pl.col("administered").cast(float).alias(self.drug_amount_col),
                # Convert administered unit to enum
                pl.col("administeredunit")
                # .replace(self.DRUG_UNIT_MAPPING)
                # .cast(self.drug_unit_dtype)
                .alias(self.drug_unit_col),
            )
            # Remove duplicate rows
            .unique()
            # Remove rows with empty lab names
            .filter(pl.col(self.drug_start_col).is_not_null())
            # Remove rows with empty lab results
            .filter(pl.col(self.drug_name_col).is_not_null() & (pl.col(self.drug_name_col) != ""))
            .drop(["intime", "outtime", "administered", "administeredunit"])
        )

    # endregion
