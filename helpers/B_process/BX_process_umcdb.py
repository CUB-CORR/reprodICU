# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script processes the UMCdb data and stores it in a structured format for further
# processing and harmonization.


import numpy as np
import pandas as pd
import polars as pl
import os

from helpers.A_extract.AX_extract_umcdb import UMCdbExtractor
from helpers.helper import GlobalHelpers
from helpers.helper_unit_conversions import UnitConverter


class UMCdbProcessor(UMCdbExtractor):
    def __init__(self, paths):
        super().__init__(paths)
        self.path = paths.umcdb_source_path
        self.helpers = GlobalHelpers()
        self.convert = UMCdbConverter()
        self.icu_stay_id = self.extract_patient_information().select(
            [self.icu_stay_id_col, self.hospital_stay_id_col, self.person_id_col]
        )
        self.icu_length_of_stay = self.extract_patient_information().select(
            [self.icu_stay_id_col, self.icu_length_of_stay_col]
        )
        self.index_cols = [self.icu_stay_id_col, self.timeseries_time_col]

    # region time series
    # Processes and combines the time series data of the eICU dataset.
    def process_timeseries(self):
        # Load the time series data.
        print("UMCdb   - Loading time series data...")

        ts_numeric = self._process_timeseries_numeric()
        ts_listitems = pl.LazyFrame()  # self._process_timeseries_listitems()

        return pl.concat(
            [ts_numeric, ts_listitems],
            how="diagonal_relaxed",
        )

    def _process_timeseries_numeric(self) -> pl.LazyFrame:
        """
        Process the numeric timeseries data of the UMCdb dataset.
        """

        if os.path.isfile(self.precalc_path + "UMCdb_B_ts_numeric.parquet"):
            # Load the preprocessed data
            return pl.scan_parquet(self.precalc_path + "UMCdb_B_ts_numeric.parquet")

        print("UMCdb   - Processing numeric time series data...")

        # Process vitals data
        ts_numeric = (
            self.extract_timeseries_numericitems()
            # Convert the lab values to the correct units
            .pipe(self.convert._convert_lab_values, labelcol="item", valuecol="value")
            # Pivot the vitals data
            .collect(streaming=True).pivot(
                on="item",
                index=self.index_cols,
                values="value",
                aggregate_function="mean",  # NOTE: mean is used here -> check if this is sensible
            )
        )

        # Drop empty rows
        droplist = list(set(ts_numeric.collect_schema().names()) - set(self.index_cols))
        ts_numeric = (
            ts_numeric.pipe(self.helpers.dropna, subset=droplist, how="all").lazy().unique()
        )

        # Save the preprocessed data
        ts_numeric.sink_parquet(self.precalc_path + "UMCdb_B_ts_numeric.parquet")

        return ts_numeric

    def _process_timeseries_listitems(self) -> pl.LazyFrame:
        """
        Process the listitems timeseries data of the UMCdb dataset.
        """

        if os.path.isfile(self.precalc_path + "UMCdb_B_ts_listitems.parquet"):
            # Load the preprocessed data
            return pl.scan_parquet(self.precalc_path + "UMCdb_B_ts_listitems.parquet")

        print("UMCdb   - Processing numeric time series data...")

        # Process vitals data
        ts_listitems = (
            self.extract_timeseries_listitems()
            # Convert the lab values to the correct units
            .pipe(self.convert._convert_lab_values, labelcol="item", valuecol="value")
            # Pivot the vitals data
            .collect(streaming=True).pivot(
                on="item",
                index=self.index_cols,
                values="value",
                aggregate_function="mean",  # NOTE: mean is used here -> check if this is sensible
            )
        )

        # Drop empty rows
        droplist = list(set(ts_listitems.collect_schema().names()) - set(self.index_cols))
        ts_listitems = (
            ts_listitems.pipe(self.helpers.dropna, subset=droplist, how="all").lazy().unique()
        )

        # Save the preprocessed data
        ts_listitems.sink_parquet(self.precalc_path + "UMCdb_B_ts_listitems.parquet")

        return ts_listitems

    # endregion


# region convert
class UMCdbConverter(UnitConverter):
    def __init__(self):
        super().__init__()

    # Convert the lab values of the eICU dataset.
    def _convert_lab_values(
        self, data, labelcol: str = "variableid", valuecol: str = "value"
    ) -> pl.LazyFrame:
        """
        Convert the lab values of the UMCdb dataset.
        """

        # Convert the lab values to the correct units.
        (
            data.pipe(
                self.convert_ratio_to_percentage,
                itemid="hematocrit",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_ratio_to_percentage,
                itemid="saO2",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_creatinine_mmol_L_to_mg_dL,
                itemid="creatinine",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_creatinine_mmol_L_to_mg_dL,
                itemid="urine_creatinine",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_cholesterol_mmol_L_to_mg_dL,
                itemid="cholesterol_HDL",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_cholesterol_mmol_L_to_mg_dL,
                itemid="cholesterol_LDL",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_cholesterol_mmol_L_to_mg_dL,
                itemid="cholesterol_total",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_cortisol_nmol_L_to_ug_dL,
                itemid="cortisol",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_g_L_to_mg_dL,
                itemid="fibriogen",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_glucose_mmol_L_to_mg_dL,
                itemid="glucose",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_hemoglobin_mmol_L_to_g_dL,
                itemid="hemoglobin",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                # same conversion due to definition of MCHC
                self.convert_hemoglobin_mmol_L_to_g_dL,
                itemid="MCHC",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_glucose_mmol_L_to_mg_dL,
                itemid="glucose",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_triglycerides_mmol_L_to_mg_dL,
                itemid="triglycerides",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_ug_L_to_ng_L,
                itemid="troponin_T",
                labelcol=labelcol,
                valuecol=valuecol,
            )
        )

        return data


# endregion
