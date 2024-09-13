# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script processes the SICdb data and stores it in a structured format for further
# processing and harmonization.


import numpy as np
import pandas as pd
import polars as pl
import os

from helpers.A_extract.AX_extract_sicdb import SICdbExtractor
from helpers.helper import GlobalHelpers
from helpers.helper_unit_conversions import UnitConverter


class SICdbProcessor(SICdbExtractor):
    def __init__(self, paths):
        super().__init__(paths)
        self.path = paths.sicdb_source_path
        self.helpers = GlobalHelpers()
        self.convert = SICdbConverter()
        self.icu_stay_id = self.extract_patient_information().select(
            [self.icu_stay_id_col, self.hospital_stay_id_col, self.person_id_col]
        )
        self.icu_length_of_stay = self.extract_patient_information().select(
            [self.icu_stay_id_col, self.icu_length_of_stay_col]
        )
        self.index_cols = [self.icu_stay_id_col, self.timeseries_time_col]

    # region timeseries
    # Processes and combines the time series data of the eICU dataset.
    def process_timeseries(self) -> pl.LazyFrame:
        # Load the time series data
        print("SICdb   - Loading time series data...")

        ts_float = self._process_timeseries_data_float()
        ts_labs = self._process_timeseries_data_labs()

        return pl.concat([ts_float, ts_labs], how="diagonal_relaxed")

    def _process_timeseries_data_float(self) -> pl.LazyFrame:
        if os.path.isfile(self.precalc_path + "SICdb_B_timeseries.parquet"):
            # Load the preprocessed data
            return pl.scan_parquet(self.precalc_path + "SICdb_B_timeseries.parquet")

        print("SICdb   - Processing time series data...")

        # Process timeseries data
        timeseries = (
            self.extract_timeseries()
            # Pivot the timeseries data
            .collect(streaming=True).pivot(
                on="DataID",
                index=self.index_cols,
                values="Val",
                aggregate_function="mean",  # NOTE: mean is used here -> check if this is sensible
            )
        )

        # Drop empty rows
        droplist = list(set(timeseries.collect_schema().names()) - set(self.index_cols))
        timeseries = (
            timeseries.pipe(self.helpers.dropna, subset=droplist, how="all").lazy().unique()
        )

        # Save the preprocessed data
        timeseries.sink_parquet(self.precalc_path + "SICdb_B_timeseries.parquet")

        return timeseries

    # endregion

    # region lab values
    def _process_timeseries_data_labs(self) -> pl.LazyFrame:
        if os.path.isfile(self.precalc_path + "SICdb_B_laboratory.parquet"):
            # Load the preprocessed data
            return pl.scan_parquet(self.precalc_path + "SICdb_B_laboratory.parquet")

        print("SICdb   - Processing laboratory data...")

        # Process timeseries data
        timeseries = (
            self.extract_laboratory_timeseries()
            # Convert the lab values to the correct units
            .pipe(
                self.convert._convert_lab_values,
                labelcol="LaboratoryID",
                valuecol="LaboratoryValue",
            )
            # Pivot the timeseries data
            .collect(streaming=True).pivot(
                on="LaboratoryID",
                index=self.index_cols,
                values="LaboratoryValue",
                aggregate_function="mean",  # NOTE: mean is used here -> check if this is sensible
            )
        )

        # Drop empty rows
        droplist = list(set(timeseries.collect_schema().names()) - set(self.index_cols))
        timeseries = (
            timeseries.pipe(self.helpers.dropna, subset=droplist, how="all").lazy().unique()
        )

        # Save the preprocessed data
        timeseries.sink_parquet(self.precalc_path + "SICdb_B_timeseries.parquet")

        return timeseries

    # endregion


# region convert
class SICdbConverter(UnitConverter):
    def __init__(self):
        super().__init__()

    # Convert the lab values of the eICU dataset.
    def _convert_lab_values(
        self, data, labelcol: str = "LaboratoryID", valuecol: str = "LaboratoryValue"
    ) -> pl.LazyFrame:
        """
        Convert the lab values of the SICdb dataset.
        """

        # Convert the lab values to the correct units.
        (
            data.pipe(
                self.convert_blood_urea_nitrogen_from_urea,
                itemid_urea="urea",
                itemid_BUN="blood_urea_nitrogen",
                labelcol=labelcol,
                valuecol=valuecol,
            )
        )

        return data


# endregion
