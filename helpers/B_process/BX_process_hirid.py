# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script processes the HiRID data and stores it in a structured format for further
# processing and harmonization.


import numpy as np
import pandas as pd
import polars as pl
import os

from helpers.A_extract.AX_extract_hirid import HiRIDExtractor
from helpers.helper import GlobalHelpers
from helpers.helper_conversions import UnitConverter


class HiRIDProcessor(HiRIDExtractor):
    def __init__(self, paths):
        super().__init__(paths)
        self.path = paths.hirid_source_path
        self.helpers = GlobalHelpers()
        self.convert = HiRIDConverter()
        self.index_cols = [self.icu_stay_id_col, self.timeseries_time_col]

    # region time series
    # Processes and combines the time series data of the eICU dataset.
    def process_timeseries(self) -> pl.LazyFrame:
        # Load the time series data
        print("HiRID   - Loading time series data...")

        if os.path.isfile(self.precalc_path + "HiRID_B_timeseries.parquet"):
            # Load the preprocessed data
            return pl.scan_parquet(self.precalc_path + "HiRID_B_timeseries.parquet")

        print("HiRID   - Processing time series data...")

        # COPY THE NEEDED DATAFRAMES FROM HiRIDExtractor.extract_timeseries() HERE
        observation_mapping = self.load_mapping(self.observation_mapping_path)
        admissiontime = (
            self._extract_admissions()
            .select([self.icu_stay_id_col, "admissiontime"])
            .cast({"admissiontime": str})
        )
        length_of_stay = self._extract_length_of_stay()

        # Create an empty DataFrame to store the timeseries data
        timeseries_processed = pl.LazyFrame()

        # Since each case has it's data in only one file, iterating over the files specifically allows
        # for a more efficient processing of the data.
        os_listdir_files = os.listdir(self.timeseries_path)
        counter, counter_max = 0, len(os_listdir_files)
        for file in os.listdir(self.timeseries_path):

            # Update the counter
            counter += 1
            print(f"Processing file {file}... \t{counter} / {counter_max}", end="\r")

            # Process timeseries data
            timeseries = (
                pl.scan_parquet(self.timeseries_path + file)
                .pipe(
                    self._extract_timeseries_helper,
                    admissiontime,
                    length_of_stay,
                    observation_mapping,
                )
                # Convert the lab values to the correct units
                .pipe(self.convert._convert_lab_values, labelcol="variableid", valuecol="value")
                # Pivot the timeseries data
                .collect(streaming=True)
                .pivot(
                    on="variableid",
                    index=self.index_cols,
                    values="value",
                    aggregate_function="mean",  # NOTE: mean is used here -> check if this is sensible
                )
            )

            # Drop empty rows
            droplist = list(set(timeseries.collect_schema().names()) - set(self.index_cols))
            timeseries = timeseries.pipe(self.helpers.dropna, subset=droplist, how="all").unique()

            # Append the data to the DataFrame
            timeseries_processed = pl.concat(
                [timeseries_processed, timeseries.lazy()], how="diagonal_relaxed"
            )

        # Save the preprocessed data
        timeseries_processed.sink_parquet(self.precalc_path + "HiRID_B_timeseries.parquet")

        return timeseries_processed

    # endregion


# region convert
class HiRIDConverter(UnitConverter):
    def __init__(self):
        super().__init__()

    # Convert the lab values of the eICU dataset.
    def _convert_lab_values(
        self, data, labelcol: str = "variableid", valuecol: str = "value"
    ) -> pl.LazyFrame:
        """
        Convert the lab values of the HiRID dataset.
        """

        # Convert the lab values to the correct units.
        (
            data.pipe(
                self.convert_absolute_count_to_relative,
                itemid="lymphocytes",
                total_itemid="leukocytes",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_blood_urea_nitrogen_from_urea,
                itemid_urea="urea",
                itemid_BUN="blood_urea_nitrogen",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_creatinine_umol_L_to_mg_dL,
                itemid="creatinine",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_g_L_to_mg_dL,
                itemid="fibrinogen",
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
                self.convert_g_L_to_g_dL,
                itemid="hemoglobin",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                # same conversion due to definition of MCHC
                self.convert_g_L_to_g_dL,
                itemid="MCHC",
                labelcol=labelcol,
                valuecol=valuecol,
            )
        )

        return data


# endregion
