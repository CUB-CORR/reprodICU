# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script combines the preprocessed patient information from the differet
# databases into one common table

import polars as pl
import os

from helpers.B_process.B_process_eicu import EICUProcessor
from helpers.B_process.BX_process_hirid import HiRIDProcessor
from helpers.B_process.B_process_mimic3 import MIMIC3Processor
from helpers.B_process.B_process_mimic4 import MIMIC4Processor
from helpers.B_process.BX_process_sicdb import SICdbProcessor
from helpers.B_process.BX_process_umcdb import UMCdbProcessor
from helpers.helper import GlobalVars
from helpers.helper import GlobalHelpers
from helpers.helper_conversions import GCSCombiner


class TimeseriesHarmonizer(GlobalVars):
    def __init__(self, paths, datasets: list, DEMO=False):
        super().__init__(paths)
        self.eicu = EICUProcessor(paths, DEMO)
        self.hirid = HiRIDProcessor(paths)
        self.mimic3 = MIMIC3Processor(paths, DEMO)
        self.mimic4 = MIMIC4Processor(paths, DEMO)
        self.sicdb = SICdbProcessor(paths)
        self.umcdb = UMCdbProcessor(paths)
        self.datasets = datasets
        self.helpers = GlobalHelpers()

    # region combine
    # Combine the timeseries data of the datasets
    def harmonize_timeseries(self) -> pl.LazyFrame:

        if self.datasets == []:
            raise ValueError("No datasets to harmonize the timeseries from.")

        # Harmonize the timeseries
        timeseries_datasets = []

        if "eICU" in self.datasets:
            eicu_timeseries = self.eicu.process_timeseries().pipe(
                self._concat_helper, "eicu-"
            )
            self._print_unique_cases(
                eicu_timeseries, "eICU", self.global_icu_stay_id_col
            )
            timeseries_datasets.append(eicu_timeseries)

        if "HiRID" in self.datasets:
            hirid_timeseries = self.hirid.process_timeseries().pipe(
                self._concat_helper, "hirid-"
            )
            self._print_unique_cases(
                hirid_timeseries, "HiRID", self.global_icu_stay_id_col
            )
            timeseries_datasets.append(hirid_timeseries)

        if "MIMIC3" in self.datasets:
            mimic3_timeseries = self.mimic3.process_timeseries().pipe(
                self._concat_helper, "mimic3-"
            )
            self._print_unique_cases(
                mimic3_timeseries, "MIMIC3", self.global_icu_stay_id_col
            )
            timeseries_datasets.append(mimic3_timeseries)

        if "MIMIC4" in self.datasets:
            mimic4_timeseries = self.mimic4.process_timeseries().pipe(
                self._concat_helper, "mimic4-"
            )
            self._print_unique_cases(
                mimic4_timeseries, "MIMIC4", self.global_icu_stay_id_col
            )
            timeseries_datasets.append(mimic4_timeseries)

        if "SICdb" in self.datasets:
            sicdb_timeseries = self.sicdb.process_timeseries().pipe(
                self._concat_helper, "sicdb-"
            )
            self._print_unique_cases(
                sicdb_timeseries, "SICdb", self.global_icu_stay_id_col
            )
            timeseries_datasets.append(sicdb_timeseries)

        if "UMCdb" in self.datasets:
            umcdb_timeseries = self.umcdb.process_timeseries().pipe(
                self._concat_helper, "umcdb-"
            )
            self._print_unique_cases(
                umcdb_timeseries, "UMCdb", self.global_icu_stay_id_col
            )
            timeseries_datasets.append(umcdb_timeseries)

        # Combine the timeseries data of the datasets
        return pl.concat(
            timeseries_datasets,
            how="diagonal_relaxed",
        )

    # endregion

    # region split
    # Split the timeseries data into vitals, labs, resp and inout
    def split_timeseries(self, path, save_to_default=True) -> None:
        """
        Splits the timeseries data into vitals, labs, resp and inout
        """
        index_cols = [self.global_icu_stay_id_col, self.timeseries_time_col]
        vitals_params = pl.Series([*index_cols, *self.relevant_vital_values])
        labs_params = pl.Series([*index_cols, *self.relevant_lab_values])
        resp_params = pl.Series(
            [*index_cols, *self.relevant_respiratory_values]
        )
        inout_params = pl.Series(
            [*index_cols, *self.relevant_intakeoutput_values]
        )

        # Load the timeseries data
        if not os.path.isfile(path):
            timeseries_all = self.harmonize_timeseries()
        else:
            timeseries_all = pl.read_parquet(path)

        timeseries_all_cols = pl.Series(timeseries_all.collect_schema().names())

        # Split off vitals
        vitals_cols = timeseries_all_cols.filter(
            timeseries_all_cols.is_in(vitals_params)
        )
        vitals_cols_not_index = list(set(vitals_cols) - set(index_cols))
        vitals = (
            timeseries_all.select(*vitals_cols)
            .pipe(self.helpers.dropna, subset=vitals_cols_not_index, how="all")
            .cast(
                {  # Convert all columns to float
                    self.global_icu_stay_id_col: str,
                    self.timeseries_time_col: float,
                    **{col: float for col in vitals_cols_not_index},
                }
            )
            .select([*index_cols, *sorted(vitals_cols_not_index)])
            .sort(index_cols)
        )

        # Split off labs
        labs_cols = timeseries_all_cols.filter(
            timeseries_all_cols.is_in(labs_params)
        )
        labs_cols_not_index = list(set(labs_cols) - set(index_cols))
        labs = (
            timeseries_all.select(*labs_cols)
            .pipe(self.helpers.dropna, subset=labs_cols_not_index, how="all")
            .cast(
                {  # Convert all columns to float
                    self.global_icu_stay_id_col: str,
                    self.timeseries_time_col: float,
                    **{col: float for col in labs_cols_not_index},
                }
                # ).pipe(
                #     self.harmonize_lab_values
            )
            .select([*index_cols, *sorted(labs_cols_not_index)])
            .sort(index_cols)
        )

        # Split off respitory
        resp_cols = timeseries_all_cols.filter(
            timeseries_all_cols.is_in(resp_params)
        )
        resp_cols_not_index = list(set(resp_cols) - set(index_cols))
        print(
            {col: float for col in resp_cols_not_index}.update(
                {"oxygen_delivery_device": str}
            )
        )
        resp = (
            timeseries_all.select(*resp_cols)
            .pipe(self.helpers.dropna, subset=resp_cols_not_index, how="all")
            .cast(
                {  # Convert all columns to float, except for oxygen_delivery_device
                    self.global_icu_stay_id_col: str,
                    self.timeseries_time_col: float,
                    **(
                        {
                            col: (
                                float
                                if col != "oxygen_delivery_device"
                                else str
                            )
                            for col in resp_cols_not_index
                        }
                    ),
                }
            )
            .select([*index_cols, *sorted(resp_cols_not_index)])
            .sort(index_cols)
        )

        # Split off inout
        inout_cols = timeseries_all_cols.filter(
            timeseries_all_cols.is_in(inout_params)
        )
        inout_cols_not_index = list(set(inout_cols) - set(index_cols))
        inout = (
            timeseries_all.select(*inout_cols)
            .pipe(self.helpers.dropna, subset=inout_cols_not_index, how="all")
            .cast(
                {  # Convert all columns to float
                    self.global_icu_stay_id_col: str,
                    self.timeseries_time_col: float,
                    **{col: float for col in inout_cols_not_index},
                }
            )
            .select([*index_cols, *sorted(inout_cols_not_index)])
            .sort(index_cols)
        )

        if save_to_default:
            vitals.sink_parquet("reprodICU_files/timeseries_vitals.parquet")
            labs.sink_parquet("reprodICU_files/timeseries_labs.parquet")
            resp.sink_parquet("reprodICU_files/timeseries_respiratory.parquet")
            inout.sink_parquet(
                "reprodICU_files/timeseries_intakeoutput.parquet"
            )

            return None

        return vitals, labs, resp, inout

    # endregion

    # region helpers
    # Concatenate the IDs with the database name to create a global ID
    def _concat_helper(self, data: pl.LazyFrame, name: str) -> pl.LazyFrame:
        return data.with_columns(
            pl.concat_str([pl.lit(name), pl.col(self.icu_stay_id_col)]).alias(
                self.global_icu_stay_id_col
            )
        )

    # Print the number of unique cases in the timeseries data
    def _print_unique_cases(
        self, data: pl.LazyFrame, name: str, count_col: str
    ) -> None:
        unique_count = (
            data.select(self.global_icu_stay_id_col)
            .unique()
            .count()
            .collect(streaming=True)
            .to_numpy()[0][0]
        )
        print(
            f"reprodICU - {unique_count:6.0f} unique cases with timeseries data in {name}."
        )
