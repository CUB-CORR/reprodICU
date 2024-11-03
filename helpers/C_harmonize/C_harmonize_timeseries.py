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


class TimeseriesHarmonizer(GlobalVars):
    def __init__(self, paths, datasets: list, DEMO=False):
        super().__init__(paths)
        self.eicu = EICUProcessor(paths, DEMO)
        self.hirid = HiRIDProcessor(paths)
        self.mimic3 = MIMIC3Processor(paths, DEMO)
        self.mimic4 = MIMIC4Processor(paths, DEMO)
        self.sicdb = SICdbProcessor(paths)
        self.umcdb = UMCdbProcessor(paths)
        self.paths = paths
        self.datasets = datasets
        self.helpers = GlobalHelpers()
        self.index_cols = [
            self.global_icu_stay_id_col,
            self.timeseries_time_col,
        ]
        self.save_path = (
            self.paths.reprodICU_files_path
            if not DEMO
            else self.paths.reprodICU_demo_files_path
        )

    # region harmonize/split
    # Split the timeseries data into vitals, labs, resp and inout
    def harmonize_split_timeseries(self, save_to_default=True) -> None:
        """
        Splits the timeseries data into vitals, labs, resp and inout
        """

        if self.datasets == []:
            raise ValueError("No datasets to harmonize the timeseries from.")

        vital_prms = pl.Series([*self.index_cols, *self.relevant_vital_values])
        labs_prms = pl.Series([*self.index_cols, *self.relevant_lab_values])
        resp_prms = pl.Series(
            [*self.index_cols, *self.relevant_respiratory_values]
        )
        inout_prms = pl.Series(
            [*self.index_cols, *self.relevant_intakeoutput_values]
        )

        # Harmonize the timeseries
        timeseries_vitals = []
        timeseries_labs = []
        timeseries_resp = []
        timeseries_inout = []

        if "eICU" in self.datasets:
            eicu_timeseries = self.eicu.process_timeseries().pipe(
                self._concat_helper, "eicu-"
            )
            eicu_timeseries_labs = self.eicu.process_timeseries_lab().pipe(
                self._concat_helper, "eicu-"
            )

            eicu_ts_names = eicu_timeseries.collect_schema().names()
            eicu_vitals = vital_prms.filter(vital_prms.is_in(eicu_ts_names))
            eicu_resp = resp_prms.filter(resp_prms.is_in(eicu_ts_names))

            eicu_ts_lab_names = eicu_timeseries_labs.collect_schema().names()
            eicu_labs = labs_prms.filter(labs_prms.is_in(eicu_ts_lab_names))

            # TODO: in/out calculation including medication
            eicu_inout = inout_prms.filter(inout_prms.is_in(eicu_ts_names))

            timeseries_vitals.append(eicu_timeseries.select(*eicu_vitals))
            timeseries_resp.append(eicu_timeseries.select(*eicu_resp))
            timeseries_labs.append(eicu_timeseries_labs.select(*eicu_labs))
            timeseries_inout.append(eicu_timeseries.select(*eicu_inout))

        if "HiRID" in self.datasets:
            hirid_timeseries = self.hirid.process_timeseries().pipe(
                self._concat_helper, "hirid-"
            )

            hirid_ts_names = hirid_timeseries.collect_schema().names()
            hirid_vitals = vital_prms.filter(vital_prms.is_in(hirid_ts_names))
            hirid_resp = resp_prms.filter(resp_prms.is_in(hirid_ts_names))
            hirid_labs = labs_prms.filter(labs_prms.is_in(hirid_ts_names))
            hirid_inout = inout_prms.filter(inout_prms.is_in(hirid_ts_names))

            timeseries_vitals.append(hirid_timeseries.select(*hirid_vitals))
            timeseries_resp.append(hirid_timeseries.select(*hirid_resp))
            timeseries_labs.append(hirid_timeseries.select(*hirid_labs))
            timeseries_inout.append(hirid_timeseries.select(*hirid_inout))

        if "MIMIC3" in self.datasets:
            mimic3_timeseries = self.mimic3.process_timeseries().pipe(
                self._concat_helper, "mimic3-"
            )

            mimic3_ts_names = mimic3_timeseries.collect_schema().names()
            mimic3_vitals = vital_prms.filter(vital_prms.is_in(mimic3_ts_names))
            mimic3_resp = resp_prms.filter(resp_prms.is_in(mimic3_ts_names))
            mimic3_labs = labs_prms.filter(labs_prms.is_in(mimic3_ts_names))
            mimic3_inout = inout_prms.filter(inout_prms.is_in(mimic3_ts_names))

            timeseries_vitals.append(mimic3_timeseries.select(*mimic3_vitals))
            timeseries_resp.append(mimic3_timeseries.select(*mimic3_resp))
            timeseries_labs.append(mimic3_timeseries.select(*mimic3_labs))
            timeseries_inout.append(mimic3_timeseries.select(*mimic3_inout))

        if "MIMIC4" in self.datasets:
            mimic4_timeseries = self.mimic4.process_timeseries().pipe(
                self._concat_helper, "mimic4-"
            )

            mimic4_ts_names = mimic4_timeseries.collect_schema().names()
            mimic4_vitals = vital_prms.filter(vital_prms.is_in(mimic4_ts_names))
            mimic4_resp = resp_prms.filter(resp_prms.is_in(mimic4_ts_names))
            mimic4_labs = labs_prms.filter(labs_prms.is_in(mimic4_ts_names))
            mimic4_inout = inout_prms.filter(inout_prms.is_in(mimic4_ts_names))

            timeseries_vitals.append(mimic4_timeseries.select(*mimic4_vitals))
            timeseries_resp.append(mimic4_timeseries.select(*mimic4_resp))
            timeseries_labs.append(mimic4_timeseries.select(*mimic4_labs))
            timeseries_inout.append(mimic4_timeseries.select(*mimic4_inout))

        if "SICdb" in self.datasets:
            sicdb_timeseries = self.sicdb.process_timeseries_data_float().pipe(
                self._concat_helper, "sicdb-"
            )
            sicdb_timeseries_lab = (
                self.sicdb.process_timeseries_data_labs().pipe(
                    self._concat_helper, "sicdb-"
                )
            )

            sicdb_ts_names = sicdb_timeseries.collect_schema().names()
            sicdb_vitals = vital_prms.filter(vital_prms.is_in(sicdb_ts_names))
            sicdb_resp = resp_prms.filter(resp_prms.is_in(sicdb_ts_names))
            sicdb_inout = inout_prms.filter(inout_prms.is_in(sicdb_ts_names))

            sicdb_ts_lab_names = sicdb_timeseries_lab.collect_schema().names()
            sicdb_labs = labs_prms.filter(labs_prms.is_in(sicdb_ts_lab_names))

            timeseries_vitals.append(sicdb_timeseries.select(*sicdb_vitals))
            timeseries_resp.append(sicdb_timeseries.select(*sicdb_resp))
            timeseries_labs.append(sicdb_timeseries_lab.select(*sicdb_labs))
            timeseries_inout.append(sicdb_timeseries.select(*sicdb_inout))

        if "UMCdb" in self.datasets:
            umcdb_timeseries = self.umcdb.process_timeseries().pipe(
                self._concat_helper, "umcdb-"
            )

            umcdb_ts_names = umcdb_timeseries.collect_schema().names()
            umcdb_vitals = vital_prms.filter(vital_prms.is_in(umcdb_ts_names))
            umcdb_resp = resp_prms.filter(resp_prms.is_in(umcdb_ts_names))
            umcdb_labs = labs_prms.filter(labs_prms.is_in(umcdb_ts_names))
            umcdb_inout = inout_prms.filter(inout_prms.is_in(umcdb_ts_names))

            timeseries_vitals.append(umcdb_timeseries.select(*umcdb_vitals))
            timeseries_resp.append(umcdb_timeseries.select(*umcdb_resp))
            timeseries_labs.append(umcdb_timeseries.select(*umcdb_labs))
            timeseries_inout.append(umcdb_timeseries.select(*umcdb_inout))

        # Combine the timeseries data of the datasets
        vitals = pl.concat(timeseries_vitals, how="diagonal_relaxed")
        vitals_cols = vitals.collect_schema().names()
        vitals_cols_not_index = list(set(vitals_cols) - set(self.index_cols))
        vitals = (
            vitals.pipe(
                self.helpers.dropna,
                "all",
                vitals_cols_not_index,
            )
            .cast(
                {  # Convert all columns to float
                    self.global_icu_stay_id_col: str,
                    self.timeseries_time_col: float,
                    **{col: float for col in vitals_cols_not_index},
                }
            )
            .select([*self.index_cols, *sorted(vitals_cols_not_index)])
            .sort(self.index_cols)
            .unique(self.index_cols)
            .sort(self.index_cols)
        )

        labs = pl.concat(timeseries_labs, how="diagonal_relaxed")
        labs_cols = labs.collect_schema().names()
        labs_cols_not_index = list(set(labs_cols) - set(self.index_cols))
        labs = (
            labs.pipe(self.helpers.dropna, "all", labs_cols_not_index)
            .cast(
                {  # Convert all columns to float
                    self.global_icu_stay_id_col: str,
                    self.timeseries_time_col: float,
                    **{col: float for col in labs_cols_not_index},
                }
            )
            .select([*self.index_cols, *sorted(labs_cols_not_index)])
            .sort(self.index_cols)
            .unique(self.index_cols)
            .sort(self.index_cols)
        )

        resp = pl.concat(timeseries_resp, how="diagonal_relaxed")
        resp_cols = resp.collect_schema().names()
        resp_cols_not_index = list(set(resp_cols) - set(self.index_cols))
        resp = (
            resp.pipe(self.helpers.dropna, "all", resp_cols_not_index)
            .cast(
                {  # Convert all columns to float, except for oxygen_delivery_device
                    self.global_icu_stay_id_col: str,
                    self.timeseries_time_col: float,
                    **{
                        col: (float if col != "oxygen_delivery_device" else str)
                        for col in resp_cols_not_index
                    },
                }
            )
            .select([*self.index_cols, *sorted(resp_cols_not_index)])
            .sort(self.index_cols)
            .unique(self.index_cols)
            .sort(self.index_cols)
        )

        inout = pl.concat(timeseries_inout, how="diagonal_relaxed")
        inout_cols = inout.collect_schema().names()
        inout_cols_not_index = list(set(inout_cols) - set(self.index_cols))
        inout = (
            inout.pipe(self.helpers.dropna, "all", inout_cols_not_index)
            .cast(
                {  # Convert all columns to float
                    self.global_icu_stay_id_col: str,
                    self.timeseries_time_col: float,
                    **{col: float for col in inout_cols_not_index},
                }
            )
            .select([*self.index_cols, *sorted(inout_cols_not_index)])
            .sort(self.index_cols)
            .unique(self.index_cols)
            .sort(self.index_cols)
        )

        if save_to_default:
            print("reprodICU - Saving timeseries...")

            print("reprodICU - Saving vitals...")
            vitals.pipe(self._print_unique_cases, "vitals").pipe(
                self._fix_temperature_values
            ).sink_parquet(self.save_path + "timeseries_vitals.parquet")

            print("reprodICU - Saving labs...")
            labs.pipe(self._print_unique_cases, "labs").sink_parquet(
                self.save_path + "timeseries_labs.parquet"
            )

            print("reprodICU - Saving respiratory...")
            resp.pipe(self._print_unique_cases, "respiratory").sink_parquet(
                self.save_path + "timeseries_respiratory.parquet"
            )

            print("reprodICU - Saving intakeoutput...")
            inout.pipe(self._print_unique_cases, "inout").sink_parquet(
                self.save_path + "timeseries_intakeoutput.parquet"
            )

            return None

        return vitals, labs, resp  # , inout

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
        self, data: pl.LazyFrame, name: str
    ) -> pl.LazyFrame:
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

        return data

    # Fix Temperature values for accidental Fahrenheit values
    def _fix_temperature_values(self, data: pl.LazyFrame) -> pl.LazyFrame:
        return data.with_columns(
            pl.when(pl.col("Temperature").gt(60))
            .then(pl.col("Temperature").sub(32).mul(5).truediv(9))
            .otherwise(pl.col("Temperature"))
            .alias("Temperature")
        )
