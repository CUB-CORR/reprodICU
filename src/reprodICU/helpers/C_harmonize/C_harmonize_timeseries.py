# Author: Finn Fassbender
# Last modified: 2025-10-30

# Description: This script combines the preprocessed patient information from the differet
# databases into one common table

import polars as pl

from ..B_process.B_process_eicu import EICUProcessor
from ..B_process.B_process_hirid import HiRIDProcessor
from ..B_process.B_process_mimic3 import MIMIC3Processor
from ..B_process.B_process_mimic4 import MIMIC4Processor
from ..B_process.B_process_nwicu import NWICUProcessor
from ..B_process.B_process_sicdb import SICdbProcessor
from ..B_process.B_process_umcdb import UMCdbProcessor
from ..helper import GlobalHelpers, GlobalVars
from ..helper_conversions import UnitConverter


class TimeseriesHarmonizer(GlobalVars):
    def __init__(self, paths, datasets: list, DEMO=False):
        """
        Initializes the TimeseriesHarmonizer class with the given paths and datasets.

        Args:
            paths (str): The file paths required for data extraction.
            datasets (list): A list of datasets to be harmonized.
            DEMO (bool, optional): A flag indicating whether to use demo data. Defaults to False.
        """
        super().__init__(paths)
        self.paths = paths
        self.datasets = datasets
        self.helpers = GlobalHelpers()
        self.convert = UnitConverter()

        if "eICU" in self.datasets:
            self.eicu = EICUProcessor(paths, DEMO)
        if "HiRID" in self.datasets:
            self.hirid = HiRIDProcessor(paths)
        if "MIMIC3" in self.datasets:
            self.mimic3 = MIMIC3Processor(paths, DEMO)
        if "MIMIC4" in self.datasets:
            self.mimic4 = MIMIC4Processor(paths, DEMO)
        if "NWICU" in self.datasets:
            self.nwicu = NWICUProcessor(paths)
        if "SICdb" in self.datasets:
            self.sicdb = SICdbProcessor(paths)
        if "UMCdb" in self.datasets:
            self.umcdb = UMCdbProcessor(paths)

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
    def harmonize_split_timeseries(
        self, timeseries=[], save_to_default=True
    ) -> None:
        """
        Split and harmonize timeseries data into vitals, labs, respiratory, and intake/output.

        Steps:
            1. Validate non-empty datasets and timeseries list.
            2. Create filter series for each timeseries category (vitals, labs, respiratory, I/O).
            3. For each dataset: process timeseries and create global identifiers.
            4. Concatenate data per category using diagonal-relaxed join.
            5. Clean and cast each category (fix temps, remove duplicates).
            6. Save to parquet files or return as tuple of DataFrames.

        Returns:
            None if save_to_default=True; otherwise tuple (vitals, labs, resp, inout) of pl.DataFrame.
        """
        if self.datasets == []:
            raise ValueError("No datasets to harmonize the timeseries from.")
        if timeseries == []:
            raise ValueError("No timeseries selected.")

        # Create filter series for each timeseries category
        # fmt: off
        vital_prms = pl.Series([*self.index_cols, *self.relevant_vital_values])
        resp_prms = pl.Series([*self.index_cols, *self.relevant_respiratory_values])
        inout_prms = pl.Series([*self.index_cols, *self.relevant_intakeoutput_values])
        labs_prms = pl.Series([*self.index_cols, *self.relevant_lab_LOINC_components])
        extra_prms = pl.Series([*self.index_cols, *self.relevant_extracorporeal_values])
        # fmt: on

        # Harmonize the timeseries per category
        timeseries_vitals = []
        timeseries_labs = []
        timeseries_resp = []
        timeseries_inout = []
        timeseries_extra = []

        # region eICU
        if "eICU" in self.datasets:
            eicu_timeseries = self.eicu.process_timeseries().pipe(
                self._concat_helper, "eicu-"
            )
            eicu_timeseries_labs = self.eicu.process_timeseries_lab().pipe(
                self._concat_helper, "eicu-"
            )
            eicu_timeseries_inout = self.eicu.process_timeseries_inout().pipe(
                self._concat_helper, "eicu-"
            )

            eicu_ts_names = eicu_timeseries.collect_schema().names()
            eicu_vitals = vital_prms.filter(vital_prms.is_in(eicu_ts_names))
            eicu_resp = resp_prms.filter(resp_prms.is_in(eicu_ts_names))

            eicu_ts_lab_names = eicu_timeseries_labs.collect_schema().names()
            eicu_labs = labs_prms.filter(labs_prms.is_in(eicu_ts_lab_names))

            eicu_ts_io_names = eicu_timeseries_inout.collect_schema().names()
            eicu_inout = inout_prms.filter(inout_prms.is_in(eicu_ts_io_names))

            timeseries_vitals.append(eicu_timeseries.select(*eicu_vitals))
            timeseries_resp.append(eicu_timeseries.select(*eicu_resp))
            timeseries_labs.append(eicu_timeseries_labs.select(*eicu_labs))
            timeseries_inout.append(eicu_timeseries_inout.select(*eicu_inout))
        # endregion

        # region HiRID
        if "HiRID" in self.datasets:
            hirid_timeseries = self.hirid.process_timeseries().pipe(
                self._concat_helper, "hirid-"
            )
            hirid_timeseries_labs = self.hirid.process_timeseries_labs().pipe(
                self._concat_helper, "hirid-"
            )

            hirid_ts_names = hirid_timeseries.collect_schema().names()
            hirid_vitals = vital_prms.filter(vital_prms.is_in(hirid_ts_names))
            hirid_resp = resp_prms.filter(resp_prms.is_in(hirid_ts_names))
            hirid_inout = inout_prms.filter(inout_prms.is_in(hirid_ts_names))

            hirid_ts_lab_names = hirid_timeseries_labs.collect_schema().names()
            hirid_labs = labs_prms.filter(labs_prms.is_in(hirid_ts_lab_names))

            timeseries_vitals.append(hirid_timeseries.select(*hirid_vitals))
            timeseries_resp.append(hirid_timeseries.select(*hirid_resp))
            timeseries_labs.append(hirid_timeseries_labs.select(*hirid_labs))
            timeseries_inout.append(hirid_timeseries.select(*hirid_inout))
        # endregion

        # region MIMIC3
        if "MIMIC3" in self.datasets:
            mimic3_timeseries = self.mimic3.process_timeseries_vitals().pipe(
                self._concat_helper, "mimic3-"
            )
            mimic3_timeseries_labs = (
                self.mimic3.process_timeseries_labevents().pipe(
                    self._concat_helper, "mimic3-"
                )
            )
            mimic3_timeseries_inout = (
                self.mimic3.process_timeseries_inputoutput().pipe(
                    self._concat_helper, "mimic3-"
                )
            )

            mimic3_ts_names = mimic3_timeseries.collect_schema().names()
            mimic3_vitals = vital_prms.filter(vital_prms.is_in(mimic3_ts_names))
            mimic3_resp = resp_prms.filter(resp_prms.is_in(mimic3_ts_names))
            mimic3_extra = extra_prms.filter(extra_prms.is_in(mimic3_ts_names))

            mimic3_ts_lab_names = (
                mimic3_timeseries_labs.collect_schema().names()
            )
            mimic3_labs = labs_prms.filter(labs_prms.is_in(mimic3_ts_lab_names))

            mimic3_ts_io_names = (
                mimic3_timeseries_inout.collect_schema().names()
            )
            mimic3_inout = inout_prms.filter(
                inout_prms.is_in(mimic3_ts_io_names)
            )

            timeseries_vitals.append(mimic3_timeseries.select(*mimic3_vitals))
            timeseries_resp.append(mimic3_timeseries.select(*mimic3_resp))
            timeseries_labs.append(mimic3_timeseries_labs.select(*mimic3_labs))
            timeseries_inout.append(mimic3_timeseries_inout.select(*mimic3_inout)) # fmt: skip
            timeseries_extra.append(mimic3_timeseries.select(*mimic3_extra))
        # endregion

        # region MIMIC4
        if "MIMIC4" in self.datasets:
            mimic4_timeseries = self.mimic4.process_timeseries_vitals().pipe(
                self._concat_helper, "mimic4-"
            )
            mimic4_timeseries_labs = (
                self.mimic4.process_timeseries_labevents().pipe(
                    self._concat_helper, "mimic4-"
                )
            )
            mimic4_timeseries_inout = (
                self.mimic4.process_timeseries_inputoutput().pipe(
                    self._concat_helper, "mimic4-"
                )
            )

            mimic4_ts_names = mimic4_timeseries.collect_schema().names()
            mimic4_vitals = vital_prms.filter(vital_prms.is_in(mimic4_ts_names))
            mimic4_resp = resp_prms.filter(resp_prms.is_in(mimic4_ts_names))
            mimic4_extra = extra_prms.filter(extra_prms.is_in(mimic4_ts_names))

            mimic4_ts_lab_names = (
                mimic4_timeseries_labs.collect_schema().names()
            )
            mimic4_labs = labs_prms.filter(labs_prms.is_in(mimic4_ts_lab_names))

            mimic4_ts_io_names = (
                mimic4_timeseries_inout.collect_schema().names()
            )
            mimic4_inout = inout_prms.filter(
                inout_prms.is_in(mimic4_ts_io_names)
            )

            timeseries_vitals.append(mimic4_timeseries.select(*mimic4_vitals))
            timeseries_resp.append(mimic4_timeseries.select(*mimic4_resp))
            timeseries_labs.append(mimic4_timeseries_labs.select(*mimic4_labs))
            timeseries_inout.append(mimic4_timeseries_inout.select(*mimic4_inout)) # fmt: skip
            timeseries_extra.append(mimic4_timeseries.select(*mimic4_extra))
        # endregion

        # region NWICU
        if "NWICU" in self.datasets:
            nwicu_timeseries = self.nwicu.process_timeseries_vitals().pipe(
                self._concat_helper, "nwicu-"
            )
            nwicu_timeseries_labs = (
                self.nwicu.process_timeseries_labevents().pipe(
                    self._concat_helper, "nwicu-"
                )
            )

            nwicu_ts_names = nwicu_timeseries.collect_schema().names()
            nwicu_vitals = vital_prms.filter(vital_prms.is_in(nwicu_ts_names))

            nwicu_ts_lab_names = nwicu_timeseries_labs.collect_schema().names()
            nwicu_labs = labs_prms.filter(labs_prms.is_in(nwicu_ts_lab_names))

            timeseries_vitals.append(nwicu_timeseries.select(*nwicu_vitals))
            timeseries_labs.append(nwicu_timeseries_labs.select(*nwicu_labs))
        # endregion

        # region SICdb
        if "SICdb" in self.datasets:
            sicdb_timeseries = self.sicdb.process_timeseries_data_float().pipe(
                self._concat_helper, "sicdb-"
            )
            sicdb_timeseries_labs = (
                self.sicdb.process_timeseries_data_labs().pipe(
                    self._concat_helper, "sicdb-"
                )
            )

            sicdb_ts_names = sicdb_timeseries.collect_schema().names()
            sicdb_vitals = vital_prms.filter(vital_prms.is_in(sicdb_ts_names))
            sicdb_resp = resp_prms.filter(resp_prms.is_in(sicdb_ts_names))
            sicdb_inout = inout_prms.filter(inout_prms.is_in(sicdb_ts_names))
            sicdb_extra = extra_prms.filter(extra_prms.is_in(sicdb_ts_names))

            sicdb_ts_lab_names = sicdb_timeseries_labs.collect_schema().names()
            sicdb_labs = labs_prms.filter(labs_prms.is_in(sicdb_ts_lab_names))

            timeseries_vitals.append(sicdb_timeseries.select(*sicdb_vitals))
            timeseries_resp.append(sicdb_timeseries.select(*sicdb_resp))
            timeseries_labs.append(sicdb_timeseries_labs.select(*sicdb_labs))
            timeseries_inout.append(sicdb_timeseries.select(*sicdb_inout))
            timeseries_extra.append(sicdb_timeseries.select(*sicdb_extra))
        # endregion

        # region UMCdb
        if "UMCdb" in self.datasets:
            umcdb_timeseries = self.umcdb.process_timeseries().pipe(
                self._concat_helper, "umcdb-"
            )
            umcdb_timeseries_labs = self.umcdb._process_timeseries_labs().pipe(
                self._concat_helper, "umcdb-"
            )

            umcdb_ts_names = umcdb_timeseries.collect_schema().names()
            umcdb_vitals = vital_prms.filter(vital_prms.is_in(umcdb_ts_names))
            umcdb_resp = resp_prms.filter(resp_prms.is_in(umcdb_ts_names))
            umcdb_inout = inout_prms.filter(inout_prms.is_in(umcdb_ts_names))
            umcdb_extra = extra_prms.filter(extra_prms.is_in(umcdb_ts_names))

            umcdb_ts_lab_names = umcdb_timeseries_labs.collect_schema().names()
            umcdb_labs = labs_prms.filter(labs_prms.is_in(umcdb_ts_lab_names))

            timeseries_vitals.append(umcdb_timeseries.select(*umcdb_vitals))
            timeseries_resp.append(umcdb_timeseries.select(*umcdb_resp))
            timeseries_labs.append(umcdb_timeseries_labs.select(*umcdb_labs))
            timeseries_inout.append(umcdb_timeseries.select(*umcdb_inout))
            timeseries_extra.append(umcdb_timeseries.select(*umcdb_extra))
        # endregion

        # Concatenate the timeseries data for each category
        # region vitals
        vitals_strs = [
            "Heart rate rhythm",
            "Confusion Assessment Method",
        ]

        vitals = pl.LazyFrame()
        for ts_vitals in timeseries_vitals:
            vitals_cols = ts_vitals.collect_schema().names()
            vitals_cols_not_index = list(set(vitals_cols) - set(self.index_cols)) # fmt: skip
            ts_vitals = (
                # Drop rows with all NaN values in vitals columns
                ts_vitals.pipe(
                    self.helpers.dropna,
                    "all",
                    vitals_cols_not_index,
                    False,
                )
                # Convert columns to appropriate types
                .cast(
                    {
                        self.global_icu_stay_id_col: str,
                        self.timeseries_time_col: float,
                        **{
                            col: str if col in vitals_strs else float
                            for col in vitals_cols_not_index
                        },
                    }
                ).select(*self.index_cols, *sorted(vitals_cols_not_index))
                # assume uniqueness & sortedness (since we're just concatenating the data)
                # .unique(self.index_cols)
                # .sort(self.index_cols)
            )

            vitals = pl.concat([vitals, ts_vitals], how="diagonal_relaxed")

        vitals_cols = vitals.collect_schema().names()
        vitals = vitals.with_columns(
            # Sum Glasgow coma score components if total is missing
            pl.coalesce(
                (
                    pl.col("Glasgow coma score total")
                    if "Glasgow coma score total" in vitals_cols
                    else None
                ),
                pl.sum_horizontal(
                    [
                        pl.when(pl.col(col) == 0)
                        .then(None)
                        .otherwise(pl.col(col))
                        for col in [
                            "Glasgow coma score eye opening",
                            "Glasgow coma score motor",
                            "Glasgow coma score verbal",
                        ]
                    ],
                    ignore_nulls=False,
                ),
            ).alias("Glasgow coma score total"),
            # Fix Temperature once more if value appears to be in Fahrenheit
            pl.when(pl.col("Temperature").gt(60))
            .then(pl.col("Temperature").sub(32).mul(5).truediv(9))
            .otherwise(pl.col("Temperature"))
            .alias("Temperature"),
        )
        # endregion

        # region labs
        labs = (
            pl.concat(timeseries_labs, how="diagonal_relaxed")
            # Convert columns to appropriate types
            .cast(
                {
                    self.global_icu_stay_id_col: str,
                    self.timeseries_time_col: float,
                }
            )
            .select(
                *self.index_cols,
                pl.exclude(
                    *self.index_cols,
                    *self.conversion_lab_LOINC_components,
                ),
            )
            .pipe(
                self.convert._decode_lab_structs,
                cols_to_exclude=self.index_cols,
            )
            # assume uniqueness & sortedness (since we're just concatenating the data)
            # .unique(self.index_cols)
            # .sort(self.index_cols)
        )

        # ensure "/leukocytes" / "/erythrocytes" columns all are in range [0, 1]
        labs_cols = labs.collect_schema().names()
        labs = labs.with_columns(
            pl.col(col)
            .struct.with_fields(
                pl.when(pl.field("value") > 1)
                .then(pl.field("value").truediv(100))
                .otherwise(pl.field("value"))
                .alias("value")
            )
            .alias(col)
            for col in [
                "Basophils/leukocytes",
                "Eosinophils/leukocytes",
                "Lymphocytes/leukocytes",
                "Monocytes/leukocytes",
                "Neutrophils/leukocytes",
                "Neutrophils.band form/leukocytes",
                "Reticulocytes/Erythrocytes",
                "Neutrophils.segmented/leukocytes",
            ]
            if col in labs_cols
        )
        # endregion

        # region respiratory
        resp_strs = [
            "Oxygen delivery system",
            "Ventilation mode Ventilator",
            "Ventilator type",
        ]

        resp = pl.LazyFrame()
        for ts_resp in timeseries_resp:
            resp_cols = ts_resp.collect_schema().names()
            resp_cols_not_index = list(set(resp_cols) - set(self.index_cols))
            ts_resp = (
                # Drop rows with all NaN values in vitals columns
                ts_resp.pipe(
                    self.helpers.dropna,
                    "all",
                    resp_cols_not_index,
                    False,
                )
                # Convert columns to appropriate types
                .cast(
                    {
                        self.global_icu_stay_id_col: str,
                        self.timeseries_time_col: float,
                        **{
                            col: str if col in resp_strs else float
                            for col in resp_cols_not_index
                        },
                    },
                    # silently fail on invalid values (i.e. don't raise an error)
                    strict=False,
                ).select(*self.index_cols, *sorted(resp_cols_not_index))
                # assume uniqueness & sortedness (since we're just concatenating the data)
                # .unique(self.index_cols)
                # .sort(self.index_cols)
            )

            resp = pl.concat([resp, ts_resp], how="diagonal_relaxed")
        # endregion

        # region intakeoutput
        inout = pl.concat(timeseries_inout, how="diagonal_relaxed")
        inout_cols = inout.collect_schema().names()
        inout_cols_not_index = list(set(inout_cols) - set(self.index_cols))
        inout = (
            # Drop rows with all NaN values in vitals columns
            inout.pipe(
                self.helpers.dropna,
                "all",
                inout_cols_not_index,
                False,
            )
            # Convert columns to appropriate types
            .cast(
                {
                    self.global_icu_stay_id_col: str,
                    self.timeseries_time_col: float,
                    **{col: float for col in inout_cols_not_index},
                }
            ).select(*self.index_cols, *sorted(inout_cols_not_index))
            # assume uniqueness & sortedness (since we're just concatenating the data)
            # .unique(self.index_cols)
            # .sort(self.index_cols)
        )
        # endregion

        # region extracorporeal
        extracorporeal_strs = [
            "Continuous renal replacement therapy mode Renal replacement therapy circuit"
        ]

        extracorporeal = pl.concat(timeseries_extra, how="diagonal_relaxed")
        extracorporeal_cols = extracorporeal.collect_schema().names()
        extracorporeal_cols_not_index = list(set(extracorporeal_cols) - set(self.index_cols)) # fmt: skip
        extracorporeal = (
            extracorporeal.pipe(
                self.helpers.dropna,
                "all",
                extracorporeal_cols_not_index,
                False,
            )
            # Convert columns to appropriate types
            .cast(
                {
                    self.global_icu_stay_id_col: str,
                    self.timeseries_time_col: float,
                    **{
                        col: str if col in extracorporeal_strs else float
                        for col in extracorporeal_cols_not_index
                    },
                }
            ).select(*self.index_cols, *sorted(extracorporeal_cols_not_index))
            # assume uniqueness & sortedness (since we're just concatenating the data)
            # .unique(self.index_cols)
            # .sort(self.index_cols)
        )
        # endregion

        # region save
        if save_to_default:
            print("reprodICU - Saving timeseries...")

            if "vitals" in timeseries:
                print("reprodICU - Saving vitals...")
                vitals.pipe(self._print_unique_cases, "vitals").sink_parquet(
                    self.save_path + "timeseries_vitals.parquet"
                )

            if "labs" in timeseries:
                print("reprodICU - Saving labs...")

                labs.pipe(self._print_unique_cases, "labs").sink_parquet(
                    self.save_path + "timeseries_labs.parquet"
                )

            if "respiratory" in timeseries:
                print("reprodICU - Saving respiratory...")
                resp.pipe(self._print_unique_cases, "respiratory").sink_parquet(
                    self.save_path + "timeseries_respiratory.parquet"
                )

            if "inout" in timeseries:
                print("reprodICU - Saving intakeoutput...")
                inout.pipe(self._print_unique_cases, "inout").sink_parquet(
                    self.save_path + "timeseries_intakeoutput.parquet"
                )

            if "extracorporeal" in timeseries:
                print("reprodICU - Saving extracorporeal...")
                extracorporeal.pipe(
                    self._print_unique_cases, "extracorporeal"
                ).sink_parquet(
                    self.save_path + "timeseries_extracorporeal.parquet"
                )

            return None

        return vitals, labs, resp, inout, extracorporeal

    # endregion

    # region metadata
    # Remove the metadata columns from the timeseries data
    # i.e. remove the structs, keeping only the value field per column
    def remove_metadata(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """
        Removes metadata from timeseries data by flattening structured columns.

        Specifically, for columns with struct types, this method:
            1. Prefixes nested field names (except "value") with the parent column name.
            2. Unnests the struct, keeping the primary "value" field intact.
            3. Excludes columns matching metadata patterns (e.g., ending in "source" or "method").

        This approach is based on the GitHub comments by @daviewales here:
        https://github.com/pola-rs/polars/issues/7078#issuecomment-2258225305
        and has been modified for LazyFrames.

        Args:
            data (pl.LazyFrame): The input LazyFrame containing metadata.

        Returns:
            pl.LazyFrame: A LazyFrame with metadata removed and sorted based on {global_icu_stay_id_col} and {timeseries_time_col}.
        """

        def _prefix_field(field):
            return pl.col(field).name.map_fields(
                lambda x: f"{field}.{x}" if x != "value" else f"{field}"
            )

        def flatten(lf: pl.LazyFrame):
            cols = lf.collect_schema().names()
            dtyp = lf.collect_schema().dtypes()

            struct_cols = [
                col
                for col, dtype in zip(cols, dtyp)
                if type(dtype) is pl.Struct
            ]
            return lf.with_columns(*map(_prefix_field, struct_cols)).unnest(
                *struct_cols
            )

        return (
            data.pipe(flatten)
            .select(
                self.global_icu_stay_id_col,
                self.timeseries_time_col,
                pl.exclude(
                    "^.*(source|method)$",
                    self.global_icu_stay_id_col,
                    self.timeseries_time_col,
                ),
            )
            .sort(self.index_cols)
        )

    # endregion

    # region helpers
    # Concatenate the IDs with the database name to create a global ID
    def _concat_helper(self, data: pl.LazyFrame, name: str) -> pl.LazyFrame:
        return data.with_columns(
            pl.concat_str(
                [pl.lit(name), pl.col(self.icu_stay_id_col).cast(int).cast(str)]
            ).alias(self.global_icu_stay_id_col)
        )

    # Print the number of unique cases in the timeseries data
    def _print_unique_cases(
        self, data: pl.LazyFrame, name: str
    ) -> pl.LazyFrame:
        unique_count = (
            data.select(self.global_icu_stay_id_col)
            .unique()
            .count()
            .collect()
            .item()
        )
        print(f"reprodICU - {unique_count:6.0f} unique cases with timeseries data in {name}.") # fmt: skip

        return data
