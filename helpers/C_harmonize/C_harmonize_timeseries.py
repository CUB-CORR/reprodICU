# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script combines the preprocessed patient information from the differet
# databases into one common table

import polars as pl

from helpers.B_process.B_process_eicu import EICUProcessor
from helpers.B_process.BX_process_hirid import HiRIDProcessor
from helpers.B_process.B_process_mimic3 import MIMIC3Processor
from helpers.B_process.B_process_mimic4 import MIMIC4Processor
from helpers.B_process.B_process_nwicu import NWICUProcessor
from helpers.B_process.BX_process_sicdb import SICdbProcessor
from helpers.B_process.BX_process_umcdb import UMCdbProcessor
from helpers.helper import GlobalHelpers, GlobalVars


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
        self.eicu = EICUProcessor(paths, DEMO)
        self.hirid = HiRIDProcessor(paths)
        self.mimic3 = MIMIC3Processor(paths, DEMO)
        self.mimic4 = MIMIC4Processor(paths, DEMO)
        self.nwicu = NWICUProcessor(paths)
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
    def harmonize_split_timeseries(
        self, timeseries=[], save_to_default=True
    ) -> None:
        """
        Splits and harmonizes the timeseries data into four categories: vitals, labs, respiratory, and intake/output.

        This function performs the following steps:
            1. Validates that {datasets} and the list of timeseries categories are provided; raises ValueError if either is empty.
            2. Constructs Polars Series for each category to represent the relevant column names:
               - {index_cols}: Identifiers such as {global_icu_stay_id_col} and {timeseries_time_col}.
               - {relevant_vital_values}, {relevant_respiratory_values}, {relevant_intakeoutput_values}, {relevant_lab_LOINC_components}: These subsets are used for filtering.
            3. For each dataset in {datasets} (e.g., "eICU", "HiRID", "MIMIC3", etc.):
               - Processes the corresponding timeseries data via dataset-specific methods.
               - Applies a helper method (_concat_helper) to create a global ID by concatenating a dataset prefix.
               - Filters the schema to select the columns that match the pre-defined Series for each category.
            4. Concatenates the data for each category using a "diagonal_relaxed" join.
            5. For each category:
               - Vitals: Cleans and casts to the appropriate data types, fixes temperature values (e.g., converting accidental Fahrenheit values), and sums subscores for Glasgow coma score.
               - Labs, Respiratory, and Intake/Output: Cast and remove duplicate records while ensuring {global_icu_stay_id_col} and {timeseries_time_col} are maintained.
            6. If save_to_default is True, writes the processed data for each category to parquet files under {save_path}; otherwise, returns a tuple containing:
               - vitals: Contains {global_icu_stay_id_col}, {timeseries_time_col} plus sorted vital measurement columns.
               - labs: Contains {global_icu_stay_id_col}, {timeseries_time_col} plus lab result columns.
               - resp: Contains {global_icu_stay_id_col}, {timeseries_time_col} and respiratory parameters.
               - inout: Contains {global_icu_stay_id_col}, {timeseries_time_col} and intake/output measurements.

        Returns:
            None if saving to files; otherwise, a tuple (vitals, labs, resp, inout) of processed Polars DataFrames.

        Raises:
            ValueError: If {datasets} is empty or if no timeseries categories are selected.
        """
        if self.datasets == []:
            raise ValueError("No datasets to harmonize the timeseries from.")
        if timeseries == []:
            raise ValueError("No timeseries selected.")

        vital_prms = pl.Series([*self.index_cols, *self.relevant_vital_values])
        resp_prms = pl.Series(
            [*self.index_cols, *self.relevant_respiratory_values]
        )
        inout_prms = pl.Series(
            [*self.index_cols, *self.relevant_intakeoutput_values]
        )

        labs_prms = pl.Series(
            [*self.index_cols, *self.relevant_lab_LOINC_components]
        )

        # Harmonize the timeseries per category
        timeseries_vitals = []
        timeseries_labs = []
        timeseries_resp = []
        timeseries_inout = []

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
            hirid_ts, hirid_ts_labs = self.hirid.process_timeseries()
            hirid_timeseries = hirid_ts.pipe(self._concat_helper, "hirid-")
            hirid_timeseries_labs = hirid_ts_labs.pipe(
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
            timeseries_inout.append(
                mimic3_timeseries_inout.select(*mimic3_inout)
            )
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
            timeseries_inout.append(
                mimic4_timeseries_inout.select(*mimic4_inout)
            )
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
            # nwicu_resp = resp_prms.filter(resp_prms.is_in(nwicu_ts_names))
            # nwicu_inout = inout_prms.filter(inout_prms.is_in(nwicu_ts_names))

            nwicu_ts_lab_names = nwicu_timeseries_labs.collect_schema().names()
            nwicu_labs = labs_prms.filter(labs_prms.is_in(nwicu_ts_lab_names))

            timeseries_vitals.append(nwicu_timeseries.select(*nwicu_vitals))
            # timeseries_resp.append(nwicu_timeseries.select(*nwicu_resp))
            timeseries_labs.append(nwicu_timeseries_labs.select(*nwicu_labs))
            # timeseries_inout.append(nwicu_timeseries.select(*nwicu_inout))
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

            sicdb_ts_lab_names = sicdb_timeseries_labs.collect_schema().names()
            sicdb_labs = labs_prms.filter(labs_prms.is_in(sicdb_ts_lab_names))

            timeseries_vitals.append(sicdb_timeseries.select(*sicdb_vitals))
            timeseries_resp.append(sicdb_timeseries.select(*sicdb_resp))
            timeseries_labs.append(sicdb_timeseries_labs.select(*sicdb_labs))
            timeseries_inout.append(sicdb_timeseries.select(*sicdb_inout))
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

            umcdb_ts_lab_names = umcdb_timeseries_labs.collect_schema().names()
            umcdb_labs = labs_prms.filter(labs_prms.is_in(umcdb_ts_lab_names))

            timeseries_vitals.append(umcdb_timeseries.select(*umcdb_vitals))
            timeseries_resp.append(umcdb_timeseries.select(*umcdb_resp))
            timeseries_labs.append(umcdb_timeseries_labs.select(*umcdb_labs))
            timeseries_inout.append(umcdb_timeseries.select(*umcdb_inout))
        # endregion

        # Concatenate the timeseries data for each category
        # region vitals
        vitals = pl.concat(timeseries_vitals, how="diagonal_relaxed")
        vitals_cols = vitals.collect_schema().names()
        vitals_cols_not_index = list(set(vitals_cols) - set(self.index_cols))
        vitals = (
            vitals.pipe(
                self.helpers.dropna, "all", vitals_cols_not_index, False
            )
            .cast(
                {  # Convert columns to appropriate types
                    self.global_icu_stay_id_col: str,
                    self.timeseries_time_col: float,
                    **{
                        col: str if col in ["Heart rate rhythm"] else float
                        for col in vitals_cols_not_index
                    },
                }
            )
            .select([*self.index_cols, *sorted(vitals_cols_not_index)])
            .with_columns(
                # Fix Temperature if value appears to be in Fahrenheit
                pl.when(pl.col("Temperature").gt(60))
                .then(pl.col("Temperature").sub(32).mul(5).truediv(9))
                .otherwise(pl.col("Temperature"))
                .alias("Temperature"),
                # Sum Glasgow coma score components if total is missing
                pl.when(pl.col("Glasgow coma score total").is_null())
                .then(
                    pl.sum_horizontal(
                        "Glasgow coma score eye opening",
                        "Glasgow coma score motor",
                        "Glasgow coma score verbal",
                        ignore_nulls=False,
                    )
                )
                .otherwise(pl.col("Glasgow coma score total"))
                .alias("Glasgow coma score total"),
            )
            # assume uniqueness (since we're just concatenating the data)
            .sort(self.index_cols)
        )
        # endregion

        # region labs
        labs = (
            pl.concat(timeseries_labs, how="diagonal_relaxed")
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
            .unique(self.index_cols)
            .sort(self.index_cols)
        )
        # endregion

        # region respiratory
        resp = pl.concat(timeseries_resp, how="diagonal_relaxed")
        resp_cols = resp.collect_schema().names()
        resp_cols_not_index = list(set(resp_cols) - set(self.index_cols))
        resp = (
            resp.pipe(self.helpers.dropna, "all", resp_cols_not_index, False)
            .cast(
                {  # Convert all columns to float, except for
                    # - Oxygen delivery system
                    # - Ventilation mode Ventilator
                    # - Ventilator type
                    self.global_icu_stay_id_col: str,
                    self.timeseries_time_col: float,
                    **{
                        col: (
                            str
                            if col
                            in [
                                "Oxygen delivery system",
                                "Ventilation mode Ventilator",
                                "Ventilator type",
                            ]
                            else float
                        )
                        for col in resp_cols_not_index
                    },
                },
                # silently fail on invalid values (i.e. don't raise an error)
                strict=False,
            )
            .select([*self.index_cols, *sorted(resp_cols_not_index)])
            .unique(self.index_cols)
            .sort(self.index_cols)
        )
        # endregion

        # region intakeoutput
        inout = pl.concat(timeseries_inout, how="diagonal_relaxed")
        inout_cols = inout.collect_schema().names()
        inout_cols_not_index = list(set(inout_cols) - set(self.index_cols))
        inout = (
            inout.pipe(self.helpers.dropna, "all", inout_cols_not_index, False)
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
        # endregion

        # region save
        if save_to_default:
            print("reprodICU - Saving timeseries...")

            if "vitals" in timeseries:
                print("reprodICU - Saving vitals...")
                vitals.pipe(self._print_unique_cases, "vitals").pipe(
                    self._fix_temperature_values
                ).sink_parquet(self.save_path + "timeseries_vitals.parquet")

            if "labs" in timeseries:
                print("reprodICU - Saving labs...")
                (
                    labs.pipe(self._print_unique_cases, "labs")
                    .pipe(self.decode_lab_values)
                    .collect(streaming=True)
                    .write_parquet(self.save_path + "timeseries_labs.parquet")
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

            return None

        return vitals, labs, resp, inout

    # endregion

    # region decode
    # Decode the lab values
    def decode_lab_values(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        Decodes lab values stored as JSON strings in columns to a structured format.

        For each non-index column in the input LazyFrame, this function:
            - Treats the column value as a JSON string.
            - Decodes it into a struct with fields:
                 "value": Numeric lab value.
                 "system": Coding system.
                 "method": Measurement method.
                 "time": Time of measurement.
                 "LOINC": LOINC code.

        Args:
            lf (pl.LazyFrame): Input LazyFrame with lab value columns.

        Returns:
            pl.LazyFrame: The LazyFrame with decoded lab value columns.
        """

        labstructdtype = pl.Struct(
            [
                pl.Field("value", pl.Float64),
                pl.Field("system", pl.String),
                pl.Field("method", pl.String),
                pl.Field("time", pl.String),
                pl.Field("LOINC", pl.String),
            ]
        )

        def decode_lab_value(lab_value):
            return pl.col(lab_value).str.json_decode(labstructdtype)

        value_cols = [
            col
            for col in lf.collect_schema().names()
            if col not in self.index_cols
        ]

        return lf.with_columns(*map(decode_lab_value, value_cols))

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
