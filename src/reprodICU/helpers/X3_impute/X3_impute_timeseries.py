# Author: Finn Fassbender
# Last modified: 2025-10-30

# Adapted from source:
# https://github.com/ratschlab/circEWS/blob/master/circews/functions/forward_filling.py#L205

# Description: This script imputes the data to remove missing values.
# It is available as a module for piping in the main script.

import polars as pl

from ..helper import GlobalVars


class TimeseriesImputer(GlobalVars):
    def __init__(self, paths, DEMO=False) -> None:
        super().__init__(paths)
        self.save_path = (
            paths.reprodICU_files_path
            if not DEMO
            else paths.reprodICU_demo_files_path
        )
        self.index_cols = [
            "Global ICU Stay ID",
            "Time Relative to Admission (seconds)",
        ]

    def _interpolate(
        self, df: pl.LazyFrame, ts_col: str, id_col: str
    ) -> pl.LazyFrame:
        """
        Interpolate missing values in timeseries data.

        Returns:
            pl.LazyFrame: Data with interpolated values filling gaps.
        """

        index_cols = [id_col, ts_col]
        schema = df.collect_schema()
        cols = schema.names()
        value_cols = [
            x for x in cols if x not in index_cols and schema[x].is_numeric()
        ]

        return df.with_columns(
            pl.col(value_col).interpolate_by(ts_col).over(partition_by=id_col)
            for value_col in value_cols
        )

    def impute_timeseries(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """
        Impute missing timeseries values via linear interpolation.

        Steps:
            1. Call _interp with time column and ICU stay ID columns.
            2. Interpolate between non-null value observations.

        Returns:
            pl.LazyFrame: Data with interpolated missing values.
        """

        return data.pipe(
            self._interpolate,
            ts_col="Time Relative to Admission (seconds)",  # Time column
            id_col="Global ICU Stay ID",  # ID columns
        )

    def impute_timeseries_vitals(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """
        Impute missing vital signs using interpolation and standardize types.

        Steps:
            1. Interpolate missing vital values.
            2. Cast all numeric vitals (except temperature) to integer.
            3. Round temperature to 1 decimal place.

        Returns:
            pl.LazyFrame: Contains columns:
                - {global_icu_stay_id_col}: Global ICU stay identifier.
                - {timeseries_time_col}: Time offset (seconds from ICU admission).
                - Temperature: Body temperature (°C, 1 decimal).
                - [Other vitals]: Vital measurements (integer).
        """

        columns = data.collect_schema().names()

        return (
            data.pipe(self.impute_timeseries)
            # Cast the values to the original data type
            .with_columns(
                pl.col(col).cast(int)
                for col in columns
                if col
                not in [*self.index_cols, "Temperature", "Heart rate rhythm"]
            )
            # Round temperature to 1 decimal place
            .with_columns(pl.col("Temperature").round(1).alias("Temperature"))
        )