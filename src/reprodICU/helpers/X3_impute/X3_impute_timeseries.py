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

    def _interp_simple(
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

    # taken from @deanm0000 on polars github, edited slightly
    # source: https://github.com/pola-rs/polars/issues/9616#issuecomment-1718358252
    def _interp(
        self, df: pl.LazyFrame, ts_col: str, id_cols=None
    ) -> pl.LazyFrame:
        """
        Interpolate missing values in timeseries data.

        Steps:
            1. Extract grid of ID and time columns from input data.
            2. For each value column: identify non-null observations.
            3. Calculate per-observation slope (change per time unit).
            4. Use forward asof join to match each row with preceding observation.
            5. Interpolate: base_value + slope × time_elapsed.
            6. Coalesce interpolated values where available.

        Returns:
            pl.LazyFrame: Data with interpolated values filling gaps.
        """

        if not isinstance(ts_col, str):
            raise ValueError("ts_col should be string")

        if isinstance(id_cols, str):
            id_cols = [id_cols]

        if id_cols is None:
            id_cols = ["__dummyid"]
            df = df.with_columns(__dummyid=0)

        lf = df.select(id_cols + [ts_col]).lazy()
        cols = df.collect_schema().names()
        value_cols = [x for x in cols if x not in id_cols and x != ts_col]

        # Iterate over all value columns, interpolating missing values
        for value_col in value_cols:
            lf = lf.join(
                # Join the original data to itself, using an asof join
                df.join_asof(
                    # Select all available data for the current value column
                    df.filter(pl.col(value_col).is_not_null()).select(
                        *id_cols,
                        ts_col,
                        # Calculate the point-wise slope of the value column
                        # (i.e. the change in value per time unit)
                        __value_slope=(
                            pl.col(value_col)
                            - pl.col(value_col).shift().over(id_cols)
                        )
                        / (
                            pl.col(ts_col)
                            - pl.col(ts_col).shift().over(id_cols)
                        ),
                        # Store previous values interpolation
                        __value_slope_since=pl.col(ts_col).shift(),
                        __value_base=pl.col(value_col).shift(),
                    ),
                    on=ts_col,
                    by=id_cols,
                    strategy="forward",
                )
                .select(
                    id_cols
                    + [ts_col]
                    + [
                        pl.coalesce(
                            # Keep the original value if it is not null
                            pl.col(value_col),
                            # Otherwise, interpolate the value by adding the base value
                            # to the slope multiplied by the time since the last known value
                            pl.coalesce(
                                pl.col("__value_base"),
                                pl.col("__value_base").shift(-1),
                            )
                            # Add the slope to the base value
                            + pl.coalesce(
                                pl.col("__value_slope"),
                                pl.col("__value_slope").shift(-1),
                            )
                            # Multiply the slope by the time since the last known value
                            * (
                                pl.col(ts_col)
                                - pl.coalesce(
                                    pl.col("__value_slope_since"),
                                    pl.col("__value_slope_since").shift(-1),
                                )
                            ),
                        ).alias(value_col)
                    ]
                )
                .lazy(),
                on=[ts_col] + id_cols,
            )

        # Drop the dummy ID column if it was added
        if id_cols[0] == "__dummyid":
            lf = lf.select(pl.exclude("__dummyid"))

        return lf

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
            self._interp_simple,
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
