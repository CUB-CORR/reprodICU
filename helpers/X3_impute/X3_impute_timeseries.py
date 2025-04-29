# Author: Finn Fassbender
# Last modified: 2024-09-11

# Adapted from source:
# https://github.com/ratschlab/circEWS/blob/master/circews/functions/forward_filling.py#L205

# Description: This script imputes the data to remove missing values.
# It is available as a module for piping in the main script.

import polars as pl
from helpers.helper import GlobalVars

DAY_ZERO = pl.datetime(year=2000, month=1, day=1, hour=0, minute=0, second=0)


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

    # taken from @deanm0000 on polars github, edited slightly for readability
    # source: https://github.com/pola-rs/polars/issues/9616#issuecomment-1718358252
    def _interp(
        self, df: pl.LazyFrame, ts_col: str, id_cols=None
    ) -> pl.LazyFrame:
        """
        Interpolates missing values in a time series.
        Accepts any number of value columns, ID columns, and a time column (ts_col).
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
                        __value_base=pl.col(value_col).shift()
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
        Impute missing values in the data using interpolation and forward filling.
        """

        return data.pipe(
            self._interp,
            "Time Relative to Admission (seconds)",  # Time column
            ["Global ICU Stay ID"],  # ID columns
        )

    def impute_timeseries_vitals(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """
        Impute missing values in the vitals data using interpolation and forward filling.
        """

        columns = data.collect_schema().names()

        return (
            data.pipe(self.impute_timeseries)
            # Cast the values to the original data type
            .with_columns(
                pl.col(col).cast(int)
                for col in columns
                if col not in [*self.index_cols, "Temperature"]
            )
            # Round temperature to 1 decimal place
            .with_columns(pl.col("Temperature").round(1).alias("Temperature"))
        )
