# Author: Finn Fassbender
# Last modified: 2024-09-11

# Adapted from source:
# https://github.com/ratschlab/circEWS/blob/master/circews/functions/forward_filling.py#L205

# Description: This script resamples the data to remove missing values.
# It is available as a module for piping in the main script.

import polars as pl
from helpers.helper import GlobalVars
from helpers.X3_impute.X3_impute_timeseries import TimeseriesImputer

DAY_ZERO = pl.datetime(year=2000, month=1, day=1, hour=0, minute=0, second=0)


class TimeseriesResampler(GlobalVars):
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
        self.timeseries_imputer = TimeseriesImputer(paths, DEMO)
        self._interp = self.timeseries_imputer._interp

    def resample_timeseries(
        self,
        data: pl.LazyFrame,
        resolution_in_seconds: int = 300,
    ) -> pl.LazyFrame:
        """
        Resample missing values in the data using interpolation and forward filling.
        """
        resampled_grid = (
            # Group by Global ICU stay ID
            data.group_by("Global ICU Stay ID")
            # Create a range of time points for each ICU stay
            .agg(
                pl.int_range(
                    0,  # Start at 0 (i.e. admission)
                    pl.col("Time Relative to Admission (seconds)").max()
                    + resolution_in_seconds,
                    resolution_in_seconds,
                )
                .cast(float)
                .alias("Time Relative to Admission (seconds)")
            ).explode("Time Relative to Admission (seconds)")
        )

        resampled_data = (
            resampled_grid
            # Join the original data to the resampled grid
            .join(data, on=self.index_cols, how="full", coalesce=True)
            # Interpolate missing values
            .pipe(
                self._interp,
                "Time Relative to Admission (seconds)",  # Time column
                ["Global ICU Stay ID"],  # ID columns
            )
        )

        # Return only the new grid with the resampled values
        return resampled_grid.join(
            resampled_data, on=self.index_cols, how="left"
        )

    def resample_timeseries_vitals(
        self,
        data: pl.LazyFrame,
        resolution_in_seconds: int = 300,
    ) -> pl.LazyFrame:
        """
        Resample missing values in the vitals data using interpolation and forward filling.
        """

        columns = data.collect_schema().names()

        return (
            data.pipe(
                self.resample_timeseries,
                resolution_in_seconds=resolution_in_seconds,
            )
            # Cast the values to the original data type
            .with_columns(
                pl.col(col).cast(int)
                for col in columns
                if col not in [*self.index_cols, "Temperature"]
            )
            # Round temperature to 1 decimal place
            .with_columns(pl.col("Temperature").round(1).alias("Temperature"))
        )
