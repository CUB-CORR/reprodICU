# Author: Finn Fassbender
# Last modified: 2024-09-11

# Adapted from source:
# https://github.com/ratschlab/circEWS/blob/master/circews/functions/forward_filling.py#L205

# Description: This script imputes the data to remove missing values.
# It is available as a module for piping in the main script.
# It can be called with command line arguments to specify the source datasets to be imputed. ! NOT IMPLEMENTED YET !

import argparse

import numpy as np
import polars as pl
from helpers.helper import GlobalVars


class TimeseriesImputer(GlobalVars):
    def __init__(self, paths, DEMO=False) -> None:
        super().__init__(paths)
        self.save_path = (
            paths.reprodICU_files_path
            if not DEMO
            else paths.reprodICU_demo_files_path
        )
        self.temp_path = self.save_path + "_tempfiles/impute/"

    def get_preadmission_data(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Returns the baseline data for each ICU stay, i.e. the data from before admission.
        """

        return data.filter(pl.col(self.timeseries_time_col) < 0)

    def get_timegrid(
        self, data: pl.LazyFrame, resolution_in_seconds: int = 300
    ) -> pl.LazyFrame:
        """
        Creates a time grid for each ICU stay.
        Default resolution is 5 minutes.
        Data is imputed to fill missing values.
        """

        # Create a time grid for each ICU stay
        return (
            data.group_by(self.global_icu_stay_id_col)
            .agg(
                pl.int_range(
                    start=0,
                    end=pl.max(self.timeseries_time_col),
                    step=resolution_in_seconds,
                ).alias(self.timeseries_time_col)
            )
            .explode(self.timeseries_time_col)
        )

    def adaptive_forward_fill(
        self,
        data: pl.DataFrame,
        timegrid: pl.DataFrame,
        global_icu_stay_id: str,
        keep_preadmission_data: bool = False,
    ) -> pl.DataFrame:
        """
        Adaptive forward fill for time series data.

        Adapted from Hyland, S.L., Faltys, M., Hüser, M. et al.
        Early prediction of circulatory failure in the intensive care unit using machine learning.
        Nat Med 26, 364–373 (2020).

        Source Code:
        https://github.com/ratschlab/circEWS/blob/master/circews/functions/forward_filling.py
        """

        # filter out the preadmission data
        preadmission_data = self.get_preadmission_data(data)
        data = data.filter(pl.col(self.timeseries_time_col) >= 0)

        # Convert the timegrid to numpy arrays
        timegrid = timegrid.drop(self.global_icu_stay_id_col).to_numpy()
        timegrid_time = timegrid[:, 0]

        # Initialize variables
        # len = length of the timegrid
        # width = number of features (columns) in the data
        numpy_data_imputed = np.full(
            (timegrid.shape[0], len(data.columns) - 2), np.nan
        )

        # Fill in each point of the timegrid.
        # Iterate over each timeseries separately, as the forward fill is adaptive.
        for i, col in enumerate(data.columns[2:]):
            # Convert the data to numpy arrays
            numpy_pre_data = (
                preadmission_data.select(self.timeseries_time_col, col)
                .filter(pl.col(col).is_not_null(), pl.col(col).is_not_nan())
                .to_numpy()
            )
            numpy_pre_data_time = numpy_pre_data[:, 0]
            numpy_pre_data_vals = numpy_pre_data[:, 1]

            numpy_data = (
                data.select(self.timeseries_time_col, col)
                .filter(pl.col(col).is_not_null(), pl.col(col).is_not_nan())
                .to_numpy()
            )
            numpy_data_time = numpy_data[:, 0]
            numpy_data_vals = numpy_data[:, 1]

            # Initialize the backward window as going forward in time.
            # Determine the first start point for the extrapolation.
            # start_point = np.median(numpy_pre_data[:, i])
            start_point = (
                numpy_pre_data_vals[-1]
                if len(numpy_pre_data_vals) > 0
                else None
            )
            start_point_time = (
                numpy_pre_data_time[-1]
                if len(numpy_pre_data_time) > 0
                else None
            )
            # Reassign for compatibility with the loop
            aim_point = start_point
            aim_point_time = start_point_time

            # Initialize the slope and aim point variables.
            slope = None
            slope_active = False
            timegrid_time_index = 0
            timegrid_time_current = 0

            # Filling using slope estimation and backward windows,
            # use the first observation after the timegrid point to estimate the slope.
            for j in range(numpy_data.shape[0]):
                # Continue if there is no start point or no data point.
                # Skip the time points that are before the current timegrid point.
                if (start_point_time is None) or (
                    numpy_data_time[j] < timegrid_time_current
                ):
                    # print(
                    #     i,
                    #     "continue ",
                    #     (start_point == np.nan)
                    #     or (start_point_time is None)
                    #     or (numpy_data[j, i] == np.nan)
                    #     or (numpy_data_time[j] < timegrid_time_current),
                    #     "slope ",
                    #     slope,
                    #     "start_point ",
                    #     start_point,
                    #     "aim_point ",
                    #     aim_point,
                    #     "start_point_time ",
                    #     start_point_time,
                    #     "aim_point_time ",
                    #     aim_point_time,
                    # )
                    continue

                # Recompute the slope and aim point if the timegrid point is reached.
                if numpy_data_time[j] > timegrid_time_current:
                    slope_active = False

                # Slope and aim point has to be recomputed.
                if not slope_active:
                    # Update the start point and time.
                    start_point = aim_point
                    start_point_time = aim_point_time

                    # Determine the aim point.
                    aim_point = numpy_data_vals[j]
                    aim_point_time = numpy_data_time[j]

                    # Determine the slope of the extrapolation.
                    slope = (aim_point - start_point) / (
                        aim_point_time - start_point_time
                    )

                    print(
                        global_icu_stay_id,
                        i,
                        "slope ",
                        slope,
                        "start_point ",
                        start_point,
                        "aim_point ",
                        aim_point,
                        "start_point_time ",
                        start_point_time,
                        "aim_point_time ",
                        aim_point_time,
                    )
                    slope_active = True

                # Fill the timegrid points with the predicted values until the next observation.
                while (timegrid_time_current <= numpy_data_time[j]) & (
                    timegrid_time_index < (len(timegrid_time) - 1)
                ):
                    # Calculate the predicted value.
                    numpy_data_imputed[timegrid_time_index, i] = (
                        start_point
                        + slope * (timegrid_time_current - start_point_time)
                    )

                    # Update the timeseries index.
                    timegrid_time_index += 1
                    timegrid_time_current = timegrid_time[timegrid_time_index]

        # add axes to times
        numpy_pre_data_time = numpy_pre_data_time[:, np.newaxis]
        timegrid_time = timegrid_time[:, np.newaxis]

        numpy_data_imputed = np.hstack((timegrid_time, numpy_data_imputed))

        if keep_preadmission_data:
            numpy_pre_data = np.hstack((numpy_pre_data_time, numpy_pre_data))
            numpy_data_imputed = np.vstack(
                (
                    preadmission_data.drop(
                        self.global_icu_stay_id_col
                    ).to_numpy(),
                    numpy_data_imputed,
                )
            )

        numpy_data_imputed = np.hstack(
            (
                np.repeat(
                    global_icu_stay_id, numpy_data_imputed.shape[0]
                ).reshape(-1, 1),
                numpy_data_imputed,
            )
        )

        return pl.DataFrame(
            numpy_data_imputed,
            schema=data.columns,
        ).with_columns(
            pl.col(self.global_icu_stay_id_col).cast(str),
            pl.exclude(self.global_icu_stay_id_col).cast(float).round(2),
        )

    def impute_timeseries(
        self,
        data: pl.LazyFrame,
        resolution_in_seconds: int = 300,
        keep_preadmission_data: bool = False,
    ) -> pl.LazyFrame:
        """
        Imputes the data to remove missing values from the time series during ICU.
        """

        # baseline = self.get_preadmission_data(data)
        # timegrid = self.fill_timegrid(data, resolution_in_seconds).group_by(
        #     self.global_icu_stay_id_col
        # )
        timegrid = self.get_timegrid(data, resolution_in_seconds)
        grouped_data = data.collect(streaming=True).group_by(
            self.global_icu_stay_id_col
        )

        # Impute the data
        counter = 0
        num_cases = (
            data.select(pl.col(self.global_icu_stay_id_col).approx_n_unique())
            .collect(streaming=True)
            .to_numpy()[0][0]
        )

        # iterate over the ICU stays
        for global_icu_stay_id, data in grouped_data:
            if counter == 50:
                break
            global_icu_stay_id = global_icu_stay_id[0]
            c, n, p = counter, num_cases, counter / num_cases
            print(
                f"Imputing ICU stay {global_icu_stay_id}...\t({c:6.0f}/{n:6.0f}, {p:5.2%})",
                end="\r",
            )

            # Select the timegrid for the current ICU stay
            _timegrid = timegrid.filter(
                pl.col(self.global_icu_stay_id_col) == global_icu_stay_id
            ).collect(streaming=True)
            # Impute the data for the current ICU stay
            data = (
                data.sort(self.timeseries_time_col)
                .pipe(
                    self.adaptive_forward_fill,
                    _timegrid,
                    global_icu_stay_id,
                    keep_preadmission_data,
                )
                .write_parquet(self.temp_path + f"{global_icu_stay_id}.parquet")
            )
            counter += 1

        # Concatenate the imputed data
        timeseries_imputed = pl.scan_parquet(self.temp_path + "*.parquet")
        timeseries_imputed.sort(
            self.global_icu_stay_id_col, self.timeseries_time_col
        ).sink_parquet(self.save_path + "timeseries_vitals_imputed.parquet")

        return timeseries_imputed


if __name__ == "__main__":
    raise NotImplementedError(
        "This script is not yet implemented as a command line tool."
    )
