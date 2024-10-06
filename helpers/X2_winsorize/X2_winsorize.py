# Author: Finn Fassbender
# Last modified: 2024-09-11

# Description: This script winsorizes the data to remove outliers.
# It is available as a module for piping in the main script.
# It can be called with command line arguments to specify the source datasets to be winsorized. ! NOT IMPLEMENTED YET !

import argparse
import polars as pl


class X2_Winsorizer:
    def __init__(self):
        pass

    def winsorize_quantiles(data, columns: list, alpha=0.99) -> pl.LazyFrame:
        """
        Winsorize the data to remove outliers.
        Clip the data to the 1-alpha quantile (lower) and alpha quantile (upper bound).

        :param data: The data to be winsorized.
        :param columns: The columns to be winsorized.
        :param alpha: The quantile to be used for winsorization.

        :return: The winsorized data.
        """

        return data.with_columns(
            *[
                pl.col(column)
                .clip(
                    pl.col(column).quantile(1 - alpha),
                    pl.col(column).quantile(alpha),
                )
                .alias(column)
                for column in columns
            ]
        )

    def winsorize_clip_lower_0_quantiles(
        data, columns: list, alpha=0.99
    ) -> pl.LazyFrame:
        """
        Winsorize the data to remove outliers.
        Clip the data to 0 (lower) and alpha quantile (upper bound).

        :param data: The data to be winsorized.
        :param columns: The columns to be winsorized.
        :param alpha: The quantile to be used for winsorization.

        :return: The winsorized data.
        """

        return data.with_columns(
            *[
                pl.col(column)
                .clip(pl.lit(0), pl.col(column).quantile(alpha))
                .alias(column)
                for column in columns
            ]
        )

    def winsorize_clip_lower_0(data, columns: list) -> pl.LazyFrame:
        """
        Winsorize the data to remove outliers.
        Clip the data to 0 (lower), the upper bound is not changed.

        :param data: The data to be winsorized.
        :param columns: The columns to be winsorized.

        :return: The winsorized data.
        """

        return data.with_columns(
            *[
                pl.col(column).clip(lower_bound=0).alias(column)
                for column in columns
            ]
        )

    def winsorize_clip_multiple(
        data, columns: list, lower: list, upper: list
    ) -> pl.LazyFrame:
        """
        Winsorize the data to remove outliers.
        Clip the data to the specified lower and upper bounds.

        :param data: The data to be winsorized.
        :param columns: The columns to be winsorized.

        :return: The winsorized data.
        """

        return data.with_columns(
            *[
                pl.col(column)
                .clip(lower_bound=_lower, upper_bound=_upper)
                .alias(column)
                for column, _lower, _upper in zip(columns, lower, upper)
            ]
        )


if __name__ == "__main__":
    raise NotImplementedError(
        "This script is not yet implemented as a command line tool."
    )


# # Author: Finn Fassbender
# # Last modified: 2024-09-05

# # Restricts the values of a DataFrame to a specified range.
# # The values below the lower bound are set to the lower bound,
# # the values above the upper bound are set to the upper bound.
# # The values within the range are not changed.

# import polars as pl
# import yaml

# from helpers.helper import GlobalVars


# class Winsorize(GlobalVars):
#     def __init__(self, limits_list_path="configs/CLINICALLY_PLAUSIBLE_VALUES.yaml"):
#         with open(limits_list_path, "r") as f:
#             self.limits = yaml.safe_load(f)

#     def winsorize(
#         self, data: pl.LazyFrame, column: str, lower: float, upper: float
#     ) -> pl.LazyFrame:
#         # Apply the winsorization to the respective column.
#         return data.with_columns(
#             pl.when(self.df[column] < lower)
#             .then(lower)
#             .when(self.df[column] > upper)
#             .then(upper)
#             .otherwise(self.df[column])
#             .alias(column),
#         )

#     def winsorize_all(self, data: pl.LazyFrame, limits: dict = None) -> pl.LazyFrame:
#         # Apply the winsorization to all columns.
#         if limits is None:
#             limits = self.limits

#         for column in data.columns:
#             if column in limits:
#                 lower = limits[column]["lower"]
#                 upper = limits[column]["upper"]
#                 data = self.winsorize(data, column, lower, upper)
#         return data
