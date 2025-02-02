# Author: Finn Fassbender
# Last modified: 2024-09-11

# Description: This script imputes the data to remove missing values.
# It is available as a module for piping in the main script.
# It can be called with command line arguments to specify the source datasets to be imputed. ! NOT IMPLEMENTED YET !

import polars as pl
from helpers.helper import GlobalVars


class IntakeOutputImprover(GlobalVars):
    def __init__(self, paths) -> None:
        super().__init__(paths)
        pass

    def improve_intake_output(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """
        Impute the intake output data to add fluid balance information.

        :param data: The intake output data to be calculated.

        :return: The calculated intake output data.
        :rtype: pl.DataFrame
        """

        inout_cols = data.collect_schema().names()
        inout_cols_series = pl.Series(inout_cols)
        input_cols = inout_cols_series.filter(
            inout_cols_series.str.contains_any(["input", "Input"])
        ).to_list()
        output_cols = inout_cols_series.filter(
            inout_cols_series.str.contains_any(["output", "Output"])
        ).to_list()

        # Impute missing values
        return data.with_columns(
            (
                pl.sum_horizontal(
                    pl.lit(0), pl.col(input_cols), ignore_nulls=True
                )
                - pl.sum_horizontal(
                    pl.lit(0), pl.col(output_cols), ignore_nulls=True
                )
            ).alias("Fluid balance")
        ).with_columns(
            # Calculate the total fluid balance
            pl.col("Fluid balance")
            .cum_sum()
            .over("Global ICU Stay ID")
            .alias("Fluid balance"),
        )


if __name__ == "__main__":
    raise NotImplementedError(
        "This script is not yet implemented as a command line tool."
    )
