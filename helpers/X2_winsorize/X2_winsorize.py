# Author: Finn Fassbender
# Last modified: 2024-09-11

# Description: This script winsorizes the data to remove outliers.
# It is available as a module for piping in the main script.
# It can be called with command line arguments to specify the source datasets to be winsorized. ! NOT IMPLEMENTED YET !

import sys
import warnings

import polars as pl

warnings.filterwarnings("ignore")


class X2_Winsorizer:
    def __init__(self):
        pass

    def winsorize_quantiles(
        self, data: pl.LazyFrame, columns: list, alpha=0.99, **kwargs
    ) -> pl.LazyFrame:
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
        self, data: pl.LazyFrame, columns: list, alpha=0.99, **kwargs
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

    def winsorize_clip_lower_0(
        self, data: pl.LazyFrame, columns: list, **kwargs
    ) -> pl.LazyFrame:
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
        self,
        data: pl.LazyFrame,
        columns: list,
        lower: list,
        upper: list,
        **kwargs,
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

    def winsorize_structs(
        self,
        data: pl.LazyFrame,
        winsorization_columns: list,
        winsorization_methods: list,
        **kwargs,
    ) -> pl.LazyFrame:
        """
        Split the struct columns and winsorize the individual columns before reassembling the struct.
        """

        # Assert methods are valid
        assert all(
            method in ["quantiles", "clip_lower_0", "clip_lower_0_quantiles"]
            for method in winsorization_methods
        )

        # Define a helper function to get unique values from a column
        def get_unique_values(data: pl.LazyFrame, column: str) -> list:
            return data.select(column).unique().collect().to_series().to_list()

        # Get the data types of the columns
        column_names = data.collect_schema().names()
        column_dtypes = data.collect_schema().dtypes()
        column_dtypes_dict = dict(zip(column_names, column_dtypes))

        counter, count = 1, len(winsorization_columns)
        # Iterate over the each column to be winsorized
        for winsorization_column, winsorization_method in zip(
            winsorization_columns, winsorization_methods
        ):
            # Print progress
            sys.stdout.write("\033[K")  # Clear to the end of line
            print(
                f"reprodICU - Winsorizing column '{winsorization_column}' ({counter:2.0f}/{count:2.0f})",
                end="\r",
            )
            counter += 1

            # Do normal winsorization for non-struct columns
            if column_dtypes_dict[winsorization_column] != pl.Struct:
                data = getattr(
                    X2_Winsorizer, f"winsorize_{winsorization_method}"
                )(data, [winsorization_column], **kwargs)

            # 1. Unnest the struct column
            # 2. Create a new column for each combination of system and method
            # 3. Apply the winsorization method to each new column respectively
            # 4. Reassemble the struct column
            else:
                data = data.unnest(winsorization_column)
                LOINC_codes = data.pipe(get_unique_values, column="LOINC")
                data = data.with_columns(
                    pl.when(pl.col("LOINC") == code)
                    .then(pl.col("value"))
                    .otherwise(None)
                    .alias(f"{code}")
                    for code in LOINC_codes
                )
                data = getattr(
                    X2_Winsorizer, f"winsorize_{winsorization_method}"
                )(
                    self, # needs explicit self reference
                    data,
                    [f"{code}" for code in LOINC_codes],
                    **kwargs,
                )
                data = data.with_columns(
                    pl.coalesce(
                        pl.col(f"{code}") for code in LOINC_codes
                    ).alias("value"),
                ).select(
                    *[
                        column
                        for column in column_names
                        if column != winsorization_column
                    ],
                    pl.struct(
                        [
                            pl.col("value"),
                            pl.col("system"),
                            pl.col("method"),
                            pl.col("time"),
                            pl.col("LOINC"),
                        ]
                    ).alias(winsorization_column),
                )

        sys.stdout.write("\033[K")  # Clear to the end of line
        print("reprodICU - Winsorization complete.")

        return data


if __name__ == "__main__":
    raise NotImplementedError(
        "This script is not yet implemented as a command line tool."
    )
