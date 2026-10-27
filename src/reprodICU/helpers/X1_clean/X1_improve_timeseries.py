# Author: Finn Fassbender
# Last modified: 2025-10-30

# Description: This script imputes the data to remove missing values.
# It is available as a module for piping in the main script.
# It can be called with command line arguments to specify the source datasets to be imputed. ! NOT IMPLEMENTED YET !

import polars as pl
from ..helper import GlobalVars


class IntakeOutputImprover(GlobalVars):
    def __init__(self, paths) -> None:
        super().__init__(paths)
        pass

    def add_infusion_volumes(
        self, data: pl.LazyFrame, medications: pl.LazyFrame
    ) -> pl.LazyFrame:
        """
        Add the volumes of medications to the intake/output data.

        Steps:
            1. Filter medications for volumes given intravenously, orally or via a tube.
            2. Extract drug amounts in ml and fluid volumes.
            3. Aggregate volumes for mixed infusions (same mixture ID).
            4. Sum drug and fluid volumes to get the total volume.
            5. Add the volumes to the intake column of their route, at their own time.

        Returns:
            pl.LazyFrame: Intake/output data with added medication volumes.
        """

        if medications is None:
            raise ValueError("The medication data is missing.")

        print("reprodICU - Adding infusion volumes to intake/output data...")

        stay = "Global ICU Stay ID"
        time = "Time Relative to Admission (seconds)"
        columns = sorted(set(self.INTAKE_COLUMN_BY_ROUTE.values()))

        # Filter the medication data for volumes (i.e. medications with a volume)
        infused_volumes = medications.filter(
            pl.col(self.drug_admin_route_col).is_in(
                list(self.INTAKE_COLUMN_BY_ROUTE)
            ),
            (pl.col(self.drug_amount_unit_col) == "ml")
            | (pl.col(self.fluid_amount_col).is_not_null()),
        ).with_columns(
            pl.when(pl.col(self.drug_amount_unit_col) == "ml")
            .then(pl.col(self.drug_amount_col))
            .otherwise(0)
            .alias("Drug Amount (ml)"),
        )

        infused_volumes = (
            pl.concat(
                [
                    infused_volumes.filter(
                        pl.col(self.drug_mixture_admin_id_col).is_null()
                    ),
                    infused_volumes.filter(
                        pl.col(self.drug_mixture_admin_id_col).is_not_null(),
                        pl.col(self.drug_mixture_admin_id_col)
                        .is_duplicated()
                        .not_(),
                    ),
                    infused_volumes.filter(
                        pl.col(self.drug_mixture_admin_id_col).is_not_null(),
                        pl.col(self.drug_mixture_admin_id_col).is_duplicated(),
                    )
                    .group_by(self.drug_mixture_admin_id_col)
                    .agg(
                        pl.exclude(
                            "Drug Amount (ml)", self.fluid_amount_col
                        ).first(),
                        pl.col(
                            "Drug Amount (ml)",
                            self.fluid_amount_col,
                        ).sum(),
                    ),
                ],
                how="diagonal_relaxed",
            )
            .with_columns(
                pl.sum_horizontal(
                    pl.col("Drug Amount (ml)"),
                    pl.col(self.fluid_amount_col),
                    ignore_nulls=True,
                ).alias("Infused volume"),
                pl.col(self.drug_admin_route_col)
                .replace_strict(self.INTAKE_COLUMN_BY_ROUTE, return_dtype=pl.String)
                .alias("Intake column"),
            )
        )

        # Sum up the volumes for each time point and intake column. Volumes without an intake
        # output row at their time get a row of their own: joining them to the next row would
        # shift them and drop those after the last row of a stay.
        volumes = (
            pl.concat(
                [
                    infused_volumes.filter(pl.col("Intake column") == column).select(
                        stay,
                        # administrations at a point in time (e.g. HiRID fluid counters)
                        # have no end
                        pl.coalesce(
                            "Drug End Relative to Admission (seconds)",
                            "Drug Start Relative to Admission (seconds)",
                        ).alias(time),
                        pl.col("Infused volume").alias(f"{column} (medication)"),
                    )
                    for column in columns
                ],
                how="diagonal_relaxed",
            )
            .group_by(stay, time)
            .agg(
                *[
                    pl.when(pl.col(f"{column} (medication)").count() > 0)
                    .then(pl.col(f"{column} (medication)").sum())
                    .alias(f"{column} (medication)")
                    for column in columns
                ]
            )
        )

        # Add the volumes to the intake output data
        existing = data.collect_schema().names()
        return (
            data.with_columns(
                [
                    pl.lit(None, dtype=pl.Float64).alias(column)
                    for column in columns
                    if column not in existing
                ]
            )
            .join(volumes, on=[stay, time], how="full", coalesce=True)
            .sort(stay, time)
            .with_columns(
                # no intake data and no medication volume stays null, not 0
                *[
                    pl.when(
                        pl.col(column).is_not_null()
                        | pl.col(f"{column} (medication)").is_not_null()
                    )
                    .then(
                        pl.sum_horizontal(column, f"{column} (medication)")
                    )
                    .alias(column)
                    for column in columns
                ]
            )
            .drop([f"{column} (medication)" for column in columns])
        )

    def improve_intake_output(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """
        Impute the intake output data to add fluid balance information.

        :param data: The intake output data to be calculated.

        :return: The calculated intake output data.
        :rtype: pl.DataFrame
        """

        print("reprodICU - Calculating fluid balances...")

        inout_cols = data.collect_schema().names()
        inout_cols_series = pl.Series(inout_cols)
        intake_cols = inout_cols_series.filter(
            inout_cols_series.str.contains_any(["intake", "Intake"])
        ).to_list()
        output_cols = inout_cols_series.filter(
            inout_cols_series.str.contains_any(["output", "Output"])
        ).to_list()

        # Impute missing values
        return data.with_columns(
            (
                pl.sum_horizontal(
                    pl.lit(0), pl.col(intake_cols), ignore_nulls=True
                )
                - pl.sum_horizontal(
                    pl.lit(0), pl.col(output_cols), ignore_nulls=True
                )
            ).alias("Fluid balance")
        ).with_columns(
            # Calculate the total fluid balance; stays without any intake record have no balance
            pl.when(
                pl.any_horizontal(pl.col(intake_cols).is_not_null())
                .any()
                .over("Global ICU Stay ID")
            )
            .then(pl.col("Fluid balance").cum_sum().over("Global ICU Stay ID"))
            .alias("Fluid balance"),
        )
