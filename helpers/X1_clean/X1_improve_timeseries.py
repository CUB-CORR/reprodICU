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

    def add_infusion_volumes(
        self, data: pl.LazyFrame, medications: pl.LazyFrame
    ) -> pl.LazyFrame:
        """
        Add the infusion volumes from the medication table to the intake output data.

        Args:
            data (pl.LazyFrame): The intake output data.
            medications (pl.LazyFrame): The medication data.

        Returns:
            pl.LazyFrame: The intake output data with added infusion volumes.
        """

        if medications is None:
            raise ValueError("The medication data is missing.")

        print("reprodICU - Adding infusion volumes to intake/output data...")

        # Filter the medication data for infusions (i.e. medications with a volume)
        infused_volumes = medications.filter(
            (pl.col("Drug Amount Unit") == "ml")
            | (pl.col("Solution Fluid Amount (ml)").is_not_null()),
            pl.col(self.drug_class_col).str.contains_any(["fluid", "drug", None]),
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
                            self.drug_amount_col, self.fluid_amount_col
                        ).first(),
                        pl.col(
                            self.drug_amount_col, self.fluid_amount_col
                        ).sum(),
                    ),
                ],
                how="diagonal_relaxed",
            )
            .with_columns(
                pl.sum_horizontal(
                    pl.col("Drug Amount"),
                    pl.col("Solution Fluid Amount (ml)"),
                    ignore_nulls=True,
                ).alias("Infused volume")
            )
            .select(
                "Global ICU Stay ID",
                "Infused volume",
                "Drug End Relative to Admission (seconds)",
            )
        )

        # Sum up the infusion volumes for each intake output time point for joining
        intake_output_times = data.select(
            "Global ICU Stay ID", "Time Relative to Admission (seconds)"
        ).with_columns(
            pl.col("Time Relative to Admission (seconds)")
            .shift()
            .over("Global ICU Stay ID")
            .fill_null(-float("inf"))  # Fill the first row with -inf
            .alias("previous Time Relative to Admission (seconds)")
        )

        infused_volumes = (
            infused_volumes.join_asof(
                intake_output_times,
                by="Global ICU Stay ID",
                left_on="Drug End Relative to Admission (seconds)",
                right_on="Time Relative to Admission (seconds)",
                strategy="forward",
            )
            .group_by(
                "Global ICU Stay ID", "Time Relative to Admission (seconds)"
            )
            .agg(pl.sum("Infused volume"))
        )

        # Add the infusion volumes to the intake output data
        return (
            data.join(
                infused_volumes,
                on=[
                    "Global ICU Stay ID",
                    "Time Relative to Admission (seconds)",
                ],
                how="left",
            )
            .with_columns(
                pl.sum_horizontal(
                    "Fluid intake intravascular", "Infused volume"
                ).alias("Fluid intake intravascular")
            )
            .drop("Infused volume")
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
