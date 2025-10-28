# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script cleans the data by manuall removing invalid values and outliers

import polars as pl


class X1_Cleaner:
    def __init__(self) -> None:
        pass

    def clean_timeseries_labs(data) -> pl.LazyFrame:
        """
        Clean the timeseries labs data by removing invalid values and outliers.

        :param data: The timeseries labs data to be cleaned.

        :return: The cleaned timeseries labs data.
        :rtype: pl.LazyFrame
        """

        # Remove rows with invalid values
        return data.with_columns(
            # Remove invalid INR values
            pl.when(pl.col("INR") == 999999.0)
            .then(None)
            .otherwise(pl.col("INR"))
            .alias("INR"),
            # Remove invalid basophils values
            pl.when(pl.col("basophils") == 999999.0)
            .then(None)
            .otherwise(pl.col("basophils"))
            .alias("basophils"),
            # Remove invalid bicarbonate values
            pl.when(pl.col("bicarbonate") == 999999.0)
            .then(None)
            .otherwise(pl.col("bicarbonate"))
            .alias("bicarbonate"),
            # Remove invalid blood urea nitrogen values
            pl.when(pl.col("blood_urea_nitrogen") == 999999.0)
            .then(None)
            .otherwise(pl.col("blood_urea_nitrogen"))
            .alias("blood_urea_nitrogen"),
            # Remove invalid chloride values
            pl.when(pl.col("chloride") == 999999.0)
            .then(None)
            .otherwise(pl.col("chloride"))
            .alias("chloride"),
            # Remove invalid creatinine values
            pl.when(pl.col("creatinine") == 999999.0)
            .then(None)
            .otherwise(pl.col("creatinine"))
            .alias("creatinine"),
            # Remove invalid glucose values
            pl.when(
                (pl.col("glucose") == -251.0)
                | (pl.col("glucose") == 999999.0)
                | (pl.col("glucose") == 1276103.0)
            )
            .then(None)
            .otherwise(pl.col("glucose"))
            .alias("glucose"),
            # Remove invalid bedside glucose values
            pl.when(pl.col("glucose_bedside") == 15454.0)
            .then(None)
            .otherwise(pl.col("glucose_bedside"))
            .alias("glucose_bedside"),
            # Remove invalid hematocrit values
            pl.when(pl.col("hematocrit") == 999999.0)
            .then(None)
            .otherwise(pl.col("hematocrit"))
            .alias("hematocrit"),
            # Remove invalid hemoglobin values
            pl.when(pl.col("hemoglobin") == 999999.0)
            .then(None)
            .otherwise(pl.col("hemoglobin"))
            .alias("hemoglobin"),
            # Remove invalid lactate values
            pl.when(
                (pl.col("lactate") == 999999.0)
                | (pl.col("lactate") == 1276103.0)
            )
            .then(None)
            .otherwise(pl.col("lactate"))
            .alias("lactate"),
            # Remove invalid platelets values
            pl.when(pl.col("lymphocytes") == 999999.0)
            .then(None)
            .otherwise(pl.col("lymphocytes"))
            .alias("lymphocytes"),
            # Remove invalid magnesium values
            pl.when(pl.col("magnesium") == 999999.0)
            .then(None)
            .otherwise(pl.col("magnesium")),
            # Remove invalid monocytes values
            pl.when(pl.col("monocytes") == 999999.0)
            .then(None)
            .otherwise(pl.col("monocytes"))
            .alias("monocytes"),
            # Remove invalid neutrophils values
            pl.when(pl.col("neutrophils") == 999999.0)
            .then(None)
            .otherwise(pl.col("neutrophils"))
            .alias("neutrophils"),
            # Remove invalid pH values
            pl.when(pl.col("pH") == 999999.0)
            .then(None)
            .otherwise(pl.col("pH"))
            .alias("pH"),
            # Remove invalid paCO2 values
            pl.when(pl.col("paCO2") == 999999.0)
            .then(None)
            .otherwise(pl.col("paCO2"))
            .alias("paCO2"),
            # Remove invalid paO2 values
            pl.when(pl.col("paO2") == 999999.0)
            .then(None)
            .otherwise(pl.col("paO2"))
            .alias("paO2"),
            # Remove invalid partial thromboplastin time values
            pl.when(pl.col("partial_thromboplastin_time") == 999999.0)
            .then(None)
            .otherwise(pl.col("partial_thromboplastin_time"))
            .alias("partial_thromboplastin_time"),
            # Remove invalid phosphate values
            pl.when(pl.col("phosphate") == 999999.0)
            .then(None)
            .otherwise(pl.col("phosphate"))
            .alias("phosphate"),
            # Remove invalid potassium values
            pl.when(pl.col("potassium") == 61259.0)
            .then(None)
            .otherwise(pl.col("potassium"))
            .alias("potassium"),
            # Remove invalid albumin values
            pl.when(pl.col("protein_albumin") == 999999.0)
            .then(None)
            .otherwise(pl.col("protein_albumin"))
            .alias("protein_albumin"),
            # Remove invalid saO2 values
            pl.when(pl.col("saO2") == 999999.0)
            .then(None)
            .otherwise(pl.col("saO2"))
            .alias("saO2"),
            # Remove invalid sodium values
            pl.when(pl.col("sodium") == 999999.0)
            .then(None)
            .otherwise(pl.col("sodium"))
            .alias("sodium"),
            # Remove invalid urine specific gravity values
            pl.when(pl.col("urine_specific_gravity") == 1025.0)
            .then(None)
            .otherwise(pl.col("urine_specific_gravity"))
            .alias("urine_specific_gravity"),
        )


if __name__ == "__main__":
    raise NotImplementedError(
        "This script is not yet implemented as a command line tool."
    )
