# Author: Finn Fassbender
# Last modified: 2025-10-30

# Description: This script cleans the data by manuall removing invalid values and outliers

import polars as pl


class X1_Cleaner:
    def __init__(self) -> None:
        pass

    def clean_timeseries_labs(data) -> pl.LazyFrame:
        """
        Remove invalid lab measurement values (sentinel values like 999999.0).

        Steps:
            1. Identify invalid values for each lab parameter (sentinel values, physiologically impossible).
            2. Replace invalid values with null.
            3. Return cleaned timeseries data.

        Returns:
            pl.LazyFrame: Cleaned timeseries data with columns:
                - INR: International Normalized Ratio.
                - basophils: Basophil percentage.
                - bicarbonate: Serum bicarbonate (mEq/L).
                - blood_urea_nitrogen: BUN (mg/dL).
                - chloride: Serum chloride (mEq/L).
                - creatinine: Serum creatinine (mg/dL).
                - glucose: Serum glucose (mg/dL).
                - glucose_bedside: Point-of-care glucose (mg/dL).
                - hematocrit: Hematocrit percentage.
                - hemoglobin: Hemoglobin (g/dL).
                - lactate: Serum lactate (mmol/L).
                - lymphocytes: Lymphocyte percentage.
                - magnesium: Serum magnesium (mg/dL).
                - monocytes: Monocyte percentage.
                - neutrophils: Neutrophil percentage.
                - pH: Blood pH.
                - paCO2: Arterial CO2 partial pressure (mmHg).
                - paO2: Arterial O2 partial pressure (mmHg).
                - partial_thromboplastin_time: PTT (seconds).
                - phosphate: Serum phosphate (mg/dL).
                - potassium: Serum potassium (mEq/L).
                - protein_albumin: Serum albumin (g/dL).
                - saO2: Arterial oxygen saturation (%).
                - sodium: Serum sodium (mEq/L).
                - urine_specific_gravity: Urine specific gravity.
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
