"""
ALIGNED_UNITS: Normalize drug rates to standard units.

Standardizes drug rates to:
- mcg/kg/min (for most drugs)
- U/kg/min (for insulin/vasopressin/etc)

Handles conversion from various input units (mcg/min, mg/hr, etc.) and
calculates rates from amounts if rate is missing but amount/duration is known.
"""

from typing import Optional

import polars as pl

from ...common import get_patient_information

STAY_KEY = "Global ICU Stay ID"
WEIGHT_COL = "Admission Weight (kg)"


def ALIGNED_UNITS(
    medications: pl.LazyFrame,
    patient_information: Optional[pl.LazyFrame] = None,
) -> pl.LazyFrame:
    """
    Normalize drug rates to standard units (mcg/kg/min or U/min).

    Arguments
    ---------
        medications : pl.LazyFrame
            Medication administrations with drug ingredient, rate, unit, and
            timing information.
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information; must contain Global ICU Stay ID and
            Admission Weight (kg). Loaded automatically if None.

    Returns
    -------
        pl.LazyFrame
            Medications dataframe with additional columns:
            - Drug Rate (fixed units)
            - Drug Rate Unit (fixed units)
            Rows with null fixed units are dropped.
    """
    if patient_information is None:
        patient_information = get_patient_information()

    # Base frames
    patient_information = patient_information.lazy()
    medications = medications.lazy()

    # Select relevant columns
    weights = patient_information.select(STAY_KEY, WEIGHT_COL)

    # Fix rates - handle cases where Drug Amount is provided but Drug Rate is not
    PREDICATES = (
        pl.col("Drug Rate").is_null(),
        pl.col("Drug Rate Unit").is_null(),
        pl.col("Drug Amount").is_not_null(),
        pl.col("Drug Amount Unit").is_in(
            ["g", "mg", "mcg", "U", "IE", "units"]
        ),
    )
    medications = medications.with_columns(
        pl.when(*PREDICATES)
        .then(
            pl.col("Drug Amount")
            / (
                pl.col("Drug End Relative to Admission (seconds)")
                - pl.col("Drug Start Relative to Admission (seconds)")
            ).truediv(60)
        )
        .otherwise(pl.col("Drug Rate"))
        .alias("Drug Rate"),
        pl.when(*PREDICATES)
        .then(pl.concat_str(pl.col("Drug Amount Unit"), pl.lit("/min")))
        .otherwise(pl.col("Drug Rate Unit"))
        .alias("Drug Rate Unit"),
    )

    # Fix units - normalize all rates to mcg/kg/min
    medications = (
        medications.join(weights, on=STAY_KEY, how="left")
        .with_columns(
            # CONVERTING UNITS
            # Convert mcg / mg / g to mcg/kg/min
            pl.when(pl.col("Drug Rate Unit") == "mcg/min")
            .then(pl.col("Drug Rate") / pl.col(WEIGHT_COL))
            .when(pl.col("Drug Rate Unit") == "mcg/hr")
            .then(pl.col("Drug Rate") / pl.col(WEIGHT_COL) / 60)
            .when(pl.col("Drug Rate Unit") == "mcg/kg/hr")
            .then(pl.col("Drug Rate") / 60)
            .when(pl.col("Drug Rate Unit") == "mg/hr")
            .then(pl.col("Drug Rate") * 1000 / pl.col(WEIGHT_COL) / 60)
            .when(pl.col("Drug Rate Unit") == "mg/min")
            .then(pl.col("Drug Rate") * 1000 / pl.col(WEIGHT_COL))
            .when(pl.col("Drug Rate Unit") == "mg/kg/min")
            .then(pl.col("Drug Rate") * 1000)
            .when(pl.col("Drug Rate Unit") == "g/hr")
            .then(pl.col("Drug Rate") * 1_000_000 / pl.col(WEIGHT_COL) / 60)
            .when(pl.col("Drug Rate Unit") == "g/min")
            .then(pl.col("Drug Rate") * 1_000_000 / pl.col(WEIGHT_COL))
            .when(pl.col("Drug Rate Unit") == "g/kg/hr")
            .then(pl.col("Drug Rate") * 1_000_000 / 60)
            .when(pl.col("Drug Rate Unit") == "g/kg/min")
            .then(pl.col("Drug Rate") * 1_000_000)
            # Convert Units
            .when(pl.col("Drug Rate Unit").is_in(["U/hr", "units/hr"]))
            .then(pl.col("Drug Rate") / 60)
            .when(pl.col("Drug Rate Unit").is_in(["U/min", "units/min", "IE/min"]))
            .then(pl.col("Drug Rate"))
            # Keep unchanged
            .when(pl.col("Drug Rate Unit") == "mcg/kg/min")
            .then(pl.col("Drug Rate"))
            .otherwise(None)
            .alias("Drug Rate (fixed units)"),
            # RENAMING UNITS
            pl.when(
                pl.col("Drug Rate Unit").is_in(
                    [
                        "mcg/kg/min",
                        "mcg/min",
                        "mcg/hr",
                        "mcg/kg/hr",
                        "mg/hr",
                        "mg/min",
                        "mg/kg/min",
                        "g/hr",
                        "g/min",
                        "g/kg/hr",
                        "g/kg/min",
                    ]
                )
            )
            .then(pl.lit("mcg/kg/min"))
            .when(
                pl.col("Drug Rate Unit").is_in(
                    ["U/min", "U/hr", "units/hr", "units/min", "IE/min"]
                )
            )
            .then(pl.lit("U/min"))
            .otherwise(None)
            .alias("Drug Rate Unit (fixed units)"),
        )
        .drop_nulls(["Drug Rate (fixed units)", "Drug Rate Unit (fixed units)"])
    )

    return medications


def ALIGNED_UNITS_VIS(
    medications: pl.LazyFrame,
    patient_information: Optional[pl.LazyFrame] = None,
) -> pl.LazyFrame:
    """
    Normalize drug rates to standard units (mcg/kg/min or U/kg/min).
    Used for VIS calculation where vasopressin should be in U/kg/min, not U/min.

    Arguments
    ---------
        medications : pl.LazyFrame
            Medication administrations with drug ingredient, rate, unit, and
            timing information.
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information; must contain Global ICU Stay ID and
            Admission Weight (kg). Loaded automatically if None.

    Returns
    -------
        pl.LazyFrame
            Medications dataframe with additional columns:
            - Drug Rate (fixed units)
            - Drug Rate Unit (fixed units)
            Rows with null fixed units are dropped.
    """

    return ALIGNED_UNITS(medications, patient_information).with_columns(
        pl.when(pl.col("Drug Rate Unit (fixed units)") == "U/min")
        .then(pl.col("Drug Rate (fixed units)") / pl.col(WEIGHT_COL))
        .otherwise(pl.col("Drug Rate (fixed units)"))
        .alias("Drug Rate (fixed units)"),
        pl.when(pl.col("Drug Rate Unit (fixed units)") == "U/min")
        .then(pl.lit("U/kg/min"))
        .otherwise(pl.col("Drug Rate Unit (fixed units)"))
        .alias("Drug Rate Unit (fixed units)"),
    )


__all__ = ["ALIGNED_UNITS", "ALIGNED_UNITS_VIS"]
