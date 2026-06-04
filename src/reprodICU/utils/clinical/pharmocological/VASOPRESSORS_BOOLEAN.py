"""
VASOPRESSORS_BOOLEAN: Create boolean indicator for vasopressor/inotrope use.

Automatically concatenates valid infusions of vasopressors/inotropes to continuous
intervals of use with a positive boolean indicator.
"""

from typing import Optional

import polars as pl

from ...common import (
    _validate_required_data,
    get_medications,
    get_patient_information,
)
from .ALIGNED_UNITS import ALIGNED_UNITS

STAY_KEY = "Global ICU Stay ID"
START_KEY = "Drug Start Relative to Admission (seconds)"
END_KEY = "Drug End Relative to Admission (seconds)"

SECONDS_IN_5MIN = (
    5 * 60
)  # grace period to allow for minor gaps in continuous infusions

VASOPRESSORS = {
    # "vasopressor": (MIN_RATE, MAX_RATE),
    "epinephrine":       (0.005,  2),  # µg/kg/min
    "norepinephrine":    (0.001,  5),  # µg/kg/min
    "phenylephrine":     (0.05,  10),  # µg/kg/min
    "vasopressin (USP)": (0.01,   1),  # U/min
} # fmt: skip


def VASOPRESSORS_BOOLEAN(
    patient_information: Optional[pl.LazyFrame] = None,
    medications: Optional[pl.LazyFrame] = None,
    VASOPRESSORS: dict = VASOPRESSORS,
) -> pl.LazyFrame:
    """
    Normalize drug rates to standard units (mcg/kg/min or U/kg/min).

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information; must contain Global ICU Stay ID and
            Admission Weight (kg). Loaded automatically if None.
        medications : pl.LazyFrame, optional
            Medication administrations with drug ingredient, rate, unit, and
            timing information.
        VASOPRESSORS : dict, optional
            List of drug ingredients to consider as vasopressors/inotropes. Must
            match the "Drug Ingredient" column in the medications dataframe.
            Defaults to a common set of vasoactive agents.

    Returns
    -------
        pl.LazyFrame
            Medications dataframe with additional columns:
            - Drug Rate (fixed units)
            - Drug Rate Unit (fixed units)
            Rows with null fixed units are dropped.
    """
    # Load defaults if not provided
    if patient_information is None:
        patient_information = get_patient_information()
    if medications is None:
        medications = get_medications()

    # Validate all required data is available
    required = {
        "patient_information": patient_information,
        "medications": medications,
    }
    _validate_required_data(concept="VASOPRESSORS_BOOLEAN", required_data=required) # fmt: skip

    # Base frames
    patient_information = patient_information.lazy()
    medications = medications.lazy()

    # Select relevant columns
    medications = medications.filter(
        pl.col("Drug Ingredient").is_in(VASOPRESSORS.keys()),
    ).pipe(ALIGNED_UNITS, patient_information=patient_information)

    # Drop rows missing the fixed‑unit rate and zero‑out anything
    # at or below the minimum for that ingredient
    min_rate_map = {k: v[0] for k, v in VASOPRESSORS.items()}

    medications = medications.drop_nulls(
        ["Drug Rate (fixed units)", "Drug Rate Unit (fixed units)"]
    ).with_columns(
        pl.when(
            pl.col("Drug Rate (fixed units)")
            <= pl.col("Drug Ingredient").replace_strict(
                min_rate_map, default=0, return_dtype=float
            )
        )
        .then(0)
        .otherwise(1)
        .alias("Vasopressor use (boolean)")
    )

    # Combine subsequent administrations of the same drug into continuous intervals of use
    return (
        medications.with_columns(
            pl.col(START_KEY)
            .le(
                pl.col(END_KEY)
                .shift(1)
                .over(
                    partition_by=[STAY_KEY, "Drug Ingredient"],
                    order_by=START_KEY,
                )
                + SECONDS_IN_5MIN
            )
            .fill_null(True)
            .alias("Starts at previous end"),
        )
        .with_columns(
            pl.struct(
                pl.col("Vasopressor use (boolean)"),
                pl.col("Starts at previous end"),
            )
            .rle_id()
            .over(
                partition_by=[STAY_KEY, "Drug Ingredient"],
                order_by=START_KEY,
            )
            .alias("Vasopressor use interval ID"),
        )
        .group_by(STAY_KEY, "Drug Ingredient", "Vasopressor use interval ID")
        .agg(
            pl.col("Vasopressor use (boolean)").max().cast(bool),
            pl.col(START_KEY).min(),
            pl.col(END_KEY).max(),
        )
    )


__all__ = ["VASOPRESSORS_BOOLEAN"]
