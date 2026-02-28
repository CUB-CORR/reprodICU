"""
HED: compute Hydrocortisone Equivalent Dose from raw inputs in long format.

Output columns per row:
- Global ICU Stay ID
- Drug Start Relative to Admission (seconds)
- Drug End Relative to Admission (seconds)
- Drug Amount
- Drug Amount Unit
- Hydrocortisone Equivalent Amount (mg)

Sources
-------
- https://www.nadf.us/uploads/1/3/0/1/130191972/corticosteroid_comparison_chart.pdf
- https://litfl.com/steroid-conversion/
"""

from typing import Optional

import polars as pl

from ...common import get_medications


def HYDROCORTISONE_EQUIVALENT_DOSAGE(
    medications: Optional[pl.LazyFrame] = None,
) -> pl.LazyFrame:
    """
    Compute Hydrocortisone Equivalent Dose in long format from raw inputs.

    Included corticosteroids and their conversion factors to hydrocortisone:
    - Hydrocortisone: 1 × dose (mg)
    - Betamethasone: 33.33 × dose (mg)
    - Dexamethasone: 30 × dose (mg)
    - Methylprednisolone: 5 × dose (mg)
    - Prednisolone: 4 × dose (mg)
    - Prednisone: 4 × dose (mg)
    - Triamcinolone: 5 × dose (mg)

    Arguments
    ---------
        medications : pl.LazyFrame, optional
            Medication administrations with drug ingredient, amount, unit, and
            timing information. Loaded automatically if None.

    Returns
    -------
        pl.LazyFrame
            One row per medication administration with columns:
            - Global ICU Stay ID
            - Drug Start Relative to Admission (seconds)
            - Drug End Relative to Admission (seconds)
            - Drug Amount
            - Drug Amount Unit
            - Hydrocortisone Equivalent Amount (mg)
    """
    # Load defaults if not provided
    if medications is None:
        medications = get_medications()

    # Validate all required data is available
    if medications is None:
        raise ValueError(
            "Cannot compute HEC: Missing required dataset 'medications'. "
            "Ensure it is configured in ~/.reprodICU/PATHS.yaml or provide it explicitly."
        )

    CORTICOSTEROIDS = [
        "hydrocortisone",
        "betamethasone",
        "dexamethasone",
        "methylprednisolone",
        "prednisolone",
        "prednisone",
        "triamcinolone",
    ]

    # Base frames
    medications = medications.lazy()

    # Select relevant medications
    medications = medications.filter(
        pl.col("Drug Ingredient").is_in(CORTICOSTEROIDS)
    )

    # Convert amounts to mg
    medications = medications.with_columns(
        pl.when(pl.col("Drug Amount Unit") == "g")
        .then(pl.col("Drug Amount") * 1000)
        .when(pl.col("Drug Amount Unit") == "mcg")
        .then(pl.col("Drug Amount") / 1000)
        .otherwise(pl.col("Drug Amount"))
        .alias("Drug Amount (mg)"),
        pl.when(pl.col("Drug Amount Unit").is_in(["g", "mcg"]))
        .then(pl.lit("mg"))
        .otherwise(pl.col("Drug Amount Unit"))
        .alias("Drug Amount Unit (fixed)"),
    )

    # Convert to Hydrocortisone Equivalent Amount
    return medications.with_columns(
        pl.when(pl.col("Drug Ingredient") == "betamethasone")
        .then(pl.col("Drug Amount (mg)") * 33.33)
        .when(pl.col("Drug Ingredient") == "dexamethasone")
        .then(pl.col("Drug Amount (mg)") * 30)
        .when(pl.col("Drug Ingredient").is_in(["methylprednisolone", "triamcinolone"]))
        .then(pl.col("Drug Amount (mg)") * 5)
        .when(pl.col("Drug Ingredient").is_in(["prednisone", "prednisolone"]))
        .then(pl.col("Drug Amount (mg)") * 4)
        .otherwise(pl.col("Drug Amount (mg)"))
        .alias("Hydrocortisone Equivalent Amount (mg)"),
    )


__all__ = ["HYDROCORTISONE_EQUIVALENT_DOSAGE"]
