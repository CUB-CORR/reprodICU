"""
MED: compute Morphine Equivalent Dose from raw inputs in long format.

Output columns per row:
- Global ICU Stay ID
- Drug Start Relative to Admission (seconds)
- Drug End Relative to Admission (seconds)
- Drug Amount
- Drug Amount Unit
- Morphine Equivalent Amount (mg)

Formula: Morphine Equivalent Dose = Drug Amount (mg) / Conversion Factor

Where Conversion Factor is the morphine equivalents (mg) from the table.

Sources
-------
- Provided table of opioid conversion factors
"""

from typing import Optional

import polars as pl

from ...common import get_medications


def MORPHINE_EQUIVALENT_DOSAGE(
    medications: Optional[pl.LazyFrame] = None,
) -> pl.LazyFrame:
    """
    Compute Morphine Equivalent Dose in long format from raw inputs.

    Included opioids and their conversion factors (dose / factor = morphine equivalents):
    - Morphine (Oral): 25
    - Morphine (Parenteral): 10
    - Buprenorphine (SL): 0.4
    - Buprenorphine (Parenteral): 0.3
    - Codeine (Oral): 200
    - Codeine (Parenteral): 100
    - Fentanyl (Parenteral): 0.15
    - Hydrocodone (Oral): 25
    - Hydromorphone (Parenteral): 2
    - Hydromorphone (Oral): 5
    - Meperidine (Oral): 300
    - Meperidine (Parenteral): 100
    - Oxycodone (Oral): 20
    - Oxycodone (Parenteral): 10
    - Oxymorphone (Parenteral): 1
    - Oxymorphone (Oral): 10
    - Tapentadol (Oral): 100
    - Tramadol (Oral): 120
    - Tramadol (Parenteral): 100

    Arguments
    ---------
        medications : pl.LazyFrame, optional
            Medication administrations with drug ingredient, amount, unit, route, and
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
            - Morphine Equivalent Amount (mg)
    """
    # Load defaults if not provided
    if medications is None:
        medications = get_medications()

    # Validate all required data is available
    if medications is None:
        raise ValueError(
            "Cannot compute MED: Missing required dataset 'medications'. "
            "Ensure it is configured in ~/.reprodICU/PATHS.yaml or provide it explicitly."
        )

    OPIOIDS = [
        "morphine",
        "buprenorphine",
        "codeine",
        "fentanyl",
        "hydrocodone",
        "hydromorphone",
        "meperidine",
        "oxycodone",
        "oxymorphone",
        "tapentadol",
        "tramadol",
    ]

    # Base frames
    medications = medications.lazy()

    # Select relevant medications
    medications = medications.filter(
        pl.col("Drug Ingredient").is_in(OPIOIDS)
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

    # Convert to Morphine Equivalent Amount
    return medications.with_columns(
        pl.when(
            pl.col("Drug Ingredient") == "morphine",
            pl.col("Drug Administration Route") == "oral"
        )
        .then(pl.col("Drug Amount (mg)") / 25)
        .when(
            pl.col("Drug Ingredient") == "morphine",
            pl.col("Drug Administration Route").is_in(["intravenous", "parenteral", "subcutaneous"])
        )
        .then(pl.col("Drug Amount (mg)") / 10)
        .when(
            pl.col("Drug Ingredient") == "buprenorphine",
            pl.col("Drug Administration Route") == "sublingual"
        )
        .then(pl.col("Drug Amount (mg)") / 0.4)
        .when(
            pl.col("Drug Ingredient") == "buprenorphine",
            pl.col("Drug Administration Route").is_in(["intravenous", "parenteral", "subcutaneous"])
        )
        .then(pl.col("Drug Amount (mg)") / 0.3)
        .when(
            pl.col("Drug Ingredient") == "codeine",
            pl.col("Drug Administration Route") == "oral"
        )
        .then(pl.col("Drug Amount (mg)") / 200)
        .when(
            pl.col("Drug Ingredient") == "codeine",
            pl.col("Drug Administration Route").is_in(["intravenous", "parenteral", "subcutaneous"])
        )
        .then(pl.col("Drug Amount (mg)") / 100)
        .when(
            pl.col("Drug Ingredient") == "fentanyl",
            pl.col("Drug Administration Route").is_in(["intravenous", "parenteral", "subcutaneous"])
        )
        .then(pl.col("Drug Amount (mg)") / 0.15)
        .when(
            pl.col("Drug Ingredient") == "hydrocodone",
            pl.col("Drug Administration Route") == "oral"
        )
        .then(pl.col("Drug Amount (mg)") / 25)
        .when(
            pl.col("Drug Ingredient") == "hydromorphone",
            pl.col("Drug Administration Route") == "oral"
        )
        .then(pl.col("Drug Amount (mg)") / 5)
        .when(
            pl.col("Drug Ingredient") == "hydromorphone",
            pl.col("Drug Administration Route").is_in(["intravenous", "parenteral", "subcutaneous"])
        )
        .then(pl.col("Drug Amount (mg)") / 2)
        .when(
            pl.col("Drug Ingredient") == "meperidine",
            pl.col("Drug Administration Route") == "oral"
        )
        .then(pl.col("Drug Amount (mg)") / 300)
        .when(
            pl.col("Drug Ingredient") == "meperidine",
            pl.col("Drug Administration Route").is_in(["intravenous", "parenteral", "subcutaneous"])
        )
        .then(pl.col("Drug Amount (mg)") / 100)
        .when(
            pl.col("Drug Ingredient") == "oxycodone",
            pl.col("Drug Administration Route") == "oral"
        )
        .then(pl.col("Drug Amount (mg)") / 20)
        .when(
            pl.col("Drug Ingredient") == "oxycodone",
            pl.col("Drug Administration Route").is_in(["intravenous", "parenteral", "subcutaneous"])
        )
        .then(pl.col("Drug Amount (mg)") / 10)
        .when(
            pl.col("Drug Ingredient") == "oxymorphone",
            pl.col("Drug Administration Route") == "oral"
        )
        .then(pl.col("Drug Amount (mg)") / 10)
        .when(
            pl.col("Drug Ingredient") == "oxymorphone",
            pl.col("Drug Administration Route").is_in(["intravenous", "parenteral", "subcutaneous"])
        )
        .then(pl.col("Drug Amount (mg)") / 1)
        .when(
            pl.col("Drug Ingredient") == "tapentadol",
            pl.col("Drug Administration Route") == "oral"
        )
        .then(pl.col("Drug Amount (mg)") / 100)
        .when(
            pl.col("Drug Ingredient") == "tramadol",
            pl.col("Drug Administration Route") == "oral"
        )
        .then(pl.col("Drug Amount (mg)") / 120)
        .when(
            pl.col("Drug Ingredient") == "tramadol",
            pl.col("Drug Administration Route").is_in(["intravenous", "parenteral", "subcutaneous"])
        )
        .then(pl.col("Drug Amount (mg)") / 100)
        .otherwise(None)
        .alias("Morphine Equivalent Amount (mg)"),
    )


__all__ = ["MORPHINE_EQUIVALENT_DOSAGE"]
