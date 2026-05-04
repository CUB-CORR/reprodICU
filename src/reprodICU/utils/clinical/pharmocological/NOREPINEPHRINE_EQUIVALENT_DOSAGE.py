"""
NEE: compute Norepinephrine Equivalent Dose from raw inputs in long format.

Output columns per row:
- Global ICU Stay ID
- T_0 (seconds from admission used as reference)
- timeframe (0-indexed integer window)
- Norepinephrine Equivalent Dose (mcg/kg/min)

Time is in seconds. Windows determined by floor((time - T_0)/window_size).

Sources
-------
- Kotani Y, Di Gioia A, Landoni G, Belletti A, Khanna AK.
  An updated "norepinephrine equivalent" score in intensive care as a marker of shock severity.
  Crit Care. 2023 Jan 20;27(1):29.
  doi: 10.1186/s13054-023-04322-y. PMID: 36670410; PMCID: PMC9854213.
"""

from typing import Optional

import polars as pl

from ...common import get_medications, get_patient_information
from .ALIGNED_UNITS import ALIGNED_UNITS

SECONDS_IN_1H = 60 * 60
SECONDS_IN_1D = 24 * SECONDS_IN_1H

VASOPRESSORS_INOTROPES = [
    "angiotensin II",  # 0.0025 * dose in ng/kg/min
    "dopamine",  # 1/100 * dose in mcg/kg/min
    "epinephrine",  # 1 * dose in mcg/kg/min
    "hydroxocobalamin",  # 0.02 * dose in g
    "metaraminol",  # 1/8 * dose in mcg/kg/min
    "methylene blue",  # 0.2 * dose in mg/kg/h
    "midodrine",  # 0.4 * dose in mcg/kg/min
    "norepinephrine",  # 1 * dose in mcg/kg/min
    "phenylephrine",  # 0.06 * dose in mcg/kg/min
    "terlipressin",  # 10 * dose in mcg/h
    "vasopressin (USP)",  # 2.5 * dose in units/kg
]


def NOREPINEPHRINE_EQUIVALENT_DOSAGE(
    patient_information: Optional[pl.LazyFrame] = None,
    medications: Optional[pl.LazyFrame] = None,
) -> pl.LazyFrame:
    """
    Compute Norepinephrine Equivalent Dose in long format from raw inputs.

    Included vasoactive agents and their weights:
    - Angiotensin II: 0.0025 × dose (ng/kg/min)
    - Dopamine: 1/100 × dose (mcg/kg/min)
    - Epinephrine: 1 × dose (mcg/kg/min)
    - Hydroxocobalamin: 0.02 × dose (g)
    - Metaraminol: 1/8 × dose (mcg/kg/min)
    - Methylene Blue: 0.2 × dose (mg/kg/h)
    - Midodrine: 0.4 × dose (mcg/kg/min)
    - Norepinephrine: 1 × dose (mcg/kg/min)
    - Phenylephrine: 0.06 × dose (mcg/kg/min)
    - Terlipressin: 10 × dose (mcg/h)
    - Vasopressin (USP): 2.5 × dose (units/kg/min)

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information; must contain Global ICU Stay ID and
            Admission Weight (kg). Loaded automatically if None.
        medications : pl.LazyFrame, optional
            Medication administrations with drug ingredient, rate, unit, and
            timing information. Loaded automatically if None.

    Returns
    -------
        pl.LazyFrame
            One row per (stay, timeframe) with columns:
            - Global ICU Stay ID
            - Drug Start Relative to Admission (seconds)
            - Drug End Relative to Admission (seconds)
            - Drug Rate
            - Drug Rate Unit
            - Norepinephrine Equivalent Dose (mcg/kg/min)
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

    missing = [name for name, data in required.items() if data is None]
    if missing:
        raise ValueError(
            f"Cannot compute VIS: Missing required datasets: {', '.join(missing)}. "
            f"Ensure they are configured in ~/.reprodICU/PATHS.yaml or provide them explicitly."
        )

    # Base frames
    patient_information = patient_information.lazy()
    medications = medications.lazy()

    # Select relevant columns
    medications = medications.filter(
        pl.col("Drug Ingredient").is_in(VASOPRESSORS_INOTROPES)
    ).pipe(ALIGNED_UNITS, patient_information=patient_information)

    # Convert to Norepinephrine Equivalent Dose
    return medications.with_columns(
        pl.when(
            pl.col("Drug Ingredient") == "angiotensin II",
            pl.col("Drug Rate Unit (fixed units)") == "mcg/kg/min",
        )
        .then(pl.col("Drug Rate (fixed units)") * 0.0025)
        .when(
            pl.col("Drug Ingredient") == "dopamine",
            pl.col("Drug Rate Unit (fixed units)") == "mcg/kg/min",
        )
        .then(pl.col("Drug Rate (fixed units)") * 0.01)
        .when(
            pl.col("Drug Ingredient") == "metaraminol",
            pl.col("Drug Rate Unit (fixed units)") == "mcg/kg/min",
        )
        .then(pl.col("Drug Rate (fixed units)") * 0.125)
        .when(
            pl.col("Drug Ingredient") == "midodrine",
            pl.col("Drug Rate Unit (fixed units)") == "mcg/kg/min",
        )
        .then(pl.col("Drug Rate (fixed units)") * 0.4)
        .when(
            pl.col("Drug Ingredient") == "phenylephrine",
            pl.col("Drug Rate Unit (fixed units)") == "mcg/kg/min",
        )
        .then(pl.col("Drug Rate (fixed units)") * 0.06)
        .when(
            pl.col("Drug Ingredient") == "terlipressin",
            pl.col("Drug Rate Unit (fixed units)") == "mcg/kg/min",
        )
        .then(pl.col("Drug Rate (fixed units)") * 10)
        .when(
            pl.col("Drug Ingredient") == "methylene blue",
            pl.col("Drug Rate Unit (fixed units)") == "mcg/kg/min",
        )
        .then(pl.col("Drug Rate (fixed units)") * 0.2 / 60)
        .when(
            pl.col("Drug Ingredient") == "hydroxocobalamin",
            pl.col("Drug Rate Unit (fixed units)") == "mcg/kg/min",
        )
        .then(pl.col("Drug Rate (fixed units)") * 0.02)
        .when(
            pl.col("Drug Ingredient").is_in(["epinephrine", "norepinephrine"]),
            pl.col("Drug Rate Unit (fixed units)") == "mcg/kg/min",
        )
        .then(pl.col("Drug Rate (fixed units)") * 1)
        .when(
            pl.col("Drug Ingredient") == "vasopressin (USP)",
            pl.col("Drug Rate Unit (fixed units)") == "U/min",
        )
        .then(pl.col("Drug Rate (fixed units)") * 2.5)
        .otherwise(None)
        .alias("Norepinephrine Equivalent Dose (mcg/kg/min)"),
    )


__all__ = ["NOREPINEPHRINE_EQUIVALENT_DOSAGE"]
