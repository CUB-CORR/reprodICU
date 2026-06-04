"""
VIS: compute Vasoactive-Inotropic Score in long format directly from raw inputs.

Output columns per row:
- Global ICU Stay ID
- T_0 (seconds from admission used as reference)
- timeframe (0-indexed integer window)
- Vasoactive-Inotropic Score (VIS) (sum of vasoactive agent contributions)

Time is in seconds. Windows determined by floor((time - T_0)/window_size).

SOURCES
-------
- Belletti, A., Lerose, C. C., Zangrillo, A., & Landoni, G. (2021).
  Vasoactive-inotropic score: Evolution, clinical utility, and pitfalls.
  Journal of Cardiothoracic and Vascular Anesthesia, 35(10), 3067–3077.
  doi:10.1053/j.jvca.2020.09.117
"""

from typing import Optional

import polars as pl

from ..clinical.pharmocological.ALIGNED_UNITS import ALIGNED_UNITS_VIS
from ..common import (
    _build_t0,
    _get_timeframe_name,
    _optional_time_bounds_filter,
    _validate_required_data,
    get_medications,
    get_patient_information,
)
from ..FIX_WINDOW_BORDERS import FIX_WINDOW_BORDERS

STAY_KEY = "Global ICU Stay ID"

SECONDS_IN_1H = 60 * 60
SECONDS_IN_1D = 24 * SECONDS_IN_1H


def VIS(
    patient_information: Optional[pl.LazyFrame] = None,
    medications: Optional[pl.LazyFrame] = None,
    *,
    t_0: Optional[int] = 0,
    t_0_per_stay: Optional[pl.LazyFrame] = None,
    t_1: Optional[int] = None,
    window_size: int = SECONDS_IN_1D,
    timeframe_name: str = None,
) -> pl.LazyFrame:
    """
    Compute Vasoactive-Inotropic Score (VIS) in long format from raw inputs.

    VIS is calculated per hour as the weighted sum of vasoactive agent contributions:
    VIS = (dopamine dose) + (dobutamine dose) + (100 × epinephrine dose) +
          (100 × norepinephrine dose) + ... for 13 supported agents.

    Included vasoactive agents and their contributions:
    - Angiotensin II: 0.25 × dose (ng/kg/min)
    - Dobutamine: dose (mcg/kg/min)
    - Dopamine: dose (mcg/kg/min)
    - Enoximone: dose (mcg/kg/min)
    - Epinephrine: 100 × dose (mcg/kg/min)
    - Levosimendan: 50 × dose (mcg/kg/min)
    - Methylene Blue: 20 × dose (mg/kg/h)
    - Milrinone: 10 × dose (mcg/kg/min)
    - Norepinephrine: 100 × dose (mcg/kg/min)
    - Olprinone: 25 × dose (mcg/kg/min)
    - Phenylephrine: 100 × dose (mcg/kg/min)
    - Terlipressin: 10 × dose (mcg/h)
    - Vasopressin (USP): 10000 × dose (units/kg/min)

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information; must contain Global ICU Stay ID and
            Admission Weight (kg). Loaded automatically if None.
        medications : pl.LazyFrame, optional
            Medication administrations with drug ingredient, rate, unit, and
            timing information. Loaded automatically if None.
        t_0 : int, optional
            Scalar reference time (seconds from admission). Defaults to 0 (admission).
            Ignored when t_0_per_stay is provided.
        t_0_per_stay : pl.LazyFrame, optional
            Per-stay T_0 overrides with columns [Global ICU Stay ID, T_0].
        t_1 : int, optional
            Optional upper time bound (seconds from admission) for filtering inputs.
        window_size : int, optional
            Timeframe width in seconds (default: 86400 = 1 day). Window index is
            floor((time - T_0)/window_size).
        timeframe_name : str, optional
            Name for output timeframe column. Auto-generated if None.

    Returns
    -------
        pl.LazyFrame
            One row per (stay, timeframe) with columns:
            - Global ICU Stay ID
            - T_0
            - timeframe (or custom name)
            - Vasoactive-Inotropic Score (VIS)
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
    _validate_required_data("VIS", required)

    VASOPRESSORS_INOTROPES = [
        "angiotensin II",  # 0.25 * dose in ng/kg/min
        "dobutamine",  # dose in mcg/kg/min
        "dopamine",  # dose in mcg/kg/min
        "enoximone",  # dose in mcg/kg/min
        "epinephrine",  # 100 * dose in mcg/kg/min
        "levosimendan",  # 50 * dose in mcg/kg/min
        "methylene blue",  # 20 * dose in mg/kg/h
        "milrinone",  # 10 * dose in mcg/kg/min
        "norepinephrine",  # 100 * dose in mcg/kg/min
        "olprinone",  # 25 * dose in mcg/kg/min
        "phenylephrine",  # 100 * dose in mcg/kg/min
        "terlipressin",  # 10 * dose in mcg/h
        "vasopressin (USP)",  # 10000 * dose in units/kg/min
    ]

    # Base frames
    patient_information = patient_information.lazy()
    medications = medications.lazy()

    ALL_STAYS = patient_information.select(STAY_KEY)
    ALL_STAYS_T0 = _build_t0(ALL_STAYS, t_0_per_stay=t_0_per_stay, t_0=t_0)
    timeframe_name = _get_timeframe_name(
        timeframe_name, window_size, t_0, t_0_per_stay
    )

    # Select relevant columns and build T_0
    medications = (
        medications.filter(
            pl.col("Drug Ingredient").is_in(VASOPRESSORS_INOTROPES)
        )
        .pipe(ALIGNED_UNITS_VIS, patient_information=patient_information)
        .join(ALL_STAYS_T0, on=STAY_KEY, how="inner")
        .with_columns(
            # Calculate start and end relative to T_0
            (
                pl.col("Drug Start Relative to Admission (seconds)")
                - pl.col("T_0")
            ).alias("Drug Start Relative to T_0 (seconds)"),
            (
                pl.col("Drug End Relative to Admission (seconds)")
                - pl.col("T_0")
            ).alias("Drug End Relative to T_0 (seconds)"),
        )
    )

    # Convert to VIS components
    medications = medications.with_columns(
        pl.when(
            pl.col("Drug Ingredient") == "angiotensin II",
            pl.col("Drug Rate Unit (fixed units)") == "mcg/kg/min",
        )
        .then(pl.col("Drug Rate (fixed units)") * 0.25 * 1000) # mcg -> ng
        .when(
            pl.col("Drug Ingredient").is_in(["dopamine", "dobutamine", "enoximone"]),
            pl.col("Drug Rate Unit (fixed units)") == "mcg/kg/min",
        )
        .then(pl.col("Drug Rate (fixed units)"))
        .when(
            pl.col("Drug Ingredient").is_in(["milrinone", "phenylephrine"]),
            pl.col("Drug Rate Unit (fixed units)") == "mcg/kg/min",
        )
        .then(pl.col("Drug Rate (fixed units)") * 10)
        .when(
            pl.col("Drug Ingredient") == "terlipressin",
            pl.col("Drug Rate Unit (fixed units)") == "mcg/kg/min",
        )
        .then(pl.col("Drug Rate (fixed units)") * pl.col("Admission Weight (kg)"))
        .when(
            pl.col("Drug Ingredient") == "methylene blue",
            pl.col("Drug Rate Unit (fixed units)") == "mg/kg/min",
        )
        .then(pl.col("Drug Rate (fixed units)") * 20 * 60) # min -> h
        .when(
            pl.col("Drug Ingredient") == "olprinone",
            pl.col("Drug Rate Unit (fixed units)") == "mcg/kg/min",
        )
        .then(pl.col("Drug Rate (fixed units)") * 25)
                .when(
            pl.col("Drug Ingredient") == "levosimendan",
            pl.col("Drug Rate Unit (fixed units)") == "mcg/kg/min",
        )
        .then(pl.col("Drug Rate (fixed units)") * 50)
        .when(
            pl.col("Drug Ingredient").is_in(["epinephrine", "norepinephrine"]),
            pl.col("Drug Rate Unit (fixed units)") == "mcg/kg/min",
        )
        .then(pl.col("Drug Rate (fixed units)") * 100)
        .when(
            pl.col("Drug Ingredient") == "vasopressin (USP)",
            pl.col("Drug Rate Unit (fixed units)") == "U/kg/min",
        )
        .then(pl.col("Drug Rate (fixed units)") * 10_000)
        .otherwise(None)
        .alias("VIS Component"),
    ) # fmt: skip

    # Fix window borders
    medications = FIX_WINDOW_BORDERS(
        medications, TIMEWINDOW_IN_SECONDS=SECONDS_IN_1H
    ).rename(
        {
            "Window Relative to T_0": "Hour Relative to T_0",
            "Drug Duration (windows)": "Drug Duration (hours)",
        }
    )

    # Calculate VIS per hour
    vis = (
        medications.group_by(
            STAY_KEY, "Hour Relative to T_0", "Drug Ingredient"
        )
        .agg(
            pl.col("VIS Component")
            .mul(pl.col("Drug Duration (hours)"))
            .sum()
            .truediv(pl.sum("Drug Duration (hours)"))
            .alias("VIS Component"),
            pl.col("T_0").first(),
        )
        .group_by(STAY_KEY, "Hour Relative to T_0")
        .agg(
            pl.sum("VIS Component").alias("Vasoactive-Inotropic Score (VIS)"),
            pl.col("T_0").first(),
        )
        .filter(
            _optional_time_bounds_filter(
                "Hour Relative to T_0", SECONDS_IN_1H, t_0, t_1
            )
        )
        .sort(STAY_KEY, "Hour Relative to T_0")
        .select(
            STAY_KEY,
            "T_0",
            pl.col("Hour Relative to T_0").alias(timeframe_name),
            "Vasoactive-Inotropic Score (VIS)",
        )
    )

    return vis


__all__ = ["VIS"]
