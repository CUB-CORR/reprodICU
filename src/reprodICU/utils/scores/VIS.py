"""
VIS: compute Vasoactive-Inotropic Score in long format directly from raw inputs.

Output columns per row:
- Global ICU Stay ID
- T_0 (seconds from admission used as reference)
- timeframe (integer window index counted from T_0)
- Vasoactive-Inotropic Score (VIS) (hourly VIS aggregated per window)

Time is in seconds. Windows determined by floor((time - T_0)/window_size).
VIS is computed per hour (no vasoactives = 0) and aggregated per window.

SOURCES
-------
- Belletti, A., Lerose, C. C., Zangrillo, A., & Landoni, G. (2021).
  Vasoactive-inotropic score: Evolution, clinical utility, and pitfalls.
  Journal of Cardiothoracic and Vascular Anesthesia, 35(10), 3067–3077.
  doi:10.1053/j.jvca.2020.09.117
"""

from typing import Optional

import polars as pl

from ..base import (
    SECONDS_IN_1H,
    STAY_KEY,
    Window,
    _add_sparse_timeframes,
    _build_base_timeframes,
    _optional_timeframe_bounds_filter,
    _window_context,
    with_windows,
)
from ..clinical.pharmocological.ALIGNED_UNITS import ALIGNED_UNITS_VIS
from ..common import (
    _validate_required_data,
    get_medications,
    get_patient_information,
)
from ..FIX_WINDOW_BORDERS import PROJECT_TO_WINDOWS, TIME_WEIGHTED_MEAN_RATE




@with_windows
def VIS(
    patient_information: Optional[pl.LazyFrame] = None,
    medications: Optional[pl.LazyFrame] = None,
    *,
    aggregation: str = "mean",
    window: Window,
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
    - Phenylephrine: 10 × dose (mcg/kg/min)
    - Terlipressin: 10 × dose (mcg/min)
    - Vasopressin (USP): 10000 × dose (units/kg/min)

    The windows are aggregated from hourly values, so `window_size` must be a multiple of 3600.

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information; must contain Global ICU Stay ID and
            Admission Weight (kg). Loaded automatically if None.
        medications : pl.LazyFrame, optional
            Medication administrations with drug ingredient, rate, unit, and
            timing information. Loaded automatically if None.
        aggregation : {"mean", "median"}, optional
            How hourly VIS values are aggregated over a window (default: "mean").

    Returns
    -------
        pl.LazyFrame
            One row per (stay, timeframe) with columns:
            - Global ICU Stay ID
            - T_0
            - timeframe (or custom name)
            - Vasoactive-Inotropic Score (VIS)
    """
    if window.window_size % SECONDS_IN_1H != 0:
        raise ValueError(f"window_size must be a multiple of 3600, got {window.window_size}")
    if aggregation not in ("mean", "median"):
        raise ValueError(f"aggregation must be 'mean' or 'median', got {aggregation!r}")

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

    ctx = _window_context(patient_information, window)

    # Select relevant columns and build T_0
    medications = (
        medications.filter(
            pl.col("Drug Ingredient").is_in(VASOPRESSORS_INOTROPES)
        )
        .pipe(ALIGNED_UNITS_VIS, patient_information=patient_information)
        .join(ctx.all_stays_t0, on=STAY_KEY, how="inner")
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
        .then(pl.col("Drug Rate (fixed units)") * pl.col("Admission Weight (kg)") / 1000 * 10) # mcg/kg/min -> mg/min, x10
        .when(
            pl.col("Drug Ingredient") == "methylene blue",
            pl.col("Drug Rate Unit (fixed units)") == "mcg/kg/min",
        )
        .then(pl.col("Drug Rate (fixed units)") * 60 / 1000 * 20) # mcg/kg/min -> mg/kg/h, x20
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

    meds_agg = PROJECT_TO_WINDOWS(
        medications,
        group_cols=[STAY_KEY, "Drug Ingredient", "T_0"],
        value_col="VIS Component",
        TIMEWINDOW_IN_SECONDS=SECONDS_IN_1H,
        prefix="Drug",
        reference="T_0",
        unit="seconds",
    ).with_columns(
        pl.col("Window Relative to T_0").alias("Hour Relative to T_0"),
    )

    # Calculate VIS per hour: sum each drug's own time-weighted average contribution
    hourly_vis = (
        meds_agg.group_by(STAY_KEY, "Drug Ingredient", "Hour Relative to T_0")
        .agg(TIME_WEIGHTED_MEAN_RATE(SECONDS_IN_1H).alias("VIS Component"))
        .group_by(STAY_KEY, "Hour Relative to T_0")
        .agg(pl.sum("VIS Component").alias("Vasoactive-Inotropic Score (VIS)"))
    )

    # All hours of the stay (no vasoactives = VIS 0), aggregated per window
    vis = (
        _add_sparse_timeframes(
            _build_base_timeframes(ctx.all_stays_t0, patient_information, SECONDS_IN_1H),
            hourly_vis.rename({"Hour Relative to T_0": "timeframe"}),
        )
        .rename({"timeframe": "Hour Relative to T_0"})
        .filter(
            _optional_timeframe_bounds_filter(
                "Hour Relative to T_0", SECONDS_IN_1H, t_minus1=window.t_minus1, t_1=window.t_1
            )
        )
        .join(hourly_vis, on=[STAY_KEY, "Hour Relative to T_0"], how="left")
        .with_columns(pl.col("Vasoactive-Inotropic Score (VIS)").fill_null(0.0))
        .group_by(
            STAY_KEY,
            "T_0",
            (pl.col("Hour Relative to T_0") // (window.window_size // SECONDS_IN_1H)).alias(ctx.name),
        )
        .agg(getattr(pl.col("Vasoactive-Inotropic Score (VIS)"), aggregation)())
        .sort(STAY_KEY, ctx.name)
    )

    return vis


__all__ = ["VIS"]
