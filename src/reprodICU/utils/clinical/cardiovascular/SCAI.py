"""
SCAI: compute SCAI cardiogenic shock stages in long format directly from raw inputs.

Output columns per row:
- Global ICU Stay ID
- T_0 (seconds from admission used as reference)
- timeframe (0-indexed integer window)
- SCAI Stage (A-E, where E is worst)

Time is in seconds. Windows determined by floor((time - T_0)/window_size).
Worst-within-window aggregation is applied per stage criteria.

SCAI Staging Criteria (simplified):

Stage A (At risk): Normal hemodynamics, no shock signs
  - SBP ≥100 mmHg AND lactate <2 mmol/L AND no vasoactive support
  - If invasive hemodynamics available: CI ≥2.5 L/min/m2, CVP ≤10, PCWP ≤15

Stage B (Beginning CS): Hemodynamic instability without hypoperfusion
  - (SBP <90 OR HR >100) AND lactate <2 mmol/L AND no vasoactive support
  - May include MAP <60 or >30 mmHg drop from baseline

Stage C (Classic CS): Hypoperfusion requiring intervention
  - Lactate ≥2 mmol/L OR creatinine ↑1.5x baseline OR urine output <30 mL/h
  - Requires vasoactive support
  - If invasive hemodynamics: CI <2.2 L/min/m2 or PCWP >15

Stage D (Deteriorating): Worsening shock despite therapy
  - Lactate ≥2 AND escalating/persistent vasopressor requirement

Stage E (Extremis): Circulatory collapse
  - Lactate ≥8 mmol/L OR pH <7.2 OR base deficit >10 mEq/L

SOURCES
-------
- Naidu SS, Baran DA, Jentzer JC, Hollenberg SM, van Diepen S, Basir MB, Grines CL, Diercks DB, Hall S, Kapur NK, Kent W, Rao SV, Samsky MD, Thiele H, Truesdell AG, Henry TD.
  SCAI SHOCK Stage Classification Expert Consensus Update: A Review and Incorporation of Validation Studies.
  J Am Coll Cardiol. 2022 Mar 8;79(9):933-946.
  doi: 10.1016/j.jacc.2022.01.018. Epub 2022 Jan 31. PMID: 35115207.
"""

from typing import Optional

import polars as pl

from ...common import (
    _assign_timeframe,
    _build_t0,
    _optional_time_bounds_filter,
    _to_lazy,
    get_medications,
    get_patient_information,
    get_timeseries_labs,
    get_timeseries_vitals,
)
from ...scores.SOFA2 import get_vasopressor_points

# seconds constants
SECONDS_PER_HOUR = 60 * 60
SECONDS_PER_DAY = 24 * SECONDS_PER_HOUR

# strict time column name used across helpers
STAY_COL = "Global ICU Stay ID"
TIME_COL = "Time Relative to Admission (seconds)"

# SCAI stages as Polars Enum
SCAI_STAGES = pl.Enum(["A", "B", "C", "D", "E"])


# region staging helpers


def _vitals_stage_expr() -> pl.Expr:
    """
    Compute SCAI stages A and B based on vital signs only.

    Stage B: Beginning CS - Hemodynamic instability (SBP<90 or MAP<60 or HR>100)
    Stage A: At risk - SBP ≥100
    """
    return (
        pl.when(
            (pl.col("Systolic arterial pressure") < 90)
            | (pl.col("Mean arterial pressure") < 60)
            | (pl.col("Heart rate") > 100)
        )
        .then(pl.lit("B"))
        .when(pl.col("Systolic arterial pressure") >= 100)
        .then(pl.lit("A"))
        .otherwise(None)
    )


def _labs_stage_expr() -> pl.Expr:
    """
    Compute SCAI stages C, D, E based on laboratory markers only.

    Stage E: Extremis - lactate ≥8 or pH <7.2 or base excess < -10
    Stage D: Worsening hypoperfusion - lactate ≥2
    Stage C: Hypoperfusion - lactate ≥2 or creatinine >1.2
    """
    return (
        pl.when(
            (pl.col("Lactate").struct.field("value") >= 8)
            | (pl.col("pH").struct.field("value") < 7.2)
            | (pl.col("Base excess").struct.field("value") < -10)
        )
        .then(pl.lit("E"))
        .when(pl.col("Lactate").struct.field("value") >= 2)
        .then(pl.lit("D"))
        .when(
            (pl.col("Lactate").struct.field("value") >= 2)
            | (pl.col("Creatinine").struct.field("value") > 1.2)
        )
        .then(pl.lit("C"))
        .otherwise(None)
    )


def _vasopressors_stage_expr() -> pl.Expr:
    """
    Compute SCAI stages C, D, E based on vasopressor requirements only.

    Stage E: High-dose vasopressors (points ≥4)
    Stage D: Medium-dose vasopressors (points ≥3)
    Stage C: Low-dose vasopressors (points ≥2)
    """
    return (
        pl.when(pl.col("vasopressor_points") >= 4)
        .then(pl.lit("E"))
        .when(pl.col("vasopressor_points") >= 3)
        .then(pl.lit("D"))
        .when(pl.col("vasopressor_points") >= 2)
        .then(pl.lit("C"))
        .otherwise(None)
    )


# endregion staging helpers


def SCAI(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_vitals: Optional[pl.LazyFrame] = None,
    timeseries_labs: Optional[pl.LazyFrame] = None,
    medications: Optional[pl.LazyFrame] = None,
    *,
    t_0: Optional[int] = 0,
    t_0_per_stay: Optional[pl.LazyFrame] = None,
    t_1: Optional[int] = None,
    window_size: int = SECONDS_PER_HOUR,
    timeframe_unit: str = "Hours",
) -> pl.LazyFrame:
    """
    Compute SCAI cardiogenic shock stages with automatic dataset loading.

    All data parameters are optional and will be automatically loaded from the
    package datasets if not provided. This makes it convenient for quick analysis
    while maintaining flexibility for custom data.

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient information dataset. Loaded automatically if None.
        timeseries_vitals : pl.LazyFrame, optional
            Timeseries vitals data. Loaded automatically if None.
        timeseries_labs : pl.LazyFrame, optional
            Timeseries labs data. Loaded automatically if None.
        medications : pl.LazyFrame, optional
            Medications data. Loaded automatically if None.
        t_0 : int, optional
            Scalar reference time (seconds from admission). Defaults to 0 (admission).
            Ignored when t_0_per_stay is provided.
        t_0_per_stay : pl.LazyFrame, optional
            Per-stay T_0 overrides with columns [Global ICU Stay ID, T_0].
        t_1 : int, optional
            Optional upper time bound (seconds from admission) for filtering inputs.
        window_size : int, optional
            Timeframe width in seconds (default: 3600 = 1 hour). Window index is
            floor((time - T_0)/window_size).
        timeframe_unit : str, optional
            Semantic only; output column remains a numeric timeframe.

    Returns
    -------
        pl.LazyFrame
            SCAI stages with all stage components (A-E).
    """
    if patient_information is None:
        patient_information = get_patient_information()
    if timeseries_vitals is None:
        timeseries_vitals = get_timeseries_vitals()
    if timeseries_labs is None:
        timeseries_labs = get_timeseries_labs()
    if medications is None:
        medications = get_medications()

    patient_information = _to_lazy(patient_information)
    timeseries_vitals = _to_lazy(timeseries_vitals)
    timeseries_labs = _to_lazy(timeseries_labs)
    medications = _to_lazy(medications)

    # Build T_0
    all_stays_t0 = _build_t0(
        patient_information.select(STAY_COL),
        t_0=t_0,
        t_0_per_stay=t_0_per_stay,
    )

    # region vitals (SBP, MAP, HR)
    vitals_tf = (
        timeseries_vitals.select(
            STAY_COL,
            TIME_COL,
            pl.coalesce(
                "Invasive systolic arterial pressure",
                "Non-invasive systolic arterial pressure",
            ).alias("Systolic arterial pressure"),
            pl.coalesce(
                "Invasive mean arterial pressure",
                "Non-invasive mean arterial pressure",
            ).alias("Mean arterial pressure"),
            "Heart rate",
        )
        .join(all_stays_t0, on=STAY_COL, how="inner")
        .filter(pl.col(TIME_COL) >= pl.col("T_0").sub(SECONDS_PER_DAY * 7))
        .with_columns(
            _assign_timeframe(TIME_COL, window_size).alias("timeframe"),
            _vitals_stage_expr().alias("vitals_stage"),
        )
        .group_by(STAY_COL, "timeframe")
        .agg(pl.col("vitals_stage").max())
    )

    # region labs (Lactate, pH, Base excess, Creatinine)
    labs_tf = (
        timeseries_labs.with_columns(
            pl.when(
                pl.col("Creatinine")
                .struct.field("system")
                .str.contains_any(["Blood", "Serum"])
            )
            .then(pl.col("Creatinine"))
            .alias("Creatinine")
        )
        .select(
            STAY_COL, TIME_COL, "Lactate", "Creatinine", "pH", "Base excess"
        )
        .join(all_stays_t0, on=STAY_COL, how="inner")
        .filter(pl.col(TIME_COL) >= pl.col("T_0").sub(SECONDS_PER_DAY * 7))
        .with_columns(
            _assign_timeframe(TIME_COL, window_size).alias("timeframe"),
            _labs_stage_expr().alias("labs_stage"),
        )
        .group_by(STAY_COL, "timeframe")
        .agg(pl.col("labs_stage").max())
    )

    # region vasopressors (vasopressor_points per timeframe)
    vp_tf = (
        get_vasopressor_points(
            medications,
            patient_information,
            all_stays_t0,
            t_1=t_1,
            window_size=window_size,
        )
        .select(STAY_COL, "timeframe", "vasopressor_points")
        .with_columns(pl.col("vasopressor_points").fill_null(0))
        .with_columns(_vasopressors_stage_expr().alias("vasopressors_stage"))
        .group_by(STAY_COL, "timeframe")
        .agg(pl.col("vasopressors_stage").max())
    )

    # Generate timeframe name
    unit = (
        "Days"
        if window_size == SECONDS_PER_DAY
        else "Hours" if window_size == SECONDS_PER_HOUR else "Windows"
    )
    reference = "T_0" if t_0 != 0 or t_0_per_stay is not None else "Admission"
    timeframe_name = f"{unit} Relative to {reference}"

    los_col = "ICU Length of Stay (days)"

    # Base frames
    base = (
        all_stays_t0.join(patient_information, on=STAY_COL, how="left")
        .select(STAY_COL, "T_0", los_col)
        .with_columns(
            pl.int_ranges(
                start=0 - pl.col("T_0").floordiv(window_size).sub(1),
                end=pl.col(los_col)
                .mul(SECONDS_PER_DAY)
                .sub("T_0")
                .floordiv(window_size)
                .add(1),
                step=1,
            )
            .cast(pl.List(float))
            .alias("timeframe")
        )
        .explode("timeframe")
        .unique()
        .select(STAY_COL, "T_0", "timeframe")
    )

    # Assemble all staging components
    out = base
    for part in [vitals_tf, labs_tf, vp_tf]:
        out = out.join(part, on=[STAY_COL, "timeframe"], how="left")

    return (
        out.filter(
            _optional_time_bounds_filter("timeframe", window_size, t_0, t_1)
        )
        .with_columns(
            pl.max_horizontal(
                pl.col("vitals_stage"),
                pl.col("labs_stage"),
                pl.col("vasopressors_stage"),
            ).alias("SCAI Stage")
        )
        .select(
            STAY_COL,
            "T_0",
            pl.col("timeframe").alias(timeframe_name),
            "SCAI Stage",
        )
        .sort(STAY_COL, timeframe_name)
    )


__all__ = ["SCAI"]
