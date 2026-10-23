"""
SCAI: compute SCAI cardiogenic shock stages in long format directly from raw inputs.

Output columns per row:
- Global ICU Stay ID
- T_0 (seconds from admission used as reference)
- timeframe (integer window index counted from T_0)
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

from ...base import (
    SECONDS_IN_1H,
    STAY_KEY,
    TIME_KEY,
    Window,
    _assemble,
    _assign_timeframe,
    _finalize,
    _to_lazy,
    _window_context,
    with_windows,
)
from ...common import (
    _validate_required_data,
    get_medications,
    get_patient_information,
    get_timeseries_labs,
    get_timeseries_vitals,
)
from ...core import BLOOD_PRESSURES
from ...scores.SOFA2 import get_vasopressor_points

# strict time column name used across helpers

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


@with_windows(window_size=SECONDS_IN_1H)
def SCAI(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_vitals: Optional[pl.LazyFrame] = None,
    timeseries_labs: Optional[pl.LazyFrame] = None,
    medications: Optional[pl.LazyFrame] = None,
    *,
    window: Window,
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

    # Validate all required data is available
    required = {
        "patient_information": patient_information,
        "timeseries_vitals": timeseries_vitals,
        "timeseries_labs": timeseries_labs,
        "medications": medications,
    }
    _validate_required_data("SCAI", required)

    patient_information = _to_lazy(patient_information)
    timeseries_vitals = _to_lazy(timeseries_vitals)
    timeseries_labs = _to_lazy(timeseries_labs)
    medications = _to_lazy(medications)

    ctx = _window_context(patient_information, window)

    # region vitals (SBP, MAP, HR)
    vitals_tf = (
        BLOOD_PRESSURES(timeseries_vitals=timeseries_vitals)
        .rename({"SBP": "Systolic arterial pressure", "MAP": "Mean arterial pressure"})
        .select(
            STAY_KEY,
            TIME_KEY,
            "Systolic arterial pressure",
            "Mean arterial pressure",
            "Heart rate",
        )
        .join(ctx.all_stays_t0, on=STAY_KEY, how="inner")
        .with_columns(
            _assign_timeframe(TIME_KEY, window.window_size).alias("timeframe"),
            _vitals_stage_expr().alias("vitals_stage"),
        )
        .group_by(STAY_KEY, "timeframe")
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
            STAY_KEY, TIME_KEY, "Lactate", "Creatinine", "pH", "Base excess"
        )
        .join(ctx.all_stays_t0, on=STAY_KEY, how="inner")
        .with_columns(
            _assign_timeframe(TIME_KEY, window.window_size).alias("timeframe"),
            _labs_stage_expr().alias("labs_stage"),
        )
        .group_by(STAY_KEY, "timeframe")
        .agg(pl.col("labs_stage").max())
    )

    # region vasopressors (vasopressor_points per timeframe)
    vp_tf = (
        get_vasopressor_points(
            medications,
            patient_information,
            ctx.all_stays_t0,
            t_minus1=window.t_minus1,
            t_1=window.t_1,
            window_size=window.window_size,
        )
        .select(STAY_KEY, "timeframe", "vasopressor_points")
        .with_columns(pl.col("vasopressor_points").fill_null(0))
        .with_columns(_vasopressors_stage_expr().alias("vasopressors_stage"))
        .group_by(STAY_KEY, "timeframe")
        .agg(pl.col("vasopressors_stage").max())
    )

    # Assemble all staging components
    out = _assemble(ctx, vitals_tf, labs_tf, vp_tf)

    return _finalize(
        out.with_columns(
            pl.max_horizontal(
                pl.col("vitals_stage"),
                pl.col("labs_stage"),
                pl.col("vasopressors_stage"),
            ).alias("SCAI Stage")
        ),
        ctx,
        "SCAI Stage",
        rest=False,
    )


__all__ = ["SCAI"]
