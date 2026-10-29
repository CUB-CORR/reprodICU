"""
SIRS: compute SIRS in long format directly from raw inputs.

Output columns per row:
- Global ICU Stay ID
- T_0 (seconds from admission used as reference)
- timeframe (integer window index counted from T_0)
- Temperature criterion (1 if <36°C or >38°C, 0 otherwise)
- Heart rate criterion (1 if >90 bpm, 0 otherwise)
- Respiratory rate criterion (1 if >20 bpm or PaCO2 <32 mmHg, 0 otherwise)
- White blood cells criterion (1 if >12,000/mm^3, <4,000/mm^3, or >10% bands, 0 otherwise)
- SIRS Criteria (True if ≥2 criteria met)

Time is in seconds. Windows determined by floor((time - T_0)/window_size).
Worst-within-window aggregation is applied per criterion.

SOURCES
-------
- Bone RC, Balk RA, Cerra FB, Dellinger RP, Fein AM, Knaus WA, Schein RM, Sibbald WJ.
  Definitions for sepsis and organ failure and guidelines for the use of innovative therapies in sepsis.
  The ACCP/SCCM Consensus Conference Committee.
  American College of Chest Physicians/Society of Critical Care Medicine.
  Chest. 1992 Jun;101(6):1644-55. doi: 10.1378/chest.101.6.1644. PMID: 1303622.
"""

from typing import Optional

import polars as pl

from ..base import (
    STAY_KEY,
    TIME_KEY,
    Window,
    _assemble,
    _assign_timeframe,
    _finalize,
    _null_if_incomplete_pre_icu,
    _window_context,
    with_windows,
)
from ..common import (
    FILTER_NOT_NULL,
    _validate_required_data,
    get_patient_information,
    get_timeseries_labs,
    get_timeseries_vitals,
)
from ..scores.SOFA import _forward_fill_within


################################################################################
################################################################################
# region data helpers
def _improve_labs(labs: pl.LazyFrame) -> pl.LazyFrame:
    return labs.with_columns(
        pl.col("Leukocytes").struct.field("value").alias("Leukocytes"),
        pl.col("Neutrophils.band form/leukocytes")
        .struct.field("value")
        .alias("Neutrophils.band form/leukocytes"),
        pl.when(
            pl.col("Carbon dioxide")
            .struct.field("system")
            .is_in(["Blood", "Blood arterial"])
        )
        .then(pl.col("Carbon dioxide").struct.field("value"))
        .alias("Carbon dioxide"),
    ).pipe(
        FILTER_NOT_NULL,
        ["Leukocytes", "Neutrophils.band form/leukocytes", "Carbon dioxide"],
    )


################################################################################
################################################################################
# region criterion helpers
def _temperature_criterion(temp: pl.Expr) -> pl.Expr:
    """
    Temperature (°C)
    >38    1
    <36    1
    else   0
    """
    return (
        pl.when((temp > 38) | (temp < 36))
        .then(1)
        .when(temp.is_not_null())
        .then(0)
        .otherwise(None)
    )


def _hr_criterion(hr: pl.Expr) -> pl.Expr:
    """
    Heart rate (bpm)
    >90    1
    else   0
    """
    return (
        pl.when(hr > 90).then(1).when(hr.is_not_null()).then(0).otherwise(None)
    )


def _rr_criterion(rr: pl.Expr, paco2: pl.Expr) -> pl.Expr:
    """
    Respiratory rate (bpm) or PaCO2 (mmHg)
    RR >20 or PaCO2 <32    1
    else                   0
    """
    return (
        pl.when((rr > 20) | (paco2 < 32))
        .then(1)
        .when(rr.is_not_null() | paco2.is_not_null())
        .then(0)
        .otherwise(None)
    )


def _wbc_criterion(wbc: pl.Expr, bands: pl.Expr) -> pl.Expr:
    """
    White blood cells (×10^3/mm^3) or bands (%)
    WBC >12 or <4, or bands >10    1
    else                           0
    """
    return (
        pl.when((wbc > 12) | (wbc < 4) | (bands > 10))
        .then(1)
        .when(wbc.is_not_null() | bands.is_not_null())
        .then(0)
        .otherwise(None)
    )


################################################################################
################################################################################


# region SIRS
@with_windows
def SIRS(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_vitals: Optional[pl.LazyFrame] = None,
    timeseries_labs: Optional[pl.LazyFrame] = None,
    *,
    forward_fill: bool = True,
    window: Window,
) -> pl.LazyFrame:
    """
    Compute SIRS criteria with automatic dataset loading.

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
        forward_fill : bool, optional
            Whether to forward-fill values within windows. Defaults to True.

    Sources
    -------
    - Bone RC, Balk RA, Cerra FB, et al.
      Definitions for sepsis and organ failure and guidelines for the use of innovative therapies in sepsis.
      The ACCP/SCCM Consensus Conference Committee. American College of Chest Physicians/Society of Critical Care Medicine.
      Chest. 1992 Jun;101(6):1644-55.
      doi: 10.1378/chest.101.6.1644

    Returns
    -------
        pl.LazyFrame
            SIRS criteria with all criterion components
    """
    # Load defaults if not provided
    if patient_information is None:
        patient_information = get_patient_information()
    if timeseries_vitals is None:
        timeseries_vitals = get_timeseries_vitals()
    if timeseries_labs is None:
        timeseries_labs = get_timeseries_labs()

    # Validate all required data is available
    required = {
        "patient_information": patient_information,
        "timeseries_vitals": timeseries_vitals,
        "timeseries_labs": timeseries_labs,
    }
    _validate_required_data("SIRS", required)

    # Strict original column names

    # Vitals columns
    vitals = timeseries_vitals.lazy()
    hr_col = "Heart rate"
    rr_col = "Respiratory rate"
    temp_col = "Temperature"

    # Labs
    labs = _improve_labs(timeseries_labs.lazy())
    wbc_col = "Leukocytes"
    bands_col = "Neutrophils.band form/leukocytes"
    paco2_col = "Carbon dioxide"

    # Base frames
    patient_information = patient_information.lazy()
    ctx = _window_context(patient_information, window)

    # region temperature criterion
    temp_tf = (
        vitals.select(STAY_KEY, TIME_KEY, temp_col)
        .drop_nulls(temp_col)
        .join(ctx.all_stays_t0, on=STAY_KEY, how="inner")
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window.window_size))
        .group_by(STAY_KEY, "timeframe")
        .agg(
            _temperature_criterion(pl.col(temp_col))
            .max()
            .alias("temperature_criterion")
        )
    )

    # region heart rate criterion
    hr_tf = (
        vitals.select(STAY_KEY, TIME_KEY, hr_col)
        .drop_nulls(hr_col)
        .join(ctx.all_stays_t0, on=STAY_KEY, how="inner")
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window.window_size))
        .group_by(STAY_KEY, "timeframe")
        .agg(_hr_criterion(pl.col(hr_col)).max().alias("hr_criterion"))
    )

    # region respiratory rate criterion
    # RR (vitals) and PaCO2 (labs) computed separately, then ORed per window
    rr_ge20_tf = (
        vitals.select(STAY_KEY, TIME_KEY, rr_col)
        .drop_nulls(rr_col)
        .join(ctx.all_stays_t0, on=STAY_KEY, how="inner")
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window.window_size))
        .group_by(STAY_KEY, "timeframe")
        .agg((pl.col(rr_col).cast(pl.Float64) > 20).max().alias("rr_ge20"))
    )

    paco2_lt32_tf = (
        labs.select(STAY_KEY, TIME_KEY, paco2_col)
        .drop_nulls(paco2_col)
        .join(ctx.all_stays_t0, on=STAY_KEY, how="inner")
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window.window_size))
        .group_by(STAY_KEY, "timeframe")
        .agg((pl.col(paco2_col).cast(pl.Float64) < 32).max().alias("paco2_lt32"))
    )

    resp_tf = (
        rr_ge20_tf.join(paco2_lt32_tf, on=[STAY_KEY, "timeframe"], how="full", coalesce=True)
        .select(
            STAY_KEY,
            "timeframe",
            pl.when(
                (pl.col("rr_ge20").fill_null(False)) | (pl.col("paco2_lt32").fill_null(False))
            )
            .then(1)
            .otherwise(0)
            .alias("rr_criterion"),
        )
    )

    # region white blood cells criterion
    wbc_tf = (
        labs.select(STAY_KEY, TIME_KEY, wbc_col, bands_col)
        .join(ctx.all_stays_t0, on=STAY_KEY, how="inner")
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window.window_size))
        .group_by(STAY_KEY, "timeframe")
        .agg(
            _wbc_criterion(
                pl.col(wbc_col).cast(pl.Float64),
                pl.col(bands_col).cast(pl.Float64),
            )
            .max()
            .alias("wbc_criterion")
        )
    )

    # region assemble
    out = _assemble(ctx, temp_tf, hr_tf, resp_tf, wbc_tf)

    if forward_fill:
        # carried indefinitely within the stay, but not from before ICU admission
        out = out.with_columns(
            _forward_fill_within(
                ["temperature_criterion", "hr_criterion", "rr_criterion", "wbc_criterion"],
                max_windows=2**62,
                stay_key=STAY_KEY,
                icu_start=(-pl.col("T_0")).floordiv(window.window_size),
            )
        )

    return _finalize(
        out.with_columns(
            _null_if_incomplete_pre_icu(
                pl.sum_horizontal(
                pl.col("temperature_criterion"),
                pl.col("hr_criterion"),
                pl.col("rr_criterion"),
                pl.col("wbc_criterion"),
                ignore_nulls=True,
            ),
                pl.col(["temperature_criterion", "hr_criterion", "rr_criterion", "wbc_criterion"]),
                window.window_size,
            )
            .ge(2)
            .alias("SIRS Criteria fulfilled")
        ),
        ctx,
        "SIRS Criteria fulfilled",
        pl.col("temperature_criterion").alias("SIRS Temperature Criterion"),
        pl.col("hr_criterion").alias("SIRS Heart rate Criterion"),
        pl.col("rr_criterion").alias("SIRS Respiratory rate Criterion"),
        pl.col("wbc_criterion").alias("SIRS WBC Criterion"),
        rest=False,
    )


# endregion


__all__ = ["SIRS"]
