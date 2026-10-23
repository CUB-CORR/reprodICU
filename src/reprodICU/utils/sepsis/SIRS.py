"""
SIRS: compute SIRS in long format directly from raw inputs.

Output columns per row:
- Global ICU Stay ID
- T_0 (seconds from admission used as reference)
- timeframe (0-indexed integer window)
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

from ..common import (
    _null_if_incomplete_pre_icu,
    T_MINUS1_DEFAULT,
    _add_sparse_timeframes,
    FILTER_NOT_NULL,
    _assign_timeframe,
    _build_base_timeframes,
    _build_t0,
    _get_timeframe_name,
    _optional_time_bounds_filter,
    _validate_required_data,
    get_patient_information,
    get_timeseries_labs,
    get_timeseries_vitals,
)

SECONDS_IN_1H = 60 * 60
SECONDS_IN_1D = 24 * SECONDS_IN_1H


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
def SIRS(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_vitals: Optional[pl.LazyFrame] = None,
    timeseries_labs: Optional[pl.LazyFrame] = None,
    *,
    t_0: Optional[int] = 0,
    t_0_per_stay: Optional[pl.LazyFrame] = None,
    t_1: Optional[int] = None,
    t_minus1: Optional[int] = T_MINUS1_DEFAULT,
    window_size: int = SECONDS_IN_1D,
    timeframe_unit: str = "Days",  # semantics only; output timeframe is numeric
    forward_fill: bool = True,
    timeframe_name: str = None,
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
        t_0 : int, optional
            Scalar reference time (seconds from admission). Defaults to 0 (admission).
            Ignored when t_0_per_stay is provided.
        t_0_per_stay : pl.LazyFrame, optional
            Per-stay T_0 overrides with columns [Global ICU Stay ID, T_0].
        t_1 : int, optional
            Optional upper time bound (seconds from admission) for filtering inputs.
        t_minus1 : int, optional
            Optional lower time bound (seconds from admission, may be negative);
            timeframes ending at or before it are dropped. Defaults to 30 days before
            admission; None keeps everything.
        window_size : int, optional
            Timeframe width in seconds (default: 86400 = 1 day). Window index is
            floor((time - T_0)/window_size).
        timeframe_unit : str, optional
            Semantic only; output column remains a numeric timeframe.
        forward_fill : bool, optional
            Whether to forward-fill values within windows. Defaults to True.
        timeframe_name : str, optional
            Name for output timeframe column. Auto-generated if None.

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
    STAY_KEY = "Global ICU Stay ID"
    TIME_KEY = "Time Relative to Admission (seconds)"

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
    ALL_STAYS = patient_information.select(STAY_KEY)
    ALL_STAYS_T0 = _build_t0(ALL_STAYS, t_0_per_stay=t_0_per_stay, t_0=t_0)
    timeframe_name = _get_timeframe_name(
        timeframe_name, window_size, t_0, t_0_per_stay
    )

    # region temperature criterion
    temp_tf = (
        vitals.select(STAY_KEY, TIME_KEY, temp_col)
        .drop_nulls(temp_col)
        .join(ALL_STAYS_T0, on=STAY_KEY, how="inner")
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window_size))
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
        .join(ALL_STAYS_T0, on=STAY_KEY, how="inner")
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window_size))
        .group_by(STAY_KEY, "timeframe")
        .agg(_hr_criterion(pl.col(hr_col)).max().alias("hr_criterion"))
    )

    # region respiratory rate criterion
    # RR (vitals) and PaCO2 (labs) computed separately, then ORed per window
    rr_ge20_tf = (
        vitals.select(STAY_KEY, TIME_KEY, rr_col)
        .drop_nulls(rr_col)
        .join(ALL_STAYS_T0, on=STAY_KEY, how="inner")
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window_size))
        .group_by(STAY_KEY, "timeframe")
        .agg((pl.col(rr_col).cast(pl.Float64) > 20).max().alias("rr_ge20"))
    )

    paco2_lt32_tf = (
        labs.select(STAY_KEY, TIME_KEY, paco2_col)
        .drop_nulls(paco2_col)
        .join(ALL_STAYS_T0, on=STAY_KEY, how="inner")
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window_size))
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
        .join(ALL_STAYS_T0, on=STAY_KEY, how="inner")
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window_size))
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
    out = _add_sparse_timeframes(_build_base_timeframes(ALL_STAYS_T0, patient_information, window_size), temp_tf, hr_tf, resp_tf, wbc_tf)
    for part in [temp_tf, hr_tf, resp_tf, wbc_tf]:
        out = out.join(part, on=[STAY_KEY, "timeframe"], how="left")

    if forward_fill:
        # carried indefinitely within the stay, but not from before ICU admission
        icu_start = (-pl.col("T_0")).floordiv(window_size)
        out = out.with_columns(
            pl.when(
                pl.col(col).is_null()
                & pl.when(pl.col(col).is_not_null())
                .then(pl.col("timeframe"))
                .forward_fill()
                .ge(icu_start)
            )
            .then(pl.col(col).forward_fill())
            .otherwise(pl.col(col))
            .over(partition_by=STAY_KEY, order_by="timeframe")
            .alias(col)
            for col in (
                "temperature_criterion",
                "hr_criterion",
                "rr_criterion",
                "wbc_criterion",
            )
        )

    return (
        out.filter(
            _optional_time_bounds_filter("timeframe", window_size, t_0, t_1, t_minus1=t_minus1)
        )
        .with_columns(
            _null_if_incomplete_pre_icu(
                pl.sum_horizontal(
                pl.col("temperature_criterion"),
                pl.col("hr_criterion"),
                pl.col("rr_criterion"),
                pl.col("wbc_criterion"),
                ignore_nulls=True,
            ),
                pl.col(["temperature_criterion", "hr_criterion", "rr_criterion", "wbc_criterion"]),
                window_size,
            )
            .ge(2)
            .alias("SIRS Criteria fulfilled")
        )
        .select(
            STAY_KEY,
            "T_0",
            pl.col("timeframe").alias(timeframe_name),
            "SIRS Criteria fulfilled",
            pl.col("temperature_criterion").alias("SIRS Temperature Criterion"),
            pl.col("hr_criterion").alias("SIRS Heart rate Criterion"),
            pl.col("rr_criterion").alias("SIRS Respiratory rate Criterion"),
            pl.col("wbc_criterion").alias("SIRS WBC Criterion"),
        )
        .sort(STAY_KEY, timeframe_name)
    )


# endregion


__all__ = ["SIRS"]
