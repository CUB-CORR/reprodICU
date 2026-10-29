"""
AKI_KDIGO: compute KDIGO-based Acute Kidney Injury stages in long format directly from raw inputs.

Output columns per row:
- Global ICU Stay ID
- T_0 (seconds from admission used as reference)
- timeframe (integer window index counted from T_0)
- Creatinine AKI Stage (0-3 based on KDIGO creatinine criteria)
- UO Consecutive AKI Stage (0-3 based on consecutive urine output method, if enabled)
- UO Any Period AKI Stage (0-3 based on any period urine output method, if enabled)
- UO Fixed Block AKI Stage (0-3 based on fixed block urine output method, if enabled)
- Overall KDIGO AKI Stage (max of all available stages)

Time is in seconds. Windows determined by floor((time - T_0)/window_size).
Worst-within-window aggregation is applied per stage method.

Creatinine is staged per measurement, urine output over hourly windows. With
version="2026" an admission baseline is only a reference after its +12h window and
the hindsight fallback "lowest value of the admission" requires hindsight_baseline=True.

Similar in implementation to pyAKI, but adapted for polars and reprodICU framework.
- cf. with demo data from https://github.com/aidh-ms/pyAKI/blob/main/tests/data/validation_data.csv
- MIMIC-IV IDs: 30849778, 32119961, 32314488, 32391858, 33281088, 34592300, 34617352, 35009126, 35258379, 35514836, 37267577, 38383343, 38540883, 39268883, 39635619

Sources
-------
- KDIGO 2024 Clinical Practice Guideline for the Evaluation and Management of Chronic Kidney Disease
  (see doi:10.1016/j.kint.2024.06.014)
- Kidney Disease: Improving Global Outcomes (KDIGO) Acute Kidney Injury Work Group.
  KDIGO Clinical Practice Guideline for Acute Kidney Injury. Kidney Int Suppl. 2012;2(1):1–138.
- KDIGO 2026 Clinical Practice Guideline for Acute Kidney Injury (AKI) and Acute Kidney Disease (AKD).
  Public Review Draft, March 2026. Practice Point 1.1.3 (baseline SCr hierarchy).
"""

from typing import Optional

import polars as pl

from ...base import (
    SECONDS_IN_12H,
    SECONDS_IN_1D,
    SECONDS_IN_1H,
    SECONDS_IN_1W,
    STAY_KEY,
    TIME_KEY,
    Window,
    _add_sparse_timeframes,
    _assign_timeframe,
    _build_base_timeframes,
    _to_lazy,
    _window_context,
    intervention_per_timeframe,
    with_windows,
)
from ...common import (
    FILTER_NOT_NULL,
    NULL_NON_FINITE,
    _validate_required_data,
    extract_struct_value,
    get_patient_information,
    get_rrt,
    get_timeseries_intakeoutput,
    get_timeseries_labs,
)
from ...laboratory.renal.CREATININE import reverse_CKD_EPI, reverse_MDRD
from ..IDEAL_BODY_WEIGHT import IDEAL_BODY_WEIGHT_DEVINE
from .URINE_OUTPUT import URINE_OUTPUT_RATE

HOSPITAL_KEY = "Global Hospital Stay ID"
HOSPITAL_TIME_KEY = "Time Relative to Hospital Admission (seconds)"
PRE_ICU_COL = "Pre-ICU Length of Stay (days)"
TIMEFRAME_KEY = "timeframe"


################################################################################
################################################################################
# region creatinine baselines
def _prepare_creatinine_baseline_2012(
    creatinine_data: pl.LazyFrame,
    patient_information: pl.LazyFrame,
) -> pl.LazyFrame:
    """Establish creatinine baseline for AKI staging per KDIGO 2012 criteria.

    KDIGO 2012 requires a baseline creatinine within 7 days prior to admission.
    This function implements a hierarchy:
    1. Actual measurement within 7 days pre-admission (preferred)
    2. MDRD-estimated creatinine at eGFR=75 (fallback if no prior measurement)

    Inputs are keyed by hospital stay on the hospital clock, as in
    `_prepare_creatinine_baseline_2026`.

    Returns
    -------
        LazyFrame with columns:
        - Global Hospital Stay ID
        - SCr Baseline: Serum creatinine baseline value
        - SCr Baseline source: Where baseline came from (measurement or estimate)

    Notes
    -----
        MDRD estimation assumes normal kidney function (eGFR=75) if no prior data exists.
        This is a conservative assumption for patients without documented baseline.
    """
    # STEP 1: Extract prior creatinine measurements from 7 days before to 12 hours after admission
    # Using absolute distance to time=0 (admission) to find closest measurement
    prior_data = (
        creatinine_data.filter(
            pl.col(HOSPITAL_TIME_KEY).is_between(-SECONDS_IN_1W, SECONDS_IN_12H)
        )
        .group_by(HOSPITAL_KEY)
        .agg(
            pl.col("Creatinine")
            .sort_by(pl.col(HOSPITAL_TIME_KEY).abs(), pl.col("Creatinine"))
            .first()  # closest to admission; ties -> lower value
            .alias("SCr Baseline prior measurement"),
            pl.lit("Prior measurement").alias(
                "Baseline source prior measurement"
            ),
        )
    )

    # STEP 2: Estimate baseline creatinine using MDRD equation
    # If no prior measurement exists, use MDRD to reverse-calculate creatinine
    # assuming eGFR of 75 (normal kidney function)
    mdrd_baseline = (
        patient_information.select(HOSPITAL_KEY, "Admission Age (years)", "Gender", "Ethnicity")
        .pipe(reverse_MDRD, target_egfr=75)
        .select(
            HOSPITAL_KEY,
            pl.col("estimated Creatinine (MDRD eGFR=75)").alias("SCr Baseline MDRD"),
            pl.lit("MDRD estimate (eGFR=75)").alias("Baseline source MDRD"),
        )
    ) # fmt: skip

    # STEP 3: Combine both sources with priority order
    # Prefer actual measurement over estimate (coalesce: first non-null wins)
    return (
        patient_information.select(HOSPITAL_KEY)
        .join(prior_data, on=HOSPITAL_KEY, how="left")
        .join(mdrd_baseline, on=HOSPITAL_KEY, how="left")
        .with_columns(
            # Priority: prior measurement > MDRD estimate
            pl.coalesce(
                pl.col("SCr Baseline prior measurement"),
                pl.col("SCr Baseline MDRD"),
            ).alias("SCr Baseline"),
            # Track source for transparency
            pl.coalesce(
                pl.col("Baseline source prior measurement"),
                pl.col("Baseline source MDRD"),
            ).alias("SCr Baseline source"),
        )
        .select(HOSPITAL_KEY, "SCr Baseline", "SCr Baseline source")
    )


def _prepare_creatinine_baseline_2026(
    creatinine_data: pl.LazyFrame,
    patient_information: pl.LazyFrame,
    rrt_start: Optional[pl.LazyFrame] = None,
    hindsight_baseline: bool = False,
) -> pl.LazyFrame:
    """Establish creatinine baseline for AKI staging per KDIGO 2026 draft (Practice Point 1.1.3).

    Implements the baseline hierarchy, the first available step wins:

    1. Outpatient measurement: Median of values more than 7 days and up to 365 days
       pre-admission ("8-365 days" counted in whole days)
    2. Admission measurement: Value closest to admission (within ±12 hours).
       A reference only after this window, see `_creatinine_stage_points`.
    3. Lowest inpatient value (only if `hindsight_baseline`): Lowest value of the
       admission before the first RRT, including values after the AKI. Uses
       measurements after the time being staged and KDIGO considers it less desirable.
    4. CKD-EPI estimate: Creatinine at eGFR=75 mL/min/1.73m² if no measurement exists

    The baseline belongs to the hospital stay: admission is hospital admission and all
    inputs are on the hospital clock (see `_prepare_creatinine`).

    Arguments
    ---------
        creatinine_data : pl.LazyFrame
            LazyFrame with columns [Global Hospital Stay ID, Time Relative to Hospital Admission (seconds), Creatinine]
        patient_information : pl.LazyFrame
            LazyFrame with Global Hospital Stay ID, age and gender
        rrt_start : pl.LazyFrame, optional
            LazyFrame with columns [Global Hospital Stay ID, first_rrt_start]
            (required if `hindsight_baseline`)
        hindsight_baseline : bool
            Include step 3

    Returns
    -------
        LazyFrame with columns:
        - Global Hospital Stay ID
        - SCr Baseline: Serum creatinine baseline value
        - SCr Baseline source: Step of the hierarchy the baseline came from
    """
    near_admission = creatinine_data.filter(
        pl.col(HOSPITAL_TIME_KEY).is_between(-SECONDS_IN_12H, SECONDS_IN_12H)
    )
    sources = [
        # STEP 1: Median of outpatient creatinine more than 7 days and up to 365 days before admission
        # (the day count starts at 7 days before admission, so no value falls between this step and the 7 days before admission)
        creatinine_data.filter(
            pl.col(HOSPITAL_TIME_KEY).is_between(-365 * SECONDS_IN_1D, -SECONDS_IN_1W, closed="left")
        )
        .group_by(HOSPITAL_KEY)
        .agg(pl.col("Creatinine").median().alias("SCr Baseline"))
        .with_columns(pl.lit("Outpatient").alias("SCr Baseline source")),
        # STEP 2: Creatinine closest to admission (within ±12 hours), ties -> lower value
        near_admission.group_by(HOSPITAL_KEY)
        .agg(
            pl.col("Creatinine")
            .sort_by(pl.col(HOSPITAL_TIME_KEY).abs(), pl.col("Creatinine"))
            .first()
            .alias("SCr Baseline")
        )
        .with_columns(pl.lit("Admission measurement").alias("SCr Baseline source")),
    ]
    if hindsight_baseline:
        # STEP 3 (hindsight_baseline only): lowest value from hospital admission until the first RRT
        sources.append(
            creatinine_data.join(rrt_start, on=HOSPITAL_KEY, how="left")
            .filter(
                pl.col(HOSPITAL_TIME_KEY) >= 0,
                pl.col("first_rrt_start").is_null()
                | (pl.col(HOSPITAL_TIME_KEY) < pl.col("first_rrt_start")),
            )
            .group_by(HOSPITAL_KEY)
            .agg(pl.col("Creatinine").min().alias("SCr Baseline"))
            .with_columns(pl.lit("Lowest inpatient").alias("SCr Baseline source"))
        )
    # STEP 4: CKD-EPI estimate (eGFR = 75 mL/min/1.73m²)
    # Use reverse CKD-EPI formula to estimate creatinine at eGFR=75
    sources.append(
        patient_information.select(HOSPITAL_KEY, "Admission Age (years)", "Gender")
        .pipe(reverse_CKD_EPI, target_egfr=75)
        .select(
            HOSPITAL_KEY,
            pl.col("estimated Creatinine (CKD-EPI eGFR=75)").alias("SCr Baseline"),
            pl.lit("CKD-EPI estimate (eGFR=75)").alias("SCr Baseline source"),
        )
    )

    # first non-null source per stay, in order of the hierarchy
    best = (
        pl.concat(
            [
                src.drop_nulls("SCr Baseline").select(
                    HOSPITAL_KEY, "SCr Baseline", "SCr Baseline source", pl.lit(i).alias("_step")
                )
                for i, src in enumerate(sources)
            ]
        )
        .sort("_step")
        .unique(HOSPITAL_KEY, keep="first", maintain_order=True)
    )
    return patient_information.select(HOSPITAL_KEY).join(
        best.drop("_step"), on=HOSPITAL_KEY, how="left"
    )


# endregion


################################################################################
################################################################################
# region creatinine helpers
def _prepare_creatinine_48h_baseline(
    creatinine_data: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Calculate the 48-hour rolling baseline for creatinine.

    KDIGO measures a rise of ≥0.3 mg/dL against the lowest value of the preceding
    48 hours (including the measurement itself).

    Arguments
    ---------
        creatinine_data : pl.LazyFrame
            LazyFrame sorted by hospital stay and time, one row per
            (hospital stay, time), with columns [Global Hospital Stay ID,
            Time Relative to Hospital Admission (seconds), Creatinine]

    Returns
    -------
        LazyFrame with columns:
        - Global Hospital Stay ID
        - Time Relative to Hospital Admission (seconds)
        - 48-hour SCr Baseline
    """
    return creatinine_data.rolling(
        index_column=HOSPITAL_TIME_KEY,
        period=f"{48 * SECONDS_IN_1H}i",
        group_by=HOSPITAL_KEY,
        closed="both",
    ).agg(pl.col("Creatinine").min().alias("48-hour SCr Baseline"))


def _hospital_clock(patient_information: pl.LazyFrame) -> pl.LazyFrame:
    """
    Map every ICU stay to its hospital stay and the hospital clock.

    Arguments
    ---------
        patient_information : pl.LazyFrame
            LazyFrame with Global ICU Stay ID and, if available,
            Global Hospital Stay ID and Pre-ICU Length of Stay (days)

    Returns
    -------
        LazyFrame with columns:
        - Global ICU Stay ID
        - Global Hospital Stay ID (the ICU stay itself if unknown)
        - hospital_offset: Seconds from hospital admission to ICU admission
    """
    names = patient_information.collect_schema().names()
    return patient_information.select(
        STAY_KEY,
        (
            pl.col(HOSPITAL_KEY).fill_null(pl.col(STAY_KEY))
            if HOSPITAL_KEY in names
            else pl.col(STAY_KEY)
        ).alias(HOSPITAL_KEY),
        (
            pl.col(PRE_ICU_COL).fill_null(0).mul(SECONDS_IN_1D).round()
            if PRE_ICU_COL in names
            else pl.lit(0)
        )
        .cast(pl.Int64)
        .alias("hospital_offset"),
    )


def _prepare_creatinine(
    timeseries_labs: pl.LazyFrame,
    patient_information: Optional[pl.LazyFrame] = None,
    version: str = "2012",
    rrt_start: Optional[pl.LazyFrame] = None,
    hindsight_baseline: bool = False,
) -> pl.LazyFrame:
    """
    Prepare creatinine timeseries with baseline calculations.

    Baselines are established per hospital stay: admission is hospital admission, so
    the creatinine of all ICU stays of a hospital stay is pooled on the hospital clock
    (Pre-ICU Length of Stay offsets the ICU clock). Pooled measurements belong to the
    ICU stay with the lowest ID; the 48-hour reference only looks backwards.

    For version 2012: Calculates 7-day and 48-hour rolling SCr Baseline values.
    For version 2026: Uses KDIGO 2026 established baseline hierarchy and 48-hour rolling baseline.

    Arguments
    ---------
        timeseries_labs : pl.LazyFrame
            LazyFrame with creatinine struct data
        patient_information : pl.LazyFrame, optional
            LazyFrame with patient demographics
        version : str
            KDIGO version ("2012" or "2026")
        rrt_start : pl.LazyFrame, optional
            LazyFrame with columns [Global ICU Stay ID, first_rrt_start]
            (version 2026 with `hindsight_baseline`)
        hindsight_baseline : bool
            Allow the lowest inpatient value as baseline (version 2026)

    Returns
    -------
        LazyFrame with columns:
        - Global ICU Stay ID
        - Time Relative to Admission (seconds)
        - Time Relative to Hospital Admission (seconds)
        - Creatinine (max per timestamp)
        - SCr Baseline for the hospital stay
        - SCr Baseline source
        - 48-hour SCr Baseline
    """
    stays = _hospital_clock(patient_information)

    # Extract creatinine data (filter for serum/plasma, non-null), pooled per hospital stay
    creatinine_data = (
        timeseries_labs.select(
            STAY_KEY,
            TIME_KEY,
            extract_struct_value(
                # HiRID reports its (laboratory) creatinine as "Blood"
                "Creatinine", ["Serum or Plasma", "Blood"], exact_match=True
            ).alias("Creatinine"),
        )
        .pipe(NULL_NON_FINITE)
        .pipe(FILTER_NOT_NULL, ["Creatinine"])
        .join(stays, on=STAY_KEY)
        # worst per timestamp: 1:1 joins, deterministic
        .group_by(
            HOSPITAL_KEY,
            pl.col(TIME_KEY)
            .add(pl.col("hospital_offset"))
            .round()
            .cast(pl.Int64)  # integer seconds: index of the rolling window
            .alias(HOSPITAL_TIME_KEY),
        )
        .agg(pl.col("Creatinine").max(), pl.col(STAY_KEY).min())
        .sort(HOSPITAL_KEY, HOSPITAL_TIME_KEY)
    )

    # Calculate 48-hour baseline (used for both versions)
    baseline_48h = _prepare_creatinine_48h_baseline(creatinine_data)

    # Hospital-level inputs of the baselines
    hospital_info = (
        patient_information.join(stays.select(STAY_KEY, HOSPITAL_KEY), on=STAY_KEY)
        .sort(STAY_KEY)
        .group_by(HOSPITAL_KEY, maintain_order=True)
        .agg(
            pl.col(col).first()
            for col in ["Admission Age (years)", "Gender", "Ethnicity"]
            if col in patient_information.collect_schema().names()
        )
    )

    # Calculate long-term baseline depending on version
    if version == "2026":
        # Use KDIGO 2026 established baseline hierarchy
        hospital_rrt_start = (
            rrt_start.join(stays, on=STAY_KEY)
            .group_by(HOSPITAL_KEY)
            .agg(pl.col("first_rrt_start").add(pl.col("hospital_offset")).min())
            if rrt_start is not None
            else None
        )
        baseline_long = _prepare_creatinine_baseline_2026(
            creatinine_data, hospital_info, hospital_rrt_start, hindsight_baseline
        )
    else:
        # Use 7-day baseline for KDIGO 2012
        baseline_long = _prepare_creatinine_baseline_2012(
            creatinine_data, hospital_info
        )

    # Join baselines with creatinine data
    return (
        creatinine_data.join(baseline_long, on=HOSPITAL_KEY, how="left")
        .join(baseline_48h, on=[HOSPITAL_KEY, HOSPITAL_TIME_KEY], how="left")
        .join(stays.select(STAY_KEY, "hospital_offset"), on=STAY_KEY)
        .select(
            STAY_KEY,
            pl.col(HOSPITAL_TIME_KEY).sub(pl.col("hospital_offset")).alias(TIME_KEY),
            HOSPITAL_TIME_KEY,
            "Creatinine",
            "SCr Baseline",
            "SCr Baseline source",
            "48-hour SCr Baseline",
        )
    )


def _creatinine_stage_points(
    creatinine: pl.Expr,
    baseline: pl.Expr,
    baseline_48h: pl.Expr,
) -> pl.Expr:
    """
    Calculate KDIGO AKI stage (0-3) based on creatinine criteria.

    KDIGO Creatinine Criteria:
    - Stage 1: ≥1.5× baseline OR ≥0.3 mg/dL increase (48h)
    - Stage 2: ≥2× baseline
    - Stage 3: ≥3× baseline OR increase to ≥4 mg/dL

    Arguments
    ---------
        creatinine : pl.Expr
            Current creatinine value expression
        baseline : pl.Expr
            SCr Baseline (from KDIGO 2026 hierarchy or 7-day rolling window);
            null if it is not yet a valid reference
        baseline_48h : pl.Expr
            Lowest creatinine of the preceding 48 hours (absolute rise criterion)

    Returns
    -------
        Integer (0-3) representing AKI stage, or None if neither baseline is available.
    """
    return (
        pl.when(baseline.is_null() & baseline_48h.is_null())
        .then(None)
        .when(
            pl.any_horizontal(
                creatinine.truediv(baseline).ge(3),
                creatinine.ge(4),
            )
        )
        .then(3)
        .when(creatinine.truediv(baseline).ge(2))
        .then(2)
        .when(
            pl.any_horizontal(
                creatinine.truediv(baseline).ge(1.5),
                creatinine.sub(baseline_48h).round(2).ge(0.3),
            )
        )
        .then(1)
        .otherwise(0)
    )


# endregion


################################################################################
################################################################################
# region resolution
def _compute_aki_resolution(
    aki_data: pl.LazyFrame,
    window_size: int,
    creatinine_col: str = "Creatinine",
    baseline_col: str = "SCr Baseline",
) -> pl.LazyFrame:
    """
    Compute AKI resolution per stay (KDIGO 2026 draft, Practice Point 1.4.1, Table 10).

    The AKI episode starts at the first timeframe with an overall stage ≥1. The
    resolution is judged by the last creatinine within 7 days after the onset
    (Sawhney et al., as cited by the draft; creatinine is the maximum per timeframe):
    - Complete Resolution: SCr <1.2× baseline
    - Partial Resolution: SCr ≥1.2× to <1.5× baseline
    - No Resolution: SCr ≥1.5× baseline
    The status is null if no creatinine was measured after the onset. Resolution
    uses follow-up measurements and is therefore only assigned from the onset on.

    Arguments
    ---------
        aki_data : pl.LazyFrame
            LazyFrame with columns [Global ICU Stay ID, timeframe,
            Overall KDIGO AKI Stage, Creatinine, SCr Baseline]
        window_size : int
            Timeframe width in seconds
        creatinine_col : str
            Name of the creatinine column
        baseline_col : str
            Name of the baseline column

    Returns
    -------
        LazyFrame with columns:
        - Global ICU Stay ID
        - timeframe
        - AKI Resolution Status (Complete Resolution | Partial Resolution | No Resolution)
        - Days from AKI Onset (from the onset on)
        - Days to Resolution (to the first timeframe with SCr <1.2× baseline within 7 days)
    """
    onset = (
        aki_data.filter(pl.col("Overall KDIGO AKI Stage").ge(1))
        .group_by(STAY_KEY)
        .agg(pl.col(TIMEFRAME_KEY).min().alias("_onset"))
    )
    days = (pl.col(TIMEFRAME_KEY) - pl.col("_onset")) * window_size / SECONDS_IN_1D
    follow_up = (
        aki_data.join(onset, on=STAY_KEY)
        .with_columns(
            (pl.col(creatinine_col) / pl.col(baseline_col)).alias("_ratio"),
            days.alias("_days"),
        )
        .filter(
            pl.col("_days").gt(0),
            pl.col("_days").le(SECONDS_IN_1W / SECONDS_IN_1D),
            pl.col("_ratio").is_not_null(),
        )
        .group_by(STAY_KEY)
        .agg(
            pl.col("_ratio").sort_by(TIMEFRAME_KEY).last().alias("_last_ratio"),
            pl.col("_days").filter(pl.col("_ratio").lt(1.2)).min().alias("Days to Resolution"),
        )
    )
    return (
        aki_data.select(STAY_KEY, TIMEFRAME_KEY)
        .join(onset, on=STAY_KEY)
        .join(follow_up, on=STAY_KEY, how="left")
        .filter(days.ge(0))
        .select(
            STAY_KEY,
            TIMEFRAME_KEY,
            pl.when(pl.col("_last_ratio").lt(1.2))
            .then(pl.lit("Complete Resolution"))
            .when(pl.col("_last_ratio").lt(1.5))
            .then(pl.lit("Partial Resolution"))
            .when(pl.col("_last_ratio").is_not_null())
            .then(pl.lit("No Resolution"))
            .alias("AKI Resolution Status"),
            days.alias("Days from AKI Onset"),
            "Days to Resolution",
        )
    )


# endregion


################################################################################
################################################################################
# region UO staging
def _uo_stages(
    uo_df: pl.LazyFrame,
    column: str,
    window_agg: str,
    limits: tuple[float, float, float],
    fixed_block: bool = False,
) -> pl.LazyFrame:
    """
    Calculate AKI stages (0-3) from hourly urine output (mL/kg) over 6/12/24h windows.

    A window only counts if all of its hours have a weight-normalised value. Stage 0
    means a complete 6h window without a criterion met. The stage is attached to the
    last hour of the window. Anuria (no urine for 12h) is stage 3.

    Arguments
    ---------
        uo_df : pl.LazyFrame
            LazyFrame with columns [Global ICU Stay ID, timeframe (hourly), uo_interval_ml_per_kg]
        column : str
            Name of the output stage column
        window_agg : str
            "max" (every hour below limit) or "sum" (total below limit)
        limits : tuple[float, float, float]
            Stage 1 (6h), 2 (12h) and 3 (24h) thresholds on the window aggregate
        fixed_block : bool
            Only evaluate windows aligned to T_0 (hours 0-5, 6-11, ...)

    Returns
    -------
        LazyFrame with columns:
        - Global ICU Stay ID
        - timeframe (hourly window, last hour of the UO window)
        - `column` (0-3)
    """
    uo = pl.col("uo_interval_ml_per_kg")
    hour = pl.col(TIMEFRAME_KEY)
    window_expr = {"max": uo.rolling_max_by, "sum": uo.rolling_sum_by}[window_agg]

    def window(n: int) -> tuple[pl.Expr, pl.Expr]:
        # rolling over a key that is unique per stay: windows never cross stays
        full = (
            uo.is_not_null().cast(pl.Int8).rolling_sum_by("_key", f"{n}i").ge(n)
        )
        if fixed_block:
            full &= (hour + 1) % n == 0
        # rounded: a rolling sum of 7.2 can come out as 7.199999999999999
        return full, window_expr("_key", f"{n}i").round(6)

    (full6, v6), (full12, v12), (full24, v24) = window(6), window(12), window(24)
    limit_1, limit_2, limit_3 = limits
    return (
        uo_df.sort(STAY_KEY, TIMEFRAME_KEY)
        .with_columns(
            (pl.col(STAY_KEY).rle_id().cast(pl.Int64) * 2**32 + hour + 2**31).alias("_key")
        )
        .select(
            STAY_KEY,
            TIMEFRAME_KEY,
            pl.max_horizontal(
                pl.when(full6).then(0),
                pl.when(full6 & v6.lt(limit_1)).then(1),
                pl.when(full12 & v12.lt(limit_2)).then(2),
                pl.when(full24 & v24.lt(limit_3)).then(3),
                pl.when(full12 & v12.eq(0)).then(3),
            ).alias(column),
        )
        .drop_nulls(column)
    )


def _uo_consecutive_stages(uo_df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Calculate AKI stages based on consecutive hourly urine output.

    KDIGO Urine Output Criteria (Consecutive; every hour of the window below threshold):
    - Stage 1: UO <0.5 mL/kg/h for ≥6 consecutive hours
    - Stage 2: UO <0.5 mL/kg/h for ≥12 consecutive hours
    - Stage 3: UO <0.3 mL/kg/h for ≥24 consecutive hours OR anuria for ≥12h
    - Stage 0: Data available but does not meet criteria for stages 1-3
    """
    return _uo_stages(
        uo_df, "UO Consecutive AKI Stage", "max", (0.5, 0.5, 0.3)
    )


def _uo_any_period_stages(uo_df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Calculate AKI stages based on total urine output over any rolling window.

    KDIGO Urine Output Criteria (Any Period):
    - Stage 1: UO <3 mL/kg during any 6-hour period (avg <0.5 mL/kg/h)
    - Stage 2: UO <6 mL/kg during any 12-hour period (avg <0.5 mL/kg/h)
    - Stage 3: UO <7.2 mL/kg during any 24-hour period (avg <0.3 mL/kg/h)
      OR anuria for ≥12h
    - Stage 0: Data available but does not meet criteria for stages 1-3
    """
    return _uo_stages(uo_df, "UO Any Period AKI Stage", "sum", (3, 6, 7.2))


def _uo_fixed_block_stages(uo_df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Calculate AKI stages based on fixed 6h, 12h and 24h blocks aligned to T_0.

    Same thresholds as any_period, evaluated only at the end of each block.
    """
    return _uo_stages(
        uo_df, "UO Fixed Block AKI Stage", "sum", (3, 6, 7.2), fixed_block=True
    )


# endregion


################################################################################
################################################################################
# region KDIGO AKI main function (unified with version support)
@with_windows
def AKI_KDIGO(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_labs: Optional[pl.LazyFrame] = None,
    timeseries_inout: Optional[pl.LazyFrame] = None,
    rrt: Optional[pl.LazyFrame] = None,
    *,
    version: str = "2012",
    weight_type: str = "ideal",
    consecutive: bool = False,
    any_period: bool = True,
    fixed_block: bool = False,
    include_resolution: bool = False,
    hindsight_baseline: bool = False,
    window: Window,
) -> pl.LazyFrame:
    """
    Compute KDIGO-based Acute Kidney Injury (AKI) stages in long format.

    Supports KDIGO 2012 and 2026 criteria with configurable options.

    KDIGO Creatinine Criteria:
    - Stage 1: ≥1.5× baseline (7 days) OR ≥0.3 mg/dL increase (48 hours)
    - Stage 2: ≥2× baseline (7 days)
    - Stage 3: ≥3× baseline (7 days) OR ≥4 mg/dL absolute

    KDIGO Urine Output Criteria (Stage-dependent; multiple methods supported):
    - Consecutive: Each hour must be below threshold for all hours in the window
    - Any Period: Total urine output over any rolling window must be below threshold
    - Fixed Block: Same as any_period but uses fixed 6h/12h/24h blocks

    KDIGO 2026 Enhancements (when version="2026"):
    - Optional ideal body weight (Devine formula) for UO thresholds
    - Optional AKI resolution (KDIGO 2026 Table 10, within 7 days of the onset)

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information. Must contain Global ICU Stay ID,
            Admission Weight (kg), age, and sex (for version="2026" with ideal weight).
            Loaded automatically if None.
        timeseries_labs : pl.LazyFrame, optional
            Timeseries lab values including Creatinine. Loaded automatically if None.
        timeseries_inout : pl.LazyFrame, optional
            Intake/output timeseries data. Loaded automatically if None.
        rrt : pl.LazyFrame, optional
            RRT timeseries data. Loaded automatically if None.
        version : str, optional
            "2012" (default) or "2026". Controls baseline strategy and feature availability.
        weight_type : str, optional
            "ideal" (Devine formula) or "actual". Only used when version="2026". Default: "ideal".
        consecutive : bool, optional
            Include AKI stages based on consecutive hourly UO criteria. Default: False.
        any_period : bool, optional
            Include AKI stages based on any-period rolling UO criteria (mean over the window, as in corr-vars). Default: True.
        fixed_block : bool, optional
            Include AKI stages based on fixed-block UO criteria. Default: False.
        include_resolution : bool, optional
            Include AKI resolution (complete, partial or none within 7 days after the
            onset, KDIGO 2026). Only used when version="2026". Default: False.
        hindsight_baseline : bool, optional
            Only used when version="2026". Allow the KDIGO fallback "lowest value during
            the admission (before RRT), including post-AKI values" as baseline when no
            outpatient or admission value exists. Allows retrospective AKI diagnosis, but
            uses measurements after the time being staged (hindsight). Default: False.

    Returns
    -------
        pl.LazyFrame
            One row per (stay, timeframe) with columns:
            - Global ICU Stay ID
            - T_0 (reference time in seconds)
            - timeframe (or custom name)
            - SCr Baseline source (where the creatinine baseline came from)
            - Creatinine AKI Stage (0-3)
            - UO Consecutive/Any Period/Fixed Block AKI Stage (if method enabled)
            - Overall KDIGO AKI Stage (max across all methods)
            - AKI Resolution Status / Days from AKI Onset / Days to Resolution
              (if version="2026" and include_resolution=True)

    Raises
    ------
        ValueError
            If version not in ["2012", "2026"], or weight_type not in ["ideal", "actual"].
    """
    # Validate version parameter
    if version not in ["2012", "2026"]:
        raise ValueError(
            f"Unknown KDIGO version: {version}. Must be '2012' or '2026'."
        )

    # Validate weight_type for version 2026
    if version == "2026" and weight_type not in ["ideal", "actual"]:
        raise ValueError(
            f"weight_type must be 'ideal' or 'actual', got {weight_type}"
        )

    # Load defaults if not provided
    if patient_information is None:
        patient_information = get_patient_information()
    if timeseries_labs is None:
        timeseries_labs = get_timeseries_labs()
    if timeseries_inout is None:
        timeseries_inout = get_timeseries_intakeoutput()
    if rrt is None:
        rrt = get_rrt()

    # Validate all required data is available
    required = {
        "patient_information": patient_information,
        "timeseries_labs": timeseries_labs,
        "timeseries_inout": timeseries_inout,
        "rrt": rrt,
    }
    _validate_required_data("AKI_KDIGO", required)

    # Ensure lazy frames
    patient_information = _to_lazy(patient_information)
    timeseries_labs = _to_lazy(timeseries_labs)
    timeseries_inout = _to_lazy(timeseries_inout)
    rrt = _to_lazy(rrt)

    # Renal Replacement Therapy (RRT) staging: Stage 3 if RRT received during timeframe
    rrt = (
        rrt.pipe(
            intervention_per_timeframe,
            patient_information,
            start_col="Renal Replacement Therapy Start Relative to Admission (seconds)",
            end_col="Renal Replacement Therapy End Relative to Admission (seconds)",
            window=window,
        )
        .cast({"timeframe": int})
        .select(STAY_KEY, "timeframe", pl.lit(3).alias("RRT AKI Stage"))
    )

    first_rrt_start = rrt.group_by(STAY_KEY).agg(
        pl.col(
            "Renal Replacement Therapy Start Relative to Admission (seconds)"
        )
        .min()
        .alias("first_rrt_start")
    )

    # region 1. Build T_0
    ctx = _window_context(patient_information, window)

    # region 2. Prepare creatinine data
    creatinine_data = _prepare_creatinine(
        timeseries_labs,
        patient_information=patient_information,
        version=version,
        rrt_start=first_rrt_start,
        hindsight_baseline=hindsight_baseline,
    )

    # Join with T_0 and assign timeframes
    creatinine_data = (
        creatinine_data.join(ctx.all_stays_t0, on=STAY_KEY, how="inner")
        .with_columns(
            _assign_timeframe(TIME_KEY, window.window_size)
            .cast(int)
            .alias("timeframe"),
            (pl.col(TIME_KEY) - pl.col("T_0")).alias(
                "Time Relative to T_0 (seconds)"
            ),
        )
    )

    if window.t_1 is not None:
        creatinine_data = creatinine_data.filter(pl.col(TIME_KEY) < window.t_1)

    # Aggregate to worst per timeframe
    creatinine_per_frame = (
        creatinine_data.with_columns(
            _creatinine_stage_points(
                pl.col("Creatinine"),
                # an admission baseline is only a reference after its +12 h window
                pl.when(
                    pl.col("SCr Baseline source").ne("Admission measurement")
                    | pl.col(HOSPITAL_TIME_KEY).gt(SECONDS_IN_12H)
                )
                .then(pl.col("SCr Baseline"))
                .otherwise(None),
                pl.col("48-hour SCr Baseline"),
            ).alias("Creatinine AKI Stage")
        )
        .group_by(STAY_KEY, "timeframe")
        .agg(
            pl.col("Creatinine").max(),
            pl.col("SCr Baseline").first(),
            pl.col("SCr Baseline source").first(),
            pl.col("Creatinine AKI Stage").max(),
        )
    )

    # region 3. Prepare body weight for UO calculations
    weight_col = "Admission Weight (kg)"
    if version == "2026" and weight_type == "ideal":
        # Compute ideal body weight via Devine formula and add to patient_information
        weight_col = "Ideal Body Weight (Effective)"
        patient_information = patient_information.join(
            IDEAL_BODY_WEIGHT_DEVINE(patient_information),
            on=STAY_KEY,
            how="left",
        ).with_columns(
            pl.coalesce(
                pl.col("Ideal Body Weight (Devine)"),
                pl.col("Admission Weight (kg)"),
            ).alias(weight_col)
        )

    # region 3.5. Prepare urine output data
    urine_data = (
        URINE_OUTPUT_RATE(
            patient_information=patient_information,
            timeseries_inout=timeseries_inout,
            t_0=window.t_0,
            t_0_per_stay=window.t_0_per_stay,
            t_minus1=window.t_minus1,
            t_1=window.t_1,
            window_size=SECONDS_IN_1H,
            timeframe_name="timeframe",
            weight_col=weight_col,
        )
        # Keep hourly data for UO staging calculations
        .join(ctx.all_stays_t0, on=STAY_KEY, how="inner")
        .cast({"timeframe": int})
    )
    if window.t_1 is not None:
        urine_data = urine_data.filter(
            pl.col("timeframe") * SECONDS_IN_1H + pl.col("T_0") < window.t_1
        )

    # region 4. Calculate UO-based AKI stages (on hourly data)
    uo_stages_dict = {}

    if consecutive:
        uo_stages_dict["consecutive"] = _uo_consecutive_stages(urine_data)
    if any_period:
        uo_stages_dict["any_period"] = _uo_any_period_stages(urine_data)
    if fixed_block:
        uo_stages_dict["fixed_block"] = _uo_fixed_block_stages(urine_data)
    # endregion

    # region 4.5. Aggregate UO stages to daily timeframes
    for method_name, uo_method_df in uo_stages_dict.items():
        # Convert hourly timeframes to daily timeframes
        stage_col = f"UO {method_name.replace('_', ' ').title()} AKI Stage"

        uo_stages_dict[method_name] = (
            uo_method_df.with_columns(
                (pl.col("timeframe") * SECONDS_IN_1H)
                .floordiv(window.window_size)
                .cast(int)
                .alias("windowly_timeframe")
            )
            .group_by(STAY_KEY, "windowly_timeframe")
            .agg(pl.col(stage_col).max())
            .rename({"windowly_timeframe": "timeframe"})
        )
    # endregion

    # region 5. Join all data
    urine_per_window = urine_data.group_by(
        STAY_KEY,
        pl.col("timeframe").mul(SECONDS_IN_1H).floordiv(window.window_size),
    ).agg(
        pl.col("uo_interval_ml").sum().alias("UO ml/window"),
        pl.col("uo_interval_ml_per_kg").sum().alias("UO ml/kg/window"),
    )
    creatinine_stages = creatinine_per_frame.select(
        STAY_KEY,
        "timeframe",
        "Creatinine",
        "SCr Baseline",
        "SCr Baseline source",
        "Creatinine AKI Stage",
    )
    result = (
        _add_sparse_timeframes(
            _build_base_timeframes(ctx.all_stays_t0, patient_information, window.window_size),
            rrt,
            urine_per_window,
            creatinine_stages,
        )
        .sort(STAY_KEY, "timeframe")
        .join(rrt, on=[STAY_KEY, "timeframe"], how="left")
        .join(urine_per_window, on=[STAY_KEY, "timeframe"], how="left")
        .join(creatinine_stages, on=[STAY_KEY, "timeframe"], how="left")
    )

    for method_name, uo_method_df in uo_stages_dict.items():
        result = result.join(
            uo_method_df, on=[STAY_KEY, "timeframe"], how="left"
        )

    # Calculate overall stage as max of all available stages
    stage_cols = ["RRT AKI Stage", "Creatinine AKI Stage"]
    for method_name in uo_stages_dict.keys():
        method_col = f"UO {method_name.replace('_', ' ').title()} AKI Stage"
        stage_cols.append(method_col)

    result = result.with_columns(
        pl.max_horizontal(*stage_cols).alias("Overall KDIGO AKI Stage")
    ).sort(STAY_KEY, "timeframe")

    # Add the AKI resolution if requested (version 2026 only)
    if version == "2026" and include_resolution:
        resolution_data = _compute_aki_resolution(result, window.window_size)
        result = result.join(
            resolution_data, on=[STAY_KEY, "timeframe"], how="left"
        )

    result = result.filter(*ctx.bounds)

    # Rename timeframe column if needed
    if ctx.name != "timeframe":
        result = result.with_columns(
            pl.col("timeframe").alias(ctx.name)
        ).drop("timeframe")

    return result


# endregion

__all__ = ["AKI_KDIGO"]
