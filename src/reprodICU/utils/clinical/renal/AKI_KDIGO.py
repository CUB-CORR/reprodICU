"""
AKI_KDIGO: compute KDIGO-based Acute Kidney Injury stages in long format directly from raw inputs.

Output columns per row:
- Global ICU Stay ID
- T_0 (seconds from admission used as reference)
- timeframe (0-indexed integer window)
- Creatinine AKI Stage (0-3 based on KDIGO creatinine criteria)
- UO Consecutive AKI Stage (0-3 based on consecutive urine output method, if enabled)
- UO Any Period AKI Stage (0-3 based on any period urine output method, if enabled)
- UO Fixed Block AKI Stage (0-3 based on fixed block urine output method, if enabled)
- Overall KDIGO AKI Stage (max of all available stages)

Time is in seconds. Windows determined by floor((time - T_0)/window_size).
Worst-within-window aggregation is applied per stage method.

Similar in implementation to pyAKI, but adapted for polars and reprodICU framework.
- cf. with demo data from https://github.com/aidh-ms/pyAKI/blob/main/tests/data/validation_data.csv
- MIMIC-IV IDs: 30849778, 32119961, 32314488, 32391858, 33281088, 34592300, 34617352, 35009126, 35258379, 35514836, 37267577, 38383343, 38540883, 39268883, 39635619

SOURCES
-------
- KDIGO 2024 Clinical Practice Guideline for the Evaluation and Management of Chronic Kidney Disease
  (see doi:10.1016/j.kint.2024.06.014)
- Kidney Disease: Improving Global Outcomes (KDIGO) Acute Kidney Injury Work Group.
  KDIGO Clinical Practice Guideline for Acute Kidney Injury. Kidney Int Suppl. 2012;2(1):1–138.
"""

from typing import Optional

import polars as pl

from ...common import (
    _assign_timeframe,
    _build_base_timeframes,
    _build_t0,
    _get_timeframe_name,
    _optional_time_bounds_filter,
    _to_lazy,
    get_patient_information,
    get_rrt,
    get_timeseries_intakeoutput,
    get_timeseries_labs,
    intervention_per_timeframe,
)
from ..IDEAL_BODY_WEIGHT import IDEAL_BODY_WEIGHT_DEVINE
from .CREATININE import reverse_CKD_EPI, reverse_MDRD
from .URINE_OUTPUT import URINE_OUTPUT

SECONDS_IN_1H = 60 * 60
SECONDS_IN_12H = 12 * SECONDS_IN_1H
SECONDS_IN_48H = 48 * SECONDS_IN_1H
SECONDS_IN_1D = 24 * SECONDS_IN_1H
SECONDS_IN_7D = 7 * SECONDS_IN_1D
SECONDS_IN_365D = 365 * SECONDS_IN_1D

STAY_KEY = "Global ICU Stay ID"
TIME_KEY = "Time Relative to Admission (seconds)"
TIMEFRAME_KEY = "timeframe"


################################################################################
################################################################################
# region creatinine baselines
def _prepare_creatinine_baseline_2012(
    creatinine_data: pl.LazyFrame,
    patient_information: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Prepare creatinine with baseline hierarchy (KDIGO 2012).

    KDIGO 2012 baseline:
    - baseline, which is known or presumed to have occurred within the prior 7 days

    Returns:
        LazyFrame with columns:
        - Global ICU Stay ID
        - SCr Baseline (established baseline for this stay)
        - SCr Baseline source (where baseline came from)
    """
    # Measurement within 7 days prior to admission
    prior_data = (
        creatinine_data.filter(
            pl.col(TIME_KEY).is_between(-SECONDS_IN_7D, SECONDS_IN_12H)
        )
        .group_by(STAY_KEY)
        .agg(
            pl.col("Creatinine")
            .sort_by(pl.col(TIME_KEY).abs())
            .last()
            .alias("SCr Baseline prior measurement"),
            pl.lit("Prior measurement").alias(
                "Baseline source prior measurement"
            ),
        )
    )

    # MDRD estimate (eGFR = 75 mL/min/1.73m²)
    # Use reverse MDRD formula to estimate creatinine at eGFR=75
    mdrd_baseline = (
        patient_information.select(STAY_KEY, "Admission Age (years)", "Gender", "Ethnicity")
        .pipe(reverse_MDRD, target_egfr=75)
        .select(
            STAY_KEY,
            pl.col("estimated Creatinine (MDRD eGFR=75)").alias("SCr Baseline MDRD"),
            pl.lit("MDRD estimate (eGFR=75)").alias("Baseline source MDRD"),
        )
    ) # fmt: skip

    # Combine sources with priority: prior measurement > mdrd
    return (
        patient_information.select(STAY_KEY)
        .join(prior_data, on=STAY_KEY, how="left")
        .join(mdrd_baseline, on=STAY_KEY, how="left")
        .with_columns(
            pl.coalesce(
                pl.col("SCr Baseline prior measurement"),
                pl.col("SCr Baseline MDRD"),
            ).alias("SCr Baseline"),
            pl.coalesce(
                pl.col("Baseline source prior measurement"),
                pl.col("Baseline source MDRD"),
            ).alias("SCr Baseline source"),
        )
        .select(STAY_KEY, "SCr Baseline", "SCr Baseline source")
    )


def _prepare_creatinine_baseline_2026(
    creatinine_data: pl.LazyFrame,
    patient_information: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Prepare creatinine with baseline hierarchy (KDIGO 2026).

    KDIGO 2026 baseline hierarchy:
    1. Representative outpatient measurement prior to admission < 365 days
    2. Admission measurement (unless AKI on admission suspected and absence of previous values)
    3. Lowest measurement during index admission (including lowest post AKI measurement) when not receiving RRT
    4. Estimate SCr or Cys C using CKD-EPI equation assuming eGFR is 75 ml/min per 1.73 m²

    Returns:
        LazyFrame with columns:
        - Global ICU Stay ID
        - SCr Baseline (established baseline for this stay)
        - SCr Baseline source (where baseline came from)
    """
    # Step 1: Try to get outpatient measurement (0-365 days pre-admission, from TIME_KEY < 0)
    outpatient_data = (
        creatinine_data.filter(pl.col(TIME_KEY).is_between(-SECONDS_IN_365D, 0))
        .group_by(STAY_KEY)
        .agg(
            pl.col("Creatinine").max().alias("SCr Baseline outpatient"),
            pl.lit("Outpatient").alias("Baseline source outpatient"),
        )
    )

    # Step 2: Get admission measurement (TIME_KEY ~= 0, first/earliest measurement at admission)
    admission_data = (
        creatinine_data.filter(
            pl.col(TIME_KEY).is_between(-SECONDS_IN_12H, SECONDS_IN_12H)
        )
        .group_by(STAY_KEY)
        .agg(
            pl.col("Creatinine")
            .sort_by(pl.col(TIME_KEY).abs())
            .last()
            .alias("SCr Baseline at admission"),
            pl.lit("Admission measurement").alias("Baseline source admission"),
        )
    )

    # Step 4: CKD-EPI estimate (eGFR = 75 mL/min/1.73m²)
    # Use reverse CKD-EPI formula to estimate creatinine at eGFR=75
    ckd_epi_baseline = (
        patient_information.select(STAY_KEY, "Admission Age (years)", "Gender")
        .pipe(reverse_CKD_EPI, target_egfr=75)
        .select(
            STAY_KEY,
            pl.col("estimated Creatinine (CKD-EPI eGFR=75)").alias("SCr Baseline CKD-EPI"),
            pl.lit("CKD-EPI estimate (eGFR=75)").alias("Baseline source CKD-EPI"),
        ) # fmt: skip
    )

    # Combine all sources with priority: outpatient > admission > ckd_epi
    return (
        patient_information.select(STAY_KEY)
        .join(outpatient_data, on=STAY_KEY, how="left")
        .join(admission_data, on=STAY_KEY, how="left")
        .join(ckd_epi_baseline, on=STAY_KEY, how="left")
        .with_columns(
            pl.coalesce(
                pl.col("SCr Baseline outpatient"),
                pl.col("SCr Baseline at admission"),
                pl.col("SCr Baseline CKD-EPI"),
            ).alias("SCr Baseline"),
            pl.coalesce(
                pl.col("Baseline source outpatient"),
                pl.col("Baseline source admission"),
                pl.col("Baseline source CKD-EPI"),
            ).alias("SCr Baseline source"),
        )
        .select(STAY_KEY, "SCr Baseline", "SCr Baseline source")
    )


# endregion


################################################################################
################################################################################
# region creatinine helpers
def _prepare_creatinine_48h_baseline(
    creatinine_data: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Calculate 48-hour rolling baseline for creatinine using forward-fill join_asof.

    Finds the most recent creatinine value within the 48-hour lookback window.

    Args:
        creatinine_data: LazyFrame with columns [Global ICU Stay ID, Time Relative to Admission (seconds), Creatinine]

    Returns:
        LazyFrame with columns:
        - Global ICU Stay ID
        - Time Relative to Admission (seconds)
        - 48-hour SCr Baseline
    """
    # Prepare left dataframe with 48-hour lookback time
    left_data = creatinine_data.with_columns(
        (pl.col(TIME_KEY) - SECONDS_IN_48H).alias("time_minus_48h"),
    )

    # Prepare right dataframe for join_asof
    right_data_48h = creatinine_data.select(
        STAY_KEY,
        TIME_KEY,
        pl.col("Creatinine").alias("48-hour SCr Baseline"),
    )

    # Join with 48-hour baseline
    return left_data.join_asof(
        right_data_48h,
        by=STAY_KEY,
        left_on="time_minus_48h",
        right_on=TIME_KEY,
        strategy="forward",
        suffix="_48h",
        coalesce=True,
    ).select(STAY_KEY, TIME_KEY, "48-hour SCr Baseline")


def _prepare_creatinine(
    timeseries_labs: pl.LazyFrame,
    patient_information: Optional[pl.LazyFrame] = None,
    version: str = "2012",
) -> pl.LazyFrame:
    """
    Prepare creatinine timeseries with baseline calculations.

    For version 2012: Calculates 7-day and 48-hour rolling SCr Baseline values.
    For version 2026: Uses KDIGO 2026 established baseline hierarchy and 48-hour rolling baseline.

    Args:
        timeseries_labs: LazyFrame with creatinine struct data
        patient_information: LazyFrame with patient demographics (required for version 2026)
        version: KDIGO version ("2012" or "2026")

    Returns:
        LazyFrame with columns:
        - Global ICU Stay ID
        - Time Relative to Admission (seconds)
        - Creatinine (max per hour)
        - SCr Baseline for this stay
        - 48-hour SCr Baseline
    """

    # Extract creatinine data (filter for serum/plasma, non-null)
    creatinine_data = (
        timeseries_labs.filter(
            pl.col("Creatinine").struct.field("value").is_not_null(),
            pl.col("Creatinine").struct.field("system") == "Serum or Plasma",
        )
        .select(
            STAY_KEY,
            TIME_KEY,
            pl.col("Creatinine").struct.field("value").alias("Creatinine"),
        )
        .sort(STAY_KEY, TIME_KEY)
    )

    # Calculate 48-hour baseline (used for both versions)
    baseline_48h = _prepare_creatinine_48h_baseline(creatinine_data)

    # Calculate long-term baseline depending on version
    if version == "2026":
        # Use KDIGO 2026 established baseline hierarchy
        baseline_long = _prepare_creatinine_baseline_2026(
            creatinine_data, patient_information
        )
    else:
        # Use 7-day baseline for KDIGO 2012
        baseline_long = _prepare_creatinine_baseline_2012(
            creatinine_data, patient_information
        )

    # Join baselines with creatinine data
    return (
        creatinine_data.join(baseline_long, on=STAY_KEY, how="left")
        .join(baseline_48h, on=[STAY_KEY, TIME_KEY], how="left")
        .select(
            STAY_KEY,
            TIME_KEY,
            "Creatinine",
            "SCr Baseline",
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
    - Stage 3: ≥3× baseline OR ≥4 mg/dL absolute

    Args:
        creatinine: Current creatinine value expression
        baseline: SCr Baseline (from KDIGO 2026 hierarchy or 7-day rolling window)
        baseline_48h: 48-hour baseline for alternative Stage 1 criterion

    Returns:
        Integer (0-3) representing AKI stage, or None if data insufficient.
    """
    return (
        pl.when(
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
                creatinine.sub(baseline_48h).ge(0.3),
            )
        )
        .then(1)
        .otherwise(0)
    )


# endregion


################################################################################
################################################################################
# region UO consecutive
def _uo_consecutive_stages(uo_df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Calculate AKI stages based on consecutive hourly urine output.

    Method: Each hour must be below threshold for all hours in window.

    KDIGO Urine Output Criteria (Consecutive):
    - Stage 1: UO <0.5 mL/kg/h for ≥6 consecutive hours
    - Stage 2: UO <0.5 mL/kg/h for ≥12 consecutive hours
    - Stage 3: UO <0.3 mL/kg/h for ≥24 consecutive hours OR anuria for ≥12h
    - Stage 0: Data available but does not meet criteria for stages 1-3

    Returns:
        LazyFrame with columns:
        - Global ICU Stay ID
        - timeframe (hourly window)
        - UO Consecutive AKI Stage (0-3)
    """

    # Stage 1: 6+ consecutive hours <0.5 mL/kg/h
    stage1 = (
        uo_df.rolling(
            index_column=TIMEFRAME_KEY,
            period="6i",
            group_by=STAY_KEY,
        )
        .agg(
            pl.when(pl.len() >= 6)
            .then(pl.col("uo_interval_ml_per_kg").lt(0.5).all())
            .otherwise(None)
            .alias("_stage1_flag")
        )
        .with_columns(
            pl.when(pl.col("_stage1_flag")).then(1).otherwise(None).alias("_s1")
        )
        .select(STAY_KEY, TIMEFRAME_KEY, "_s1")
    )

    # Stage 2: 12+ consecutive hours <0.5 mL/kg/h OR anuria 12h
    stage2 = (
        uo_df.rolling(
            index_column=TIMEFRAME_KEY,
            period="12i",
            group_by=STAY_KEY,
        )
        .agg(
            pl.when(pl.len() >= 12)
            .then(
                pl.any_horizontal(
                    pl.col("uo_interval_ml_per_kg").lt(0.5).all(),
                    pl.col("uo_interval_ml_per_kg").eq(0).all(),
                )
            )
            .otherwise(None)
            .alias("_stage2_flag")
        )
        .with_columns(
            pl.when(pl.col("_stage2_flag")).then(2).otherwise(None).alias("_s2")
        )
        .select(STAY_KEY, TIMEFRAME_KEY, "_s2")
    )

    # Stage 3: 24+ consecutive hours <0.3 mL/kg/h OR anuria 12h
    stage3 = (
        uo_df.rolling(
            index_column=TIMEFRAME_KEY,
            period="24i",
            group_by=STAY_KEY,
        )
        .agg(
            pl.when(pl.len() >= 24)
            .then(
                pl.any_horizontal(
                    pl.col("uo_interval_ml_per_kg").lt(0.3).all(),
                    pl.col("uo_interval_ml_per_kg").eq(0).all(),
                )
            )
            .otherwise(None)
            .alias("_stage3_flag")
        )
        .with_columns(
            pl.when(pl.col("_stage3_flag")).then(3).otherwise(None).alias("_s3")
        )
        .select(STAY_KEY, TIMEFRAME_KEY, "_s3")
    )

    # Stage 0: Data available but does not meet criteria for stages 1-3
    stage0 = (
        uo_df.rolling(
            index_column=TIMEFRAME_KEY,
            period="6i",
            group_by=STAY_KEY,
        )
        .agg(pl.when(pl.len() >= 6).then(0).otherwise(None).alias("_s0"))
        .select(STAY_KEY, TIMEFRAME_KEY, "_s0")
    )

    # Combine stages: take max of available stages
    result = (
        stage0.join(
            stage1, on=[STAY_KEY, TIMEFRAME_KEY], how="left", coalesce=True
        )
        .join(stage2, on=[STAY_KEY, TIMEFRAME_KEY], how="left", coalesce=True)
        .join(stage3, on=[STAY_KEY, TIMEFRAME_KEY], how="left", coalesce=True)
        .with_columns(
            pl.max_horizontal(
                pl.col("_s0"), pl.col("_s1"), pl.col("_s2"), pl.col("_s3")
            ).alias("UO Consecutive AKI Stage")
        )
        .select(STAY_KEY, TIMEFRAME_KEY, "UO Consecutive AKI Stage")
    )

    return result


# endregion


# region UO any period
def _uo_any_period_stages(uo_df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Calculate AKI stages based on any time period urine output.

    Method: Total urine output over any consecutive window (averaged).

    KDIGO Urine Output Criteria (Any Period):
    - Stage 1: UO <3 mL/kg during any 6-hour period (avg <0.5 mL/kg/h)
    - Stage 2: UO <6 mL/kg during any 12-hour period (avg <0.5 mL/kg/h)
    - Stage 3: UO <7.2 mL/kg during any 24-hour period (avg <0.3 mL/kg/h)
    - Stage 0: Data available but does not meet criteria for stages 1-3

    Returns:
        LazyFrame with columns:
        - Global ICU Stay ID
        - timeframe (hourly window)
        - UO Any Period AKI Stage (0-3)
    """

    # Stage 1: 6-hour rolling sum <3 mL/kg
    stage1 = (
        uo_df.rolling(
            index_column=TIMEFRAME_KEY,
            period="6i",
            group_by=STAY_KEY,
        )
        .agg(
            pl.when(pl.len() >= 6)
            .then(pl.col("uo_interval_ml_per_kg").sum().lt(3))
            .otherwise(None)
            .alias("_stage1_flag")
        )
        .with_columns(
            pl.when(pl.col("_stage1_flag")).then(1).otherwise(None).alias("_s1")
        )
        .select(STAY_KEY, TIMEFRAME_KEY, "_s1")
    )

    # Stage 2: 12-hour rolling sum <6 mL/kg
    stage2 = (
        uo_df.rolling(
            index_column=TIMEFRAME_KEY,
            period="12i",
            group_by=STAY_KEY,
        )
        .agg(
            pl.when(pl.len() >= 12)
            .then(pl.col("uo_interval_ml_per_kg").sum().lt(6))
            .otherwise(None)
            .alias("_stage2_flag")
        )
        .with_columns(
            pl.when(pl.col("_stage2_flag")).then(2).otherwise(None).alias("_s2")
        )
        .select(STAY_KEY, TIMEFRAME_KEY, "_s2")
    )

    # Stage 3: 24-hour rolling sum <7.2 mL/kg
    stage3 = (
        uo_df.rolling(
            index_column=TIMEFRAME_KEY,
            period="24i",
            group_by=STAY_KEY,
        )
        .agg(
            pl.when(pl.len() >= 24)
            .then(pl.col("uo_interval_ml_per_kg").sum().lt(7.2))
            .otherwise(None)
            .alias("_stage3_flag")
        )
        .with_columns(
            pl.when(pl.col("_stage3_flag")).then(3).otherwise(None).alias("_s3")
        )
        .select(STAY_KEY, TIMEFRAME_KEY, "_s3")
    )

    # Stage 0: Data available but does not meet criteria for stages 1-3
    stage0 = (
        uo_df.rolling(
            index_column=TIMEFRAME_KEY,
            period="6i",
            group_by=STAY_KEY,
        )
        .agg(pl.when(pl.len() >= 6).then(0).otherwise(None).alias("_s0"))
        .select(STAY_KEY, TIMEFRAME_KEY, "_s0")
    )

    # Combine stages
    result = (
        stage0.join(
            stage1, on=[STAY_KEY, TIMEFRAME_KEY], how="left", coalesce=True
        )
        .join(stage2, on=[STAY_KEY, TIMEFRAME_KEY], how="left", coalesce=True)
        .join(stage3, on=[STAY_KEY, TIMEFRAME_KEY], how="left", coalesce=True)
        .with_columns(
            pl.max_horizontal(
                pl.col("_s0"), pl.col("_s1"), pl.col("_s2"), pl.col("_s3")
            ).alias("UO Any Period AKI Stage")
        )
        .select(STAY_KEY, TIMEFRAME_KEY, "UO Any Period AKI Stage")
    )

    return result


# endregion


# region UO fixed block
def _uo_fixed_block_stages(uo_df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Calculate AKI stages based on fixed time blocks.

    Method: Fixed 6-hour, 12-hour, and 24-hour blocks (midnight-aligned).

    Same thresholds as any_period method but using fixed block boundaries.
    Stage 0 is assigned when data is available but does not meet criteria for stages 1-3.

    Returns:
        LazyFrame with columns:
        - Global ICU Stay ID
        - timeframe (hourly window, representing end of block)
        - UO Fixed Block AKI Stage (0-3)
    """

    base_data = uo_df.with_columns(
        (pl.col(TIMEFRAME_KEY) // 6).alias("_6h_block"),
        (pl.col(TIMEFRAME_KEY) // 12).alias("_12h_block"),
        (pl.col(TIMEFRAME_KEY) // 24).alias("_24h_block"),
    )

    # 6-hour block: sum <3 mL/kg
    stage1 = (
        base_data.group_by(STAY_KEY, "_6h_block")
        .agg(
            pl.when(pl.len() >= 6)
            .then(pl.col("uo_interval_ml_per_kg").sum().lt(3))
            .otherwise(None)
            .alias("_stage1_flag"),
            pl.col(TIMEFRAME_KEY).max().alias(TIMEFRAME_KEY),
        )
        .with_columns(
            pl.when(pl.col("_stage1_flag")).then(1).otherwise(None).alias("_s1")
        )
        .select(STAY_KEY, TIMEFRAME_KEY, "_s1")
    )

    # 12-hour block: sum <6 mL/kg
    stage2 = (
        base_data.group_by(STAY_KEY, "_12h_block")
        .agg(
            pl.when(pl.len() >= 12)
            .then(pl.col("uo_interval_ml_per_kg").sum().lt(6))
            .otherwise(None)
            .alias("_stage2_flag"),
            pl.col(TIMEFRAME_KEY).max().alias(TIMEFRAME_KEY),
        )
        .with_columns(
            pl.when(pl.col("_stage2_flag")).then(2).otherwise(None).alias("_s2")
        )
        .select(STAY_KEY, TIMEFRAME_KEY, "_s2")
    )

    # 24-hour block: sum <7.2 mL/kg
    stage3 = (
        base_data.group_by(STAY_KEY, "_24h_block")
        .agg(
            pl.when(pl.len() >= 24)
            .then(pl.col("uo_interval_ml_per_kg").sum().lt(7.2))
            .otherwise(None)
            .alias("_stage3_flag"),
            pl.col(TIMEFRAME_KEY).max().alias(TIMEFRAME_KEY),
        )
        .with_columns(
            pl.when(pl.col("_stage3_flag")).then(3).otherwise(None).alias("_s3")
        )
        .select(STAY_KEY, TIMEFRAME_KEY, "_s3")
    )

    # Stage 0: Data available but does not meet criteria for stages 1-3
    stage0 = (
        base_data.group_by(STAY_KEY, "_6h_block")
        .agg(
            pl.when(pl.len() >= 6).then(0).otherwise(None).alias("_s0"),
            pl.col(TIMEFRAME_KEY).max().alias(TIMEFRAME_KEY),
        )
        .select(STAY_KEY, TIMEFRAME_KEY, "_s0")
    )

    result = (
        stage0
        # Combine stages
        .join(stage1, on=[STAY_KEY, TIMEFRAME_KEY], how="left", coalesce=True)
        .join(stage2, on=[STAY_KEY, TIMEFRAME_KEY], how="left", coalesce=True)
        .join(stage3, on=[STAY_KEY, TIMEFRAME_KEY], how="left", coalesce=True)
        .with_columns(
            pl.max_horizontal(
                pl.col("_s0"), pl.col("_s1"), pl.col("_s2"), pl.col("_s3")
            ).alias("UO Fixed Block AKI Stage")
        )
        .select(STAY_KEY, TIMEFRAME_KEY, "UO Fixed Block AKI Stage")
    )

    return result


# endregion


################################################################################
################################################################################
# region KDIGO AKI main function (unified with version support)
def AKI_KDIGO(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_labs: Optional[pl.LazyFrame] = None,
    timeseries_inout: Optional[pl.LazyFrame] = None,
    renal_replacement_therapy: Optional[pl.LazyFrame] = None,
    *,
    version: str = "2012",
    t_0: Optional[int] = 0,
    t_0_per_stay: Optional[pl.LazyFrame] = None,
    t_1: Optional[int] = None,
    window_size: int = SECONDS_IN_1D,
    timeframe_unit: str = "Days",
    timeframe_name: Optional[str] = None,
    weight_type: str = "ideal",
    consecutive: bool = True,
    any_period: bool = False,
    fixed_block: bool = False,
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
    - Optional AKI resolution and AKD tracking

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
        renal_replacement_therapy : pl.LazyFrame, optional
            RRT timeseries data. Loaded automatically if None.
        version : str, optional
            "2012" (default) or "2026". Controls baseline strategy and feature availability.
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
        timeframe_unit : str, optional
            Name of the time unit for documentation. Default: "Days".
        timeframe_name : str, optional
            Name for output timeframe column. Auto-generated if None.
        weight_type : str, optional
            "ideal" (Devine formula) or "actual". Only used when version="2026". Default: "ideal".
        consecutive : bool, optional
            Include AKI stages based on consecutive hourly UO criteria. Default: True.
        any_period : bool, optional
            Include AKI stages based on any-period rolling UO criteria. Default: False.
        fixed_block : bool, optional
            Include AKI stages based on fixed-block UO criteria. Default: False.
        include_criteria : bool, optional
            Include intermediate criteria columns in output. Default: False.
        include_intermediate_stages : bool, optional
            Include intermediate boolean stage columns. Default: False.

    Returns
    -------
        pl.LazyFrame
            One row per (stay, timeframe) with columns:
            - Global ICU Stay ID
            - T_0 (reference time in seconds)
            - timeframe (or custom name)
            - Creatinine AKI Stage (0-3)
            - UO Consecutive/Any Period/Fixed Block AKI Stage (if method enabled)
            - Overall KDIGO AKI Stage (max across all methods)

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
    if renal_replacement_therapy is None:
        renal_replacement_therapy = get_rrt()

    # Validate all required data is available
    required = {
        "patient_information": patient_information,
        "timeseries_labs": timeseries_labs,
        "timeseries_inout": timeseries_inout,
        "renal_replacement_therapy": renal_replacement_therapy,
    }
    missing = [name for name, data in required.items() if data is None]
    if missing:
        raise ValueError(
            f"Cannot compute AKI_KDIGO: Missing required datasets: {', '.join(missing)}. "
            f"Ensure they are configured in ~/.reprodICU/PATHS.yaml or provide them explicitly."
        )

    # Ensure lazy frames
    patient_information = _to_lazy(patient_information)
    timeseries_labs = _to_lazy(timeseries_labs)
    timeseries_inout = _to_lazy(timeseries_inout)
    renal_replacement_therapy = _to_lazy(renal_replacement_therapy)

    # Renal Replacement Therapy (RRT) staging: Stage 3 if RRT received during timeframe
    rrt = (
        renal_replacement_therapy.pipe(
            intervention_per_timeframe,
            patient_information,
            start_col="Renal Replacement Therapy Start Relative to Admission (seconds)",
            end_col="Renal Replacement Therapy End Relative to Admission (seconds)",
            t_0=t_0,
            t_0_per_stay=t_0_per_stay,
            window_size=window_size,
        )
        .cast({"timeframe": int})
        .select(STAY_KEY, "timeframe", pl.lit(3).alias("RRT AKI Stage"))
    )

    # region 1. Build T_0
    ALL_STAYS = patient_information.select(STAY_KEY)
    ALL_STAYS_T0 = _build_t0(ALL_STAYS, t_0_per_stay, t_0)
    timeframe_name = _get_timeframe_name(
        timeframe_name, window_size, t_0, t_0_per_stay
    )

    # region 2. Prepare creatinine data
    creatinine_data = _prepare_creatinine(
        timeseries_labs,
        patient_information=patient_information,
        version=version,
    )

    # Join with T_0 and assign timeframes
    creatinine_data = (
        creatinine_data.join(ALL_STAYS_T0, on=STAY_KEY, how="inner")
        .with_columns(
            _assign_timeframe(TIME_KEY, window_size)
            .cast(int)
            .alias("timeframe"),
            (pl.col(TIME_KEY) - pl.col("T_0")).alias(
                "Time Relative to T_0 (seconds)"
            ),
        )
        .filter(pl.col("Time Relative to T_0 (seconds)") >= 0)
    )

    # Apply optional time bounds
    for cond in _optional_time_bounds_filter(
        "timeframe",
        window_size,
        t_0 // window_size if t_0 is not None else None,
        t_1 // window_size if t_1 is not None else None,
    ):
        creatinine_data = creatinine_data.filter(cond)

    # Aggregate to worst per timeframe
    creatinine_per_frame = (
        creatinine_data.group_by(STAY_KEY, "timeframe")
        .agg(
            pl.col("Creatinine").max(),
            pl.col("SCr Baseline").mean(),
            pl.col("48-hour SCr Baseline").mean(),
        )
        .with_columns(
            _creatinine_stage_points(
                pl.col("Creatinine"),
                pl.col("SCr Baseline"),
                pl.col("48-hour SCr Baseline"),
            ).alias("Creatinine AKI Stage")
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
        URINE_OUTPUT(
            patient_information=patient_information,
            timeseries_inout=timeseries_inout,
            t_0=t_0,
            t_0_per_stay=t_0_per_stay,
            t_1=t_1,
            window_size=SECONDS_IN_1H,
            weight_col=weight_col,
        )
        # Keep hourly data for UO staging calculations
        .join(ALL_STAYS_T0, on=STAY_KEY, how="inner")
        .cast({"timeframe": int})
        .sort("timeframe")  # Sort for rolling operations
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
                .floordiv(window_size)
                .cast(int)
                .alias("windowly_timeframe")
            )
            .group_by(STAY_KEY, "windowly_timeframe")
            .agg(pl.col(stage_col).max())
            .rename({"windowly_timeframe": "timeframe"})
        )
    # endregion

    # region 5. Join all data
    result = (
        _build_base_timeframes(ALL_STAYS_T0, patient_information, window_size)
        .sort(STAY_KEY, "timeframe")
        .join(rrt, on=[STAY_KEY, "timeframe"], how="left")
        .join(
            urine_data.select(
                STAY_KEY,
                "timeframe",
                pl.col("uo_interval_ml").alias("UO ml/window"),
                pl.col("uo_interval_ml_per_kg").alias("UO ml/kg/window"),
            ),
            on=[STAY_KEY, "timeframe"],
            how="left",
        )
        .join(
            creatinine_per_frame.select(
                STAY_KEY,
                "timeframe",
                "Creatinine",
                "Creatinine AKI Stage",
            ),
            on=[STAY_KEY, "timeframe"],
            how="left",
        )
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

    # Rename timeframe column if needed
    if timeframe_name != "timeframe":
        result = result.with_columns(
            pl.col("timeframe").alias(timeframe_name)
        ).drop("timeframe")

    return result


# endregion

__all__ = ["AKI_KDIGO"]
