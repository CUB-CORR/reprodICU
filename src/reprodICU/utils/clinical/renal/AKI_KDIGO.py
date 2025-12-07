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
    _build_t0,
    _optional_time_bounds_filter,
    _to_lazy,
    get_patient_information,
    get_timeseries_intakeoutput,
    get_timeseries_labs,
)
from .URINE_OUTPUT import URINE_OUTPUT

SECONDS_IN_1H = 60 * 60
SECONDS_IN_48H = 48 * SECONDS_IN_1H
SECONDS_IN_1D = 24 * SECONDS_IN_1H
SECONDS_IN_7D = 7 * SECONDS_IN_1D


################################################################################
################################################################################
# region creatinine helpers


def _prepare_creatinine(
    timeseries_labs: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Prepare creatinine timeseries with baseline calculations.

    Calculates 7-day and 48-hour baseline creatinine values using forward-fill
    join_asof to find the most recent value in those windows.

    Returns:
        LazyFrame with columns:
        - Global ICU Stay ID
        - Time Relative to Admission (seconds)
        - Creatinine (max per hour)
        - 7-day Baseline Creatinine
        - 48-hour Baseline Creatinine
    """
    STAY_KEY = "Global ICU Stay ID"
    TIME_KEY = "Time Relative to Admission (seconds)"

    creatinine_data = (
        timeseries_labs.filter(
            pl.col("Creatinine").struct.field("value").is_not_null()
        )
        .select(
            STAY_KEY,
            TIME_KEY,
            pl.col("Creatinine").struct.field("value").alias("Creatinine"),
        )
        .sort(STAY_KEY, TIME_KEY)
    )

    # Prepare left dataframe with lookback times
    left_data = creatinine_data.with_columns(
        (pl.col(TIME_KEY) - SECONDS_IN_7D).alias("time_minus_7d"),
        (pl.col(TIME_KEY) - SECONDS_IN_48H).alias("time_minus_48h"),
    )

    # Prepare right dataframes for join_asof
    right_data_7d = creatinine_data.select(
        STAY_KEY,
        TIME_KEY,
        pl.col("Creatinine").alias("7-day Baseline Creatinine"),
    )

    right_data_48h = creatinine_data.select(
        STAY_KEY,
        TIME_KEY,
        pl.col("Creatinine").alias("48-hour Baseline Creatinine"),
    )

    # Join with 7-day baseline
    creatinine_with_7d = left_data.join_asof(
        right_data_7d,
        by=STAY_KEY,
        left_on="time_minus_7d",
        right_on=TIME_KEY,
        strategy="forward",
        suffix="_7d",
        coalesce=True,
    )

    # Join with 48-hour baseline
    creatinine_with_baselines = creatinine_with_7d.join_asof(
        right_data_48h,
        by=STAY_KEY,
        left_on="time_minus_48h",
        right_on=TIME_KEY,
        strategy="forward",
        suffix="_48h",
        coalesce=True,
    )

    return creatinine_with_baselines.select(
        STAY_KEY,
        TIME_KEY,
        "Creatinine",
        "7-day Baseline Creatinine",
        "48-hour Baseline Creatinine",
    )


def _creatinine_stage_points(
    creatinine: pl.Expr,
    baseline_7d: pl.Expr,
    baseline_48h: pl.Expr,
) -> pl.Expr:
    """
    Calculate KDIGO AKI stage (0-3) based on creatinine criteria.

    KDIGO Creatinine Criteria:
    - Stage 1: ≥1.5× baseline (7d) OR ≥0.3 mg/dL increase (48h)
    - Stage 2: ≥2× baseline (7d)
    - Stage 3: ≥3× baseline (7d) OR ≥4 mg/dL absolute

    Returns:
        Integer (0-3) representing AKI stage, or None if data insufficient.
    """
    return (
        pl.when(
            pl.any_horizontal(
                creatinine.truediv(baseline_7d).ge(3),
                creatinine.ge(4),
            )
        )
        .then(3)
        .when(creatinine.truediv(baseline_7d).ge(2))
        .then(2)
        .when(
            pl.any_horizontal(
                creatinine.truediv(baseline_7d).ge(1.5),
                creatinine.sub(baseline_48h).ge(0.3),
            )
        )
        .then(1)
        .otherwise(0)
    )


# endregion

################################################################################
################################################################################
# region urine output helpers


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
    STAY_KEY = "Global ICU Stay ID"
    TIMEFRAME_KEY = "timeframe"

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
    STAY_KEY = "Global ICU Stay ID"
    TIMEFRAME_KEY = "timeframe"

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
    STAY_KEY = "Global ICU Stay ID"
    TIMEFRAME_KEY = "timeframe"

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
            ).alias("UO Fixed Block AKI Stage")
        )
        .select(STAY_KEY, TIMEFRAME_KEY, "UO Fixed Block AKI Stage")
    )

    return result


# endregion


################################################################################
################################################################################
# region KDIGO AKI main function
def AKI_KDIGO(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_labs: Optional[pl.LazyFrame] = None,
    timeseries_inout: Optional[pl.LazyFrame] = None,
    *,
    t_0: Optional[int] = 0,
    t_0_per_stay: Optional[pl.LazyFrame] = None,
    t_1: Optional[int] = None,
    window_size: int = SECONDS_IN_1D,
    timeframe_unit: str = "Days",
    timeframe_name: Optional[str] = None,
    consecutive: bool = True,
    any_period: bool = False,
    fixed_block: bool = False,
    include_criteria: bool = False,
    include_intermediate_stages: bool = False,
) -> pl.LazyFrame:
    """
    Compute KDIGO-based Acute Kidney Injury (AKI) stages in long format from raw inputs.

    KDIGO Creatinine Criteria:
    - Stage 1: ≥1.5× baseline (7 days) OR ≥0.3 mg/dL increase (48 hours)
    - Stage 2: ≥2× baseline (7 days)
    - Stage 3: ≥3× baseline (7 days) OR ≥4 mg/dL absolute

    KDIGO Urine Output Criteria (Stage-dependent; multiple methods supported):
    - Consecutive: Each hour must be below threshold for all hours in the window
    - Any Period: Total urine output over any rolling window must be below threshold
    - Fixed Block: Same as any_period but uses fixed 6h/12h/24h blocks

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information; must contain Global ICU Stay ID and
            Admission Weight (kg). Loaded automatically if None.
        timeseries_labs : pl.LazyFrame, optional
            Timeseries lab values including Creatinine. Loaded automatically if None.
        timeseries_inout : pl.LazyFrame, optional
            Intake/output timeseries data. Loaded automatically if None.
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
    """
    # Load defaults if not provided
    if patient_information is None:
        patient_information = get_patient_information()
    if timeseries_labs is None:
        timeseries_labs = get_timeseries_labs()
    if timeseries_inout is None:
        timeseries_inout = get_timeseries_intakeoutput()

    # Validate all required data is available
    required = {
        "patient_information": patient_information,
        "timeseries_labs": timeseries_labs,
        "timeseries_inout": timeseries_inout,
    }
    missing = [name for name, data in required.items() if data is None]
    if missing:
        raise ValueError(
            f"Cannot compute AKI_KDIGO: Missing required datasets: {', '.join(missing)}. "
            f"Ensure they are configured in ~/.reprodICU/PATHS.yaml or provide them explicitly."
        )

    # Auto-generate timeframe_name if needed
    if timeframe_name is None:
        unit = (
            "Days"
            if window_size == SECONDS_IN_1D
            else "Hours" if window_size == SECONDS_IN_1H else "Windows"
        )
        reference = (
            "T_0" if t_0 != 0 or t_0_per_stay is not None else "Admission"
        )
        timeframe_name = f"{unit} Relative to {reference}"

    STAY_KEY = "Global ICU Stay ID"
    TIME_KEY = "Time Relative to Admission (seconds)"

    # Ensure lazy frames
    patient_information = _to_lazy(patient_information)
    timeseries_labs = _to_lazy(timeseries_labs)
    timeseries_inout = _to_lazy(timeseries_inout)

    # region 1. Build T_0
    all_stays = patient_information.select(STAY_KEY)
    all_stays_t0 = _build_t0(all_stays, t_0_per_stay, t_0)
    # endregion

    # region 2. Prepare creatinine data
    creatinine_data = _prepare_creatinine(timeseries_labs)

    # Join with T_0 and assign timeframes
    creatinine_data = (
        creatinine_data.join(all_stays_t0, on=STAY_KEY, how="inner")
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
            pl.col("Creatinine").max().alias("Creatinine"),
            pl.col("7-day Baseline Creatinine")
            .mean()
            .alias("7-day Baseline Creatinine"),
            pl.col("48-hour Baseline Creatinine")
            .mean()
            .alias("48-hour Baseline Creatinine"),
        )
        .with_columns(
            _creatinine_stage_points(
                pl.col("Creatinine"),
                pl.col("7-day Baseline Creatinine"),
                pl.col("48-hour Baseline Creatinine"),
            ).alias("Creatinine AKI Stage")
        )
    )
    # endregion

    # region 3. Prepare urine output data
    urine_data = (
        URINE_OUTPUT(
            patient_information=patient_information,
            timeseries_inout=timeseries_inout,
            t_0=t_0,
            t_0_per_stay=t_0_per_stay,
            t_1=t_1,
            window_size=SECONDS_IN_1H,
        )
        # Keep hourly data for UO staging calculations
        .join(all_stays_t0, on=STAY_KEY, how="inner")
        .cast({"timeframe": int})
        .sort("timeframe")  # Sort for rolling operations
    )
    # endregion

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
        patient_information.select(STAY_KEY, "ICU Length of Stay (days)")
        .with_columns(
            pl.int_ranges(
                0,
                (pl.col("ICU Length of Stay (days)").mul(SECONDS_IN_1D) + 1)
                // window_size,
                step=1,
            )
            .cast(pl.List(float))
            .alias("timeframe")
        )
        .explode("timeframe")
        .select(STAY_KEY, pl.col("timeframe").cast(int))
        .sort(STAY_KEY, "timeframe")
        .collect()
        .lazy()
    )

    result = result.join(
        urine_data, on=[STAY_KEY, "timeframe"], how="left"
    ).join(
        creatinine_per_frame.select(
            STAY_KEY, "timeframe", "Creatinine AKI Stage"
        ),
        on=[STAY_KEY, "timeframe"],
        how="left",
    )

    for method_name, uo_method_df in uo_stages_dict.items():
        result = result.join(
            uo_method_df, on=[STAY_KEY, "timeframe"], how="left"
        )

    # Calculate overall stage as max of all available stages
    stage_cols = [pl.col("Creatinine AKI Stage")]
    for method_name in uo_stages_dict.keys():
        method_col = f"UO {method_name.replace('_', ' ').title()} AKI Stage"
        stage_cols.append(pl.col(method_col))

    result = result.with_columns(
        pl.max_horizontal(*stage_cols).alias("Overall KDIGO AKI Stage")
    ).sort(STAY_KEY, "timeframe")

    # Rename timeframe column if needed
    if timeframe_name != "timeframe":
        result = result.with_columns(
            pl.col("timeframe").alias(timeframe_name)
        ).drop("timeframe")
    # endregion

    return result


__all__ = ["AKI_KDIGO"]
