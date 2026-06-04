"""
Model for End-Stage Liver Disease (MELD) score: predict survival in end-stage liver disease.

This module implements MELD score calculation including serum sodium adjustment.
MELD is a prognostic score used to predict short-term mortality risk in patients with
chronic liver disease or cirrhosis, calculated using INR, bilirubin, and creatinine,
with optional adjustment for serum sodium.

References:
- Kamath PS, Wiesner RH, Malinchoc M, Kremers W, Therneau TM, Kosberg CL, D'Amico G, Dickson ER, Kim WR.
  A model to predict survival in patients with end-stage liver disease.
  Hepatology. 2001 Feb;33(2):464-70.
- OPTN MELD Serum Sodium Policy Changes (January 2016):
  https://optn.transplant.hrsa.gov/news/meld-serum-sodium-policy-changes/
"""

from typing import Optional

import polars as pl

from ...common import (
    _assign_timeframe,
    _build_t0,
    _optional_time_bounds_filter,
    _to_lazy,
    _validate_required_data,
    extract_struct_value,
    get_diagnoses,
    get_patient_information,
    get_rrt,
    get_timeseries_labs,
)

SECONDS_IN_1H = 60 * 60
SECONDS_IN_1D = 24 * SECONDS_IN_1H
SECONDS_IN_7D = 7 * SECONDS_IN_1D


################################################################################
################################################################################
# region MELD helpers
def _adjust_sodium_for_glucose(
    sodium_expr: pl.Expr, glucose_expr: pl.Expr
) -> pl.Expr:
    """
    Adjust serum sodium based on glucose levels using the Hiller formula.

    The formula accounts for pseudohyponatremia in hyperglycemic states:
    Corrected sodium = Measured sodium + 0.024 * (Serum glucose - 100)

    Returns
    -------
        pl.Expr
            Adjusted sodium value
    """
    return (
        pl.when(sodium_expr.is_not_null() & glucose_expr.is_not_null())
        .then(sodium_expr + 0.024 * (glucose_expr - 100))
        .otherwise(sodium_expr)
    )


def _determine_liver_etiology(diagnoses: pl.LazyFrame) -> pl.LazyFrame:
    """
    Determine liver disease etiology from diagnosis codes.

    The original MELD equation included an etiology coefficient. In the MELD-Na version,
    this coefficient is not used, but we calculate it for completeness.

    Etiology categorization:
    - Cholestatic: Primary biliary cirrhosis, primary sclerosing cholangitis
    - Alcoholic: Alcohol-related liver disease
    - Other: All other liver diseases

    Returns
    -------
        pl.LazyFrame
            DataFrame with etiology classification
    """
    filter_str = r"(?i)liver|hepatic|cirrhosis"
    cholestatic_str = (
        r"(?i)cholestatic|primary biliary|primary sclerosing|biliary cirrhosis"
    )
    alcoholic_str = r"(?i)alcoholic|alcohol|ethanol"

    return (
        diagnoses.filter(
            pl.col("Diagnosis Description").str.contains(filter_str)
        )
        .with_columns(
            pl.when(pl.col("Diagnosis Description").str.contains(cholestatic_str))
            .then(pl.lit("cholestatic"))
            .when(pl.col("Diagnosis Description").str.contains(alcoholic_str))
            .then(pl.lit("alcoholic"))
            .otherwise(pl.lit("other"))
            .alias("Liver Disease Etiology")
        )
        .group_by("Global ICU Stay ID")
        .agg(
            pl.when(pl.col("Liver Disease Etiology").eq(pl.lit("cholestatic")).any())
            .then(pl.lit("cholestatic"))
            .when(pl.col("Liver Disease Etiology").eq(pl.lit("alcoholic")).any())
            .then(pl.lit("alcoholic"))
            .otherwise(pl.lit("other"))
            .alias("Liver Disease Etiology")
        )
    ) # fmt: skip


def _calculate_meld_score(
    creatinine_expr: pl.Expr,
    bilirubin_expr: pl.Expr,
    inr_expr: pl.Expr,
    sodium_expr: pl.Expr,
    glucose_expr: pl.Expr,
    rrt_status_expr: pl.Expr,
    etiology_expr: pl.Expr,
) -> pl.Expr:
    """
    Calculate MELD score based on laboratory values.

    MELD formula:
    Initial MELD(i) = 0.957 * ln(creatinine) + 0.378 * ln(bilirubin) + 1.120 * ln(INR) + 0.643

    For MELD > 11 with sodium adjustment:
    MELD = MELD(i) + 1.32*(137-Na) – [0.033*MELD(i)*(137-Na)]

    Special considerations:
    - Laboratory values less than 1.0 are set to 1.0
    - Creatinine is capped at 4.0 mg/dL
    - Creatinine is set to 4.0 mg/dL for patients on dialysis
    - Sodium values are limited to range 125-137 mmol/L
    - Maximum MELD score is 40

    Returns
    -------
        pl.Expr
            MELD score
    """
    # Apply minimum values and adjustments
    creatinine_adj = (
        pl.when((creatinine_expr.clip(1.0, None) > 4.0) | rrt_status_expr)
        .then(4.0)
        .otherwise(creatinine_expr.clip(1.0, None))
    )

    bilirubin_adj = bilirubin_expr.clip(1.0, None)
    inr_adj = inr_expr.clip(1.0, None)
    sodium_adj = _adjust_sodium_for_glucose(sodium_expr, glucose_expr).clip(125, 137) # fmt: skip

    # Etiology coefficient
    etiology_coeff = (
        pl.when(etiology_expr.is_in(["cholestatic", "alcoholic"]))
        .then(0)
        .otherwise(1)
    )

    # Calculate initial MELD
    meld_initial = (
        0.957 * creatinine_adj.log()
        + 0.378 * bilirubin_adj.log()
        + 1.120 * inr_adj.log()
        + 0.643  # * etiology_coeff not included in updated version
    )

    # Apply sodium adjustment for MELD > 11
    meld_final = (
        pl.when(meld_initial > 11)
        .then(
            meld_initial
            + 1.32 * (137 - sodium_adj)
            - 0.033 * meld_initial * (137 - sodium_adj)
        )
        .otherwise(meld_initial)
    )

    # Round to tenth decimal place, multiply by 10, and cap at 40
    return meld_final.round(1).mul(10).clip(None, 40)


# endregion


################################################################################
################################################################################
# region MELD main function
def MELD(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_labs: Optional[pl.LazyFrame] = None,
    rrt_data: Optional[pl.LazyFrame] = None,
    diagnoses: Optional[pl.LazyFrame] = None,
    *,
    t_0: Optional[int] = 0,
    t_0_per_stay: Optional[pl.LazyFrame] = None,
    t_1: Optional[int] = None,
    window_size: int = SECONDS_IN_1D,
    timeframe_unit: str = "Days",
    timeframe_name: Optional[str] = None,
) -> pl.LazyFrame:
    """
    Compute MELD (Model for End-Stage Liver Disease) scores in long format from raw inputs.

    The MELD score is used to estimate disease severity, determine prognosis, and prioritize
    patients for liver transplantation. This implementation follows the 2016 update which
    includes sodium adjustment.

    MELD Formula:
    Initial MELD(i) = 0.957 × ln(creatinine) + 0.378 × ln(bilirubin) + 1.120 × ln(INR) + 0.643

    For MELD > 11 with sodium adjustment:
    MELD = MELD(i) + 1.32×(137-Na) - [0.033×MELD(i)×(137-Na)]

    Special considerations:
    - Laboratory values less than 1.0 are set to 1.0
    - Creatinine is capped at 4.0 mg/dL
    - Creatinine is set to 4.0 mg/dL for patients on dialysis (2+ treatments in past week or 24+ hours CVVHD)
    - Sodium values are limited to range 125-137 mmol/L
    - Maximum MELD score is 40

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information; must contain Global ICU Stay ID.
            Loaded automatically if None.
        timeseries_labs : pl.LazyFrame, optional
            Laboratory timeseries data with creatinine, bilirubin, INR, sodium, glucose measurements.
            Loaded automatically if None.
        rrt_data : pl.LazyFrame, optional
            Renal Replacement Therapy data for dialysis status determination.
            Loaded automatically if None.
        diagnoses : pl.LazyFrame, optional
            Diagnosis data for liver disease etiology determination.
            Loaded automatically if None.
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
            Semantic unit for timeframe (default: "Days"). Does not affect calculation.
        timeframe_name : str, optional
            Name for output timeframe column. Auto-generated if None.

    Returns
    -------
        pl.LazyFrame
            One row per (stay, timeframe) with columns:
            - Global ICU Stay ID
            - T_0
            - timeframe (or custom name)
            - MELD Score
            - Creatinine (mg/dL)
            - Bilirubin (mg/dL)
            - INR
            - Sodium (mmol/L)
            - Glucose (mg/dL)
            - Had Recent RRT (boolean)
            - Liver Disease Etiology
    """
    # Load defaults if not provided
    if patient_information is None:
        patient_information = get_patient_information()
    if timeseries_labs is None:
        timeseries_labs = get_timeseries_labs()
    if rrt_data is None:
        rrt_data = get_rrt()
    if diagnoses is None:
        diagnoses = get_diagnoses()

    # Validate all required data is available
    required = {
        "patient_information": patient_information,
        "timeseries_labs": timeseries_labs,
    }
    _validate_required_data(concept="MELD", required_data=required)

    STAY_KEY = "Global ICU Stay ID"
    TIME_KEY = "Time Relative to Admission (seconds)"

    patient_information = _to_lazy(patient_information)
    timeseries_labs = _to_lazy(timeseries_labs)
    rrt_data = _to_lazy(rrt_data) if rrt_data is not None else None
    diagnoses = _to_lazy(diagnoses) if diagnoses is not None else None

    # Build T_0
    all_stays = patient_information.select(STAY_KEY)
    all_stays_t0 = _build_t0(all_stays, t_0_per_stay, t_0)

    # Filter and prepare lab data
    LABS = ["Creatinine", "Bilirubin", "INR", "Sodium", "Glucose"]
    lab_data = (
        timeseries_labs.filter(
            pl.any_horizontal(pl.col(col).is_not_null() for col in LABS)
        )
        .join(all_stays_t0, on=STAY_KEY, how="inner")
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
        lab_data = lab_data.filter(cond)

    # Extract lab values from structs
    lab_data = lab_data.with_columns(
        extract_struct_value("Creatinine", ["Serum or Plasma", "Blood"]).alias("Creatinine"),
        extract_struct_value("Bilirubin",  ["Serum or Plasma", "Blood"]).alias("Bilirubin"),
        extract_struct_value("INR",        ["Serum or Plasma", "Blood"]).alias("INR"),
        extract_struct_value("Sodium",     ["Serum or Plasma", "Blood"]).alias("Sodium"),
        extract_struct_value("Glucose",    ["Serum or Plasma", "Blood"]).alias("Glucose"),
    ).select(
        STAY_KEY,
        TIME_KEY,
        "timeframe",
        "T_0",
        "Time Relative to T_0 (seconds)",
        *LABS
    ) # fmt: skip

    # Aggregate labs to worst per timeframe
    lab_per_frame = lab_data.group_by(STAY_KEY, "timeframe", "T_0").agg(
        # Take most recent (worst) values per timeframe
        pl.col("Creatinine").max(),
        pl.col("Bilirubin").max(),
        pl.col("INR").max(),
        pl.col("Sodium").max(),
        pl.col("Glucose").max(),
    )

    # Get RRT status for each timeframe (check if patient had recent RRT at end of timeframe)
    rrt_status_per_frame = None
    if rrt_data is not None:
        # For each timeframe, check RRT status at the end of that timeframe
        timeframe_end_times = (
            lab_per_frame.select(STAY_KEY, "timeframe", "T_0")
            .with_columns(
                pl.col("timeframe")
                .add(1)
                .mul(window_size)
                .add(pl.col("T_0"))
                .alias("timeframe_end")
            )
            .select(STAY_KEY, "timeframe", "timeframe_end")
        )

        rrt_status_per_frame = (
            timeframe_end_times.join(
                rrt_data.select(
                    STAY_KEY,
                    "Renal Replacement Therapy Start Relative to Admission (seconds)",
                    "Renal Replacement Therapy End Relative to Admission (seconds)",
                    "Renal Replacement Therapy Duration (hours)",
                ),
                on=STAY_KEY,
                how="left",
            )
            .filter(
                pl.col("Renal Replacement Therapy Start Relative to Admission (seconds)")
                <= pl.col("timeframe_end"),
                pl.col("Renal Replacement Therapy End Relative to Admission (seconds)")
                >= (pl.col("timeframe_end") - SECONDS_IN_7D),
            )
            .group_by(STAY_KEY, "timeframe")
            .agg(
                pl.len().alias("rrt_count"),
                pl.col("Renal Replacement Therapy Duration (hours)")
                .sum()
                .alias("rrt_duration_hours"),
            )
            .with_columns(
                (
                    (pl.col("rrt_count") >= 2)
                    | (pl.col("rrt_duration_hours") >= 24)
                ).alias("Had Recent RRT")
            )
            .select(STAY_KEY, "timeframe", "Had Recent RRT")
        ) # fmt: skip

    # Get liver disease etiology (static per patient)
    etiology_per_patient = None
    if diagnoses is not None:
        etiology_per_patient = _determine_liver_etiology(diagnoses)

    # Combine all data
    result = lab_per_frame

    if rrt_status_per_frame is not None:
        result = result.join(
            rrt_status_per_frame, on=[STAY_KEY, "timeframe"], how="left"
        )

    if etiology_per_patient is not None:
        result = result.join(etiology_per_patient, on=STAY_KEY, how="left")

    # Fill missing values
    result = result.with_columns(
        pl.col("Had Recent RRT").fill_null(False),
        pl.col("Liver Disease Etiology").fill_null("other"),
    )

    # Calculate MELD score
    result = result.with_columns(
        _calculate_meld_score(
            pl.col("Creatinine"),
            pl.col("Bilirubin"),
            pl.col("INR"),
            pl.col("Sodium"),
            pl.col("Glucose"),
            pl.col("Had Recent RRT"),
            pl.col("Liver Disease Etiology"),
        ).alias("MELD Score")
    )

    # Handle timeframe naming
    if timeframe_name is not None:
        result = result.with_columns(pl.col("timeframe").alias(timeframe_name))

    select_cols = [
        STAY_KEY,
        "T_0",
        "timeframe" if timeframe_name is None else timeframe_name,
        "MELD Score",
        "Creatinine",
        "Bilirubin",
        "INR",
        "Sodium",
        "Glucose",
        "Had Recent RRT",
        "Liver Disease Etiology",
    ]

    return result.sort(STAY_KEY, "timeframe").select(select_cols)


# endregion


__all__ = ["MELD"]
