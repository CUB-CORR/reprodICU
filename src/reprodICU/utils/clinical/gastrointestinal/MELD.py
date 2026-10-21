"""
Model for End-Stage Liver Disease (MELD) score: predict survival in end-stage liver disease.

This module implements the variants of the MELD score used by the OPTN:
- MELD (2002): creatinine, bilirubin and INR.
- MELD-Na (2016): adds an adjustment for serum sodium.
- MELD 3.0: adds sex and albumin, new coefficients and interaction terms, and lowers the creatinine cap from 4.0 to 3.0 mg/dL.
MELD is a prognostic score used to predict short-term mortality risk in patients with
chronic liver disease or cirrhosis.

References:
- Kamath PS, Wiesner RH, Malinchoc M, Kremers W, Therneau TM, Kosberg CL, D'Amico G, Dickson ER, Kim WR.
  A model to predict survival in patients with end-stage liver disease.
  Hepatology. 2001 Feb;33(2):464-70.
- OPTN. Improving Liver Allocation: MELD, PELD, Status 1A, Status 1B. Notice of OPTN Policy and Guidance Changes (policy 9.1.D: MELD Score).
  https://www.hrsa.gov/sites/default/files/hrsa/optn/policy-guid-change_impr-liv-alloc-meld-peld-sta-1a-sta-1b_liv.pdf
- Kim WR, Mannalithara A, Heimbach JK, Kamath PS, Asrani SK, Biggins SW, Wood NL, Gentry SE, Kwong AJ.
  MELD 3.0: The Model for End-Stage Liver Disease Updated for the Modern Era.
  Gastroenterology. 2021 Dec;161(6):1887-1895.e4. doi: 10.1053/j.gastro.2021.08.050.
  Epub 2021 Sep 3. PMID: 34481845; PMCID: PMC8608337.
"""

from typing import Literal, Optional

import polars as pl

from ...common import (
    FILTER_NOT_NULL,
    _assign_timeframe,
    _build_base_timeframes,
    _build_t0,
    _get_timeframe_name,
    _optional_timeframe_bounds_filter,
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

STAY_KEY = "Global ICU Stay ID"
TIME_KEY = "Time Relative to Admission (seconds)"


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
    cholestatic_str = r"(?i)cholestatic|primary biliary|primary sclerosing|biliary cirrhosis" # fmt: skip
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


def _improve_labs(labs: pl.LazyFrame, albumin: bool = False) -> pl.LazyFrame:
    LABS = ["Creatinine", "Bilirubin", "INR", "Sodium", "Glucose"]
    if albumin:
        LABS.append("Albumin")
    return (
        labs.with_columns(
            extract_struct_value(lab, ["Serum", "Blood"]).alias(lab)
            for lab in LABS
        )
        .pipe(FILTER_NOT_NULL, LABS)
        .select(STAY_KEY, TIME_KEY, *LABS)
    )


def _recent_rrt_per_timeframe(
    rrt: pl.LazyFrame, base: pl.LazyFrame, window_size: int
) -> pl.LazyFrame:
    """
    Whether a patient had dialysis in the 7 days up to the end of each timeframe:
    2 or more treatments, or 24 hours or more of CVVHD.
    """
    start = "Renal Replacement Therapy Start Relative to Admission (seconds)"
    end   = "Renal Replacement Therapy End Relative to Admission (seconds)"
    hours = "Renal Replacement Therapy Duration (hours)"
    window_end = (pl.col("timeframe") + 1) * window_size + pl.col("T_0")
    return (
        base.join(rrt.select(STAY_KEY, start, end, hours), on=STAY_KEY)
        .filter(
            pl.col(start) <= window_end,
            pl.col(end) >= window_end - SECONDS_IN_7D,
        )
        .group_by(STAY_KEY, "timeframe")
        .agg(((pl.len() >= 2) | (pl.col(hours).sum() >= 24)).alias("Had Recent RRT"))
    )


def _creatinine_for_meld(
    creatinine_expr: pl.Expr, rrt_status_expr: pl.Expr, cap: float
) -> pl.Expr:
    """Creatinine of at least 1.0 and at most `cap`; `cap` for patients on dialysis."""
    return (
        pl.when((creatinine_expr.clip(1.0, None) > cap) | rrt_status_expr)
        .then(cap)
        .otherwise(creatinine_expr.clip(1.0, None))
    )


def _calculate_meld_score(
    creatinine_expr: pl.Expr,
    bilirubin_expr: pl.Expr,
    inr_expr: pl.Expr,
    rrt_status_expr: pl.Expr,
) -> pl.Expr:
    """
    Calculate the original MELD score (2002) from laboratory values.

    MELD = 10 * round(0.957 * ln(creatinine) + 0.378 * ln(bilirubin) + 1.120 * ln(INR) + 0.643, 1)

    - Laboratory values less than 1.0 are set to 1.0
    - Creatinine is capped at 4.0 mg/dL and set to 4.0 mg/dL for patients on dialysis
      (2+ treatments in past week or 24+ hours CVVHD)
    - Maximum MELD score is 40
    """
    creatinine = _creatinine_for_meld(creatinine_expr, rrt_status_expr, 4.0)
    return (
        (
            0.957 * creatinine.log()
            + 0.378 * bilirubin_expr.clip(1.0, None).log()
            + 1.120 * inr_expr.clip(1.0, None).log()
            + 0.643
        )
        .round(1)
        .mul(10)
        .clip(None, 40)
    )


def _calculate_meld_na_score(
    creatinine_expr: pl.Expr,
    bilirubin_expr: pl.Expr,
    inr_expr: pl.Expr,
    sodium_expr: pl.Expr,
    glucose_expr: pl.Expr,
    rrt_status_expr: pl.Expr,
) -> pl.Expr:
    """
    Calculate the MELD-Na score (2016) from laboratory values.

    Initial MELD(i) is the original MELD score (see `_calculate_meld_score`),
    on the 6-40 scale. For MELD(i) > 11:
    MELD-Na = MELD(i) + 1.32*(137-Na) - [0.033*MELD(i)*(137-Na)]

    - Sodium values are limited to range 125-137 mmol/L
    - Maximum MELD score is 40

    Returns
    -------
        pl.Expr
            MELD-Na score
    """
    meld_initial = _calculate_meld_score(creatinine_expr, bilirubin_expr, inr_expr, rrt_status_expr) # fmt: skip
    sodium_adj = _adjust_sodium_for_glucose(sodium_expr, glucose_expr).clip(125, 137) # fmt: skip

    meld_final = (
        pl.when(meld_initial > 11)
        .then(
            meld_initial
            + 1.32 * (137 - sodium_adj)
            - 0.033 * meld_initial * (137 - sodium_adj)
        )
        .otherwise(meld_initial)
    )

    return meld_final.round(0).clip(None, 40)


def _calculate_meld3_score(
    creatinine_expr: pl.Expr,
    bilirubin_expr: pl.Expr,
    inr_expr: pl.Expr,
    sodium_expr: pl.Expr,
    glucose_expr: pl.Expr,
    rrt_status_expr: pl.Expr,
    albumin_expr: pl.Expr,
    female_expr: pl.Expr,
    age_expr: pl.Expr,
) -> pl.Expr:
    """
    Calculate the MELD 3.0 score from laboratory values, sex and age.

    MELD =   1.33 (if female)
           + 4.56 ln(bilirubin)
           + 0.82 (137-Na)
           - 0.24 (137-Na) ln(bilirubin)
           + 9.09 ln(INR)
           + 11.14 ln(creatinine)
           + 1.85 (3.5-albumin)
           - 1.83 (3.5-albumin) ln(creatinine)
           + 6

    For candidates at least 12 and less than 18 years old, the sex term is dropped and
    the constant is 7.33. There is no MELD score under 12 years (PELD instead).

    - Bilirubin, INR and creatinine less than 1.0 are set to 1.0
    - Creatinine is capped at 3.0 mg/dL and set to 3.0 mg/dL for patients on dialysis
    - Sodium values are limited to range 125-137 mmol/L
    - Albumin values are limited to range 1.5-3.5 g/dL
    - The score is limited to range 6-40 and rounded to the nearest whole number

    Returns
    -------
        pl.Expr
            MELD 3.0 score
    """
    creatinine = _creatinine_for_meld(creatinine_expr, rrt_status_expr, 3.0)
    bilirubin = bilirubin_expr.clip(1.0, None)
    sodium = _adjust_sodium_for_glucose(sodium_expr, glucose_expr).clip(125, 137)
    albumin = albumin_expr.clip(1.5, 3.5)

    score = (
        4.56 * bilirubin.log()
        + 0.82 * (137 - sodium)
        - 0.24 * (137 - sodium) * bilirubin.log()
        + 9.09 * inr_expr.clip(1.0, None).log()
        + 11.14 * creatinine.log()
        + 1.85 * (3.5 - albumin)
        - 1.83 * (3.5 - albumin) * creatinine.log()
    )
    return (
        pl.when(age_expr >= 18).then(score + 6 + 1.33 * female_expr.cast(int))
          .when(age_expr >= 12).then(score + 7.33)
        .clip(6, 40)
        .round(0)
    ) # fmt: skip


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
    version: Literal["MELD", "MELD-Na", "MELD 3.0"] = "MELD-Na",
) -> pl.LazyFrame:
    """
    Compute MELD (Model for End-Stage Liver Disease) scores in long format from raw inputs.

    The MELD score is used to estimate disease severity, determine prognosis, and prioritize
    patients for liver transplantation. `version` selects the formula: "MELD" (2002),
    "MELD-Na" (2016, with sodium adjustment; default) or "MELD 3.0" (with sex and albumin).
    See `_calculate_meld_score`, `_calculate_meld_na_score` and
    `_calculate_meld3_score` for the formulas and special considerations.

    MELD 3.0 needs the albumin lab and the Gender and Admission Age (years) of the patient
    information. The admission age stands in for the age at registration (at least 18: adult
    formula, 12 to 17: adolescent formula, under 12: no score).

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
        version : {"MELD", "MELD-Na", "MELD 3.0"}, optional
            Formula to use (default: "MELD-Na").

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
            - Albumin (g/dL; MELD 3.0 only)
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
    _validate_required_data("MELD", required)

    patient_information = _to_lazy(patient_information)
    timeseries_labs = _to_lazy(timeseries_labs)

    # Build T_0 and base timeframes
    ALL_STAYS = patient_information.select(STAY_KEY)
    ALL_STAYS_T0 = _build_t0(ALL_STAYS, t_0_per_stay, t_0)
    timeframe_name = _get_timeframe_name(
        timeframe_name, window_size, t_0, t_0_per_stay
    )
    base = _build_base_timeframes(ALL_STAYS_T0, patient_information, window_size)

    # region component data
    labs = _improve_labs(timeseries_labs, albumin=version == "MELD 3.0")
    lab_names = [c for c in labs.collect_schema().names() if c not in (STAY_KEY, TIME_KEY)]
    labs_tf = (
        labs.join(ALL_STAYS_T0, on=STAY_KEY, how="inner")
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window_size))
        .group_by(STAY_KEY, "timeframe")
        .agg(
            # the worst is the highest value, except for albumin
            pl.col(lab_names).exclude("Albumin").max(),
            *([pl.col("Albumin").min()] if "Albumin" in lab_names else []),
        )
    )

    out = base.join(labs_tf, on=[STAY_KEY, "timeframe"], how="left")
    if rrt_data is not None:
        rrt_tf = _recent_rrt_per_timeframe(_to_lazy(rrt_data), base, window_size)
        out = out.join(rrt_tf, on=[STAY_KEY, "timeframe"], how="left")
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Boolean).alias("Had Recent RRT"))
    if diagnoses is not None:
        etiology = _determine_liver_etiology(_to_lazy(diagnoses))
        out = out.join(etiology, on=STAY_KEY, how="left")
    else:
        out = out.with_columns(
            pl.lit(None, dtype=pl.String).alias("Liver Disease Etiology")
        )
    out = out.with_columns(
        pl.col("Had Recent RRT").fill_null(False),
        pl.col("Liver Disease Etiology").fill_null("other"),
    )
    # endregion

    # region score
    if version == "MELD 3.0":
        out = out.join(
            patient_information.select(STAY_KEY, "Gender", "Admission Age (years)"),
            on=STAY_KEY,
            how="left",
        )
        score = _calculate_meld3_score(
            pl.col("Creatinine"),
            pl.col("Bilirubin"),
            pl.col("INR"),
            pl.col("Sodium"),
            pl.col("Glucose"),
            pl.col("Had Recent RRT"),
            pl.col("Albumin"),
            pl.col("Gender") == "Female",
            pl.col("Admission Age (years)"),
        )
    elif version == "MELD":
        score = _calculate_meld_score(
            pl.col("Creatinine"),
            pl.col("Bilirubin"),
            pl.col("INR"),
            pl.col("Had Recent RRT"),
        )
    else:
        score = _calculate_meld_na_score(
            pl.col("Creatinine"),
            pl.col("Bilirubin"),
            pl.col("INR"),
            pl.col("Sodium"),
            pl.col("Glucose"),
            pl.col("Had Recent RRT"),
        )
    # endregion

    return (
        out.filter(
            _optional_timeframe_bounds_filter("timeframe", window_size, t_0, t_1)
        )
        .select(
            STAY_KEY,
            "T_0",
            pl.col("timeframe").alias(timeframe_name),
            score.alias("MELD Score"),
            *lab_names,
            "Had Recent RRT",
            "Liver Disease Etiology",
        )
        .sort(STAY_KEY, timeframe_name)
    )


# endregion


__all__ = ["MELD"]
