"""
AKI-FB risk score: predict a strongly positive fluid balance (> 2 L within 72 h)
after the diagnosis of acute kidney injury (AKI) in critically ill adults.

Output columns per row:
- Global ICU Stay ID
- T_0 (seconds from admission used as reference; intended to be the AKI onset)
- timeframe (0-indexed integer window; 0 = day of AKI diagnosis)
- AKI-FB Risk Score (0-61)
- The components the score is built from

Time is in seconds. Windows determined by floor((time - T_0)/window_size).
Missing information counts as 0 points.

Scoring (variables measured in the window; sepsis must start at or before T_0):
- Post-trauma admission: 5
- Sepsis: 4 / septic shock: 2 (as published)
- Highest creatinine > 100 umol/L (1.13 mg/dL): 6
- Accumulated fluid balance: < -100 mL: 0, -100 to < 100 mL: 7, >= 100 mL: 19
- Mechanical ventilation: 3
- Norepinephrine equivalent: none 0, <= 0.07 mcg/kg/min: 1, > 0.07 mcg/kg/min: 8
- Lactate >= 2 mmol/L: 3
- Transfusion of blood products: 2
- Nutrition support (enteral or parenteral): 11

Reported cut-off: 32 (derivation) / 26 (validation).

References:
- White KC, Bellomo R, Laupland KB, Gatton ML, Ostermann M, McIlroy P, Luke S, Garrett P, Tabah A, Whebell S, Marella P, McCullough J, Shekar K, Attokaran AG, Kumar A, Meyer J, Sanderson B, Serpa-Neto A; Queensland Critical Care Research Network (QCCRN).
  Predicting a strongly positive fluid balance in critically ill patients with acute kidney injury: A multicentre, international study.
  J Crit Care. 2025 Jun;87:155016.
  doi: 10.1016/j.jcrc.2025.155016. Epub 2025 Jan 23. PMID: 39855144.
"""

from typing import Optional

import polars as pl

from ...admission import COMMON_ADMISSION_CATEGORIES
from ...common import (
    _assign_timeframe,
    _build_base_timeframes,
    _build_t0,
    _get_timeframe_name,
    _to_lazy,
    _validate_required_data,
    extract_struct_value,
    get_medications,
    get_patient_information,
    get_timeseries_intakeoutput,
    get_timeseries_labs,
    get_ventilation,
    intervention_per_timeframe,
)
from ...sepsis.SEPSIS import SEPSIS
from ..pharmocological.NOREPINEPHRINE_EQUIVALENT_DOSAGE import (
    NOREPINEPHRINE_EQUIVALENT_DOSAGE,
)

SECONDS_IN_1H = 60 * 60
SECONDS_IN_1D = 24 * SECONDS_IN_1H

STAY_KEY = "Global ICU Stay ID"
TIME_KEY = "Time Relative to Admission (seconds)"
DRUG_START_KEY = "Drug Start Relative to Admission (seconds)"
DRUG_END_KEY = "Drug End Relative to Admission (seconds)"

CREATININE_THRESHOLD = 100 / 88.4  # 100 umol/L in mg/dL
NEE_HIGH_THRESHOLD = 0.07  # mcg/kg/min


################################################################################
################################################################################
# region AKI-FB helpers
def _calculate_aki_fb_score(
    *,
    trauma: pl.Expr,
    sepsis: pl.Expr,
    creatinine: pl.Expr,
    fluid_balance: pl.Expr,
    ventilated: pl.Expr,
    norepinephrine: pl.Expr,
    norepinephrine_high: pl.Expr,
    lactate: pl.Expr,
    transfusion: pl.Expr,
    nutrition: pl.Expr,
) -> pl.Expr:
    """
    Sum the AKI-FB risk score points; null inputs contribute 0 points.

    Arguments
    ---------
        trauma, ventilated, norepinephrine, norepinephrine_high, transfusion, nutrition : pl.Expr
            Boolean flags.
        sepsis : pl.Expr
            "SEPSIS" | "SHOCK" | null.
        creatinine : pl.Expr
            Highest creatinine (mg/dL).
        fluid_balance : pl.Expr
            Accumulated fluid balance (mL).
        lactate : pl.Expr
            Highest lactate (mmol/L).

    Returns
    -------
        pl.Expr
            AKI-FB risk score (0-61)
    """
    return (
        pl.when(trauma).then(5).otherwise(0)
        + pl.when(sepsis == "SEPSIS").then(4)
        .when(sepsis == "SHOCK").then(2)
        .otherwise(0)
        + pl.when(creatinine > CREATININE_THRESHOLD).then(6).otherwise(0)
        + pl.when(fluid_balance >= 100).then(19)
        .when(fluid_balance >= -100).then(7)
        .otherwise(0)
        + pl.when(ventilated).then(3).otherwise(0)
        + pl.when(norepinephrine_high).then(8)
        .when(norepinephrine).then(1)
        .otherwise(0)
        + pl.when(lactate >= 2).then(3).otherwise(0)
        + pl.when(transfusion).then(2).otherwise(0)
        + pl.when(nutrition).then(11).otherwise(0)
    )  # fmt: skip


# endregion


################################################################################
################################################################################
# region AKI-FB main function
def AKI_FB(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_labs: Optional[pl.LazyFrame] = None,
    timeseries_inout: Optional[pl.LazyFrame] = None,
    medications: Optional[pl.LazyFrame] = None,
    ventilation: Optional[pl.LazyFrame] = None,
    sepsis: Optional[pl.LazyFrame] = None,
    *,
    t_0: Optional[int] = 0,
    t_0_per_stay: Optional[pl.LazyFrame] = None,
    t_1: Optional[int] = None,
    window_size: int = SECONDS_IN_1D,
    timeframe_unit: str = "Days",  # semantics only; output timeframe is numeric
    timeframe_name: Optional[str] = None,
) -> pl.LazyFrame:
    """
    Compute the AKI-FB risk score in long format from raw inputs.

    Intended use: pass the AKI onset per stay as `t_0_per_stay`; timeframe 0 is then
    the day of AKI diagnosis, which is what the score was derived on.

    Creatinine and lactate use the window maximum, fluid balance the window sum, and
    ventilation, norepinephrine, blood products and nutrition count if active in the
    window. Sepsis is the last label (SEPSIS or SHOCK) at or before T_0 and carries
    over to later timeframes. Missing information scores 0 points.

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information; must contain Global ICU Stay ID, ICU Length
            of Stay (days), Admission Weight (kg) and the admission columns used by
            COMMON_ADMISSION_CATEGORIES. Loaded automatically if None.
        timeseries_labs : pl.LazyFrame, optional
            Laboratory timeseries with Creatinine (mg/dL) and Lactate (mmol/L).
            Loaded automatically if None.
        timeseries_inout : pl.LazyFrame, optional
            Intake/output timeseries with "Fluid balance" (mL).
            Loaded automatically if None.
        medications : pl.LazyFrame, optional
            Medication administrations (vasopressors, blood products, nutrition).
            Also used to compute SEPSIS if not provided. Loaded automatically if None.
        ventilation : pl.LazyFrame, optional
            Ventilation episodes; invasive ventilation / tracheostomy counts.
            Also used to compute SEPSIS if not provided. Loaded automatically if None.
        sepsis : pl.LazyFrame, optional
            Output of SEPSIS (Seymour anchor) computed relative to admission (t_0=0) with
            window_size=3600. Computed automatically if None.
        t_0 : int, optional
            Scalar reference time (seconds from admission). Defaults to 0 (admission).
            Ignored when t_0_per_stay is provided.
        t_0_per_stay : pl.LazyFrame, optional
            Per-stay T_0 overrides (AKI onset) with columns [Global ICU Stay ID, T_0].
        t_1 : int, optional
            Optional upper time bound (seconds from admission); timeframes starting at
            or after it are dropped.
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
            - AKI-FB Risk Score (0-61; maximum without missing information)
            - Post-Trauma Admission (boolean)
            - Sepsis Before T_0 ("SEPSIS" | "SHOCK" | null; last label at or before T_0)
            - Creatinine (mg/dL; window max)
            - Fluid Balance (mL; window sum)
            - Ventilated (boolean)
            - Norepinephrine Use (boolean; norepinephrine equivalent dose > 0)
            - High-Dose Norepinephrine Use (boolean; norepinephrine equivalent dose > 0.07 mcg/kg/min)
            - Lactate (mmol/L; window max)
            - Blood Product Transfusion (boolean)
            - Nutrition Support (boolean)
    """
    # Load defaults if not provided
    if patient_information is None:
        patient_information = get_patient_information()
    if timeseries_labs is None:
        timeseries_labs = get_timeseries_labs()
    if timeseries_inout is None:
        timeseries_inout = get_timeseries_intakeoutput()
    if medications is None:
        medications = get_medications()
    if ventilation is None:
        ventilation = get_ventilation()

    # Validate all required data is available
    required = {
        "patient_information": patient_information,
        "timeseries_labs": timeseries_labs,
        "timeseries_inout": timeseries_inout,
        "medications": medications,
        "ventilation": ventilation,
    }
    _validate_required_data("AKI_FB", required)

    # Ensure lazy
    patient_information = _to_lazy(patient_information)
    timeseries_labs = _to_lazy(timeseries_labs)
    timeseries_inout = _to_lazy(timeseries_inout)
    medications = _to_lazy(medications)
    ventilation = _to_lazy(ventilation)

    # Build T_0 and base timeframes
    all_stays_t0 = _build_t0(
        patient_information.select(STAY_KEY), t_0_per_stay, t_0
    )
    base = _build_base_timeframes(all_stays_t0, patient_information, window_size)
    if t_1 is not None:
        base = base.filter(
            pl.col("T_0") + pl.col("timeframe") * window_size < t_1
        )
    timeframe_name = _get_timeframe_name(
        timeframe_name, window_size, t_0, t_0_per_stay
    )

    # Trauma admission (static per stay)
    trauma = (
        COMMON_ADMISSION_CATEGORIES(patient_information)
        .select(STAY_KEY, "Trauma Admission")
        .rename({"Trauma Admission": "Post-Trauma Admission"})
    )

    # Sepsis (hourly, relative to admission): last label at or before T_0
    if sepsis is None:
        sepsis = SEPSIS(
            patient_information,
            timeseries_labs=timeseries_labs,
            timeseries_inout=timeseries_inout,
            medications=medications,
            ventilation=ventilation,
            t_0=0,
            window_size=SECONDS_IN_1H,
            anchors={"seymour"},
        )
    sepsis_before_t0 = (
        _to_lazy(sepsis)
        .filter(pl.col("SEPSIS").is_not_null())
        .join(all_stays_t0, on=STAY_KEY, how="inner")
        .filter(pl.col("timeframe") * SECONDS_IN_1H <= pl.col("T_0"))
        .sort(STAY_KEY, "timeframe")
        .group_by(STAY_KEY)
        .last()
        .select(STAY_KEY, pl.col("SEPSIS").alias("Sepsis Before T_0"))
    )

    # Worst laboratory values per timeframe
    labs = (
        timeseries_labs.join(all_stays_t0, on=STAY_KEY, how="inner")
        .with_columns(
            _assign_timeframe(TIME_KEY, window_size).alias("timeframe"),
            extract_struct_value("Creatinine", ["Serum", "Blood"]).alias("Creatinine"),
            extract_struct_value("Lactate", ["Blood", "Plasma", "Serum"]).alias("Lactate"),
        )
        .filter(pl.col("timeframe") >= 0)
        .group_by(STAY_KEY, "timeframe")
        .agg(pl.col("Creatinine").max(), pl.col("Lactate").max())
    )  # fmt: skip

    # Fluid balance per timeframe
    fluid_balance = (
        timeseries_inout.join(all_stays_t0, on=STAY_KEY, how="inner")
        .with_columns(_assign_timeframe(TIME_KEY, window_size).alias("timeframe"))
        .filter(pl.col("timeframe") >= 0)
        .group_by(STAY_KEY, "timeframe")
        .agg(
            # the sum of no readings is not a balance of 0
            pl.when(pl.col("Fluid balance").is_not_null().any())
            .then(pl.col("Fluid balance").sum())
            .alias("Fluid Balance")
        )
    )

    # Interventions active per timeframe: name -> (data, start column, end column)
    nee = NOREPINEPHRINE_EQUIVALENT_DOSAGE(patient_information, medications)
    interventions = {
        "Ventilated": (
            ventilation.filter(
                pl.col("Ventilation Type").is_in(["invasive ventilation", "tracheostomy"])
                | pl.col("Ventilation Type").is_null()
            ),
            "Ventilation Start Relative to Admission (seconds)",
            "Ventilation End Relative to Admission (seconds)",
        ),
        "Norepinephrine Use": (nee.filter(pl.col("Norepinephrine Equivalent Dose (mcg/kg/min)") > 0), DRUG_START_KEY, DRUG_END_KEY),
        "High-Dose Norepinephrine Use": (
            nee.filter(pl.col("Norepinephrine Equivalent Dose (mcg/kg/min)") > NEE_HIGH_THRESHOLD),
            DRUG_START_KEY,
            DRUG_END_KEY,
        ),
        "Blood Product Transfusion": (
            medications.filter(pl.col("Drug Class").str.starts_with("blood product")),
            DRUG_START_KEY,
            DRUG_END_KEY,
        ),
        "Nutrition Support": (
            medications.filter(pl.col("Drug Class").is_in(["enteral", "parenteral"])),
            DRUG_START_KEY,
            DRUG_END_KEY,
        ),
    }  # fmt: skip

    # Combine all components starting from base timeframes
    keys = [STAY_KEY, "timeframe"]
    result = base.join(labs, on=keys, how="left").join(fluid_balance, on=keys, how="left")
    for name, (data, start_col, end_col) in interventions.items():
        active = intervention_per_timeframe(
            data,
            patient_information,
            start_col=start_col,
            end_col=end_col,
            t_0=t_0,
            t_0_per_stay=t_0_per_stay,
            window_size=window_size,
        ).select(STAY_KEY, pl.col("timeframe").cast(int), pl.col("intervention").alias(name))  # fmt: skip
        result = result.join(active, on=keys, how="left")
    result = (
        result.join(trauma, on=STAY_KEY, how="left")
        .join(sepsis_before_t0, on=STAY_KEY, how="left")
        .with_columns(pl.col(*interventions, "Post-Trauma Admission").fill_null(False))
    )

    # Calculate AKI-FB risk score
    result = result.with_columns(
        _calculate_aki_fb_score(
            trauma=pl.col("Post-Trauma Admission"),
            sepsis=pl.col("Sepsis Before T_0"),
            creatinine=pl.col("Creatinine"),
            fluid_balance=pl.col("Fluid Balance"),
            ventilated=pl.col("Ventilated"),
            norepinephrine=pl.col("Norepinephrine Use"),
            norepinephrine_high=pl.col("High-Dose Norepinephrine Use"),
            lactate=pl.col("Lactate"),
            transfusion=pl.col("Blood Product Transfusion"),
            nutrition=pl.col("Nutrition Support"),
        ).alias("AKI-FB Risk Score")
    )

    # Final output
    return (
        result.with_columns(pl.col("timeframe").alias(timeframe_name))
        .sort(STAY_KEY, "timeframe")
        .select(
            STAY_KEY,
            "T_0",
            timeframe_name,
            "AKI-FB Risk Score",
            "Post-Trauma Admission",
            "Sepsis Before T_0",
            "Creatinine",
            "Fluid Balance",
            "Ventilated",
            "Norepinephrine Use",
            "High-Dose Norepinephrine Use",
            "Lactate",
            "Blood Product Transfusion",
            "Nutrition Support",
        )
    )


# endregion


__all__ = ["AKI_FB"]
