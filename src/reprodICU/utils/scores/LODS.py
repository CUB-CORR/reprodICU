"""
LODS: compute Logistic Organ Dysfunction System (LODS and mLODS) in long format directly from raw inputs.

Output columns per row:
- Global ICU Stay ID
- T_0 (seconds from admission used as reference)
- timeframe (0-indexed integer window)
- LODS Score or mLODS Score (sum of component points)
- Component points (GCS, heart rate, systolic BP, P/F ratio, BUN, creatinine, urine output, WBC, bilirubin, platelets, INR)

Time is in seconds. Windows determined by floor((time - T_0)/window_size).
Worst-within-window aggregation is applied per component.

SOURCES
-------
- Le Gall JR, Klar J, Lemeshow S, Saulnier F, Alberti C, Artigas A, Teres D.
  The Logistic Organ Dysfunction system. A new way to assess organ dysfunction in the intensive care unit. ICU Scoring Group.
  JAMA. 1996 Sep 11;276(10):802-10.
  doi: 10.1001/jama.276.10.802. PMID: 8769590.
- Seymour CW, Liu VX, Iwashyna TJ, Brunkhorst FM, Rea TD, Scherag A, Rubenfeld G, Kahn JM, Shankar-Hari M, Singer M, Deutschman CS, Escobar GJ, Angus DC.
  Assessment of Clinical Criteria for Sepsis: For the Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3).
  JAMA. 2016 Feb 23;315(8):762-74.
  doi: 10.1001/jama.2016.0288. PMID: 26903335; PMCID: PMC5433435.
"""

from typing import Optional

import polars as pl

from ..clinical.renal.URINE_OUTPUT import URINE_OUTPUT
from ..common import (
    ScoringTable,
    _assign_timeframe,
    _build_base_timeframes,
    _build_t0,
    _get_timeframe_name,
    _optional_time_bounds_filter,
    _validate_required_data,
    extract_struct_value,
    get_patient_information,
    get_timeseries_intakeoutput,
    get_timeseries_labs,
    get_timeseries_respiratory,
    get_timeseries_vitals,
    get_ventilation,
    intervention_per_timeframe,
)
from ..laboratory.oxygenation.PF_RATIO import PaO2_FiO2_RATIO

STAY_KEY = "Global ICU Stay ID"
TIME_KEY = "Time Relative to Admission (seconds)"

SECONDS_IN_1H = 60 * 60
SECONDS_IN_1D = 24 * SECONDS_IN_1H


################################################################################
################################################################################
# region data helpers


def _improve_vitals(vitals: pl.LazyFrame) -> pl.LazyFrame:
    return vitals.select(
        STAY_KEY,
        TIME_KEY,
        "Heart rate",
        pl.coalesce(
            "Invasive systolic arterial pressure",
            "Non-invasive systolic arterial pressure",
        ).alias("Systolic arterial pressure"),
        "Glasgow coma score total",
    )


def _improve_labs(labs: pl.LazyFrame) -> pl.LazyFrame:
    sources = ["Serum or Plasma", "Blood"]
    LABS    = ["Urea nitrogen", "Creatinine", "Leukocytes", "Bilirubin", "Platelets", "INR"] # fmt: skip
    return labs.with_columns(
        extract_struct_value(lab, sources).alias(lab) for lab in LABS
    ).select(STAY_KEY, TIME_KEY, *LABS)


# endregion


################################################################################
################################################################################
# region organ scoring helpers


def _gcs_points_lods(gcs: pl.Expr) -> pl.Expr:
    return ScoringTable([          # Glasgow Coma Scale | Points
        (  14, 15, "both",    0),  # 14-15 .............. 0
        (   9, 13, "both",    1),  #  9-13                1
        (   6,  8, "both",    2),  #  6- 8                2
        (   4,  5, "both",    3),  #  4- 5                3
        (None,  4, "neither", 4),  #    <4                4
    ]).to_expr(gcs) # fmt: skip


def _pf_ratio_points_lods(pf: pl.Expr, ventilated: pl.Expr) -> pl.Expr:
    """
    PaO2/FiO2 ratio, mmHg

    Any      and NOT mechanically ventilated  0
    ≥250     and mechanically ventilated     +1
     150-249 and mechanically ventilated     +2
      50-149 and mechanically ventilated     +3
     <50     and mechanically ventilated     +4
    """
    return (
        pl.when(~ventilated)
        .then(0)
        .when(pf >= 250)
        .then(1)
        .when(pf.is_between(150, 250, closed="left"))
        .then(2)
        .when(pf.is_between(50, 150, closed="left"))
        .then(3)
        .when(pf < 50)
        .then(4)
        .otherwise(None)
    )


def _hr_points_lods(hr: pl.Expr) -> pl.Expr:
    return ScoringTable([            # Heart rate (bpm) | Points
        (None,   30, "neither", 3),  #  <30               3
        (  30,  140, "left",    0),  #   30-139 ......... 0
        ( 140,  160, "left",    1),  #  140-159           1
        ( 160, None, "left",    2),  # ≥160               2
    ]).to_expr(hr) # fmt: skip


def _sbp_points_lods(sbp: pl.Expr) -> pl.Expr:
    return ScoringTable([             # Systolic blood pressure (mmHg) | Points
        (None,   40, "neither", 3),   #  <40                             3
        (  40,   70, "left",    2),   #   40- 69                         2
        (  70,   90, "left",    1),   #   70- 89                         1
        (  90,  240, "left",    0),   #   90-239 ....................... 0
        ( 240,  270, "left",    1),   #  240-269                         1
        ( 270, None, "left",    2),   # ≥270                             2
    ]).to_expr(sbp) # fmt: skip


def _bun_points_lods(bun: pl.Expr) -> pl.Expr:
    urea = bun / 0.467
    return ScoringTable([            # Urea (mmol/L) | Points
        (None,    6, "neither", 0),  #  <6 ........... 0
        (   6,   10, "left",    1),  #   6- 9.9        1
        (  10,   20, "left",    2),  #  10-19.9        2
        (  20, None, "left",    3),  # ≥20             3
    ]).to_expr(urea) # fmt: skip


def _creatinine_points_lods(crea: pl.Expr) -> pl.Expr:
    return ScoringTable([            # Creatinine (mg/dL) | Points
        (None,  1.2, "neither", 0),  # <1.2 ............... 0
        ( 1.2,  1.6, "both",    1),  #  1.2-1.6             1
        ( 1.6, None, "neither", 2),  # >1.6                 2
    ]).to_expr(crea) # fmt: skip


def _uo_points_lods(uo_l_day: pl.Expr) -> pl.Expr:
    return ScoringTable([              # Urine output (L/day) | Points
        (None,   0.5,  "neither", 2),  # <0.5                   2
        ( 0.5,   0.75, "left",    1),  #  0.5 -0.74             1
        ( 0.75, 10,    "left",    0),  #  0.75-9.99 ........... 0
        (10,    None,  "left",    1),  # ≥10                    1
    ]).to_expr(uo_l_day) # fmt: skip


def _wbc_points_lods(wbc: pl.Expr) -> pl.Expr:
    return ScoringTable([            # Leukocytes (10^9/L) | Points
        (None,  1.0, "neither", 2),  #  <1.0                 2
        ( 1.0,  2.5, "left",    1),  #   1.0- 2.4            1
        ( 2.5,  50,  "left",    0),  #   2.5-49.9 .......... 0
        (50,   None, "left",    1),  # ≥50.                  1
    ]).to_expr(wbc) # fmt: skip


def _bilirubin_points_lods(bili: pl.Expr) -> pl.Expr:
    return ScoringTable([            # Bilirubin (mg/dL) | Points
        (None,  2.0, "neither", 0),  # <2.0 .............. 0
        ( 2.0,  4.0, "both",    1),  #  2.0-4.0            1
        ( 4.0, None, "neither", 2),  # >4.0                2
    ]).to_expr(bili) # fmt: skip


def _platelet_points_lods(plt: pl.Expr) -> pl.Expr:
    return ScoringTable([            # Platelets (10^3/µL) | Points
        (None,   50, "neither", 1),  # <50                   1
        (  50, None, "left",    0),  # ≥50 ................. 0
    ]).to_expr(plt) # fmt: skip


def _inr_points_lods(inr: pl.Expr) -> pl.Expr:
    return ScoringTable([            # INR (ratio) | Points
        (None, 0.25, "neither", 1),  # <0.25         1
        (0.25, None, "left",    0),  # ≥0.25 ....... 0
    ]).to_expr(inr) # fmt: skip


# endregion


################################################################################
################################################################################
# region LODS
def LODS(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_vitals: Optional[pl.LazyFrame] = None,
    timeseries_labs: Optional[pl.LazyFrame] = None,
    timeseries_resp: Optional[pl.LazyFrame] = None,
    timeseries_inout: Optional[pl.LazyFrame] = None,
    ventilation: Optional[pl.LazyFrame] = None,
    *,
    window_size: int = SECONDS_IN_1D,
    t_0: int = 0,
    t_1: Optional[int] = None,
    t_0_per_stay: Optional[pl.LazyFrame] = None,
    timeframe_name: Optional[str] = None,
) -> pl.LazyFrame:
    """
    Compute Logistic Organ Dysfunction System (LODS) score in long format.

    LODS is an organ dysfunction score computed per timeframe (default 24h)
    based on the worst values recorded within each window.

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information. Loaded automatically if None.
        timeseries_vitals : pl.LazyFrame, optional
            Timeseries vital signs data. Loaded automatically if None.
        timeseries_labs : pl.LazyFrame, optional
            Timeseries labs data. Loaded automatically if None.
        timeseries_resp : pl.LazyFrame, optional
            Timeseries respiratory data. Loaded automatically if None.
        timeseries_inout : pl.LazyFrame, optional
            Timeseries intake/output data. Loaded automatically if None.
        ventilation : pl.LazyFrame, optional
            Ventilation data with start/end intervals. Loaded automatically if None.
        window_size : int, optional
            Timeframe width in seconds (default: 86400 = 1 day). Window index is
            floor((time - T_0)/window_size).
        t_0 : int, optional
            Scalar reference time (seconds from admission). Defaults to 0 (admission).
            Ignored when t_0_per_stay is provided.
        t_1 : int, optional
            Optional upper time bound (seconds from admission) for filtering inputs.
        t_0_per_stay : pl.LazyFrame, optional
            Per-stay T_0 overrides with columns [Global ICU Stay ID, T_0].
        timeframe_name : str, optional
            Name for output timeframe column. Auto-generated if None.

    Returns
    -------
        pl.LazyFrame
            One row per (stay, timeframe) with columns:
            - Global ICU Stay ID
            - T_0
            - timeframe (or custom name)
            - LODS Score
            - Component points (gcs_points, hr_points, sbp_points, pf_ratio_points,
              bun_points, creatinine_points, uo_points, wbc_points, bilirubin_points,
              platelet_points, inr_points)
    """
    # region data loading
    if patient_information is None:
        patient_information = get_patient_information()
    if timeseries_vitals is None:
        timeseries_vitals = get_timeseries_vitals()
    if timeseries_labs is None:
        timeseries_labs = get_timeseries_labs()
    if timeseries_resp is None:
        timeseries_resp = get_timeseries_respiratory()
    if timeseries_inout is None:
        timeseries_inout = get_timeseries_intakeoutput()
    if ventilation is None:
        ventilation = get_ventilation()

    # Validate all required data is available
    required = {
        "patient_information": patient_information,
        "timeseries_vitals": timeseries_vitals,
        "timeseries_labs": timeseries_labs,
        "timeseries_resp": timeseries_resp,
        "timeseries_inout": timeseries_inout,
        "ventilation": ventilation,
    }
    _validate_required_data("LODS", required)

    vitals = _improve_vitals(timeseries_vitals.lazy())
    labs = _improve_labs(timeseries_labs.lazy())

    patient_information = patient_information.lazy()
    ALL_STAYS = patient_information.select(STAY_KEY)
    ALL_STAYS_T0 = _build_t0(ALL_STAYS, t_0_per_stay=t_0_per_stay, t_0=t_0)
    timeframe_name = _get_timeframe_name(
        timeframe_name, window_size, t_0, t_0_per_stay
    )

    # Ventilation status
    vent = (
        ventilation.filter(
            ~pl.col("Ventilation Type").is_in(["other", "supplemental oxygen"])
        )
        .pipe(
            intervention_per_timeframe,
            patient_information,
            start_col="Ventilation Start Relative to Admission (seconds)",
            end_col="Ventilation End Relative to Admission (seconds)",
            t_0=t_0,
            t_0_per_stay=t_0_per_stay,
            window_size=window_size,
        )
        .rename({"intervention": "ventilated"})
    )
    # endregion

    # region component scoring
    vitals_tf = (
        vitals.join(ALL_STAYS_T0, on=STAY_KEY, how="inner")
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window_size))
        .group_by(STAY_KEY, "timeframe")
        .agg(
            _gcs_points_lods(pl.col("Glasgow coma score total"))
            .max()
            .alias("gcs_points"),
            _hr_points_lods(pl.col("Heart rate")).max().alias("hr_points"),
            _sbp_points_lods(pl.col("Systolic arterial pressure"))
            .max()
            .alias("sbp_points"),
        )
    )

    labs_tf = (
        labs.join(ALL_STAYS_T0, on=STAY_KEY, how="inner")
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window_size))
        .group_by(STAY_KEY, "timeframe")
        .agg(
            _bun_points_lods(pl.col("Urea nitrogen")).max().alias("bun_points"),
            _creatinine_points_lods(pl.col("Creatinine"))
            .max()
            .alias("creatinine_points"),
            _wbc_points_lods(pl.col("Leukocytes")).max().alias("wbc_points"),
            _bilirubin_points_lods(pl.col("Bilirubin"))
            .max()
            .alias("bilirubin_points"),
            _platelet_points_lods(pl.col("Platelets"))
            .max()
            .alias("platelet_points"),
            _inr_points_lods(pl.col("INR")).max().alias("inr_points"),
        )
    )

    resp_tf = (
        PaO2_FiO2_RATIO(
            patient_information=patient_information,
            timeseries_resp=timeseries_resp,
            timeseries_labs=timeseries_labs,
            t_0=t_0,
            t_0_per_stay=t_0_per_stay,
        )
        .join(ALL_STAYS_T0, on=STAY_KEY, how="inner")
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window_size))
        .join(vent, [STAY_KEY, "timeframe"], how="left")
        .with_columns(pl.col("ventilated").fill_null(False))
        .group_by(STAY_KEY, "timeframe")
        .agg(
            _pf_ratio_points_lods(
                pl.col("PaO2/FiO2 Ratio"), pl.col("ventilated")
            )
            .max()
            .alias("pf_ratio_points")
        )
    )

    uo_tf = (
        URINE_OUTPUT(
            patient_information=patient_information,
            timeseries_inout=timeseries_inout,
            t_0=t_0,
            t_0_per_stay=t_0_per_stay,
            window_size=window_size,
        )
        .with_columns(
            uo_l_day=pl.col("uo_interval_ml")
            .truediv(1000)
            .mul(SECONDS_IN_1D / window_size)
        )
        .group_by(STAY_KEY, "timeframe")
        .agg(_uo_points_lods(pl.col("uo_l_day")).max().alias("uo_points"))
    )
    # endregion

    # region union of all (stay,timeframe)
    base = _build_base_timeframes(ALL_STAYS_T0, patient_information, window_size) # fmt: skip

    # region assemble
    out = base
    for part in [vitals_tf, labs_tf, resp_tf, uo_tf]:
        out = out.join(part, on=[STAY_KEY, "timeframe"], how="left")

    return (
        out.filter(
            _optional_time_bounds_filter("timeframe", window_size, t_0, t_1)
        )
        .with_columns(
            pl.sum_horizontal(pl.exclude(STAY_KEY, "T_0", "timeframe")).alias(
                "LODS Score"
            )
        )
        .select(
            STAY_KEY,
            "T_0",
            pl.col("timeframe").alias(timeframe_name),
            "LODS Score",
            pl.all().exclude(STAY_KEY, "T_0", timeframe_name, "LODS Score"),
        )
        .sort(STAY_KEY, timeframe_name)
    )


# endregion


################################################################################
################################################################################
# region mLODS
def mLODS(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_vitals: Optional[pl.LazyFrame] = None,
    timeseries_labs: Optional[pl.LazyFrame] = None,
    timeseries_resp: Optional[pl.LazyFrame] = None,
    ventilation: Optional[pl.LazyFrame] = None,
    *,
    window_size: int = SECONDS_IN_1D,
    t_0: int = 0,
    t_1: Optional[int] = None,
    t_0_per_stay: Optional[pl.LazyFrame] = None,
    timeframe_name: Optional[str] = None,
) -> pl.LazyFrame:
    """
    Compute modified Logistic Organ Dysfunction System (mLODS) score in long format.

    mLODS is a simplified version of LODS used in the Sepsis-3 definitions.
    It is computed per timeframe (default 24h) based on the worst values
    recorded within each window.

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information. Loaded automatically if None.
        timeseries_vitals : pl.LazyFrame, optional
            Timeseries vital signs data. Loaded automatically if None.
        timeseries_labs : pl.LazyFrame, optional
            Timeseries labs data. Loaded automatically if None.
        timeseries_resp : pl.LazyFrame, optional
            Timeseries respiratory data. Loaded automatically if None.
        ventilation : pl.LazyFrame, optional
            Ventilation data with start/end intervals. Loaded automatically if None.
        window_size : int, optional
            Timeframe width in seconds (default: 86400 = 1 day). Window index is
            floor((time - T_0)/window_size).
        t_0 : int, optional
            Scalar reference time (seconds from admission). Defaults to 0 (admission).
            Ignored when t_0_per_stay is provided.
        t_1 : int, optional
            Optional upper time bound (seconds from admission) for filtering inputs.
        t_0_per_stay : pl.LazyFrame, optional
            Per-stay T_0 overrides with columns [Global ICU Stay ID, T_0].
        timeframe_name : str, optional
            Name for output timeframe column. Auto-generated if None.

    Returns
    -------
        pl.LazyFrame
            One row per (stay, timeframe) with columns:
            - Global ICU Stay ID
            - T_0
            - timeframe (or custom name)
            - mLODS Score
            - Component points (gcs_points, hr_points, sbp_points, pf_ratio_points,
              creatinine_points, wbc_points, bilirubin_points, platelet_points)
    """
    # region data loading
    if patient_information is None:
        patient_information = get_patient_information()
    if timeseries_vitals is None:
        timeseries_vitals = get_timeseries_vitals()
    if timeseries_labs is None:
        timeseries_labs = get_timeseries_labs()
    if timeseries_resp is None:
        timeseries_resp = get_timeseries_respiratory()
    if ventilation is None:
        ventilation = get_ventilation()

    # Validate all required data is available
    required = {
        "patient_information": patient_information,
        "timeseries_vitals": timeseries_vitals,
        "timeseries_labs": timeseries_labs,
        "timeseries_resp": timeseries_resp,
        "ventilation": ventilation,
    }
    _validate_required_data("mLODS", required)

    vitals = _improve_vitals(timeseries_vitals.lazy())
    labs = _improve_labs(timeseries_labs.lazy())

    patient_information = patient_information.lazy()
    ALL_STAYS = patient_information.select(STAY_KEY)
    ALL_STAYS_T0 = _build_t0(ALL_STAYS, t_0_per_stay=t_0_per_stay, t_0=t_0)
    timeframe_name = _get_timeframe_name(
        timeframe_name, window_size, t_0, t_0_per_stay
    )

    # Ventilation status
    vent = (
        ventilation.filter(
            ~pl.col("Ventilation Type").is_in(["other", "supplemental oxygen"])
        )
        .pipe(
            intervention_per_timeframe,
            patient_information,
            start_col="Ventilation Start Relative to Admission (seconds)",
            end_col="Ventilation End Relative to Admission (seconds)",
            t_0=t_0,
            t_0_per_stay=t_0_per_stay,
            window_size=window_size,
        )
        .rename({"intervention": "ventilated"})
    )
    # endregion

    # region component scoring
    vitals_tf = (
        vitals.join(ALL_STAYS_T0, on=STAY_KEY, how="inner")
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window_size))
        .group_by(STAY_KEY, "timeframe")
        .agg(
            _gcs_points_lods(pl.col("Glasgow coma score total"))
            .max()
            .alias("gcs_points"),
            _hr_points_lods(pl.col("Heart rate")).max().alias("hr_points"),
            _sbp_points_lods(pl.col("Systolic arterial pressure"))
            .max()
            .alias("sbp_points"),
        )
    )

    labs_tf = (
        labs.join(ALL_STAYS_T0, on=STAY_KEY, how="inner")
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window_size))
        .group_by(STAY_KEY, "timeframe")
        .agg(
            _creatinine_points_lods(pl.col("Creatinine"))
            .max()
            .alias("creatinine_points"),
            _wbc_points_lods(pl.col("Leukocytes")).max().alias("wbc_points"),
            _bilirubin_points_lods(pl.col("Bilirubin"))
            .max()
            .alias("bilirubin_points"),
            _platelet_points_lods(pl.col("Platelets"))
            .max()
            .alias("platelet_points"),
        )
    )

    resp_tf = (
        PaO2_FiO2_RATIO(
            patient_information=patient_information,
            timeseries_resp=timeseries_resp,
            timeseries_labs=timeseries_labs,
            t_0=t_0,
            t_0_per_stay=t_0_per_stay,
        )
        .join(ALL_STAYS_T0, on=STAY_KEY, how="inner")
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window_size))
        .join(vent, [STAY_KEY, "timeframe"], how="left")
        .with_columns(pl.col("ventilated").fill_null(False))
        .group_by(STAY_KEY, "timeframe")
        .agg(
            _pf_ratio_points_lods(
                pl.col("PaO2/FiO2 Ratio"), pl.col("ventilated")
            )
            .max()
            .alias("pf_ratio_points")
        )
    )
    # endregion

    # region union of all (stay,timeframe)
    base = _build_base_timeframes(ALL_STAYS_T0, patient_information, window_size) # fmt: skip

    # region assemble
    out = base
    for part in [vitals_tf, labs_tf, resp_tf]:
        out = out.join(part, on=[STAY_KEY, "timeframe"], how="left")

    return (
        out.filter(
            _optional_time_bounds_filter("timeframe", window_size, t_0, t_1)
        )
        .with_columns(
            pl.sum_horizontal(pl.exclude(STAY_KEY, "T_0", "timeframe")).alias(
                "mLODS Score"
            )
        )
        .select(
            STAY_KEY,
            "T_0",
            pl.col("timeframe").alias(timeframe_name),
            "mLODS Score",
            pl.all().exclude(STAY_KEY, "T_0", timeframe_name, "mLODS Score"),
        )
        .sort(STAY_KEY, timeframe_name)
    )


# endregion


################################################################################
################################################################################
# region mortality
def LODS_mortality(lods_score: pl.Expr) -> pl.Expr:
    """
    Calculate predicted mortality rate from LODS score.

    Formula
    -------
        logit = -3.4043 + 0.4173 * LODS
        mortality = exp(logit) / (1 + exp(logit))
    """
    logit = -3.4043 + 0.4173 * lods_score
    return logit.exp() / (1 + logit.exp())


# endregion

__all__ = ["LODS", "mLODS", "LODS_mortality"]
