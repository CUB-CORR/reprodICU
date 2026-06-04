"""
SAPS-II: compute Simplified Acute Physiology Score II in long format.

Output columns per row:
- Global ICU Stay ID
- T_0 (seconds from admission used as reference)
- SAPS-II Score (sum of component points)
- Component points (age, heart rate, systolic blood pressure, temperature, Glasgow coma score, urea nitrogen, WBC, sodium, potassium, bicarbonate, bilirubin, PaO2/FiO2 ratio, urine output, admission type & urgency, chronic disease)

Time is in seconds. Windows determined by floor((time - T_0)/window_size).
Worst-within-window aggregation is applied per component.

SOURCES
-------
- Le Gall JR, Lemeshow S, Saulnier F.
  A new Simplified Acute Physiology Score (SAPS II) based on a European/North American multicenter study.
  JAMA. 1993 Dec 22-29;270(24):2957-63. doi: 10.1001/jama.270.24.2957. Erratum in: JAMA 1994 May 4;271(17):1321. PMID: 8254858.
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
    get_diagnoses,
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
SECONDS_IN_1W = 7 * SECONDS_IN_1D


################################################################################
################################################################################
# region data helpers


def _improve_vitals(vitals: pl.LazyFrame) -> pl.LazyFrame:
    return (
        vitals.filter(
            pl.any_horizontal(
                pl.col(col).is_finite()
                for col in [
                    "Heart rate", "Invasive systolic arterial pressure", 
                    "Non-invasive systolic arterial pressure", "Temperature", 
                    "Glasgow coma score total",
                ]  # fmt: skip
            )
        )
        .with_columns(
            pl.coalesce(
                pl.col("Invasive systolic arterial pressure"),
                pl.col("Non-invasive systolic arterial pressure"),
            ).alias("Systolic blood pressure"),
        )
        .select(
            STAY_KEY,
            TIME_KEY,
            "Heart rate",
            "Systolic blood pressure",
            "Temperature",
            "Glasgow coma score total",
        )
    )


def _improve_labs(labs: pl.LazyFrame) -> pl.LazyFrame:
    sources = ["Serum or Plasma", "Blood", "Blood arterial"]
    LABS    = ["Sodium", "Potassium", "Urea nitrogen", "Bilirubin", "Leukocytes", "Bicarbonate"] # fmt: skip
    return (
        labs.with_columns(
            extract_struct_value(col, sources, exact_match=True).alias(col)
            for col in LABS
        )
        .filter(pl.any_horizontal(pl.col(col).is_finite() for col in LABS))
        .select(STAY_KEY, TIME_KEY, *LABS)
    )


# endregion


################################################################################
################################################################################
# region SAPS-II scoring helpers


def _age_points_saps2(age: pl.Expr) -> pl.Expr:
    return ScoringTable([            # Age (years) | Points
        (None,   40, "neither", 0),  # <40 .......... 0
        (  40,   60, "left",    7),  #  40-59         7
        (  60,   70, "left",   12),  #  60-69        12
        (  70,   75, "left",   15),  #  70-74        15
        (  75,   80, "left",   16),  #  75-79        16
        (  80, None, "left",   18),  # ≥80           18
    ]).to_expr(age) # fmt: skip


def _hr_points_saps2(hr: pl.Expr) -> pl.Expr:
    return ScoringTable([             # Heart rate (bpm) | Points
        (None,   40, "neither", 11),  #  <40               11
        (  40,   70, "left",     2),  #   40- 69            2
        (  70,  120, "left",     0),  #   70-119 .......... 0
        ( 120,  160, "left",     4),  #  120-159            4
        ( 160, None, "left",     7),  # ≥160                7
    ]).to_expr(hr) # fmt: skip


def _sbp_points_saps2(sbp: pl.Expr) -> pl.Expr:
    return ScoringTable([             # Systolic blood pressure (mmHg) | Points
        (None,   70, "neither", 13),  #  <70                             13
        (  70,  100, "left",     5),  #   70- 99                          5
        ( 100,  200, "left",     0),  #  100-199 ........................ 0
        ( 200, None, "left",     2),  # ≥200                              2
    ]).to_expr(sbp) # fmt: skip


def _temp_points_saps2(temp: pl.Expr) -> pl.Expr:
    return ScoringTable([            # Temperature (°C) | Points
        (None, 39.0, "neither", 0),  # <39.0 ............ 0
        (39.0, None, "left",    3),  # ≥39.0              3
    ]).to_expr(temp) # fmt: skip


def _pao2fio2_points_saps2(paf: pl.Expr) -> pl.Expr:
    return ScoringTable([             # PaO2/FiO2 ratio | Points
        (None,  100, "neither", 11),  # <100              11
        ( 100,  200, "left",     9),  #  100-199           9
        ( 200, None, "left",     6),  # ≥200               6
    ]).to_expr(paf) # fmt: skip


def _uo_points_saps2(uo_l_day: pl.Expr) -> pl.Expr:
    return ScoringTable([             # Urine output (L/day) | Points
        (None,  0.5, "neither", 11),  # <0.5                   11
        ( 0.5,  1.0, "left",     4),  #  0.5-0.99               4
        ( 1.0, None, "left",     0),  # ≥1.0 .................. 0
    ]).to_expr(uo_l_day) # fmt: skip


def _bun_points_saps2(bun: pl.Expr) -> pl.Expr:
    return ScoringTable([             # BUN (mg/dL) | Points
        (None,   28, "neither", 0),   # <28 .......... 0
        (  28,   84, "left",    6),   #  28-83         6
        (  84, None, "left",   10),   # ≥84           10
    ]).to_expr(bun) # fmt: skip


def _wbc_points_saps2(wbc: pl.Expr) -> pl.Expr:
    return ScoringTable([             # Leukocytes (10^9/L) | Points
        (None,  1.0, "neither", 12),  #  <1.0                 12
        ( 1.0, 20.0, "left",     0),  #   1.0-19.9 ........... 0
        (20.0, None, "left",     3),  # ≥20.0                  3
    ]).to_expr(wbc) # fmt: skip


def _sodium_points_saps2(na: pl.Expr) -> pl.Expr:
    return ScoringTable([            # Sodium (mmol/L) | Points
        (None,  125, "neither", 5),  # <125              5
        ( 125,  145, "left",    0),  #  125-144 ........ 0
        ( 145, None, "left",    1),  # ≥145              1
    ]).to_expr(na) # fmt: skip


def _potassium_points_saps2(k: pl.Expr) -> pl.Expr:
    return ScoringTable([            # Potassium (mmol/L) | Points
        (None,  3.0, "neither", 3),  # <3.0                 3
        ( 3.0,  5.0, "left",    0),  #  3.0-4.9 ........... 0
        ( 5.0, None, "left",    3),  # ≥5.0                 3
    ]).to_expr(k) # fmt: skip


def _bicarbonate_points_saps2(hco3: pl.Expr) -> pl.Expr:
    return ScoringTable([            # Bicarbonate (mmol/L) | Points
        (None,   15, "neither", 6),  # <15                    6
        (  15,   20, "left",    3),  #  15-19                 3
        (  20, None, "left",    0),  # ≥20 .................. 0
    ]).to_expr(hco3) # fmt: skip


def _bilirubin_points_saps2(bili: pl.Expr) -> pl.Expr:
    return ScoringTable([            # Bilirubin (mg/dL) | Points
        (None,  4.0, "neither", 0),  # <4.0 .............. 0
        ( 4.0,  6.0, "left",    4),  #  4.0-5.9            4
        ( 6.0, None, "left",    9),  # ≥6.0                9
    ]).to_expr(bili) # fmt: skip


def _gcs_points_saps2(gcs: pl.Expr) -> pl.Expr:
    return ScoringTable([           # Glasgow Coma Scale | Points
        (None,  6, "neither", 26),  # <6                   26
        (   6,  9, "left",    13),  #  6- 8                13
        (   9, 11, "left",     7),  #  9-10                 7
        (  11, 14, "left",     5),  # 11-13                 5
        (  14, 16, "left",     0),  # 14-15 ............... 0
    ]).to_expr(gcs) # fmt: skip


# endregion


def _chronic_disease_points_saps2(diagnoses: pl.LazyFrame) -> pl.LazyFrame:
    """
    Chronic diseases:
    - AIDS: 17 points (HIV positive with complications)
    - Hematologic malignancy: 10 points (Lymphoma, Acute Leukemia, Multiple Myeloma)
    - Metastatic cancer: 9 points (Proven by surgery, CT scan, or other method)

    Args:
        diagnoses (pl.LazyFrame): Standardized diagnoses table.

    Returns:
        pl.LazyFrame: Contains columns:
            - {STAY_KEY}: Global ICU Stay ID.
            - chronic_pts: Points for chronic disease.
    """
    # AIDS: B20–B24
    # Hematologic malignancy: C81-C85, C88, C90-C96
    # Metastatic cancer: C77-C79
    ICD_COL = "Diagnosis ICD-10 Code"

    return (
        diagnoses.with_columns(
            aids=pl.col(ICD_COL).str.contains(r"^B2[0-4]"),
            heme=pl.col(ICD_COL).str.contains(r"^C8[1-58]|C9[0-6]"),
            metc=pl.col(ICD_COL).str.contains(r"^C7[7-9]"),
        )
        .group_by(STAY_KEY)
        .agg(
            pl.max_horizontal(
                pl.when(pl.col("aids").any()).then(17).otherwise(0),
                pl.when(pl.col("heme").any()).then(10).otherwise(0),
                pl.when(pl.col("metc").any()).then(9).otherwise(0),
            ).alias("chronic_pts")
        )
    )


def _admission_type_points_saps2(
    adm_type: pl.Expr, adm_urgency: pl.Expr
) -> pl.Expr:
    """
    Type of admission:
    - Scheduled surgical: 0
    - Medical: 6
    - Unscheduled surgical: 8
    """
    return (
        pl.when(adm_type == "Medical")
        .then(6)
        .when(adm_type == "Surgical", adm_urgency != "Elective")
        .then(8)
        .otherwise(0)
    )


################################################################################
################################################################################
def SAPS2(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_vitals: Optional[pl.LazyFrame] = None,
    timeseries_labs: Optional[pl.LazyFrame] = None,
    timeseries_resp: Optional[pl.LazyFrame] = None,
    timeseries_inout: Optional[pl.LazyFrame] = None,
    diagnoses: Optional[pl.LazyFrame] = None,
    ventilation: Optional[pl.LazyFrame] = None,
    *,
    window_size: int = SECONDS_IN_1D,
    t_0: int = 0,
    t_1: Optional[int] = None,
    t_0_per_stay: Optional[pl.LazyFrame] = None,
    timeframe_name: Optional[str] = None,
) -> pl.LazyFrame:
    """
    Compute Simplified Acute Physiology Score II (SAPS II) with automatic dataset loading.

    All data parameters are optional and will be automatically loaded from the
    package datasets if not provided. This makes it convenient for quick analysis
    while maintaining flexibility for custom data.

    Returns
    -------
        pl.LazyFrame
            SAPS-II scores with all organ subscore components
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
    if diagnoses is None:
        diagnoses = get_diagnoses()
    if ventilation is None:
        ventilation = get_ventilation()

    # Validate
    required = {
        "patient_information": patient_information,
        "timeseries_vitals": timeseries_vitals,
        "timeseries_labs": timeseries_labs,
        "timeseries_resp": timeseries_resp,
        "timeseries_inout": timeseries_inout,
        "diagnoses": diagnoses,
        "ventilation": ventilation,
    }
    _validate_required_data("SAPS-II", required)

    vitals = _improve_vitals(timeseries_vitals)
    labs = _improve_labs(timeseries_labs)

    patient_information = patient_information.lazy()
    ALL_STAYS = patient_information.select(STAY_KEY)
    ALL_STAYS_T0 = _build_t0(ALL_STAYS, t_0_per_stay=t_0_per_stay, t_0=t_0)
    timeframe_name = _get_timeframe_name(
        timeframe_name, window_size, t_0, t_0_per_stay
    )

    # region component scoring
    # 1. Physiological Points
    vitals_tf = (
        vitals.join(ALL_STAYS_T0, on=STAY_KEY, how="inner")
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window_size))
        .group_by(STAY_KEY, "timeframe")
        .agg(
            _hr_points_saps2(pl.col("Heart rate")).max().alias("hr_pts"),
            _sbp_points_saps2(pl.col("Systolic blood pressure"))
            .max()
            .alias("sbp_pts"),
            _temp_points_saps2(pl.col("Temperature")).max().alias("temp_pts"),
            _gcs_points_saps2(pl.col("Glasgow coma score total"))
            .max()
            .alias("gcs_pts"),
        )
    )

    labs_tf = (
        labs.join(ALL_STAYS_T0, on=STAY_KEY, how="inner")
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window_size))
        .group_by(STAY_KEY, "timeframe")
        .agg(
            _bun_points_saps2(pl.col("Urea nitrogen")).max().alias("bun_pts"),
            _wbc_points_saps2(pl.col("Leukocytes")).max().alias("wbc_pts"),
            _sodium_points_saps2(pl.col("Sodium")).max().alias("na_pts"),
            _potassium_points_saps2(pl.col("Potassium")).max().alias("k_pts"),
            _bicarbonate_points_saps2(pl.col("Bicarbonate"))
            .max()
            .alias("hco3_pts"),
            _bilirubin_points_saps2(pl.col("Bilirubin"))
            .max()
            .alias("bili_pts"),
        )
    )

    # 2. PaO2/FiO2 (Ventilated/CPAP only)
    vent_tf = (
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

    pf_ratio = (
        PaO2_FiO2_RATIO(
            timeseries_resp=timeseries_resp,
            timeseries_labs=timeseries_labs,
            t_0=t_0,
            t_0_per_stay=t_0_per_stay,
        )
        .join(ALL_STAYS_T0, on=STAY_KEY, how="inner")
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window_size))
        .join(vent_tf, on=[STAY_KEY, "timeframe"], how="left")
        .filter(pl.col("ventilated"))
        .group_by(STAY_KEY, "timeframe")
        .agg(
            _pao2fio2_points_saps2(pl.col("PaO2/FiO2 Ratio"))
            .max()
            .alias("pafi_pts")
        )
    )

    # 3. Urine Output
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
        .agg(_uo_points_saps2(pl.col("uo_l_day")).max().alias("uo_pts"))
    )

    # 4. Static Points (Age, Adm Type, Chronic)
    static_pts = (
        patient_information.select(
            STAY_KEY,
            _age_points_saps2(pl.col("Admission Age (years)")).alias("age_pts"),
            _admission_type_points_saps2(
                pl.col("Admission Type"), pl.col("Admission Urgency")
            ).alias("adm_pts"),
        )
        .join(_chronic_disease_points_saps2(diagnoses), on=STAY_KEY, how="left")
        .with_columns(pl.col("chronic_pts").fill_null(0))
    )

    # region union of all (stay,timeframe)
    base = _build_base_timeframes(ALL_STAYS_T0, patient_information, window_size) # fmt: skip

    # region assemble
    out = base.join(static_pts, on=STAY_KEY, how="left")
    for part in [vitals_tf, labs_tf, pf_ratio, uo_tf]:
        out = out.join(part, on=[STAY_KEY, "timeframe"], how="left")

    return (
        out.filter(
            _optional_time_bounds_filter("timeframe", window_size, t_0, t_1)
        )
        .with_columns(
            pl.sum_horizontal(pl.exclude(STAY_KEY, "T_0", "timeframe")).alias(
                "SAPS-II Score"
            )
        )
        .select(
            STAY_KEY,
            "T_0",
            pl.col("timeframe").alias(timeframe_name),
            "SAPS-II Score",
            pl.all().exclude(STAY_KEY, "T_0", timeframe_name, "SAPS-II Score"),
        )
    )


__all__ = ["SAPS2"]
