"""
OASIS: compute Oxford Acute Severity of Illness Score in long format directly from raw inputs.

Output columns per row:
- Global ICU Stay ID
- T_0 (seconds from admission used as reference)
- timeframe (0-indexed integer window)
- OASIS Score (sum of component points)
- Component scores (pre-ICU LOS, age, GCS, heart rate, MAP, respiratory rate, temperature, ventilation, urine output, elective surgery)

Time is in seconds. Windows determined by floor((time - T_0)/window_size).
Worst-within-window aggregation is applied per component.

SOURCES
-------
- Johnson AE, Kramer AA, Clifford GD.
  A new severity of illness scale using a subset of Acute Physiology And Chronic Health Evaluation data elements shows comparable predictive accuracy.
  Crit Care Med. 2013 Jul;41(7):1711-8.
  doi: 10.1097/CCM.0b013e31828a24fe. PMID: 23660729.
"""

from typing import Optional

import numpy as np
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
    get_patient_information,
    get_timeseries_intakeoutput,
    get_timeseries_vitals,
    get_ventilation,
    intervention_per_timeframe,
)
from ..core import BLOOD_PRESSURES

STAY_KEY = "Global ICU Stay ID"
TIME_KEY = "Time Relative to Admission (seconds)"

SECONDS_IN_1H = 60 * 60
SECONDS_IN_1D = 24 * SECONDS_IN_1H
SECONDS_IN_1W = 7 * SECONDS_IN_1D


################################################################################
################################################################################
# region helpers
def _improve_vitals(vitals: pl.LazyFrame) -> pl.LazyFrame:
    return (
        BLOOD_PRESSURES(timeseries_vitals=vitals)
        .rename({"MAP": "Mean arterial pressure"})
        .with_columns(
        pl.when(pl.col("Mean arterial pressure").is_finite())
        .then(pl.col("Mean arterial pressure"))
        .otherwise(None)
        .alias("Mean arterial pressure")
        )
    )


################################################################################
################################################################################
# region pre-ICU LOS
def _pre_icu_los_points(pre_los: pl.Expr) -> pl.Expr:
    pre_los_hours = pre_los * 24
    return ScoringTable([                # Pre-ICU length of stay (hours) | Points
        (  None,   0.17, "neither", 5),  # <0.17             5
        (  0.17,   4.94, "right",   4),  #  0.17-  4.94      4
        (  4.94,  24   , "right",   3),  #  4.94- 24         3
        ( 24   , 311.8 , "right",   2),  # 24.  -311.8       2
        (311.8 ,   None, "neither", 1),  #      >311.8       1
    ]).to_expr(pre_los_hours) # fmt: skip


# region age
def _age_points(age: pl.Expr) -> pl.Expr:
    return ScoringTable([            # Age (years) | Points
        (None,   24, "neither", 0),  # <24 ......... 0
        (  24,   53, "left",    3),  #  24-53        3
        (  54,   77, "left",    6),  #  54-77        6
        (  78,   89, "right",   9),  #  78-89        9
        (  89, None, "neither", 7),  # >89           7
    ]).to_expr(age) # fmt: skip


# region GCS
def _gcs_points(gcs: pl.Expr) -> pl.Expr:
    return ScoringTable([           # Glasgow Coma Scale | Points
        (None,  8, "neither", 10),  # <8                   10
        (   8, 13, "right",    5),  #  8-13                 5 
        (  14, 14, "both",     3),  #    14                 3
        (  15, 15, "both",     0),  #    15 ............... 0
    ]).to_expr(gcs) # fmt: skip


# region heart rate
def _hr_points(hr: pl.Expr) -> pl.Expr:
    return ScoringTable([            # Heart rate (bpm) | Points
        (None,   33, "neither", 4),  #  <33               4
        (  33,   88, "left",    0),  #   33- 88 ......... 0
        (  89,  106, "left",    1),  #   89-106           1
        ( 107,  125, "right",   3),  #  107-125           3
        ( 125, None, "neither", 6),  # >125               6
    ]).to_expr(hr) # fmt: skip


# region MAP
def _map_points(map: pl.Expr) -> pl.Expr:
    return ScoringTable([                # Mean Arterial Pressure (mmHg) | Points
        (  None,  20.65, "neither", 4),  #  <20.65                         4
        ( 20.65,  51.00, "left",    3),  #   20.65- 50.99                  3
        ( 51.00,  61.32, "left",    2),  #   51.00- 61.32                  2
        ( 61.33, 143.44, "right",   0),  #   61.33-143.44 ................ 0
        (143.44,   None, "neither", 3),  # >143.44                         3
    ]).to_expr(map) # fmt: skip


# region respiratory rate
def _rr_points(rr: pl.Expr) -> pl.Expr:
    return ScoringTable([             # Respiratory rate (bpm) | Points
        (None,    6, "neither", 10),  #  <6                      10
        (   6,   12, "right",    1),  #   6-12                    1
        (  13,   22, "right",    0),  #  13-22 .................. 0
        (  23,   30, "right",    1),  #  23-30                    1
        (  31,   44, "right",    6),  #  31-44                    6
        (  44, None, "neither",  9),  # >44                       9
    ]).to_expr(rr) # fmt: skip


# region temperature
def _temperature_points(temp: pl.Expr) -> pl.Expr:
    return ScoringTable([              # Temperature (°C) | Points
        ( None, 33.22, "neither", 3),  # <33.22.            3
        (33.22, 35.93, "right",   4),  #  33.22-35.93       4
        (35.94, 36.39, "right",   2),  #  35.94-36.39       2
        (36.40, 36.88, "right",   0),  #  36.40-36.88 ..... 0
        (36.89, 39.88, "right",   2),  #  36.89-39.88       2
        (39.88,  None, "neither", 6),  # ≥39.88             6
    ]).to_expr(temp) # fmt: skip


# region urine output
def _urine_output_points(uo_ml: pl.Expr) -> pl.Expr:
    return ScoringTable([             # Urine output (mL/24h) | Points
        (None,  671, "neither", 10),  #  <671                   10
        ( 671, 1427, "left",     5),  #   671-1426               5
        (1427, 2544, "left",     1),  #  1427-2543               1
        (2544, 6896, "left",     0),  #  2544-6895 ............. 0
        (6896, None, "neither",  4),  # ≥6896                    4
    ]).to_expr(uo_ml) # fmt: skip


# region ventilation
def _ventilation_points(ventilated: pl.Expr) -> pl.Expr:
    """
    Mechanical ventilation

    not ventilated  0
    ventilated      9
    """
    return pl.when(ventilated).then(9).otherwise(0)


# region elective surgery
def _elective_surgery_points(elective: pl.Expr) -> pl.Expr:
    """
    Elective surgery admission

    non-elective surgery  6
    elective surgery      0
    """
    return pl.when(~elective).then(6).otherwise(0)


################################################################################
################################################################################
# region OASIS
def OASIS(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_vitals: Optional[pl.LazyFrame] = None,
    timeseries_inout: Optional[pl.LazyFrame] = None,
    ventilation: Optional[pl.LazyFrame] = None,
    *,
    t_0: Optional[int] = 0,
    t_0_per_stay: Optional[pl.LazyFrame] = None,
    t_1: Optional[int] = None,
    window_size: int = SECONDS_IN_1D,
    timeframe_name: str = None,
) -> pl.LazyFrame:
    """
    Compute OASIS score in long format from raw inputs.

    OASIS is an organ dysfunction score computed per day based on vital signs,
    laboratory values, and administrative data recorded in the first 24 hours
    after ICU admission.

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information including age, admission urgency/type,
            and pre-ICU length of stay. Loaded automatically if None.
        timeseries_vitals : pl.LazyFrame, optional
            Timeseries vital signs data. Loaded automatically if None.
        timeseries_inout : pl.LazyFrame, optional
            Timeseries intake/output data for urine output. Loaded automatically if None.
        ventilation : pl.LazyFrame, optional
            Ventilation data with start/end intervals. Loaded automatically if None.
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
        timeframe_name : str, optional
            Name for output timeframe column. Auto-generated if None.

    Returns
    -------
        pl.LazyFrame
            One row per (stay, timeframe) with columns:
            - Global ICU Stay ID
            - T_0
            - timeframe (or custom name)
            - OASIS Score
            - Individual component scores
    """
    # Load defaults if not provided
    if patient_information is None:
        patient_information = get_patient_information()
    if timeseries_vitals is None:
        timeseries_vitals = get_timeseries_vitals()
    if timeseries_inout is None:
        timeseries_inout = get_timeseries_intakeoutput()
    if ventilation is None:
        ventilation = get_ventilation()

    # Validate all required data is available
    required = {
        "patient_information": patient_information,
        "timeseries_vitals": timeseries_vitals,
        "timeseries_inout": timeseries_inout,
        "ventilation": ventilation,
    }
    _validate_required_data("OASIS", required)

    # Strict original column names
    age_col = "Admission Age (years)"
    pre_los_col = "Pre-ICU Length of Stay (days)"
    urgency_col = "Admission Urgency"
    admission_type_col = "Admission Type"

    gcs_col = "Glasgow coma score total"
    hr_col = "Heart rate"
    map_col = "Mean arterial pressure"
    rr_col = "Respiratory rate"
    temp_col = "Temperature"

    # Base frames
    patient_information = patient_information.lazy()
    timeseries_vitals = timeseries_vitals.lazy()
    timeseries_inout = timeseries_inout.lazy()
    ventilation = ventilation.lazy()

    ALL_STAYS = patient_information.select(STAY_KEY)
    ALL_STAYS_T0 = _build_t0(ALL_STAYS, t_0_per_stay=t_0_per_stay, t_0=t_0)
    timeframe_name = _get_timeframe_name(
        timeframe_name, window_size, t_0, t_0_per_stay
    )

    # region patient-level scores (pre-ICU LOS, age, elective surgery)
    patient_scores = (
        patient_information.join(ALL_STAYS_T0, on=STAY_KEY, how="inner")
        .select(STAY_KEY, age_col, pre_los_col, urgency_col, admission_type_col)
        .with_columns(
            pl.when(
                pl.col(urgency_col) == "Elective",
                pl.col(admission_type_col) == "Surgical",
            )
            .then(True)
            .otherwise(False)
            .alias("elective_surgery")
        )
        # fmt: off
        .with_columns(
            _age_points(pl.col(age_col)).alias("age_points"),
            _pre_icu_los_points(pl.col(pre_los_col)).alias("pre_icu_los_points"),
            _elective_surgery_points(pl.col("elective_surgery")).alias("elective_surgery_points"),
        )
        # fmt: on
        .select(
            STAY_KEY,
            "age_points",
            "pre_icu_los_points",
            "elective_surgery_points",
        )
    )

    # region vitals (GCS, HR, MAP, RR, Temperature)
    vitals = _improve_vitals(timeseries_vitals)
    vitals_scores = (
        vitals.select(
            STAY_KEY,
            TIME_KEY,
            gcs_col,
            hr_col,
            map_col,
            rr_col,
            temp_col,
        )
        .join(ALL_STAYS_T0, on=STAY_KEY, how="inner")
        .filter(pl.col(TIME_KEY) >= pl.col("T_0").sub(SECONDS_IN_1W))
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window_size))
        .with_columns(
            _gcs_points(pl.col(gcs_col)).alias("gcs_points"),
            _hr_points(pl.col(hr_col)).alias("hr_points"),
            _map_points(pl.col(map_col)).alias("map_points"),
            _rr_points(pl.col(rr_col)).alias("rr_points"),
            _temperature_points(pl.col(temp_col)).alias("temperature_points"),
        )
        .group_by(STAY_KEY, "timeframe")
        .agg(
            pl.max("gcs_points"),
            pl.max("hr_points"),
            pl.max("map_points"),
            pl.max("rr_points"),
            pl.max("temperature_points"),
        )
    )

    # region ventilation (per day)
    ventilation_scores = (
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
        .with_columns(
            _ventilation_points(pl.col("ventilated") > 0).alias(
                "ventilation_points"
            )
        )
        .select(STAY_KEY, "timeframe", "ventilation_points")
    )

    # region urine output (daily totals)
    urine_scores = (
        URINE_OUTPUT(
            patient_information=patient_information,
            timeseries_inout=timeseries_inout,
            t_0=t_0,
            t_0_per_stay=t_0_per_stay,
            window_size=window_size,
        )
        .with_columns(
            _urine_output_points(
                pl.col("uo_interval_ml") * (SECONDS_IN_1D / window_size)
            ).alias("urine_output_points")
        )
        .select(STAY_KEY, "timeframe", "urine_output_points")
    )

    # region union of all (stay, timeframe)
    base = _build_base_timeframes(ALL_STAYS_T0, patient_information, window_size) # fmt: skip

    # region assemble
    out = base
    for part in [vitals_scores, ventilation_scores, urine_scores]:
        out = out.join(part, on=[STAY_KEY, "timeframe"], how="left")

    # Join patient-level scores (constant across timeframes)

    return (
        out.join(patient_scores, on=STAY_KEY, how="left")
        .filter(
            _optional_time_bounds_filter("timeframe", window_size, t_0, t_1)
        )
        .with_columns(
            pl.sum_horizontal(
                "age_points",
                "pre_icu_los_points",
                "elective_surgery_points",
                "gcs_points",
                "hr_points",
                "map_points",
                "rr_points",
                "temperature_points",
                "ventilation_points",
                "urine_output_points",
                ignore_nulls=True,
            ).alias("OASIS Score")
        )
        .select(
            STAY_KEY,
            "T_0",
            pl.col("timeframe").alias(timeframe_name),
            "OASIS Score",
            "age_points",
            "pre_icu_los_points",
            "elective_surgery_points",
            "gcs_points",
            "hr_points",
            "map_points",
            "rr_points",
            "temperature_points",
            "ventilation_points",
            "urine_output_points",
        )
        .sort(STAY_KEY, timeframe_name)
    )


################################################################################
################################################################################
# region mortality
def OASIS_icu_mortality(oasis_score: float) -> float:
    """
    Calculate predicted ICU mortality rate from OASIS score.

    Values found in Table 4.10 in
      Johnson, A. (2014).
      Mortality prediction and acuity assessment in critical care [PhD thesis].
      University of Oxford.
    """
    x = -7.4225 + 0.1434 * oasis_score
    return np.exp(x) / (1 + np.exp(x))


def OASIS_hospital_mortality(oasis_score: float) -> float:
    """
    Calculate predicted hospital mortality rate from OASIS score.

    Values found in Table 4.10 in
      Johnson, A. (2014).
      Mortality prediction and acuity assessment in critical care [PhD thesis].
      University of Oxford.
    """
    x = -6.1746 + 0.1275 * oasis_score
    return np.exp(x) / (1 + np.exp(x))


__all__ = ["OASIS", "OASIS_icu_mortality", "OASIS_hospital_mortality"]
