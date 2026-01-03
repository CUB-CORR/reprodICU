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
    _assign_timeframe,
    _build_t0,
    _get_timeframe_name,
    _optional_time_bounds_filter,
    get_patient_information,
    get_timeseries_intakeoutput,
    get_timeseries_vitals,
    get_ventilation,
    intervention_per_timeframe,
)

SECONDS_IN_1H = 60 * 60
SECONDS_IN_1D = 24 * SECONDS_IN_1H
SECONDS_IN_1W = 7 * SECONDS_IN_1D


################################################################################
################################################################################
# region helpers


def _improve_vitals(vitals: pl.LazyFrame) -> pl.LazyFrame:
    return vitals.with_columns(
        pl.coalesce(
            pl.col("Invasive mean arterial pressure"),
            pl.col("Non-invasive mean arterial pressure"),
            1 / 3 * pl.col("Invasive systolic arterial pressure")
            + 2 / 3 * pl.col("Invasive diastolic arterial pressure"),
            1 / 3 * pl.col("Non-invasive systolic arterial pressure")
            + 2 / 3 * pl.col("Non-invasive diastolic arterial pressure"),
        ).alias("Mean arterial pressure"),
    ).with_columns(
        pl.when(pl.col("Mean arterial pressure").is_finite())
        .then(pl.col("Mean arterial pressure"))
        .otherwise(None)
        .alias("Mean arterial pressure")
    )


################################################################################
################################################################################
# region pre-ICU LOS
def _pre_icu_los_points(pre_los: pl.Expr) -> pl.Expr:
    """
    Pre-ICU length of stay points (in days)

    <0.17 hours         5
     0.17-  4.94 hours  4
     4.94- 24    hours  3
    24.01-311.8  hours  2
    >311.8 hours        1
    """
    pre_los_hours = pre_los * 24
    return (
        pl.when(pre_los_hours < 0.17)
        .then(5)
        .when(pre_los_hours.is_between(0.17, 4.94, closed="right"))
        .then(4)
        .when(pre_los_hours.is_between(4.94, 24, closed="right"))
        .then(3)
        .when(pre_los_hours.is_between(24, 311.8, closed="right"))
        .then(2)
        .when(pre_los_hours > 311.8)
        .then(1)
        .otherwise(None)
    )


# region age
def _age_points(age: pl.Expr) -> pl.Expr:
    """
    Age in years

    <24 years     0
     24-53 years  3
     54-77 years  6
     78-89 years  9
    >89 years     7
    """
    return (
        pl.when(age < 24)
        .then(0)
        .when(age.is_between(24, 53, closed="left"))
        .then(3)
        .when(age.is_between(54, 77, closed="left"))
        .then(6)
        .when(age.is_between(78, 89, closed="right"))
        .then(9)
        .when(age > 89)
        .then(7)
        .otherwise(None)
    )


# region GCS
def _gcs_points(gcs: pl.Expr) -> pl.Expr:
    """
    Glasgow Coma Scale

     3- 7  10
     8-13   5
    14      3
    15      0
    """
    return (
        pl.when(gcs < 8)
        .then(10)
        .when(gcs.is_between(8, 13, closed="right"))
        .then(5)
        .when(gcs == 14)
        .then(3)
        .when(gcs == 15)
        .then(0)
        .otherwise(None)
    )


# region heart rate
def _hr_points(hr: pl.Expr) -> pl.Expr:
    """
    Heart rate in bpm

     <33 bpm      4
      33- 88 bpm  0
      89-106 bpm  1
     107-125 bpm  3
    >125 bpm      6
    """
    return (
        pl.when(hr < 33)
        .then(4)
        .when(hr.is_between(33, 88, closed="left"))
        .then(0)
        .when(hr.is_between(89, 106, closed="left"))
        .then(1)
        .when(hr.is_between(107, 125, closed="right"))
        .then(3)
        .when(hr > 125)
        .then(6)
        .otherwise(None)
    )


# region MAP
def _map_points(map: pl.Expr) -> pl.Expr:
    """
    Mean Arterial Pressure in mmHg

     <20.65 mmHg         4
      20.65- 50.99 mmHg  3
      51.00- 61.32 mmHg  2
      61.33-143.44 mmHg  0
    >143.44 mmHg         3
    """
    return (
        pl.when(map < 20.65)
        .then(4)
        .when(map.is_between(20.65, 51.00, closed="left"))
        .then(3)
        .when(map.is_between(51.00, 61.32, closed="left"))
        .then(2)
        .when(map.is_between(61.33, 143.44, closed="right"))
        .then(0)
        .when(map > 143.44)
        .then(3)
        .otherwise(None)
    )


# region respiratory rate
def _rr_points(rr: pl.Expr) -> pl.Expr:
    """
    Respiratory rate in bpm

    <6 bpm     10
     6-12 bpm   1
    13-22 bpm   0
    23-30 bpm   1
    31-44 bpm   6
    >44 bpm     9
    """
    return (
        pl.when(rr < 6)
        .then(10)
        .when(rr.is_between(6, 12, closed="right"))
        .then(1)
        .when(rr.is_between(13, 22, closed="right"))
        .then(0)
        .when(rr.is_between(23, 30, closed="right"))
        .then(1)
        .when(rr.is_between(31, 44, closed="right"))
        .then(6)
        .when(rr > 44)
        .then(9)
        .otherwise(None)
    )


# region temperature
def _temperature_points(temp: pl.Expr) -> pl.Expr:
    """
    Temperature in Celsius

    <33.22 °C        3
     33.22-35.93 °C  4
     35.94-36.39 °C  2
     36.40-36.88 °C  0
     36.89-39.88 °C  2
    >39.88 °C        6
    """
    return (
        pl.when(temp < 33.22)
        .then(3)
        .when(temp.is_between(33.22, 35.93, closed="right"))
        .then(4)
        .when(temp.is_between(35.94, 36.39, closed="right"))
        .then(2)
        .when(temp.is_between(36.40, 36.88, closed="right"))
        .then(0)
        .when(temp.is_between(36.89, 39.88, closed="right"))
        .then(2)
        .when(temp >= 39.88)
        .then(6)
        .otherwise(None)
    )


# region urine output
def _urine_output_points(uo_ml: pl.Expr) -> pl.Expr:
    """
    Urine output in mL/24h

    < 671 mL/24h           10
      671 to <1427 mL/24h   5
     1427 to <2544 mL/24h   1
     2544 to <6896 mL/24h   0
    >6896 mL/24h            4
    """
    return (
        pl.when(uo_ml < 671)
        .then(10)
        .when(uo_ml.is_between(671, 1427, closed="left"))
        .then(5)
        .when(uo_ml.is_between(1427, 2544, closed="left"))
        .then(1)
        .when(uo_ml.is_between(2544, 6896, closed="left"))
        .then(0)
        .when(uo_ml >= 6896)
        .then(4)
        .otherwise(None)
    )


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

    missing = [name for name, data in required.items() if data is None]
    if missing:
        raise ValueError(
            f"Cannot compute OASIS: Missing required datasets: {', '.join(missing)}. "
            f"Ensure they are configured in ~/.reprodICU/PATHS.yaml or provide them explicitly."
        )

    # Strict original column names
    STAY_KEY = "Global ICU Stay ID"
    TIME_KEY = "Time Relative to Admission (seconds)"
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
    base = (
        ALL_STAYS_T0.join(patient_information, on=STAY_KEY, how="left")
        .select(STAY_KEY, "T_0", "ICU Length of Stay (days)")
        .with_columns(
            pl.int_ranges(
                start=0 - pl.col("T_0").floordiv(window_size).sub(1),
                end=pl.col("ICU Length of Stay (days)")
                .mul(SECONDS_IN_1D)
                .sub("T_0")
                .truediv(window_size)
                .ceil()
                .add(1),
                step=1,
            )
            .cast(pl.List(float))
            .alias("timeframe")
        )
        .explode("timeframe")
        .unique()
        .select(STAY_KEY, "T_0", "timeframe")
    )

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
