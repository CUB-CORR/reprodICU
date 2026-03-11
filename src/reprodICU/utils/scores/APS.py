"""
APS: compute Acute Physiology Score (from APACHE II & III) in long format directly from raw inputs.

Output columns per row:
- Global ICU Stay ID
- T_0 (seconds from admission used as reference)
- APS Score or APS3 Score (sum of component points)
- Component points (temperature, MAP, heart rate, respiratory rate, GCS, sodium, potassium, BUN, creatinine, glucose, albumin, bilirubin, hematocrit, WBC, pH, pCO2, oxygenation, urine output)

Time is in seconds. Windows determined by floor((time - T_0)/window_size).
Worst-within-window aggregation is applied per component.

SOURCES
-------
- Knaus WA, Draper EA, Wagner DP, Zimmerman JE.
  APACHE II: a severity of disease classification system.
  Crit Care Med. 1985 Oct;13(10):818-29. PMID: 3928249.
- Knaus WA, Wagner DP, Draper EA, Zimmerman JE, Bergner M, Bastos PG, Sirio CA, Murphy DJ, Lotring T, Damiano A, et al.
  The APACHE III prognostic system. Risk prediction of hospital mortality for critically ill hospitalized adults.
  Chest. 1991 Dec;100(6):1619-36.
  doi: 10.1378/chest.100.6.1619. PMID: 1959406.
"""

from typing import Optional

import polars as pl

from ..clinical.renal.URINE_OUTPUT import URINE_OUTPUT
from ..clinical.respiratory.ALVEOLAR_ARTERIAL_GRADIENT import Aa_GRADIENT
from ..common import (
    _assign_timeframe,
    _build_t0,
    _get_timeframe_name,
    _optional_time_bounds_filter,
    get_patient_information,
    get_timeseries_intakeoutput,
    get_timeseries_labs,
    get_timeseries_respiratory,
    get_timeseries_vitals,
    get_ventilation,
    intervention_per_timeframe,
)

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
        vitals.with_columns(
            pl.coalesce(
                pl.col("Invasive mean arterial pressure"),
                pl.col("Non-invasive mean arterial pressure"),
                1 / 3 * pl.col("Invasive systolic arterial pressure")
                + 2 / 3 * pl.col("Invasive diastolic arterial pressure"),
                1 / 3 * pl.col("Non-invasive systolic arterial pressure")
                + 2 / 3 * pl.col("Non-invasive diastolic arterial pressure"),
            ).alias("Mean arterial pressure"),
        )
        .filter(
            pl.any_horizontal(
                pl.col(col).is_finite()
                for col in [
                    "Heart rate", "Mean arterial pressure", "Respiratory rate",
                    "Temperature", "Glasgow coma score total",
                ] # fmt: skip
            )
        )
        .select(
            STAY_KEY,
            TIME_KEY,
            "Heart rate",
            "Mean arterial pressure",
            "Respiratory rate",
            "Temperature",
            "Glasgow coma score total",
            "Glasgow coma score eye opening",
            "Glasgow coma score motor",
            "Glasgow coma score verbal",
        )
    )


def _improve_labs(labs: pl.LazyFrame) -> pl.LazyFrame:
    sources = ["Serum or Plasma", "Blood", "Blood arterial", None]
    LABS = [
        "Oxygen", "Carbon dioxide", "pH", "Sodium", "Potassium", "Creatinine",
        "Erythrocyte/Blood", "Leukocytes", "Urea nitrogen",  "Albumin",
        "Bilirubin", "Glucose",
    ] # fmt: skip
    return (
        labs.with_columns(
            pl.when(pl.col(col).struct.field("system").is_in(sources))
            .then(pl.col(col).struct.field("value"))
            .alias(col)
            for col in LABS
        )
        .filter(pl.any_horizontal(pl.col(col).is_finite() for col in LABS))
        .select(STAY_KEY, TIME_KEY, *LABS)
    )


def _improve_resp(resp: pl.LazyFrame) -> pl.LazyFrame:
    return (
        resp.with_columns(
            pl.max_horizontal(
                "Oxygen/Total gas setting [Volume Fraction] Ventilator",
                "Oxygen/Gas total [Pure volume fraction] Inhaled gas",
            ).alias("FiO2")
        )
        .with_columns(
            pl.when(pl.col("FiO2").is_between(0, 1))
            .then(pl.col("FiO2") * 100)
            .when(pl.col("FiO2").is_between(1, 100))
            .then(pl.col("FiO2"))
            .otherwise(None)
            .alias("FiO2")
        )
        .filter(pl.col("FiO2").is_finite())
        .select(STAY_KEY, TIME_KEY, "FiO2")
    )


# endregion


################################################################################
################################################################################
# region APS (APACHE II) scoring helpers


def _temp_points_aps(temp: pl.Expr) -> pl.Expr:
    """
    Temperature, °C
    ≥41.0        +4
     39.0-40.9   +3
     38.5-38.9   +1
     36.0-38.4    0
     34.0-35.9   +1
     32.0-33.9   +2
     30.0-31.9   +3
    <30.0        +4
    """
    return (
        pl.when(temp >= 41.0)
        .then(4)
        .when(temp.is_between(39.0, 41.0, closed="left"))
        .then(3)
        .when(temp.is_between(38.5, 39.0, closed="left"))
        .then(1)
        .when(temp.is_between(36.0, 38.5, closed="left"))
        .then(0)
        .when(temp.is_between(34.0, 36.0, closed="left"))
        .then(1)
        .when(temp.is_between(32.0, 34.0, closed="left"))
        .then(2)
        .when(temp.is_between(30.0, 32.0, closed="left"))
        .then(3)
        .when(temp < 30.0)
        .then(4)
        .otherwise(None)
    )


def _map_points_aps(map_val: pl.Expr) -> pl.Expr:
    """
    Mean Arterial Pressure, mmHg
    ≥160        +4
     130-159    +2
     110-129     0
      70-109    +2
      50- 69    +3
     <50        +4
    """
    return (
        pl.when(map_val >= 160)
        .then(4)
        .when(map_val.is_between(130, 160, closed="left"))
        .then(2)
        .when(map_val.is_between(110, 130, closed="left"))
        .then(0)
        .when(map_val.is_between(70, 110, closed="left"))
        .then(2)
        .when(map_val.is_between(50, 70, closed="left"))
        .then(3)
        .when(map_val < 50)
        .then(4)
        .otherwise(None)
    )


def _hr_points_aps(hr: pl.Expr) -> pl.Expr:
    """
    Heart Rate, bpm
    ≥180        +4
     140-179    +3
     110-139    +2
      70-109     0
      55- 69    +2
      40- 54    +3
     <40        +4
    """
    return (
        pl.when(hr >= 180)
        .then(4)
        .when(hr.is_between(140, 180, closed="left"))
        .then(3)
        .when(hr.is_between(110, 140, closed="left"))
        .then(2)
        .when(hr.is_between(70, 110, closed="left"))
        .then(0)
        .when(hr.is_between(55, 70, closed="left"))
        .then(2)
        .when(hr.is_between(40, 55, closed="left"))
        .then(3)
        .when(hr < 40)
        .then(4)
        .otherwise(None)
    )


def _rr_points_aps(rr: pl.Expr) -> pl.Expr:
    """
    Respiratory Rate, bpm
    ≥50         +4
     35-49      +3
     25-34      +1
     12-24       0
     10-11      +1
      6- 9      +3
     <6         +4
    """
    return (
        pl.when(rr >= 50)
        .then(4)
        .when(rr.is_between(35, 50, closed="left"))
        .then(3)
        .when(rr.is_between(25, 35, closed="left"))
        .then(1)
        .when(rr.is_between(12, 25, closed="left"))
        .then(0)
        .when(rr.is_between(10, 12, closed="left"))
        .then(1)
        .when(rr.is_between(6, 10, closed="left"))
        .then(3)
        .when(rr < 6)
        .then(4)
        .otherwise(None)
    )


def _oxygenation_points_aps(
    pao2: pl.Expr, fio2: pl.Expr, aa_grad: pl.Expr
) -> pl.Expr:
    """
    Oxygenation (PaO2 or A-a gradient)
    If FiO2 ≥ 50%, use A-a gradient:
        ≥500        +4
         350-499    +3
         200-349    +2
        <200         0
    If FiO2 < 50%, use PaO2:
        >70          0
         61-70      +1
         55-60      +3
        <55         +4
    """
    return (
        pl.when(fio2 >= 50)
        .then(
            pl.when(aa_grad >= 500)
            .then(4)
            .when(aa_grad.is_between(350, 500, closed="left"))
            .then(3)
            .when(aa_grad.is_between(200, 350, closed="left"))
            .then(2)
            .when(aa_grad < 200)
            .then(0)
            .otherwise(None)
        )
        .otherwise(
            pl.when(pao2 > 70)
            .then(0)
            .when(pao2.is_between(61, 71, closed="left"))
            .then(1)
            .when(pao2.is_between(55, 61, closed="left"))
            .then(3)
            .when(pao2 < 55)
            .then(4)
            .otherwise(None)
        )
    )


def _ph_points_aps(ph: pl.Expr) -> pl.Expr:
    """
    Arterial pH
    ≥7.70        +4
     7.60-7.69   +3
     7.50-7.59   +1
     7.33-7.49    0
     7.25-7.32   +2
     7.15-7.24   +3
    <7.15        +4
    """
    return (
        pl.when(ph >= 7.70)
        .then(4)
        .when(ph.is_between(7.60, 7.70, closed="left"))
        .then(3)
        .when(ph.is_between(7.50, 7.60, closed="left"))
        .then(1)
        .when(ph.is_between(7.33, 7.50, closed="left"))
        .then(0)
        .when(ph.is_between(7.25, 7.33, closed="left"))
        .then(2)
        .when(ph.is_between(7.15, 7.25, closed="left"))
        .then(3)
        .when(ph < 7.15)
        .then(4)
        .otherwise(None)
    )


def _sodium_points_aps(na: pl.Expr) -> pl.Expr:
    """
    Serum Sodium, mmol/L
    ≥180         +4
     160-179     +3
     155-159     +2
     150-154     +1
     130-149      0
     120-129     +2
     111-119     +3
    <111         +4
    """
    return (
        pl.when(na >= 180)
        .then(4)
        .when(na.is_between(160, 180, closed="left"))
        .then(3)
        .when(na.is_between(155, 160, closed="left"))
        .then(2)
        .when(na.is_between(150, 155, closed="left"))
        .then(1)
        .when(na.is_between(130, 150, closed="left"))
        .then(0)
        .when(na.is_between(120, 130, closed="left"))
        .then(2)
        .when(na.is_between(111, 120, closed="left"))
        .then(3)
        .when(na < 111)
        .then(4)
        .otherwise(None)
    )


def _potassium_points_aps(k: pl.Expr) -> pl.Expr:
    """
    Serum Potassium, mmol/L
    ≥7.0        +4
     6.0-6.9    +3
     5.5-5.9    +1
     3.5-5.4     0
     3.0-3.4    +1
     2.5-2.9    +2
    <2.5        +4
    """
    return (
        pl.when(k >= 7.0)
        .then(4)
        .when(k.is_between(6.0, 7.0, closed="left"))
        .then(3)
        .when(k.is_between(5.5, 6.0, closed="left"))
        .then(1)
        .when(k.is_between(3.5, 5.5, closed="left"))
        .then(0)
        .when(k.is_between(3.0, 3.5, closed="left"))
        .then(1)
        .when(k.is_between(2.5, 3.0, closed="left"))
        .then(2)
        .when(k < 2.5)
        .then(4)
        .otherwise(None)
    )


def _creatinine_points_aps(crea: pl.Expr) -> pl.Expr:
    """
    Serum Creatinine, mg/dL
    (Points are doubled if acute renal failure, not handled here)
    ≥3.5        +4
     2.0-3.4    +3
     1.5-1.9    +2
     0.6-1.4     0
    <0.6        +2
    """
    return (
        pl.when(crea >= 3.5)
        .then(4)
        .when(crea.is_between(2.0, 3.5, closed="left"))
        .then(3)
        .when(crea.is_between(1.5, 2.0, closed="left"))
        .then(2)
        .when(crea.is_between(0.6, 1.5, closed="left"))
        .then(0)
        .when(crea < 0.6)
        .then(2)
        .otherwise(None)
    )


def _hematocrit_points_aps(hct: pl.Expr) -> pl.Expr:
    """
    Hematocrit, %
    ≥60.0        +4
     50.0-59.9   +2
     46.0-49.9   +1
     30.0-45.9    0
     20.0-29.9   +2
    <20.0        +4
    """
    return (
        pl.when(hct >= 60.0)
        .then(4)
        .when(hct.is_between(50.0, 60.0, closed="left"))
        .then(2)
        .when(hct.is_between(46.0, 50.0, closed="left"))
        .then(1)
        .when(hct.is_between(30.0, 46.0, closed="left"))
        .then(0)
        .when(hct.is_between(20.0, 30.0, closed="left"))
        .then(2)
        .when(hct < 20.0)
        .then(4)
        .otherwise(None)
    )


def _wbc_points_aps(wbc: pl.Expr) -> pl.Expr:
    """
    White Blood Count, 10^3/µL
    ≥40.0         +4
     20.0-39.9    +2
     15.0-19.9    +1
      3.0-14.9     0
      1.0- 2.9    +2
     <1.0         +4
    """
    return (
        pl.when(wbc >= 40.0)
        .then(4)
        .when(wbc.is_between(20.0, 40.0, closed="left"))
        .then(2)
        .when(wbc.is_between(15.0, 20.0, closed="left"))
        .then(1)
        .when(wbc.is_between(3.0, 15.0, closed="left"))
        .then(0)
        .when(wbc.is_between(1.0, 3.0, closed="left"))
        .then(2)
        .when(wbc < 1.0)
        .then(4)
        .otherwise(None)
    )


def _gcs_points_aps(gcs: pl.Expr) -> pl.Expr:
    """
    Glasgow Coma Scale
    15 - GCS
    """
    return (15 - gcs).clip(0, 12)


# endregion


################################################################################
################################################################################
# region APS
def APS(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_vitals: Optional[pl.LazyFrame] = None,
    timeseries_labs: Optional[pl.LazyFrame] = None,
    timeseries_resp: Optional[pl.LazyFrame] = None,
    *,
    window_size: int = SECONDS_IN_1D,
    t_0: int = 0,
    t_1: Optional[int] = None,
    t_0_per_stay: Optional[pl.LazyFrame] = None,
    timeframe_name: Optional[str] = None,
) -> pl.LazyFrame:
    """
    Compute Acute Physiology Score (APACHE II) in long format.

    The Acute Physiology Score (APS) is a component of the APACHE II severity
    of disease classification system, representing the physiological derangement
    of the patient.

    Steps:
        1. Load and improve vitals, labs, and respiratory data.
        2. Calculate component points (temperature, MAP, heart rate, respiratory
           rate, oxygenation, pH, sodium, potassium, creatinine, hematocrit,
           WBC, GCS) per observation.
        3. Aggregate to worst-within-window points per component.
        4. Sum component points to get the total APS Score.

    Returns:
        pl.LazyFrame: Contains columns:
            - {STAY_KEY}: Global ICU Stay ID.
            - {TIME_KEY}: Reference time (T_0).
            - timeframe: 0-indexed window.
            - APS Score: Sum of component points.
            - Individual component points (temp_pts, map_pts, etc.).
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

    # Validate all required data is available
    required = {
        "patient_information": patient_information,
        "timeseries_vitals": timeseries_vitals,
        "timeseries_labs": timeseries_labs,
        "timeseries_resp": timeseries_resp,
    }

    missing = [name for name, data in required.items() if data is None]
    if missing:
        raise ValueError(
            f"Cannot compute APS: Missing required datasets: {', '.join(missing)}. "
            f"Ensure they are configured in ~/.reprodICU/PATHS.yaml or provide them explicitly."
        )

    vitals = _improve_vitals(timeseries_vitals)
    labs = _improve_labs(timeseries_labs)
    resp = _improve_resp(timeseries_resp)

    patient_information = patient_information.lazy()
    ALL_STAYS = patient_information.select(STAY_KEY)
    ALL_STAYS_T0 = _build_t0(ALL_STAYS, t_0_per_stay=t_0_per_stay, t_0=t_0)
    timeframe_name = _get_timeframe_name(
        timeframe_name, window_size, t_0, t_0_per_stay
    )

    # region component scoring
    vitals_tf = (
        vitals.join(ALL_STAYS_T0, on=STAY_KEY, how="inner")
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window_size))
        .group_by(STAY_KEY, "timeframe")
        .agg(
            _temp_points_aps(pl.col("Temperature")).max().alias("temp_pts"),
            _map_points_aps(pl.col("Mean arterial pressure")).max().alias("map_pts"),
            _hr_points_aps(pl.col("Heart rate")).max().alias("hr_pts"),
            _rr_points_aps(pl.col("Respiratory rate")).max().alias("rr_pts"),
            _gcs_points_aps(pl.col("Glasgow coma score total")).max().alias("gcs_pts"),
        )
    ) # fmt: skip

    labs_tf = (
        labs.join(ALL_STAYS_T0, on=STAY_KEY, how="inner")
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window_size))
        .group_by(STAY_KEY, "timeframe")
        .agg(
            _ph_points_aps(pl.col("pH")).max().alias("ph_pts"),
            _sodium_points_aps(pl.col("Sodium")).max().alias("na_pts"),
            _potassium_points_aps(pl.col("Potassium")).max().alias("k_pts"),
            _creatinine_points_aps(pl.col("Creatinine")).max().alias("crea_pts"),
            _hematocrit_points_aps(pl.col("Erythrocyte/Blood")).max().alias("hct_pts"),
            _wbc_points_aps(pl.col("Leukocytes")).max().alias("wbc_pts"),
        )
    ) # fmt: skip

    # Oxygenation needs PaO2, FiO2, and Aa-gradient
    resp_tf = resp.join(ALL_STAYS_T0, on=STAY_KEY, how="inner").with_columns(
        timeframe=_assign_timeframe(TIME_KEY, window_size)
    )

    aa_grad_tf = (
        Aa_GRADIENT(
            patient_information=patient_information,
            timeseries_resp=timeseries_resp,
            timeseries_labs=timeseries_labs,
            t_0=t_0,
            t_0_per_stay=t_0_per_stay,
        )
        .join(ALL_STAYS_T0, on=STAY_KEY, how="inner")
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window_size))
    )

    oxy_tf = (
        aa_grad_tf.join(
            labs.select(STAY_KEY, TIME_KEY, "Oxygen")
            .filter(pl.col("Oxygen").is_not_null()),
            on=[STAY_KEY, TIME_KEY],
            how="outer",
        )
        .join_asof(
            resp_tf.select(STAY_KEY, TIME_KEY, "FiO2"),
            on=TIME_KEY,
            by=STAY_KEY,
            strategy="backward",
            tolerance=4 * SECONDS_IN_1H,
        )
        .group_by(STAY_KEY, "timeframe")
        .agg(
            _oxygenation_points_aps(
                pl.col("Oxygen"),
                pl.col("FiO2"),
                pl.col("Alveolar-arterial oxygen Partial pressure difference"),
            )
            .max()
            .alias("oxy_pts"),
        )
    )
    # endregion

    # region union of all (stay,timeframe)
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
            .alias("timeframe"),
        )
        .explode("timeframe")
        .unique()
        .select(STAY_KEY, "T_0", "timeframe")
    )

    # region assemble
    out = base
    for part in [labs_tf, vitals_tf, oxy_tf]:
        out = out.join(part, on=[STAY_KEY, "timeframe"], how="left")

    return (
        out.filter(
            _optional_time_bounds_filter("timeframe", window_size, t_0, t_1)
        )
        .with_columns(
            pl.sum_horizontal(pl.exclude(STAY_KEY, "T_0", "timeframe")).alias(
                "APS Score"
            )
        )
        .select(
            STAY_KEY,
            "T_0",
            pl.col("timeframe").alias(timeframe_name),
            "APS Score",
            pl.all().exclude(STAY_KEY, "T_0", timeframe_name, "APS Score"),
        )
    )


# endregion

################################################################################
################################################################################
# region APS3 (APACHE III) scoring helpers


def _hr_points_aps3(hr: pl.Expr) -> pl.Expr:
    """
    Heart Rate, bpm
    ≥155        +17
     140-154    +14
     120-139     +7
     110-119     +5
     100-109     +1
      50- 99      0
      40- 49     +5
     <40         +8
    """
    return (
        pl.when(hr >= 155)
        .then(17)
        .when(hr.is_between(140, 155, closed="left"))
        .then(14)
        .when(hr.is_between(120, 140, closed="left"))
        .then(7)
        .when(hr.is_between(110, 120, closed="left"))
        .then(5)
        .when(hr.is_between(100, 110, closed="left"))
        .then(1)
        .when(hr.is_between(50, 100, closed="left"))
        .then(0)
        .when(hr.is_between(40, 50, closed="left"))
        .then(5)
        .when(hr < 40)
        .then(8)
        .otherwise(None)
    )


def _map_points_aps3(map_val: pl.Expr) -> pl.Expr:
    """
    Mean Arterial Pressure, mmHg
    ≥140        +10
     130-139     +7
     120-129     +6
     100-119     +4
      80- 99      0
      70- 79     +6
      60- 69     +7
      40- 59    +15
     <40        +23
    """
    return (
        pl.when(map_val >= 140)
        .then(10)
        .when(map_val.is_between(130, 140, closed="left"))
        .then(7)
        .when(map_val.is_between(120, 130, closed="left"))
        .then(6)
        .when(map_val.is_between(100, 120, closed="left"))
        .then(4)
        .when(map_val.is_between(80, 100, closed="left"))
        .then(0)
        .when(map_val.is_between(70, 80, closed="left"))
        .then(6)
        .when(map_val.is_between(60, 70, closed="left"))
        .then(7)
        .when(map_val.is_between(40, 60, closed="left"))
        .then(15)
        .when(map_val < 40)
        .then(23)
        .otherwise(None)
    )


def _temp_points_aps3(temp: pl.Expr) -> pl.Expr:
    """
    Temperature, °C
    ≥40.0         +4
     35.0-39.9     0
     34.0-34.9    +8
     33.5-33.9   +13
     33.0-33.4   +16
    <33.0        +20
    """
    return (
        pl.when(temp >= 40.0)
        .then(4)
        .when(temp.is_between(35.0, 40.0, closed="left"))
        .then(0)
        .when(temp.is_between(34.0, 35.0, closed="left"))
        .then(8)
        .when(temp.is_between(33.5, 34.0, closed="left"))
        .then(13)
        .when(temp.is_between(33.0, 33.5, closed="left"))
        .then(16)
        .when(temp < 33.0)
        .then(20)
        .otherwise(None)
    )


def _rr_points_aps3(rr: pl.Expr, ventilated: pl.Expr) -> pl.Expr:
    """
    Respiratory Rate, bpm
    ≥50         +18
     35-49      +11
     30-34       +9
     25-29       +6
     14-24        0
    (if ventilated, 6-13 is 0)
     12-13       +7
      6-11       +8
     <6         +17
    """
    return (
        pl.when(rr >= 50)
        .then(18)
        .when(rr.is_between(35, 50, closed="left"))
        .then(11)
        .when(rr.is_between(30, 35, closed="left"))
        .then(9)
        .when(rr.is_between(25, 30, closed="left"))
        .then(6)
        .when(rr.is_between(14, 25, closed="left"))
        .then(0)
        .when(ventilated & rr.is_between(6, 14, closed="left"))
        .then(0)
        .when(rr.is_between(12, 14, closed="left"))
        .then(7)
        .when(rr.is_between(6, 12, closed="left"))
        .then(8)
        .when(rr < 6)
        .then(17)
        .otherwise(None)
    )


def _oxygenation_points_aps3(
    pao2: pl.Expr, fio2: pl.Expr, aa_grad: pl.Expr
) -> pl.Expr:
    """
    Oxygenation (PaO2 or A-a gradient)
    If FiO2 ≥ 50%, use A-a gradient:
        ≥500         +14
         350-499     +11
         250-349      +9
         100-249      +7
        <100           0
    If FiO2 < 50%, use PaO2:
         ≥80           0
          70- 79      +2
          50- 69      +5
         <50         +15
    """
    return (
        pl.when(fio2 >= 50)
        .then(
            pl.when(aa_grad >= 500)
            .then(14)
            .when(aa_grad.is_between(350, 500, closed="left"))
            .then(11)
            .when(aa_grad.is_between(250, 350, closed="left"))
            .then(9)
            .when(aa_grad.is_between(100, 250, closed="left"))
            .then(7)
            .when(aa_grad < 100)
            .then(0)
            .otherwise(None)
        )
        .otherwise(
            pl.when(pao2 >= 80)
            .then(0)
            .when(pao2.is_between(70, 80, closed="left"))
            .then(2)
            .when(pao2.is_between(50, 70, closed="left"))
            .then(5)
            .when(pao2 < 50)
            .then(15)
            .otherwise(None)
        )
    )


def _hct_points_aps3(hct: pl.Expr) -> pl.Expr:
    """
    Hematocrit, %
    ≥50.0        +3
     41.0-49.9    0
    <41.0        +3
    """
    return (
        pl.when(hct >= 50.0)
        .then(3)
        .when(hct.is_between(41.0, 50.0, closed="left"))
        .then(0)
        .when(hct < 41.0)
        .then(3)
        .otherwise(None)
    )


def _wbc_points_aps3(wbc: pl.Expr) -> pl.Expr:
    """
    White Blood Count, 10^3/µL
    ≥25.0         +5
     20.0-24.9    +1
      3.0-19.9     0
      1.0- 2.9    +5
     <1.0        +19
    """
    return (
        pl.when(wbc >= 25.0)
        .then(5)
        .when(wbc.is_between(20.0, 25.0, closed="left"))
        .then(1)
        .when(wbc.is_between(3.0, 20.0, closed="left"))
        .then(0)
        .when(wbc.is_between(1.0, 3.0, closed="left"))
        .then(5)
        .when(wbc < 1.0)
        .then(19)
        .otherwise(None)
    )


def _creatinine_points_aps3(crea: pl.Expr) -> pl.Expr:
    """
    Serum Creatinine, mg/dL
    (Without acute renal failure)
    ≥1.95        +7
     1.50-1.94   +4
     0.50-1.49    0
    <0.50        +3
    """
    return (
        pl.when(crea >= 1.95)
        .then(7)
        .when(crea.is_between(1.50, 1.95, closed="left"))
        .then(4)
        .when(crea.is_between(0.50, 1.50, closed="left"))
        .then(0)
        .when(crea < 0.50)
        .then(3)
        .otherwise(None)
    )


def _uo_points_aps3(uo_l_day: pl.Expr) -> pl.Expr:
    """
    Urine Output, L/day
    <0.40        +15
     0.40-0.59    +8
     0.60-0.89    +7
     0.90-1.49    +5
     1.50-1.99    +4
     2.00-3.99     0
    ≥4.00         +1
    """
    return (
        pl.when(uo_l_day < 0.40)
        .then(15)
        .when(uo_l_day.is_between(0.40, 0.60, closed="left"))
        .then(8)
        .when(uo_l_day.is_between(0.60, 0.90, closed="left"))
        .then(7)
        .when(uo_l_day.is_between(0.90, 1.50, closed="left"))
        .then(5)
        .when(uo_l_day.is_between(1.50, 2.00, closed="left"))
        .then(4)
        .when(uo_l_day.is_between(2.00, 4.00, closed="left"))
        .then(0)
        .when(uo_l_day >= 4.00)
        .then(1)
        .otherwise(None)
    )


def _bun_points_aps3(bun: pl.Expr) -> pl.Expr:
    """
    BUN, mg/dL
    ≥80         +12
     40-79       +8
     20-39       +6
     17-19       +2
    <17           0
    """
    return (
        pl.when(bun >= 80)
        .then(12)
        .when(bun.is_between(40, 80, closed="left"))
        .then(8)
        .when(bun.is_between(20, 40, closed="left"))
        .then(6)
        .when(bun.is_between(17, 20, closed="left"))
        .then(2)
        .when(bun < 17)
        .then(0)
        .otherwise(None)
    )


def _sodium_points_aps3(na: pl.Expr) -> pl.Expr:
    """
    Serum Sodium, mmol/L
    ≥155         +4
     135-154      0
     120-134     +2
    <120         +3
    """
    return (
        pl.when(na >= 155)
        .then(4)
        .when(na.is_between(135, 155, closed="left"))
        .then(0)
        .when(na.is_between(120, 135, closed="left"))
        .then(2)
        .when(na < 120)
        .then(3)
        .otherwise(None)
    )


def _albumin_points_aps3(alb: pl.Expr) -> pl.Expr:
    """
    Albumin, g/dL
    ≥4.5         +4
     2.5-4.4      0
     2.0-2.4     +6
    <2.0        +11
    """
    return (
        pl.when(alb >= 4.5)
        .then(4)
        .when(alb.is_between(2.5, 4.5, closed="left"))
        .then(0)
        .when(alb.is_between(2.0, 2.5, closed="left"))
        .then(6)
        .when(alb < 2.0)
        .then(11)
        .otherwise(None)
    )


def _bilirubin_points_aps3(bili: pl.Expr) -> pl.Expr:
    """
    Bilirubin, mg/dL
    ≥8.0        +16
     5.0-7.9     +8
     3.0-4.9     +6
     2.0-2.9     +5
    <2.0          0
    """
    return (
        pl.when(bili >= 8.0)
        .then(16)
        .when(bili.is_between(5.0, 8.0, closed="left"))
        .then(8)
        .when(bili.is_between(3.0, 5.0, closed="left"))
        .then(6)
        .when(bili.is_between(2.0, 3.0, closed="left"))
        .then(5)
        .when(bili < 2.0)
        .then(0)
        .otherwise(None)
    )


def _glucose_points_aps3(glu: pl.Expr) -> pl.Expr:
    """
    Glucose, mg/dL
    ≥350         +5
     200-349     +3
      60-199      0
      40- 59     +9
     <40         +8
    """
    return (
        pl.when(glu >= 350)
        .then(5)
        .when(glu.is_between(200, 350, closed="left"))
        .then(3)
        .when(glu.is_between(60, 200, closed="left"))
        .then(0)
        .when(glu.is_between(40, 60, closed="left"))
        .then(9)
        .when(glu < 40)
        .then(8)
        .otherwise(None)
    )


def _acid_base_points_aps3(ph: pl.Expr, paco2: pl.Expr) -> pl.Expr:
    """
    Acid-Base Status (pH and PaCO2)
    Table based on both values.
    """
    return (
        pl.when(ph.is_null() | paco2.is_null())
        .then(None)
        .when(ph < 7.20)
        .then(pl.when(paco2 < 50).then(12).otherwise(4))
        .when(ph.is_between(7.20, 7.30, closed="left"))
        .then(
            pl.when(paco2 < 30)
            .then(9)
            .when(paco2 < 40)
            .then(6)
            .when(paco2 < 50)
            .then(3)
            .otherwise(2)
        )
        .when(ph.is_between(7.30, 7.35, closed="left"))
        .then(pl.when(paco2 < 30).then(9).when(paco2 < 45).then(0).otherwise(1))
        .when(ph.is_between(7.35, 7.45, closed="left"))
        .then(pl.when(paco2 < 30).then(5).when(paco2 < 45).then(0).otherwise(1))
        .when(ph.is_between(7.45, 7.50, closed="left"))
        .then(
            pl.when(paco2 < 30)
            .then(5)
            .when(paco2 < 35)
            .then(0)
            .when(paco2 < 45)
            .then(2)
            .otherwise(12)
        )
        .when(ph.is_between(7.50, 7.60, closed="left"))
        .then(pl.when(paco2 < 40).then(3).otherwise(12))
        .otherwise(
            pl.when(paco2 < 25).then(0).when(paco2 < 40).then(3).otherwise(12)
        )
    )


def _gcs_points_aps3(eyes: pl.Expr, motor: pl.Expr, verbal: pl.Expr) -> pl.Expr:
    """
    Glasgow Coma Scale (APACHE III version)
    Table based on eyes, motor, and verbal components.
    """
    return (
        pl.when(eyes == 1)
        .then(
            pl.when(verbal == 1)
            .then(
                pl.when(motor.is_in([1, 2]))
                .then(48)
                .when(motor.is_in([3, 4]))
                .then(33)
                .when(motor.is_in([5, 6]))
                .then(16)
            )
            .when(verbal.is_in([2, 3]))
            .then(
                pl.when(motor.is_in([1, 2]))
                .then(29)
                .when(motor.is_in([3, 4]))
                .then(24)
                .otherwise(None)
            )
            .otherwise(None)
        )
        .when(eyes > 1)
        .then(
            pl.when(verbal == 1)
            .then(
                pl.when(motor.is_in([1, 2]))
                .then(29)
                .when(motor.is_in([3, 4]))
                .then(24)
                .when(motor.is_in([5, 6]))
                .then(15)
            )
            .when(verbal.is_in([2, 3]))
            .then(
                pl.when(motor.is_in([1, 2]))
                .then(29)
                .when(motor.is_in([3, 4]))
                .then(24)
                .when(motor == 5)
                .then(13)
                .when(motor == 6)
                .then(10)
            )
            .when(verbal == 4)
            .then(
                pl.when(motor.is_in([1, 2, 3, 4]))
                .then(13)
                .when(motor == 5)
                .then(8)
                .when(motor == 6)
                .then(3)
            )
            .when(verbal == 5)
            .then(
                pl.when(motor.is_in([1, 2, 3, 4, 5]))
                .then(3)
                .when(motor == 6)
                .then(0)
            )
        )
        .otherwise(None)
    )


# endregion


################################################################################
################################################################################
# region APS3
def APS3(
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
    Acute Physiology Score III (APACHE III) in long format.

    Steps:
        1. Load and improve vitals, labs, respiratory, and urine data.
        2. Assign timeframes based on T_0 and window_size.
        3. Calculate component points for each observation.
        4. Aggregate to worst-within-window points.
        5. Sum component points to get APS3 Score.

    Returns:
        pl.LazyFrame: Contains columns:
            - {STAY_KEY}: Global ICU Stay ID.
            - {TIME_KEY}: Reference time (T_0).
            - timeframe: 0-indexed window.
            - APS3 Score: Sum of component points.
            - Component points: hr_pts, map_pts, temp_pts, rr_pts, oxy_pts, hct_pts, wbc_pts, crea_pts, uo_pts, bun_pts, na_pts, alb_pts, bili_pts, glu_pts, ab_pts, gcs_pts.
            - Helpers: PaO2/FiO2 Ratio, uo_l_day.
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

    missing = [name for name, data in required.items() if data is None]
    if missing:
        raise ValueError(
            f"Cannot compute APS3: Missing required datasets: {', '.join(missing)}. "
            f"Ensure they are configured in ~/.reprodICU/PATHS.yaml or provide them explicitly."
        )

    vitals = _improve_vitals(timeseries_vitals)
    labs = _improve_labs(timeseries_labs)
    resp = _improve_resp(timeseries_resp)

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
        .join(vent, [STAY_KEY, "timeframe"], how="left")
        .with_columns(pl.col("ventilated").fill_null(False))
        .group_by(STAY_KEY, "timeframe")
        .agg(
            _hr_points_aps3(pl.col("Heart rate")).max().alias("hr_pts"),
            _map_points_aps3(pl.col("Mean arterial pressure")).max().alias("map_pts"),
            _temp_points_aps3(pl.col("Temperature")).max().alias("temp_pts"),
            _rr_points_aps3(pl.col("Respiratory rate"), pl.col("ventilated")).max().alias("rr_pts"),
            _gcs_points_aps3(
                pl.col("Glasgow coma score eye opening"),
                pl.col("Glasgow coma score motor"),
                pl.col("Glasgow coma score verbal"),
            ).max().alias("gcs_pts"),
        )
    ) # fmt: skip

    labs_tf = (
        labs.join(ALL_STAYS_T0, on=STAY_KEY, how="inner")
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window_size))
        .group_by(STAY_KEY, "timeframe")
        .agg(
            _hct_points_aps3(pl.col("Erythrocyte/Blood")).max().alias("hct_pts"),
            _wbc_points_aps3(pl.col("Leukocytes")).max().alias("wbc_pts"),
            _creatinine_points_aps3(pl.col("Creatinine")).max().alias("crea_pts"),
            _bun_points_aps3(pl.col("Urea nitrogen")).max().alias("bun_pts"),
            _sodium_points_aps3(pl.col("Sodium")).max().alias("na_pts"),
            _albumin_points_aps3(pl.col("Albumin")).max().alias("alb_pts"),
            _bilirubin_points_aps3(pl.col("Bilirubin")).max().alias("bili_pts"),
            _glucose_points_aps3(pl.col("Glucose")).max().alias("glu_pts"),
            _acid_base_points_aps3(pl.col("pH"), pl.col("Oxygen")).max().alias("ab_pts"),
        )
    ) # fmt: skip

    # Oxygenation needs PaO2, FiO2, and Aa-gradient
    resp_tf = resp.join(ALL_STAYS_T0, on=STAY_KEY, how="inner").with_columns(
        timeframe=_assign_timeframe(TIME_KEY, window_size)
    )

    aa_grad_tf = (
        Aa_GRADIENT(
            patient_information=patient_information,
            timeseries_resp=timeseries_resp,
            timeseries_labs=timeseries_labs,
            t_0=t_0,
            t_0_per_stay=t_0_per_stay,
        )
        .join(ALL_STAYS_T0, on=STAY_KEY, how="inner")
        .with_columns(
            _assign_timeframe(TIME_KEY, window_size).alias("timeframe")
        )
    )

    oxy_tf = (
        aa_grad_tf.join(
            labs.select(STAY_KEY, TIME_KEY, "Oxygen"),
            [STAY_KEY, TIME_KEY],
            how="inner",
        )
        .sort(STAY_KEY, TIME_KEY)
        .join_asof(
            resp_tf.select(STAY_KEY, TIME_KEY, "FiO2").sort(STAY_KEY, TIME_KEY),
            on=TIME_KEY,
            by=STAY_KEY,
            strategy="backward",
            tolerance=4 * SECONDS_IN_1H,
        )
        .group_by(STAY_KEY, "timeframe")
        .agg(
            _oxygenation_points_aps3(
                pl.col("Oxygen"),
                pl.col("FiO2"),
                pl.col("Alveolar-arterial oxygen Partial pressure difference"),
            )
            .max()
            .alias("oxy_pts")
        )
    )

    # Urine output (scaled to 24h)
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
        .agg(_uo_points_aps3(pl.col("uo_l_day")).max().alias("uo_pts"))
    )
    # endregion

    # region union of all (stay,timeframe)
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
            .alias("timeframe"),
        )
        .explode("timeframe")
        .unique()
        .select(STAY_KEY, "T_0", "timeframe")
    )

    # region assemble
    out = base
    for part in [labs_tf, vitals_tf, oxy_tf, uo_tf]:
        out = out.join(part, on=[STAY_KEY, "timeframe"], how="left")

    return (
        out.filter(
            _optional_time_bounds_filter("timeframe", window_size, t_0, t_1)
        )
        .with_columns(
            pl.sum_horizontal(pl.exclude(STAY_KEY, "T_0", "timeframe")).alias(
                "APS3 Score"
            )
        )
        .select(
            STAY_KEY,
            "T_0",
            pl.col("timeframe").alias(timeframe_name),
            "APS3 Score",
            pl.all().exclude(STAY_KEY, "T_0", timeframe_name, "APS3 Score"),
        )
    )


# endregion

__all__ = ["APS", "APS3"]
