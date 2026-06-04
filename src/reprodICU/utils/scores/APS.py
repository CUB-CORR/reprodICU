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
from ..common import (
    ScoringTable,
    _assign_timeframe,
    _build_base_timeframes,
    _build_t0,
    _get_timeframe_name,
    _optional_time_bounds_filter,
    extract_struct_value,
    get_patient_information,
    get_timeseries_intakeoutput,
    get_timeseries_labs,
    get_timeseries_respiratory,
    get_timeseries_vitals,
    get_ventilation,
    intervention_per_timeframe,
)
from ..core import BLOOD_PRESSURES
from ..laboratory.oxygenation.ALVEOLAR_ARTERIAL_GRADIENT import Aa_GRADIENT

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
        BLOOD_PRESSURES(timeseries_vitals=vitals)
        .rename({"MAP": "Mean arterial pressure"})
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
    sources = ["Serum or Plasma", "Blood", "Blood arterial"]
    LABS = [
        "Oxygen", "Carbon dioxide", "pH", "Sodium", "Potassium", "Creatinine",
        "Erythrocyte/Blood", "Leukocytes", "Urea nitrogen",  "Albumin",
        "Bilirubin", "Glucose",
    ] # fmt: skip
    return (
        labs.with_columns(
            extract_struct_value(col, allowed_systems=sources).alias(col)
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
    return ScoringTable([            # Temperature (°C) | Points
        (41.0, None, "left",    4),  # ≥41.0              4
        (39.0, 41.0, "left",    3),  #  39.0-40.9         3
        (38.5, 39.0, "left",    1),  #  38.5-38.9         1
        (36.0, 38.5, "left",    0),  #  36.0-38.4 ....... 0
        (34.0, 36.0, "left",    1),  #  34.0-35.9         1
        (32.0, 34.0, "left",    2),  #  32.0-33.9         2
        (30.0, 32.0, "left",    3),  #  30.0-31.9         3
        (None, 30.0, "neither", 4),  # <30.0              4
    ]).to_expr(temp) # fmt: skip


def _map_points_aps(map_val: pl.Expr) -> pl.Expr:
    return ScoringTable([            # Mean Arterial Pressure (mmHg) | Points
        ( 160, None, "left",    4),  # ≥160                            4
        ( 130,  160, "left",    2),  #  130-159                        2
        ( 110,  130, "left",    0),  #  110-129 ...................... 0
        (  70,  110, "left",    2),  #   70-109                        2
        (  50,   70, "left",    3),  #   50- 69                        3
        (None,   50, "neither", 4),  #  <50                            4
    ]).to_expr(map_val) # fmt: skip


def _hr_points_aps(hr: pl.Expr) -> pl.Expr:
    return ScoringTable([            # Heart rate (bpm) | Points
        ( 180, None, "left",    4),  # ≥180               4
        ( 140,  180, "left",    3),  #  140-179           3
        ( 110,  140, "left",    2),  #  110-139           2
        (  70,  110, "left",    0),  #   70-109 ......... 0
        (  55,   70, "left",    2),  #   55- 69           2
        (  40,   55, "left",    3),  #   40- 54           3
        (None,   40, "neither", 4),  #  <40               4
    ]).to_expr(hr) # fmt: skip


def _rr_points_aps(rr: pl.Expr) -> pl.Expr:
    return ScoringTable([            # Respiratory rate (bpm) | Points
        (  50, None, "left",    4),  # ≥50                      4
        (  35,   50, "left",    3),  #  35-49                   3
        (  25,   35, "left",    1),  #  25-34                   1
        (  12,   25, "left",    0),  #  12-24 ................. 0
        (  10,   12, "left",    1),  #  10-11                   1
        (   6,   10, "left",    3),  #   6- 9                   3
        (None,    6, "neither", 4),  #  <6                      4
    ]).to_expr(rr) # fmt: skip


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
    return ScoringTable([            # pH          | Points
        (7.70, None, "left",    4),  # ≥7.70         4
        (7.60, 7.70, "left",    3),  #  7.60-7.69    3
        (7.50, 7.60, "left",    1),  #  7.50-7.59    1
        (7.33, 7.50, "left",    0),  #  7.33-7.49 .. 0
        (7.25, 7.33, "left",    2),  #  7.25-7.32    2
        (7.15, 7.25, "left",    3),  #  7.15-7.24    3
        (None, 7.15, "neither", 4),  # <7.15         4
    ]).to_expr(ph) # fmt: skip


def _sodium_points_aps(na: pl.Expr) -> pl.Expr:
    return ScoringTable([            # Sodium (mmol/L) | Points
        ( 180, None, "left",    4),  # ≥180              4
        ( 160,  180, "left",    3),  #  160-179          3
        ( 155,  160, "left",    2),  #  155-159          2
        ( 150,  155, "left",    1),  #  150-154          1
        ( 130,  150, "left",    0),  #  130-149 ........ 0
        ( 120,  130, "left",    2),  #  120-129          2
        ( 111,  120, "left",    3),  #  111-119          3
        (None,  111, "neither", 4),  # <111              4
    ]).to_expr(na) # fmt: skip


def _potassium_points_aps(k: pl.Expr) -> pl.Expr:
    return ScoringTable([            # Potassium (mmol/L) | Points
        ( 7.0, None, "left",    4),  # ≥7.0                 4
        ( 6.0,  7.0, "left",    3),  #  6.0-6.9             3
        ( 5.5,  6.0, "left",    1),  #  5.5-5.9             1
        ( 3.5,  5.5, "left",    0),  #  3.5-5.4 ........... 0
        ( 3.0,  3.5, "left",    1),  #  3.0-3.4             1
        ( 2.5,  3.0, "left",    2),  #  2.5-2.9             2
        (None,  2.5, "neither", 4),  # <2.5                 4
    ]).to_expr(k) # fmt: skip


def _creatinine_points_aps(crea: pl.Expr) -> pl.Expr:
    return ScoringTable([            # Creatinine (mg/dL) | Points
        ( 3.5, None, "left",    4),  # ≥3.5                 4
        ( 2.0,  3.5, "left",    3),  #  2.0-3.4             3
        ( 1.5,  2.0, "left",    2),  #  1.5-1.9             2
        ( 0.6,  1.5, "left",    0),  #  0.6-1.4 ........... 0
        (None,  0.6, "neither", 2),  # <0.6                 2
    ]).to_expr(crea) # fmt: skip


def _hematocrit_points_aps(hct: pl.Expr) -> pl.Expr:
    return ScoringTable([            # Hematocrit (%) | Points
        (60.0, None, "left",    4),  # ≥60.0            4
        (50.0, 60.0, "left",    2),  #  50.0-59.9       2
        (46.0, 50.0, "left",    1),  #  46.0-49.9       1
        (30.0, 46.0, "left",    0),  #  30.0-45.9 ..... 0
        (20.0, 30.0, "left",    2),  #  20.0-29.9       2
        (None, 20.0, "neither", 4),  # <20.0            4
    ]).to_expr(hct) # fmt: skip


def _wbc_points_aps(wbc: pl.Expr) -> pl.Expr:
    return ScoringTable([            # WBC (10^9/L) | Points
        (40.0, None, "left",    4),  # ≥40.0          4
        (20.0, 40.0, "left",    2),  #  20.0-39.9     2
        (15.0, 20.0, "left",    1),  #  15.0-19.9     1
        ( 3.0, 15.0, "left",    0),  #   3.0-14.9 ... 0
        ( 1.0,  3.0, "left",    2),  #   1.0- 2.9     2
        (None,  1.0, "neither", 4),  #  <1.0          4
    ]).to_expr(wbc) # fmt: skip


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
    base = _build_base_timeframes(ALL_STAYS_T0, patient_information, window_size) # fmt: skip

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
    return ScoringTable([            # Heart rate (bpm) | Points
        ( 155, None, "left",   17),  # ≥155               17
        ( 140,  155, "left",   14),  #  140-154           14
        ( 120,  140, "left",    7),  #  120-139            7
        ( 110,  120, "left",    5),  #  110-119            5
        ( 100,  110, "left",    1),  #  100-109            1
        (  50,  100, "left",    0),  #   50- 99 .......... 0
        (None,   50, "neither", 6),  #  <50                6
    ]).to_expr(hr) # fmt: skip


def _map_points_aps3(map_val: pl.Expr) -> pl.Expr:
    return ScoringTable([             # MAP (mmHg) | Points
        ( 140, None, "left",    10),  # ≥140         10
        ( 130,  140, "left",     7),  #  130-139      7
        ( 120,  130, "left",     6),  #  120-129      6
        ( 100,  120, "left",     4),  #  100-119      4
        (  80,  100, "left",     0),  #   80- 99 .... 0
        (  70,   80, "left",     6),  #   70- 79      6
        (None,   70, "neither", 18),  #  <70         18
    ]).to_expr(map_val) # fmt: skip


def _temp_points_aps3(temp: pl.Expr) -> pl.Expr:
    return ScoringTable([             # Temperature (°C) | Points
        (40.0, None, "left",     4),  # ≥40.0               4
        (35.0, 40.0, "left",     0),  #  35.0-39.9 ........ 0
        (34.0, 35.0, "left",     8),  #  34.0-34.9          8
        (33.5, 34.0, "left",    13),  #  33.5-33.9         13
        (33.0, 33.5, "left",    16),  #  33.0-33.4         16
        (None, 33.0, "neither", 20),  # <33.0              20
    ]).to_expr(temp) # fmt: skip


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
    return ScoringTable([            # Hematocrit (%) | Points
        (60.0, None, "left",    4),  # ≥60.0            4
        (50.0, 60.0, "left",    2),  #  50.0-59.9       2
        (46.0, 50.0, "left",    1),  #  46.0-49.9       1
        (30.0, 46.0, "left",    0),  #  30.0-45.9 ..... 0
        (20.0, 30.0, "left",    2),  #  20.0-29.9       2
        (None, 20.0, "neither", 4),  # <20.0            4
    ]).to_expr(hct) # fmt: skip


def _wbc_points_aps3(wbc: pl.Expr) -> pl.Expr:
    return ScoringTable([            # WBC (10^9/L) | Points
        (40.0, None, "left",    4),  # ≥40.0          4
        (20.0, 40.0, "left",    2),  #  20.0-39.9     2
        (15.0, 20.0, "left",    1),  #  15.0-19.9     1
        ( 3.0, 15.0, "left",    0),  #   3.0-14.9 ... 0
        ( 1.0,  3.0, "left",    2),  #   1.0-2.9      2
        (None,  1.0, "neither", 4),  #  <1.0          4
    ]).to_expr(wbc) # fmt: skip


def _creatinine_points_aps3(crea: pl.Expr) -> pl.Expr:
    return ScoringTable([            # Creatinine (mg/dL) | Points
        ( 3.5, None, "left",   10),  # ≥3.5                 10
        ( 2.0,  3.5, "left",    7),  #  2.0-3.4              7
        ( 1.5,  2.0, "left",    5),  #  1.5-1.9              5
        ( 1.2,  1.5, "left",    3),  #  1.2-1.4              3
        ( 0.6,  1.2, "left",    0),  #  0.6-1.1 ............ 0
        (None,  0.6, "neither", 7),  # <0.6                  7
    ]).to_expr(crea) # fmt: skip


def _uo_points_aps3(uo_l_day: pl.Expr) -> pl.Expr:
    return ScoringTable([             # Urine output (L/day) | Points
        (None,  0.5, "neither", 11),  # <0.5                   11
        ( 0.5,  1.0, "left",     5),  #  0.5-0.99               5
         (1.0, None, "left",     0),  # ≥1.0 .................. 0
    ]).to_expr(uo_l_day) # fmt: skip


def _bun_points_aps3(bun: pl.Expr) -> pl.Expr:
    return ScoringTable([            # BUN (mg/dL) | Points
        (84.0, None, "left",   11),  # ≥84           11
        (70.0, 84.0, "left",    7),  #  70-83         7
        (28.0, 70.0, "left",    0),  #  28-69 ....... 0
        (None, 28.0, "neither", 3),  # <28            3
    ]).to_expr(bun) # fmt: skip


def _sodium_points_aps3(na: pl.Expr) -> pl.Expr:
    return ScoringTable([              # Sodium (mmol/L) | Points
        (180.0,  None, "left",    8),  # ≥180              8
        (160.0, 180.0, "left",    5),  #  160-179          5
        (155.0, 160.0, "left",    4),  #  155-159          4
        (150.0, 155.0, "left",    3),  #  150-154          3
        (130.0, 150.0, "left",    0),  #  130-149 ........ 0
        (120.0, 130.0, "left",    4),  #  120-129          4
        (111.0, 120.0, "left",    6),  #  111-119          6
        ( None, 111.0, "neither", 8),  # <111              8
    ]).to_expr(na) # fmt: skip


def _albumin_points_aps3(alb: pl.Expr) -> pl.Expr:
    return ScoringTable([            # Albumin (g/dL) | Points
        (4.5, None, "left",     0),  # ≥4.5 ............ 0
        (3.5, 4.5,  "left",     4),  #  3.5-4.4          4
        (2.5, 3.5,  "left",     7),  #  2.5-3.4          7
        (None, 2.5, "neither", 11),  # <2.5             11
    ]).to_expr(alb) # fmt: skip


def _bilirubin_points_aps3(bili: pl.Expr) -> pl.Expr:
    return ScoringTable([            # Bilirubin (mg/dL) | Points
        ( 6.0, None, "left",   16),  # ≥6.0                16
        ( 4.0,  6.0, "left",    9),  #  4.0-5.9             9
        ( 2.0,  4.0, "left",    6),  #  2.0-3.9             6
        (None,  2.0, "neither", 0),  # <2.0 ............... 0
    ]).to_expr(bili) # fmt: skip


def _glucose_points_aps3(glu: pl.Expr) -> pl.Expr:
    return ScoringTable([              # Glucose (mg/dL) | Points
        (400.0,  None, "left",   11),  # ≥400              11
        (300.0, 400.0, "left",    8),  #  300-399           8
        (150.0, 300.0, "left",    5),  #  150-299           5
        ( 70.0, 150.0, "left",    0),  #   70-149 ......... 0
        ( None,  70.0, "neither", 8),  #  <70               8
    ]).to_expr(glu) # fmt: skip


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
                pl.when(motor.is_in([1, 2])).then(48)
                  .when(motor.is_in([3, 4])).then(33)
                  .when(motor.is_in([5, 6])).then(16)
            )
            .when(verbal.is_in([2, 3]))
            .then(
                pl.when(motor.is_in([1, 2])).then(29)
                  .when(motor.is_in([3, 4])).then(24)
                  .otherwise(None)
            )
            .otherwise(None)
        )
        .when(eyes > 1)
        .then(
            pl.when(verbal == 1)
            .then(
                pl.when(motor.is_in([1, 2])).then(29)
                  .when(motor.is_in([3, 4])).then(24)
                  .when(motor.is_in([5, 6])).then(15)
            )
            .when(verbal.is_in([2, 3]))
            .then(
                pl.when(motor.is_in([1, 2])).then(29)
                  .when(motor.is_in([3, 4])).then(24)
                  .when(    motor == 5     ).then(13)
                  .when(    motor == 6     ).then(10)
            )
            .when(verbal == 4)
            .then(
                pl.when(motor.is_in([1, 2, 3, 4])).then(13)
                  .when(        motor == 5       ).then( 8)
                  .when(        motor == 6       ).then( 3)
            )
            .when(verbal == 5)
            .then(
                pl.when(motor.is_in([1, 2, 3, 4, 5])).then(3)
                  .when(         motor == 6         ).then(0)
            )
        )
        .otherwise(None)
    ) # fmt: skip


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
    base = _build_base_timeframes(ALL_STAYS_T0, patient_information, window_size) # fmt: skip

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
