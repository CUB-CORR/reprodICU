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
    _assign_timeframe,
    _build_base_timeframes,
    _build_t0,
    _get_timeframe_name,
    _optional_time_bounds_filter,
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
    sources = ["Serum or Plasma", "Blood", "Blood arterial", None]
    LABS = [
        "Sodium", "Potassium", "Urea nitrogen", "Bilirubin", "Leukocytes",
        "Bicarbonate",
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


# endregion


################################################################################
################################################################################
# region SAPS-II scoring helpers


def _age_points_saps2(age: pl.Expr) -> pl.Expr:
    """
    Age (years)
    <40           0
     40-59        7
     60-69       12
     70-74       15
     75-79       16
      >=80       18
    """
    return (
        pl.when(age < 40)
        .then(0)
        .when(age.is_between(40, 60, closed="left"))
        .then(7)
        .when(age.is_between(60, 70, closed="left"))
        .then(12)
        .when(age.is_between(70, 75, closed="left"))
        .then(15)
        .when(age.is_between(75, 80, closed="left"))
        .then(16)
        .when(age >= 80)
        .then(18)
        .otherwise(None)
    )


def _hr_points_saps2(hr: pl.Expr) -> pl.Expr:
    """
    Heart Rate (bpm)
    <40         11
     40- 69      2
     70-119      0
    120-159      4
      >=160      7
    """
    return (
        pl.when(hr < 40)
        .then(11)
        .when(hr.is_between(40, 70, closed="left"))
        .then(2)
        .when(hr.is_between(70, 120, closed="left"))
        .then(0)
        .when(hr.is_between(120, 160, closed="left"))
        .then(4)
        .when(hr >= 160)
        .then(7)
        .otherwise(None)
    )


def _sbp_points_saps2(sbp: pl.Expr) -> pl.Expr:
    """
    Systolic Arterial Pressure (mmHg)
    <70         13
     70- 99      5
    100-199      0
      >=200      2
    """
    return (
        pl.when(sbp < 70)
        .then(13)
        .when(sbp.is_between(70, 100, closed="left"))
        .then(5)
        .when(sbp.is_between(100, 200, closed="left"))
        .then(0)
        .when(sbp >= 200)
        .then(2)
        .otherwise(None)
    )


def _temp_points_saps2(temp: pl.Expr) -> pl.Expr:
    """
    Temperature (°C)
     <39.0       0
    >=39.0       3
    """
    return (
        pl.when(temp < 39.0).then(0).when(temp >= 39.0).then(3).otherwise(None)
    )


def _pao2fio2_points_saps2(paf: pl.Expr) -> pl.Expr:
    """
    PaO2/FiO2 ratio (mmHg)
    <100        11
     100-199     9
       >=200     6
    """
    return (
        pl.when(paf < 100)
        .then(11)
        .when(paf.is_between(100, 200, closed="left"))
        .then(9)
        .when(paf >= 200)
        .then(6)
        .otherwise(0)
    )


def _uo_points_saps2(uo_l_day: pl.Expr) -> pl.Expr:
    """
    Urine Output (L/day)
    <0.5         11
     0.5-0.99     4
       >=1.0      0
    """
    return (
        pl.when(uo_l_day < 0.5)
        .then(11)
        .when(uo_l_day.is_between(0.5, 1.0, closed="left"))
        .then(4)
        .when(uo_l_day >= 1.0)
        .then(0)
        .otherwise(None)
    )


def _bun_points_saps2(bun: pl.Expr) -> pl.Expr:
    """
    Serum urea nitrogen (mg/dL)
    <28          0
     28-83       6
      >=84      10
    """
    return (
        pl.when(bun < 28)
        .then(0)
        .when(bun.is_between(28, 84, closed="left"))
        .then(6)
        .when(bun >= 84)
        .then(10)
        .otherwise(None)
    )


def _wbc_points_saps2(wbc: pl.Expr) -> pl.Expr:
    """
    WBC (10^3/µL)
    <1.0         12
     1.0-19.9     0
       >=20.0     3
    """
    return (
        pl.when(wbc < 1.0)
        .then(12)
        .when(wbc.is_between(1, 20, closed="left"))
        .then(0)
        .when(wbc >= 20)
        .then(3)
        .otherwise(None)
    )


def _sodium_points_saps2(na: pl.Expr) -> pl.Expr:
    """
    Serum Sodium (mEq/L)
    <125          5
     125-144      0
       >=145      1
    """
    return (
        pl.when(na < 125)
        .then(5)
        .when(na.is_between(125, 145, closed="left"))
        .then(0)
        .when(na >= 145)
        .then(1)
        .otherwise(None)
    )


def _potassium_points_saps2(k: pl.Expr) -> pl.Expr:
    """
    Serum Potassium (mEq/L)
    <3.0         3
     3.0-4.9     0
       >=5.0     3
    """
    return (
        pl.when(k < 3.0)
        .then(3)
        .when(k.is_between(3.0, 5.0, closed="left"))
        .then(0)
        .when(k >= 5.0)
        .then(3)
        .otherwise(None)
    )


def _bicarbonate_points_saps2(hco3: pl.Expr) -> pl.Expr:
    """
    Serum Bicarbonate (mEq/L)
    <15         6
     15-19      3
      >=20      0
    """
    return (
        pl.when(hco3 < 15)
        .then(6)
        .when(hco3.is_between(15, 20, closed="left"))
        .then(3)
        .when(hco3 >= 20)
        .then(0)
        .otherwise(None)
    )


def _bilirubin_points_saps2(bili: pl.Expr) -> pl.Expr:
    """
    Bilirubin (mg/dL)
    <4.0          0
     4.0-5.9      4
       >=6.0      9
    """
    return (
        pl.when(bili < 4.0)
        .then(0)
        .when(bili.is_between(4.0, 6.0, closed="left"))
        .then(4)
        .when(bili >= 6.0)
        .then(9)
        .otherwise(None)
    )


def _gcs_points_saps2(gcs: pl.Expr) -> pl.Expr:
    """
    Glasgow Coma Scale
    <6          26
     6- 8       13
     9-10        7
    11-13        5
    14-15        0
    """
    return (
        pl.when(gcs < 6)
        .then(26)
        .when(gcs.is_between(6, 9, closed="left"))
        .then(13)
        .when(gcs.is_between(9, 11, closed="left"))
        .then(7)
        .when(gcs.is_between(11, 14, closed="left"))
        .then(5)
        .when(gcs >= 14)
        .then(0)
        .otherwise(None)
    )


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

    missing = [name for name, data in required.items() if data is None]
    if missing:
        raise ValueError(
            f"Cannot compute SAPS-II: Missing required datasets: {', '.join(missing)}. "
            f"Ensure they are configured in ~/.reprodICU/PATHS.yaml or provide them explicitly."
        )

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
