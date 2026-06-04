"""
EWS: compute Early Warning Scores (EWS, MEWS, NEWS, NEWS2) in long format directly from raw inputs.

Output columns per row:
- Global ICU Stay ID
- T_0 (seconds from admission used as reference)
- timeframe (0-indexed integer window)
- [Score Name] Score (sum of component points)
- Component points (respiratory rate, heart rate, systolic BP, temperature, GCS, SpO2, supplemental oxygen)

Time is in seconds. Windows determined by floor((time - T_0)/window_size).
Worst-within-window aggregation is applied per component.

SOURCES
-------
- EWS:   Morgan RJM, Williams F, Wright MM. An early warning scoring system for detecting developing critical illness. Clin Intensive Care. 1997;8(2):100.
- MEWS:  Subbe CP, Kruger M, Rutherford P, Gemmel L. Validation of a modified Early Warning Score in medical admissions. QJM. 2001 Oct;94(10):521-6. doi: 10.1093/qjmed/94.10.521. PMID: 11588210.
- NEWS:  Royal College of Physicians. National Early Warning Score (NEWS): Standardising the assessment of acute-illness severity in the NHS. Report of a working party. London: RCP, 2012.
- NEWS2: Royal College of Physicians. National Early Warning Score (NEWS) 2: Standardising the assessment of acute-illness severity in the NHS. Updated report of a working party. London: RCP, 2017.
"""

from typing import Optional

import polars as pl

from ..common import (
    ScoringTable,
    _assign_timeframe,
    _build_base_timeframes,
    _build_t0,
    _get_timeframe_name,
    _optional_time_bounds_filter,
    extract_struct_value,
    get_patient_information,
    get_timeseries_labs,
    get_timeseries_respiratory,
    get_timeseries_vitals,
)

STAY_KEY = "Global ICU Stay ID"
TIME_KEY = "Time Relative to Admission (seconds)"

SECONDS_IN_1H = 60 * 60
SECONDS_IN_1D = 24 * SECONDS_IN_1H


################################################################################
################################################################################
# region common helpers
def _improve_vitals(vitals: pl.LazyFrame) -> pl.LazyFrame:
    return vitals.select(
        STAY_KEY,
        TIME_KEY,
        "Respiratory rate",
        "Temperature",
        pl.coalesce(
            "Invasive systolic arterial pressure",
            "Non-invasive systolic arterial pressure",
        ).alias("Systolic arterial pressure"),
        "Heart rate",
        "Glasgow coma score total",
        "Peripheral oxygen saturation",
    )


def _improve_respiratory(resp: pl.LazyFrame) -> pl.LazyFrame:
    return resp.select(
        STAY_KEY,
        TIME_KEY,
        pl.col("Oxygen gas flow Oxygen delivery system").alias(
            "Supplemental oxygen"
        ),
    )


def _improve_labs(labs: pl.LazyFrame) -> pl.LazyFrame:
    return labs.with_columns(
        extract_struct_value("Carbon dioxide").alias("paCO2")
    ).select(STAY_KEY, TIME_KEY, "paCO2")


def _validate_data(required: dict) -> None:
    missing = [name for name, data in required.items() if data is None]
    if missing:
        raise ValueError(
            f"Missing required datasets: {', '.join(missing)}. "
            f"Ensure they are configured or provide them explicitly."
        )


# endregion

################################################################################
################################################################################


def _avpu_points_standard(gcs: pl.Expr) -> pl.Expr:
    """
    GCS to AVPU mapping. Used by EWS and MEWS.

    GCS       AVPU          Points
    15        Alert         0
    13-14     Voice         1
     9-12     Pain          2
     3- 8     Unresponsive  3

    Source:
    Kelly CA, Upex A, Bateman DN.
    Comparison of consciousness level assessment in the poisoned patient using the alert/verbal/painful/unresponsive scale and the Glasgow Coma Scale.
    Ann Emerg Med. 2004 Aug;44(2):108-13.
    doi: 10.1016/j.annemergmed.2004.03.028. PMID: 15278081.
    """
    return (
        pl.when(gcs == 15)
        .then(0)
        .when(gcs.is_between(13, 14))
        .then(1)
        .when(gcs.is_between(9, 12))
        .then(2)
        .when(gcs.is_between(3, 8))
        .then(3)
        .otherwise(None)
    )


def _avpu_points_news(gcs: pl.Expr) -> pl.Expr:
    """
    GCS to AVPU mapping for NEWS/NEWS2.

    GCS       AVPU          Points
    15        Alert         0
    <15       V, P, or U    3
    """
    return pl.when(gcs == 15).then(0).when(gcs < 15).then(3).otherwise(None)


################################################################################
################################################################################
# region EWS


def _rr_points_ews(rr: pl.Expr) -> pl.Expr:
    return ScoringTable([          # Respiratory rate (bpm) | Points
        (None,    8, "right", 2),  #  ≤8                      2
        (   8,   14, "right", 0),  #   9-14 ................. 0
        (  14,   20, "right", 1),  #  15-20                   1
        (  20,   29, "right", 2),  #  21-29                   2
        (  30, None, "left",  3),  # ≥30                      3
    ]).to_expr(rr) # fmt: skip


def _temp_points_ews(temp: pl.Expr) -> pl.Expr:
    return ScoringTable([            # Temperature (°C) | Points
        (None, 35.0, "neither", 2),  # <35.0              2
        (35.0, 36.5, "right",   1),  #  35.0-36.5         1
        (36.5, 37.4, "right",   0),  #  36.5-37.4 ....... 0
        (37.4, None, "neither", 2),  # >37.4              2
    ]).to_expr(temp) # fmt: skip


def _sbp_points_ews(sbp: pl.Expr) -> pl.Expr:
    return ScoringTable([            # Systolic blood pressure (mmHg) | Points
        (None,   70, "right",   3),  #  ≤70                             3
        (  70,   80, "right",   2),  #   71- 80                         2
        (  80,  100, "right",   1),  #   81-100                         1
        ( 100,  200, "neither", 0),  #  101-199 ....................... 0
        ( 200, None, "left",    2),  # ≥200                             2
    ]).to_expr(sbp) # fmt: skip


def _hr_points_ews(hr: pl.Expr) -> pl.Expr:
    return ScoringTable([            # Heart rate (bpm) | Points
        (None,   40, "right",   3),  #  ≤40               3
        (  40,   50, "right",   1),  #   41- 50           1
        (  50,  100, "right",   0),  #   51-100 ......... 0
        ( 100,  110, "right",   1),  #  101-110           1
        ( 110,  130, "right",   2),  #  111-129           2
        ( 130, None, "neither", 3),  # >130               3
    ]).to_expr(hr) # fmt: skip


def EWS(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_vitals: Optional[pl.LazyFrame] = None,
    *,
    window_size: int = SECONDS_IN_1D,
    t_0: int = 0,
    t_1: Optional[int] = None,
    t_0_per_stay: Optional[pl.LazyFrame] = None,
    timeframe_name: Optional[str] = None,
) -> pl.LazyFrame:
    """
    Compute Early Warning Score (EWS) in long format.

    Output columns:
    - Global ICU Stay ID
    - T_0 (seconds from admission used as reference)
    - timeframe (0-indexed integer window)
    - EWS Score (sum of component points)
    - Component points: rr_points, temp_points, sbp_points, hr_points, avpu_points

    Args:
        patient_information: Patient information LazyFrame.
        timeseries_vitals: Vital signs LazyFrame.
        window_size: Size of the timeframe window in seconds (default: 24h).
        t_0: Global time anchor in seconds from admission (default: 0).
        t_1: Global end time in seconds from admission (optional).
        t_0_per_stay: LazyFrame with stay-specific T_0 (optional).
        timeframe_name: Custom name for the timeframe column (optional).

    Returns:
        pl.LazyFrame: EWS scores in long format.
    """
    # region data loading
    if patient_information is None:
        patient_information = get_patient_information()
    if timeseries_vitals is None:
        timeseries_vitals = get_timeseries_vitals()

    _validate_data(
        {
            "patient_information": patient_information,
            "timeseries_vitals": timeseries_vitals,
        }
    )

    vitals = _improve_vitals(timeseries_vitals.lazy())

    patient_information = patient_information.lazy()
    ALL_STAYS = patient_information.select(STAY_KEY)
    ALL_STAYS_T0 = _build_t0(ALL_STAYS, t_0_per_stay=t_0_per_stay, t_0=t_0)
    timeframe_name = _get_timeframe_name(
        timeframe_name, window_size, t_0, t_0_per_stay
    )
    # endregion

    # region component scoring
    vitals_tf = (
        vitals.join(ALL_STAYS_T0, on=STAY_KEY, how="inner")
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window_size))
        .group_by(STAY_KEY, "timeframe")
        .agg(
            _rr_points_ews(pl.col("Respiratory rate")).max().alias("rr_points"),
            _temp_points_ews(pl.col("Temperature")).max().alias("temp_points"),
            _sbp_points_ews(pl.col("Systolic arterial pressure")).max().alias("sbp_points"),
            _hr_points_ews(pl.col("Heart rate")).max().alias("hr_points"),
            _avpu_points_standard(pl.col("Glasgow coma score total")).max().alias("avpu_points"),
        )
    ) # fmt: skip
    # endregion

    # region union of all (stay,timeframe)
    base = _build_base_timeframes(ALL_STAYS_T0, patient_information, window_size) # fmt: skip

    # region assemble
    return (
        base.join(vitals_tf, on=[STAY_KEY, "timeframe"], how="left")
        .filter(
            _optional_time_bounds_filter("timeframe", window_size, t_0, t_1)
        )
        .select(
            STAY_KEY,
            "T_0",
            pl.col("timeframe").alias(timeframe_name),
            pl.sum_horizontal(pl.exclude(STAY_KEY, "T_0", "timeframe")).alias(
                "EWS Score"
            ),
            pl.all().exclude(STAY_KEY, "T_0", timeframe_name),
        )
        .sort(STAY_KEY, timeframe_name)
    )


# endregion


################################################################################
################################################################################
# region MEWS


def _rr_points_mews(rr: pl.Expr) -> pl.Expr:
    return ScoringTable([            # Respiratory rate (bpm) | Points
        (None,    8, "right",   3),  #  ≤8                      3
        (   8,   11, "neither", 1),  #   9-10                   1
        (  11,   20, "left",    0),  #  12-20 ................. 0
        (  20,   25, "left",    2),  #  21-24                   2
        (  25, None, "left",    3),  # ≥25                      3
    ]).to_expr(rr) # fmt: skip


def _temp_points_mews(temp: pl.Expr) -> pl.Expr:
    return ScoringTable([            # Temperature (°C) | Points
        (None, 35.0, "neither", 2),  # <35.0              2
        (35.0, 38.5, "both",    0),  #  35.0-38.5 ....... 0
        (38.5, None, "neither", 2),  # >38.5              2
    ]).to_expr(temp) # fmt: skip


def MEWS(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_vitals: Optional[pl.LazyFrame] = None,
    *,
    window_size: int = SECONDS_IN_1D,
    t_0: int = 0,
    t_1: Optional[int] = None,
    t_0_per_stay: Optional[pl.LazyFrame] = None,
    timeframe_name: Optional[str] = None,
) -> pl.LazyFrame:
    """
    Compute Modified Early Warning Score (MEWS) in long format.

    Output columns:
    - Global ICU Stay ID
    - T_0 (seconds from admission used as reference)
    - timeframe (0-indexed integer window)
    - MEWS Score (sum of component points)
    - Component points: rr_points, temp_points, sbp_points, hr_points, avpu_points

    Args:
        patient_information: Patient information LazyFrame.
        timeseries_vitals: Vital signs LazyFrame.
        window_size: Size of the timeframe window in seconds (default: 24h).
        t_0: Global time anchor in seconds from admission (default: 0).
        t_1: Global end time in seconds from admission (optional).
        t_0_per_stay: LazyFrame with stay-specific T_0 (optional).
        timeframe_name: Custom name for the timeframe column (optional).

    Returns:
        pl.LazyFrame: MEWS scores in long format.
    """
    # region data loading
    if patient_information is None:
        patient_information = get_patient_information()
    if timeseries_vitals is None:
        timeseries_vitals = get_timeseries_vitals()

    _validate_data(
        {
            "patient_information": patient_information,
            "timeseries_vitals": timeseries_vitals,
        }
    )

    vitals = _improve_vitals(timeseries_vitals.lazy())

    patient_information = patient_information.lazy()
    ALL_STAYS = patient_information.select(STAY_KEY)
    ALL_STAYS_T0 = _build_t0(ALL_STAYS, t_0_per_stay=t_0_per_stay, t_0=t_0)
    timeframe_name = _get_timeframe_name(
        timeframe_name, window_size, t_0, t_0_per_stay
    )
    # endregion

    # region component scoring
    vitals_tf = (
        vitals.join(ALL_STAYS_T0, on=STAY_KEY, how="inner")
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window_size))
        .group_by(STAY_KEY, "timeframe")
        .agg(
            _rr_points_mews(pl.col("Respiratory rate")).max().alias("rr_points"),
            _temp_points_mews(pl.col("Temperature")).max().alias("temp_points"),
            _sbp_points_ews(pl.col("Systolic arterial pressure")).max().alias("sbp_points"),
            _hr_points_ews(pl.col("Heart rate")).max().alias("hr_points"),
            _avpu_points_standard(pl.col("Glasgow coma score total")).max().alias("avpu_points"),
        )
    ) # fmt: skip
    # endregion

    # region union of all (stay,timeframe)
    base = _build_base_timeframes(ALL_STAYS_T0, patient_information, window_size) # fmt: skip

    # region assemble
    return (
        base.join(vitals_tf, on=[STAY_KEY, "timeframe"], how="left")
        .filter(
            _optional_time_bounds_filter("timeframe", window_size, t_0, t_1)
        )
        .select(
            STAY_KEY,
            "T_0",
            pl.col("timeframe").alias(timeframe_name),
            pl.sum_horizontal(pl.exclude(STAY_KEY, "T_0", "timeframe")).alias(
                "MEWS Score"
            ),
            pl.exclude(STAY_KEY, "T_0", timeframe_name),
        )
        .sort(STAY_KEY, timeframe_name)
    )


# endregion


################################################################################
################################################################################
# region NEWS


def _spo2_points_news(spo2: pl.Expr) -> pl.Expr:
    return ScoringTable([            # SpO2 (%) | Points
        (None,   91, "right",   3),  # ≤91        3
        (  91,   94, "neither", 2),  #  92-93     2
        (  94,   96, "left",    1),  #  94-95     1
        (  96, None, "left",    0),  # ≥96 ...... 0
    ]).to_expr(spo2) # fmt: skip


def _suppo2_points_news(suppo2: pl.Expr) -> pl.Expr:
    """
    Supplemental oxygen points for NEWS and NEWS2.

    Oxygen      Points
    Yes         2
    No          0
    """
    return pl.when(suppo2.cast(pl.Boolean)).then(2).otherwise(0)


def _temp_points_news(temp: pl.Expr) -> pl.Expr:
    return ScoringTable([            # Temperature (°C) | Points
        (None, 35.0, "right",   3),  # ≤35.0              3
        (35.0, 36.0, "right",   1),  #  35.1-36.0         1
        (36.0, 38.0, "right",   0),  #  36.1-38.0 ....... 0
        (38.0, 39.0, "both",    1),  #  38.1-39.0         1
        (39.0, None, "neither", 2),  # ≥39.1              2
    ]).to_expr(temp) # fmt: skip


def _sbp_points_news(sbp: pl.Expr) -> pl.Expr:
    return ScoringTable([            # Systolic blood pressure (mmHg) | Points
        (None,   90, "right",   3),  #  ≤90                             3
        (  90,  100, "right",   2),  #   91-100                         2
        ( 100,  110, "right",   1),  #  101-110                         1
        ( 110,  220, "neither", 0),  #  111-219 ....................... 0
        ( 220, None, "left",    3),  # ≥220                             3
    ]).to_expr(sbp) # fmt: skip


def _hr_points_news(hr: pl.Expr) -> pl.Expr:
    return ScoringTable([            # Heart rate (bpm) | Points
        (None,   40, "right",   3),  #  ≤40               3
        (  40,   50, "right",   1),  #   41- 50           1
        (  50,   90, "right",   0),  #   51- 90 ......... 0
        (  90,  110, "right",   1),  #   91-110           1
        ( 110,  130, "right",   2),  #  111-130           2
        ( 130, None, "neither", 3),  # ≥131               3
    ]).to_expr(hr) # fmt: skip


def NEWS(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_vitals: Optional[pl.LazyFrame] = None,
    timeseries_resp: Optional[pl.LazyFrame] = None,
    *,
    window_size: int = SECONDS_IN_1D,
    t_0: int = 0,
    t_1: Optional[int] = None,
    t_0_per_stay: Optional[pl.LazyFrame] = None,
    timeframe_name: Optional[str] = None,
) -> pl.LazyFrame:
    """
    Compute National Early Warning Score (NEWS) in long format.

    Output columns:
    - Global ICU Stay ID
    - T_0 (seconds from admission used as reference)
    - timeframe (0-indexed integer window)
    - NEWS Score (sum of component points)
    - Component points: rr_points, spo2_points, suppo2_points, temp_points, sbp_points, hr_points, avpu_points

    Args:
        patient_information: Patient information LazyFrame.
        timeseries_vitals: Vital signs LazyFrame.
        timeseries_resp: Respiratory LazyFrame (for supplemental oxygen).
        window_size: Size of the timeframe window in seconds (default: 24h).
        t_0: Global time anchor in seconds from admission (default: 0).
        t_1: Global end time in seconds from admission (optional).
        t_0_per_stay: LazyFrame with stay-specific T_0 (optional).
        timeframe_name: Custom name for the timeframe column (optional).

    Returns:
        pl.LazyFrame: NEWS scores in long format.
    """
    # region data loading
    if patient_information is None:
        patient_information = get_patient_information()
    if timeseries_vitals is None:
        timeseries_vitals = get_timeseries_vitals()
    if timeseries_resp is None:
        timeseries_resp = get_timeseries_respiratory()

    _validate_data(
        {
            "patient_information": patient_information,
            "timeseries_vitals": timeseries_vitals,
            "timeseries_resp": timeseries_resp,
        }
    )

    vitals = _improve_vitals(timeseries_vitals.lazy())
    resp = _improve_respiratory(timeseries_resp.lazy())

    patient_information = patient_information.lazy()
    ALL_STAYS = patient_information.select(STAY_KEY)
    ALL_STAYS_T0 = _build_t0(ALL_STAYS, t_0_per_stay=t_0_per_stay, t_0=t_0)
    timeframe_name = _get_timeframe_name(
        timeframe_name, window_size, t_0, t_0_per_stay
    )
    # endregion

    # region component scoring
    vitals_tf = (
        vitals.join(ALL_STAYS_T0, on=STAY_KEY, how="inner")
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window_size))
        .group_by(STAY_KEY, "timeframe")
        .agg(
            _rr_points_mews(pl.col("Respiratory rate")).max().alias("rr_points"),
            _spo2_points_news(pl.col("Peripheral oxygen saturation")).max().alias("spo2_points"),
            _temp_points_news(pl.col("Temperature")).max().alias("temp_points"),
            _sbp_points_news(pl.col("Systolic arterial pressure")).max().alias("sbp_points"),
            _hr_points_news(pl.col("Heart rate")).max().alias("hr_points"),
            _avpu_points_news(pl.col("Glasgow coma score total")).max().alias("avpu_points"),
        )
    ) # fmt: skip

    resp_tf = (
        resp.join(ALL_STAYS_T0, on=STAY_KEY, how="inner")
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window_size))
        .group_by(STAY_KEY, "timeframe")
        .agg(
            _suppo2_points_news(pl.col("Supplemental oxygen"))
            .max()
            .alias("suppo2_points"),
        )
    )
    # endregion

    # region union of all (stay,timeframe)
    base = _build_base_timeframes(ALL_STAYS_T0, patient_information, window_size) # fmt: skip

    # region assemble
    return (
        base.join(vitals_tf, on=[STAY_KEY, "timeframe"], how="left")
        .join(resp_tf, on=[STAY_KEY, "timeframe"], how="left")
        .filter(
            _optional_time_bounds_filter("timeframe", window_size, t_0, t_1)
        )
        .select(
            STAY_KEY,
            "T_0",
            pl.col("timeframe").alias(timeframe_name),
            pl.sum_horizontal(pl.exclude(STAY_KEY, "T_0", "timeframe")).alias(
                "NEWS Score"
            ),
            pl.exclude(STAY_KEY, "T_0", timeframe_name),
        )
        .sort(STAY_KEY, timeframe_name)
    )


# endregion


################################################################################
################################################################################
# region NEWS2


def _spo2_points_news2(
    spo2: pl.Expr, suppo2: pl.Expr, paco2: pl.Expr
) -> pl.Expr:
    """
    Peripheral oxygen saturation points for NEWS2.

    Scale 1 (Normal):
    SpO2 (%)    Points
    ≤91         3
     92-93      2
     94-95      1
    ≥96         0

    Scale 2 (Hypercapnic Respiratory Failure, paCO2 > 50 mmHg):
    SpO2 (%)    Points
    ≤83         3
     84-85      2
     86-87      1
     88-92      0
    ≥93 (RA)    0
     93-94 (O2) 1
     95-96 (O2) 2
    ≥97 (O2)    3
    """
    scale1 = (
        pl.when(spo2 <= 91).then(3)
          .when(spo2.is_between(91, 94, closed="right")).then(2)
          .when(spo2.is_between(93, 96, closed="none" )).then(1)
          .when(spo2 >= 96).then(0)
          .otherwise(None)
    ) # fmt: skip
    scale2 = (
        pl.when(spo2 <= 83).then(3)
          .when(spo2.is_between(83, 86, closed="right")).then(2)
          .when(spo2.is_between(85, 88, closed="right")).then(1)
          .when(spo2.is_between(87, 93, closed="none" )).then(0)
        .when(suppo2.cast(pl.Boolean))
        .then(
            pl.when(spo2.is_between(92, 95, closed="right")).then(1)
              .when(spo2.is_between(94, 97, closed="none" )).then(2)
              .when(spo2 >= 97).then(3)
              .otherwise(None)
        )
        .when(spo2 >= 93).then(0)
        .otherwise(None)
    ) # fmt: skip
    return (
        pl.when(paco2.is_null() | (paco2 <= 50)).then(scale1).otherwise(scale2)
    )


def NEWS2(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_vitals: Optional[pl.LazyFrame] = None,
    timeseries_resp: Optional[pl.LazyFrame] = None,
    timeseries_labs: Optional[pl.LazyFrame] = None,
    *,
    window_size: int = SECONDS_IN_1D,
    t_0: int = 0,
    t_1: Optional[int] = None,
    t_0_per_stay: Optional[pl.LazyFrame] = None,
    timeframe_name: Optional[str] = None,
) -> pl.LazyFrame:
    """
    Compute National Early Warning Score 2 (NEWS2) in long format.

    Output columns:
    - Global ICU Stay ID
    - T_0 (seconds from admission used as reference)
    - timeframe (0-indexed integer window)
    - NEWS2 Score (sum of component points)
    - Component points: rr_points, spo2_points, suppo2_points, temp_points, sbp_points, hr_points, avpu_points

    Args:
        patient_information: Patient information LazyFrame.
        timeseries_vitals: Vital signs LazyFrame.
        timeseries_resp: Respiratory LazyFrame (for supplemental oxygen).
        timeseries_labs: Labs LazyFrame (for paCO2).
        window_size: Size of the timeframe window in seconds (default: 24h).
        t_0: Global time anchor in seconds from admission (default: 0).
        t_1: Global end time in seconds from admission (optional).
        t_0_per_stay: LazyFrame with stay-specific T_0 (optional).
        timeframe_name: Custom name for the timeframe column (optional).

    Returns:
        pl.LazyFrame: NEWS2 scores in long format.
    """
    # region data loading
    if patient_information is None:
        patient_information = get_patient_information()
    if timeseries_vitals is None:
        timeseries_vitals = get_timeseries_vitals()
    if timeseries_resp is None:
        timeseries_resp = get_timeseries_respiratory()
    if timeseries_labs is None:
        timeseries_labs = get_timeseries_labs()

    _validate_data(
        {
            "patient_information": patient_information,
            "timeseries_vitals": timeseries_vitals,
            "timeseries_resp": timeseries_resp,
            "timeseries_labs": timeseries_labs,
        }
    )

    vitals = _improve_vitals(timeseries_vitals.lazy())
    resp = _improve_respiratory(timeseries_resp.lazy())
    labs = _improve_labs(timeseries_labs.lazy())

    patient_information = patient_information.lazy()
    ALL_STAYS = patient_information.select(STAY_KEY)
    ALL_STAYS_T0 = _build_t0(ALL_STAYS, t_0_per_stay=t_0_per_stay, t_0=t_0)
    timeframe_name = _get_timeframe_name(
        timeframe_name, window_size, t_0, t_0_per_stay
    )
    # endregion

    # region component scoring
    vitals_tf = (
        vitals.join(ALL_STAYS_T0, on=STAY_KEY, how="inner")
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window_size))
        .group_by(STAY_KEY, "timeframe")
        .agg(
            _rr_points_mews(pl.col("Respiratory rate")).max().alias("rr_points"),
            _temp_points_news(pl.col("Temperature")).max().alias("temp_points"),
            _sbp_points_news(pl.col("Systolic arterial pressure")).max().alias("sbp_points"),
            _hr_points_news(pl.col("Heart rate")).max().alias("hr_points"),
            _avpu_points_news(pl.col("Glasgow coma score total")).max().alias("avpu_points"),
        )
    ) # fmt: skip

    resp_tf = (
        vitals.select(STAY_KEY, TIME_KEY, "Peripheral oxygen saturation")
        .drop_nulls()
        .join(resp.drop_nulls(), on=[STAY_KEY, TIME_KEY], how="left")
        .join(labs.drop_nulls(), on=[STAY_KEY, TIME_KEY], how="left")
        .join(ALL_STAYS_T0, on=STAY_KEY, how="inner")
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window_size))
        .group_by(STAY_KEY, "timeframe")
        .agg(
            _spo2_points_news2(
                pl.col("Peripheral oxygen saturation"),
                pl.col("Supplemental oxygen"),
                pl.col("paCO2"),
            )
            .max()
            .alias("spo2_points"),
            _suppo2_points_news(pl.col("Supplemental oxygen"))
            .max()
            .alias("suppo2_points"),
        )
    )
    # endregion

    # region union of all (stay,timeframe)
    base = _build_base_timeframes(ALL_STAYS_T0, patient_information, window_size) # fmt: skip

    # region assemble
    return (
        base.join(vitals_tf, on=[STAY_KEY, "timeframe"], how="left")
        .join(resp_tf, on=[STAY_KEY, "timeframe"], how="left")
        .filter(
            _optional_time_bounds_filter("timeframe", window_size, t_0, t_1)
        )
        .select(
            STAY_KEY,
            "T_0",
            pl.col("timeframe").alias(timeframe_name),
            pl.sum_horizontal(pl.exclude(STAY_KEY, "T_0", "timeframe")).alias(
                "NEWS2 Score"
            ),
            pl.exclude(STAY_KEY, "T_0", timeframe_name),
        )
        .sort(STAY_KEY, timeframe_name)
    )


# endregion

__all__ = ["EWS", "MEWS", "NEWS", "NEWS2"]
