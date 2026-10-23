"""
SOFA: compute SOFA in long format directly from raw inputs.

Output columns per row:
- Global ICU Stay ID
- T_0 (seconds from admission used as reference)
- timeframe (integer window index counted from T_0)
- Respiratory points (by P/F ratio)
- Coagulation points (by platelets)
- Liver points (by bilirubin)
- Cardiovascular points (by MAP & vasoactive meds)
    - includes both MAP and vasoactive meds subscores
- Central nervous system points (by GCS)
- Renal points (by creatinine & urine output)
    - includes both creatinine and urine output subscores
- SOFA Score (sum of organ points)

Time is in seconds. Windows determined by floor((time - T_0)/window_size).
Worst-within-window aggregation is applied per organ.

SOURCES
-------
- Vincent JL, Moreno R, Takala J, Willatts S, De Mendonça A, Bruining H, Reinhart CK, Suter PM, Thijs LG.
  The SOFA (Sepsis-related Organ Failure Assessment) score to describe organ dysfunction/failure. On behalf of the Working Group on Sepsis-Related Problems of the European Society of Intensive Care Medicine.
  Intensive Care Med. 1996 Jul;22(7):707-10.
  doi: 10.1007/BF01709751. PMID: 8844239.
"""

from typing import Optional

import polars as pl

from ..base import (
    SECONDS_IN_1D,
    SECONDS_IN_1H,
    SECONDS_IN_1W,
    STAY_KEY,
    TIME_KEY,
    Window,
    _assemble,
    _assign_timeframe,
    _finalize,
    _null_if_incomplete_pre_icu,
    _window_context,
    with_windows,
)
from ..clinical.neurological.GCS import GCS_NT_CARRY_FORWARD
from ..clinical.pharmocological.ALIGNED_UNITS import ALIGNED_UNITS
from ..clinical.renal.URINE_OUTPUT import URINE_OUTPUT
from ..common import (
    FILTER_NOT_NULL,
    ScoringTable,
    _validate_required_data,
    extract_struct_value,
    get_medications,
    get_patient_information,
    get_timeseries_intakeoutput,
    get_timeseries_labs,
    get_timeseries_respiratory,
    get_timeseries_vitals,
    get_ventilation,
)
from ..core import BLOOD_PRESSURES
from ..FIX_WINDOW_BORDERS import PROJECT_TO_WINDOWS, TIME_WEIGHTED_MEAN_RATE
from ..laboratory.oxygenation.PF_RATIO import PaO2_FiO2_RATIO


################################################################################
################################################################################
# region data helpers
def _improve_vitals(vitals: pl.LazyFrame) -> pl.LazyFrame:
    return (
        BLOOD_PRESSURES(timeseries_vitals=vitals)
        .rename({"MAP": "Mean arterial pressure"})
        .pipe(FILTER_NOT_NULL, ["Mean arterial pressure", "Glasgow coma score total"])
    )


def _improve_vitals_quick(vitals: pl.LazyFrame) -> pl.LazyFrame:
    return (
        BLOOD_PRESSURES(timeseries_vitals=vitals)
        .rename({"SBP": "Systolic arterial pressure"})
        .pipe(FILTER_NOT_NULL, ["Systolic arterial pressure", "Respiratory rate", "Glasgow coma score total"])
    )


def _improve_labs(labs: pl.LazyFrame) -> pl.LazyFrame:
    return labs.with_columns(
        extract_struct_value("Platelets").alias("Platelets"),
        extract_struct_value("Creatinine", ["Serum", "Blood"]).alias("Creatinine"),
        extract_struct_value("Bilirubin",  ["Serum", "Blood"]).alias("Bilirubin"),
    ).pipe(FILTER_NOT_NULL, ["Platelets", "Creatinine", "Bilirubin"])


################################################################################
################################################################################
# region organ scoring helpers
def _pf_ratio_points(pf_ratio: pl.Expr, ventilated: pl.Expr) -> pl.Expr:
    """
    PaO2/FiO2, mmHg
    ≥400                                   0
     300-399                              +1
     200-299                              +2
    ≤199 and NOT mechanically ventilated  +2
     100-199 and mechanically ventilated  +3
    <100 and mechanically ventilated      +4
    """
    return (
        pl.when((pf_ratio >= 400) | pf_ratio.is_null())
        .then(0)
        .when(pf_ratio.is_between(300, 400, closed="left"))
        .then(1)
        .when(pf_ratio.is_between(200, 300, closed="left"))
        .then(2)
        .when((pf_ratio < 200) & (~ventilated))
        .then(2)
        .when(pf_ratio.is_between(100, 200, closed="left") & ventilated)
        .then(3)
        .when((pf_ratio < 100) & ventilated)
        .then(4)
        .otherwise(None)
    )


def _sf_ratio_points(pf_ratio: pl.Expr, ventilated: pl.Expr) -> pl.Expr:
    """
    spO2/FiO2, mmHg
    ≥400                                   0
     300-399                              +1
     200-299                              +2
    ≤199 and NOT mechanically ventilated  +2
     100-199 and mechanically ventilated  +3
    <100 and mechanically ventilated      +4
    """
    return (
        pl.when((pf_ratio >= 400) | pf_ratio.is_null())
        .then(0)
        .when(pf_ratio.is_between(300, 400, closed="left"))
        .then(1)
        .when(pf_ratio.is_between(200, 300, closed="left"))
        .then(2)
        .when((pf_ratio < 200) & (~ventilated))
        .then(2)
        .when(pf_ratio.is_between(100, 200, closed="left") & ventilated)
        .then(3)
        .when((pf_ratio < 100) & ventilated)
        .then(4)
        .otherwise(None)
    )


def _platelet_points(platelets_value: pl.Expr) -> pl.Expr:
    return ScoringTable([            # Platelets (×10^3/µL) | Points
        ( 150, None, "left",    0),  # ≥150 ................. 0
        ( 100,  150, "left",    1),  #  100-149               1
        (  50,  100, "left",    2),  #   50- 99               2
        (  20,   50, "left",    3),  #   20- 49               3
        (None,   20, "neither", 4),  #  <20                   4
    ]).to_expr(platelets_value) # fmt: skip


def _gcs_points(gcs: pl.Expr) -> pl.Expr:
    return ScoringTable([            # Glasgow Coma Scale | Points
        (  15, None, "left",    0),  # 15 ................. 0
        (  13,   15, "left",    1),  # 13-14                1
        (  10,   13, "left",    2),  # 10-12                2
        (   6,   10, "left",    3),  #  6- 9                3
        (None,    6, "neither", 4),  # <6                   4
    ]).to_expr(gcs) # fmt: skip


def _bilirubin_points(bili: pl.Expr) -> pl.Expr:
    return ScoringTable([            # Bilirubin (mg/dL) | Points
        (None,  1.2, "neither", 0),  #  <1.2 ............. 0
        ( 1.2,  2.0, "left",    1),  #   1.2- 1.9          1
        ( 2.0,  6.0, "left",    2),  #   2.0- 5.9          2
        ( 6.0, 12.0, "left",    3),  #   6.0-11.9          3
        (12.0, None, "left",    4),  # ≥12.0               4
    ]).to_expr(bili) # fmt: skip


def _creatinine_points(crea: pl.Expr) -> pl.Expr:
    return ScoringTable([            # Creatinine (mg/dL) | Points
        (None,  1.2, "neither", 0),  # <1.2 ............... 0
        ( 1.2,  2.0, "left",    1),  #  1.2-1.9             1
        ( 2.0,  3.5, "left",    2),  #  2.0-3.4             2
        ( 3.5,  5.0, "left",    3),  #  3.5-4.9             3
        ( 5.0, None, "left",    4),  # ≥5.0                 4
    ]).to_expr(crea) # fmt: skip


def _uo_points(uo_ml: pl.Expr) -> pl.Expr:
    return ScoringTable([            # Urine output (mL/day) | Points
        (None,  200, "neither", 4),  # <200                    4
        ( 200,  500, "left",    3),  #  200-499                3
        ( 500, None, "left",    0),  # ≥500 .................. 0
    ]).to_expr(uo_ml) # fmt: skip


def _map_points(map_val: pl.Expr) -> pl.Expr:
    return ScoringTable([            # Mean Arterial Pressure (mmHg) | Points
        (None,   70, "neither", 1),  # <70                             1
        (  70, None, "left",    0),  # ≥70 ........................... 0
    ]).to_expr(map_val) # fmt: skip


def _vasopressor_points(
    dopa: pl.Expr, dobu: pl.Expr, epi: pl.Expr, norepi: pl.Expr
) -> pl.Expr:
    """
    Administration of vasoactive agents required (mcg/kg/min):
    DOPamine  ≤5 or DOBUTamine (any dose)                     +2
    DOPamine  >5, EPINEPHrine ≤0.1, or norEPINEPHrine ≤0.1    +3
    DOPamine >15, EPINEPHrine >0.1, or norEPINEPHrine >0.1    +4
    """
    # Order matters: check the highest tier first, then lower tiers
    return (
        pl.when((dopa > 15) | (epi > 0.1) | (norepi > 0.1))
        .then(4)
        .when(
            (dopa > 5)
            | ((epi > 0) & (epi <= 0.1))
            | ((norepi > 0) & (norepi <= 0.1))
        )
        .then(3)
        .when(((dopa > 0) & (dopa <= 5)) | (dobu > 0))
        .then(2)
        .otherwise(0)
    )


################################################################################
################################################################################
# region qSOFA
@with_windows
def qSOFA(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_vitals: Optional[pl.LazyFrame] = None,
    *,
    window: Window,
) -> pl.LazyFrame:
    """
    Compute qSOFA score with automatic dataset loading.

    All data parameters are optional and will be automatically loaded from the
    package datasets if not provided. This makes it convenient for quick analysis
    while maintaining flexibility for custom data.

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient information dataset. Loaded automatically if None.
        timeseries_vitals : pl.LazyFrame, optional
            Timeseries vitals data. Loaded automatically if None.

    Sources
    -------

    - original source:
        Seymour CW, Liu VX, Iwashyna TJ, Brunkhorst FM, Rea TD, Scherag A, Rubenfeld G, Kahn JM, Shankar-Hari M, Singer M, Deutschman CS, Escobar GJ, Angus DC. Assessment of Clinical Criteria for Sepsis: For the Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). JAMA. 2016 Feb 23;315(8):762-74. doi: 10.1001/jama.2016.0288. Erratum in: JAMA. 2016 May 24-31;315(20):2237. doi: 10.1001/jama.2016.5850. PMID: 26903335; PMCID: PMC5433435.

    Returns
    -------
        pl.LazyFrame
            qSOFA scores with all subscore components
    """
    # Load defaults if not provided
    if patient_information is None:
        patient_information = get_patient_information()
    if timeseries_vitals is None:
        timeseries_vitals = get_timeseries_vitals()

    # Validate all required data is available
    required = {
        "patient_information": patient_information,
        "timeseries_vitals": timeseries_vitals,
    }
    _validate_required_data("qSOFA", required)

    # Strict original column names

    # Vitals
    vitals = _improve_vitals_quick(timeseries_vitals.lazy())
    sbp_col = "Systolic arterial pressure"
    rr_col = "Respiratory rate"
    gcs_col = "Glasgow coma score total"

    # Base frames
    patient_information = patient_information.lazy()
    ctx = _window_context(patient_information, window)

    # vitals (SBP, RR & GCS)
    vitals_tf = (
        vitals.select(STAY_KEY, TIME_KEY, sbp_col, rr_col, gcs_col)
        .join(ctx.all_stays_t0, on=STAY_KEY, how="inner")
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window.window_size))
        .group_by(STAY_KEY, "timeframe")
        # 1 point each for systolic hypotension, tachypnea, or altered mentation
        .agg(
            (pl.col(sbp_col) <= 100).cast(int).max().alias("sbp_points"),
            (pl.col(rr_col) >= 22).cast(int).max().alias("rr_points"),
            (pl.col(gcs_col) != 15).cast(int).max().alias("gcs_points"),
        )
    )

    # assemble
    out = _assemble(ctx, vitals_tf)

    return _finalize(
        out.with_columns(
            _null_if_incomplete_pre_icu(
                pl.sum_horizontal(
                pl.col("sbp_points"),
                pl.col("rr_points"),
                pl.col("gcs_points"),
                ignore_nulls=True,
            ),
                pl.col(["sbp_points", "rr_points", "gcs_points"]),
                window.window_size,
            ).alias("qSOFA Score")
        ),
        ctx,
        "qSOFA Score",
        pl.col("sbp_points").alias("Systolic hypotension (<=100 mmHg)"),
        pl.col("rr_points").alias("Tachypnea (>=22/min)"),
        pl.col("gcs_points").alias("Altered mentation (GCS < 15)"),
        rest=False,
    )


def _forward_fill_within(
    cols: list[str],
    max_windows: int,
    stay_key: str = "Global ICU Stay ID",
    icu_start: Optional[pl.Expr] = None,
) -> list[pl.Expr]:
    """
    Forward-fill `cols` for at most `max_windows` timeframes after the last value.

    Rows are sparse before ICU admission, so a limit counted in rows would carry
    values across gaps. The limit is the distance in timeframes. With `icu_start`
    (first timeframe of the ICU stay), values from before it are not carried in.
    """
    tf = pl.col("timeframe")
    out = []
    for col in cols:
        last = pl.when(pl.col(col).is_not_null()).then(tf).forward_fill()
        can_fill = pl.col(col).is_null() & (tf - last).le(max_windows)
        if icu_start is not None:
            can_fill = can_fill & last.ge(icu_start)
        out.append(
            pl.when(can_fill)
            .then(pl.col(col).forward_fill())
            .otherwise(pl.col(col))
            .over(partition_by=stay_key, order_by="timeframe")
            .alias(col)
        )
    return out


def _organ_score_forward_fill(
    out: pl.LazyFrame,
    *,
    fast_cols: list[str],
    slow_cols: list[str],
    uo_col: str,
    window_size: int,
    stay_key: str = "Global ICU Stay ID",
    fill_from_pre_icu: bool = True,
) -> pl.LazyFrame:
    """
    Apply forward-fill to organ score columns with different time horizons.

    Fast columns (6h): PF/respiratory, GCS, MAP, vasopressor points.
    Slow columns (1 week): platelet, bilirubin, creatinine points.
    Urine output (24h): persistent within day, reset daily.
    The horizons are distances in timeframes, so they also hold across the sparse
    timeframes before ICU admission. With fill_from_pre_icu=False, values from before
    ICU admission are not carried into the ICU stay.
    """
    # Convert the horizons from hours to windows
    windows_per_day = max(1, SECONDS_IN_1D // window_size)
    icu_start = None if fill_from_pre_icu else (-pl.col("T_0")).floordiv(window_size)
    return out.with_columns(
        *_forward_fill_within(fast_cols, (6 * SECONDS_IN_1H) // window_size, stay_key, icu_start),
        *_forward_fill_within(slow_cols, SECONDS_IN_1W // window_size, stay_key, icu_start),
        pl.col(uo_col)
        .forward_fill()
        .backward_fill()
        .over(partition_by=[stay_key, pl.col("timeframe").floordiv(windows_per_day)]),
    ) # fmt: skip


################################################################################
################################################################################
# region SOFA
@with_windows
def SOFA(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_vitals: Optional[pl.LazyFrame] = None,
    timeseries_labs: Optional[pl.LazyFrame] = None,
    timeseries_resp: Optional[pl.LazyFrame] = None,
    timeseries_inout: Optional[pl.LazyFrame] = None,
    medications: Optional[pl.LazyFrame] = None,
    ventilation: Optional[pl.LazyFrame] = None,
    *,
    forward_fill: bool = True,
    _fill_from_pre_icu: bool = True,  # internal: used by SEPSIS
    window: Window,
) -> pl.LazyFrame:
    """
    Compute SOFA score with automatic dataset loading.

    All data parameters are optional and will be automatically loaded from the
    package datasets if not provided. This makes it convenient for quick analysis
    while maintaining flexibility for custom data.

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient information dataset. Loaded automatically if None.
        timeseries_vitals : pl.LazyFrame, optional
            Timeseries vitals data. Loaded automatically if None.
        timeseries_labs : pl.LazyFrame, optional
            Timeseries labs data. Loaded automatically if None.
        timeseries_resp : pl.LazyFrame, optional
            Timeseries respiratory data. Loaded automatically if None.
        timeseries_inout : pl.LazyFrame, optional
            Timeseries intake/output data. Loaded automatically if None.
        medications : pl.LazyFrame, optional
            Medications data. Loaded automatically if None.
        ventilation : pl.LazyFrame, optional
            Ventilation data. Loaded automatically if None.
        forward_fill : bool, optional
            Carry the last organ points forward over windows without a measurement: up to
            6 hours for respiration, nervous system and the cardiovascular points, up to a
            week for platelets, bilirubin and creatinine, and within the same day for urine
            output. Defaults to True.
        _fill_from_pre_icu : bool, optional
            Internal, used by SEPSIS. If False, values from before ICU admission do not fill
            windows of the ICU stay. Defaults to True.

    Sources
    -------

    - original source:
        Vincent, J.-L., Moreno, R., Takala, J., Willatts, S., De Mendonça, A., Bruining, H., … Thijs, L. G. (1996). The SOFA (Sepsis-related Organ Failure Assessment) score to describe organ dysfunction/failure. Intensive Care Medicine, 22(7), 707–710. doi:10.1007/bf01709751
    - modified for usage without mechanical ventilation data following:
        Jones, A. E., Trzeciak, S., & Kline, J. A. (2009). The Sequential Organ Failure Assessment score for predicting outcome in patients with severe sepsis and evidence of hypoperfusion at the time of emergency department presentation. Critical Care Medicine, 37(5), 1649–1654. doi:10.1097/CCM.0b013e31819def97
            * source of the substitution:
                Pandharipande PP, Sanders N, Jacques P, et al. Calculating SOFA scores when arterial blood gasses are not available: Validating SpO2/FIO2 ratios for imputing PaO2/FIO2 ratios in the SOFA scores. Crit Care Med. 2006;34:A1.
    - CNS of patients with a not testable verbal GCS (verbal 0, e.g. intubated) carries the last assessed GCS forward, or GCS 15 if there is none, following (Proposal 1):
        Lambden S, Laterre PF, Levy MM, Francois B. The SOFA score-development, utility and challenges of accurate assessment in clinical trials. Crit Care. 2019 Nov 27;23(1):374. doi: 10.1186/s13054-019-2663-7. PMID: 31775846; PMCID: PMC6880479.

    Returns
    -------
        pl.LazyFrame
            SOFA scores with all organ subscore components
    """
    # Load defaults if not provided
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
    if medications is None:
        medications = get_medications()
    if ventilation is None:
        ventilation = get_ventilation()

    # Validate all required data is available
    required = {
        "patient_information": patient_information,
        "timeseries_vitals": timeseries_vitals,
        "timeseries_labs": timeseries_labs,
        "timeseries_resp": timeseries_resp,
        "timeseries_inout": timeseries_inout,
        "medications": medications,
        "ventilation": ventilation,
    }
    _validate_required_data("SOFA", required)

    # Strict original column names

    # Vitals
    vitals = _improve_vitals(timeseries_vitals.lazy())
    map_col = "Mean arterial pressure"
    gcs_col = "Glasgow coma score total"
    gcs_verbal_col = "Glasgow coma score verbal"

    # Labs
    labs = _improve_labs(timeseries_labs.lazy())
    platelets_col = "Platelets"
    bilirubin_col = "Bilirubin"
    creatinine_col = "Creatinine"
    pf_ratio_col = "PaO2/FiO2 Ratio"

    # Ventilation
    vent = ventilation.lazy()
    vent_start_col = "Ventilation Start Relative to Admission (seconds)"

    # Meds
    meds = medications.lazy()
    drug_ingredient_col = "Drug Ingredient"
    drug_start_col = "Drug Start Relative to Admission (seconds)"
    drug_end_col = "Drug End Relative to Admission (seconds)"

    VASOACTIVE_AGENTS = [
        "dopamine",
        "dobutamine",
        "epinephrine",
        "norepinephrine",
    ]

    # Base frames
    patient_information = patient_information.lazy()
    ctx = _window_context(patient_information, window)

    # Pre-compute ventilation flags per timeframe to avoid interval checking
    # at every respiratory measurement
    ventilation_tf = (
        vent.filter(
            ~pl.col("Ventilation Type").is_in(["other", "supplemental oxygen"])
            | pl.col(STAY_KEY).str.starts_with("eicu") # eICU ventilation type is not readily available
        )
        .join(ctx.all_stays_t0, on=STAY_KEY, how="inner")
        .with_columns(
            pl.lit(True).alias("ventilation"),
            timeframe=(pl.col(vent_start_col) - pl.col("T_0"))
            .floordiv(window.window_size)
            .cast(int),
        )
        .group_by(STAY_KEY, "timeframe")
        .agg(pl.col("ventilation").max())
    )

    # region respiratory (P/F ratio)
    resp_tf = (
        PaO2_FiO2_RATIO(
            patient_information=patient_information,
            timeseries_resp=timeseries_resp,
            timeseries_labs=timeseries_labs,
            t_0=window.t_0,
            t_0_per_stay=window.t_0_per_stay,
        )
        .select(STAY_KEY, TIME_KEY, pf_ratio_col)
        .join(ctx.all_stays_t0, on=STAY_KEY, how="inner")
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window.window_size))
        # attach ventilation flag at measurement time using intervals
        .join(ventilation_tf, on=[STAY_KEY, "timeframe"], how="left")
        .with_columns(
            _pf_ratio_points(
                pl.col(pf_ratio_col).cast(pl.Float64),
                pl.col("ventilation").fill_null(False),
            ).alias("pf_points")
        )
        .group_by(STAY_KEY, "timeframe")
        .agg(pl.max("pf_points").alias("pf_ratio_points"))
    )

    # region labs (platelets, bilirubin, creatinine)
    labs_tf = (
        labs.select(
            STAY_KEY, TIME_KEY, platelets_col, bilirubin_col, creatinine_col
        )
        .join(ctx.all_stays_t0, on=STAY_KEY, how="inner")
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window.window_size))
        .group_by(STAY_KEY, "timeframe")
        .agg(
            _platelet_points(pl.col(platelets_col))
            .max()
            .alias("platelet_points"),
            _bilirubin_points(pl.col(bilirubin_col))
            .max()
            .alias("bilirubin_points"),
            _creatinine_points(pl.col(creatinine_col))
            .max()
            .alias("creatinine_points"),
        )
    )

    # region vitals (GCS & MAP)
    vitals_tf = (
        vitals.select(STAY_KEY, TIME_KEY, gcs_col, gcs_verbal_col, map_col)
        .join(ctx.all_stays_t0, on=STAY_KEY, how="inner")
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window.window_size))
        .group_by(STAY_KEY, "timeframe")
        .agg(
            _gcs_points(pl.col(gcs_col)).max().alias("gcs_points"),
            (pl.col(gcs_verbal_col) == 0).any().alias("gcs_nt"),
            _map_points(pl.col(map_col)).max().alias("map_points"),
        )
        .pipe(GCS_NT_CARRY_FORWARD, "gcs_points", "gcs_nt")
    )

    # region medications (vasoactive) -> doses in mcg/kg/min per timeframe
    meds_tf = (
        meds.filter(pl.col(drug_ingredient_col).is_in(VASOACTIVE_AGENTS))
        .pipe(ALIGNED_UNITS, patient_information=patient_information)
        .join(ctx.all_stays_t0, on=STAY_KEY, how="inner")
        .with_columns(
            pl.col(drug_start_col)
            .sub(pl.col("T_0"))
            .alias("Drug Start Relative to T_0 (seconds)"),
            pl.col(drug_end_col)
            .sub(pl.col("T_0"))
            .alias("Drug End Relative to T_0 (seconds)"),
        )
    )

    meds_agg = PROJECT_TO_WINDOWS(
        meds_tf,
        group_cols=[STAY_KEY, drug_ingredient_col, "T_0"],
        value_col="Drug Rate (fixed units)",
        TIMEWINDOW_IN_SECONDS=window.window_size,
        prefix="Drug",
        reference="T_0",
        unit="seconds",
    )

    if window.t_1 is not None:
        meds_agg = meds_agg.filter(
            pl.col("Window Relative to T_0")
            < (pl.lit(int(window.t_1)).sub(pl.col("T_0")).floordiv(window.window_size) + 1)
        )

    meds_tf = (
        meds_agg
        .group_by(STAY_KEY, "Window Relative to T_0")
        .agg(
            TIME_WEIGHTED_MEAN_RATE(
                window.window_size, where=pl.col(drug_ingredient_col) == agent
            )
            .alias(agent)
            for agent in VASOACTIVE_AGENTS
        )
        .select(
            STAY_KEY,
            pl.col("Window Relative to T_0").alias("timeframe"),
            _vasopressor_points(
                *[pl.col(agent) for agent in VASOACTIVE_AGENTS]  # ABC-sorted
            ).alias("vasopressor_points"),
        )
    )

    # region urine output (refactored to call URINE_OUTPUT)
    uo_tf = (
        URINE_OUTPUT(
            patient_information=patient_information,
            timeseries_inout=timeseries_inout,
            t_0=window.t_0,
            t_0_per_stay=window.t_0_per_stay,
            t_minus1=window.t_minus1,
            t_1=window.t_1,
            window_size=window.window_size,
            timeframe_name="timeframe",
        )
        .with_columns(
            pl.sum("uo_interval_ml")
            .over(
                partition_by=[
                    STAY_KEY,
                    (pl.col("timeframe") * window.window_size)
                    .floordiv(SECONDS_IN_1D)
                    .alias("uo_day_index"),
                ]
            )
            .alias("uo_daily_ml")
        )
        .with_columns(_uo_points(pl.col("uo_daily_ml")).alias("uo_points"))
    )

    # region assemble
    out = _assemble(ctx, resp_tf, labs_tf, vitals_tf, meds_tf, uo_tf)

    if forward_fill:
        out = _organ_score_forward_fill(
            out,
            fast_cols=["pf_ratio_points", "gcs_points", "map_points", "vasopressor_points"],
            slow_cols=["platelet_points", "bilirubin_points", "creatinine_points"],
            uo_col="uo_points",
            window_size=window.window_size,
            stay_key=STAY_KEY,
            fill_from_pre_icu=_fill_from_pre_icu,
        )

    return _finalize(
        out.with_columns(
            # renal_points: max(creatinine_points, uo_points)
            pl.max_horizontal(
                pl.col("creatinine_points"), pl.col("uo_points")
            ).alias("renal_points"),
            # cardiovascular_points: max of MAP-based and meds-based
            pl.max_horizontal(
                pl.col("map_points"), pl.col("vasopressor_points")
            ).alias("cardiovascular_points"),
        )
        .with_columns(
            _null_if_incomplete_pre_icu(
                pl.sum_horizontal(
                pl.col("pf_ratio_points"),
                pl.col("platelet_points"),
                pl.col("bilirubin_points"),
                pl.col("cardiovascular_points"),
                pl.col("gcs_points"),
                pl.col("renal_points"),
                ignore_nulls=True,
            ),
                pl.col(["pf_ratio_points", "platelet_points", "bilirubin_points", "cardiovascular_points", "gcs_points", "renal_points"]),
                window.window_size,
            ).alias("SOFA Score")
        ),
        ctx,
        "SOFA Score",
        pl.col("pf_ratio_points").alias("Respiration"),
        pl.col("platelet_points").alias("Coagulation"),
        pl.col("bilirubin_points").alias("Liver"),
        pl.col("map_points").alias("Cardiovascular (MAP)"),
        pl.col("vasopressor_points").alias("Cardiovascular (VPs)"),
        pl.col("cardiovascular_points").alias("Cardiovascular"),
        pl.col("gcs_points").alias("Central nervous system"),
        pl.col("creatinine_points").alias("Renal (creatinine)"),
        pl.col("uo_points").alias("Renal (urine output)"),
        pl.col("renal_points").alias("Renal"),
        rest=False,
    )


__all__ = ["SOFA"]
