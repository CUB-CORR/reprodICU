"""
SOFA: compute SOFA in long format directly from raw inputs.

Output columns per row:
- Global ICU Stay ID
- T_0 (seconds from admission used as reference)
- timeframe (0-indexed integer window)
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

from ..clinical.pharmocological.ALIGNED_UNITS import ALIGNED_UNITS
from ..clinical.renal.URINE_OUTPUT import URINE_OUTPUT
from ..clinical.respiratory.PF_RATIO import PaO2_FiO2_RATIO
from ..common import (
    _assign_timeframe,
    _build_t0,
    _get_timeframe_name,
    _optional_time_bounds_filter,
    get_medications,
    get_patient_information,
    get_timeseries_intakeoutput,
    get_timeseries_labs,
    get_timeseries_respiratory,
    get_timeseries_vitals,
    get_ventilation,
)
from ..FIX_WINDOW_BORDERS import FIX_WINDOW_BORDERS

SECONDS_IN_1H = 60 * 60
SECONDS_IN_4H = 4 * SECONDS_IN_1H
SECONDS_IN_1D = 24 * SECONDS_IN_1H
SECONDS_IN_1W = 7 * SECONDS_IN_1D


################################################################################
################################################################################
# region data helpers
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
    ).filter(
        pl.col("Mean arterial pressure").is_finite(),
        pl.any_horizontal(
            "Mean arterial pressure",
            "Glasgow coma score total",
        ),
    )


def _improve_vitals_quick(vitals: pl.LazyFrame) -> pl.LazyFrame:
    return vitals.with_columns(
        pl.coalesce(
            pl.col("Invasive systolic arterial pressure"),
            pl.col("Non-invasive systolic arterial pressure"),
        ).alias("Systolic arterial pressure"),
    ).filter(
        pl.any_horizontal(
            "Systolic arterial pressure",
            "Respiratory rate",
            "Glasgow coma score total",
        ),
    )


def _improve_labs(labs: pl.LazyFrame) -> pl.LazyFrame:
    return labs.with_columns(
        pl.col("Platelets").struct.field("value").alias("Platelets"),
        pl.when(
            pl.col("Creatinine")
            .struct.field("system")
            .str.contains_any(["Serum", "Blood"])
        )
        .then(pl.col("Creatinine").struct.field("value"))
        .alias("Creatinine"),
        pl.when(
            pl.col("Bilirubin")
            .struct.field("system")
            .str.contains_any(["Serum", "Blood"])
        )
        .then(pl.col("Bilirubin").struct.field("value"))
        .alias("Bilirubin"),
    ).filter(pl.any_horizontal("Platelets", "Creatinine", "Bilirubin"))


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
    """
    Platelets, ×10^3/µL
    ≥150       0
     100-149  +1
      50- 99  +2
      20- 49  +3
     <20      +4
    """
    return (
        pl.when(platelets_value >= 150)
        .then(0)
        .when(platelets_value.is_between(100, 150, closed="left"))
        .then(1)
        .when(platelets_value.is_between(50, 100, closed="left"))
        .then(2)
        .when(platelets_value.is_between(20, 50, closed="left"))
        .then(3)
        .when(platelets_value < 20)
        .then(4)
        .otherwise(None)
    )


def _gcs_points(gcs: pl.Expr) -> pl.Expr:
    """
    Glasgow Coma Scale
    15      0
    13-14  +1
    10-12  +2
     6- 9  +3
    <6     +4
    """
    return (
        pl.when(gcs == 15)
        .then(0)
        .when(gcs.is_between(13, 14))
        .then(1)
        .when(gcs.is_between(10, 12))
        .then(2)
        .when(gcs.is_between(6, 9))
        .then(3)
        .when(gcs < 6)
        .then(4)
        .otherwise(None)
    )


def _bilirubin_points(bili: pl.Expr) -> pl.Expr:
    """
    Bilirubine, mg/dL (μmol/L)
     <1.2      (<20)       0
      1.2–1.9   (20- 32)  +1
      2.0–5.9   (33-101)  +2
      6.0–11.9 (102-204)  +3
    ≥12.0         (>204)  +4
    """
    return (
        pl.when(bili < 1.2)
        .then(0)
        .when(bili.is_between(1.2, 2.0, closed="left"))
        .then(1)
        .when(bili.is_between(2.0, 6.0, closed="left"))
        .then(2)
        .when(bili.is_between(6.0, 12.0, closed="left"))
        .then(3)
        .when(bili >= 12.0)
        .then(4)
        .otherwise(None)
    )


def _creatinine_points(crea: pl.Expr) -> pl.Expr:
    """
    Creatinine, mg/dL (μmol/L)
    <1.2        (<110)   0
     1.2–1.9 (110-170)  +1
     2.0–3.4 (171-299)  +2
     3.5–4.9 (300-440)  +3
    ≥5.0        (>440)  +4
    """
    return (
        pl.when(crea < 1.2)
        .then(0)
        .when(crea.is_between(1.2, 2.0, closed="left"))
        .then(1)
        .when(crea.is_between(2.0, 3.5, closed="left"))
        .then(2)
        .when(crea.is_between(3.5, 5.0, closed="left"))
        .then(3)
        .when(crea >= 5.0)
        .then(4)
        .otherwise(None)
    )


def _uo_points(uo_ml: pl.Expr) -> pl.Expr:
    """
    Urine output, mL/day
    <500 mL/day  +3
    <200 mL/day  +4
    """
    return pl.when(uo_ml < 200).then(4).when(uo_ml < 500).then(3).otherwise(0)


def _map_points(map_val: pl.Expr) -> pl.Expr:
    """
    Mean Arterial Pressure (MAP), mmHg
    <70 mmHg  +1
    """
    return pl.when(map_val < 70).then(1).otherwise(0)


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
def qSOFA(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_vitals: Optional[pl.LazyFrame] = None,
    *,
    t_0: Optional[int] = 0,
    t_0_per_stay: Optional[pl.LazyFrame] = None,
    t_1: Optional[int] = None,
    window_size: int = SECONDS_IN_1D,
    timeframe_unit: str = "Days",  # semantics only; output timeframe is numeric
    timeframe_name: str = None,
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
        timeframe_unit : str, optional
            Semantic only; output column remains a numeric timeframe.
        forward_fill : bool, optional
            Whether to forward-fill values within windows. Defaults to True.
        timeframe_name : str, optional
            Name for output timeframe column. Auto-generated if None.

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

    missing = [name for name, data in required.items() if data is None]
    if missing:
        raise ValueError(
            f"Cannot compute SOFA: Missing required datasets: {', '.join(missing)}. "
            f"Ensure they are configured in ~/.reprodICU/PATHS.yaml or provide them explicitly."
        )

    # Strict original column names
    STAY_KEY = "Global ICU Stay ID"
    TIME_KEY = "Time Relative to Admission (seconds)"
    los_col = "ICU Length of Stay (days)"

    # Vitals
    vitals = _improve_vitals_quick(timeseries_vitals.lazy())
    sbp_col = "Systolic arterial pressure"
    rr_col = "Respiratory rate"
    gcs_col = "Glasgow coma score total"

    # Base frames
    patient_information = patient_information.lazy()
    ALL_STAYS = patient_information.select(STAY_KEY)
    ALL_STAYS_T0 = _build_t0(ALL_STAYS, t_0_per_stay=t_0_per_stay, t_0=t_0)
    timeframe_name = _get_timeframe_name(
        timeframe_name, window_size, t_0, t_0_per_stay
    )

    # vitals (SBP, RR & GCS)
    vitals_tf = (
        vitals.select(STAY_KEY, TIME_KEY, sbp_col, rr_col, gcs_col)
        .join(ALL_STAYS_T0, on=STAY_KEY, how="inner")
        .filter(pl.col(TIME_KEY) >= pl.col("T_0").sub(SECONDS_IN_1W))
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window_size))
        .group_by(STAY_KEY, "timeframe")
        # 1 point each for systolic hypotension, tachypnea, or altered mentation
        .agg(
            (pl.col(sbp_col) <= 100).cast(int).max().alias("sbp_points"),
            (pl.col(rr_col) >= 22).cast(int).max().alias("rr_points"),
            (pl.col(gcs_col) != 15).cast(int).max().alias("gcs_points"),
        )
    )

    # union of all (stay,timeframe)
    base = (
        ALL_STAYS_T0.join(patient_information, on=STAY_KEY, how="left")
        .select(STAY_KEY, "T_0", los_col)
        .with_columns(
            pl.int_ranges(
                start=0 - pl.col("T_0").floordiv(window_size).sub(1),
                end=pl.col(los_col)
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

    # assemble
    out = base.join(vitals_tf, on=[STAY_KEY, "timeframe"], how="left")

    return (
        out.filter(
            _optional_time_bounds_filter("timeframe", window_size, t_0, t_1)
        )
        .with_columns(
            pl.sum_horizontal(
                pl.col("sbp_points"),
                pl.col("rr_points"),
                pl.col("gcs_points"),
                ignore_nulls=True,
            ).alias("qSOFA Score")
        )
        .select(
            STAY_KEY,
            "T_0",
            pl.col("timeframe").alias(timeframe_name),
            "qSOFA Score",
            pl.col("sbp_points").alias("Systolic hypotension (<=100 mmHg)"),
            pl.col("rr_points").alias("Tachypnea (>=22/min)"),
            pl.col("gcs_points").alias("Altered mentation (GCS < 15)"),
        )
        .sort(STAY_KEY, timeframe_name)
    )


################################################################################
################################################################################
# region SOFA
def SOFA(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_vitals: Optional[pl.LazyFrame] = None,
    timeseries_labs: Optional[pl.LazyFrame] = None,
    timeseries_resp: Optional[pl.LazyFrame] = None,
    timeseries_inout: Optional[pl.LazyFrame] = None,
    medications: Optional[pl.LazyFrame] = None,
    ventilation: Optional[pl.LazyFrame] = None,
    *,
    t_0: Optional[int] = 0,
    t_0_per_stay: Optional[pl.LazyFrame] = None,
    t_1: Optional[int] = None,
    window_size: int = SECONDS_IN_1D,
    timeframe_unit: str = "Days",  # semantics only; output timeframe is numeric
    forward_fill: bool = True,
    timeframe_name: str = None,
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
        timeframe_unit : str, optional
            Semantic only; output column remains a numeric timeframe.
        forward_fill : bool, optional
            Whether to forward-fill values within windows. Defaults to True.
        timeframe_name : str, optional
            Name for output timeframe column. Auto-generated if None.

    Sources
    -------

    - original source:
        Vincent, J.-L., Moreno, R., Takala, J., Willatts, S., De Mendonça, A., Bruining, H., … Thijs, L. G. (1996). The SOFA (Sepsis-related Organ Failure Assessment) score to describe organ dysfunction/failure. Intensive Care Medicine, 22(7), 707–710. doi:10.1007/bf01709751
    - modified for usage *without mechanical ventilation* data following:
        Jones, A. E., Trzeciak, S., & Kline, J. A. (2009). The Sequential Organ Failure Assessment score for predicting outcome in patients with severe sepsis and evidence of hypoperfusion at the time of emergency department presentation. Critical Care Medicine, 37(5), 1649–1654. doi:10.1097/CCM.0b013e31819def97
            * source of the substitution:
                Pandharipande PP, Sanders N, Jacques P, et al. Calculating SOFA scores when arterial blood gasses are not available: Validating SpO2/FIO2 ratios for imputing PaO2/FIO2 ratios in the SOFA scores. Crit Care Med. 2006;34:A1.

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

    missing = [name for name, data in required.items() if data is None]
    if missing:
        raise ValueError(
            f"Cannot compute SOFA: Missing required datasets: {', '.join(missing)}. "
            f"Ensure they are configured in ~/.reprodICU/PATHS.yaml or provide them explicitly."
        )

    # Strict original column names
    STAY_KEY = "Global ICU Stay ID"
    TIME_KEY = "Time Relative to Admission (seconds)"
    los_col = "ICU Length of Stay (days)"

    # Vitals
    vitals = _improve_vitals(timeseries_vitals.lazy())
    map_col = "Mean arterial pressure"
    gcs_col = "Glasgow coma score total"

    # Labs
    labs = _improve_labs(timeseries_labs.lazy())
    platelets_col = "Platelets"
    bilirubin_col = "Bilirubin"
    creatinine_col = "Creatinine"
    pf_ratio_col = "PaO2/FiO2 Ratio"

    if ventilation is not None:
        vent = ventilation.lazy()
        vent_start_col = "Ventilation Start Relative to Admission (seconds)"
        vent_end_col = "Ventilation End Relative to Admission (seconds)"

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
    ALL_STAYS = patient_information.select(STAY_KEY)
    ALL_STAYS_T0 = _build_t0(ALL_STAYS, t_0_per_stay=t_0_per_stay, t_0=t_0)
    timeframe_name = _get_timeframe_name(
        timeframe_name, window_size, t_0, t_0_per_stay
    )

    # Ventilation is handled in the respiratory section using start/end intervals

    # region respiratory (P/F ratio)
    resp_tf = (
        PaO2_FiO2_RATIO(t_0=t_0, t_0_per_stay=t_0_per_stay)
        .select(STAY_KEY, TIME_KEY, pf_ratio_col)
        .join(ALL_STAYS_T0, on=STAY_KEY, how="inner")
        .filter(pl.col(TIME_KEY) >= pl.col("T_0").sub(SECONDS_IN_1W))
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window_size))
        # attach ventilation flag at measurement time using intervals
        .join(
            vent.filter(
                ~pl.col("Ventilation Type").is_in(
                    ["other", "supplemental oxygen"]
                )
            ).select(STAY_KEY, vent_start_col, vent_end_col),
            STAY_KEY,
            how="left",
        )
        .with_columns(
            pl.when(
                pl.col(TIME_KEY) >= pl.col(vent_start_col),
                pl.col(vent_end_col).is_null()
                | (pl.col(TIME_KEY) < pl.col(vent_end_col)),
            )
            .then(True)
            .otherwise(False)
            .alias("ventilation")
        )
        .group_by(STAY_KEY, TIME_KEY, "timeframe", pf_ratio_col)
        .agg(pl.max("ventilation"))
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
        .join(ALL_STAYS_T0, on=STAY_KEY, how="inner")
        .filter(pl.col(TIME_KEY) >= pl.col("T_0").sub(SECONDS_IN_1W))
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window_size))
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
        vitals.select(STAY_KEY, TIME_KEY, gcs_col, map_col)
        .join(ALL_STAYS_T0, on=STAY_KEY, how="inner")
        .filter(pl.col(TIME_KEY) >= pl.col("T_0").sub(SECONDS_IN_1W))
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window_size))
        .group_by(STAY_KEY, "timeframe")
        .agg(
            _gcs_points(pl.col(gcs_col)).max().alias("gcs_points"),
            _map_points(pl.col(map_col)).max().alias("map_points"),
        )
    )

    # region medications (vasoactive) -> doses in mcg/kg/min per timeframe
    # Attribute to timeframe using FIX_WINDOW_BORDERS (window-based attribution like VIS.py)
    meds_tf = (
        meds.filter(pl.col(drug_ingredient_col).is_in(VASOACTIVE_AGENTS))
        .pipe(ALIGNED_UNITS, patient_information=patient_information)
        .join(ALL_STAYS_T0, on=STAY_KEY, how="inner")
        .filter(pl.col(drug_end_col) >= pl.col("T_0").sub(SECONDS_IN_1W))
        .with_columns(
            pl.col(drug_start_col)
            .sub(pl.col("T_0"))
            .alias("Drug Start Relative to T_0 (seconds)"),
            pl.col(drug_end_col)
            .sub(pl.col("T_0"))
            .alias("Drug End Relative to T_0 (seconds)"),
        )
    )

    meds_tf = FIX_WINDOW_BORDERS(
        meds_tf,
        TIMEWINDOW_IN_SECONDS=window_size,
        prefix="Drug",
        reference="T_0",
        unit="seconds",
    ).with_columns(
        pl.col("Window Relative to T_0").alias("timeframe"),
        (
            pl.col("Drug Rate (fixed units)")
            * pl.col("Drug Duration (windows)")
        ).alias("time-weighted Rate"),
    )

    if t_1 is not None:
        meds_tf = meds_tf.filter(
            pl.col("timeframe")
            < (pl.lit(int(t_1)).sub(pl.col("T_0")).floordiv(window_size) + 1)
        )

    meds_tf = (
        meds_tf.group_by(STAY_KEY, "timeframe")
        .agg(
            pl.when(pl.col(drug_ingredient_col) == agent)
            .then(
                pl.sum("time-weighted Rate").truediv(
                    pl.sum("Drug Duration (windows)")
                )
            )
            .otherwise(0)
            .max()
            .alias(agent)
            for agent in VASOACTIVE_AGENTS
        )
        .select(
            STAY_KEY,
            "timeframe",
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
            t_0=t_0,
            t_0_per_stay=t_0_per_stay,
            t_1=t_1,
            window_size=window_size,
        )
        .with_columns(
            pl.sum("uo_interval_ml")
            .over(
                partition_by=[
                    STAY_KEY,
                    (pl.col("timeframe") * window_size)
                    .floordiv(SECONDS_IN_1D)
                    .alias("uo_day_index"),
                ]
            )
            .alias("uo_daily_ml")
        )
        .with_columns(
            _uo_points(pl.col("uo_daily_ml")).alias("uo_points"),
            pl.col("timeframe").cast(float),
        )
    )

    # region union of all (stay,timeframe)
    base = (
        ALL_STAYS_T0.join(patient_information, on=STAY_KEY, how="left")
        .select(STAY_KEY, "T_0", los_col)
        .with_columns(
            pl.int_ranges(
                start=0 - pl.col("T_0").floordiv(window_size).sub(1),
                end=pl.col(los_col)
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
    for part in [resp_tf, labs_tf, vitals_tf, meds_tf, uo_tf]:
        out = out.join(part, on=[STAY_KEY, "timeframe"], how="left")

    if forward_fill:
        out = out.with_columns(
            # forward-fill within stay at most 6 hours
            pl.col(
                "pf_ratio_points",
                "gcs_points",
                "map_points",
                "vasopressor_points",
            )
            .forward_fill((window_size // SECONDS_IN_1H) * 6)
            .over(partition_by=STAY_KEY, order_by="timeframe"),
            # forward-fill within stay at most a week
            pl.col(
                "platelet_points",
                "bilirubin_points",
                "creatinine_points",
            )
            .forward_fill((window_size // SECONDS_IN_1H) * 168)
            .over(partition_by=STAY_KEY, order_by="timeframe"),
            # make urine output persistent for 24h
            pl.col("uo_points")
            .forward_fill()
            .backward_fill()
            .over(
                partition_by=[
                    STAY_KEY,
                    pl.col("timeframe").floordiv(
                        (window_size // SECONDS_IN_1H) * 24
                    ),
                ],
            ),
        )

    return (
        out.filter(
            _optional_time_bounds_filter("timeframe", window_size, t_0, t_1)
        )
        .with_columns(
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
            pl.sum_horizontal(
                pl.col("pf_ratio_points"),
                pl.col("platelet_points"),
                pl.col("bilirubin_points"),
                pl.col("cardiovascular_points"),
                pl.col("gcs_points"),
                pl.col("renal_points"),
                ignore_nulls=True,
            ).alias("SOFA Score")
        )
        .select(
            STAY_KEY,
            "T_0",
            pl.col("timeframe").alias(timeframe_name),
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
        )
        .sort(STAY_KEY, timeframe_name)
    )


__all__ = ["SOFA"]
