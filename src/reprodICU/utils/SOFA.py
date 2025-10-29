"""
SOFA_LONG: compute SOFA in long format directly from raw inputs.

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
"""

# author: Finn Fassbender
# version: 31.08.2025


# Implements the Sequential Organ Failure Assessment (SOFA) score
# as in https://www.mdcalc.com/calc/691/sequential-organ-failure-assessment-sofa-score

from typing import Optional

import polars as pl
from .URINE_OUTPUT import URINE_OUTPUT
from .FIX_WINDOW_BORDERS import FIX_WINDOW_BORDERS

SECONDS_IN_1MIN = 60
SECONDS_IN_1H = 60 * SECONDS_IN_1MIN
SECONDS_IN_4H = 4 * SECONDS_IN_1H
SECONDS_IN_12H = 12 * SECONDS_IN_1H
SECONDS_IN_1D = 24 * SECONDS_IN_1H
SECONDS_IN_1W = 7 * SECONDS_IN_1D


# region helpers
def _build_t0(
    all_stays: pl.LazyFrame,
    t_0_per_stay: Optional[pl.LazyFrame],
    t_0: Optional[int],
) -> pl.LazyFrame:
    all_stays = all_stays.lazy()

    if t_0_per_stay is not None:
        t_0_per_stay = t_0_per_stay.lazy()
        return (
            all_stays.select("Global ICU Stay ID")
            .join(
                t_0_per_stay.select("Global ICU Stay ID", "T_0"),
                "Global ICU Stay ID",
                how="left",
            )
            .with_columns(pl.col("T_0").fill_null(0).cast(pl.Int64))
        )

    t0_val = 0 if t_0 is None else int(t_0)
    return all_stays.select(
        "Global ICU Stay ID", pl.lit(t0_val).cast(pl.Int64).alias("T_0")
    )


def _assign_timeframe(time_col: str, window_size: int) -> pl.Expr:
    return pl.col(time_col).sub(pl.col("T_0")).floordiv(window_size)


def _optional_time_bounds_filter(
    time_col: str, window_size: int, t_0: Optional[int], t_1: Optional[int]
) -> list[pl.Expr]:
    conds: list[pl.Expr] = []
    if t_0 is not None:
        conds.append(
            pl.col(time_col) >= pl.lit(int(t_0)).floordiv(window_size).sub(1)
        )
    if t_1 is not None:
        conds.append(
            pl.col(time_col) < pl.lit(int(t_1)).floordiv(window_size).add(1)
        )
    return conds


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
        pl.col("Glasgow coma score total").is_finite(),
        pl.any_horizontal("Mean arterial pressure", "Glasgow coma score total"),
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
        pl.when(
            pl.col("Oxygen")
            .struct.field("system")
            .is_in(["Blood arterial", "Blood"])
            | pl.col("Oxygen").struct.field("system").is_null()
        )
        .then(pl.col("Oxygen").struct.field("value"))
        .otherwise(None)
        .alias("Oxygen in Arterial blood"),
    ).filter(
        pl.any_horizontal(
            "Platelets", "Creatinine", "Bilirubin", "Oxygen in Arterial blood"
        )
    )


def _improve_resp(resp: pl.LazyFrame) -> pl.LazyFrame:
    return (
        resp.with_columns(
            pl.max_horizontal(
                "Oxygen/Total gas setting [Volume Fraction] Ventilator",
                "Oxygen/Gas total [Pure volume fraction] Inhaled gas",
            ).alias("FiO2")
        )
        .select(
            "Global ICU Stay ID",
            "Time Relative to Admission (seconds)",
            pl.when(pl.col("FiO2").is_between(0, 1))
            .then(pl.col("FiO2") * 100)
            .when(pl.col("FiO2").is_between(1, 100))
            .then(pl.col("FiO2"))
            .otherwise(None)
            .alias("FiO2"),
        )
        .drop_nulls("FiO2")
    )


# endregion


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


# endregion


################################################################################
################################################################################
# region SOFA
def SOFA_LONG(
    patient_information: pl.LazyFrame,
    timeseries_vitals: pl.LazyFrame,
    timeseries_labs: pl.LazyFrame,
    timeseries_resp: pl.LazyFrame,
    timeseries_inout: pl.LazyFrame,
    medications: pl.LazyFrame,
    ventilation: Optional[pl.LazyFrame] = None,
    *,
    t_0: Optional[int] = 0,
    t_0_per_stay: Optional[pl.LazyFrame] = None,
    t_1: Optional[int] = None,
    window_size: int = SECONDS_IN_1D,
    forward_fill: bool = True,
    timeframe_name: str = None,
) -> pl.LazyFrame:
    """
    Compute SOFA score in long format directly from raw inputs.

    Arguments
    ---------
        patient_information: pl.LazyFrame
            _description_
        timeseries_vitals: pl.LazyFrame
            _description_
        timeseries_labs: pl.LazyFrame
            _description_
        timeseries_resp: pl.LazyFrame
            _description_
        timeseries_inout: pl.LazyFrame
            _description_
        medications: pl.LazyFrame
            _description_
        ventilation : pl.LazyFrame, optional
            _description_. Defaults to None.
        t_0 : int, optional
            _description_. Defaults to 0.
        t_1 : int, optional
            _description_. Defaults to None.
        window_size : int, optional
            _description_. Defaults to SECONDS_IN_1D.
        forward_fill : bool, optional
            _description_. Defaults to True.
        t_0_per_stay : pl.LazyFrame, optional
            _description_. Defaults to None.
        timeframe_name : str, optional
            _description_. Defaults to None.

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
        pl.LazyFrame: _description_
    """

    if timeframe_name is None:
        unit = (
            "Days"
            if window_size == SECONDS_IN_1D
            else "Hours" if window_size == SECONDS_IN_1H else "Windows"
        )
        reference = (
            "T_0" if t_0 != 0 or t_0_per_stay is not None else "Admission"
        )
        timeframe_name = f"{unit} Relative to {reference}"

    # Strict original column names
    STAY_KEY = "Global ICU Stay ID"
    TIME_KEY = "Time Relative to Admission (seconds)"
    weight_col = "Admission Weight (kg)"
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

    # Resp
    resp = _improve_resp(timeseries_resp.lazy())
    pf_ratio_col = "PaO2/FiO2 Ratio"

    if ventilation is not None:
        vent = ventilation.lazy()
        vent_start_col = "Ventilation Start Relative to Admission (seconds)"
        vent_end_col = "Ventilation End Relative to Admission (seconds)"

    # Meds
    meds = medications.lazy()
    drug_ingredient_col = "Drug Ingredient"
    drug_rate_col = "Drug Rate"
    drug_rate_unit_col = "Drug Rate Unit"
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

    # Ventilation is handled in the respiratory section using start/end intervals

    # region respiratory (P/F ratio)
    pf = (
        labs.select(STAY_KEY, TIME_KEY, "Oxygen in Arterial blood")
        .drop_nulls("Oxygen in Arterial blood")
        # FiO2 occurred at most 4 hours before this blood gas
        .join_asof(
            resp,
            on=TIME_KEY,
            by=STAY_KEY,
            strategy="backward",
            tolerance=SECONDS_IN_4H,
            coalesce=True,
        )
        .with_columns(
            pl.col("Oxygen in Arterial blood")
            .truediv(
                pl.col("FiO2").fill_null(21).truediv(100)
            )  # 21% if FiO2 missing
            .alias(pf_ratio_col)
        )
        .with_columns(
            pl.when(pl.col(pf_ratio_col).is_finite())
            .then(pl.col(pf_ratio_col))
            .otherwise(None)
            .alias(pf_ratio_col)
        )
    )

    resp_tf = (
        pf.select(STAY_KEY, TIME_KEY, pf_ratio_col)
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
    # endregion

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
    # endregion

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
    # Normalize to mcg/kg/min
    # Join patient weight
    meds_norm = (
        meds.join(
            patient_information.select(STAY_KEY, weight_col),
            on=STAY_KEY,
            how="left",
        )
        .with_columns(
            # convert absolute doses to per-kg when needed
            pl.when(
                pl.col(drug_rate_unit_col).is_in(
                    ["mcg/hr", "mcg/min", "mg/hr", "mg/min"]
                )
            )
            .then(pl.col(drug_rate_col) / pl.col(weight_col))
            .otherwise(pl.col(drug_rate_col))
            .alias("rate_perkg"),
            # update units to reflect per-kg when we converted
            pl.when(pl.col(drug_rate_unit_col) == "mcg/hr")
            .then(pl.lit("mcg/kg/hr"))
            .when(pl.col(drug_rate_unit_col) == "mcg/min")
            .then(pl.lit("mcg/kg/min"))
            .when(pl.col(drug_rate_unit_col) == "mg/hr")
            .then(pl.lit("mg/kg/hr"))
            .when(pl.col(drug_rate_unit_col) == "mg/min")
            .then(pl.lit("mg/kg/min"))
            .otherwise(pl.col(drug_rate_unit_col))
            .alias("unit_perkg"),
        )
        .with_columns(
            # convert mg to mcg
            pl.when(pl.col("unit_perkg") == "mg/kg/hr")
            .then(pl.col("rate_perkg") * 1000)
            .when(pl.col("unit_perkg") == "mg/kg/min")
            .then(pl.col("rate_perkg") * 1000)
            .otherwise(pl.col("rate_perkg"))
            .alias("rate_mcg"),
            pl.when(pl.col("unit_perkg") == "mg/kg/hr")
            .then(pl.lit("mcg/kg/hr"))
            .when(pl.col("unit_perkg") == "mg/kg/min")
            .then(pl.lit("mcg/kg/min"))
            .otherwise(pl.col("unit_perkg"))
            .alias("unit_mcg"),
        )
        .with_columns(
            # convert /hr to /min
            pl.when(pl.col("unit_mcg") == "mcg/kg/hr")
            .then(pl.col("rate_mcg") / 60)
            .otherwise(pl.col("rate_mcg"))
            .alias("rate_final"),
            pl.lit("mcg/kg/min").alias("unit_final"),
        )
    )

    # Attribute to timeframe using FIX_WINDOW_BORDERS (window-based attribution like VIS.py)
    meds_tf = (
        meds_norm.filter(pl.col(drug_ingredient_col).is_in(VASOACTIVE_AGENTS))
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
        (pl.col("rate_mcg") * pl.col("Drug Duration (windows)")).alias(
            "time-weighted Rate"
        ),
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
    # endregion

    # region urine output (refactored to call URINE_OUTPUT)
    uo_base = URINE_OUTPUT(
        patient_information=patient_information,
        timeseries_inout=timeseries_inout,
        t_0=t_0,
        t_0_per_stay=t_0_per_stay,
        t_1=t_1,
        window_size=window_size,
    )
    uo_tf = uo_base.with_columns(
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
    ).with_columns(
        _uo_points(pl.col("uo_daily_ml")).alias("uo_points"),
        pl.col("timeframe").cast(float),
    )
    # endregion

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
                .floordiv(window_size)
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
    # endregion

    # region assemble
    out = base
    for part in [resp_tf, labs_tf, vitals_tf, meds_tf, uo_tf]:
        out = out.join(part, on=[STAY_KEY, "timeframe"], how="left")

    if forward_fill:
        out = out.with_columns(
            pl.col(
                "pf_ratio_points",
                "gcs_points",
                "map_points",
                "vasopressor_points",
            )
            # forward-fill within stay at most 6 hours
            .forward_fill(window_size // SECONDS_IN_1H * 6).over(
                partition_by=STAY_KEY, order_by="timeframe"
            ),
            pl.col(
                "platelet_points",
                "bilirubin_points",
                "creatinine_points",
            )
            # forward-fill within stay at most a week
            .forward_fill(window_size // SECONDS_IN_1H * 168).over(
                partition_by=STAY_KEY, order_by="timeframe"
            ),
            pl.col("uo_points")
            # make urine output persistent for 24h
            .forward_fill()
            .backward_fill()
            .over(
                partition_by=[
                    STAY_KEY,
                    pl.col("timeframe").floordiv(
                        window_size // SECONDS_IN_1H * 24
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


__all__ = ["SOFA_LONG"]
