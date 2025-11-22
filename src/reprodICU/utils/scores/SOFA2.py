"""
SOFA-2: compute SOFA-2 in long format directly from raw inputs.

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
- SOFA-2 Score (sum of organ points)

Time is in seconds. Windows determined by floor((time - T_0)/window_size).
Worst-within-window aggregation is applied per organ.

SOURCES
-------
- Ranzani OT, Singer M, Salluh JIF, et al.
  Development and Validation of the Sequential Organ Failure Assessment (SOFA)-2 Score.
  JAMA. Published online October 29, 2025.
  doi:10.1001/jama.2025.20516
"""

from typing import Optional

import polars as pl

from ..clinical.renal.URINE_OUTPUT import URINE_OUTPUT
from ..clinical.respiratory.PF_RATIO import PAO2_FIO2_RATIO, SPO2_FIO2_RATIO
from ..common import (
    _assign_timeframe,
    _build_t0,
    _optional_time_bounds_filter,
    get_medications,
    get_patient_information,
    get_rrt,
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
    ).filter(pl.any_horizontal("Platelets", "Creatinine", "Bilirubin"))


################################################################################
################################################################################
# region respiratory
def _pf_ratio_points(pf_ratio: pl.Expr, ventilated: pl.Expr) -> pl.Expr:
    """
    PaO2/FIO2, mmHg
     >300                                    0 # changed from 400 to 300 in SOFA-2
    <=300                                   +1
    <=225                                   +2 # changed from 300 to 225 in SOFA-2
    <=150 AND advanced ventilatory support  +3
    <= 75 AND advanced ventilatory support  +4

    NOTE:
    - Advanced ventilatory support is defined as receipt of high-flow nasal
      cannula, continuous positive airflow pressure, bilevel positive airway
      pressure, noninvasive ventilation, invasive mechanical ventilation, or
      long-term home ventilation. This is required to score 3 to 4 points, in
      addition to the PaO2:FIO2 or SpO2:FIO2 ratio being within the specified
      range. Changes in PaO2/FIO2 or SpO2:FIO2 within an 1 hour (eg, after
      suctioning) should not be considered.
    - Patients not receiving advanced respiratory support can score a maximum of
      2 points unless ventilatory support is (1) not available or (2) precluded
      due to the ceiling of treatment; if so, severity is scored by the
      PaO2:FIO2 or SpO2:FIO2 ratio.
    - If used for respiratory failure, ECMO (all forms) should be scored 4 in
      the respiratory component (regardless of PaO2;FIO2 ratio), but not in the
      cardiovascular component. If used for cardiovascular indications (all
      forms), it should be automatically scored in both the cardiovascular and
      the respiratory systems.
    """
    return (
        pl.when((pf_ratio > 300) | pf_ratio.is_null())
        .then(0)
        .when(pf_ratio.is_between(225, 300, closed="right"))
        .then(1)
        .when(pf_ratio.is_between(75, 150, closed="right") & ventilated)
        .then(3)
        .when((pf_ratio < 75) & ventilated)
        .then(4)
        # ensure this is the last condition checked to not override other tiers
        .when(pf_ratio.is_between(0, 225, closed="right"))
        .then(2)
        .otherwise(None)
    )


def _sf_ratio_points(sf_ratio: pl.Expr, ventilated: pl.Expr) -> pl.Expr:
    """
    SpO2/FIO2, mmHg
     >300                                    0
    <=300                                   +1
    <=250                                   +2
    <=200 AND advanced ventilatory support  +3
    <=120 AND advanced ventilatory support  +4

    NOTE:
    - Use the arterial oxygen saturation (SpO2) to FIO2 ratio only when the
      PaO2:FIO2 ratio is unavailable and when the SpO2 is less than 98%. Cutoffs:
      - 0 points, greater than 300 mm Hg
      - 1 point, 300 mm Hg or less
      - 2 points, 250 mm Hg or less
      - 3 points, 200 mm Hg or less with ventilatory support
      - 4 points, 120 mm Hg or less with ventilatory support or ECMO.
    """
    return (
        pl.when((sf_ratio > 300) | sf_ratio.is_null())
        .then(0)
        .when(sf_ratio.is_between(250, 300, closed="right"))
        .then(1)
        .when(sf_ratio.is_between(120, 200, closed="right") & ventilated)
        .then(3)
        .when((sf_ratio < 120) & ventilated)
        .then(4)
        # ensure this is the last condition checked to not override other tiers
        .when(sf_ratio.is_between(0, 250, closed="right"))
        .then(2)
        .otherwise(None)
    )


# region hemostasis
def _platelet_points(platelets_value: pl.Expr) -> pl.Expr:
    """
    Platelets, ×10^3/µL
    >150       0
     101-150  +1
      81-100  +2    # cutoff changed from 50 to 80 in SOFA-2
      51- 80  +3    # cutoff changed from 20 to 50 in SOFA-2
    <=50      +4
    """
    return (
        pl.when(platelets_value > 150)
        .then(0)
        .when(platelets_value.is_between(100, 150, closed="right"))
        .then(1)
        .when(platelets_value.is_between(80, 100, closed="right"))
        .then(2)
        .when(platelets_value.is_between(50, 80, closed="right"))
        .then(3)
        .when(platelets_value <= 50)
        .then(4)
        .otherwise(None)
    )


# region brain
def _gcs_points(
    gcs: pl.Expr,
    gcs_motor: pl.Expr,
    sedation_flag: pl.Expr,
    delirium_treatment_flag: pl.Expr,
) -> pl.Expr:
    """
    Glasgow Coma Scale
    15    (or GCS-M   6)     0
    13-14 (or GCS-M   5)    +1
     9-12 (or GCS-M   4)    +2  # cutoff changed from 10 to 9 in SOFA-2
     6- 8 (or GCS-M   3)    +3
    <6    (or GCS-M <=2)    +4

    NOTE:
    - For sedated patients, use the last recorded GCS before sedation. If the
      previous GCS is unknown, score 0.
    - When not possible to evaluate the 3 domains of GCS, use the best
      achieved score in the motor-scale domain.
    - If receiving drug treatment for delirium (short- or long-term), score 1
      point even if GCS is 15. For relevant drugs, see the International
      Management of Pain, Agitation, and Delirium in Adult Patients in the ICU
      Guidelines.
    """
    return (
        # For sedated patients, use the last recorded GCS before sedation.
        pl.when(sedation_flag)
        .then(None)
        .when(gcs.is_not_null())
        .then(
            # If receiving drug treatment for delirium (short- or long-term),
            # score 1 point even if GCS is 15.
            pl.when(delirium_treatment_flag)
            .then(1)
            .when(gcs == 15)
            .then(0)
            .when(gcs.is_between(13, 14))
            .then(1)
            .when(gcs.is_between(9, 12))
            .then(2)
            .when(gcs.is_between(6, 8))
            .then(3)
            .when(gcs < 6)
            .then(4)
        )
        # When not possible to evaluate the 3 domains of GCS, use the best
        # achieved score in the motor-scale domain
        .when(gcs_motor.is_not_null())
        .then(
            pl.when(gcs_motor == 6)
            .then(0)
            .when(gcs_motor == 5)
            .then(1)
            .when(gcs_motor == 4)
            .then(2)
            .when(gcs_motor == 3)
            .then(3)
            .when(gcs_motor <= 2)
            .then(4)
        )
        .otherwise(None)
    )


# region liver
def _bilirubin_points(bili: pl.Expr) -> pl.Expr:
    """
    Bilirubine, mg/dL (μmol/L)
    <= 1.2 (<= 20.6)     0
    <= 3.0 (<= 51.3)    +1  # cutoff changed from 2.0 to 3.0 in SOFA-2
    <= 6.0 (<=102.6)    +2
    <=12.0 (<=205.2)    +3
     >12.0 (> 205.2)    +4
    """
    return (
        pl.when(bili <= 1.2)
        .then(0)
        .when(bili.is_between(1.2, 3.0, closed="right"))
        .then(1)
        .when(bili.is_between(3.0, 6.0, closed="right"))
        .then(2)
        .when(bili.is_between(6.0, 12.0, closed="right"))
        .then(3)
        .when(bili >= 12.0)
        .then(4)
        .otherwise(None)
    )


# region kidney
def _creatinine_points(crea: pl.Expr, rrt: pl.Expr) -> pl.Expr:
    """
    Creatinine, mg/dL (μmol/L)
    <= 1.2 (<=110)   0
    <= 2.0 (<=170)  +1
    <= 3.5 (<=300)  +2
     > 3.5 ( >300)  +3
    receiving RRT   +4  # higher cutoff of 5.0 in SOFA-1 is not used in SOFA-2

    NOTE:
    - Excludes patients receiving RRT exclusively for nonrenal causes (eg,
      removal of toxic products, bacterial toxins, cytokines).
    - For patients not receiving RRT (eg, ceiling of treatment, machine
      unavailability, or decision to delay commencement), score 4 points if they
      otherwise meet criteria for RRT, ie, creatinine level greater than
      1.2 mg/dL (>110 μmol/L) or oliguria (<0.3 mL/kg/h) for more than 6 hours
      plus at least 1 of either serum potassium of 6.0 mmol/L or greater or
      metabolic acidosis with pH of 7.20 or less and serum bicarbonate of 12
      mmol/L or less.
    - For patients receiving intermittent RRT, score 4 points on days not
      receiving RRT until RRT use is terminated.
    """
    return (
        pl.when(crea <= 1.2)
        .then(0)
        .when(crea.is_between(1.2, 2.0, closed="right"))
        .then(1)
        .when(crea.is_between(2.0, 3.5, closed="right"))
        .then(2)
        .when(crea > 3.5)
        .then(3)
        .when(rrt)
        .then(4)
        .otherwise(None)
    )


def _uo_points(uo_df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Urine output
    <0.5 mL/kg/h for 6–12 h     +2
    <0.5 mL/kg/h for  >12 h     +3
    <0.3 mL/kg/h for  >24 h
       or anuria for  >12 h     +4

    Arguments
    ---------
        uo_df : pl.LazyFrame
            Dataframe with hourly urine output data including:
            - Global ICU Stay ID
            - time (seconds or hours relative to T_0)
            - uo_interval_ml_per_kg (hourly urine output in mL/kg/h)

    Returns
    -------
        pl.LazyFrame
            With added uo_points column containing rolling window scores
    """
    STAY_KEY = "Global ICU Stay ID"

    return uo_df.with_columns(
        # Calculate rolling averages over lookback windows
        pl.col("uo_interval_ml_per_kg")
        .rolling_mean(window_size=6, center=False)
        .over(STAY_KEY, order_by="timeframe")
        .alias("uo_6h_avg"),
        pl.col("uo_interval_ml_per_kg")
        .rolling_mean(window_size=12, center=False)
        .over(STAY_KEY, order_by="timeframe")
        .alias("uo_12h_avg"),
        pl.col("uo_interval_ml_per_kg")
        .rolling_mean(window_size=24, center=False)
        .over(STAY_KEY, order_by="timeframe")
        .alias("uo_24h_avg"),
        # Check for anuria in lookback windows
        (pl.col("uo_interval_ml_per_kg") == 0)
        .cast(int)
        .rolling_sum(window_size=12, center=False)
        .over(STAY_KEY, order_by="timeframe")
        .alias("anuria_hours_12h"),
    ).select(
        STAY_KEY,
        "timeframe",
        # 4 points: <0.3 mL/kg/h for >24h
        pl.when(pl.col("uo_24h_avg") < 0.3).then(4)
        # 4 points: anuria for >12h
        .when(pl.col("anuria_hours_12h") > 12).then(4)
        # 3 points: <0.5 mL/kg/h for >12h
        .when(pl.col("uo_12h_avg") < 0.5).then(3)
        # 2 points: <0.5 mL/kg/h for 6-12h
        .when(pl.col("uo_6h_avg") < 0.5)
        .then(2)
        .otherwise(None)
        .alias("uo_points"),
    )


# region cardiovascular
def _map_points(map_val: pl.Expr) -> pl.Expr:
    """
    Mean Arterial Pressure (MAP), mmHg
    >=70 mmHg    0
     <70 mmHg   +1
    """
    return pl.when(map_val < 70).then(1).otherwise(0)


def _vasopressor_points(
    dopamine: pl.Expr,
    dobutamine: pl.Expr,
    epinephrine: pl.Expr,
    norepinephrine: pl.Expr,
    phenylephrine: pl.Expr,
    vasopressin: pl.Expr,
) -> pl.Expr:
    """
    Administration of vasoactive agents required (mcg/kg/min):
    +2 Low-dose vasopressor (sum of norepinephrine and epinephrine ≤0.2 μg/kg/min)
       OR any dose of other vasopressor or inotrope
    +3 Medium-dose vasopressor (sum of norepinephrine and epinephrine >0.2 to ≤0.4 μg/kg/min)
       OR low-dose vasopressor (sum norepinephrine and epinephrine ≤0.2 μg/kg/min)
          with any other vasopressor or inotrope
    +4 High-dose vasopressor (sum of norepinephrine and epinephrine >0.4 μg/kg/min)
       OR medium-dose vasopressor (sum of norepinephrine and epinephrine >0.2 to ≤0.4 μg/kg/min)
          with any other vasopressor or inotrope or mechanical support

    NOTE:
    - Vasopressor medication is only scored if given by continuous intravenous
      infusion for at least 1 hour.
    - Norepinephrine is usually dispensed as the salt (eg, hemitartrate or
      bitartrate). Dose should be expressed as the base. 1 mg of norepinephrine
      base is equivalent to 2 mg of norepinephrine bitartrate monohydrate,
      1.89 mg of the anhydrous bitartrate (also called hydrogen tartrate, acid
      tartrate, or tartrate), and 1.22 mg of the hydrochloride.
    - If dopamine is used as a single vasopressor, scoring is based on the
      following cutoffs:
      - 2 points (<=20 μg/kg/min)
      - 3 points (>20 to <=40 μg/kg/min)
      - 4 points (>40 μg/kg/min)
      These cutoffs are based on norepinephrine equipotency studies.
    - When vasoactive drugs are unavailable or precluded due to a ceiling of
      treatment, use the following MAP cutoffs for scoring:
      - 0 point, 70 mm Hg or higher
      - 1 point, 60 to 69 mm Hg
      - 2 points, 50 to 59 mm Hg
      - 3 points, 40 to 49 mm Hg
      - 4 points, less than 40 mm Hg
    """
    # Extract helper expressions for clarity
    nor_epi_sum = norepinephrine + epinephrine
    dopamine_only = dopamine.is_not_null() & pl.all_horizontal(
        dobutamine.is_null(),
        epinephrine.is_null(),
        norepinephrine.is_null(),
        phenylephrine.is_null(),
        vasopressin.is_null(),
    )
    has_other_agent = pl.any_horizontal(
        dobutamine.is_not_null(),
        phenylephrine.is_not_null(),
        vasopressin.is_not_null(),
    )
    has_any_agent = pl.any_horizontal(
        dopamine.is_not_null(),
        dobutamine.is_not_null(),
        epinephrine.is_not_null(),
        norepinephrine.is_not_null(),
        phenylephrine.is_not_null(),
        vasopressin.is_not_null(),
    )

    return (
        # Dopamine as sole agent: special cutoffs per spec
        pl.when(dopamine_only)
        .then(
            pl.when(dopamine <= 20)
            .then(2)
            .when(dopamine <= 40)
            .then(3)
            .otherwise(4)
        )
        # High-dose vasopressor: always 4 points
        .when(nor_epi_sum > 0.4)
        .then(4)
        # Medium-dose vasopressor with other agent: 4 points
        .when((nor_epi_sum > 0.2) & has_other_agent)
        .then(4)
        # Medium-dose vasopressor alone: 3 points
        .when(nor_epi_sum > 0.2)
        .then(3)
        # Low-dose vasopressor with other agent: 3 points
        .when((nor_epi_sum > 0) & has_other_agent)
        .then(3)
        # Low-dose vasopressor alone: 2 points
        .when(nor_epi_sum > 0)
        .then(2)
        # Any other vasoactive agent: 2 points
        .when(has_any_agent)
        .then(2)
        # No vasoactive agents: 0 points
        .otherwise(0)
    )


###################
###################
# region vasopressor
def get_vasopressor_points(
    meds: pl.LazyFrame,
    patient_information: pl.LazyFrame,
    ALL_STAYS_T0: pl.LazyFrame,
    *,
    t_1: Optional[int] = None,
    window_size: int = SECONDS_IN_1D,
) -> pl.LazyFrame:
    """
    Compute vasopressor points over time windows from medication administration
    data.

    Arguments
    ---------
        meds : pl.LazyFrame
            Medications administration data.
        patient_information : pl.LazyFrame
            Patient information data including weights.
        ALL_STAYS_T0 : pl.LazyFrame
            Dataframe with all stays and their T_0 times.
        t_1 : int, optional
            Optional upper time bound (seconds from admission) for filtering inputs.
        window_size : int, optional
            Timeframe width in seconds (default: 86400 = 1 day). Window index is
            floor((time - T_0)/window_size).

    Returns
    -------
        pl.LazyFrame
            Vasopressor points per stay and timeframe.
    """

    STAY_KEY = "Global ICU Stay ID"
    weight_col = "Admission Weight (kg)"
    drug_ingredient_col = "Drug Ingredient"
    drug_start_col = "Drug Start Relative to Admission (seconds)"
    drug_end_col = "Drug End Relative to Admission (seconds)"
    drug_rate_col = "Drug Rate"
    drug_rate_unit_col = "Drug Rate Unit"

    VASOACTIVE_AGENTS = [
        "dobutamine",
        "dopamine",
        "epinephrine",
        "norepinephrine",
        "phenylephrine",
        "vasopressin (USP)",
    ]

    # Normalize to mcg/kg/min
    vp_tf = (
        meds.filter(
            pl.col(drug_ingredient_col).is_in(VASOACTIVE_AGENTS),
            # filter for infusions of at least 1 hour
            (pl.col(drug_end_col) - pl.col(drug_start_col)) >= SECONDS_IN_1H,
        )
        # Join patient weight
        .join(
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
            .alias(drug_rate_col),
            pl.lit("mcg/kg/min").alias(drug_rate_unit_col),
        )
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

    # Attribute to timeframe using FIX_WINDOW_BORDERS
    vp_tf = FIX_WINDOW_BORDERS(
        vp_tf,
        TIMEWINDOW_IN_SECONDS=window_size,
        prefix="Drug",
        reference="T_0",
        unit="seconds",
    ).with_columns(
        pl.col("Window Relative to T_0").alias("timeframe"),
        pl.col(drug_rate_col)
        .mul(pl.col("Drug Duration (windows)"))
        .alias("time-weighted Rate"),
    )

    if t_1 is not None:
        vp_tf = vp_tf.filter(
            pl.col("timeframe")
            < (pl.lit(int(t_1)).sub(pl.col("T_0")).floordiv(window_size) + 1)
        )

    return (
        vp_tf.group_by(STAY_KEY, "timeframe")
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


################################################################################
################################################################################
# region SOFA-2
def SOFA2(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_vitals: Optional[pl.LazyFrame] = None,
    timeseries_labs: Optional[pl.LazyFrame] = None,
    timeseries_resp: Optional[pl.LazyFrame] = None,
    timeseries_inout: Optional[pl.LazyFrame] = None,
    medications: Optional[pl.LazyFrame] = None,
    ventilation: Optional[pl.LazyFrame] = None,
    rrt: Optional[pl.LazyFrame] = None,
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
    Compute SOFA-2 score with automatic dataset loading.

    All data parameters are optional and will be automatically loaded from the
    package datasets if not provided. This makes it convenient for quick analysis
    while maintaining flexibility for custom data.

    NOTE: For missing values at day 1, the general recommendation is to score
    these as 0 points. This may vary for specific purposes (eg, bedside use,
    research, etc). For sequential scoring, for missing data after day 1, it is
    to carry forward the last observation, the rationale being that
    nonmeasurement suggests stability.

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
        rrt : pl.LazyFrame, optional
            Renal replacement therapy data. Loaded automatically if None.
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
    - Ranzani OT, Singer M, Salluh JIF, et al.
      Development and Validation of the Sequential Organ Failure Assessment (SOFA)-2 Score.
      JAMA. Published online October 29, 2025.
      doi:10.1001/jama.2025.20516

    Returns
    -------
        pl.LazyFrame
            SOFA-2 scores with all organ subscore components
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
    if rrt is None:
        rrt = get_rrt()

    # Validate all required data is available
    required = {
        "patient_information": patient_information,
        "timeseries_vitals": timeseries_vitals,
        "timeseries_labs": timeseries_labs,
        "timeseries_resp": timeseries_resp,
        "timeseries_inout": timeseries_inout,
        "medications": medications,
        "ventilation": ventilation,
        "rrt": rrt,
    }

    missing = [name for name, data in required.items() if data is None]
    if missing:
        raise ValueError(
            f"Cannot compute SOFA-2: Missing required datasets: {', '.join(missing)}. "
            f"Ensure they are configured in ~/.reprodICU/PATHS.yaml or provide them explicitly."
        )

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
    los_col = "ICU Length of Stay (days)"

    # Vitals
    vitals = _improve_vitals(timeseries_vitals.lazy())
    map_col = "Mean arterial pressure"
    gcs_col = "Glasgow coma score total"
    gcs_motor_col = "Glasgow coma score motor"
    sf_ratio_col = "SpO2/FiO2 Ratio"

    # Labs
    labs = _improve_labs(timeseries_labs.lazy())
    platelets_col = "Platelets"
    bilirubin_col = "Bilirubin"
    creatinine_col = "Creatinine"
    pf_ratio_col = "PaO2/FiO2 Ratio"

    # Resp
    resp = timeseries_resp.lazy()
    oxygen_delivery_col = "Oxygen delivery system"

    # Ventilation
    vent = ventilation.lazy()
    vent_start_col = "Ventilation Start Relative to Admission (seconds)"
    vent_end_col = "Ventilation End Relative to Admission (seconds)"

    # RRT
    rrt = rrt.lazy()
    rrt_start_col = "Renal Replacement Therapy Start Relative to Admission (seconds)" # fmt: skip
    rrt_end_col = "Renal Replacement Therapy End Relative to Admission (seconds)" # fmt: skip

    # Meds
    meds = medications.lazy()
    drug_ingredient_col = "Drug Ingredient"
    drug_start_col = "Drug Start Relative to Admission (seconds)"
    drug_end_col = "Drug End Relative to Admission (seconds)"

    SEDATION_DRUGS = [
        "propofol",
        "midazolam",
        "lorazepam",
        "dexmedetomidine",
    ]

    DELIRIUM_DRUGS = [
        "haloperidol",
        "quetiapine",
        "risperidone",
        "olanzapine",
        "dexmedetomidine",
    ]

    # Base frames
    patient_information = patient_information.lazy()
    ALL_STAYS = patient_information.select(STAY_KEY)
    ALL_STAYS_T0 = _build_t0(ALL_STAYS, t_0_per_stay=t_0_per_stay, t_0=t_0)

    # region respiratory (P/F ratio)
    resp_tf = (
        PAO2_FIO2_RATIO(t_0=t_0, t_0_per_stay=t_0_per_stay)
        .select(STAY_KEY, TIME_KEY, pf_ratio_col)
        .join(
            SPO2_FIO2_RATIO(t_0=t_0, t_0_per_stay=t_0_per_stay)
            .select(STAY_KEY, TIME_KEY, sf_ratio_col)
            .filter(pl.col(sf_ratio_col).is_not_null()),
            on=[STAY_KEY, TIME_KEY],
            how="outer",
            coalesce=True,
        )
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
            on=STAY_KEY,
            how="left",
        )
        .join(
            resp.select(STAY_KEY, TIME_KEY, oxygen_delivery_col).drop_nulls(),
            on=[STAY_KEY, TIME_KEY],
            how="outer",
            coalesce=True,
        )
        .with_columns(
            pl.when(
                pl.col(TIME_KEY) >= pl.col(vent_start_col),
                pl.col(vent_end_col).is_null()
                | (pl.col(TIME_KEY) < pl.col(vent_end_col)),
            )
            .then(True)
            .when(
                pl.col(oxygen_delivery_col).is_in(
                    [
                        "Mechanical ventilator",
                        "Continuous positive airway pressure/Bilevel positive airway pressure mask",
                        "High flow oxygen nasal cannula",
                    ]
                )
            )
            .then(True)
            .otherwise(False)
            .alias("ventilation")
        )
        .with_columns(
            _pf_ratio_points(
                pl.col(pf_ratio_col).cast(pl.Float64),
                pl.col("ventilation").fill_null(False),
            ).alias("pf_points"),
            _sf_ratio_points(
                pl.col(sf_ratio_col).cast(pl.Float64),
                pl.col("ventilation").fill_null(False),
            ).alias("sf_points"),
        )
        .group_by(STAY_KEY, "timeframe")
        .agg(
            pl.max("pf_points").alias("pf_ratio_points"),
            pl.max("sf_points").alias("sf_ratio_points"),
        )
        .select(
            STAY_KEY,
            "timeframe",
            # Use the arterial oxygen saturation (SpO2) to FIO2 ratio only when
            # the PaO2:FIO2 ratio is unavailable
            pl.coalesce(
                pl.col("pf_ratio_points"), pl.col("sf_ratio_points")
            ).alias("respiratory_points"),
        )
    )

    # region labs (platelets, bilirubin, creatinine)
    labs_tf = (
        labs.select(
            STAY_KEY, TIME_KEY, platelets_col, bilirubin_col, creatinine_col
        )
        .join(ALL_STAYS_T0, on=STAY_KEY, how="inner")
        .filter(pl.col(TIME_KEY) >= pl.col("T_0").sub(SECONDS_IN_1W))
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window_size))
        # attach rrt flag at measurement time using intervals
        .join(
            rrt.select(STAY_KEY, rrt_start_col, rrt_end_col),
            STAY_KEY,
            how="left",
        )
        .with_columns(
            pl.when(
                pl.col(TIME_KEY) >= pl.col(rrt_start_col),
                pl.col(rrt_end_col).is_null()
                | (pl.col(TIME_KEY) < pl.col(rrt_end_col)),
            )
            .then(True)
            .otherwise(False)
            .alias("rrt")
        )
        .group_by(STAY_KEY, "timeframe")
        .agg(
            _platelet_points(pl.col(platelets_col))
            .max()
            .alias("platelet_points"),
            _bilirubin_points(pl.col(bilirubin_col))
            .max()
            .alias("bilirubin_points"),
            _creatinine_points(
                pl.col(creatinine_col),
                pl.col("rrt").fill_null(False),
            )
            .max()
            .alias("creatinine_points"),
        )
    )

    # region vitals (GCS & MAP)
    vitals_tf = (
        vitals.select(STAY_KEY, TIME_KEY, gcs_col, gcs_motor_col, map_col)
        .join(ALL_STAYS_T0, on=STAY_KEY, how="inner")
        .filter(pl.col(TIME_KEY) >= pl.col("T_0").sub(SECONDS_IN_1W))
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window_size))
        # attach sedation / delirium flag at measurement time using intervals
        .join(
            meds.filter(
                pl.col(drug_ingredient_col).is_in(
                    SEDATION_DRUGS + DELIRIUM_DRUGS
                )
            ).select(
                STAY_KEY,
                drug_ingredient_col,
                drug_start_col,
                drug_end_col,
            ),
            on=STAY_KEY,
            how="left",
        )
        .with_columns(
            pl.when(
                pl.col(drug_ingredient_col).is_in(SEDATION_DRUGS),
                pl.col(TIME_KEY) >= pl.col(drug_start_col),
                pl.col(drug_end_col).is_null()
                | (pl.col(TIME_KEY) < pl.col(drug_end_col)),
            )
            .then(True)
            .otherwise(False)
            .alias("sedation"),
            pl.when(
                pl.col(drug_ingredient_col).is_in(DELIRIUM_DRUGS),
                pl.col(TIME_KEY) >= pl.col(drug_start_col),
                pl.col(drug_end_col).is_null()
                | (pl.col(TIME_KEY) < pl.col(drug_end_col)),
            )
            .then(True)
            .otherwise(False)
            .alias("delirium_treatment"),
        )
        .group_by(STAY_KEY, "timeframe")
        .agg(
            _gcs_points(
                pl.col(gcs_col),
                pl.col(gcs_motor_col),
                pl.col("sedation"),
                pl.col("delirium_treatment"),
            )
            .max()
            .alias("gcs_points"),
            _map_points(pl.col(map_col)).max().alias("map_points"),
        )
    )

    # region medications (vasoactive) -> doses in mcg/kg/min per timeframe
    vp_tf = get_vasopressor_points(
        meds=meds,
        patient_information=patient_information,
        ALL_STAYS_T0=ALL_STAYS_T0,
        window_size=window_size,
    )

    # region urine output (refactored to call URINE_OUTPUT)
    uo_base = URINE_OUTPUT(
        patient_information=patient_information,
        timeseries_inout=timeseries_inout,
        t_0=t_0,
        t_0_per_stay=t_0_per_stay,
        window_size=SECONDS_IN_1H,  # Return hourly values
    )
    # Apply rolling window assessment to compute urine output points
    # Aggregate back to day-level for integration with SOFA-2 daily scores
    uo_tf = (
        _uo_points(uo_base)
        .join(ALL_STAYS_T0, on=STAY_KEY, how="inner")
        .with_columns((pl.col("timeframe") * SECONDS_IN_1H).alias("timeframe"))
        .with_columns(timeframe=_assign_timeframe("timeframe", window_size))
        .group_by(STAY_KEY, "timeframe")
        .agg(pl.max("uo_points").alias("uo_points"))
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

    # region assemble
    out = base
    for part in [resp_tf, labs_tf, vitals_tf, vp_tf, uo_tf]:
        out = out.join(part, on=[STAY_KEY, "timeframe"], how="left")

    if forward_fill:
        out = out.with_columns(
            # forward-fill within stay at most 6 hours
            pl.col(
                "respiratory_points",
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
                pl.col("respiratory_points"),
                pl.col("platelet_points"),
                pl.col("bilirubin_points"),
                pl.col("cardiovascular_points"),
                pl.col("gcs_points"),
                pl.col("renal_points"),
                ignore_nulls=True,
            ).alias("SOFA-2 Score")
        )
        .select(
            STAY_KEY,
            "T_0",
            pl.col("timeframe").alias(timeframe_name),
            "SOFA-2 Score",
            pl.col("respiratory_points").alias("Respiration"),
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


__all__ = ["SOFA2"]
