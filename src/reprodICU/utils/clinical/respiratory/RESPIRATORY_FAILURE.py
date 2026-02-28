"""
Respiratory failure classification: identify acute hypoxemic and acute hypercapnic respiratory failure.

This module implements respiratory failure classification based on arterial blood gas parameters:
- Acute Hypoxemic Respiratory Failure: SpO2 < 88% OR PaO2 < 60 mmHg OR SaO2 < 88%
  Results from V/Q mismatch, shunt, hypoventilation, diffusion limitation, or low inspired oxygen.
- Acute Hypercapnic Respiratory Failure: PaCO2 ≥ 45 mmHg AND pH < 7.35
  Results from alveolar hypoventilation, increased fraction of dead space, or increased CO2 production.
"""

from typing import Optional

import polars as pl

from ...common import (
    _build_t0,
    _to_lazy,
    get_patient_information,
    get_timeseries_labs,
    get_timeseries_vitals,
)

respiratory_failure_type_col = "Respiratory Failure Type"


# region helpers
def _improve_labs(labs: pl.LazyFrame) -> pl.LazyFrame:
    return labs.select(
        "Global ICU Stay ID",
        "Time Relative to Admission (seconds)",
        # PaO2
        pl.when(
            pl.col("Oxygen")
            .struct.field("system")
            .is_in(["Blood arterial", "Blood"])
            | pl.col("Oxygen").struct.field("system").is_null()
        )
        .then(pl.col("Oxygen").struct.field("value"))
        .otherwise(None)
        .alias("PaO2"),
        # PaCO2
        pl.when(
            pl.col("Carbon dioxide")
            .struct.field("system")
            .is_in(["Blood arterial", "Blood"])
            | pl.col("Carbon dioxide").struct.field("system").is_null()
        )
        .then(pl.col("Carbon dioxide").struct.field("value"))
        .otherwise(None)
        .alias("PaCO2"),
        # pH
        pl.when(
            pl.col("pH")
            .struct.field("system")
            .is_in(["Blood arterial", "Blood"])
            | pl.col("pH").struct.field("system").is_null()
        )
        .then(pl.col("pH").struct.field("value"))
        .otherwise(None)
        .alias("pH"),
        # SaO2
        pl.when(
            pl.col("Oxygen saturation")
            .struct.field("system")
            .is_in(["Blood arterial", "Blood"])
            | pl.col("Oxygen saturation").struct.field("system").is_null()
        )
        .then(pl.col("Oxygen saturation").struct.field("value"))
        .otherwise(None)
        .alias("SaO2"),
    ).filter(
        pl.any_horizontal(
            pl.col(col).is_not_null() for col in ["PaO2", "PaCO2", "pH", "SaO2"]
        )
    )


def _improve_vitals(vitals: pl.LazyFrame) -> pl.LazyFrame:
    return vitals.filter(
        pl.col("Peripheral oxygen saturation").is_not_null(),
        pl.col("Peripheral oxygen saturation").is_between(0, 100),
    ).select(
        "Global ICU Stay ID",
        "Time Relative to Admission (seconds)",
        pl.col("Peripheral oxygen saturation").alias("SpO2"),
    )


# endregion helpers


# region Respiratory Failure
def RESPIRATORY_FAILURE(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_vitals: Optional[pl.LazyFrame] = None,
    timeseries_labs: Optional[pl.LazyFrame] = None,
    *,
    t_0: Optional[int] = 0,
    t_0_per_stay: Optional[pl.LazyFrame] = None,
) -> pl.LazyFrame:
    """
    Classify respiratory failure into acute hypoxemic and acute hypercapnic types.

    Identifies episodes of acute respiratory failure based on arterial blood gas parameters
    and oxygen saturation measurements.

    Steps:
        1. Extract PaO2, SaO2, PaCO2, and pH from laboratory timeseries.
           Extract SpO2 from vital signs timeseries.
        2. Classify as acute hypoxemic respiratory failure if SpO2 < 88% or PaO2 < 60 mmHg or SaO2 < 88%.
        3. Classify as acute hypercapnic respiratory failure if PaCO2 ≥ 45 mmHg AND pH < 7.35.

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information. Loaded automatically if None.
        timeseries_vitals : pl.LazyFrame, optional
            Vital signs timeseries data. Loaded automatically if None.
        timeseries_labs : pl.LazyFrame, optional
            Laboratory timeseries data. Loaded automatically if None.
        t_0 : int, optional
            Scalar reference time (seconds from admission). Defaults to 0 (admission).
            Ignored when t_0_per_stay is provided.
        t_0_per_stay : pl.LazyFrame, optional
            Per-stay T_0 overrides with columns [Global ICU Stay ID, T_0].

    Returns
    -------
        pl.LazyFrame
            Respiratory failure observations with columns:
            - Global ICU Stay ID
            - Time Relative to Admission (seconds)
            - {respiratory_failure_type_col}: Type of respiratory failure:
                "Acute Hypoxemic", "Acute Hypercapnic", "Mixed", or None if insufficient data
            - PaO2: Arterial oxygen partial pressure (mmHg)
            - PaCO2: Arterial carbon dioxide partial pressure (mmHg)
            - pH: Arterial pH
            - SaO2: Arterial oxygen saturation (%)
            - SpO2: Peripheral oxygen saturation (%)
    """
    # Load defaults if not provided
    if patient_information is None:
        patient_information = get_patient_information()
    if timeseries_vitals is None:
        timeseries_vitals = get_timeseries_vitals()
    if timeseries_labs is None:
        timeseries_labs = get_timeseries_labs()

    STAY_KEY = "Global ICU Stay ID"
    TIME_KEY = "Time Relative to Admission (seconds)"
    LAB_COLS = ["PaO2", "PaCO2", "pH", "SaO2", "SpO2"]

    patient_information = _to_lazy(patient_information)
    timeseries_vitals = _to_lazy(timeseries_vitals)
    timeseries_labs = _to_lazy(timeseries_labs)
    t_0_per_stay = _to_lazy(t_0_per_stay) if t_0_per_stay is not None else None

    all_stays = patient_information.select(STAY_KEY)
    all_stays_t0 = _build_t0(all_stays, t_0_per_stay, t_0)

    timeseries_labs = _improve_labs(timeseries_labs)
    timeseries_vitals = _improve_vitals(timeseries_vitals)

    # Merge labs and vitals to have all relevant data available
    resp_failure = (
        timeseries_labs.join(
            timeseries_vitals,
            on=[STAY_KEY, TIME_KEY],
            how="outer",
            coalesce=True,
        )
        .with_columns(
            # Classify Acute Hypoxemic: SpO2 < 88% OR PaO2 < 60 mmHg OR SaO2 < 88%
            pl.when(
                (pl.col("SpO2") < 88)
                | (pl.col("PaO2") < 60)
                | (pl.col("SaO2") < 88)
            )
            .then(True)
            .when(
                (pl.col("PaO2").is_not_null())
                | (pl.col("SpO2").is_not_null())
                | (pl.col("SaO2").is_not_null())
            )
            .then(False)
            .otherwise(None)
            .alias("is_hypoxemic"),
            # Classify Acute Hypercapnic: PaCO2 ≥ 45 mmHg AND pH < 7.35
            pl.when(
                (pl.col("PaCO2") >= 45)
                & (pl.col("pH") < 7.35)
            )
            .then(True)
            .when(
                (pl.col("PaCO2").is_not_null())
                & (pl.col("pH").is_not_null())
            )
            .then(False)
            .otherwise(None)
            .alias("is_hypercapnic"),
        )
        # Combine into single categorical column
        .with_columns(
            pl.when(pl.col("is_hypoxemic") & pl.col("is_hypercapnic"))
            .then(pl.lit("Mixed"))
            .when(pl.col("is_hypoxemic"))
            .then(pl.lit("Acute Hypoxemic"))
            .when(pl.col("is_hypercapnic"))
            .then(pl.lit("Acute Hypercapnic"))
            .otherwise(None)
            .alias(respiratory_failure_type_col)
        )
        .filter(pl.col(respiratory_failure_type_col).is_not_null())
        .select(STAY_KEY, TIME_KEY, *LAB_COLS, respiratory_failure_type_col)
    ) # fmt: skip

    if (t_0 != 0) or (t_0_per_stay is not None):
        resp_failure = (
            resp_failure.join(all_stays_t0, on=STAY_KEY, how="inner")
            .with_columns(
                pl.col(TIME_KEY)
                .sub(pl.col("T_0"))
                .alias("Time Relative to T_0 (seconds)")
            )
            .drop("T_0")
        )

    return resp_failure


__all__ = ["RESPIRATORY_FAILURE"]
