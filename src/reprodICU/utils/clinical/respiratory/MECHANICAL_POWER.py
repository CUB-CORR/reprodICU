"""
Mechanical Power: calculate mechanical power from ventilator data.

This module implements mechanical power measurement:
- Mechanical Power (MP): Measures the energy delivered to the respiratory system
  by the ventilator per unit of time.

Output column:
- Mechanical Power: MP in J/min

Formula
-------
- Mechanical Power (MP):
    MP (in J/min) = 0.098 x V_T (in Liters) x RR x (Ppeak - 0.5 x DP)
    where:
    - V_T: Tidal Volume in Liters
    - RR: Respiratory Rate in breaths per minute
    - Ppeak: Peak Inspiratory Pressure in cm H2O
    - DP: Driving Pressure = Ppeak - PEEP
"""

from typing import Optional

import polars as pl

from ...common import (
    _build_t0,
    _to_lazy,
    get_patient_information,
    get_timeseries_respiratory,
)

SECONDS_IN_6H = 6 * 60 * 60
STAY_KEY = "Global ICU Stay ID"
TIME_KEY = "Time Relative to Admission (seconds)"


# region helpers
def _improve_resp(resp: pl.LazyFrame) -> pl.LazyFrame:
    columns = ["VT", "RR", "Pplat", "PIP", "PEEP", "FiO2"]
    return (
        resp.select(
            "Global ICU Stay ID",
            "Time Relative to Admission (seconds)",
            pl.max_horizontal(
                "Tidal volume setting Ventilator",
                "Tidal volume.spontaneous --on ventilator",
                "Tidal volume.spontaneous+mechanical --on ventilator",
            ).alias("VT"),
            pl.max_horizontal(
                "Breath rate setting Ventilator",
                "Breath rate spontaneous and mechanical --on ventilator",
            ).alias("RR"),
            pl.col(
                "Pressure.plateau Respiratory system airway --on ventilator"
            ).alias("Pplat"),
            pl.col(
                "Pressure.max Respiratory system airway --on ventilator"
            ).alias("PIP"),
            pl.max_horizontal(
                "Positive end expiratory pressure setting Ventilator",
                "PEEP Respiratory system",
            ).alias("PEEP"),
            pl.max_horizontal(
                "Oxygen/Total gas setting [Volume Fraction] Ventilator",
                "Oxygen/Gas total [Pure volume fraction] Inhaled gas",
            ).alias("FiO2"),
        )
        .with_columns(
            pl.when(pl.col("FiO2").is_between(0, 1))
            .then(pl.col("FiO2") * 100)
            .when(pl.col("FiO2").is_between(1, 100))
            .then(pl.col("FiO2"))
            .otherwise(None)
            .alias("FiO2"),
        )
        .filter(pl.any_horizontal(pl.col(col).is_not_null() for col in columns))
        .sort(STAY_KEY, TIME_KEY)
        .with_columns(
            pl.when(pl.col(col).is_not_null())
            .then(pl.col(TIME_KEY))
            .otherwise(None)
            .forward_fill()
            .over(STAY_KEY)
            .alias(f"_ff_time_{col}")
            for col in columns
        )
        .with_columns(
            pl.when(
                pl.col(TIME_KEY) <= (pl.col(f"_ff_time_{col}") + SECONDS_IN_6H)
            )
            .then(pl.col(col).forward_fill().over(STAY_KEY))
            .otherwise(None)
            .alias(col)
            for col in columns
        )
        .drop(f"_ff_time_{col}" for col in columns)
        .filter(pl.all_horizontal(pl.col(col).is_not_null() for col in columns))
    )


# region mechanical power
def MECHANICAL_POWER(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_resp: Optional[pl.LazyFrame] = None,
    *,
    t_0: Optional[int] = 0,
    t_0_per_stay: Optional[pl.LazyFrame] = None,
) -> pl.LazyFrame:
    """
    Calculate mechanical power timeseries from respiratory timeseries.

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information. Loaded automatically if None.
        timeseries_resp : pl.LazyFrame, optional
            Respiratory timeseries data. Loaded automatically if None.
        t_0 : int, optional
            Scalar reference time (seconds from admission). Defaults to 0 (admission).
            Ignored when t_0_per_stay is provided.
        t_0_per_stay : pl.LazyFrame, optional
            Per-stay T_0 overrides with columns [Global ICU Stay ID, T_0].

    Returns
    -------
        pl.LazyFrame
            Mechanical Power timeseries with columns:
            - Global ICU Stay ID
            - Time Relative to Admission (seconds)
            - Mechanical Power (J/min)
    """
    # Load defaults if not provided
    if patient_information is None:
        patient_information = get_patient_information()
    if timeseries_resp is None:
        timeseries_resp = get_timeseries_respiratory()

    # Validate all required data is available
    required = {
        "patient_information": patient_information,
        "timeseries_resp": timeseries_resp,
    }

    missing = [name for name, data in required.items() if data is None]
    if missing:
        raise ValueError(
            f"Cannot compute MECHANICAL_POWER: Missing required datasets: {', '.join(missing)}. "
            f"Ensure they are configured in ~/.reprodICU/PATHS.yaml or provide them explicitly."
        )

    patient_information = _to_lazy(patient_information)
    timeseries_resp = _to_lazy(timeseries_resp)
    t_0_per_stay = _to_lazy(t_0_per_stay) if t_0_per_stay is not None else None

    all_stays = patient_information.select(STAY_KEY)
    all_stays_t0 = _build_t0(all_stays, t_0_per_stay, t_0)

    timeseries_resp = _improve_resp(timeseries_resp)

    # MP (in J/minutes) = 0.098 x V_T (in Liters) x RR x (Ppeak - 0.5 x DP)
    mechanical_power = (
        timeseries_resp.with_columns(
            pl.col("VT").truediv(1000).alias("VT_Liters"),
            (pl.col("PIP") - pl.col("PEEP")).alias("DP"),  # Driving Pressure
        )
        .with_columns(
            (
                0.098
                * pl.col("VT_Liters")
                * pl.col("RR")
                * (pl.col("PIP") - 0.5 * pl.col("DP"))
            ).alias("Mechanical Power (J/min)")
        )
        .select(STAY_KEY, TIME_KEY, "Mechanical Power (J/min)")
    )

    if (t_0 != 0) or (t_0_per_stay is not None):
        mechanical_power = (
            mechanical_power.join(all_stays_t0, on=STAY_KEY, how="inner")
            .with_columns(
                pl.col(TIME_KEY)
                .sub(pl.col("T_0"))
                .alias("Time Relative to T_0 (seconds)")
            )
            .drop("T_0")
        )

    return mechanical_power


__all__ = ["MECHANICAL_POWER"]
