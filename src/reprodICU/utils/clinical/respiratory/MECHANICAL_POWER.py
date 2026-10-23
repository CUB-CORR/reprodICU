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

from ...base import (
    Reference,
    SECONDS_IN_6H,
    STAY_KEY,
    TIME_KEY,
    _reference_t0,
    _relative_to_t0,
    _to_lazy,
    with_reference_time,
)
from ...common import (
    FILTER_NOT_NULL,
    _validate_required_data,
    get_patient_information,
    get_timeseries_respiratory,
)
from ...core import RESPIRATORY_PARAMS


# region helpers
def _improve_resp(resp: pl.LazyFrame) -> pl.LazyFrame:
    columns = ["TV", "RR", "Pplat", "PIP", "PEEP", "FiO2"]
    return (
        RESPIRATORY_PARAMS(timeseries_resp=resp)
        .select(STAY_KEY, TIME_KEY, *columns)
        .pipe(FILTER_NOT_NULL, columns)
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
@with_reference_time
def MECHANICAL_POWER(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_resp: Optional[pl.LazyFrame] = None,
    *,
    ref: Reference,
) -> pl.LazyFrame:
    """
    Calculate mechanical power timeseries from respiratory timeseries.

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information. Loaded automatically if None.
        timeseries_resp : pl.LazyFrame, optional
            Respiratory timeseries data. Loaded automatically if None.

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
    _validate_required_data("MECHANICAL_POWER", required)

    patient_information = _to_lazy(patient_information)
    timeseries_resp = _to_lazy(timeseries_resp)
    all_stays_t0 = _reference_t0(patient_information, ref)

    timeseries_resp = _improve_resp(timeseries_resp)

    # MP (in J/minutes) = 0.098 x V_T (in Liters) x RR x (Ppeak - 0.5 x DP)
    mechanical_power = (
        timeseries_resp.with_columns(
            pl.col("TV").truediv(1000).alias("TV_Liters"),
            (pl.col("Pplat") - pl.col("PEEP")).alias("DP"),  # Driving Pressure
        )
        .with_columns(
            (
                0.098
                * pl.col("TV_Liters")
                * pl.col("RR")
                * (pl.col("PIP") - 0.5 * pl.col("DP"))
            ).alias("Mechanical Power (J/min)")
        )
        .select(STAY_KEY, TIME_KEY, "Mechanical Power (J/min)")
    )

    mechanical_power = _relative_to_t0(mechanical_power, all_stays_t0, ref)

    return mechanical_power


__all__ = ["MECHANICAL_POWER"]
