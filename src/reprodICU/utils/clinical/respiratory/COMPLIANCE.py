"""
Respiratory Compliance: calculate static and dynamic lung compliance from ventilator data.

This module implements compliance measurements:
- Static Compliance (Cstat): Measures elastic recoil of the lungs and chest wall
- Dynamic Compliance (Cdyn): Measures combined effects of elasticity and airway resistance

Output columns depend on the function:
- Static Compliance: Cstat in mL/cm H2O
- Dynamic Compliance: Cdyn in mL/cm H2O

Formulas
--------
- Static Compliance (Cstat):
    Cstat = Tidal Volume (mL) / (Plateau Pressure (Pplat) - PEEP)

- Dynamic Compliance (Cdyn):
    Cdyn = Tidal Volume (mL) / (Peak Inspiratory Pressure (PIP) - PEEP)
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
    columns = ["TV", "Pplat", "PIP", "PEEP"]
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


# region Compliance Calculations
@with_reference_time
def STATIC_COMPLIANCE(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_resp: Optional[pl.LazyFrame] = None,
    *,
    ref: Reference,
) -> pl.LazyFrame:
    """
    Calculate static lung compliance (Cstat) timeseries from respiratory timeseries.

    Static compliance measures the elastic recoil of the lungs and chest wall at
    zero flow conditions using the plateau pressure.

    Formula: Cstat = VT / (Pplat - PEEP)

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information. Loaded automatically if None.
        timeseries_resp : pl.LazyFrame, optional
            Respiratory timeseries data. Loaded automatically if None.

    Returns
    -------
        pl.LazyFrame
            Static Compliance timeseries with columns:
            - Global ICU Stay ID
            - Time Relative to Admission (seconds)
            - Static Compliance (mL/cm H2O)
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
    _validate_required_data("STATIC_COMPLIANCE", required)

    patient_information = _to_lazy(patient_information)
    timeseries_resp = _to_lazy(timeseries_resp)
    all_stays_t0 = _reference_t0(patient_information, ref)

    timeseries_resp = _improve_resp(timeseries_resp)

    static_compliance = timeseries_resp.with_columns(
        pl.when(pl.col("Pplat") > pl.col("PEEP"))
        .then(pl.col("TV") / (pl.col("Pplat") - pl.col("PEEP")))
        .alias("Static Compliance (mL/cm H2O)")
    ).select(STAY_KEY, TIME_KEY, "Static Compliance (mL/cm H2O)")

    static_compliance = _relative_to_t0(static_compliance, all_stays_t0, ref)

    return static_compliance


@with_reference_time
def DYNAMIC_COMPLIANCE(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_resp: Optional[pl.LazyFrame] = None,
    *,
    ref: Reference,
) -> pl.LazyFrame:
    """
    Calculate dynamic lung compliance (Cdyn) timeseries from respiratory timeseries.

    Dynamic compliance measures the combined effects of lung elasticity and
    airway resistance using the peak inspiratory pressure.

    Formula: Cdyn = VT / (PIP - PEEP)

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information. Loaded automatically if None.
        timeseries_resp : pl.LazyFrame, optional
            Respiratory timeseries data. Loaded automatically if None.

    Returns
    -------
        pl.LazyFrame
            Dynamic Compliance timeseries with columns:
            - Global ICU Stay ID
            - Time Relative to Admission (seconds)
            - Dynamic Compliance (mL/cm H2O)
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
    _validate_required_data("DYNAMIC_COMPLIANCE", required)

    patient_information = _to_lazy(patient_information)
    timeseries_resp = _to_lazy(timeseries_resp)
    all_stays_t0 = _reference_t0(patient_information, ref)

    timeseries_resp = _improve_resp(timeseries_resp)

    dynamic_compliance = timeseries_resp.with_columns(
        pl.when(pl.col("PIP") > pl.col("PEEP"))
        .then(pl.col("TV") / (pl.col("PIP") - pl.col("PEEP")))
        .alias("Dynamic Compliance (mL/cm H2O)")
    ).select(STAY_KEY, TIME_KEY, "Dynamic Compliance (mL/cm H2O)")

    dynamic_compliance = _relative_to_t0(dynamic_compliance, all_stays_t0, ref)

    return dynamic_compliance


__all__ = ["STATIC_COMPLIANCE", "DYNAMIC_COMPLIANCE"]
