"""
Base excess calculator from blood-gas pH and pCO2.

The module implements base excess calculations from blood-gas values:
- Base excess (BE)
- Standard base excess (SBE)

Formulas
--------
- BE = 0.02786 × PaCO2 × 10^(pH - 6.1) + 13.77 × pH − 124.58
- SBE = 0.9287 × HCO3- + 13.77 × pH − 124.58
"""

from typing import Optional

import polars as pl

from ...base import (
    Reference,
    STAY_KEY,
    TIME_KEY,
    _reference_t0,
    _relative_to_t0,
    _to_lazy,
    with_reference_time,
)
from ...common import (
    _validate_required_data,
    get_patient_information,
    get_timeseries_labs,
)
from .BICARBONATE import _bicarbonate_from_paCO2_and_pH, _improve_labs

BASE_EXCESS_COL          = "Base excess"
STANDARD_BASE_EXCESS_COL = "Standard base excess"


# region helpers
def _base_excess_from_paCO2_and_pH(pco2: pl.Expr, ph: pl.Expr) -> pl.Expr:
    """Calculate base excess from PaCO2 and pH."""
    return 0.02786 * pco2 * (10 ** (ph - 6.1)) + 13.77 * ph - 124.58


def _standard_base_excess_from_bicarbonate_and_pH(
    hco3: pl.Expr, ph: pl.Expr
) -> pl.Expr:
    """Calculate standard base excess from bicarbonate and pH."""
    return 0.9287 * hco3 + 13.77 * ph - 124.58


# region BASE_EXCESS
@with_reference_time
def BASE_EXCESS(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_labs: Optional[pl.LazyFrame] = None,
    *,
    ref: Reference,
) -> pl.LazyFrame:
    """
    Calculate base excess from blood-gas pH and pCO2.

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information. Loaded automatically if None.
        timeseries_labs : pl.LazyFrame, optional
            Laboratory timeseries data. Loaded automatically if None.

    Returns
    -------
        pl.LazyFrame
            Base excess timeseries with columns:
            - Global ICU Stay ID
            - Time Relative to Admission (seconds)
            - Base excess
    """
    if patient_information is None:
        patient_information = get_patient_information()
    if timeseries_labs is None:
        timeseries_labs = get_timeseries_labs()

    required = {
        "patient_information": patient_information,
        "timeseries_labs": timeseries_labs,
    }
    _validate_required_data("BASE_EXCESS", required)

    patient_information = _to_lazy(patient_information)
    timeseries_labs = _to_lazy(timeseries_labs)
    all_stays_t0 = _reference_t0(patient_information, ref)

    timeseries_labs = _improve_labs(timeseries_labs)

    base_excess = (
        timeseries_labs.with_columns(
            _base_excess_from_paCO2_and_pH(pl.col("pCO2"), pl.col("pH")).alias(
                BASE_EXCESS_COL
            )
        )
        .with_columns(
            pl.when(pl.col(BASE_EXCESS_COL).is_finite())
            .then(pl.col(BASE_EXCESS_COL))
            .otherwise(None)
            .alias(BASE_EXCESS_COL)
        )
        .select(STAY_KEY, TIME_KEY, BASE_EXCESS_COL)
    )

    base_excess = _relative_to_t0(base_excess, all_stays_t0, ref)

    return base_excess


# region STANDARD_BASE_EXCESS
@with_reference_time
def STANDARD_BASE_EXCESS(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_labs: Optional[pl.LazyFrame] = None,
    *,
    ref: Reference,
) -> pl.LazyFrame:
    """
    Calculate standard base excess from blood-gas pH and pCO2.

    Standard base excess is computed as 0.9287 × HCO3- + 13.77 × pH − 124.58.

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information. Loaded automatically if None.
        timeseries_labs : pl.LazyFrame, optional
            Laboratory timeseries data. Loaded automatically if None.

    Returns
    -------
        pl.LazyFrame
            Standard base excess timeseries with columns:
            - Global ICU Stay ID
            - Time Relative to Admission (seconds)
            - Standard base excess
    """
    if patient_information is None:
        patient_information = get_patient_information()
    if timeseries_labs is None:
        timeseries_labs = get_timeseries_labs()

    required = {
        "patient_information": patient_information,
        "timeseries_labs": timeseries_labs,
    }
    _validate_required_data("STANDARD_BASE_EXCESS", required)

    patient_information = _to_lazy(patient_information)
    timeseries_labs = _to_lazy(timeseries_labs)
    all_stays_t0 = _reference_t0(patient_information, ref)

    timeseries_labs = _improve_labs(timeseries_labs)

    standard_base_excess = (
        timeseries_labs.with_columns(
            _standard_base_excess_from_bicarbonate_and_pH(
                _bicarbonate_from_paCO2_and_pH(pl.col("pCO2"), pl.col("pH")),
                pl.col("pH"),
            ).alias(STANDARD_BASE_EXCESS_COL)
        )
        .with_columns(
            pl.when(pl.col(STANDARD_BASE_EXCESS_COL).is_finite())
            .then(pl.col(STANDARD_BASE_EXCESS_COL))
            .otherwise(None)
            .alias(STANDARD_BASE_EXCESS_COL)
        )
        .select(STAY_KEY, TIME_KEY, STANDARD_BASE_EXCESS_COL)
    )

    standard_base_excess = _relative_to_t0(standard_base_excess, all_stays_t0, ref)

    return standard_base_excess


__all__ = ["BASE_EXCESS", "STANDARD_BASE_EXCESS"]
