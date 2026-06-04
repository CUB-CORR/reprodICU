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

from ...common import (
    _build_t0,
    _to_lazy,
    _validate_required_data,
    get_patient_information,
    get_timeseries_labs,
)
from .BICARBONATE import _bicarbonate_from_paCO2_and_pH, _improve_labs

STAY_KEY = "Global ICU Stay ID"
TIME_KEY = "Time Relative to Admission (seconds)"

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
def BASE_EXCESS(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_labs: Optional[pl.LazyFrame] = None,
    *,
    t_0: Optional[int] = 0,
    t_0_per_stay: Optional[pl.LazyFrame] = None,
) -> pl.LazyFrame:
    """
    Calculate base excess from blood-gas pH and pCO2.

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information. Loaded automatically if None.
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
    _validate_required_data(concept="BASE_EXCESS", required_data=required)

    patient_information = _to_lazy(patient_information)
    timeseries_labs = _to_lazy(timeseries_labs)
    t_0_per_stay = _to_lazy(t_0_per_stay) if t_0_per_stay is not None else None

    all_stays = patient_information.select(STAY_KEY)
    all_stays_t0 = _build_t0(all_stays, t_0_per_stay, t_0)

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

    if (t_0 != 0) or (t_0_per_stay is not None):
        base_excess = (
            base_excess.join(all_stays_t0, on=STAY_KEY, how="inner")
            .with_columns(
                pl.col(TIME_KEY)
                .sub(pl.col("T_0"))
                .alias("Time Relative to T_0 (seconds)")
            )
            .drop("T_0")
        )

    return base_excess


# region STANDARD_BASE_EXCESS
def STANDARD_BASE_EXCESS(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_labs: Optional[pl.LazyFrame] = None,
    *,
    t_0: Optional[int] = 0,
    t_0_per_stay: Optional[pl.LazyFrame] = None,
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
        t_0 : int, optional
            Scalar reference time (seconds from admission). Defaults to 0 (admission).
            Ignored when t_0_per_stay is provided.
        t_0_per_stay : pl.LazyFrame, optional
            Per-stay T_0 overrides with columns [Global ICU Stay ID, T_0].

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
    _validate_required_data(concept="STANDARD_BASE_EXCESS", required_data=required) # fmt: skip

    patient_information = _to_lazy(patient_information)
    timeseries_labs = _to_lazy(timeseries_labs)
    t_0_per_stay = _to_lazy(t_0_per_stay) if t_0_per_stay is not None else None

    all_stays = patient_information.select(STAY_KEY)
    all_stays_t0 = _build_t0(all_stays, t_0_per_stay, t_0)

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

    if (t_0 != 0) or (t_0_per_stay is not None):
        standard_base_excess = (
            standard_base_excess.join(all_stays_t0, on=STAY_KEY, how="inner")
            .with_columns(
                pl.col(TIME_KEY)
                .sub(pl.col("T_0"))
                .alias("Time Relative to T_0 (seconds)")
            )
            .drop("T_0")
        )

    return standard_base_excess


__all__ = ["BASE_EXCESS", "STANDARD_BASE_EXCESS"]
