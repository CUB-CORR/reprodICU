"""
Bicarbonate calculators from blood-gas pH and pCO2.

The module implements three related acid-base quantities from blood-gas values:
- Actual bicarbonate (HCO3-)
- Standard bicarbonate (SBC)
- Total CO2 (TCO2)

Formulas
--------
- HCO3- = 0.0307 × PaCO2 × 10^(pH - 6.1)
- SBC = HCO3- with PaCO2 fixed at 40 mmHg
- TCO2 = HCO3- + (PaCO2 × 0.03)
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
    extract_struct_value,
    get_patient_information,
    get_timeseries_labs,
)

BICARBONATE_COL = "Bicarbonate"
STANDARD_BICARBONATE_COL = "Standard Bicarbonate"
TOTAL_CO2_COL = "Total CO2"


# region helpers
def _improve_labs(labs: pl.LazyFrame) -> pl.LazyFrame:
    return labs.select(
        STAY_KEY,
        TIME_KEY,
        extract_struct_value("pH", ["Blood"]).alias("pH"),
        extract_struct_value("Carbon dioxide", ["Blood"]).alias("pCO2"),
    ).drop_nulls(["pH", "pCO2"])


def _bicarbonate_from_paCO2_and_pH(pco2: pl.Expr, ph: pl.Expr) -> pl.Expr:
    """Calculate HCO3- from PaCO2 and pH.

    HCO3- = 0.0307 × PaCO2 × 10^(pH - 6.1)
    """
    return 0.0307 * pco2 * (10 ** (ph - 6.1))


# region BICARBONATE
@with_reference_time
def BICARBONATE(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_labs: Optional[pl.LazyFrame] = None,
    *,
    ref: Reference,
) -> pl.LazyFrame:
    """
    Calculate bicarbonate values from blood-gas pH and pCO2.

    HCO3- = 0.0307 × PaCO2 × 10^(pH - 6.1)

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information. Loaded automatically if None.
        timeseries_labs : pl.LazyFrame, optional
            Laboratory timeseries data. Loaded automatically if None.

    Returns
    -------
        pl.LazyFrame
            Bicarbonate timeseries with columns:
            - Global ICU Stay ID
            - Time Relative to Admission (seconds)
            - Bicarbonate
    """
    if patient_information is None:
        patient_information = get_patient_information()
    if timeseries_labs is None:
        timeseries_labs = get_timeseries_labs()

    required = {
        "patient_information": patient_information,
        "timeseries_labs": timeseries_labs,
    }
    _validate_required_data("BICARBONATE", required)

    patient_information = _to_lazy(patient_information)
    timeseries_labs = _to_lazy(timeseries_labs)
    all_stays_t0 = _reference_t0(patient_information, ref)

    timeseries_labs = _improve_labs(timeseries_labs)

    bicarbonate = (
        timeseries_labs.with_columns(
            _bicarbonate_from_paCO2_and_pH(pl.col("pCO2"), pl.col("pH")).alias(
                BICARBONATE_COL
            )
        )
        .with_columns(
            pl.when(pl.col(BICARBONATE_COL).is_finite())
            .then(pl.col(BICARBONATE_COL))
            .otherwise(None)
            .alias(BICARBONATE_COL)
        )
        .select(STAY_KEY, TIME_KEY, BICARBONATE_COL)
    )

    bicarbonate = _relative_to_t0(bicarbonate, all_stays_t0, ref)

    return bicarbonate


# region STANDARD_BICARBONATE
@with_reference_time
def STANDARD_BICARBONATE(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_labs: Optional[pl.LazyFrame] = None,
    *,
    ref: Reference,
) -> pl.LazyFrame:
    """
    Calculate standard bicarbonate from blood-gas pH and pCO2 = 40mmHg.

    HCO3-_std = 0.0307 × 40 × 10^(pH - 6.1)

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information. Loaded automatically if None.
        timeseries_labs : pl.LazyFrame, optional
            Laboratory timeseries data. Loaded automatically if None.

    Returns
    -------
        pl.LazyFrame
            Standard bicarbonate timeseries with columns:
            - Global ICU Stay ID
            - Time Relative to Admission (seconds)
            - Standard Bicarbonate
    """
    if patient_information is None:
        patient_information = get_patient_information()
    if timeseries_labs is None:
        timeseries_labs = get_timeseries_labs()

    required = {
        "patient_information": patient_information,
        "timeseries_labs": timeseries_labs,
    }
    _validate_required_data("STANDARD_BICARBONATE", required)

    patient_information = _to_lazy(patient_information)
    timeseries_labs = _to_lazy(timeseries_labs)
    all_stays_t0 = _reference_t0(patient_information, ref)

    timeseries_labs = _improve_labs(timeseries_labs)

    standard_bicarbonate = (
        timeseries_labs.with_columns(
            _bicarbonate_from_paCO2_and_pH(pl.lit(40), pl.col("pH")).alias(
                STANDARD_BICARBONATE_COL
            )
        )
        .with_columns(
            pl.when(pl.col(STANDARD_BICARBONATE_COL).is_finite())
            .then(pl.col(STANDARD_BICARBONATE_COL))
            .otherwise(None)
            .alias(STANDARD_BICARBONATE_COL)
        )
        .select(STAY_KEY, TIME_KEY, STANDARD_BICARBONATE_COL)
    )

    standard_bicarbonate = _relative_to_t0(standard_bicarbonate, all_stays_t0, ref)

    return standard_bicarbonate


# region TOTAL_CO2
@with_reference_time
def TOTAL_CO2(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_labs: Optional[pl.LazyFrame] = None,
    *,
    ref: Reference,
) -> pl.LazyFrame:
    """
    Calculate total CO2 from blood-gas pH and pCO2.

    tCO2 = HCO3- + (PaCO2 × 0.03)

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information. Loaded automatically if None.
        timeseries_labs : pl.LazyFrame, optional
            Laboratory timeseries data. Loaded automatically if None.

    Returns
    -------
        pl.LazyFrame
            Total CO2 timeseries with columns:
            - Global ICU Stay ID
            - Time Relative to Admission (seconds)
            - Total CO2
    """
    if patient_information is None:
        patient_information = get_patient_information()
    if timeseries_labs is None:
        timeseries_labs = get_timeseries_labs()

    required = {
        "patient_information": patient_information,
        "timeseries_labs": timeseries_labs,
    }
    _validate_required_data("TOTAL_CO2", required)

    patient_information = _to_lazy(patient_information)
    timeseries_labs = _to_lazy(timeseries_labs)
    all_stays_t0 = _reference_t0(patient_information, ref)

    timeseries_labs = _improve_labs(timeseries_labs)

    total_co2 = (
        timeseries_labs.with_columns(
            (
                _bicarbonate_from_paCO2_and_pH(pl.col("pCO2"), pl.col("pH"))
                + (pl.col("pCO2") * 0.03)
            ).alias(TOTAL_CO2_COL)
        )
        .with_columns(
            pl.when(pl.col(TOTAL_CO2_COL).is_finite())
            .then(pl.col(TOTAL_CO2_COL))
            .otherwise(None)
            .alias(TOTAL_CO2_COL)
        )
        .select(STAY_KEY, TIME_KEY, TOTAL_CO2_COL)
    )

    total_co2 = _relative_to_t0(total_co2, all_stays_t0, ref)

    return total_co2


__all__ = ["BICARBONATE", "STANDARD_BICARBONATE", "TOTAL_CO2"]
