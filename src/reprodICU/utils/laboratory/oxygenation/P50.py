"""
P50: calculate the partial pressure of oxygen (pO2) at which hemoglobin is 50% saturated.

P50 represents the pO2 at which hemoglobin saturation is exactly 50%, derived
from the oxygen-hemoglobin dissociation curve using the Hill equation.
A lower P50 indicates increased affinity (left-shifted curve), while a higher
P50 indicates decreased affinity (right-shifted curve).

Output column:
- P50: P50 value in mmHg

Formula
-------
- P50:
    P50 (in mmHg) = pO2 * ((100 - sO2) / sO2) ^ (1/n)
    where:
    - pO2: Arterial partial pressure of oxygen in mmHg
    - sO2: Arterial oxygen saturation in percentage (0-100%)
    - n: Hill coefficient (~2.8)

Standard conditions (37°C, pH 7.4, PCO2 40 mmHg):
- Normal P50 ≈ 26-27 mmHg

Sources
-------
- Doyle DJ. A simple method to calculate P50 from a single blood sample.
  Int J Clin Monit Comput. 1997;14(2):109-11. doi: 10.1007/BF03356585. PMID: 9336736.
"""

from typing import Optional

import polars as pl

from ...base import (
    STAY_KEY,
    TIME_KEY,
    Reference,
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

HILL_COEFFICIENT = 2.711


# region helpers
def _improve_labs(labs: pl.LazyFrame) -> pl.LazyFrame:
    return (
        labs.filter(
            pl.col("Oxygen")
            .struct.field("system")
            .eq_missing(pl.col("Oxygen saturation").struct.field("system")),
        )
        .select(
            STAY_KEY,
            TIME_KEY,
            extract_struct_value("Oxygen", ["Blood"]).alias("pO2"),
            extract_struct_value("Oxygen saturation", ["Blood"]).alias("sO2"),
        )
        .filter(
            pl.col("pO2").is_between(20, 700),  # Physiologically valid range
            pl.col("sO2").is_between(10, 100),  # Percentage range
        )
    )


# region P50
@with_reference_time
def P50(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_labs: Optional[pl.LazyFrame] = None,
    *,
    hill_coefficient: float = HILL_COEFFICIENT,
    ref: Reference,
) -> pl.LazyFrame:
    """
    Calculate P50 (oxygen pressure at 50% hemoglobin saturation) timeseries from blood gas data.

    P50 represents the partial pressure of oxygen at which hemoglobin is 50% saturated
    and is a measure of hemoglobin's affinity for oxygen. It is calculated from arterial
    blood gas measurements using the Hill equation.

    Formula: P50 = pO2 * ((100 - sO2) / sO2) ^ (1/n)

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information. Loaded automatically if None.
        timeseries_labs : pl.LazyFrame, optional
            Laboratory timeseries data. Loaded automatically if None.
        hill_coefficient : float, optional
            Hill coefficient n of the oxygen dissociation curve. Defaults to 2.711.

    Returns
    -------
        pl.LazyFrame
            P50 timeseries with columns:
            - Global ICU Stay ID
            - Time Relative to Admission (seconds)
            - P50 (mmHg)
    """
    # Load defaults if not provided
    if patient_information is None:
        patient_information = get_patient_information()
    if timeseries_labs is None:
        timeseries_labs = get_timeseries_labs()

    # Validate all required data is available
    required = {
        "patient_information": patient_information,
        "timeseries_labs": timeseries_labs,
    }
    _validate_required_data("P50", required)

    patient_information = _to_lazy(patient_information)
    timeseries_labs = _to_lazy(timeseries_labs)
    all_stays_t0 = _reference_t0(patient_information, ref)

    timeseries_labs = _improve_labs(timeseries_labs)

    p50 = (
        timeseries_labs.with_columns(
            pl.col("pO2")
            .mul(
                (100 - pl.col("sO2"))
                .truediv(pl.col("sO2"))
                .pow(1 / hill_coefficient)
            )
            .alias("P50 (mmHg)")
        )
        .with_columns(
            pl.when(
                pl.col("P50 (mmHg)").is_finite(),
                pl.col("P50 (mmHg)") > 0
            )
            .then(pl.col("P50 (mmHg)"))
            .otherwise(None)
            .alias("P50 (mmHg)")
        )
        .select(STAY_KEY, TIME_KEY, "P50 (mmHg)")
    )

    p50 = _relative_to_t0(p50, all_stays_t0, ref)

    return p50


__all__ = ["P50"]
