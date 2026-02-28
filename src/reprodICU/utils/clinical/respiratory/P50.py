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

from ...common import (
    _build_t0,
    _to_lazy,
    get_patient_information,
    get_timeseries_labs,
)

STAY_KEY = "Global ICU Stay ID"
TIME_KEY = "Time Relative to Admission (seconds)"
HILL_COEFFICIENT = 2.711

SECONDS_IN_4H = 4 * 60 * 60


# region helpers
def _improve_labs(labs: pl.LazyFrame) -> pl.LazyFrame:
    return labs.select(
        STAY_KEY,
        TIME_KEY,
        pl.when(
            pl.col("Oxygen")
            .struct.field("system")
            .is_in(["Blood arterial", "Blood venous", "Blood"])
            | pl.col("Oxygen").struct.field("system").is_null()
        )
        .then(pl.col("Oxygen").struct.field("value"))
        .otherwise(None)
        .alias("pO2"),
        pl.when(
            pl.col("Oxygen saturation")
            .struct.field("system")
            .is_in(["Blood arterial", "Blood venous", "Blood"])
            | pl.col("Oxygen saturation").struct.field("system").is_null()
        )
        .then(pl.col("Oxygen saturation").struct.field("value"))
        .otherwise(None)
        .alias("sO2"),
    ).filter(
        pl.col("Oxygen").struct.field("system")
        == pl.col("Oxygen saturation").struct.field("system"),
        pl.col("pO2").is_between(20, 700),  # Physiologically valid range
        pl.col("sO2").is_between(10, 100),  # Percentage range
    )


# region P50
def P50(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_labs: Optional[pl.LazyFrame] = None,
    *,
    t_0: Optional[int] = 0,
    t_0_per_stay: Optional[pl.LazyFrame] = None,
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
        t_0 : int, optional
            Scalar reference time (seconds from admission). Defaults to 0 (admission).
            Ignored when t_0_per_stay is provided.
        t_0_per_stay : pl.LazyFrame, optional
            Per-stay T_0 overrides with columns [Global ICU Stay ID, T_0].

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

    missing = [name for name, data in required.items() if data is None]
    if missing:
        raise ValueError(
            f"Cannot compute P50: Missing required datasets: {', '.join(missing)}. "
            f"Ensure they are configured in ~/.reprodICU/PATHS.yaml or provide them explicitly."
        )

    patient_information = _to_lazy(patient_information)
    timeseries_labs = _to_lazy(timeseries_labs)
    t_0_per_stay = _to_lazy(t_0_per_stay) if t_0_per_stay is not None else None

    all_stays = patient_information.select(STAY_KEY)
    all_stays_t0 = _build_t0(all_stays, t_0_per_stay, t_0)

    timeseries_labs = _improve_labs(timeseries_labs)

    p50 = (
        timeseries_labs.with_columns(
            pl.col("pO2")
            .mul(
                (100 - pl.col("sO2"))
                .truediv(pl.col("sO2"))
                .pow(1 / HILL_COEFFICIENT)
            )
            .alias("P50 (mmHg)")
        )
        .with_columns(
            pl.when(pl.col("P50 (mmHg)").is_finite())
            .then(pl.col("P50 (mmHg)"))
            .otherwise(None)
            .alias("P50 (mmHg)")
        )
        .select(STAY_KEY, TIME_KEY, "P50 (mmHg)")
    )

    if (t_0 != 0) or (t_0_per_stay is not None):
        p50 = (
            p50.join(all_stays_t0, on=STAY_KEY, how="inner")
            .with_columns(
                pl.col(TIME_KEY)
                .sub(pl.col("T_0"))
                .alias("Time Relative to T_0 (seconds)")
            )
            .drop("T_0")
        )

    return p50


__all__ = ["P50"]
