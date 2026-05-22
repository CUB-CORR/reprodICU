from typing import Optional

import polars as pl

from ..common import _to_lazy, get_timeseries_vitals


def BLOOD_PRESSURES(
    timeseries_vitals: Optional[pl.LazyFrame] = None,
) -> pl.LazyFrame:
    """
    Coalesce invasive and non-invasive blood pressure measurements into a unified timeseries with columns:
    - Systolic Blood Pressure (-> `SBP`)
    - Mean Arterial Pressure (-> `MAP`)
    - Diastolic Blood Pressure (-> `DBP`)

    MAP is calculated using the formula:
    MAP = (SBP + 2 * DBP) / 3

    Parameters:
        timeseries_vitals : pl.LazyFrame, optional
            Vital signs timeseries data. Loaded automatically if None.

    Returns
    -------
        pl.LazyFrame
            vitals timeseries with additional columns:
            - `SBP`: Systolic Blood Pressure (mmHg)
            - `MAP`: Mean Arterial Pressure (mmHg)
            - `DBP`: Diastolic Blood Pressure (mmHg)
    """
    # Load defaults if not provided
    if timeseries_vitals is None:
        timeseries_vitals = get_timeseries_vitals()

    timeseries_vitals = _to_lazy(timeseries_vitals)

    return timeseries_vitals.with_columns(
        pl.coalesce(
            pl.col("Invasive systolic arterial pressure"),
            pl.col("Non-invasive systolic arterial pressure"),
        ).alias("SBP"),
        pl.coalesce(
            pl.col("Invasive mean arterial pressure"),
            pl.col("Non-invasive mean arterial pressure"),
            (
                pl.col("Invasive systolic arterial pressure")
                + 2 * pl.col("Invasive diastolic arterial pressure")
            ).truediv(3),
            (
                pl.col("Non-invasive systolic arterial pressure")
                + 2 * pl.col("Non-invasive diastolic arterial pressure")
            ).truediv(3),
        ).alias("MAP"),
        pl.coalesce(
            pl.col("Invasive diastolic arterial pressure"),
            pl.col("Non-invasive diastolic arterial pressure"),
        ).alias("DBP"),
    )


__all__ = ["BLOOD_PRESSURES"]
