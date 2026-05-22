from typing import Optional

import polars as pl

from ..common import _to_lazy, get_timeseries_respiratory


def RESPIRATORY_PARAMS(
    timeseries_respiratory: Optional[pl.LazyFrame] = None,
) -> pl.LazyFrame:
    """
    Coalesce invasive and non-invasive blood pressure measurements into a unified timeseries with columns:
    - Tidal volume (-> `TV`)
    - Minute ventilation (-> `MV`)
    - Respiratory rate (-> `RR`)
    - Plateau pressure (-> `Pplat`)
    - Peak inspiratory pressure (-> `PIP`)
    - Positive end expiratory pressure (-> `PEEP`)
    - Fraction of inspired oxygen (-> `FiO2`)
    - Oxygen flow in liters per minute (-> `LPM`)

    MAP is calculated using the formula:
    MAP = (SBP + 2 * DBP) / 3

    Parameters:
        timeseries_respiratory : pl.LazyFrame, optional
            Respiratory parameters timeseries data. Loaded automatically if None.

    Returns
    -------
        pl.LazyFrame
            respiratory timeseries with additional columns:
            - `TV`: Tidal volume (mL)
            - `MV`: Minute ventilation (L/min)
            - `RR`: Respiratory rate (breaths/min)
            - `Pplat`: Plateau pressure (cmH2O)
            - `PIP`: Peak inspiratory pressure (cmH2O)
            - `PEEP`: Positive end expiratory pressure (cmH2O)
            - `FiO2`: Fraction of inspired oxygen (%)
            - `LPM`: Oxygen flow in liters per minute (LPM)
    """
    # Load defaults if not provided
    if timeseries_respiratory is None:
        timeseries_respiratory = get_timeseries_respiratory()

    timeseries_respiratory = _to_lazy(timeseries_respiratory)

    return (
        timeseries_respiratory.with_columns(
            pl.max_horizontal(
                "Tidal volume setting Ventilator",
                "Tidal volume.spontaneous --on ventilator",
                "Tidal volume.spontaneous+mechanical --on ventilator",
            ).alias("TV"),
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
            pl.col("Oxygen gas flow Oxygen delivery system").alias("LPM")
        )
        .with_columns(
            pl.max_horizontal(
                "Minute volume setting Ventilator",
                pl.col("TV") * pl.col("RR"),
            ).alias("MV"),
        )
        .with_columns(
            pl.when(pl.col("FiO2").is_between(0.21, 1))
            .then(pl.col("FiO2") * 100)
            .when(pl.col("FiO2").is_between(21, 100))
            .then(pl.col("FiO2"))
            .otherwise(None)
            .alias("FiO2"),
        )
    )


__all__ = ["RESPIRATORY_PARAMS"]
