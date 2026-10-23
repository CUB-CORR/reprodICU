"""
Mortality Measures: compute common mortality endpoints from admission and outcome data.

Output columns per row:
- Global ICU Stay ID
- Mortality in ICU: Binary flag for in-hospital mortality during ICU stay
- Mortality in Hospital: Binary flag for in-hospital mortality
- Mortality After ICU Admission (days): Days from ICU admission to death (or censored)
- Mortality [N] Days After ICU Admission: Binary flags for time-point mortality (7d, 28d, 30d, 90d, 180d, 360d, 1y)

Mortality is determined from a combination of:
- ICU length of stay with in-ICU death flag
- Hospital length of stay with in-hospital death flag
- Post-discharge mortality data

SOURCES
-------
- Standard clinical endpoints for mortality assessment in critical care trials
"""

from typing import Optional

import polars as pl

from .base import _to_lazy
from .common import get_patient_information

__all__ = ["COMMON_MORTALITY_MEASURES", "MORTALITY_WITHIN_DAYS"]


def MORTALITY_WITHIN_DAYS(days: int) -> pl.Expr:
    """Flag for death within `days` days after ICU admission (needs "Mortality After ICU Admission (days)")."""
    return (pl.col("Mortality After ICU Admission (days)") <= days).alias(f"Mortality {days} Days After ICU Admission")


def COMMON_MORTALITY_MEASURES(
    patient_information: Optional[pl.LazyFrame] = None,
) -> pl.LazyFrame:
    """
    Calculate common mortality measures based on admission and outcome data.

    Computes mortality flags at standard timepoints:
    - 7-day mortality
    - 28-day mortality
    - 30-day mortality
    - 90-day mortality
    - 180-day (6-month) mortality
    - 360-day mortality
    - 365-day (1-year) mortality

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information with mortality and length-of-stay data.
            Loaded automatically if None. Must contain:
            - Global ICU Stay ID
            - ICU Length of Stay (days)
            - Mortality in ICU
            - Mortality After ICU Discharge (days) [optional]
            - Hospital Length of Stay (days) [optional]
            - Pre-ICU Length of Stay (days) [optional]
            - Mortality in Hospital [optional]

    Returns
    -------
        pl.LazyFrame
            One row per stay with:
            - Global ICU Stay ID
            - Mortality After ICU Admission (days): Time to death from admission
            - Mortality [N] Days After ICU Admission: Boolean for each timepoint (7d, 28d, 30d, 90d, 180d, 360d, 365d)
    """
    # Load defaults if not provided
    if patient_information is None:
        patient_information = get_patient_information()

    # Validate data is available
    if patient_information is None:
        raise ValueError(
            "Cannot compute COMMON_MORTALITY_MEASURES: Missing patient_information dataset. "
            "Ensure it is configured in ~/.reprodICU/PATHS.yaml or provide it explicitly."
        )

    # Ensure lazy
    patient_information = _to_lazy(patient_information)

    return (
        patient_information.with_columns(
            pl.coalesce(
                pl.col("ICU Length of Stay (days)")
                + pl.coalesce(
                    pl.col("Mortality After ICU Discharge (days)"),
                    pl.when(pl.col("Mortality in ICU"))
                    .then(pl.lit(0))
                    .otherwise(None),
                ),
                pl.when(pl.col("Mortality in Hospital"))
                .then(
                    pl.col("Hospital Length of Stay (days)")
                    - pl.col("Pre-ICU Length of Stay (days)")
                )
                .otherwise(None),
            ).alias("Mortality After ICU Admission (days)"),
            pl.coalesce(
                pl.col("ICU Length of Stay (days)")
                + pl.col("Mortality After ICU Discharge Censor Cutoff (days)"),
                pl.col("ICU Length of Stay (days)")
                + pl.coalesce(
                    pl.col("Mortality After ICU Discharge (days)"),
                    pl.when(pl.col("Mortality in ICU"))
                    .then(pl.lit(0))
                    .otherwise(None),
                ),
                pl.when(pl.col("Mortality in Hospital"))
                .then(
                    pl.col("Hospital Length of Stay (days)")
                    - pl.col("Pre-ICU Length of Stay (days)")
                )
                .otherwise(None),
            ).alias("Mortality After ICU Admission Censor Cutoff (days)"),
        )
        .with_columns(
            pl.when(pl.col("Mortality After ICU Admission (days)") <= 0)
            .then(None)
            .otherwise(pl.col("Mortality After ICU Admission (days)"))
            .alias("Mortality After ICU Admission (days)")
        )
        .with_columns(
            *[MORTALITY_WITHIN_DAYS(days) for days in (7, 28, 30, 90, 180, 360)],
            MORTALITY_WITHIN_DAYS(365).alias("Mortality 1 Year After ICU Admission"),
        )
    )
