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

from ...common import (
    _build_t0,
    _to_lazy,
    get_patient_information,
    get_timeseries_respiratory,
)

SECONDS_IN_6H = 60 * 60
STAY_KEY = "Global ICU Stay ID"
TIME_KEY = "Time Relative to Admission (seconds)"


# region helpers
def _improve_resp(resp: pl.LazyFrame) -> pl.LazyFrame:
    columns = ["VT", "Pplat", "PIP", "PEEP"]
    return (
        resp.select(
            STAY_KEY,
            TIME_KEY,
            pl.max_horizontal(
                "Tidal volume setting Ventilator",
                "Tidal volume.spontaneous --on ventilator",
                "Tidal volume.spontaneous+mechanical --on ventilator",
            ).alias("VT"),
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
        )
        .filter(pl.any_horizontal(pl.col(col).is_not_null() for col in columns))
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
def STATIC_COMPLIANCE(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_resp: Optional[pl.LazyFrame] = None,
    *,
    t_0: Optional[int] = 0,
    t_0_per_stay: Optional[pl.LazyFrame] = None,
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
        t_0 : int, optional
            Scalar reference time (seconds from admission). Defaults to 0 (admission).
            Ignored when t_0_per_stay is provided.
        t_0_per_stay : pl.LazyFrame, optional
            Per-stay T_0 overrides with columns [Global ICU Stay ID, T_0].

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

    missing = [name for name, data in required.items() if data is None]
    if missing:
        raise ValueError(
            f"Cannot compute STATIC_COMPLIANCE: Missing required datasets: {', '.join(missing)}. "
            f"Ensure they are configured in ~/.reprodICU/PATHS.yaml or provide them explicitly."
        )

    patient_information = _to_lazy(patient_information)
    timeseries_resp = _to_lazy(timeseries_resp)
    t_0_per_stay = _to_lazy(t_0_per_stay) if t_0_per_stay is not None else None

    all_stays = patient_information.select(STAY_KEY)
    all_stays_t0 = _build_t0(all_stays, t_0_per_stay, t_0)

    timeseries_resp = _improve_resp(timeseries_resp)

    static_compliance = timeseries_resp.with_columns(
        (pl.col("VT") / (pl.col("Pplat") - pl.col("PEEP"))).alias(
            "Static Compliance (mL/cm H2O)"
        )
    ).select(STAY_KEY, TIME_KEY, "Static Compliance (mL/cm H2O)")

    if (t_0 != 0) or (t_0_per_stay is not None):
        static_compliance = (
            static_compliance.join(all_stays_t0, on=STAY_KEY, how="inner")
            .with_columns(
                pl.col(TIME_KEY)
                .sub(pl.col("T_0"))
                .alias("Time Relative to T_0 (seconds)")
            )
            .drop("T_0")
        )

    return static_compliance


def DYNAMIC_COMPLIANCE(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_resp: Optional[pl.LazyFrame] = None,
    *,
    t_0: Optional[int] = 0,
    t_0_per_stay: Optional[pl.LazyFrame] = None,
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
        t_0 : int, optional
            Scalar reference time (seconds from admission). Defaults to 0 (admission).
            Ignored when t_0_per_stay is provided.
        t_0_per_stay : pl.LazyFrame, optional
            Per-stay T_0 overrides with columns [Global ICU Stay ID, T_0].

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

    missing = [name for name, data in required.items() if data is None]
    if missing:
        raise ValueError(
            f"Cannot compute DYNAMIC_COMPLIANCE: Missing required datasets: {', '.join(missing)}. "
            f"Ensure they are configured in ~/.reprodICU/PATHS.yaml or provide them explicitly."
        )

    patient_information = _to_lazy(patient_information)
    timeseries_resp = _to_lazy(timeseries_resp)
    t_0_per_stay = _to_lazy(t_0_per_stay) if t_0_per_stay is not None else None

    all_stays = patient_information.select(STAY_KEY)
    all_stays_t0 = _build_t0(all_stays, t_0_per_stay, t_0)

    timeseries_resp = _improve_resp(timeseries_resp)

    dynamic_compliance = timeseries_resp.with_columns(
        (pl.col("VT") / (pl.col("PIP") - pl.col("PEEP"))).alias(
            "Dynamic Compliance (mL/cm H2O)"
        )
    ).select(STAY_KEY, TIME_KEY, "Dynamic Compliance (mL/cm H2O)")

    if (t_0 != 0) or (t_0_per_stay is not None):
        dynamic_compliance = (
            dynamic_compliance.join(all_stays_t0, on=STAY_KEY, how="inner")
            .with_columns(
                pl.col(TIME_KEY)
                .sub(pl.col("T_0"))
                .alias("Time Relative to T_0 (seconds)")
            )
            .drop("T_0")
        )

    return dynamic_compliance


__all__ = ["STATIC_COMPLIANCE", "DYNAMIC_COMPLIANCE"]
