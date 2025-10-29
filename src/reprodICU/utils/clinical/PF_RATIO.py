from typing import Optional

import polars as pl

from ..common import (
    _build_t0,
    _to_lazy,
    get_patient_information,
    get_timeseries_labs,
    get_timeseries_respiratory,
)

SECONDS_IN_4H = 4 * 60 * 60
pf_ratio_col = "PaO2/FiO2 Ratio"


def _improve_resp(resp: pl.LazyFrame) -> pl.LazyFrame:
    return (
        resp.with_columns(
            pl.max_horizontal(
                "Oxygen/Total gas setting [Volume Fraction] Ventilator",
                "Oxygen/Gas total [Pure volume fraction] Inhaled gas",
            ).alias("FiO2")
        )
        .select(
            "Global ICU Stay ID",
            "Time Relative to Admission (seconds)",
            pl.when(pl.col("FiO2").is_between(0, 1))
            .then(pl.col("FiO2") * 100)
            .when(pl.col("FiO2").is_between(1, 100))
            .then(pl.col("FiO2"))
            .otherwise(None)
            .alias("FiO2"),
        )
        .drop_nulls("FiO2")
    )


def _improve_labs(labs: pl.LazyFrame) -> pl.LazyFrame:
    return labs.with_columns(
        pl.when(
            pl.col("Oxygen")
            .struct.field("system")
            .is_in(["Blood arterial", "Blood"])
            | pl.col("Oxygen").struct.field("system").is_null()
        )
        .then(pl.col("Oxygen").struct.field("value"))
        .otherwise(None)
        .alias("Oxygen in Arterial blood"),
    ).drop_nulls("Oxygen in Arterial blood")


def PAO2_FIO2_RATIO(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_resp: Optional[pl.LazyFrame] = None,
    timeseries_labs: Optional[pl.LazyFrame] = None,
    *,
    t_0: Optional[int] = 0,
    t_0_per_stay: Optional[pl.LazyFrame] = None,
    timeframe_name: Optional[str] = None,
) -> pl.LazyFrame:
    """_summary_

    Calculate urine output per time window from intake/output timeseries.

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information; must contain Global ICU Stay ID and
            Admission Weight (kg). Loaded automatically if None.
        timeseries_resp : pl.LazyFrame, optional
            Respiratory timeseries data. Loaded automatically if None.
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
            PaO2/FiO2 Ratio timeseries with columns:
            - Global ICU Stay ID
            - Time Relative to Admission (seconds)
            - PaO2/FiO2 Ratio
    """
    # Load defaults if not provided
    if patient_information is None:
        patient_information = get_patient_information()
    if timeseries_resp is None:
        timeseries_resp = get_timeseries_respiratory()
    if timeseries_labs is None:
        timeseries_labs = get_timeseries_labs()

    # Validate all required data is available
    required = {
        "patient_information": patient_information,
        "timeseries_resp": timeseries_resp,
        "timeseries_labs": timeseries_labs,
    }

    missing = [name for name, data in required.items() if data is None]
    if missing:
        raise ValueError(
            f"Cannot compute URINE_OUTPUT: Missing required datasets: {', '.join(missing)}. "
            f"Ensure they are configured in ~/.reprodICU/PATHS.yaml or provide them explicitly."
        )

    STAY_KEY = "Global ICU Stay ID"
    TIME_KEY = "Time Relative to Admission (seconds)"

    patient_information = _to_lazy(patient_information)
    timeseries_resp = _to_lazy(timeseries_resp)
    timeseries_labs = _to_lazy(timeseries_labs)
    t_0_per_stay = _to_lazy(t_0_per_stay) if t_0_per_stay is not None else None

    all_stays = patient_information.select(STAY_KEY)
    all_stays_t0 = _build_t0(all_stays, t_0_per_stay, t_0)

    timeseries_resp = _improve_resp(timeseries_resp)
    timeseries_labs = _improve_labs(timeseries_labs)

    pf_ratio = (
        timeseries_labs.select(STAY_KEY, TIME_KEY, "Oxygen in Arterial blood")
        .drop_nulls("Oxygen in Arterial blood")
        # FiO2 occurred at most 4 hours before this blood gas
        .join_asof(
            timeseries_resp,
            on=TIME_KEY,
            by=STAY_KEY,
            strategy="backward",
            tolerance=SECONDS_IN_4H,
            coalesce=True,
        )
        .with_columns(
            pl.col("Oxygen in Arterial blood")
            .truediv(
                pl.col("FiO2").fill_null(21).truediv(100)
            )  # 21% if FiO2 missing
            .alias(pf_ratio_col)
        )
        .with_columns(
            pl.when(pl.col(pf_ratio_col).is_finite())
            .then(pl.col(pf_ratio_col))
            .otherwise(None)
            .alias(pf_ratio_col)
        )
        .select(STAY_KEY, TIME_KEY, pf_ratio_col)
    )

    if t_0 or t_0_per_stay:
        pf_ratio = (
            pf_ratio.join(all_stays_t0, on=STAY_KEY, how="inner")
            .with_columns(
                pl.col(TIME_KEY)
                .sub(pl.col("T_0"))
                .alias("Time Relative to T_0 (seconds)")
            )
            .drop("T_0")
        )
        
    return pf_ratio


__all__ = ["PAO2_FIO2_RATIO"]
