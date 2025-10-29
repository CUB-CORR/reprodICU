from typing import Optional

import polars as pl

from ..common import (
    _build_t0,
    _to_lazy,
    get_timeseries_intakeoutput,
    get_patient_information,
)
from ..FIX_WINDOW_BORDERS import FIX_WINDOW_BORDERS

SECONDS_IN_1MIN = 60
SECONDS_IN_1H = 60 * SECONDS_IN_1MIN
SECONDS_IN_12H = 12 * SECONDS_IN_1H
SECONDS_IN_1D = 24 * SECONDS_IN_1H
SECONDS_IN_1W = 7 * SECONDS_IN_1D


def _improve_inout(inout: pl.LazyFrame) -> pl.LazyFrame:
    return (
        _to_lazy(inout)
        .select(
            "Global ICU Stay ID",
            "Time Relative to Admission (seconds)",
            pl.sum_horizontal(
                "Fluid output urine in and out urethral catheter",
                "Fluid output urine nephrostomy",
                "Urine output",
            ).alias("Urine output"),
        )
        .drop_nulls("Urine output")
    )


def URINE_OUTPUT(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_inout: Optional[pl.LazyFrame] = None,
    *,
    t_0: Optional[int] = 0,
    t_0_per_stay: Optional[pl.LazyFrame] = None,
    t_1: Optional[int] = None,
    window_size: int = SECONDS_IN_1D,
    timeframe_unit: str = "Days",  # semantics only; output timeframe is numeric
    timeframe_name: Optional[str] = None,
) -> pl.LazyFrame:
    """
    Calculate urine output per time window from intake/output timeseries.

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information; must contain Global ICU Stay ID and
            Admission Weight (kg). Loaded automatically if None.
        timeseries_inout : pl.LazyFrame, optional
            Intake/output timeseries data. Loaded automatically if None.
        t_0 : int, optional
            Scalar reference time (seconds from admission). Defaults to 0 (admission).
            Ignored when t_0_per_stay is provided.
        t_0_per_stay : pl.LazyFrame, optional
            Per-stay T_0 overrides with columns [Global ICU Stay ID, T_0].
        t_1 : int, optional
            Optional upper time bound (seconds from admission) for filtering inputs.
        window_size : int, optional
            Timeframe width in seconds (default: 86400 = 1 day). Window index is
            floor((time - T_0)/window_size).
        timeframe_name : str, optional
            Name for output timeframe column. Auto-generated if None.

    Returns
    -------
        pl.LazyFrame
            Urine output aggregated per stay and time window, with columns:
            - Global ICU Stay ID
            - timeframe (or custom name)
            - uo_interval_ml: Total urine output in ml for the window
            - uo_interval_ml_per_kg: (optional) Urine output per kg body weight
    """
    # Load defaults if not provided
    if patient_information is None:
        patient_information = get_patient_information()
    if timeseries_inout is None:
        timeseries_inout = get_timeseries_intakeoutput()

    # Validate all required data is available
    required = {
        "patient_information": patient_information,
        "timeseries_inout": timeseries_inout,
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
    timeseries_inout = _to_lazy(timeseries_inout)
    t_0_per_stay = _to_lazy(t_0_per_stay) if t_0_per_stay is not None else None

    all_stays = patient_information.select(STAY_KEY)
    all_stays_t0 = _build_t0(all_stays, t_0_per_stay, t_0)

    inout = _improve_inout(timeseries_inout)

    urine = (
        inout.join(all_stays_t0, on=STAY_KEY, how="inner")
        .filter(pl.col(TIME_KEY) >= pl.col("T_0").sub(SECONDS_IN_1W))
        .sort(STAY_KEY, TIME_KEY)
        .with_columns(
            pl.col(TIME_KEY)
            .shift(1)
            .over(partition_by=STAY_KEY, order_by=TIME_KEY)
            .alias("prev_time")
        )
        .with_columns(
            pl.max_horizontal(
                pl.col("prev_time"),
                pl.col(TIME_KEY).sub(SECONDS_IN_12H),
            ).alias("uo_start")
        )
        .with_columns(
            (pl.col("uo_start") - pl.col("T_0")).alias("Urine Start Relative to T_0 (seconds)"),
            (pl.col(TIME_KEY) - pl.col("T_0")).alias("Urine End Relative to T_0 (seconds)"),
        )
        .drop("prev_time", "uo_start")
    ) # fmt: skip

    windowed = FIX_WINDOW_BORDERS(
        urine,
        TIMEWINDOW_IN_SECONDS=window_size,
        prefix="Urine",
        reference="T_0",
        unit="seconds",
    ).with_columns(
        pl.col("Window Relative to T_0").alias("timeframe"),
        pl.when(
            pl.col("Urine End Relative to T_0 (seconds)")
            <= pl.col("Urine Start Relative to T_0 (seconds)")
        )
        .then(pl.col("Urine output"))
        .otherwise(
            pl.col("Urine output")
            * pl.col("Urine Duration (seconds)")
            / (
                pl.col("Urine End Relative to T_0 (seconds)")
                - pl.col("Urine Start Relative to T_0 (seconds)")
            )
        )
        .alias("uo_window_ml"),
    )

    if t_1 is not None:
        windowed = windowed.filter(
            pl.col("timeframe")
            < (pl.lit(int(t_1)).sub(pl.col("T_0")).floordiv(window_size).add(1))
        )

    aggregated = (
        windowed.group_by(STAY_KEY, "timeframe")
        .agg(pl.sum("uo_window_ml").alias("uo_interval_ml"))
        .join(
            patient_information.select(STAY_KEY, "Admission Weight (kg)"),
            on=STAY_KEY,
            how="left",
        )
        .with_columns(
            pl.when(pl.col("Admission Weight (kg)") > 0)
            .then(pl.col("uo_interval_ml") / pl.col("Admission Weight (kg)"))
            .otherwise(None)
            .alias("uo_interval_ml_per_kg")
        )
    )

    if timeframe_name is not None:
        aggregated = aggregated.with_columns(
            pl.col("timeframe").alias(timeframe_name)
        )

    select_cols = [
        STAY_KEY,
        "timeframe" if timeframe_name is None else timeframe_name,
        "uo_interval_ml",
        "uo_interval_ml_per_kg",
    ]

    return aggregated.select(select_cols)


__all__ = ["URINE_OUTPUT"]
