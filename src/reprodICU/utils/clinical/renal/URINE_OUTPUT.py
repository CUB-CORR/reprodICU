from typing import Optional

import polars as pl

from ...base import (
    SECONDS_IN_12H,
    STAY_KEY,
    TIME_KEY,
    Window,
    _build_t0,
    _get_timeframe_name,
    _optional_timeframe_bounds_filter,
    _to_lazy,
    with_windows,
)
from ...common import (
    FILTER_NOT_NULL,
    _validate_required_data,
    get_patient_information,
    get_timeseries_intakeoutput,
)
from ...FIX_WINDOW_BORDERS import PROJECT_TO_WINDOWS



def _improve_inout(inout: pl.LazyFrame) -> pl.LazyFrame:
    return (
        _to_lazy(inout)
        .pipe(
            FILTER_NOT_NULL,
            [
                "Fluid output urine in and out urethral catheter",
                "Fluid output urine nephrostomy",
                "Urine output",
            ],
        )
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


@with_windows
def URINE_OUTPUT(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_inout: Optional[pl.LazyFrame] = None,
    *,
    weight_col: str = "Admission Weight (kg)",
    window: Window,
) -> pl.LazyFrame:
    """
    Calculate urine output per time window from intake/output timeseries.

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information; must contain Global ICU Stay ID and
            weight column specified by weight_col. Loaded automatically if None.
        timeseries_inout : pl.LazyFrame, optional
            Intake/output timeseries data. Loaded automatically if None.
        weight_col : str, optional
            Name of weight column in patient_information to use for per-kg calculations.
            Defaults to "Admission Weight (kg)".

    Returns
    -------
        pl.LazyFrame
            Urine output aggregated per stay and time window, with columns:
            - Global ICU Stay ID
            - timeframe (or custom name)
            - uo_interval_ml: Total urine output in ml for the window
            - uo_interval_ml_per_kg: Urine output per kg body weight
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
    _validate_required_data("URINE_OUTPUT", required)

    patient_information = _to_lazy(patient_information)
    timeseries_inout = _to_lazy(timeseries_inout)

    all_stays_t0 = _build_t0(patient_information.select(STAY_KEY), window.t_0_per_stay, window.t_0)

    inout = _improve_inout(timeseries_inout)

    urine = (
        inout.join(all_stays_t0, on=STAY_KEY, how="inner")
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

    urine_with_rate = urine.with_columns(
        pl.when(
            pl.col("Urine End Relative to T_0 (seconds)")
            <= pl.col("Urine Start Relative to T_0 (seconds)")
        )
        .then(pl.col("Urine output"))
        .otherwise(
            pl.col("Urine output")
            / (
                pl.col("Urine End Relative to T_0 (seconds)")
                - pl.col("Urine Start Relative to T_0 (seconds)")
            )
        )
        .alias("urine_rate_ml_per_sec"),
    )

    windowed = PROJECT_TO_WINDOWS(
        urine_with_rate,
        group_cols=[STAY_KEY],
        value_col="urine_rate_ml_per_sec",
        TIMEWINDOW_IN_SECONDS=window.window_size,
        prefix="Urine",
        reference="T_0",
        unit="seconds",
    ).select(
        STAY_KEY,
        pl.col("Window Relative to T_0").alias("timeframe"),
        pl.col("Urine Projection Sum").alias("uo_interval_ml"),
    )

    aggregated = (
        windowed
        # Join with weight column from patient_information
        .join(
            patient_information.select(STAY_KEY, weight_col),
            on=STAY_KEY,
            how="left",
        )
        .with_columns(
            pl.when(pl.col(weight_col) > 0)
            .then(pl.col("uo_interval_ml") / pl.col(weight_col))
            .otherwise(None)
            .alias("uo_interval_ml_per_kg")
        )
    )

    name = _get_timeframe_name(window.timeframe_name, window.window_size, window.t_0, window.t_0_per_stay)
    return (
        aggregated.join(all_stays_t0, on=STAY_KEY, how="left")
        .filter(
            *_optional_timeframe_bounds_filter(
                "timeframe", window.window_size, t_minus1=window.t_minus1, t_1=window.t_1
            )
        )
        .select(STAY_KEY, pl.col("timeframe").alias(name), "uo_interval_ml", "uo_interval_ml_per_kg")
        .sort(STAY_KEY, name)
    )


__all__ = ["URINE_OUTPUT"]
