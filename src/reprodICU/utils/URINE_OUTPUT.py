from typing import Optional

import polars as pl

from .common import _build_t0, _to_lazy
from .FIX_WINDOW_BORDERS import FIX_WINDOW_BORDERS

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
    patient_information: pl.LazyFrame,
    timeseries_inout: pl.LazyFrame,
    *,
    t_0: Optional[int] = 0,
    t_0_per_stay: Optional[pl.LazyFrame] = None,
    t_1: Optional[int] = None,
    window_size: int = SECONDS_IN_1D,
    timeframe_unit: str = "Days",  # semantics only; output timeframe is numeric
    timeframe_name: Optional[str] = None,
    weight_per_stay: Optional[pl.LazyFrame] = None,
    weight_per_stay_col: Optional[str] = None,
) -> pl.LazyFrame:
    STAY_KEY = "Global ICU Stay ID"
    TIME_KEY = "Time Relative to Admission (seconds)"

    patient_information = _to_lazy(patient_information)
    timeseries_inout = _to_lazy(timeseries_inout)
    t_0_per_stay = _to_lazy(t_0_per_stay) if t_0_per_stay is not None else None
    weight_per_stay = _to_lazy(weight_per_stay) if weight_per_stay is not None else None # fmt: skip

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

    aggregated = windowed.group_by(STAY_KEY, "timeframe").agg(
        pl.sum("uo_window_ml").alias("uo_interval_ml")
    )

    has_weight_rate = False
    if weight_per_stay is not None:
        aggregated = aggregated.join(
            weight_per_stay, on=STAY_KEY, how="left"
        ).with_columns(
            pl.when(pl.col(weight_per_stay_col) > 0)
            .then(pl.col("uo_interval_ml") / pl.col(weight_per_stay_col))
            .otherwise(None)
            .alias("uo_interval_ml_per_kg")
        )
        has_weight_rate = True

    if timeframe_name is not None:
        aggregated = aggregated.with_columns(
            pl.col("timeframe").alias(timeframe_name)
        )

    select_cols = [
        STAY_KEY,
        "timeframe",
        "uo_interval_ml",
    ]
    if has_weight_rate:
        select_cols.append("uo_interval_ml_per_kg")
    if timeframe_name is not None:
        select_cols.append(timeframe_name)

    return aggregated.select(select_cols)


__all__ = ["URINE_OUTPUT"]
