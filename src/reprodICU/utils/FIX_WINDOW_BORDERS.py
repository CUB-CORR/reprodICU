from typing import Optional

import polars as pl

from .common import _to_lazy

SECONDS_IN_1H = 60 * 60


def FIX_WINDOW_BORDERS(
    DATA: pl.LazyFrame,
    TIMEWINDOW_IN_SECONDS: int = SECONDS_IN_1H,
    prefix: str = "Drug",
    reference: str = "T_0",
    unit: str = "seconds",
) -> pl.LazyFrame:
    DATA = _to_lazy(DATA)
    suffix = f"{reference} ({unit})"
    start_col = prefix + " Start Relative to " + suffix
    end_col = prefix + " End Relative to " + suffix
    window_col = "Window Relative to " + reference

    return (
        DATA.with_columns(
            pl.int_ranges(
                pl.col(start_col) // TIMEWINDOW_IN_SECONDS,
                (pl.col(end_col) - 1) // TIMEWINDOW_IN_SECONDS + 1,
            ).alias(window_col)
        )
        .explode(window_col)
        .with_columns(
            (
                pl.min_horizontal(
                    pl.col(end_col), (pl.col(window_col) + 1) * TIMEWINDOW_IN_SECONDS
                )
                - pl.max_horizontal(
                    pl.col(start_col), pl.col(window_col) * TIMEWINDOW_IN_SECONDS
                )
            ).alias(prefix + " Duration (seconds)")
        )
        .with_columns(
            (pl.col(prefix + " Duration (seconds)") / TIMEWINDOW_IN_SECONDS)
            .alias(prefix + " Duration (windows)"),
            pl.col(window_col).cast(int),
        )
    ) # fmt: skip


def PROJECT_TO_WINDOWS(
    DATA: pl.LazyFrame,
    group_cols: list[str],
    value_col: str,
    TIMEWINDOW_IN_SECONDS: int = SECONDS_IN_1H,
    prefix: str = "Drug",
    reference: str = "T_0",
    unit: str = "seconds",
) -> pl.LazyFrame:
    """Project interval values onto a regular window grid.

    Redistributes each interval's value to windows proportional to overlap duration.
    Returns columns: group_cols..., "Window Relative to {reference}",
    "{prefix} Projection Sum", "{prefix} Duration (windows)".
    """
    window_col = "Window Relative to " + reference
    exploded = FIX_WINDOW_BORDERS(DATA, TIMEWINDOW_IN_SECONDS, prefix, reference, unit)

    return (
        exploded.group_by(*group_cols, window_col)
        .agg(
            (pl.col(value_col) * pl.col(prefix + " Duration (seconds)"))
            .sum()
            .alias(prefix + " Projection Sum"),
            pl.col(prefix + " Duration (windows)")
            .sum()
            .alias(prefix + " Duration (windows)"),
        )
        .sort(*group_cols, window_col)
    )


def TIME_WEIGHTED_MEAN_RATE(
    window_size: int,
    prefix: str = "Drug",
    where: Optional[pl.Expr] = None,
) -> pl.Expr:
    """Mean rate while infused in a window, from the output of PROJECT_TO_WINDOWS.

    sum(rate x seconds) / seconds infused, over the rows (optionally only those
    where `where` holds); 0 if there are none.
    """
    projection = pl.col(prefix + " Projection Sum")
    duration = pl.col(prefix + " Duration (windows)")
    if where is not None:
        projection = projection.filter(where)
        duration = duration.filter(where)
    return (projection.sum() / (duration.sum() * window_size)).fill_nan(0)
