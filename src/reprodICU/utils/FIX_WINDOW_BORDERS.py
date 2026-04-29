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
    return (
        DATA.with_columns(
            pl.int_ranges(
                -(
                    pl.col(prefix + " Start Relative to " + suffix)
                    // -TIMEWINDOW_IN_SECONDS
                )
                * TIMEWINDOW_IN_SECONDS,
                pl.col(prefix + " End Relative to " + suffix),
                TIMEWINDOW_IN_SECONDS,
            ).alias(prefix + " Window Borders")
        )
        .with_columns(
            pl.concat_list(
                pl.col(prefix + " Start Relative to " + suffix),
                pl.col(prefix + " Window Borders"),
            ).alias(prefix + " Window Borders Start"),
            pl.concat_list(
                pl.col(prefix + " Window Borders"),
                pl.col(prefix + " End Relative to " + suffix),
            ).alias(prefix + " Window Borders End"),
        )
        .drop(prefix + " Window Borders")
        .explode(
            prefix + " Window Borders Start",
            prefix + " Window Borders End",
        )
        .with_columns(
            (
                pl.col(prefix + " Window Borders End")
                - pl.col(prefix + " Window Borders Start")
            ).alias(prefix + " Duration (seconds)"),
            (
                pl.col(prefix + " Window Borders End")
                - pl.col(prefix + " Window Borders Start")
            )
            .truediv(TIMEWINDOW_IN_SECONDS)
            .alias(prefix + " Duration (windows)"),
            pl.col(prefix + " Window Borders Start")
            .floordiv(TIMEWINDOW_IN_SECONDS)
            .cast(int)
            .alias("Window Relative to " + reference),
        )
        .drop(prefix + " Window Borders Start", prefix + " Window Borders End")
    )
