"""Windowing basis of the concepts: reference time, timeframe index, dense and sparse frames, bounds."""
import functools
import inspect
import textwrap
from dataclasses import dataclass
from numbers import Integral
from typing import NamedTuple, Optional

import polars as pl

STAY_KEY = "Global ICU Stay ID"
TIME_KEY = "Time Relative to Admission (seconds)"
SECONDS_IN_1MIN = 60
SECONDS_IN_1H = 60 * SECONDS_IN_1MIN
SECONDS_IN_4H = 4 * SECONDS_IN_1H
SECONDS_IN_6H = 6 * SECONDS_IN_1H
SECONDS_IN_12H = 12 * SECONDS_IN_1H
SECONDS_IN_1D = 24 * SECONDS_IN_1H
SECONDS_IN_1W = 7 * SECONDS_IN_1D
# default lower time bound (seconds from admission) of the concepts; None = unbounded
T_MINUS1_DEFAULT = -30 * SECONDS_IN_1D


def _to_lazy(frame) -> pl.LazyFrame:
    return frame if isinstance(frame, pl.LazyFrame) else frame.lazy()


# region time helpers
def _build_t0(
    all_stays: pl.LazyFrame,
    t_0_per_stay: Optional[pl.LazyFrame],
    t_0: Optional[int],
) -> pl.LazyFrame:
    all_stays = _to_lazy(all_stays)

    if t_0_per_stay is not None:
        t_0_per_stay = _to_lazy(t_0_per_stay)
        return (
            all_stays.select(STAY_KEY)
            .join(t_0_per_stay.select(STAY_KEY, "T_0"), STAY_KEY, how="left")
            .with_columns(pl.col("T_0").fill_null(0).cast(pl.Int64))
        )

    t0_val = 0 if t_0 is None else int(t_0)
    return all_stays.select(
        STAY_KEY, pl.lit(t0_val).cast(pl.Int64).alias("T_0")
    )


def _build_base_timeframes(
    all_stays_t_0: pl.LazyFrame,
    patient_information: pl.LazyFrame,
    window_size: int,
    los_col: str = "ICU Length of Stay (days)",
) -> pl.LazyFrame:
    """
    One row per stay and window, dense from ICU admission to the end of the stay.

    Timeframes are floor((time - T_0)/window_size). T_0 is only the reference, so the
    first window is the one containing ICU admission (negative for T_0 > 0). Windows
    before it are added where data exists with `_add_sparse_timeframes`.
    """
    all_stays_t_0 = _to_lazy(all_stays_t_0)

    return (
        all_stays_t_0.join(patient_information, on=STAY_KEY, how="left")
        .select(STAY_KEY, "T_0", los_col)
        .with_columns(
            pl.int_ranges(
                # dense from the window containing ICU admission; T_0 is only the reference
                start=(-pl.col("T_0")).truediv(window_size).floor().cast(int),
                end=(pl.col(los_col) * SECONDS_IN_1D - pl.col("T_0"))
                .truediv(window_size)
                .clip(lower_bound=1)  # at least the first window
                .ceil()
                .cast(int),
                step=1,
            ).alias("timeframe")
        )
        .explode("timeframe")
        .select(STAY_KEY, "T_0", "timeframe")
    )


def _add_sparse_timeframes(
    base: pl.LazyFrame, *parts: pl.LazyFrame
) -> pl.LazyFrame:
    """
    Add the timeframes before the dense range of `base` that hold data.

    `_build_base_timeframes` is dense from the window containing ICU admission.
    Windows before it exist only where one of the `parts` (frames with
    Global ICU Stay ID and timeframe) has a row, which keeps the frame small for
    long pre-ICU histories.
    """
    start = base.group_by(STAY_KEY).agg(
        pl.col("timeframe").min().alias("_start"), pl.col("T_0").first()
    )
    extra = (
        pl.concat(
            [_to_lazy(p).select(STAY_KEY, "timeframe") for p in parts],
            how="vertical_relaxed",
        )
        .drop_nulls()
        .unique()
        .join(start, on=STAY_KEY, how="inner")
        .filter(pl.col("timeframe") < pl.col("_start"))
        .select(STAY_KEY, "T_0", "timeframe")
    )
    return pl.concat([base, extra], how="vertical_relaxed")


def _null_if_incomplete_pre_icu(
    score: pl.Expr, components: pl.Expr, window_size: int
) -> pl.Expr:
    """
    Null `score` before ICU admission unless every column of `components` has a value.

    Missing components count as normal inside the stay, but the windows before the
    dense range exist only where some input has data, so a missing component there
    means the score cannot be assessed. Needs `T_0` and `timeframe` columns.
    """
    pre_icu = pl.col("timeframe") < (-pl.col("T_0")).floordiv(window_size)
    return (
        pl.when(pre_icu & pl.any_horizontal(components.is_null()))
        .then(None)
        .otherwise(score)
    )


def _sum_points(window_size: int) -> pl.Expr:
    """Sum of all points columns, null for incomplete windows before ICU admission."""
    points = pl.exclude(STAY_KEY, "T_0", "timeframe")
    return _null_if_incomplete_pre_icu(pl.sum_horizontal(points), points, window_size)


def _assign_timeframe(time_col: str, window_size: int) -> pl.Expr:
    return pl.col(time_col).sub(pl.col("T_0")).floordiv(window_size).cast(int)


def _optional_timeframe_bounds_filter(
    timeframe_col: str,
    window_size: int,
    t_minus1: Optional[int] = None,
    t_1: Optional[int] = None,
    t0_col: str = "T_0",
) -> list[pl.Expr]:
    """
    Predicates restricting timeframes to the range of interest.

    The timeframe column counts from each stay's T_0, floor((time - T_0)/window_size).
    t_0 (scalar or per stay) therefore enters through the `t0_col` column and is only
    the reference of the index, never a cut. `t_minus1` and `t_1` are on the admission
    clock (seconds), so the frame needs the T_0 column.

    Arguments
    ---------
        timeframe_col : str
            Timeframe column (index from T_0).
        window_size : int
            Timeframe width in seconds.
        t_minus1 : int, optional
            Keep the windows ending after t_minus1 (seconds from admission, may be
            negative to reach back before ICU admission).
        t_1 : int, optional
            Keep the windows starting before t_1 (seconds from admission).
        t0_col : str
            Column with the T_0 of each stay (seconds from admission).

    Returns
    -------
        list of pl.Expr to pass to `.filter()`
    """
    # never empty: some callers pass the list itself to `.filter()`, where [] is a null mask
    conds: list[pl.Expr] = [pl.lit(True)]
    if t_minus1 is not None:
        conds.append(
            (pl.col(timeframe_col) + 1) * window_size + pl.col(t0_col) > int(t_minus1)
        )
    if t_1 is not None:
        conds.append(
            pl.col(timeframe_col) * window_size + pl.col(t0_col) < int(t_1)
        )
    return conds


def _get_timeframe_name(
    timeframe_name: Optional[str],
    window_size: int,
    t_0: int,
    t_0_per_stay: Optional[pl.LazyFrame],
) -> str:
    if timeframe_name is not None:
        return timeframe_name
    unit = (
        "Days"
        if window_size == SECONDS_IN_1D
        else "Hours" if window_size == SECONDS_IN_1H else "Windows"
    )
    reference = "T_0" if t_0 != 0 or t_0_per_stay is not None else "Admission"
    return f"{unit} Relative to {reference}"


# region timeframe helpers
def intervention_per_timeframe(
    data: pl.LazyFrame,
    patient_information: pl.LazyFrame,
    start_col: str,
    end_col: str,
    window: "Window",
) -> pl.LazyFrame:
    """Determine whether an intervention was active in each timeframe per stay."""
    all_stays_t0 = _reference_t0(patient_information, window)
    window_size = window.window_size
    return (
        data.select(STAY_KEY, start_col, end_col)
        .join(all_stays_t0, on=STAY_KEY, how="inner")
        .with_columns(
            (pl.col(start_col) - pl.col("T_0")).alias("start_rel"),
            (pl.col(end_col) - pl.col("T_0")).alias("end_rel"),
        )
        .with_columns(
            timeframe_start=pl.col("start_rel").floordiv(window_size),
            timeframe_end=pl.col("end_rel").floordiv(window_size),
        )
        .select(
            STAY_KEY,
            pl.int_ranges(
                pl.col("timeframe_start"),
                pl.col("timeframe_end").add(1),
                step=1,
            ).alias("timeframe"),
        )
        .explode("timeframe")
        .with_columns(pl.col("timeframe").is_not_null().alias("intervention"))
        .group_by(STAY_KEY, "timeframe")
        .agg(pl.max("intervention"))
    )


# region decorators for the time arguments
@dataclass(frozen=True)
class Reference:
    """The reference time of a concept: a scalar, or one per stay."""

    t_0: Optional[int] = 0
    t_0_per_stay: Optional[pl.LazyFrame] = None

    def __post_init__(self):
        _check_int("t_0", self.t_0)


@dataclass(frozen=True)
class Window(Reference):
    """The reference time and the windows of a concept, see `with_windows`."""

    t_minus1: Optional[int] = T_MINUS1_DEFAULT
    t_1: Optional[int] = None
    window_size: int = SECONDS_IN_1D
    timeframe_name: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        _check_int("t_minus1", self.t_minus1)
        _check_int("t_1", self.t_1)
        _check_int("window_size", self.window_size, optional=False)
        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {self.window_size}")
        if self.t_minus1 is not None and self.t_1 is not None and self.t_minus1 >= self.t_1:
            raise ValueError(f"t_minus1 ({self.t_minus1}) must be smaller than t_1 ({self.t_1})")


def _check_int(name, value, optional=True):
    if value is None and optional:
        return
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer number of seconds, got {value!r}")


# The arguments the decorators add: name, annotation, default, the type and the
# description for the docstring
_T_0 = (
    "t_0",
    Optional[int],
    0,
    "int",
    "Scalar reference time (seconds from admission). Timeframes count from it, "
    "it never cuts the range. Ignored when t_0_per_stay is provided.",
)
_T_0_PER_STAY = (
    "t_0_per_stay",
    Optional[pl.LazyFrame],
    None,
    "pl.LazyFrame",
    "Per-stay T_0 overrides with columns [Global ICU Stay ID, T_0].",
)
_T_MINUS1 = (
    "t_minus1",
    Optional[int],
    T_MINUS1_DEFAULT,
    "int",
    "Lower time bound (seconds from admission, may be negative). Timeframes "
    "ending at or before it are dropped. Defaults to 30 days before "
    "admission, None keeps everything.",
)
_T_1 = (
    "t_1",
    Optional[int],
    None,
    "int",
    "Upper time bound (seconds from admission). Timeframes starting at or "
    "after it are dropped.",
)
_TIMEFRAME_NAME = (
    "timeframe_name",
    Optional[str],
    None,
    "str",
    "Name of the timeframe column. Generated from the window size if None.",
)


def _document(doc, params):
    """Add an entry for each of `params` to the Arguments of `doc`."""
    lines = (doc or "").split("\n")
    returns = [i for i, line in enumerate(lines) if line.strip() == "Returns"]
    end = returns[0] if returns else len(lines)
    indent = len(lines[end]) - len(lines[end].lstrip()) if returns else 4
    while end > 0 and not lines[end - 1].strip():
        end -= 1

    entries = []
    if "Arguments" not in [line.strip() for line in lines]:
        entries = ["", " " * indent + "Arguments", " " * indent + "-" * 9]
    for name, _, _, kind, text in params:
        entries.append(" " * (indent + 4) + f"{name} : {kind}, optional")
        entries += textwrap.wrap(
            text,
            width=80,
            initial_indent=" " * (indent + 8),
            subsequent_indent=" " * (indent + 8),
        )
    return "\n".join(lines[:end] + entries + lines[end:])


def _inject(func, kind, params, make):
    """
    Give `func` the keyword-only `params` and pass their values to it as one
    object, built by `make`, in its keyword-only argument `kind`.
    """
    signature = inspect.signature(func)
    own = [p for p in signature.parameters.values() if p.name != kind]
    added = [
        inspect.Parameter(
            name,
            inspect.Parameter.KEYWORD_ONLY,
            default=default,
            annotation=annotation,
        )
        for name, annotation, default, _, _ in params
    ]

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if kind in kwargs:
            raise TypeError(f"{func.__name__}() got an unexpected keyword argument '{kind}'")
        values = {p[0]: kwargs.pop(p[0], p[2]) for p in params}
        return func(*args, **kwargs, **{kind: make(**values)})

    wrapper.__signature__ = signature.replace(parameters=own + added)
    wrapper.__doc__ = _document(func.__doc__, params)
    return wrapper


def with_reference_time(func):
    """
    Give an event-level concept the keyword arguments `t_0` and `t_0_per_stay`.

    The decorated function takes a keyword-only `ref` (a `Reference`) and reads
    the two values from it. Callers see, and the docstring documents, the two
    arguments.
    """
    return _inject(func, "ref", [_T_0, _T_0_PER_STAY], Reference)


def with_windows(func=None, *, window_size=SECONDS_IN_1D):
    """
    Give a windowed concept the keyword arguments `t_0`, `t_0_per_stay`,
    `t_minus1`, `t_1`, `window_size` and `timeframe_name`.

    The decorated function takes a keyword-only `window` (a `Window`) and reads
    the values from it. Callers see, and the docstring documents, the six
    arguments. `window_size` is their default width.
    """
    width = {SECONDS_IN_1D: " = 1 day", SECONDS_IN_1H: " = 1 hour"}
    window_size_param = (
        "window_size",
        int,
        window_size,
        "int",
        f"Timeframe width in seconds (default: {window_size}"
        f"{width.get(window_size, '')}). The window index is "
        "floor((time - T_0) / window_size).",
    )
    params = [
        _T_0,
        _T_0_PER_STAY,
        _T_MINUS1,
        _T_1,
        window_size_param,
        _TIMEFRAME_NAME,
    ]

    def decorate(f):
        return _inject(f, "window", params, Window)

    return decorate(func) if func is not None else decorate


# endregion


# region the reference time of the event-level concepts
def _reference_t0(patient_information, ref: Reference) -> pl.LazyFrame:
    """T_0 per stay: the per-stay table of the reference if given, else its scalar."""
    return _build_t0(_to_lazy(patient_information).select(STAY_KEY), ref.t_0_per_stay, ref.t_0)


def _relative_to_t0(
    df: pl.LazyFrame, all_stays_t0: pl.LazyFrame, ref: Reference, time_col: str = "Time Relative to Admission (seconds)"
) -> pl.LazyFrame:
    """
    Add "Time Relative to T_0 (seconds)" (and drop T_0) if the reference is not the admission, else return `df` as it is.

    The stays without a T_0 are dropped, as for the windowed concepts.
    """
    if ref.t_0 == 0 and ref.t_0_per_stay is None:
        return df
    return (
        df.join(all_stays_t0, on=STAY_KEY, how="inner")
        .with_columns(pl.col(time_col).sub(pl.col("T_0")).alias("Time Relative to T_0 (seconds)"))
        .drop("T_0")
    )


# endregion


# region the shared skeleton of the windowed concepts
class Context(NamedTuple):
    """The reference time, the timeframes and the bounds of a windowed concept."""

    all_stays_t0: pl.LazyFrame
    base: pl.LazyFrame
    name: str
    bounds: list
    window_size: int


def _window_context(patient_information, window: Window) -> Context:
    """
    T_0 per stay, the dense timeframes from ICU admission, the name of the timeframe column and the bounds filter.
    """
    patient_information = _to_lazy(patient_information)
    all_stays_t0 = _build_t0(patient_information.select(STAY_KEY), t_0_per_stay=window.t_0_per_stay, t_0=window.t_0)
    return Context(
        all_stays_t0,
        _build_base_timeframes(all_stays_t0, patient_information, window.window_size),
        _get_timeframe_name(window.timeframe_name, window.window_size, window.t_0, window.t_0_per_stay),
        _optional_timeframe_bounds_filter("timeframe", window.window_size, t_minus1=window.t_minus1, t_1=window.t_1),
        window.window_size,
    )


def _assemble(ctx: Context, *window_parts: pl.LazyFrame, stay_parts=()) -> pl.LazyFrame:
    """
    The dense timeframes (plus the earlier ones where a part has data) with the parts joined on.

    window_parts have [stay, timeframe, ...], stay_parts [stay, ...]. The bounds are applied last, by `_finalize`, so that
    a fill over the windows sees the ones it is about to drop.
    """
    out = _add_sparse_timeframes(ctx.base, *window_parts)
    for part in window_parts:
        out = out.join(part, on=[STAY_KEY, "timeframe"], how="left")
    for part in stay_parts:
        out = out.join(part, on=STAY_KEY, how="left")
    return out


def _finalize(df: pl.LazyFrame, ctx: Context, *lead, rest: bool = True) -> pl.LazyFrame:
    """The windows within the bounds: stay, T_0 and the timeframe under its name, then the `lead` columns and (unless `rest=False`) all the others, sorted by stay and timeframe."""
    columns = [STAY_KEY, "T_0", pl.col("timeframe").alias(ctx.name), *lead]
    if rest:
        columns.append(pl.all().exclude(STAY_KEY, "T_0", "timeframe", ctx.name, *[c for c in lead if isinstance(c, str)]))
    return df.filter(*ctx.bounds).select(*columns).sort(STAY_KEY, ctx.name)


# endregion


# region calendar days
def _admission_time_to_seconds(data: pl.LazyFrame) -> pl.LazyFrame:
    """Extract seconds since midnight from a pl.Time column.

    Returns
    -------
        Expression yielding seconds since midnight (0-86399)

    Example: 14:30:00 -> 14*3600 + 30*60 = 52200 seconds
    """
    if "Admission Time (24h)" not in data.collect_schema().names():
        return data.with_columns(pl.lit(0, dtype=pl.Int64).alias("Admission Time (seconds)"))  # midnight
    return data.with_columns(
        (pl.col("Admission Time (24h)") - pl.time(0, 0, 0))
        .dt.total_seconds()
        .fill_null(0)  # Default to midnight if missing
        .cast(pl.Int64)
        .alias("Admission Time (seconds)")
    )


def _calendar_day_from_admission(
    time_seconds: pl.Expr, admission_seconds_into_day: pl.Expr
) -> pl.Expr:
    """Compute calendar day number from admission given admission time.

    Calendar day 0 is the day of admission (from midnight to next midnight).
    Takes into account the admission time within that day.

    Arguments
    ---------
        time_seconds : pl.Expr
            Time relative to admission in seconds
        admission_seconds_into_day : pl.Expr
            Seconds from midnight to admission time (0-86399)

    Returns
    -------
        Calendar day index (0 = admission day, 1 = next day, -1 = day before, etc.)

    Examples
    --------
        Admission at 14:00 (50400 seconds into day)
        - time=-3600 (1h before admission = 13:00) -> day 0 (same calendar day)
        - time=-39600 (11h before admission = 03:00) -> day -1 (previous calendar day)
        - time=36000 (10h after admission = 00:00 next day) -> day 1
    """
    # Seconds from the start of admission day
    seconds_from_day_start = time_seconds + admission_seconds_into_day
    # Integer division by SECONDS_IN_1D gives calendar day
    return seconds_from_day_start.floordiv(SECONDS_IN_1D).cast(int)


# endregion
