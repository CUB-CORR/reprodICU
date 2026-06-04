from typing import Any, Dict, List, Optional, Tuple, Union

import polars as pl

from ..config import ConfigManager

STAY_KEY = "Global ICU Stay ID"
SECONDS_IN_1H = 60 * 60
SECONDS_IN_1D = 24 * SECONDS_IN_1H

# Load config for plausible values
_config = ConfigManager()
CIV = _config.load_config(
    "CLINICALLY_PLAUSIBLE_VALUES.yaml", user_override=True
)


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
    all_stays_t_0 = _to_lazy(all_stays_t_0)

    return (
        all_stays_t_0.join(patient_information, on=STAY_KEY, how="left")
        .select(STAY_KEY, "T_0", los_col)
        .with_columns(
            pl.int_ranges(
                start=0,
                end=(pl.col(los_col) * SECONDS_IN_1D - pl.col("T_0"))
                .truediv(window_size)
                .clip(lower_bound=0)
                .ceil()
                .add(1)
                .cast(int),
                step=1,
            ).alias("timeframe")
        )
        .explode("timeframe")
        .unique()
        .select(STAY_KEY, "T_0", "timeframe")
    )


def _assign_timeframe(time_col: str, window_size: int) -> pl.Expr:
    return pl.col(time_col).sub(pl.col("T_0")).floordiv(window_size).cast(int)


def _optional_time_bounds_filter(
    time_col: str, window_size: int, t_0: Optional[int], t_1: Optional[int]
) -> list[pl.Expr]:
    conds: list[pl.Expr] = []
    if t_0 is not None:
        conds.append(
            pl.col(time_col) >= pl.lit(int(t_0)).floordiv(window_size).sub(1)
        )
    if t_1 is not None:
        conds.append(
            pl.col(time_col) < pl.lit(int(t_1)).floordiv(window_size).add(1)
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


# region dataset helpers
def _load_dataset(dataset_name: str) -> Optional[pl.LazyFrame]:
    """
    Safely load a dataset from the reprodICU package.

    Avoids circular imports by importing reprodICU only when needed.

    Args:
        dataset_name: Name of the dataset (e.g., 'patient_information', 'timeseries_vitals')

    Returns:
        pl.LazyFrame if dataset exists, None otherwise
    """
    import reprodICU

    try:
        if reprodICU.dataset_exists(dataset_name):
            return getattr(reprodICU, dataset_name)
        return None
    except Exception:
        return None


def _load_concept(concept_name: str) -> Optional[pl.LazyFrame]:
    """
    Safely load a concept from the reprodICU package.

    Avoids circular imports by importing reprodICU only when needed.

    Args:
        concept_name: Name of the concept (e.g., 'ventilation', 'rrt')

    Returns:
        pl.LazyFrame if concept exists, None otherwise
    """
    import reprodICU

    try:
        if reprodICU.concept_exists(concept_name):
            return getattr(reprodICU, concept_name)
        return None
    except Exception:
        return None


def get_patient_information() -> Optional[pl.LazyFrame]:
    """Load patient information dataset from reprodICU."""
    return _load_dataset("patient_information")


def get_timeseries_vitals() -> Optional[pl.LazyFrame]:
    """Load timeseries vitals dataset from reprodICU."""
    return _load_dataset("timeseries_vitals")


def get_timeseries_labs() -> Optional[pl.LazyFrame]:
    """Load timeseries labs dataset from reprodICU."""
    return _load_dataset("timeseries_labs")


def get_timeseries_respiratory() -> Optional[pl.LazyFrame]:
    """Load timeseries respiratory dataset from reprodICU."""
    return _load_dataset("timeseries_respiratory")


def get_timeseries_intakeoutput() -> Optional[pl.LazyFrame]:
    """Load timeseries intake/output dataset from reprodICU."""
    return _load_dataset("timeseries_intakeoutput")


def get_medications() -> Optional[pl.LazyFrame]:
    """Load medications dataset from reprodICU."""
    return _load_dataset("medications")


def get_prescriptions() -> Optional[pl.LazyFrame]:
    """Load prescriptions dataset from reprodICU."""
    return _load_dataset("prescriptions")


def get_diagnoses() -> Optional[pl.LazyFrame]:
    """Load diagnoses dataset from reprodICU."""
    return _load_dataset("diagnoses")


def get_procedures() -> Optional[pl.LazyFrame]:
    """Load procedures dataset from reprodICU."""
    return _load_dataset("procedures")


def get_notes() -> Optional[pl.LazyFrame]:
    """Load notes dataset from reprodICU."""
    return _load_dataset("notes")


def get_microbiology() -> Optional[pl.LazyFrame]:
    """Load microbiology dataset from reprodICU."""
    return _load_dataset("microbiology")


def get_ventilation() -> Optional[pl.LazyFrame]:
    """Load ventilation concept from reprodICU."""
    return _load_concept("VENTILATION_DURATION")


def get_rrt() -> Optional[pl.LazyFrame]:
    """Load RRT concept from reprodICU."""
    return _load_concept("RENAL_REPLACEMENT_THERAPY_DURATION")


# region timeframe helpers
def intervention_per_timeframe(
    data: pl.LazyFrame,
    patient_information: pl.LazyFrame,
    start_col: str,
    end_col: str,
    t_0: Optional[int] = 0,
    t_0_per_stay: Optional[pl.LazyFrame] = None,
    window_size: int = SECONDS_IN_1H,
) -> pl.LazyFrame:
    """Determine whether an intervention was active in each timeframe per stay."""

    ALL_STAYS = patient_information.select(STAY_KEY)
    ALL_STAYS_T0 = _build_t0(ALL_STAYS, t_0_per_stay=t_0_per_stay, t_0=t_0)
    return (
        data.select(STAY_KEY, start_col, end_col)
        .join(ALL_STAYS_T0, on=STAY_KEY, how="inner")
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
        .with_columns(
            pl.when(pl.col("timeframe").is_not_null())
            .then(True)
            .otherwise(False)
            .alias("intervention"),
        )
        .group_by(STAY_KEY, "timeframe")
        .agg(pl.max("intervention"))
    )


# region data cleaning
def _plausible_values(
    obj: Union[pl.LazyFrame, pl.DataFrame, pl.Expr, str],
    columns: Optional[Union[str, List[str]]],
    column_config: Dict[str, Any],
    mode: str,  # "clip" or "drop"
) -> Union[pl.LazyFrame, pl.DataFrame, pl.Expr]:
    """Internal implementation for clipping or dropping implausible values."""
    if isinstance(obj, (pl.LazyFrame, pl.DataFrame)):
        # Dataframe mode
        target_cols = columns if columns else obj.collect_schema().names()
        if isinstance(target_cols, str):
            target_cols = [target_cols]

        # Check if all target columns are present in column_config
        missing_cols = [c for c in target_cols if c not in column_config]
        if missing_cols:
            raise ValueError(
                f"The following columns are not present in the column_config: {missing_cols}. "
                "Please add them to CLINICALLY_PLAUSIBLE_VALUES.yaml or provide a custom config."
            )

        expressions = [
            _plausible_values(pl.col(col), None, column_config, mode)
            for col in target_cols
        ]

        return obj.with_columns(expressions)

    # Expression mode
    expr = pl.col(obj) if isinstance(obj, str) else obj

    # Determine column name for config lookup
    col_name = columns if isinstance(columns, str) else None
    if col_name is None:
        try:
            col_name = expr.meta.output_name()
        except Exception:
            raise ValueError(
                "Could not determine column name from expression. "
                "Please provide 'columns' as a string for config lookup."
            )

    if col_name not in column_config:
        raise ValueError(f"Column '{col_name}' not in config.")

    limits = column_config[col_name]
    min_val = limits.get("min", float("-inf"))
    max_val = limits.get("max", float("inf"))

    if mode == "clip":
        return expr.clip(
            lower_bound=min_val,
            upper_bound=max_val,
        ).alias(col_name)
    else:  # mode == "drop"
        return (
            pl.when(expr.is_between(min_val, max_val))
            .then(expr)
            .otherwise(None)
            .alias(col_name)
        )


def CLIP_PLAUSIBLE_VALUES(
    obj: Union[pl.LazyFrame, pl.DataFrame, pl.Expr, str],
    columns: Optional[Union[str, List[str]]] = None,
    column_config: Dict[str, Any] = CIV,
) -> Union[pl.LazyFrame, pl.DataFrame, pl.Expr]:
    """
    Clip columns to clinically plausible values.

    Can be used as a dataframe pipe or an expression pipe.

    Arguments:
        obj: Input LazyFrame, DataFrame, Expr, or column name.
        columns: Optional list of columns to clean (if obj is a dataframe)
                 or the name to use for config lookup (if obj is an expression).
        column_config: Dictionary specifying min and max values for each column.

    Returns:
        Union[pl.LazyFrame, pl.DataFrame, pl.Expr]: Clipped data or clipping expression.
    """
    return _plausible_values(obj, columns, column_config, mode="clip")


def DROP_IMPLAUSIBLE_VALUES(
    obj: Union[pl.LazyFrame, pl.DataFrame, pl.Expr, str],
    columns: Optional[Union[str, List[str]]] = None,
    column_config: Dict[str, Any] = CIV,
) -> Union[pl.LazyFrame, pl.DataFrame, pl.Expr]:
    """
    Drop (set to null) values outside the clinically plausible range.

    Can be used as a dataframe pipe or an expression pipe.

    Arguments:
        obj: Input LazyFrame, DataFrame, Expr, or column name.
        columns: Optional list of columns to clean (if obj is a dataframe)
                 or the name to use for config lookup (if obj is an expression).
        column_config: Dictionary specifying min and max values for each column.

    Returns:
        Union[pl.LazyFrame, pl.DataFrame, pl.Expr]: Cleaned data or cleaning expression.
    """
    return _plausible_values(obj, columns, column_config, mode="drop")


# region struct extraction utilities


def extract_struct_value(
    col_name: str,
    allowed_systems: Optional[List[str]] = None,
    include_none: bool = True,
) -> pl.Expr:
    """
    Extract numeric value from measurement struct column.

    Handles struct columns containing 'system' and 'value' fields.
    Optionally filters by specimen system (Serum, Blood, Plasma, CSF, etc.).

    Arguments
    ---------
        col_name : str
            Name of struct column to extract from
        allowed_systems : list of str, optional
            Filter to only these specimen systems. If None, accept all systems.
            Examples: ["Serum", "Blood"], ["Arterial", "Mixed venous"]
        include_none : bool, default True
            If True, preserve None values in result.
            If False, filter out rows where value is None.

    Returns
    -------
        pl.Expr
            Expression that extracts the value (filtered if systems specified)

    Examples
    --------
        # Extract creatinine allowing only serum/blood specimens
        expr = extract_struct_value("Creatinine", ["Serum", "Blood"])

        # Extract platelets from any specimen
        expr = extract_struct_value("Platelets")

        # Extract without None values
        expr = extract_struct_value("Sodium", include_none=False)
    """
    value_expr = pl.col(col_name).struct.field("value")

    if allowed_systems:
        system_expr = pl.col(col_name).struct.field("system")
        system_filter = system_expr.str.contains_any(allowed_systems)
        expr = pl.when(system_filter).then(value_expr).otherwise(None)
    else:
        expr = value_expr

    if not include_none:
        expr = expr.filter(pl.col(col_name).is_not_null())

    return expr


# endregion


# region scoring utilities


class ScoringTable:
    """
    Map numeric value ranges to clinical severity points with closure control.

    Implements range-based scoring where intervals map to integer points.
    Each range specifies lower/upper bounds and closure type to handle
    complex threshold logic accurately.

    Attributes
    ----------
        ranges : List[Tuple[float, float, str, int]]
            List of (lower, upper, closed, points) tuples defining scoring ranges.
            closed: "left" [lower, upper), "right" (lower, upper], "both" [lower, upper],
            or "neither" (lower, upper).

    Examples
    --------
        # Score with multiple ranges and different closure types
        scores = ScoringTable([          # Heart rate (bpm) | Points
            (None,   33, "neither", 4),  #  <33               4
            (  33,   88, "left",    0),  #   33- 88 ......... 0
            (  89,  106, "left",    1),  #   89-106           1
            ( 107,  125, "right",   3),  #  107-125           3
            ( 125, None, "neither", 6),  # >125               6
        ])

        df = data.with_columns(
            scores.to_expr(pl.col("HeartRate")).alias("HR_Points")
        )
    """

    def __init__(self, ranges: List[Tuple[float, float, str, int]]):
        """
        Initialize scoring table with explicit ranges and closure types.

        Arguments
        ---------
            ranges : List[Tuple[float, float, str, int]]
                List of (lower, upper, closed, points) defining scoring intervals.
                Closure types:
                - "left":    [lower, upper) - includes lower, excludes upper
                - "right":   (lower, upper] - excludes lower, includes upper
                - "both":    [lower, upper] - includes both
                - "neither": (lower, upper) - excludes both
        """
        self.ranges = ranges

    def to_expr(self, col: pl.Expr) -> pl.Expr:
        """Convert scoring table to Polars when/then/otherwise expression."""
        expr = None
        for lower, upper, closed, points in self.ranges:
            if lower is None or upper is None:
                # Handle infinite boundaries
                if lower is None:
                    condition = col < upper  # (-∞, upper)
                else:
                    condition = col > lower  # (lower, ∞)
            else:
                # Finite interval with specified closure
                if closed == "left":
                    condition = col.is_between(lower, upper, closed="left")
                elif closed == "right":
                    condition = col.is_between(lower, upper, closed="right")
                elif closed == "both":
                    condition = col.is_between(lower, upper, closed="both")
                elif closed == "neither":
                    condition = col.is_between(lower, upper, closed="neither")
                else:
                    raise ValueError(f"Unknown closure type: {closed}")

            if expr is None:
                expr = pl.when(condition).then(points)
            else:
                expr = expr.when(condition).then(points)

        return expr.otherwise(None)

    def __repr__(self) -> str:
        """Return descriptive representation."""
        ranges_str = ", ".join(
            f"({l},{u}:{c})→{p}" for l, u, c, p in self.ranges
        )
        return f"ScoringTable([{ranges_str}])"


# endregion


__all__ = [
    # common utils
    "_to_lazy",
    "_build_t0",
    "_build_base_timeframes",
    "_assign_timeframe",
    "_optional_time_bounds_filter",
    "_get_timeframe_name",
    # dataset loaders
    "get_patient_information",
    "get_timeseries_vitals",
    "get_timeseries_labs",
    "get_timeseries_respiratory",
    "get_timeseries_intakeoutput",
    "get_medications",
    "get_diagnoses",
    "get_procedures",
    "get_notes",
    "get_microbiology",
    # concept loaders
    "get_ventilation",
    "get_rrt",
    # measurement utilities
    "extract_struct_value",
    "ScoringTable",
    # data cleaning
    "CLIP_PLAUSIBLE_VALUES",
    "DROP_IMPLAUSIBLE_VALUES",
]
