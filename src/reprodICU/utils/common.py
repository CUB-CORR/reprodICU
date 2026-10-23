from typing import Any, Dict, List, Optional, Tuple, Union

import polars as pl
import polars.selectors as cs

from ..config import ConfigManager

# the windowing helpers moved to base.py; re-exported for the modules that still import them from here
from .base import (  # noqa: F401
    SECONDS_IN_1D,
    SECONDS_IN_1H,
    STAY_KEY,
    T_MINUS1_DEFAULT,
    _add_sparse_timeframes,
    _assign_timeframe,
    _build_base_timeframes,
    _build_t0,
    _get_timeframe_name,
    _null_if_incomplete_pre_icu,
    _optional_timeframe_bounds_filter,
    _sum_points,
    _to_lazy,
    intervention_per_timeframe,
)

# Load config for plausible values
_config = ConfigManager()
CIV = _config.load_config(
    "CLINICALLY_PLAUSIBLE_VALUES.yaml", user_override=True
)


def _validate_required_data(
    concept: str,
    required_data: dict[str, Optional[pl.LazyFrame]],
) -> None:
    """
    Ensure all required datasets/concepts are loaded before computing a concept.

    Arguments
    ---------
        required_data : dict
            Dictionary mapping dataset/concept names to their loaded LazyFrames
            (or None if not loaded).

    Raises
    ------
        ValueError if any required dataset/concept is missing.
    """
    missing = [name for name, data in required_data.items() if data is None]
    if missing:
        raise ValueError(
            f"Cannot compute {concept}: Missing required datasets: {', '.join(missing)}. "
            f"Ensure they are configured in ~/.reprodICU/PATHS.yaml or provide them explicitly."
        )


# region dataset helpers
def _load_dataset(dataset_name: str) -> Optional[pl.LazyFrame]:
    """
    Safely load a dataset from the reprodICU package.

    Avoids circular imports by importing reprodICU only when needed.

    Arguments
    ---------
        dataset_name : str
            Name of the dataset (e.g., 'patient_information', 'timeseries_vitals')

    Returns
    -------
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

    Arguments
    ---------
        concept_name : str
            Name of the concept (e.g., 'ventilation', 'rrt')

    Returns
    -------
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
    """Load timeseries intake/output dataset (with infusion volumes and fluid balance) from reprodICU."""
    return _load_dataset("timeseries_intakeoutput_balanced")


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

    Arguments
    ---------
        obj : Union[pl.LazyFrame, pl.DataFrame, pl.Expr, str]
            Input LazyFrame, DataFrame, Expr, or column name.
        columns : Union[str, List[str]], optional
            Optional list of columns to clean (if obj is a dataframe)
            or the name to use for config lookup (if obj is an expression).
        column_config : Dict[str, Any]
            Dictionary specifying min and max values for each column.

    Returns
    -------
        Union[pl.LazyFrame, pl.DataFrame, pl.Expr]
            Clipped data or clipping expression.
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

    Arguments
    ---------
        obj : Union[pl.LazyFrame, pl.DataFrame, pl.Expr, str]
            Input LazyFrame, DataFrame, Expr, or column name.
        columns : Union[str, List[str]], optional
            Optional list of columns to clean (if obj is a dataframe)
            or the name to use for config lookup (if obj is an expression).
        column_config : Dict[str, Any]
            Dictionary specifying min and max values for each column.

    Returns
    -------
        Union[pl.LazyFrame, pl.DataFrame, pl.Expr]
            Cleaned data or cleaning expression.
    """
    return _plausible_values(obj, columns, column_config, mode="drop")


def NULL_NON_FINITE(
    data: Union[pl.LazyFrame, pl.DataFrame],
) -> Union[pl.LazyFrame, pl.DataFrame]:
    """Set non-finite values (NaN, +-inf) of all float columns to null."""
    return data.with_columns(pl.when(cs.float().is_finite()).then(cs.float()))


def FILTER_NOT_NULL(
    data: Union[pl.LazyFrame, pl.DataFrame], columns: List[str]
) -> Union[pl.LazyFrame, pl.DataFrame]:
    """Keep the rows with a value in at least one of the columns."""
    return data.filter(pl.any_horizontal(pl.col(columns).is_not_null()))


# region struct extraction utilities
def extract_struct_value(
    col_name: str,
    allowed_systems: Optional[List[str]] = None,
    include_none_source: bool = True,
    exact_match: bool = False,
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
        include_none_source : bool, default True
            If True, preserve None values in result.
            If False, filter out rows where value is None.
        exact_match : bool, default False
            If True, system must exactly match one of allowed_systems.
            If False, system can contain allowed_systems as substring (case-insensitive).

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
        expr = extract_struct_value("Sodium", include_none_source=False)

        # Extract with exact system match
        expr = extract_struct_value("pH", ["Blood"], exact_match=True)
    """
    struct = pl.col(col_name).struct
    value  = struct.field("value")
    expr   = value

    if allowed_systems:
        system = struct.field("system")
        match_expr = (
            system.is_in(allowed_systems)
            if exact_match
            else system.str.contains_any(allowed_systems)
        )
        
        if include_none_source:
            match_expr = match_expr | system.is_null()

        expr = pl.when(match_expr).then(value).otherwise(None)

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
                    condition = col <= upper if closed in ("right", "both") else col < upper  # (-∞, upper)
                else:
                    condition = col >= lower if closed in ("left", "both") else col > lower   # (lower, ∞)
            else:
                # Finite interval with specified closure
                if closed in ("left", "right", "both"):
                    condition = col.is_between(lower, upper, closed=closed)
                elif closed == "neither":
                    condition = col.is_between(lower, upper, closed="none")
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
    "_optional_timeframe_bounds_filter",
    "_get_timeframe_name",
    "_validate_required_data",
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
    "NULL_NON_FINITE",
    "FILTER_NOT_NULL",
]
