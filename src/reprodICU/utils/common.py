from typing import Any, Dict, List, Optional, Union

import polars as pl

from ..config import ConfigManager

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
            all_stays.select("Global ICU Stay ID")
            .join(
                t_0_per_stay.select("Global ICU Stay ID", "T_0"),
                "Global ICU Stay ID",
                how="left",
            )
            .with_columns(pl.col("T_0").fill_null(0).cast(pl.Int64))
        )

    t0_val = 0 if t_0 is None else int(t_0)
    return all_stays.select(
        "Global ICU Stay ID", pl.lit(t0_val).cast(pl.Int64).alias("T_0")
    )


def _assign_timeframe(time_col: str, window_size: int) -> pl.Expr:
    return pl.col(time_col).sub(pl.col("T_0")).floordiv(window_size)


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


__all__ = [
    # common utils
    "_to_lazy",
    "_build_t0",
    "_assign_timeframe",
    "_optional_time_bounds_filter",
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
    # data cleaning
    "CLIP_PLAUSIBLE_VALUES",
    "DROP_IMPLAUSIBLE_VALUES",
]
