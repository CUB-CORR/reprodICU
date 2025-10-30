"""
Free Days Calculations: compute ventilator-free days, RRT-free days, and vasopressor-free days.

Free days are calendar days within a timeframe (typically 28 days post-admission) where a patient
was alive and not receiving a specific intervention (ventilation, RRT, or vasopressors).

Output columns per row:
- Global ICU Stay ID
- [Free Days Type] Free Days ([timeframe]d): Number of calendar days free from intervention
- Total [Free Days Type] Days ([timeframe]d): Number of calendar days on intervention
- Days Alive in Timeframe: Total calendar days the patient was alive
- Mortality [timeframe] Days After ICU Admission: Binary mortality flag

Time is in seconds. Calendar day indices are computed from 24-hour blocks.

SOURCES
-------
- Yehya N, Harhay MO, Curley MAQ, Schoenfeld DA, Reeder RW.
  Reappraisal of Ventilator-Free Days in Critical Care Research.
  Am J Respir Crit Care Med. 2019 Oct 1;200(7):828-836.
  doi: 10.1164/rccm.201810-2050CP. PMID: 31034248; PMCID: PMC6812447.
"""

from typing import Optional

import polars as pl

from ..common import (
    _to_lazy,
    get_patient_information,
    get_medications,
    get_ventilation,
    get_rrt,
)
from ..mortality import COMMON_MORTALITY_MEASURES

SECONDS_IN_1D = 24 * 60 * 60
SECONDS_IN_1H = 60 * 60
SUCCESSFUL_EXTUBATION_GAP_SECONDS = 48 * 60 * 60
VASOPRESSOR_MEDICATIONS = [
    "dopamine",
    "dobutamine",
    "epinephrine",
    "norepinephrine",
    "terlipressin",
    "vasopressin",
]

__all__ = [
    "VENTILATOR_FREE_DAYS",
    "RENAL_REPLACEMENT_THERAPY_FREE_DAYS",
    "VASOPRESSOR_FREE_DAYS",
]


# region time helpers
def _time_str_to_seconds(time_col: pl.Expr) -> pl.Expr:
    """Convert time column to seconds from midnight."""
    return (time_col - pl.time(0, 0, 0)).dt.total_seconds()


# region calendar day
def _calculate_event_calendar_days_from_intervals(
    event_data: pl.LazyFrame,
    patient_information: pl.LazyFrame,
    timeframe_days: int,
    start_time_col: str,
    end_time_col: str,
    output_calendar_days_col: str,
    additional_filter_expr: Optional[pl.Expr] = None,
) -> pl.LazyFrame:
    """
    Calculate which calendar days within a timeframe contain events from time intervals.

    Arguments
    ---------
        event_data : pl.LazyFrame
            Event data with start and end times relative to admission (seconds).
        patient_information : pl.LazyFrame
            Patient info with admission times and stay IDs.
        timeframe_days : int
            Total timeframe in calendar days.
        start_time_col : str
            Column name for event start time.
        end_time_col : str
            Column name for event end time.
        output_calendar_days_col : str
            Name for output column listing calendar day indices with events.
        additional_filter_expr : pl.Expr, optional
            Additional filter to apply to event_data (e.g., drug filtering).

    Returns
    -------
        pl.LazyFrame
            One row per stay with column of calendar day indices containing events.
    """
    timeframe_total_seconds = timeframe_days * SECONDS_IN_1D
    event_data_filtered = (
        event_data
        if additional_filter_expr is None
        else event_data.filter(additional_filter_expr)
    )

    event_with_admission_time = event_data_filtered.join(
        patient_information.select(
            "Global ICU Stay ID", "Admission Time (24h)"
        ),
        on="Global ICU Stay ID",
        how="left",
        coalesce=True,
    ).with_columns(
        _time_str_to_seconds(pl.col("Admission Time (24h)"))
        .fill_null(0)
        .alias("Admission Time Offset (seconds)")
    )

    return (
        event_with_admission_time.with_columns(
            pl.col(start_time_col)
            .clip(lower_bound=0)
            .alias("Start Relative to Admission (seconds)"),
            pl.col(end_time_col).alias("End Relative to Admission (seconds)"),
        )
        .filter(
            (
                pl.col("End Relative to Admission (seconds)")
                > pl.col("Start Relative to Admission (seconds)")
            )
            & (
                pl.col("Start Relative to Admission (seconds)")
                < timeframe_total_seconds
            )
        )
        .with_columns(
            pl.min_horizontal(
                pl.col("End Relative to Admission (seconds)"),
                pl.lit(timeframe_total_seconds),
            ).alias("End Relative to Admission (seconds)")
        )
        .filter(
            pl.col("End Relative to Admission (seconds)")
            > pl.col("Start Relative to Admission (seconds)")
        )
        .with_columns(
            (
                (
                    pl.col("Start Relative to Admission (seconds)")
                    + pl.col("Admission Time Offset (seconds)")
                ).floordiv(SECONDS_IN_1D)
            )
            .cast(int)
            .alias("Start Day (index)"),
            (
                (
                    pl.col("End Relative to Admission (seconds)")
                    + pl.col("Admission Time Offset (seconds)")
                    - 1
                ).floordiv(SECONDS_IN_1D)
            )
            .cast(int)
            .alias("End Day (index)"),
        )
        .with_columns(
            pl.when(pl.col("Start Day (index)") <= pl.col("End Day (index)"))
            .then(
                pl.int_ranges(
                    pl.col("Start Day (index)"),
                    pl.col("End Day (index)") + 1,
                )
            )
            .otherwise(pl.lit([], dtype=pl.List(int)))
            .alias("Calendar Day Index")
        )
        .explode("Calendar Day Index")
        .filter(
            pl.col("Calendar Day Index").is_not_null(),
            pl.col("Calendar Day Index") < timeframe_days,
            pl.col("Calendar Day Index") >= 0,
        )
        .group_by("Global ICU Stay ID")
        .agg(
            pl.col("Calendar Day Index")
            .unique()
            .sort()
            .alias(output_calendar_days_col)
        )
    )


# region event-free days
def calculate_event_free_days(
    patient_information: pl.LazyFrame,
    event_summary: pl.LazyFrame,
    mortality: pl.LazyFrame,
    timeframe_days: int,
    event_calendar_days_colname: str,
    free_days_type_name: str,
) -> pl.LazyFrame:
    """
    Calculate free days by comparing alive calendar days with event calendar days.

    Arguments
    ---------
        patient_information : pl.LazyFrame
            Patient/stay-level information.
        event_summary : pl.LazyFrame
            Summary of calendar days with events (from _calculate_event_calendar_days_from_intervals).
        mortality : pl.LazyFrame
            Mortality data with mortality flags and timing.
        timeframe_days : int
            Total timeframe in calendar days.
        event_calendar_days_colname : str
            Name of the column listing event calendar days.
        free_days_type_name : str
            Human-readable name for the event type (e.g., "Ventilator", "RRT").

    Returns
    -------
        pl.LazyFrame
            One row per stay with free days, total event days, and mortality info.
    """
    return (
        patient_information.select(
            "Global ICU Stay ID",
            "ICU Length of Stay (days)",
            "Admission Time (24h)",
            "Mortality After ICU Discharge (days)",
        )
        .join(
            event_summary.select(
                "Global ICU Stay ID", event_calendar_days_colname
            ),
            on="Global ICU Stay ID",
            how="left",
            coalesce=True,
        )
        .join(
            mortality.select(
                "Global ICU Stay ID",
                f"Mortality {timeframe_days} Days After ICU Admission",
                "Mortality in ICU",
            ),
            on="Global ICU Stay ID",
            how="left",
            coalesce=True,
        )
        .with_columns(
            pl.col(event_calendar_days_colname).fill_null(
                pl.lit([], dtype=pl.List(int))
            ),
            pl.col(
                f"Mortality {timeframe_days} Days After ICU Admission"
            ).fill_null(False),
            pl.col("Mortality in ICU").fill_null(False),
            pl.col("Mortality After ICU Discharge (days)").fill_null(int(1e6)),
            _time_str_to_seconds(pl.col("Admission Time (24h)"))
            .fill_null(0)
            .alias("Admission Time Offset (seconds)"),
        )
        .with_columns(
            pl.when(pl.col("Mortality in ICU"))
            .then(pl.col("ICU Length of Stay (days)"))
            .otherwise(
                pl.col("ICU Length of Stay (days)")
                + pl.col("Mortality After ICU Discharge (days)")
            )
            .alias("Survival Duration (days)")
        )
        .with_columns(
            pl.when(pl.col("Survival Duration (days)") < 0)
            .then(pl.lit(0))
            .otherwise(pl.col("Survival Duration (days)"))
            .alias("Survival Duration (days)")
        )
        .with_columns(
            (pl.col("Survival Duration (days)") * SECONDS_IN_1D).alias(
                "Survival Duration (seconds)"
            )
        )
        .with_columns(
            pl.when(pl.col("Survival Duration (seconds)") > 0)
            .then(
                (
                    pl.col("Survival Duration (seconds)")
                    + pl.col("Admission Time Offset (seconds)")
                    - 1
                ).floordiv(SECONDS_IN_1D)
            )
            .otherwise(-1)
            .cast(int)
            .alias("Last Day Alive (index)")
        )
        .with_columns(
            pl.min_horizontal(
                pl.col("Last Day Alive (index)"),
                pl.lit(timeframe_days - 1),
            ).alias("Last Day Alive (index)")
        )
        .with_columns(
            pl.when(pl.col("Last Day Alive (index)") >= 0)
            .then(pl.int_ranges(0, pl.col("Last Day Alive (index)") + 1))
            .otherwise(pl.lit([], dtype=pl.List(int)))
            .alias("Alive Calendar Days")
        )
        .with_columns(
            pl.col("Alive Calendar Days")
            .list.len()
            .cast(int)
            .alias("Days Alive in Timeframe")
        )
        .with_columns(
            pl.when(
                pl.col(f"Mortality {timeframe_days} Days After ICU Admission")
            )
            .then(pl.lit(0))
            .otherwise(
                pl.col("Alive Calendar Days")
                .list.set_difference(pl.col(event_calendar_days_colname))
                .list.len()
                .cast(int)
            )
            .alias(f"{free_days_type_name} Free Days ({timeframe_days}d)")
        )
        .select(
            "Global ICU Stay ID",
            f"{free_days_type_name} Free Days ({timeframe_days}d)",
            pl.col(event_calendar_days_colname)
            .list.len()
            .alias(f"Total {free_days_type_name} Days ({timeframe_days}d)"),
            "Days Alive in Timeframe",
            f"Mortality {timeframe_days} Days After ICU Admission",
        )
    )


# region vent-free days
def VENTILATOR_FREE_DAYS(
    patient_information: Optional[pl.LazyFrame] = None,
    ventilation: Optional[pl.LazyFrame] = None,
    timeframe_days: int = 28,
) -> pl.LazyFrame:
    """
    Calculate ventilator-free days (28d by default).

    Ventilator-free days are calendar days where the patient was alive and not receiving
    invasive mechanical ventilation or tracheostomy.

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information. Loaded automatically if None.
        ventilation : pl.LazyFrame, optional
            Ventilation timeseries data. Loaded automatically if None.
        timeframe_days : int, optional
            Timeframe in calendar days (default: 28).

    Returns
    -------
        pl.LazyFrame
            One row per stay with:
            - Global ICU Stay ID
            - Ventilator Free Days (28d) (or other timeframe)
            - Total Ventilator Days (28d)
            - Days Alive in Timeframe
            - Mortality 28 Days After ICU Admission
    """
    # Load defaults if not provided
    if patient_information is None:
        patient_information = get_patient_information()
    if ventilation is None:
        ventilation = get_ventilation()

    # Validate all required data is available
    required = {
        "patient_information": patient_information,
        "ventilation": ventilation,
    }

    missing = [name for name, data in required.items() if data is None]
    if missing:
        raise ValueError(
            f"Cannot compute VENTILATOR_FREE_DAYS: Missing required datasets: {', '.join(missing)}. "
            f"Ensure they are configured in ~/.reprodICU/PATHS.yaml or provide them explicitly."
        )

    # Ensure lazy
    patient_information = _to_lazy(patient_information)
    ventilation = _to_lazy(ventilation)

    # Compute mortality measures
    mortality = COMMON_MORTALITY_MEASURES(patient_information)

    # Filter to invasive ventilation types
    filtered_ventilation = ventilation.filter(
        pl.col("Ventilation Type").is_in(
            ["invasive ventilation", "tracheostomy"]
        )
        | pl.col("Ventilation Type").is_null()
    )

    # Merge consecutive ventilation blocks separated by less than 48 hours (reintubation)
    processed_ventilation_data = (
        filtered_ventilation.sort(
            "Global ICU Stay ID",
            "Ventilation Start Relative to Admission (seconds)",
        )
        .with_columns(
            pl.col("Ventilation End Relative to Admission (seconds)")
            .shift(1)
            .over("Global ICU Stay ID")
            .alias("Previous Ventilation End")
        )
        .with_columns(
            (
                (
                    (
                        pl.col(
                            "Ventilation Start Relative to Admission (seconds)"
                        )
                        - pl.col("Previous Ventilation End")
                    )
                    > SUCCESSFUL_EXTUBATION_GAP_SECONDS
                )
                | pl.col("Previous Ventilation End").is_null()
            ).alias("is_new_block")
        )
        .with_columns(
            pl.col("is_new_block")
            .cum_sum()
            .over("Global ICU Stay ID")
            .alias("block_id")
        )
        .group_by("Global ICU Stay ID", "block_id")
        .agg(
            pl.col("Ventilation Start Relative to Admission (seconds)")
            .min()
            .alias("Ventilation Start Relative to Admission (seconds)"),
            pl.col("Ventilation End Relative to Admission (seconds)")
            .max()
            .alias("Ventilation End Relative to Admission (seconds)"),
        )
        .drop("block_id")
    )

    ventilation_summary = _calculate_event_calendar_days_from_intervals(
        event_data=processed_ventilation_data,
        patient_information=patient_information,
        timeframe_days=timeframe_days,
        start_time_col="Ventilation Start Relative to Admission (seconds)",
        end_time_col="Ventilation End Relative to Admission (seconds)",
        output_calendar_days_col="Ventilator Calendar Days",
    )

    return calculate_event_free_days(
        patient_information=patient_information,
        event_summary=ventilation_summary,
        mortality=mortality,
        timeframe_days=timeframe_days,
        event_calendar_days_colname="Ventilator Calendar Days",
        free_days_type_name="Ventilator",
    )


# region rrt-free days
def RENAL_REPLACEMENT_THERAPY_FREE_DAYS(
    patient_information: Optional[pl.LazyFrame] = None,
    rrt: Optional[pl.LazyFrame] = None,
    timeframe_days: int = 28,
) -> pl.LazyFrame:
    """
    Calculate renal replacement therapy (RRT)-free days (28d by default).

    RRT-free days are calendar days where the patient was alive and not receiving
    any renal replacement therapy.

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information. Loaded automatically if None.
        rrt : pl.LazyFrame, optional
            RRT timeseries data with start/end times relative to admission.
            Loaded automatically if None.
        timeframe_days : int, optional
            Timeframe in calendar days (default: 28).

    Returns
    -------
        pl.LazyFrame
            One row per stay with:
            - Global ICU Stay ID
            - RRT Free Days (28d) (or other timeframe)
            - Total RRT Days (28d)
            - Days Alive in Timeframe
            - Mortality 28 Days After ICU Admission
    """
    # Load defaults if not provided
    if patient_information is None:
        patient_information = get_patient_information()
    if rrt is None:
        rrt = get_rrt()

    # Validate all required data is available
    required = {
        "patient_information": patient_information,
        "rrt": rrt,
    }

    missing = [name for name, data in required.items() if data is None]
    if missing:
        raise ValueError(
            f"Cannot compute RENAL_REPLACEMENT_THERAPY_FREE_DAYS: Missing required datasets: {', '.join(missing)}. "
            f"Ensure they are configured in ~/.reprodICU/PATHS.yaml or provide them explicitly."
        )

    # Ensure lazy
    patient_information = _to_lazy(patient_information)
    rrt = _to_lazy(rrt)

    # Compute mortality measures
    mortality = COMMON_MORTALITY_MEASURES(patient_information)

    rrt_summary = _calculate_event_calendar_days_from_intervals(
        event_data=rrt,
        patient_information=patient_information,
        timeframe_days=timeframe_days,
        start_time_col="Renal Replacement Therapy Start Relative to Admission (seconds)",
        end_time_col="Renal Replacement Therapy End Relative to Admission (seconds)",
        output_calendar_days_col="RRT Calendar Days",
    )

    return calculate_event_free_days(
        patient_information=patient_information,
        event_summary=rrt_summary,
        mortality=mortality,
        timeframe_days=timeframe_days,
        event_calendar_days_colname="RRT Calendar Days",
        free_days_type_name="RRT",
    )


# region rrt-free days
def VASOPRESSOR_FREE_DAYS(
    patient_information: Optional[pl.LazyFrame] = None,
    medications: Optional[pl.LazyFrame] = None,
    timeframe_days: int = 28,
) -> pl.LazyFrame:
    """
    Calculate vasopressor-free days (28d by default).

    Vasopressor-free days are calendar days where the patient was alive and not receiving
    vasoactive medications (dopamine, epinephrine, norepinephrine, vasopressin, terlipressin,
    or dobutamine) at positive rates.

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information. Loaded automatically if None.
        medications : pl.LazyFrame, optional
            Medication administrations data. Loaded automatically if None.
        timeframe_days : int, optional
            Timeframe in calendar days (default: 28).

    Returns
    -------
        pl.LazyFrame
            One row per stay with:
            - Global ICU Stay ID
            - Vasopressor Free Days (28d) (or other timeframe)
            - Total Vasopressor Days (28d)
            - Days Alive in Timeframe
            - Mortality 28 Days After ICU Admission
    """
    # Load defaults if not provided
    if patient_information is None:
        patient_information = get_patient_information()
    if medications is None:
        medications = get_medications()

    # Validate all required data is available
    required = {
        "patient_information": patient_information,
        "medications": medications,
    }

    missing = [name for name, data in required.items() if data is None]
    if missing:
        raise ValueError(
            f"Cannot compute VASOPRESSOR_FREE_DAYS: Missing required datasets: {', '.join(missing)}. "
            f"Ensure they are configured in ~/.reprodICU/PATHS.yaml or provide them explicitly."
        )

    # Ensure lazy
    patient_information = _to_lazy(patient_information)
    medications = _to_lazy(medications)

    # Compute mortality measures
    mortality = COMMON_MORTALITY_MEASURES(patient_information)

    vasopressor_filter = pl.col("Drug Ingredient").is_in(
        VASOPRESSOR_MEDICATIONS
    ) & pl.col("Drug Rate").gt(0)

    vasopressor_summary = _calculate_event_calendar_days_from_intervals(
        event_data=medications,
        patient_information=patient_information,
        timeframe_days=timeframe_days,
        start_time_col="Drug Start Relative to Admission (seconds)",
        end_time_col="Drug End Relative to Admission (seconds)",
        output_calendar_days_col="Vasopressor Calendar Days",
        additional_filter_expr=vasopressor_filter,
    )

    return calculate_event_free_days(
        patient_information=patient_information,
        event_summary=vasopressor_summary,
        mortality=mortality,
        timeframe_days=timeframe_days,
        event_calendar_days_colname="Vasopressor Calendar Days",
        free_days_type_name="Vasopressor",
    )
