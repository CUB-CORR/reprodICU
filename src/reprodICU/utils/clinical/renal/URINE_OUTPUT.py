from typing import Optional

import polars as pl

from ...base import (
    SECONDS_IN_1D,
    STAY_KEY,
    TIME_KEY,
    Window,
    _admission_time_to_seconds,
    _assign_timeframe,
    _calendar_day_from_admission,
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

IRRIGATION = "Fluid intake urinary bladder irrigation"


def _improve_inout(inout: pl.LazyFrame) -> pl.LazyFrame:
    """The urine reading of each row (the urine sources summed); rows without a urine value are dropped."""
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
            STAY_KEY,
            TIME_KEY,
            pl.sum_horizontal(
                "Fluid output urine in and out urethral catheter",
                "Fluid output urine nephrostomy",
                "Urine output",
            ).alias("Urine output"),
        )
        .drop_nulls("Urine output")
    )


def _without_irrigation_days(readings: pl.LazyFrame, timeseries_inout: pl.LazyFrame, patient_information: pl.LazyFrame) -> pl.LazyFrame:
    """
    The readings without those of calendar days with bladder irrigation.

    The catheter output of a patient with bladder irrigation contains the irrigant and the irrigant is not charted
    at the time it comes out, so such a day is left out as a whole (as in corr-vars' aki_urine_output).
    """
    if IRRIGATION not in timeseries_inout.collect_schema().names():
        return readings
    admission = _admission_time_to_seconds(patient_information).select(STAY_KEY, "Admission Time (seconds)")
    day = _calendar_day_from_admission(pl.col(TIME_KEY), pl.col("Admission Time (seconds)")).alias("day")
    irrigation_days = (
        timeseries_inout.filter(pl.col(IRRIGATION) > 0)
        .join(admission, on=STAY_KEY)
        .select(STAY_KEY, day)
        .unique()
    )
    return (
        readings.join(admission, on=STAY_KEY, how="left")
        .with_columns(day)
        .join(irrigation_days, on=[STAY_KEY, "day"], how="anti")
        .drop("day", "Admission Time (seconds)")
    )


def _urine_readings(timeseries_inout: pl.LazyFrame, all_stays_t0: pl.LazyFrame, patient_information: pl.LazyFrame) -> pl.LazyFrame:
    """
    The urine readings of the stays, each with the interval it covers relative to T_0.

    A reading covers the time since the previous reading of the stay. The first reading of a stay
    and a reading more than 24 hours after the previous one have no known interval (start is null):
    a long gap often means the patient was away from the ICU, so the urine produced in it is not
    documented and no start is made up. Calendar days with bladder irrigation are left out.
    """
    return (
        _improve_inout(timeseries_inout)
        .pipe(_without_irrigation_days, timeseries_inout, patient_information)
        .join(all_stays_t0, on=STAY_KEY, how="inner")
        .sort(STAY_KEY, TIME_KEY)
        .with_columns(pl.col(TIME_KEY).shift(1).over(STAY_KEY).alias("prev_time"))
        .with_columns(
            pl.when((pl.col("prev_time") < pl.col(TIME_KEY)) & (pl.col(TIME_KEY) - pl.col("prev_time") <= SECONDS_IN_1D))
            .then(pl.col("prev_time") - pl.col("T_0"))
            .alias("Urine Start Relative to T_0 (seconds)"),
            (pl.col(TIME_KEY) - pl.col("T_0")).alias("Urine End Relative to T_0 (seconds)"),
        )
        .drop("prev_time")
    )  # fmt: skip


def _urine_per_window(
    concept: str,
    patient_information: Optional[pl.LazyFrame],
    timeseries_inout: Optional[pl.LazyFrame],
    weight_col: str,
    window: Window,
    first_readings: bool,
) -> pl.LazyFrame:
    """
    Urine output per stay and window: the known intervals spread over the windows they overlap.

    With `first_readings` the volume of a reading without a known interval (the first of the stay, or after
    a gap of more than 24 hours) is added to the window of its time stamp.
    """
    if patient_information is None:
        patient_information = get_patient_information()
    if timeseries_inout is None:
        timeseries_inout = get_timeseries_intakeoutput()
    _validate_required_data(concept, {"patient_information": patient_information, "timeseries_inout": timeseries_inout})

    patient_information = _to_lazy(patient_information)
    all_stays_t0 = _build_t0(patient_information.select(STAY_KEY), window.t_0_per_stay, window.t_0)
    timeseries_inout = _to_lazy(timeseries_inout)
    readings = _urine_readings(timeseries_inout, all_stays_t0, patient_information)

    known = readings.filter(pl.col("Urine Start Relative to T_0 (seconds)").is_not_null()).with_columns(
        (pl.col("Urine output") / (pl.col("Urine End Relative to T_0 (seconds)") - pl.col("Urine Start Relative to T_0 (seconds)"))).alias("urine_rate_ml_per_sec")
    )  # fmt: skip
    per_window = PROJECT_TO_WINDOWS(
        known,
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
    if first_readings:
        first = readings.filter(pl.col("Urine Start Relative to T_0 (seconds)").is_null()).select(
            STAY_KEY,
            _assign_timeframe(TIME_KEY, window.window_size).alias("timeframe"),
            pl.col("Urine output").alias("uo_interval_ml"),
        )
        per_window = (
            pl.concat([per_window, first], how="vertical_relaxed")
            .group_by(STAY_KEY, "timeframe")
            .agg(pl.col("uo_interval_ml").sum())
        )

    name = _get_timeframe_name(window.timeframe_name, window.window_size, window.t_0, window.t_0_per_stay)
    return (
        per_window.join(patient_information.select(STAY_KEY, weight_col), on=STAY_KEY, how="left")
        .with_columns(
            pl.when(pl.col(weight_col) > 0)
            .then(pl.col("uo_interval_ml") / pl.col(weight_col))
            .alias("uo_interval_ml_per_kg")
        )
        .join(all_stays_t0, on=STAY_KEY, how="left")
        .filter(*_optional_timeframe_bounds_filter("timeframe", window.window_size, t_minus1=window.t_minus1, t_1=window.t_1))
        .select(STAY_KEY, pl.col("timeframe").alias(name), "uo_interval_ml", "uo_interval_ml_per_kg")
        .sort(STAY_KEY, name)
    )


@with_windows
def URINE_OUTPUT_VOLUME(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_inout: Optional[pl.LazyFrame] = None,
    *,
    weight_col: str = "Admission Weight (kg)",
    window: Window,
) -> pl.LazyFrame:
    """
    Urine volume per time window, for totals such as the daily urine output of the organ scores.

    A reading covers the time since the previous reading of the stay and its volume is spread over
    the windows that interval overlaps. The first reading of a stay and a reading more than 24 hours
    after the previous one have no known interval (a long gap often means the patient was away from
    the ICU); their volume counts in the window of their time stamp. Calendar days with bladder
    irrigation are left out: the catheter output contains the irrigant. Only windows with urine are
    returned.

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information; must contain Global ICU Stay ID and the weight column.
            Loaded automatically if None.
        timeseries_inout : pl.LazyFrame, optional
            Intake/output timeseries data. Loaded automatically if None.
        weight_col : str, optional
            Weight column of patient_information for the per-kg volume.
            Defaults to "Admission Weight (kg)".

    Returns
    -------
        pl.LazyFrame
            One row per stay and window with urine, with columns:
            - Global ICU Stay ID
            - timeframe (or custom name)
            - uo_interval_ml: Urine volume in mL in the window
            - uo_interval_ml_per_kg: The volume per kg of body weight
    """
    return _urine_per_window("URINE_OUTPUT_VOLUME", patient_information, timeseries_inout, weight_col, window, first_readings=True)


@with_windows
def URINE_OUTPUT_RATE(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_inout: Optional[pl.LazyFrame] = None,
    *,
    weight_col: str = "Admission Weight (kg)",
    window: Window,
) -> pl.LazyFrame:
    """
    Urine produced per time window over known intervals, for rates such as mL/kg/h.

    Same as URINE_OUTPUT_VOLUME, but readings without a known interval (the first of the stay, or
    after a gap of more than 24 hours) are left out: they say nothing about a rate. The windows
    before the second reading and the gaps of more than 24 hours therefore have no value. Calendar
    days with bladder irrigation are left out as in URINE_OUTPUT_VOLUME. Divide a window's volume by the window length (1 h windows give mL/h and mL/kg/h).

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information; must contain Global ICU Stay ID and the weight column.
            Loaded automatically if None.
        timeseries_inout : pl.LazyFrame, optional
            Intake/output timeseries data. Loaded automatically if None.
        weight_col : str, optional
            Weight column of patient_information for the per-kg volume.
            Defaults to "Admission Weight (kg)".

    Returns
    -------
        pl.LazyFrame
            One row per stay and window with urine from known intervals, with columns:
            - Global ICU Stay ID
            - timeframe (or custom name)
            - uo_interval_ml: Urine volume in mL in the window
            - uo_interval_ml_per_kg: The volume per kg of body weight
    """
    return _urine_per_window("URINE_OUTPUT_RATE", patient_information, timeseries_inout, weight_col, window, first_readings=False)


__all__ = ["URINE_OUTPUT_VOLUME", "URINE_OUTPUT_RATE"]
