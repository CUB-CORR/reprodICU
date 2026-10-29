from typing import Literal, Optional

import polars as pl

from ...base import (
    SECONDS_IN_4H,
    STAY_KEY,
    TIME_KEY,
    Reference,
    _reference_t0,
    _relative_to_t0,
    _to_lazy,
    with_reference_time,
)
from ...common import (
    _validate_required_data,
    extract_struct_value,
    get_patient_information,
    get_timeseries_labs,
    get_timeseries_respiratory,
    get_timeseries_vitals,
)
from ...core import RESPIRATORY_PARAMS

pf_ratio_col = "PaO2/FiO2 Ratio"
sf_ratio_col = "SpO2/FiO2 Ratio"


# region helpers
def _improve_resp(
    resp: pl.LazyFrame,
    fio2_type: Literal["invasive", "non-invasive", "both"] = "both",
) -> pl.LazyFrame:
    resp = RESPIRATORY_PARAMS(timeseries_resp=resp)

    VENT_FIO2 = pl.col("FiO2")
    FLOW_FIO2 = pl.col("LPM").mul(3).add(21).clip(upper_bound=100)

    if fio2_type == "non-invasive":
        resp = resp.filter(
            pl.col("Oxygen delivery system").str.contains_any(
                ["High flow", "Continuous positive"]
            )
        )
        combined_fio2 = FLOW_FIO2
    elif fio2_type == "invasive":
        combined_fio2 = VENT_FIO2
    else:  # fio2_type == "both"
        combined_fio2 = pl.coalesce(VENT_FIO2, FLOW_FIO2)

    return (
        resp.select(
            "Global ICU Stay ID",
            "Time Relative to Admission (seconds)",
            combined_fio2.alias("FiO2"),
        )
        .drop_nulls("FiO2")
    )


def _improve_labs(labs: pl.LazyFrame) -> pl.LazyFrame:
    return labs.select(
        "Global ICU Stay ID",
        "Time Relative to Admission (seconds)",
        extract_struct_value(
            "Oxygen", ["Blood arterial", "Blood"], exact_match=True
        ).alias("Oxygen in Arterial blood"),
    ).drop_nulls("Oxygen in Arterial blood")


def _improve_vitals(vitals: pl.LazyFrame) -> pl.LazyFrame:
    return vitals.filter(
        pl.col("Peripheral oxygen saturation").is_not_null(),
        pl.col("Peripheral oxygen saturation").is_between(0, 98),
    ).select(
        "Global ICU Stay ID",
        "Time Relative to Admission (seconds)",
        pl.col("Peripheral oxygen saturation").alias("SpO2"),
    )


# region PaO2/FiO2 Ratio
@with_reference_time
def PaO2_FiO2_RATIO(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_resp: Optional[pl.LazyFrame] = None,
    timeseries_labs: Optional[pl.LazyFrame] = None,
    *,
    tolerance: Optional[int] = SECONDS_IN_4H,
    fio2_type: Literal["invasive", "non-invasive", "both"] = "both",
    ref: Reference,
) -> pl.LazyFrame:
    """
    Calculate PaO2/FiO2 ratio timeseries from respiratory and laboratory timeseries.

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information. Loaded automatically if None.
        timeseries_resp : pl.LazyFrame, optional
            Respiratory timeseries data. Loaded automatically if None.
        timeseries_labs : pl.LazyFrame, optional
            Laboratory timeseries data. Loaded automatically if None.
        tolerance : int, optional
            Join tolerance in seconds. Default is 4 hours.
        fio2_type : str, optional
            Source of FiO2: 'invasive' (ventilator), 'non-invasive' (flow-based),
            or 'both' (ventilator with flow-based fallback). Defaults to 'both'.

    Returns
    -------
        pl.LazyFrame
            PaO2/FiO2 Ratio timeseries with columns:
            - Global ICU Stay ID
            - Time Relative to Admission (seconds)
            - PaO2/FiO2 Ratio
    """
    # Load defaults if not provided
    if patient_information is None:
        patient_information = get_patient_information()
    if timeseries_resp is None:
        timeseries_resp = get_timeseries_respiratory()
    if timeseries_labs is None:
        timeseries_labs = get_timeseries_labs()

    # Validate all required data is available
    required = {
        "patient_information": patient_information,
        "timeseries_resp": timeseries_resp,
        "timeseries_labs": timeseries_labs,
    }
    _validate_required_data("PaO2_FiO2_RATIO", required)

    patient_information = _to_lazy(patient_information)
    timeseries_resp = _to_lazy(timeseries_resp)
    timeseries_labs = _to_lazy(timeseries_labs)
    all_stays_t0 = _reference_t0(patient_information, ref)

    timeseries_resp = _improve_resp(timeseries_resp, fio2_type=fio2_type)
    timeseries_labs = _improve_labs(timeseries_labs)

    pf_ratio = (
        timeseries_labs.sort(STAY_KEY, TIME_KEY)
        # join_asof needs both sides sorted by the keys
        # FiO2 occurred at most 4 hours before this blood gas
        .join_asof(
            timeseries_resp.sort(STAY_KEY, TIME_KEY),
            on=TIME_KEY,
            by=STAY_KEY,
            strategy="backward",
            tolerance=tolerance,
            coalesce=True,
        )
        .with_columns(
            pl.col("Oxygen in Arterial blood")
            .truediv(
                pl.col("FiO2").fill_null(21).truediv(100)
            )  # 21% if FiO2 missing
            .alias(pf_ratio_col)
        )
        .with_columns(
            pl.when(pl.col(pf_ratio_col).is_finite())
            .then(pl.col(pf_ratio_col))
            .otherwise(None)
            .alias(pf_ratio_col)
        )
        .select(STAY_KEY, TIME_KEY, pf_ratio_col)
    )

    pf_ratio = _relative_to_t0(pf_ratio, all_stays_t0, ref)

    return pf_ratio


# region SpO2/FiO2 Ratio
@with_reference_time
def SpO2_FiO2_RATIO(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_resp: Optional[pl.LazyFrame] = None,
    timeseries_vitals: Optional[pl.LazyFrame] = None,
    *,
    tolerance: Optional[int] = SECONDS_IN_4H,
    fio2_type: Literal["invasive", "non-invasive", "both"] = "both",
    ref: Reference,
) -> pl.LazyFrame:
    """
    Calculate SpO2/FiO2 ratio timeseries from respiratory and vital signs timeseries.

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information. Loaded automatically if None.
        timeseries_resp : pl.LazyFrame, optional
            Respiratory timeseries data. Loaded automatically if None.
        timeseries_vitals : pl.LazyFrame, optional
            Vital signs timeseries data. Loaded automatically if None.
        tolerance : int, optional
            Join tolerance in seconds. Default is 4 hours.
        fio2_type : str, optional
            Source of FiO2: 'invasive' (ventilator), 'non-invasive' (flow-based),
            or 'both' (ventilator with flow-based fallback). Defaults to 'both'.

    Returns
    -------
        pl.LazyFrame
            SpO2/FiO2 Ratio timeseries with columns:
            - Global ICU Stay ID
            - Time Relative to Admission (seconds)
            - SpO2/FiO2 Ratio
    """
    # Load defaults if not provided
    if patient_information is None:
        patient_information = get_patient_information()
    if timeseries_resp is None:
        timeseries_resp = get_timeseries_respiratory()
    if timeseries_vitals is None:
        timeseries_vitals = get_timeseries_vitals()

    # Validate all required data is available
    required = {
        "patient_information": patient_information,
        "timeseries_resp": timeseries_resp,
        "timeseries_vitals": timeseries_vitals,
    }
    _validate_required_data("SpO2_FiO2_RATIO", required)

    patient_information = _to_lazy(patient_information)
    timeseries_resp = _to_lazy(timeseries_resp)
    timeseries_vitals = _to_lazy(timeseries_vitals)
    all_stays_t0 = _reference_t0(patient_information, ref)

    timeseries_resp = _improve_resp(timeseries_resp, fio2_type=fio2_type)
    timeseries_vitals = _improve_vitals(timeseries_vitals)

    sf_ratio = (
        timeseries_vitals.sort(STAY_KEY, TIME_KEY)
        # join_asof needs both sides sorted by the keys
        # FiO2 occurred at most 4 hours before this blood gas
        .join_asof(
            timeseries_resp.sort(STAY_KEY, TIME_KEY),
            on=TIME_KEY,
            by=STAY_KEY,
            strategy="backward",
            tolerance=tolerance,
            coalesce=True,
        )
        .with_columns(
            pl.col("SpO2")
            .truediv(
                pl.col("FiO2").fill_null(21).truediv(100)
            )  # 21% if FiO2 missing
            .alias(sf_ratio_col)
        )
        .with_columns(
            pl.when(pl.col(sf_ratio_col).is_finite())
            .then(pl.col(sf_ratio_col))
            .otherwise(None)
            .alias(sf_ratio_col)
        )
        .select(STAY_KEY, TIME_KEY, sf_ratio_col)
    )

    sf_ratio = _relative_to_t0(sf_ratio, all_stays_t0, ref)

    return sf_ratio


__all__ = ["PaO2_FiO2_RATIO", "SpO2_FiO2_RATIO"]
