from typing import Optional, Literal

import polars as pl

from ...common import (
    _build_t0,
    _to_lazy,
    get_patient_information,
    get_timeseries_labs,
    get_timeseries_vitals,
    get_timeseries_respiratory,
)

SECONDS_IN_4H = 4 * 60 * 60
pf_ratio_col = "PaO2/FiO2 Ratio"
sf_ratio_col = "SpO2/FiO2 Ratio"


# region helpers
def _improve_resp(
    resp: pl.LazyFrame,
    fio2_type: Literal["invasive", "non-invasive", "both"] = "both",
) -> pl.LazyFrame:
    VENT_FIO2 = pl.max_horizontal(
        "Oxygen/Total gas setting [Volume Fraction] Ventilator",
        "Oxygen/Gas total [Pure volume fraction] Inhaled gas",
    )

    FLOW_FIO2 = (
        pl.col("Oxygen gas flow Oxygen delivery system")
        .mul(3)
        .add(21)
        .clip(upper_bound=100)
    )

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
        resp.with_columns(combined_fio2.alias("FiO2"))
        .select(
            "Global ICU Stay ID",
            "Time Relative to Admission (seconds)",
            pl.when(pl.col("FiO2").is_between(0.21, 1))
            .then(pl.col("FiO2") * 100)
            .when(pl.col("FiO2").is_between(21, 100))
            .then(pl.col("FiO2"))
            .otherwise(None)
            .alias("FiO2"),
        )
        .drop_nulls("FiO2")
    )


def _improve_labs(labs: pl.LazyFrame) -> pl.LazyFrame:
    return labs.select(
        "Global ICU Stay ID",
        "Time Relative to Admission (seconds)",
        pl.when(
            pl.col("Oxygen")
            .struct.field("system")
            .is_in(["Blood arterial", "Blood"])
            | pl.col("Oxygen").struct.field("system").is_null()
        )
        .then(pl.col("Oxygen").struct.field("value"))
        .otherwise(None)
        .alias("Oxygen in Arterial blood"),
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
def PaO2_FiO2_RATIO(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_resp: Optional[pl.LazyFrame] = None,
    timeseries_labs: Optional[pl.LazyFrame] = None,
    *,
    t_0: Optional[int] = 0,
    t_0_per_stay: Optional[pl.LazyFrame] = None,
    tolerance: Optional[int] = SECONDS_IN_4H,
    fio2_type: Literal["invasive", "non-invasive", "both"] = "both",
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
        t_0 : int, optional
            Scalar reference time (seconds from admission). Defaults to 0 (admission).
            Ignored when t_0_per_stay is provided.
        t_0_per_stay : pl.LazyFrame, optional
            Per-stay T_0 overrides with columns [Global ICU Stay ID, T_0].
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

    missing = [name for name, data in required.items() if data is None]
    if missing:
        raise ValueError(
            f"Cannot compute PaO2_FiO2_RATIO: Missing required datasets: {', '.join(missing)}. "
            f"Ensure they are configured in ~/.reprodICU/PATHS.yaml or provide them explicitly."
        )

    STAY_KEY = "Global ICU Stay ID"
    TIME_KEY = "Time Relative to Admission (seconds)"

    patient_information = _to_lazy(patient_information)
    timeseries_resp = _to_lazy(timeseries_resp)
    timeseries_labs = _to_lazy(timeseries_labs)
    t_0_per_stay = _to_lazy(t_0_per_stay) if t_0_per_stay is not None else None

    all_stays = patient_information.select(STAY_KEY)
    all_stays_t0 = _build_t0(all_stays, t_0_per_stay, t_0)

    timeseries_resp = _improve_resp(timeseries_resp, fio2_type=fio2_type)
    timeseries_labs = _improve_labs(timeseries_labs)

    pf_ratio = (
        timeseries_labs
        # FiO2 occurred at most 4 hours before this blood gas
        .join_asof(
            timeseries_resp,
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

    if (t_0 != 0) or (t_0_per_stay is not None):
        pf_ratio = (
            pf_ratio.join(all_stays_t0, on=STAY_KEY, how="inner")
            .with_columns(
                pl.col(TIME_KEY)
                .sub(pl.col("T_0"))
                .alias("Time Relative to T_0 (seconds)")
            )
            .drop("T_0")
        )

    return pf_ratio


# region SpO2/FiO2 Ratio
def SpO2_FiO2_RATIO(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_resp: Optional[pl.LazyFrame] = None,
    timeseries_vitals: Optional[pl.LazyFrame] = None,
    *,
    t_0: Optional[int] = 0,
    t_0_per_stay: Optional[pl.LazyFrame] = None,
    tolerance: Optional[int] = SECONDS_IN_4H,
    fio2_type: Literal["invasive", "non-invasive", "both"] = "both",
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
        t_0 : int, optional
            Scalar reference time (seconds from admission). Defaults to 0 (admission).
            Ignored when t_0_per_stay is provided.
        t_0_per_stay : pl.LazyFrame, optional
            Per-stay T_0 overrides with columns [Global ICU Stay ID, T_0].
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

    missing = [name for name, data in required.items() if data is None]
    if missing:
        raise ValueError(
            f"Cannot compute SpO2_FiO2_RATIO: Missing required datasets: {', '.join(missing)}. "
            f"Ensure they are configured in ~/.reprodICU/PATHS.yaml or provide them explicitly."
        )

    STAY_KEY = "Global ICU Stay ID"
    TIME_KEY = "Time Relative to Admission (seconds)"

    patient_information = _to_lazy(patient_information)
    timeseries_resp = _to_lazy(timeseries_resp)
    timeseries_vitals = _to_lazy(timeseries_vitals)
    t_0_per_stay = _to_lazy(t_0_per_stay) if t_0_per_stay is not None else None

    all_stays = patient_information.select(STAY_KEY)
    all_stays_t0 = _build_t0(all_stays, t_0_per_stay, t_0)

    timeseries_resp = _improve_resp(timeseries_resp, fio2_type=fio2_type)
    timeseries_vitals = _improve_vitals(timeseries_vitals)

    sf_ratio = (
        timeseries_vitals
        # FiO2 occurred at most 4 hours before this blood gas
        .join_asof(
            timeseries_resp,
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

    if (t_0 != 0) or (t_0_per_stay is not None):
        sf_ratio = (
            sf_ratio.join(all_stays_t0, on=STAY_KEY, how="inner")
            .with_columns(
                pl.col(TIME_KEY)
                .sub(pl.col("T_0"))
                .alias("Time Relative to T_0 (seconds)")
            )
            .drop("T_0")
        )

    return sf_ratio


___all__ = ["PaO2_FiO2_RATIO", "SpO2_FiO2_RATIO"]
