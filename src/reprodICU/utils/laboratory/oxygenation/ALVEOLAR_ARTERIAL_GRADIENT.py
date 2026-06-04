from typing import Optional

import polars as pl

from ...common import (
    _build_t0,
    _to_lazy,
    _validate_required_data,
    get_patient_information,
    get_timeseries_labs,
    get_timeseries_respiratory,
)

SECONDS_IN_4H = 4 * 60 * 60

ATMOSPHERIC_PRESSURE_MMHG = 760  # mmHg
WATER_VAPOR_PRESSURE_MMHG = 47  # mmHg
RESPIRATORY_QUOTIENT = 0.8  # CO2 produced / O2 consumed

PAO2_col = "Oxygen in Alveolus by calculation"
Aa_gradient_col = "Alveolar-arterial oxygen Partial pressure difference"


# region helpers
def _improve_resp(resp: pl.LazyFrame) -> pl.LazyFrame:
    return (
        resp.with_columns(
            pl.max_horizontal(
                "Oxygen/Total gas setting [Volume Fraction] Ventilator",
                "Oxygen/Gas total [Pure volume fraction] Inhaled gas",
            ).alias("FiO2")
        )
        .select(
            "Global ICU Stay ID",
            "Time Relative to Admission (seconds)",
            pl.when(pl.col("FiO2").is_between(0, 1))
            .then(pl.col("FiO2") * 100)
            .when(pl.col("FiO2").is_between(1, 100))
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
        pl.when(
            pl.col("Carbon dioxide")
            .struct.field("system")
            .is_in(["Blood arterial", "Blood"])
            | pl.col("Carbon dioxide").struct.field("system").is_null()
        )
        .then(pl.col("Carbon dioxide").struct.field("value"))
        .otherwise(None)
        .alias("Carbon dioxide in Arterial blood"),
    )


# region PAO2
def PAO2(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_resp: Optional[pl.LazyFrame] = None,
    timeseries_labs: Optional[pl.LazyFrame] = None,
    *,
    t_0: Optional[int] = 0,
    t_0_per_stay: Optional[pl.LazyFrame] = None,
) -> pl.LazyFrame:
    """
    Calculate PAO2 from respiratory and laboratory timeseries.

    PAO2 is the alveolar oxygen partial pressure, calculated using the alveolar gas equation.

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

    Returns
    -------
        pl.LazyFrame
            PAO2 timeseries with columns:
            - Global ICU Stay ID
            - Time Relative to Admission (seconds)
            - PAO2
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
    _validate_required_data(concept="PAO2", required_data=required)

    STAY_KEY = "Global ICU Stay ID"
    TIME_KEY = "Time Relative to Admission (seconds)"

    patient_information = _to_lazy(patient_information)
    timeseries_resp = _to_lazy(timeseries_resp)
    timeseries_labs = _to_lazy(timeseries_labs)
    t_0_per_stay = _to_lazy(t_0_per_stay) if t_0_per_stay is not None else None

    all_stays = patient_information.select(STAY_KEY)
    all_stays_t0 = _build_t0(all_stays, t_0_per_stay, t_0)

    timeseries_resp = _improve_resp(timeseries_resp)
    timeseries_labs = _improve_labs(timeseries_labs)

    PAO2 = (
        timeseries_labs
        # FiO2 occurred at most 4 hours before this blood gas
        .join_asof(
            timeseries_resp,
            on=TIME_KEY,
            by=STAY_KEY,
            strategy="backward",
            tolerance=SECONDS_IN_4H,
            coalesce=True,
        )
        .with_columns(
            (
                pl.col("FiO2").fill_null(21).truediv(100)  # 21% if FiO2 missing
                .mul(ATMOSPHERIC_PRESSURE_MMHG - WATER_VAPOR_PRESSURE_MMHG)
                - pl.col("Carbon dioxide in Arterial blood")
                .truediv(RESPIRATORY_QUOTIENT)
            ).alias(PAO2_col)
        )
        .filter(pl.col(PAO2_col).is_finite())
        .select(STAY_KEY, TIME_KEY, PAO2_col)
    ) # fmt: skip

    if (t_0 != 0) or (t_0_per_stay is not None):
        PAO2 = (
            PAO2.join(all_stays_t0, on=STAY_KEY, how="inner")
            .with_columns(
                pl.col(TIME_KEY)
                .sub(pl.col("T_0"))
                .alias("Time Relative to T_0 (seconds)")
            )
            .drop("T_0")
        )

    return PAO2


# endregion PAO2


# region Aa_GRADIENT
def Aa_GRADIENT(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_resp: Optional[pl.LazyFrame] = None,
    timeseries_labs: Optional[pl.LazyFrame] = None,
    *,
    t_0: Optional[int] = 0,
    t_0_per_stay: Optional[pl.LazyFrame] = None,
) -> pl.LazyFrame:
    """
    Calculate the alveolar-arterial oxygen gradient (A-a gradient).

    The A-a gradient is the difference between alveolar oxygen partial pressure
    (PAO2) and arterial oxygen partial pressure (PaO2).

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

    Returns
    -------
        pl.LazyFrame
            A-a gradient timeseries with columns:
            - Global ICU Stay ID
            - Time Relative to Admission (seconds)
            - Alveolar-Arterial Oxygen Gradient
    """
    # Load defaults if not provided
    if patient_information is None:
        patient_information = get_patient_information()
    if timeseries_resp is None:
        timeseries_resp = get_timeseries_respiratory()
    if timeseries_labs is None:
        timeseries_labs = get_timeseries_labs()

    STAY_KEY = "Global ICU Stay ID"
    TIME_KEY = "Time Relative to Admission (seconds)"

    patient_information = _to_lazy(patient_information)
    timeseries_labs = _to_lazy(timeseries_labs)
    t_0_per_stay = _to_lazy(t_0_per_stay) if t_0_per_stay is not None else None

    all_stays = patient_information.select(STAY_KEY)
    all_stays_t0 = _build_t0(all_stays, t_0_per_stay, t_0)

    # Get PAO2
    PAO2_df = PAO2(
        patient_information=patient_information,
        timeseries_resp=timeseries_resp,
        timeseries_labs=timeseries_labs,
        t_0=t_0,
        t_0_per_stay=t_0_per_stay,
    )

    timeseries_labs = _improve_labs(timeseries_labs)

    # Join PAO2 and PaO2, then calculate A-a gradient
    Aa_gradient = (
        PAO2_df.join(
            timeseries_labs.select(
                STAY_KEY,
                TIME_KEY,
                pl.col("Oxygen in Arterial blood"),
            ).drop_nulls(),
            on=[STAY_KEY, TIME_KEY],
            how="inner",
        )
        .with_columns(
            pl.col(PAO2_col)
            .sub(pl.col("Oxygen in Arterial blood"))
            .alias(Aa_gradient_col)
        )
        .select(STAY_KEY, TIME_KEY, Aa_gradient_col)
    )

    if (t_0 != 0) or (t_0_per_stay is not None):
        Aa_gradient = (
            Aa_gradient.join(all_stays_t0, on=STAY_KEY, how="inner")
            .with_columns(
                pl.col(TIME_KEY)
                .sub(pl.col("T_0"))
                .alias("Time Relative to T_0 (seconds)")
            )
            .drop("T_0")
        )

    return Aa_gradient


# endregion Aa_GRADIENT


__all__ = ["PAO2", "Aa_GRADIENT"]
