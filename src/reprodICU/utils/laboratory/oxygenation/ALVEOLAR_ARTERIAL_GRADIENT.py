from typing import Optional

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
)
from ...core import RESPIRATORY_PARAMS

ATMOSPHERIC_PRESSURE_MMHG = 760  # mmHg
WATER_VAPOR_PRESSURE_MMHG = 47  # mmHg
RESPIRATORY_QUOTIENT = 0.8  # CO2 produced / O2 consumed

PAO2_col = "Oxygen in Alveolus by calculation"
Aa_gradient_col = "Alveolar-arterial oxygen Partial pressure difference"


# region helpers
def _improve_resp(resp: pl.LazyFrame) -> pl.LazyFrame:
    return (
        RESPIRATORY_PARAMS(timeseries_resp=resp)
        .select(
            "Global ICU Stay ID",
            "Time Relative to Admission (seconds)",
            "FiO2",
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
        extract_struct_value(
            "Carbon dioxide", ["Blood arterial", "Blood"], exact_match=True
        ).alias("Carbon dioxide in Arterial blood"),
    )


# region PAO2
@with_reference_time
def PAO2(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_resp: Optional[pl.LazyFrame] = None,
    timeseries_labs: Optional[pl.LazyFrame] = None,
    *,
    ref: Reference,
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
    _validate_required_data("PAO2", required)

    patient_information = _to_lazy(patient_information)
    timeseries_resp = _to_lazy(timeseries_resp)
    timeseries_labs = _to_lazy(timeseries_labs)
    all_stays_t0 = _reference_t0(patient_information, ref)

    timeseries_resp = _improve_resp(timeseries_resp)
    timeseries_labs = _improve_labs(timeseries_labs)

    PAO2 = (
        timeseries_labs.sort(STAY_KEY, TIME_KEY)
        # join_asof needs both sides sorted by the keys
        # FiO2 occurred at most 4 hours before this blood gas
        .join_asof(
            timeseries_resp.sort(STAY_KEY, TIME_KEY),
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

    PAO2 = _relative_to_t0(PAO2, all_stays_t0, ref)

    return PAO2


# endregion PAO2


# region Aa_GRADIENT
@with_reference_time
def Aa_GRADIENT(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_resp: Optional[pl.LazyFrame] = None,
    timeseries_labs: Optional[pl.LazyFrame] = None,
    *,
    ref: Reference,
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

    # Validate all required data is available
    required = {
        "patient_information": patient_information,
        "timeseries_resp": timeseries_resp,
        "timeseries_labs": timeseries_labs,
    }
    _validate_required_data("Aa_GRADIENT", required)

    patient_information = _to_lazy(patient_information)
    timeseries_labs = _to_lazy(timeseries_labs)
    all_stays_t0 = _reference_t0(patient_information, ref)

    # Get PAO2
    PAO2_df = PAO2(
        patient_information=patient_information,
        timeseries_resp=timeseries_resp,
        timeseries_labs=timeseries_labs,
        t_0=ref.t_0,
        t_0_per_stay=ref.t_0_per_stay,
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

    Aa_gradient = _relative_to_t0(Aa_gradient, all_stays_t0, ref)

    return Aa_gradient


# endregion Aa_GRADIENT


__all__ = ["PAO2", "Aa_GRADIENT"]
