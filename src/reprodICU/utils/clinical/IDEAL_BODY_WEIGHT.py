"""
Anthropometric Calculations: compute ideal body weight, adjusted body weight, body surface area, and BMI classifications.

This module implements several anthropometric formulas:
- Devine formula for ideal body weight
- Lorentz formula for ideal body weight
- Adjusted body weight for dosing in obese patients
- Mosteller formula for body surface area
- WHO classification for body mass index (BMI)

Output columns depend on the function:
- Ideal Body Weight (Devine): Calculated ideal body weight in kg
- Ideal Body Weight (Lorentz): Calculated ideal body weight in kg
- Adjusted Body Weight: Adjusted body weight for dosing in kg
- Body Surface Area: Body surface area in m²
- Body Mass Index: BMI in kg/m²
- BMI Classification: WHO classification category

SOURCES
-------
- https://www.mdcalc.com/calc/68/ideal-body-weight-adjusted-body-weight
- https://www.mdcalc.com/calc/29/body-mass-index-bmi-body-surface-area-bsa
"""

from typing import Optional

import polars as pl

from ..common import _to_lazy, get_patient_information

height_col = "Admission Height (cm)"
weight_col = "Admission Weight (kg)"
gender_col = "Gender"


# region Helpers
def _load_patient_information(
    patient_information: Optional[pl.LazyFrame] = None,
) -> pl.LazyFrame:
    """
    Load and validate patient information data.

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information. Loaded automatically if None.

    Returns
    -------
        pl.LazyFrame
            Patient information data converted to lazy frame.

    Raises
    ------
        ValueError
            If patient information cannot be loaded.
    """
    if patient_information is None:
        patient_information = get_patient_information()

    if patient_information is None:
        raise ValueError(
            "Cannot load patient_information: Missing data parameter. "
            "Ensure patient_information is configured in ~/.reprodICU/PATHS.yaml or provide data explicitly."
        )

    return _to_lazy(patient_information)


# region IBW
def IDEAL_BODY_WEIGHT_DEVINE(
    patient_information: Optional[pl.LazyFrame] = None,
) -> pl.LazyFrame:
    """
    Calculate the ideal body weight using the Devine formula.

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information. Loaded automatically if None.
            Must contain: Global ICU Stay ID, Admission Height (cm), Gender

    Returns
    -------
        pl.LazyFrame
            Selected columns with:
            - Global ICU Stay ID
            - Ideal Body Weight (Devine): Calculated ideal body weight in kg
    """
    patient_information = _load_patient_information(patient_information)

    return patient_information.with_columns(
        pl.when(pl.col(gender_col) == "Female")
        .then(45.5 + 2.3 * (pl.col(height_col) / 2.54 - 60))
        .otherwise(50 + 2.3 * (pl.col(height_col) / 2.54 - 60))
        .alias("Ideal Body Weight (Devine)")
    ).select("Global ICU Stay ID", "Ideal Body Weight (Devine)")


# region IBW
def IDEAL_BODY_WEIGHT_LORENTZ(
    patient_information: Optional[pl.LazyFrame] = None,
) -> pl.LazyFrame:
    """
    Calculate the ideal body weight using the Lorentz formula.

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information. Loaded automatically if None.
            Must contain: Global ICU Stay ID, Admission Height (cm), Gender

    Returns
    -------
        pl.LazyFrame
            Selected columns with:
            - Global ICU Stay ID
            - Ideal Body Weight (Lorentz): Calculated ideal body weight in kg
    """
    patient_information = _load_patient_information(patient_information)

    return patient_information.with_columns(
        pl.when(pl.col(gender_col) == "Female")
        .then((pl.col(height_col) - 100) - ((pl.col(height_col) - 150) / 2))
        .otherwise(
            (pl.col(height_col) - 100) - ((pl.col(height_col) - 150) / 4)
        )
        .alias("Ideal Body Weight (Lorentz)")
    ).select("Global ICU Stay ID", "Ideal Body Weight (Lorentz)")


# region ABW
def ADJUSTED_BODY_WEIGHT(
    patient_information: Optional[pl.LazyFrame] = None,
) -> pl.LazyFrame:
    """
    Calculate the adjusted body weight for dosing in obese patients.

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information. Loaded automatically if None.
            Must contain: Global ICU Stay ID, Admission Height (cm), Admission Weight (kg), Gender

    Returns
    -------
        pl.LazyFrame
            Selected columns with:
            - Global ICU Stay ID
            - Ideal Body Weight (Devine)
            - Adjusted Body Weight: Adjusted weight for dosing in kg (or None if not obese)
    """
    patient_information = _load_patient_information(patient_information)

    return (
        patient_information.pipe(IDEAL_BODY_WEIGHT_DEVINE)
        .with_columns(
            pl.when(pl.col(weight_col) > pl.col("Ideal Body Weight (Devine)"))
            .then(
                pl.col("Ideal Body Weight (Devine)")
                + 0.4
                * (pl.col(weight_col) - pl.col("Ideal Body Weight (Devine)"))
            )
            .otherwise(None)
            .alias("Adjusted Body Weight")
        )
        .select(
            "Global ICU Stay ID",
            "Ideal Body Weight (Devine)",
            "Adjusted Body Weight",
        )
    )


# region BSA
def BODY_SURFACE_AREA(
    patient_information: Optional[pl.LazyFrame] = None,
) -> pl.LazyFrame:
    """
    Calculate the body surface area using the Mosteller formula.

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information. Loaded automatically if None.
            Must contain: Global ICU Stay ID, Admission Height (cm), Admission Weight (kg)

    Returns
    -------
        pl.LazyFrame
            Input data with additional column:
            - Body Surface Area: Body surface area in m²
    """
    patient_information = _load_patient_information(patient_information)

    return patient_information.with_columns(
        (((pl.col(height_col) * pl.col(weight_col)) / 3600) ** 0.5).alias(
            "Body Surface Area"
        )
    )


# region BMI
def BODY_MASS_INDEX(
    patient_information: Optional[pl.LazyFrame] = None,
) -> pl.LazyFrame:
    """
    Calculate the body mass index (BMI).

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information. Loaded automatically if None.
            Must contain: Global ICU Stay ID, Admission Height (cm), Admission Weight (kg)

    Returns
    -------
        pl.LazyFrame
            Input data with additional column:
            - Body Mass Index: BMI in kg/m²
    """
    patient_information = _load_patient_information(patient_information)

    return patient_information.with_columns(
        (pl.col(weight_col) / ((pl.col(height_col) / 100) ** 2)).alias(
            "Body Mass Index"
        )
    )


# region BMI Class
def CLASSIFY_BODY_MASS_INDEX(
    patient_information: Optional[pl.LazyFrame] = None,
) -> pl.LazyFrame:
    """
    Classify the body mass index (BMI) according to the WHO classification.

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information. Loaded automatically if None.
            Must contain: Global ICU Stay ID, Admission Height (cm), Admission Weight (kg)

    Returns
    -------
        pl.LazyFrame
            Input data with additional columns:
            - Body Mass Index: BMI in kg/m²
            - BMI Classification: WHO classification category (Severe Thinness, Moderate Thinness,
              Mild Thinness, Normal, Overweight, Obese Class I, II, or III)
    """
    patient_information = _load_patient_information(patient_information)

    bmi_classification_dtype = pl.Enum(
        [
            "Severe Thinness",
            "Moderate Thinness",
            "Mild Thinness",
            "Normal",
            "Overweight",
            "Obese Class I (Moderate)",
            "Obese Class II (Severe)",
            "Obese Class III (Very Severe)",
        ]
    )

    return patient_information.pipe(BODY_MASS_INDEX).with_columns(
        pl.when(pl.col("Body Mass Index").is_null())
        .then(None)
        .when(pl.col("Body Mass Index") < 16)
        .then(pl.lit("Severe Thinness"))
        .when(pl.col("Body Mass Index") < 17)
        .then(pl.lit("Moderate Thinness"))
        .when(pl.col("Body Mass Index") < 18.5)
        .then(pl.lit("Mild Thinness"))
        .when(pl.col("Body Mass Index") < 25)
        .then(pl.lit("Normal"))
        .when(pl.col("Body Mass Index") < 30)
        .then(pl.lit("Overweight"))
        .when(pl.col("Body Mass Index") < 35)
        .then(pl.lit("Obese Class I (Moderate)"))
        .when(pl.col("Body Mass Index") < 40)
        .then(pl.lit("Obese Class II (Severe)"))
        .otherwise(pl.lit("Obese Class III (Very Severe)"))
        .cast(bmi_classification_dtype)
        .alias("BMI Classification")
    )


__all__ = [
    "IDEAL_BODY_WEIGHT_DEVINE",
    "IDEAL_BODY_WEIGHT_LORENTZ",
    "ADJUSTED_BODY_WEIGHT",
    "BODY_SURFACE_AREA",
    "BODY_MASS_INDEX",
    "CLASSIFY_BODY_MASS_INDEX",
]
