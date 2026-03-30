"""
Glomerular Filtration Rate (GFR) Estimation: compute estimated GFR using multiple validated formulas.

This module implements several GFR estimation formulas:
- 2021 CKD-EPI Creatinine formula
- 2021 CKD-EPI Creatinine-Cystatin C formula
- Cockcroft-Gault equation for creatinine clearance
- MDRD GFR formula

Output columns depend on the function:
- estimated GFR (CKD-EPI Creatinine): eGFR in mL/min/1.73m² using CKD-EPI creatinine formula
- estimated GFR (CKD-EPI Creatinine-Cystatin): eGFR in mL/min/1.73m² using combined formula
- estimated GFR (Cockcroft-Gault): Creatinine clearance in mL/min using Cockcroft-Gault equation
- estimated GFR (MDRD): eGFR in mL/min/1.73m² using MDRD formula

Input values required:
- Creatinine: serum creatinine in mg/dL
- Age: patient age in years
- Gender: patient gender (Male / Female)
- Ethnicity: for MDRD formula (optional for other formulas)
- Height/Weight: for Cockcroft-Gault formula

| Formula            | Parameters                          |
|--------------------|-------------------------------------|
| CKD-EPI Creatinine | Age, Gender, Creatinine             |
| CKD-EPI Cr-Cys     | Age, Gender, Creatinine, Cystatin C |
| Cockcroft-Gault    | Age, Gender, Weight, Creatinine     |
| MDRD               | Age, Gender, Creatinine, Ethnicity  |

SOURCES
-------
- Cockcroft-Gault equation (1976):
    Cockcroft DW, Gault MH.
    Prediction of creatinine clearance from serum creatinine.
    Nephron. 1976;16(1):31-41.
    doi: 10.1159/000180580. PMID: 1244564.
- MDRD equation (2006):
    Levey AS, Coresh J, Greene T, Stevens LA, Zhang YL, Hendriksen S, Kusek JW, Van Lente F; Chronic Kidney Disease Epidemiology Collaboration.
    Using standardized serum creatinine values in the modification of diet in renal disease study equation for estimating glomerular filtration rate.
    Ann Intern Med. 2006 Aug 15;145(4):247-54.
    doi: 10.7326/0003-4819-145-4-200608150-00004. PMID: 16908915.
- CKD-EPI 2021 Creatinine and Creatinine-Cystatin C equations (2021):
    Inker LA, Eneanya ND, Coresh J, Tighiouart H, Wang D, Sang Y, Crews DC, Doria A, Estrella MM, Froissart M, Grams ME, Greene T, Grubb A, Gudnason V, Gutiérrez OM, Kalil R, Karger AB, Mauer M, Navis G, Nelson RG, Poggio ED, Rodby R, Rossing P, Rule AD, Selvin E, Seegmiller JC, Shlipak MG, Torres VE, Yang W, Ballew SH, Couture SJ, Powe NR, Levey AS; Chronic Kidney Disease Epidemiology Collaboration.
    New Creatinine- and Cystatin C-Based Equations to Estimate GFR without Race.
    N Engl J Med. 2021 Nov 4;385(19):1737-1749.
    doi: 10.1056/NEJMoa2102953. Epub 2021 Sep 23. PMID: 34554658; PMCID: PMC8822996.
"""

from typing import Optional

import numpy as np
import polars as pl

from ...common import _to_lazy, get_patient_information, get_timeseries_labs
from ..IDEAL_BODY_WEIGHT import ADJUSTED_BODY_WEIGHT

age_col = "Admission Age (years)"
height_col = "Admission Height (cm)"
weight_col = "Admission Weight (kg)"
gender_col = "Gender"
ethnicity_col = "Ethnicity"
creatinine_col = "Creatinine"
cystatin_c_col = "Cystatin C"


# region CKD
def CKD_EPI_Creatinine_parameters(data: pl.LazyFrame) -> pl.LazyFrame:
    """
    Get the parameters for the CKD-EPI Creatinine GFR formula.

    Parameters vary based on gender and serum creatinine level:

    | Female                  | Male                    |
    |-------------------------|-------------------------|
    | SCr ≤0.7   | SCr >0.7   | SCr ≤0.9   | SCr >0.9   |
    |-------------------------|-------------------------|
    | A =  0.7   | A =  0.7   | A =  0.9   | A =  0.9   |
    | B = -0.241 | B = -1.2   | B = -0.302 | B = -1.2   |
    |-------------------------|-------------------------|

    Arguments
    ---------
        data : pl.LazyFrame
            Input data frame containing gender and creatinine columns.

    Returns
    -------
        pl.LazyFrame
            Input data with additional columns:
            - A1: Creatinine reference value based on gender
            - B1: Exponent based on gender and creatinine level
    """

    return data.with_columns(
        pl.when(pl.col(gender_col) == "Female")
        .then(0.7)
        .otherwise(0.9)
        .alias("A1"),
        pl.when(pl.col(gender_col) == "Female")
        .then(
            pl.when(pl.col(creatinine_col) > 0.7).then(-1.2).otherwise(-0.241)
        )
        .otherwise(
            pl.when(pl.col(creatinine_col) > 0.9).then(-1.2).otherwise(-0.302)
        )
        .alias("B1"),
    )


def CKD_EPI_Creatinine_Cystatin_parameters(data: pl.LazyFrame) -> pl.LazyFrame:
    """
    Get the parameters for the CKD-EPI Creatinine Cystatin-C GFR formula.

    Parameters vary based on gender, creatinine, and cystatin C levels:

    |             Female                    | Male                      |
    |---------------------------------------|---------------------------|
    |           | SCr ≤0.7    | SCr >0.7    | SCr ≤0.9    | SCr >0.9    |
    |---------------------------------------|---------------------------|
    | Scys ≤0.8 | A =  0.7    | A =  0.7    | A =  0.9    | A =  0.9    |
    |           | B = -0.219  | B = -0.544  | B = -0.144  | B = -0.544  |
    |           | C =  0.8    | C =  0.8    | C =  0.8    | C =  0.8    |
    |           | D = -0.323  | D = -0.323  | D = -0.323  | D = -0.323  |
    |---------------------------------------|---------------------------|
    | Scys >0.8 | A =  0.7    | A =  0.7    | A =  0.9    | A =  0.9    |
    |           | B = -0.219  | B = -0.544  | B = -0.144  | B = -0.544  |
    |           | C =  0.8    | C =  0.8    | C =  0.8    | C =  0.8    |
    |           | D = -0.778  | D = -0.778  | D = -0.778  | D = -0.778  |
    |-------------------------------------------------------------------|

    Arguments
    ---------
        data : pl.LazyFrame
            Input data frame containing gender, creatinine, and cystatin C columns.

    Returns
    -------
        pl.LazyFrame
            Input data with additional columns:
            - A2: Creatinine reference value based on gender
            - B2: Creatinine exponent based on gender and creatinine level
            - C2: Cystatin C reference value (fixed at 0.8)
            - D2: Cystatin C exponent based on cystatin C level
    """

    return data.with_columns(
        pl.when(pl.col(gender_col) == "Female")
        .then(0.7)
        .otherwise(0.9)
        .alias("A2"),
        pl.when(pl.col(gender_col) == "Female")
        .then(
            pl.when(pl.col(creatinine_col) <= 0.7)
            .then(-0.219)
            .otherwise(-0.544)
        )
        .otherwise(
            pl.when(pl.col(creatinine_col) <= 0.9)
            .then(-0.144)
            .otherwise(-0.544)
        )
        .alias("B2"),
        pl.lit(0.8).alias("C2"),
        pl.when(pl.col(cystatin_c_col) <= 0.8)
        .then(-0.323)
        .otherwise(-0.778)
        .alias("D2"),
    )


def CKD_EPI_Creatinine(data: pl.LazyFrame) -> pl.LazyFrame:
    """
    Calculate eGFR using the 2021 CKD-EPI Creatinine formula.

    Arguments
    ---------
        data : pl.LazyFrame
            Input data frame containing gender, age, and creatinine columns.

    Returns
    -------
        pl.LazyFrame
            Input data with additional column:
            - estimated GFR (CKD-EPI Creatinine): Estimated GFR in mL/min/1.73m²
    """

    return data.pipe(CKD_EPI_Creatinine_parameters).with_columns(
        (
            142
            * ((pl.col(creatinine_col) / pl.col("A1")) ** pl.col("B1"))
            * (0.9938 ** pl.col(age_col))
            * pl.when(pl.col(gender_col) == "Female").then(1.012).otherwise(1)
        ).alias("estimated GFR (CKD-EPI Creatinine)")
    )


def CKD_EPI_Creatinine_Cystatin(data: pl.LazyFrame) -> pl.LazyFrame:
    """
    Calculate eGFR using the 2021 CKD-EPI Creatinine-Cystatin C formula.

    Arguments
    ---------
        data : pl.LazyFrame
            Input data frame containing gender, age, creatinine, and cystatin C columns.

    Returns
    -------
        pl.LazyFrame
            Input data with additional column:
            - estimated GFR (CKD-EPI Creatinine-Cystatin): Estimated GFR in mL/min/1.73m²
    """

    return data.pipe(CKD_EPI_Creatinine_Cystatin_parameters).with_columns(
        (
            135
            * (pl.col(creatinine_col) / pl.col("A2")) ** pl.col("B2")
            * (pl.col(cystatin_c_col) / pl.col("C2")) ** pl.col("D2")
            * 0.9961 ** pl.col(age_col)
            * pl.when(pl.col(gender_col) == "Female").then(0.963).otherwise(1)
        ).alias("estimated GFR (CKD-EPI Creatinine-Cystatin)")
    )


def CKD_EPI(data: pl.LazyFrame, use_cystatin_c: bool = False) -> pl.LazyFrame:
    """
    Calculate eGFR using CKD-EPI formula (creatinine alone or combined with cystatin C).

    Arguments
    ---------
        data : pl.LazyFrame
            Input data frame containing gender, age, and creatinine columns (and optionally cystatin C).
        use_cystatin_c : bool, optional
            If True, uses combined creatinine-cystatin C formula; otherwise uses creatinine-only formula.

    Returns
    -------
        pl.LazyFrame
            Selected columns with:
            - Global ICU Stay ID
            - Gender, age, and creatinine columns (and cystatin C if provided)
            - estimated GFR (CKD-EPI Creatinine) or estimated GFR (CKD-EPI Creatinine-Cystatin)
    """

    if not use_cystatin_c:
        return data.pipe(CKD_EPI_Creatinine).select(
            "Global ICU Stay ID",
            "estimated GFR (CKD-EPI Creatinine)",
        )

    return data.pipe(
        CKD_EPI_Creatinine_Cystatin,
    ).select(
        "Global ICU Stay ID", "estimated GFR (CKD-EPI Creatinine-Cystatin)"
    )


# region reverse CKD-EPI
def reverse_CKD_EPI_Creatinine(
    data: pl.LazyFrame, target_egfr: float = 75
) -> pl.LazyFrame:
    """
    Estimate creatinine from target eGFR using the reverse CKD-EPI Creatinine formula.

    Reverses the CKD-EPI 2021 creatinine formula using the upper piecewise branch:
    eGFR = 142 * ((SCr/A) ^ B) * (0.9938 ^ age) * female_factor

    Valid physiological solutions fall on the upper branch (Scr > 0.7 for females, > 0.9 for males).
    For each gender, computes:
    K = eGFR / (142 * 0.9938^age * female_factor)

    Then applies the upper branch formula:
    - Female (Scr > 0.7): Scr = 0.7 * K^(1/-1.2)
    - Male (Scr > 0.9): Scr = 0.9 * K^(1/-1.2)

    Arguments
    ---------
        data : pl.LazyFrame
            Input data frame containing gender and age columns.
        target_egfr : float, optional
            Target eGFR value to estimate creatinine for. Default: 75 mL/min/1.73m².

    Returns
    -------
        pl.LazyFrame
            Input data with additional columns:
            - estimated Creatinine (CKD-EPI eGFR={target_egfr}): Estimated serum creatinine in mg/dL
    """
    # Calculate gender-specific factor and reference value A
    F = (pl.when(pl.col(gender_col) == "Female").then(1.012).otherwise(1))
    A = pl.when(pl.col(gender_col) == "Female").then(0.7).otherwise(0.9)

    # Compute K = eGFR / (142 * 0.9938^age * F)
    K = target_egfr / (142 * (0.9938 ** pl.col(age_col)) * F)

    # Upper branch (valid solution): Scr = A * K^(1/-1.2)
    estimated_scr = A * (K ** (1.0 / -1.2))

    return data.with_columns(
        estimated_scr.alias(
            f"estimated Creatinine (CKD-EPI eGFR={target_egfr})"
        )
    )


# endregion


# region Cockcroft-Gault
def CockcroftGault(data: pl.LazyFrame) -> pl.LazyFrame:
    """
    Calculate creatinine clearance using the Cockcroft-Gault equation.

    Arguments
    ---------
        data : pl.LazyFrame
            Input data frame containing gender, age, weight, and creatinine columns.

    Returns
    -------
        pl.LazyFrame
            Input data with additional column:
            - estimated GFR (Cockcroft-Gault): Creatinine clearance in mL/min
    """

    return data.with_columns(
        (
            (140 - pl.col(age_col))
            * pl.col(weight_col)
            * pl.when(pl.col(gender_col) == "Female").then(0.85).otherwise(1)
            / (72 * pl.col(creatinine_col))
        ).alias("estimated GFR (Cockcroft-Gault)")
    )


# region MDRD
def MDRD(data: pl.LazyFrame) -> pl.LazyFrame:
    """
    Calculate eGFR using the MDRD formula.

    Arguments
    ---------
        data : pl.LazyFrame
            Input data frame containing gender, age, creatinine, and ethnicity columns.

    Returns
    -------
        pl.LazyFrame
            Input data with additional column:
            - estimated GFR (MDRD): Estimated GFR in mL/min/1.73m²
    """

    return data.with_columns(
        (
            175
            * (pl.col(creatinine_col) ** -1.154)
            * (pl.col(age_col) ** -0.203)
            * pl.when(pl.col(ethnicity_col) == "Black or African American")
            .then(1.212)
            .otherwise(1)
            * pl.when(pl.col(gender_col) == "Female").then(0.742).otherwise(1)
        ).alias("estimated GFR (MDRD)")
    )


# region helper
def _improve_labs(timeseries_labs: pl.LazyFrame) -> pl.LazyFrame:
    return timeseries_labs.select(
        "Global ICU Stay ID",
        "Time Relative to Admission (seconds)",
        pl.when(
            pl.col("Creatinine").struct.field("system").eq("Serum or Plasma")
            | pl.col("Creatinine").struct.field("system").is_null()
        )
        .then(pl.col("Creatinine").struct.field("value"))
        .otherwise(None)
        .alias("Creatinine"),
    ).drop_nulls("Creatinine")


# region GFR
def ESTIMATED_GFR(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_labs: Optional[pl.LazyFrame] = None,
) -> pl.LazyFrame:
    """
    Calculate estimated GFR using multiple validated formulas.

    Computes eGFR using CKD-EPI (creatinine), Cockcroft-Gault, and MDRD formulas.
    All estimated GFR values have infinities and NaN replaced with None.

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information; must contain Global ICU Stay ID and
            Admission Weight (kg). Loaded automatically if None.
        timeseries_labs : pl.LazyFrame, optional
            Laboratory timeseries data. Loaded automatically if None.
            Creatinine values are filtered for "Serum or Plasma" system.

    Returns
    -------
        pl.LazyFrame
            Input data with additional columns:
            - Ideal Body Weight (Devine)
            - Adjusted Body Weight
            - estimated GFR (CKD-EPI Creatinine): eGFR in mL/min/1.73m²
            - estimated GFR (Cockcroft-Gault): Creatinine clearance in mL/min
            - estimated GFR (MDRD): eGFR in mL/min/1.73m²
    """
    # Load defaults if not provided
    if patient_information is None:
        patient_information = get_patient_information()
    if timeseries_labs is None:
        timeseries_labs = get_timeseries_labs()

    # Validate all required data is available
    required = {
        "patient_information": patient_information,
        "timeseries_labs": timeseries_labs,
    }

    missing = [name for name, data in required.items() if data is None]
    if missing:
        raise ValueError(
            f"Cannot compute ESTIMATED_GFR: Missing required datasets: {', '.join(missing)}. "
            f"Ensure they are configured in ~/.reprodICU/PATHS.yaml or provide them explicitly."
        )

    patient_information = _to_lazy(patient_information)
    timeseries_labs = _to_lazy(timeseries_labs)

    # Process and filter labs data
    timeseries_labs = _improve_labs(timeseries_labs)

    # Merge anthropometric data with processed labs
    data = patient_information.join(
        timeseries_labs, on="Global ICU Stay ID", how="inner"
    ).join(
        patient_information.pipe(ADJUSTED_BODY_WEIGHT),
        on="Global ICU Stay ID",
        how="left",
    )

    CKDGFR = (
        data.pipe(CKD_EPI)
        .select("Global ICU Stay ID", "estimated GFR (CKD-EPI Creatinine)")
        .with_columns(
            pl.col("estimated GFR (CKD-EPI Creatinine)")
            .replace([-np.inf, np.inf, np.nan], None)
            .cast(float)
        )
    )

    CockcroftGaultGFR = (
        data.pipe(CockcroftGault)
        .select("Global ICU Stay ID", "estimated GFR (Cockcroft-Gault)")
        .with_columns(
            pl.col("estimated GFR (Cockcroft-Gault)")
            .replace([-np.inf, np.inf, np.nan], None)
            .cast(float)
        )
    )

    MDRDGFR = (
        data.pipe(MDRD)
        .select("Global ICU Stay ID", "estimated GFR (MDRD)")
        .with_columns(
            pl.col("estimated GFR (MDRD)")
            .replace([-np.inf, np.inf, np.nan], None)
            .cast(float)
        )
    )

    # Return enriched data with all GFR estimates
    return (
        data.join(CKDGFR, on="Global ICU Stay ID", how="left")
        .join(CockcroftGaultGFR, on="Global ICU Stay ID", how="left")
        .join(MDRDGFR, on="Global ICU Stay ID", how="left")
    )


__all__ = ["ESTIMATED_GFR"]
