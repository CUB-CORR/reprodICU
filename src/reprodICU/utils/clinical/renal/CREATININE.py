"""
Estimated Creatinine: reversing glomerular filtration rate (GFR) formulas to estimate serum creatinine.

The module implements the following reverse formulas:
- 2021 CKD-EPI Creatinine reverse formula
- MDRD GFR reverse formula

Output columns depend on the function:
- estimated Creatinine (CKD-EPI eGFR={target_egfr}): Serum creatinine in mg/dL that would produce the target eGFR using CKD-EPI formula
- estimated Creatinine (MDRD eGFR={target_egfr}): Serum creatinine in mg/dL that would produce the target eGFR using MDRD formula

Input values required:
- Age: patient age in years
- Gender: patient gender (Male / Female)
- Ethnicity: patient ethnicity (required for MDRD reverse)

| Formula            | Parameters              |
|--------------------|-------------------------|
| reverse CKD-EPI    | Age, Gender             |
| reverse MDRD       | Age, Gender, Ethnicity  |

CLINICAL NOTES
--------------
The reverse formulas are useful for establishing baseline creatinine assumptions (e.g., assuming normal
kidney function at eGFR=75 mL/min/1.73m²) or for sensitivity analyses exploring how changes in
creatinine would affect GFR values under different disease scenarios.

SOURCES
-------
- Reverse MDRD formula derived from:
    Levey AS, Coresh J, Greene T, Stevens LA, Zhang YL, Hendriksen S, Kusek JW, Van Lente F; Chronic Kidney Disease Epidemiology Collaboration.
    Using standardized serum creatinine values in the modification of diet in renal disease study equation for estimating glomerular filtration rate.
    Ann Intern Med. 2006 Aug 15;145(4):247-54.
    doi: 10.7326/0003-4819-145-4-200608150-00004. PMID: 16908915.
- Reverse CKD-EPI 2021 Creatinine formula derived from:
    Inker LA, Eneanya ND, Coresh J, Tighiouart H, Wang D, Sang Y, Crews DC, Doria A, Estrella MM, Froissart M, Grams ME, Greene T, Grubb A, Gudnason V, Gutiérrez OM, Kalil R, Karger AB, Mauer M, Navis G, Nelson RG, Poggio ED, Rodby R, Rossing P, Rule AD, Selvin E, Seegmiller JC, Shlipak MG, Torres VE, Yang W, Ballew SH, Couture SJ, Powe NR, Levey AS; Chronic Kidney Disease Epidemiology Collaboration.
    New Creatinine- and Cystatin C-Based Equations to Estimate GFR without Race.
    N Engl J Med. 2021 Nov 4;385(19):1737-1749.
    doi: 10.1056/NEJMoa2102953. Epub 2021 Sep 23. PMID: 34554658; PMCID: PMC8822996.
"""

from typing import Optional

import numpy as np
import polars as pl

from ...common import _to_lazy, get_patient_information

age_col = "Admission Age (years)"
gender_col = "Gender"
ethnicity_col = "Ethnicity"
creatinine_col = "Creatinine"


# region reverse CKD-EPI
def reverse_CKD_EPI(
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
    F = pl.when(pl.col(gender_col) == "Female").then(1.012).otherwise(1)
    A = pl.when(pl.col(gender_col) == "Female").then(0.7).otherwise(0.9)

    # Compute K = eGFR / (142 * 0.9938^age * F)
    K = target_egfr / (142 * (0.9938 ** pl.col(age_col)) * F)

    # Upper branch (valid solution): Scr = A * K^(1/-1.2)
    estimated_scr = A * (K ** (1.0 / -1.2))

    return data.with_columns(
        estimated_scr.alias(f"estimated Creatinine (CKD-EPI eGFR={target_egfr})")
    ) # fmt: skip


# endregion


# region reverse MDRD
def reverse_MDRD(data: pl.LazyFrame, target_egfr: float = 75) -> pl.LazyFrame:
    """
    Estimate creatinine from target eGFR using the reverse MDRD formula.

    Reverses the MDRD GFR equation:
    eGFR = 175 * (SCr ^ -1.154) * (age ^ -0.203) * gender_factor * ethnicity_factor

    Algebraically rearranging for serum creatinine:
    SCr = (175 * (age ^ -0.203) * gender_factor * ethnicity_factor / target_eGFR) ^ (1 / -1.154)

    Arguments
    ---------
        data : pl.LazyFrame
            Input data frame containing gender, age, and ethnicity columns.
        target_egfr : float, optional
            Target eGFR value to estimate creatinine for. Default: 75 mL/min/1.73m².

    Returns
    -------
        pl.LazyFrame
            Input data with additional columns:
            - estimated Creatinine (MDRD eGFR={target_egfr}): Estimated serum creatinine in mg/dL
    """
    # Calculate gender and ethnicity factors matching forward MDRD formula
    G = pl.when(pl.col(gender_col) == "Female").then(0.742).otherwise(1)
    E = pl.when(pl.col(ethnicity_col).eq_missing("Black or African American")).then(1.212).otherwise(1) # fmt: skip

    # Compute inverse MDRD: SCr = (175 * (age^-0.203) * gender_factor * ethnicity_factor / target_egfr) ^ (1/-1.154)
    N = 175 * (pl.col(age_col) ** -0.203) * G * E
    estimated_scr = (N / target_egfr) ** (1.0 / -1.154)

    return data.with_columns(
        estimated_scr.alias(f"estimated Creatinine (MDRD eGFR={target_egfr})")
    )


# endregion


# region ESTIMATED_CREATININE
def ESTIMATED_CREATININE(
    patient_information: Optional[pl.LazyFrame] = None,
    target_egfr: float = 75,
) -> pl.LazyFrame:
    """
    Calculate estimated creatinine from target eGFR using multiple validated reverse formulas.

    Computes estimated serum creatinine levels that would correspond to a target eGFR value
    using both CKD-EPI and MDRD reverse formulas. All estimated creatinine values have
    infinities and NaN replaced with None.

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information; must contain Global ICU Stay ID,
            Admission Age (years), Gender, and Ethnicity. Loaded automatically if None.
        target_egfr : float, optional
            Target eGFR value (mL/min/1.73m²) to estimate creatinine for. Default: 75.

    Returns
    -------
        pl.LazyFrame
            Input data with additional columns:
            - estimated Creatinine (CKD-EPI eGFR={target_egfr}): Serum creatinine in mg/dL
            - estimated Creatinine (MDRD eGFR={target_egfr}): Serum creatinine in mg/dL
    """
    # Load defaults if not provided
    if patient_information is None:
        patient_information = get_patient_information()

    # Validate all required data is available
    if patient_information is None:
        raise ValueError(
            "Cannot compute ESTIMATED_CREATININE: Missing patient_information. "
            "Ensure it is configured in ~/.reprodICU/PATHS.yaml or provide it explicitly."
        )

    patient_information = _to_lazy(patient_information)

    ckd_epi_reverse = (
        patient_information.pipe(reverse_CKD_EPI, target_egfr=target_egfr)
        .select(
            "Global ICU Stay ID",
            f"estimated Creatinine (CKD-EPI eGFR={target_egfr})",
        )
        .with_columns(
            pl.col(f"estimated Creatinine (CKD-EPI eGFR={target_egfr})")
            .replace([-np.inf, np.inf, np.nan], None)
            .cast(float)
        )
    )

    mdrd_reverse = (
        patient_information.pipe(reverse_MDRD, target_egfr=target_egfr)
        .select(
            "Global ICU Stay ID",
            f"estimated Creatinine (MDRD eGFR={target_egfr})",
        )
        .with_columns(
            pl.col(f"estimated Creatinine (MDRD eGFR={target_egfr})")
            .replace([-np.inf, np.inf, np.nan], None)
            .cast(float)
        )
    )

    # Return enriched data with all estimated creatinine values
    return patient_information.join(
        ckd_epi_reverse, on="Global ICU Stay ID", how="left"
    ).join(mdrd_reverse, on="Global ICU Stay ID", how="left")


# endregion


__all__ = ["ESTIMATED_CREATININE"]
