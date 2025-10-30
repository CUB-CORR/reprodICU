"""
Comorbidity Indices: calculate comorbidity scores using validated algorithms.

This module implements several comorbidity indices based on diagnosis data:
- Elixhauser Comorbidity Index (Quan implementation)
- Charlson Comorbidity Index (Quan implementation)
- Gagne Comorbidity Index

Output depends on the function:
- Comorbidity scores (default): Numeric scores representing disease burden
- Comorbidity categories (return_categories=True): Categorical risk levels

The module automatically loads diagnosis and patient information data from
the reprodICU database if not provided explicitly. All functions support
both ICD-9 and ICD-10 diagnosis codes.
"""

from typing import Optional

from pycomorb import comorbidity
import polars as pl
from .common import _to_lazy, get_diagnoses, get_patient_information


# region helpers
def _load_and_process_diagnoses(
    diagnoses: Optional[pl.LazyFrame] = None,
    patient_information: Optional[pl.LazyFrame] = None,
) -> pl.LazyFrame:
    """
    Load and process diagnosis data with patient information.

    Arguments
    ---------
        diagnoses : pl.LazyFrame, optional
            Diagnosis data. Loaded automatically if None.
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information. Loaded automatically if None.

    Returns
    -------
        pl.LazyFrame
            Processed diagnosis data with:
            - Global ICU Stay ID
            - Diagnosis ICD Code (unified from ICD-9/10)
            - Diagnosis ICD Code Version (source)
            - Admission Age (years)

    Raises
    ------
        ValueError
            If required datasets cannot be loaded.
    """
    if diagnoses is None:
        diagnoses = get_diagnoses()
    if patient_information is None:
        patient_information = get_patient_information()

    # Validate data is available
    required = {
        "diagnoses": diagnoses,
        "patient_information": patient_information,
    }

    missing = [name for name, data in required.items() if data is None]
    if missing:
        raise ValueError(
            f"Cannot load comorbidity data: Missing required datasets: {', '.join(missing)}. "
            f"Ensure they are configured in ~/.reprodICU/PATHS.yaml or provide them explicitly."
        )

    diagnoses = _to_lazy(diagnoses)
    patient_information = _to_lazy(patient_information)

    return (
        diagnoses.join(patient_information, on="Global ICU Stay ID", how="left")
        .with_columns(
            pl.when(pl.col("Diagnosis ICD Code Version (source)") == "ICD-9")
            .then(pl.col("Diagnosis ICD-9 Code"))
            .otherwise(pl.col("Diagnosis ICD-10 Code"))
            .alias("Diagnosis ICD Code")
        )
        .select(
            "Global ICU Stay ID",
            "Diagnosis ICD Code",
            "Diagnosis ICD Code Version (source)",
            "Admission Age (years)",
        )
        .collect()
    )


# region Comorbidity Scores
def ELIXHAUSER(
    diagnoses: Optional[pl.LazyFrame] = None,
    patient_information: Optional[pl.LazyFrame] = None,
    return_categories: bool = False,
):
    """
    Calculate Elixhauser comorbidity index using Quan implementation with van Walraven weights.

    Arguments
    ---------
        diagnoses : pl.LazyFrame, optional
            Diagnosis data. Loaded automatically if None.
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information. Loaded automatically if None.
        return_categories : bool, optional
            If True, return comorbidity categories; otherwise return index scores. Default is False.

    Returns
    -------
        pd.DataFrame
            Comorbidity scores or categories with Global ICU Stay ID as index.
    """
    return comorbidity(
        score="elixhauser",
        implementation="quan",
        df=_load_and_process_diagnoses(diagnoses, patient_information),
        id_col="Global ICU Stay ID",
        code_col="Diagnosis ICD Code",
        icd_version="icd9_10",
        icd_version_col="Diagnosis ICD Code Version (source)",
        return_categories=return_categories,
    )


def CHARLSON(
    diagnoses: Optional[pl.LazyFrame] = None,
    patient_information: Optional[pl.LazyFrame] = None,
    return_categories: bool = False,
):
    """
    Calculate Charlson comorbidity index using Quan implementation.

    Arguments
    ---------
        diagnoses : pl.LazyFrame, optional
            Diagnosis data. Loaded automatically if None.
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information. Loaded automatically if None.
        return_categories : bool, optional
            If True, return comorbidity categories; otherwise return index scores. Default is False.

    Returns
    -------
        pd.DataFrame
            Comorbidity scores or categories with Global ICU Stay ID as index.
    """
    return comorbidity(
        score="charlson",
        implementation="quan",
        df=_load_and_process_diagnoses(diagnoses, patient_information),
        id_col="Global ICU Stay ID",
        code_col="Diagnosis ICD Code",
        age_col="Admission Age (years)",
        icd_version="icd9_10",
        icd_version_col="Diagnosis ICD Code Version (source)",
        return_categories=return_categories,
    ).drop("Charlson Comorbidity Score")


def GAGNE(
    diagnoses: Optional[pl.LazyFrame] = None,
    patient_information: Optional[pl.LazyFrame] = None,
    return_categories: bool = False,
):
    """
    Calculate Gagne comorbidity index (combined implementation).

    Arguments
    ---------
        diagnoses : pl.LazyFrame, optional
            Diagnosis data. Loaded automatically if None.
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information. Loaded automatically if None.
        return_categories : bool, optional
            If True, return comorbidity categories; otherwise return index scores. Default is False.

    Returns
    -------
        pd.DataFrame
            Comorbidity scores or categories with Global ICU Stay ID as index.
    """
    return comorbidity(
        score="combined",
        df=_load_and_process_diagnoses(diagnoses, patient_information),
        id_col="Global ICU Stay ID",
        code_col="Diagnosis ICD Code",
        icd_version="icd9_10",
        icd_version_col="Diagnosis ICD Code Version (source)",
        return_categories=return_categories,
    )


__all__ = ["ELIXHAUSER", "CHARLSON", "GAGNE"]
