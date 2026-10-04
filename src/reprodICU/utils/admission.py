"""
Admission Categories: flag common admission groups from diagnosis and specialty data.

Output columns per row:
- Global ICU Stay ID
- Trauma Admission: Binary flag for trauma admissions
- Cardiac Surgery Admission: Binary flag for post-cardiac-surgery admissions
- Diabetic Emergency Admission: Binary flag for diabetic ketoacidosis / hyperosmolar admissions

Admission categories are determined from a combination of:
- Admission Diagnosis (APACHE)
- Primary (Admission) Diagnosis (ICD)
- Primary diagnosis description (priority 1 of the diagnoses table; diabetic emergency only)
- Specialty

Missing information counts as False. Admission Origin and Unit Type are not used, since they
carry no or mixed information (e.g. the MIMIC trauma ICU also admits neoplasms and strokes).
"""

from typing import Optional

import polars as pl

from .common import _to_lazy, get_diagnoses, get_patient_information

__all__ = ["COMMON_ADMISSION_CATEGORIES"]


def COMMON_ADMISSION_CATEGORIES(
    patient_information: Optional[pl.LazyFrame] = None,
    diagnoses: Optional[pl.LazyFrame] = None,
) -> pl.LazyFrame:
    """
    Flag common admission categories based on admission diagnosis and specialty.

    Computes admission flags for:
    - Trauma (APACHE trauma diagnoses, ICD-10 injuries / accidents, trauma surgery specialty)
    - Cardiac surgery (post-operative CABG / valve diagnoses, cardiac surgery specialty)
    - Diabetic emergency (diabetic ketoacidosis, hyperosmolar hyperglycaemic state)

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information. Loaded automatically if None. Must contain:
            - Global ICU Stay ID
            - Admission Diagnosis (APACHE)
            - Primary (Admission) Diagnosis (ICD)
            - Specialty
        diagnoses : pl.LazyFrame, optional
            Diagnosis data (priority 1 descriptions identify diabetic emergencies by ICD).
            Loaded automatically if None; optional.

    Returns
    -------
        pl.LazyFrame
            One row per stay with:
            - Global ICU Stay ID
            - Trauma Admission: Boolean flag
            - Cardiac Surgery Admission: Boolean flag
            - Diabetic Emergency Admission: Boolean flag
    """
    # Load defaults if not provided
    if patient_information is None:
        patient_information = get_patient_information()
    if diagnoses is None:
        diagnoses = get_diagnoses()

    # Validate data is available
    if patient_information is None:
        raise ValueError(
            "Cannot compute COMMON_ADMISSION_CATEGORIES: Missing patient_information dataset. "
            "Ensure it is configured in ~/.reprodICU/PATHS.yaml or provide it explicitly."
        )

    # Ensure lazy
    patient_information = _to_lazy(patient_information)
    # without diagnoses, the ICD route to diabetic emergencies finds nothing
    diagnoses = (
        _to_lazy(diagnoses)
        if diagnoses is not None
        else pl.LazyFrame(schema={"Global ICU Stay ID": pl.String, "Diagnosis Priority": pl.Int64, "Diagnosis Description": pl.String})
    )

    # Primary diagnosis (priority 1) is a diabetic emergency per ICD description
    diabetic_emergency_icd = (
        diagnoses.filter(
            pl.col("Diagnosis Priority") == 1,
            pl.col("Diagnosis Description").str.contains(r"(?i)ketoacidosis|hyperosmolarity"),
        )
        .select("Global ICU Stay ID")
        .unique()
        .with_columns(pl.lit(True).alias("Diabetic Emergency (ICD)"))
    )

    return patient_information.join(diabetic_emergency_icd, on="Global ICU Stay ID", how="left").select(
        "Global ICU Stay ID",
        pl.any_horizontal(
            pl.col("Admission Diagnosis (APACHE)").str.contains(r"^(Non|Post)-Operative - Trauma\b"),
            pl.col("Primary (Admission) Diagnosis (ICD)").str.contains(r"^XIX - (Injur|Certain early complications of trauma)|^XX - Accidents"),
            pl.col("Specialty") == "Trauma Surgery",
        )
        .fill_null(False)
        .alias("Trauma Admission"),
        pl.any_horizontal(
            pl.col("Admission Diagnosis (APACHE)").str.contains(r"^Post-Operative - Cardiovascular - (CABG|.*valve)"),
            pl.col("Specialty") == "Cardiac Surgery",
        )
        .fill_null(False)
        .alias("Cardiac Surgery Admission"),
        pl.any_horizontal(
            pl.col("Admission Diagnosis (APACHE)").str.contains(r"(?i)ketoacidosis|hyperosmolar"),
            pl.col("Diabetic Emergency (ICD)"),
        )
        .fill_null(False)
        .alias("Diabetic Emergency Admission"),
    )
