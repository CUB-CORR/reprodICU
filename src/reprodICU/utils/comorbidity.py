"""
Comorbidity Indices: calculate comorbidity scores using validated algorithms.

This module implements several comorbidity indices based on diagnosis data:
- Elixhauser Comorbidity Index (Quan implementation)
- Charlson Comorbidity Index (Quan implementation)
- Gagne Comorbidity Index

Output depends on the function:
- Comorbidity scores (default): Numeric scores representing disease burden
- Comorbidity categories (return_categories=True): One indicator column per comorbidity category

The module automatically loads diagnosis and patient information data from
the reprodICU database if not provided explicitly. All functions support
both ICD-9 and ICD-10 diagnosis codes. The indices are computed with pycomorb.

SOURCES
-------
Elixhauser (Quan code mapping, van Walraven weights):
- Elixhauser A, Steiner C, Harris DR, Coffey RM.
  Comorbidity measures for use with administrative data.
  Med Care. 1998 Jan;36(1):8-27.
  doi: 10.1097/00005650-199801000-00004. PMID: 9431328.
- Quan H, Sundararajan V, Halfon P, Fong A, Burnand B, Luthi JC, Saunders LD, Beck CA, Feasby TE, Ghali WA.
  Coding algorithms for defining comorbidities in ICD-9-CM and ICD-10 administrative data.
  Med Care. 2005 Nov;43(11):1130-9.
  doi: 10.1097/01.mlr.0000182534.19832.83. PMID: 16224307.
- van Walraven C, Austin PC, Jennings A, Quan H, Forster AJ.
  A modification of the Elixhauser comorbidity measures into a point system for hospital death using administrative data.
  Med Care. 2009 Jun;47(6):626-33.
  doi: 10.1097/MLR.0b013e31819432e5. PMID: 19433995.

Charlson (Quan code mapping):
- Charlson ME, Pompei P, Ales KL, MacKenzie CR.
  A new method of classifying prognostic comorbidity in longitudinal studies: development and validation.
  J Chronic Dis. 1987;40(5):373-83.
  doi: 10.1016/0021-9681(87)90171-8. PMID: 3558716.
- Quan H, Sundararajan V, Halfon P, Fong A, Burnand B, Luthi JC, Saunders LD, Beck CA, Feasby TE, Ghali WA.
  Coding algorithms for defining comorbidities in ICD-9-CM and ICD-10 administrative data.
  Med Care. 2005 Nov;43(11):1130-9.
  doi: 10.1097/01.mlr.0000182534.19832.83. PMID: 16224307.

Gagne (combined comorbidity score):
- Gagne JJ, Glynn RJ, Avorn J, Levin R, Schneeweiss S.
  A combined comorbidity score predicted mortality in elderly patients better than existing scores.
  J Clin Epidemiol. 2011 Jul;64(7):749-59.
  doi: 10.1016/j.jclinepi.2010.10.004. PMID: 21208778; PMCID: PMC3100405.
  ICD-9-CM codes combine the Charlson conditions as mapped by Romano et al. 1993
  and the Elixhauser conditions as mapped by Quan et al. 2005 (see above).
- Romano PS, Roos LL, Jollis JG.
  Adapting a clinical comorbidity index for use with ICD-9-CM administrative data: differing perspectives.
  J Clin Epidemiol. 1993 Oct;46(10):1075-9; discussion 1081-90.
  doi: 10.1016/0895-4356(93)90103-8. PMID: 8410092.
- Sun JW, Rogers JR, Her Q, Welch EC, Panozzo CA, Toh S, Gagne JJ.
  Adaptation and Validation of the Combined Comorbidity Score for ICD-10-CM.
  Med Care. 2017 Dec;55(12):1046-1051.
  doi: 10.1097/MLR.0000000000000824. PMID: 29087983.
"""

from typing import Optional

import polars as pl
from pycomorb import comorbidity

from .base import (
    _to_lazy,
)
from .common import (
    _validate_required_data,
    get_diagnoses,
    get_patient_information,
)


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
    _validate_required_data("comorbidity data", required)

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
) -> pl.DataFrame:
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
        pl.DataFrame
            One row per Global ICU Stay ID with the score column "Elixhauser Comorbidity Index",
            or, with return_categories=True, one boolean column per comorbidity category.
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
) -> pl.DataFrame:
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
        pl.DataFrame
            One row per Global ICU Stay ID with the score column "Charlson Age-Comorbidity Score",
            or, with return_categories=True, one boolean column per comorbidity category.
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
    ).drop("Charlson Comorbidity Index")  # keep the age-adjusted "Charlson Age-Comorbidity Score"


def GAGNE(
    diagnoses: Optional[pl.LazyFrame] = None,
    patient_information: Optional[pl.LazyFrame] = None,
    return_categories: bool = False,
) -> pl.DataFrame:
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
        pl.DataFrame
            One row per Global ICU Stay ID with the score column "Combined Comorbidity Score",
            or, with return_categories=True, one boolean column per comorbidity category.
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
