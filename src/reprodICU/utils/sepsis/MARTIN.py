# author: Finn Fassbender
# version: 2025-01-16

# Greg S. Martin, David M. Mannino, Stephanie Eaton, and Marc Moss.
# The epidemiology of sepsis in the united states from 1979 through 2000.
# N Engl J Med, 348(16):1546–1554, Apr 2003.
# doi: 10.1056/NEJMoa022139. URL http://dx.doi.org/10.1056/NEJMoa022139.

# This implementation is based only on the ICD-9-CM codes used in the paper.
# ICD-9-CPT codes are not used.

from typing import Optional

import polars as pl

from ..common import _to_lazy, _validate_required_data, get_diagnoses


# region Martin Sepsis
def MARTIN_SEPSIS(diagnoses: Optional[pl.LazyFrame] = None) -> pl.LazyFrame:
    """
    Calculate Martin sepsis classification from diagnosis codes.

    Implements the Martin et al. (2003) definition of sepsis, which requires:
    - Sepsis: ICD-9-CM codes indicating infection
    - Severe Sepsis: Sepsis + organ dysfunction

    Arguments
    ---------
        diagnoses : pl.LazyFrame, optional
            Diagnosis data. Loaded automatically if None.

    Returns
    -------
        pl.LazyFrame
            Martin sepsis classification with columns:
            - Global ICU Stay ID
            - Organ Dysfunction: Any organ system affected
            - Martin Sepsis: Presence of sepsis
            - Martin Severe Sepsis: Sepsis with organ dysfunction
    """
    if diagnoses is None:
        diagnoses = get_diagnoses()

    # Validate data is available
    required = {"diagnoses": diagnoses}
    _validate_required_data("Martin sepsis", required)

    diagnoses = _to_lazy(diagnoses)

    return (
        diagnoses.with_columns(
            # sepsis
            (
                # septicemia
                (pl.col("Diagnosis ICD-9 Code").str.slice(0, 3) == "038")
                # septicemic, bacteremia, disseminated fungal infection, disseminated candida infection
                | pl.col("Diagnosis ICD-9 Code")
                .str.slice(0, 4)
                .is_in(["0202", "7907", "1179", "1125"])
                # disseminated fungal endocarditis
                | (pl.col("Diagnosis ICD-9 Code").str.slice(0, 5) == "11281")
            ).alias("sepsis"),
            # respiratory
            (
                (pl.col("Diagnosis ICD-9 Code").str.slice(0, 4) == "7991")
                | pl.col("Diagnosis ICD-9 Code")
                .str.slice(0, 5)
                .is_in(["51881", "51882", "51885", "78609"])
            ).alias("respiratory"),
            # cardiovascular
            (
                pl.col("Diagnosis ICD-9 Code")
                .str.slice(0, 4)
                .is_in(["4580", "7855", "4580", "4588", "4589", "7963"])
                | pl.col("Diagnosis ICD-9 Code")
                .str.slice(0, 5)
                .is_in(["78551", "78559"])
            ).alias("cardiovascular"),
            # renal
            pl.col("Diagnosis ICD-9 Code")
            .str.slice(0, 3)
            .is_in(["584", "580", "585"])
            .alias("renal"),
            # hepatic
            (
                (pl.col("Diagnosis ICD-9 Code").str.slice(0, 3) == "570")
                | pl.col("Diagnosis ICD-9 Code")
                .str.slice(0, 4)
                .is_in(["5722", "5733"])
            ).alias("hepatic"),
            # hematologic
            pl.col("Diagnosis ICD-9 Code")
            .str.slice(0, 4)
            .is_in(["2862", "2866", "2869", "2873", "2874", "2875"])
            .alias("hematologic"),
            # metabolic
            (pl.col("Diagnosis ICD-9 Code").str.slice(0, 4) == "2762").alias(
                "metabolic"
            ),
            # neurologic
            (
                (pl.col("Diagnosis ICD-9 Code").str.slice(0, 3) == "293")
                | pl.col("Diagnosis ICD-9 Code")
                .str.slice(0, 4)
                .is_in(["3481", "3483"])
                | pl.col("Diagnosis ICD-9 Code")
                .str.slice(0, 5)
                .is_in(["78001", "78009"])
            ).alias("neurologic"),
        )
        .group_by("Global ICU Stay ID")
        .agg(pl.all().sum().cast(bool))
        .with_columns(
            pl.any_horizontal(
                pl.col("respiratory"),
                pl.col("cardiovascular"),
                pl.col("renal"),
                pl.col("hepatic"),
                pl.col("hematologic"),
                pl.col("metabolic"),
                pl.col("neurologic"),
            ).alias("Organ Dysfunction")
        )
        .with_columns(
            pl.col("sepsis").alias("Martin Sepsis"),
            pl.all_horizontal(
                pl.col("sepsis"),
                pl.col("Organ Dysfunction"),
            ).alias("Martin Severe Sepsis"),
        )
        .select(
            "Global ICU Stay ID",
            "Organ Dysfunction",
            "Martin Sepsis",
            "Martin Severe Sepsis",
        )
    )


# endregion

__all__ = ["MARTIN_SEPSIS"]
