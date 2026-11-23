# Author: Finn Fassbender
# Last modified: 2025-10-30

# Description: This script imputes the data to remove missing values.
# It is available as a module for piping in the main script.
# It can be called with command line arguments to specify the source datasets to be imputed. ! NOT IMPLEMENTED YET !

import polars as pl

from ..helper import GlobalVars


class DiagnosesMapper(GlobalVars):
    def __init__(self, paths, patient_info_path: str) -> None:
        super().__init__(paths)
        self.patient_info_path = patient_info_path

    def map_diagnoses(self, data) -> pl.LazyFrame:
        """
        Map diagnoses between ICD-9 and ICD-10 codes.

        Steps:
            1. Load patient IDs to fill missing hospital/ICU stay associations.
            2. Create bidirectional ICD-9 <-> ICD-10 mapping dictionaries.
            3. Assign missing hospital stay IDs from ICU stay IDs where available.
            4. Map all diagnoses to both ICD-9 and ICD-10 codes.
            5. Deduplicate and sort results.

        Returns:
            pl.LazyFrame: Diagnoses with both ICD-9 and ICD-10 codes added.
        """

        IDs = pl.scan_parquet(self.patient_info_path).select(
            self.global_hospital_stay_id_col,
            self.global_icu_stay_id_col,
        )

        # An ICD-9 code can have several ICD-10 targets; the mapping lists the primary one first
        # (4019 -> I10, then I169), so take the first and not the last.
        ICD9_TO_ICD10_MAPPING = dict(
            self.ICD9_TO_ICD10_DIAGS.unique("icd9", keep="first", maintain_order=True)
            .select("icd9", "icd10")
            .iter_rows()
        )
        ICD10_TO_ICD9_MAPPING = dict(
            zip(
                self.ICD10_TO_ICD9_DIAGS["icd10"],
                self.ICD10_TO_ICD9_DIAGS["icd9"],
            )
        )

        return (
            pl.concat(
                [
                    data.filter(pl.col(self.global_icu_stay_id_col).is_null())
                    .drop(self.global_icu_stay_id_col)
                    .join(
                        IDs,
                        on=self.global_hospital_stay_id_col,
                        how="left",
                        coalesce=True,
                    ),
                    data.filter(
                        pl.col(self.global_icu_stay_id_col).is_not_null()
                    ),
                ],
                how="diagonal_relaxed",
            )
            .collect()
            .with_columns(
                # Anchor on the code the source gave: a pair gives both, otherwise the code
                # of the source version.
                pl.coalesce(
                    pl.col(self.diagnosis_icd9_code_col),
                    pl.when(pl.col(self.diagnosis_icd_version_col) == 9).then(
                        pl.col(self.diagnosis_icd_code_col)
                    ),
                ).alias(self.diagnosis_icd9_code_col),
                pl.coalesce(
                    pl.col(self.diagnosis_icd10_code_col),
                    pl.when(pl.col(self.diagnosis_icd_version_col) == 10).then(
                        pl.col(self.diagnosis_icd_code_col)
                    ),
                ).alias(self.diagnosis_icd10_code_col),
                # Label of the source version, "ICD-9 / ICD-10" for a pair
                pl.when(
                    pl.col(self.diagnosis_icd9_code_col).is_not_null()
                    & pl.col(self.diagnosis_icd10_code_col).is_not_null()
                )
                .then(pl.lit("ICD-9 / ICD-10"))
                .otherwise(
                    pl.concat_str(
                        pl.lit("ICD-"),
                        pl.col(self.diagnosis_icd_version_col).cast(pl.String),
                    )
                )
                .alias(self.diagnosis_icd_source_version_col),
            )
            .with_columns(
                # Impute only the codes still missing, from the other version
                pl.col(self.diagnosis_icd9_code_col)
                .fill_null(
                    pl.col(self.diagnosis_icd10_code_col).replace_strict(
                        ICD10_TO_ICD9_MAPPING, default=None
                    )
                ),
                pl.col(self.diagnosis_icd10_code_col)
                .fill_null(
                    pl.col(self.diagnosis_icd9_code_col).replace_strict(
                        ICD9_TO_ICD10_MAPPING, default=None
                    )
                )
                .replace("NoDx", None),
            )
            .group_by(
                self.global_person_id_col,
                self.global_hospital_stay_id_col,
                self.global_icu_stay_id_col,
                self.diagnosis_icd9_code_col,
                self.diagnosis_icd10_code_col,
                self.diagnosis_start_col,
                self.diagnosis_end_col,
                self.diagnosis_priority_col,
                self.diagnosis_admission_col,
                self.diagnosis_discharge_col,
                # self.diagnosis_description_col, # Description is not used in the grouping, as it can be inconsistent
            )
            # Rows of one diagnosis that come from both ICD versions get the label "ICD-9 / ICD-10"
            .agg(
                pl.implode(self.diagnosis_icd_source_version_col),
                # Prefer the ICD-10 description (row order is not stable)
                pl.col(self.diagnosis_description_col)
                .sort_by(
                    self.diagnosis_icd_version_col,
                    descending=True,
                    nulls_last=True,
                )
                .drop_nulls()
                .first(),
            )
            .with_columns(
                pl.when(
                    pl.col(self.diagnosis_icd_source_version_col)
                    .list.unique()
                    .list.len()
                    > 1
                )
                .then(pl.lit("ICD-9 / ICD-10"))
                .otherwise(
                    pl.col(self.diagnosis_icd_source_version_col).list.first()
                )
                .alias(self.diagnosis_icd_source_version_col)
            )
            .unique()
            # Keep records that are not duplicates OR keep only active on discharge if duplicates exist
            .with_columns(
                # Count duplicates for each group
                pl.len()
                .over(
                    self.global_person_id_col,
                    self.global_hospital_stay_id_col,
                    self.global_icu_stay_id_col,
                    self.diagnosis_icd9_code_col,
                    self.diagnosis_icd10_code_col,
                    self.diagnosis_start_col,
                )
                .alias("Diagnosis Group Count")
            )
            .filter(
                (pl.col("Diagnosis Group Count") == 1)
                | (
                    (pl.col("Diagnosis Group Count") > 1)
                    & pl.col(self.diagnosis_discharge_col)
                )
            )
            .drop("Diagnosis Group Count")
            .sort(self.global_icu_stay_id_col, self.diagnosis_start_col)
            .select(
                self.global_person_id_col,
                self.global_hospital_stay_id_col,
                self.global_icu_stay_id_col,
                self.diagnosis_icd_source_version_col,
                self.diagnosis_icd9_code_col,
                self.diagnosis_icd10_code_col,
                self.diagnosis_start_col,
                self.diagnosis_end_col,
                self.diagnosis_priority_col,
                self.diagnosis_admission_col,
                self.diagnosis_discharge_col,
                self.diagnosis_description_col,
            )
            .lazy()
        )
