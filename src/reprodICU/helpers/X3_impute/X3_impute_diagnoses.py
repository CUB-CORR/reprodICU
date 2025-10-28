# Author: Finn Fassbender
# Last modified: 2024-09-11

# Description: This script imputes the data to remove missing values.
# It is available as a module for piping in the main script.
# It can be called with command line arguments to specify the source datasets to be imputed. ! NOT IMPLEMENTED YET !

import polars as pl

from helpers.helper import GlobalVars


class DiagnosesImputer(GlobalVars):
    def __init__(self, paths, patient_info_path: str) -> None:
        super().__init__(paths)
        self.patient_info_path = patient_info_path

    def impute_diagnoses(self, data) -> pl.LazyFrame:
        """
        Imputes missing ICD codes in the diagnoses data.
        -> maps ICD9 codes to ICD10 codes and vice versa (for inclusion / exclusion criteria down the line)
        """

        IDs = pl.scan_parquet(self.patient_info_path).select(
            self.global_hospital_stay_id_col,
            self.global_icu_stay_id_col,
        )

        ICD9_TO_ICD10_MAPPING = dict(
            zip(
                self.ICD9_TO_ICD10_DIAGS["icd9"],
                self.ICD9_TO_ICD10_DIAGS["icd10"],
            )
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
                # Impute missing ICD9 codes
                pl.when(pl.col(self.diagnosis_icd_version_col) == 9)
                .then(pl.col(self.diagnosis_icd_code_col))
                .otherwise(
                    pl.col(self.diagnosis_icd_code_col).replace(
                        ICD10_TO_ICD9_MAPPING
                    )
                )
                .alias(self.diagnosis_icd9_code_col),
                # Impute missing ICD10 codes
                pl.when(pl.col(self.diagnosis_icd_version_col) == 10)
                .then(pl.col(self.diagnosis_icd_code_col))
                .otherwise(
                    pl.col(self.diagnosis_icd_code_col).replace(
                        ICD9_TO_ICD10_MAPPING
                    )
                )
                .replace("NoDx", None)
                .alias(self.diagnosis_icd10_code_col),
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
                self.diagnosis_discharge_col,
                # self.diagnosis_description_col, # Description is not used in the grouping, as it can be inconsistent
            )
            # Add column for ICD version of original diagnosis code
            .agg(
                pl.implode(self.diagnosis_icd_version_col),
                pl.first(self.diagnosis_description_col),
            )
            .with_columns(
                pl.when(
                    pl.col(self.diagnosis_icd_version_col)
                    .list.unique()
                    .list.len()
                    > 1
                )
                .then(pl.lit("ICD-9 / ICD-10"))
                .otherwise(
                    pl.concat_str(
                        pl.lit("ICD-"),
                        pl.col(self.diagnosis_icd_version_col)
                        .list.first()
                        .cast(pl.String),
                    )
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
                self.diagnosis_discharge_col,
                self.diagnosis_description_col,
            )
            .lazy()
        )


if __name__ == "__main__":
    raise NotImplementedError(
        "This script is not yet implemented as a command line tool."
    )
