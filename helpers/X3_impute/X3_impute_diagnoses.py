# Author: Finn Fassbender
# Last modified: 2024-09-11

# Description: This script imputes the data to remove missing values.
# It is available as a module for piping in the main script.
# It can be called with command line arguments to specify the source datasets to be imputed. ! NOT IMPLEMENTED YET !

import argparse
import polars as pl

from helpers.helper import GlobalVars


class DiagnosesImputer(GlobalVars):
    def __init__(self, paths) -> None:
        super().__init__(paths)
        pass

    def impute_diagnoses(self, data) -> pl.LazyFrame:
        """
        Imputes missing ICD codes in the diagnoses data.
        -> maps ICD9 codes to ICD10 codes and vice versa (for inclusion / exclusion criteria down the line)
        """

        ICD9_TO_ICD10_MAPPING = dict(
            zip(self.ICD9_TO_ICD10_DIAGS["icd9"], self.ICD9_TO_ICD10_DIAGS["icd10"])
        )
        ICD10_TO_ICD9_MAPPING = dict(
            zip(self.ICD10_TO_ICD9_DIAGS["icd10"], self.ICD10_TO_ICD9_DIAGS["icd9"])
        )

        return (
            data.with_columns(
                # Impute missing ICD9 codes
                pl.when(pl.col(self.diagnosis_icd_version_col) == 9)
                .then(pl.col(self.diagnosis_icd_code_col))
                .otherwise(pl.col(self.diagnosis_icd_code_col).replace(ICD10_TO_ICD9_MAPPING))
                .alias(self.diagnosis_icd9_code_col),
                # Impute missing ICD10 codes
                pl.when(pl.col(self.diagnosis_icd_version_col) == 10)
                .then(pl.col(self.diagnosis_icd_code_col))
                .otherwise(pl.col(self.diagnosis_icd_code_col).replace(ICD9_TO_ICD10_MAPPING))
                .alias(self.diagnosis_icd10_code_col),
            )
            .select(
                [
                    self.global_person_id_col,
                    self.global_hospital_stay_id_col,
                    self.global_icu_stay_id_col,
                    self.diagnosis_icd9_code_col,
                    self.diagnosis_icd10_code_col,
                    self.diagnosis_start_col,
                    self.diagnosis_priority_col,
                    self.diagnosis_discharge_col,
                    self.diagnosis_description_col,
                ]
            )
            .unique()
            .sort([self.global_icu_stay_id_col, self.diagnosis_start_col])
        )


if __name__ == "__main__":
    raise NotImplementedError("This script is not yet implemented as a command line tool.")
