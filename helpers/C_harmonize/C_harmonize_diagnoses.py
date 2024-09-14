# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script combines the preprocessed diagnoses from the differet
# databases into one common table

import polars as pl

from helpers.B_process.B_process_eicu import EICUProcessor
from helpers.A_extract.A_extract_mimic3 import MIMIC3Extractor
from helpers.A_extract.A_extract_mimic4 import MIMIC4Extractor
from helpers.A_extract.AX_extract_sicdb import SICdbExtractor
from helpers.helper import GlobalVars


class DiagnosesHarmonizer(GlobalVars):
    def __init__(self, paths, datasets: list):
        super().__init__(paths)
        self.eicu = EICUProcessor(paths)
        # self.hirid = HiRIDExtractor(paths)
        self.mimic3 = MIMIC3Extractor(paths)
        self.mimic4 = MIMIC4Extractor(paths)
        self.sicdb = SICdbExtractor(paths)
        # self.umcdb = UMCdbExtractor(paths)
        self.datasets = datasets

    def harmonize_diagnoses(self) -> pl.LazyFrame:
        """
        Harmonize the diagnoses from the different databases.

        The final table contains the following columns:
        - Person ID
        - Hospital stay ID
        - ICU stay ID
        - Diagnosis ICD code
        - Diagnosis ICD version
        - Diagnosis start
        - Diagnosis priority
        - Diagnosis active at discharge
        - Diagnosis description

        :param self.datasets: The datasets to harmonize the diagnoses from.

        :return: The harmonized diagnoses.
        :rtype: pl.LazyFrame
        """

        if self.datasets == []:
            raise ValueError("No datasets to harmonize the diagnoses from.")

        diagnoses_datasets = []

        # Harmonize the diagnoses
        if "eICU" in self.datasets:
            diagnoses_datasets.append(
                self.eicu.process_diagnoses().pipe(
                    self._concat_helper1, "eicu-"
                )
            )

        if "MIMIC3" in self.datasets:
            diagnoses_datasets.append(
                self.mimic3.extract_diagnoses().pipe(
                    self._concat_helper2, "mimic3-"
                )
            )

        if "MIMIC4" in self.datasets:
            diagnoses_datasets.append(
                self.mimic4.extract_diagnoses().pipe(
                    self._concat_helper2, "mimic4-"
                )
            )

        if "SICdb" in self.datasets:
            diagnoses_datasets.append(
                self.sicdb.extract_diagnoses().pipe(
                    self._concat_helper3, "sicdb-"
                )
            )

        return (
            pl.concat(
                diagnoses_datasets,
                how="diagonal_relaxed",
            )
            .select(
                [
                    self.global_person_id_col,
                    self.global_hospital_stay_id_col,
                    self.global_icu_stay_id_col,
                    self.diagnosis_icd_code_col,
                    self.diagnosis_icd_version_col,
                    self.diagnosis_start_col,
                    self.diagnosis_priority_col,
                    self.diagnosis_discharge_col,
                    self.diagnosis_description_col,
                ]
            )
            .unique()
            .sort([self.global_icu_stay_id_col, self.diagnosis_start_col])
        )

    # Helper functions
    # Concatenate the IDs with the database name to create a global ID
    def _concat_helper1(self, data: pl.LazyFrame, name: str) -> pl.LazyFrame:
        return data.with_columns(
            pl.concat_str([pl.lit(name), pl.col(self.person_id_col)]).alias(
                self.global_person_id_col
            ),
            pl.concat_str(
                [pl.lit(name), pl.col(self.hospital_stay_id_col)]
            ).alias(self.global_hospital_stay_id_col),
            pl.concat_str([pl.lit(name), pl.col(self.icu_stay_id_col)]).alias(
                self.global_icu_stay_id_col
            ),
        ).lazy()

    def _concat_helper2(self, data: pl.LazyFrame, name: str) -> pl.LazyFrame:
        return data.with_columns(
            pl.concat_str([pl.lit(name), pl.col(self.person_id_col)]).alias(
                self.global_person_id_col
            ),
            pl.concat_str(
                [pl.lit(name), pl.col(self.hospital_stay_id_col)]
            ).alias(self.global_hospital_stay_id_col),
        ).lazy()

    def _concat_helper3(self, data: pl.LazyFrame, name: str) -> pl.LazyFrame:
        return data.with_columns(
            pl.concat_str([pl.lit(name), pl.col(self.person_id_col)]).alias(
                self.global_person_id_col
            ),
            pl.concat_str([pl.lit(name), pl.col(self.icu_stay_id_col)]).alias(
                self.global_icu_stay_id_col
            ),
        ).lazy()
