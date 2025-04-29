# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script combines the preprocessed diagnoses from the differet
# databases into one common table

import polars as pl

from helpers.B_process.B_process_eicu import EICUProcessor
from helpers.A_extract.A_extract_mimic3 import MIMIC3Extractor
from helpers.A_extract.A_extract_mimic4 import MIMIC4Extractor
from helpers.A_extract.A_extract_nwicu import NWICUExtractor
from helpers.A_extract.AX_extract_sicdb import SICdbExtractor
from helpers.helper import GlobalVars


class DiagnosesHarmonizer(GlobalVars):
    def __init__(self, paths, datasets: list, DEMO=False):
        """
        Initializes the DiagnosesHarmonizer class with the given paths and datasets.

        Args:
            paths (str): The file paths required for data extraction.
            datasets (list): A list of datasets to be harmonized.
            DEMO (bool, optional): A flag indicating whether to use demo data. Defaults to False.
        """
        super().__init__(paths)
        self.eicu = EICUProcessor(paths, DEMO)
        # self.hirid = HiRIDExtractor(paths)
        self.mimic3 = MIMIC3Extractor(paths, DEMO)
        self.mimic4 = MIMIC4Extractor(paths, DEMO)
        self.nwicu = NWICUExtractor(paths)
        self.sicdb = SICdbExtractor(paths)
        # self.umcdb = UMCdbExtractor(paths)
        self.datasets = datasets

    def harmonize_diagnoses(self) -> pl.LazyFrame:
        """
        Harmonizes diagnoses from multiple databases into a single LazyFrame.

        This function performs the following steps:
            1. Validates that a non-empty list of datasets is provided; raises ValueError if empty.
            2. Initializes an empty list to accumulate diagnoses datasets.
            3. For each dataset in {datasets}:
               - If "eICU" is present: Processes diagnoses using eICUProcessor and applies _concat_helper1 to generate global identifiers.
               - If "MIMIC3" is present: Extracts diagnoses via MIMIC3Extractor and applies _concat_helper2.
               - If "MIMIC4" is present: Extracts diagnoses via MIMIC4Extractor and applies _concat_helper2.
               - If "NWICU" is present: Extracts diagnoses via NWICUExtractor and applies _concat_helper2.
               - If "SICdb" is present: Extracts diagnoses via SICdbExtractor and applies _concat_helper3.
            4. Concatenates all accumulated diagnoses datasets using a "diagonal_relaxed" join.
            5. Selects specific columns and ensures uniqueness and sorting based on {global_icu_stay_id_col} and {diagnosis_start_col}.

        The final returned LazyFrame contains the following columns:
            - {global_person_id_col}: Global person identifier.
            - {global_hospital_stay_id_col}: Global hospital stay identifier.
            - {global_icu_stay_id_col}: Global ICU stay identifier.
            - {diagnosis_icd_code_col}: Diagnosis ICD code.
            - {diagnosis_icd_version_col}: Diagnosis ICD version.
            - {diagnosis_start_col}: Start time of the diagnosis.
            - {diagnosis_end_col}: End time of the diagnosis.
            - {diagnosis_priority_col}: Priority level of the diagnosis.
            - {diagnosis_discharge_col}: Indicates if the diagnosis was active at discharge.
            - {diagnosis_description_col}: Textual description of the diagnosis.

        Returns:
            pl.LazyFrame: Harmonized diagnoses data containing the columns listed above.

        Raises:
            ValueError: If no datasets are provided.
        """
        if self.datasets == []:
            raise ValueError("No datasets to harmonize the diagnoses from.")

        diagnoses_datasets = []

        # Harmonize the diagnoses per dataset
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

        if "NWICU" in self.datasets:
            diagnoses_datasets.append(
                self.nwicu.extract_diagnoses().pipe(
                    self._concat_helper2, "nwicu-"
                )
            )

        if "SICdb" in self.datasets:
            diagnoses_datasets.append(
                self.sicdb.extract_diagnoses().pipe(
                    self._concat_helper3, "sicdb-"
                )
            )

        diagnoses = pl.concat(diagnoses_datasets, how="diagonal_relaxed")
        diagnoses_cols_list = [
            self.global_person_id_col,
            self.global_hospital_stay_id_col,
            self.global_icu_stay_id_col,
            self.diagnosis_icd_code_col,
            self.diagnosis_icd_version_col,
            self.diagnosis_start_col,
            self.diagnosis_end_col,
            self.diagnosis_priority_col,
            self.diagnosis_discharge_col,
            self.diagnosis_description_col,
        ]

        return (
            diagnoses.select(
                col
                for col in diagnoses_cols_list
                if col in diagnoses.columns
            )
            .unique()
            .sort(self.global_icu_stay_id_col, self.diagnosis_start_col)
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
