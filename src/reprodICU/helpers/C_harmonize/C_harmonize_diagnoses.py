# Author: Finn Fassbender
# Last modified: 2025-10-30

# Description: This script combines the preprocessed diagnoses from the differet
# databases into one common table

import polars as pl

from ..A_extract.A_extract_mimic3 import MIMIC3Extractor
from ..A_extract.A_extract_mimic4 import MIMIC4Extractor
from ..A_extract.A_extract_nwicu import NWICUExtractor
from ..A_extract.A_extract_sicdb import SICdbExtractor
from ..B_process.B_process_eicu import EICUProcessor
from ..helper import GlobalVars


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
        self.datasets = datasets
        
        if "eICU" in datasets:
            self.eicu = EICUProcessor(paths, DEMO)
        if "MIMIC3" in datasets:
            self.mimic3 = MIMIC3Extractor(paths, DEMO)
        if "MIMIC4" in datasets:
            self.mimic4 = MIMIC4Extractor(paths, DEMO)
        if "NWICU" in datasets:
            self.nwicu = NWICUExtractor(paths)
        if "SICdb" in datasets:
            self.sicdb = SICdbExtractor(paths)

    def harmonize_diagnoses(self) -> pl.LazyFrame:
        """
        Harmonize diagnoses from multiple databases.

        Steps:
            1. Validate non-empty dataset list; raise ValueError if empty.
            2. For each dataset: process/extract diagnoses and create global identifiers.
            3. Apply database-specific identifier concatenation via helper methods.
            4. Concatenate all datasets using diagonal-relaxed join.
            5. Select columns in standardized order.
            6. Remove duplicates and sort by ICU stay and diagnosis start time.

        Returns:
            pl.LazyFrame: Contains columns:
                - {global_person_id_col}: Global person identifier.
                - {global_hospital_stay_id_col}: Global hospital stay identifier.
                - {global_icu_stay_id_col}: Global ICU stay identifier.
                - {diagnosis_icd_code_col}: ICD diagnosis code.
                - {diagnosis_icd_version_col}: ICD version (e.g., "9", "10").
                - {diagnosis_start_col}: Diagnosis start datetime.
                - {diagnosis_end_col}: Diagnosis end datetime.
                - {diagnosis_priority_col}: Diagnosis priority/sequence.
                - {diagnosis_discharge_col}: Active at discharge flag.
                - {diagnosis_description_col}: Diagnosis description text.
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
                col for col in diagnoses_cols_list if col in diagnoses.columns
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
