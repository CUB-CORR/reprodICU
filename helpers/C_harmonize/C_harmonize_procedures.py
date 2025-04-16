# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script combines the preprocessed procedures from the differet
# databases into one common table

import polars as pl

from helpers.A_extract.A_extract_eicu import EICUExtractor
from helpers.A_extract.A_extract_mimic3 import MIMIC3Extractor
from helpers.A_extract.A_extract_mimic4 import MIMIC4Extractor
from helpers.A_extract.A_extract_nwicu import NWICUExtractor
from helpers.A_extract.AX_extract_sicdb import SICdbExtractor
from helpers.A_extract.AX_extract_umcdb import UMCdbExtractor
from helpers.helper import GlobalVars


class ProceduresHarmonizer(GlobalVars):
    def __init__(self, paths, datasets: list, DEMO=False):
        """
        Initializes the ProceduresHarmonizer class with the given paths and datasets.

        Args:
            paths (str): The file paths required for data extraction.
            datasets (list): A list of datasets to be harmonized.
            DEMO (bool, optional): A flag indicating whether to use demo data. Defaults to False.
        """
        super().__init__(paths)
        self.eicu = EICUExtractor(paths, DEMO)
        # self.hirid = HiRIDExtractor(paths)
        self.mimic3 = MIMIC3Extractor(paths, DEMO)
        self.mimic4 = MIMIC4Extractor(paths, DEMO)
        self.nwicu = NWICUExtractor(paths)
        self.sicdb = SICdbExtractor(paths)
        self.umcdb = UMCdbExtractor(paths)
        self.datasets = datasets

    def harmonize_procedures(self) -> pl.LazyFrame:
        """
        Harmonizes procedure data from multiple databases into a single LazyFrame.

        This function performs the following steps:
            1. Validates that a non-empty list of datasets is provided; raises a ValueError if empty.
            2. Initializes an empty list to accumulate procedure datasets.
            3. For each dataset in {datasets}:
               - If "eICU" is present: Extracts procedures using EICUExtractor and applies _concat_helper1 to generate global IDs.
               - If "MIMIC3" is present: Extracts procedures using MIMIC3Extractor and applies _concat_helper1.
               - If "MIMIC4" is present: Extracts procedures using MIMIC4Extractor and applies _concat_helper1.
               - If "NWICU" is present: Extracts procedures using NWICUExtractor and applies _concat_helper1.
               - If "SICdb" is present: Extracts procedures using SICdbExtractor and applies _concat_helper2.
               - If "UMCdb" is present: Extracts procedures using UMCdbExtractor and applies _concat_helper2.
            4. Concatenates all accumulated procedure datasets using a "diagonal_relaxed" join.
            5. Selects specific columns and removes duplicate records.

        The final returned LazyFrame contains the following columns:
            - {global_person_id_col}: Global person identifier.
            - {global_hospital_stay_id_col}: Global hospital stay identifier.
            - {global_icu_stay_id_col}: Global ICU stay identifier.
            - {procedure_icd_code_col}: ICD code corresponding to the procedure.
            - {procedure_icd_version_col}: Version of the ICD code (e.g., ICD-9, ICD-10).
            - {procedure_category_col}: Category grouping for the procedure.
            - {procedure_start_col}: Start time of the procedure.
            - {procedure_end_col}: End time of the procedure.
            - {procedure_priority_col}: Priority level for the procedure.
            - {procedure_discharge_col}: Indicates if the procedure is active at discharge.
            - {procedure_description_col}: Description of the procedure.

        Returns:
            pl.LazyFrame: A LazyFrame containing harmonized procedure data with the columns listed above.

        Raises:
            ValueError: If no datasets are provided.
        """
        if self.datasets == []:
            raise ValueError("No datasets to harmonize the procedures from.")

        procedures_datasets = []

        if "eICU" in self.datasets:
            procedures_datasets.append(
                self.eicu.extract_treatments().pipe(
                    self._concat_helper1, "eicu-"
                )
            )

        if "MIMIC3" in self.datasets:
            procedures_datasets.append(
                self.mimic3.extract_procedures().pipe(
                    self._concat_helper1, "mimic3-"
                )
            )

        if "MIMIC4" in self.datasets:
            procedures_datasets.append(
                self.mimic4.extract_procedures().pipe(
                    self._concat_helper1, "mimic4-"
                )
            )

        if "NWICU" in self.datasets:
            procedures_datasets.append(
                self.nwicu.extract_procedures().pipe(
                    self._concat_helper1, "nwicu-"
                )
            )

        if "SICdb" in self.datasets:
            procedures_datasets.append(
                self.sicdb.extract_procedures().pipe(
                    self._concat_helper2, "sicdb-"
                )
            )

        if "UMCdb" in self.datasets:
            procedures_datasets.append(
                self.umcdb.extract_procedures().pipe(
                    self._concat_helper2, "umcdb-"
                )
            )

        procedures = pl.concat(procedures_datasets, how="diagonal_relaxed")
        procedures_cols_list = [
            self.global_person_id_col,
            self.global_hospital_stay_id_col,
            self.global_icu_stay_id_col,
            self.procedure_icd_code_col,
            self.procedure_icd_version_col,
            self.procedure_category_col,
            self.procedure_start_col,
            self.procedure_end_col,
            self.procedure_priority_col,
            self.procedure_discharge_col,
            self.procedure_description_col,
        ]

        return (
            procedures.select(
                col
                for col in procedures_cols_list
                if col in procedures.columns
            )
            .unique()
            .sort(self.global_icu_stay_id_col, self.procedure_start_col)
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
        )

    def _concat_helper2(self, data: pl.LazyFrame, name: str) -> pl.LazyFrame:
        return data.with_columns(
            pl.concat_str([pl.lit(name), pl.col(self.person_id_col)]).alias(
                self.global_person_id_col
            ),
            pl.concat_str([pl.lit(name), pl.col(self.icu_stay_id_col)]).alias(
                self.global_icu_stay_id_col
            ),
        )
