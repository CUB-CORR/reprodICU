# Author: Finn Fassbender
# Last modified: 2025-10-30

# Description: This script combines the preprocessed procedures from the differet
# databases into one common table

import polars as pl

from ..A_extract.A_extract_eicu import EICUExtractor
from ..A_extract.A_extract_mimic3 import MIMIC3Extractor
from ..A_extract.A_extract_mimic4 import MIMIC4Extractor
from ..A_extract.A_extract_nwicu import NWICUExtractor
from ..A_extract.A_extract_sicdb import SICdbExtractor
from ..A_extract.A_extract_umcdb import UMCdbExtractor
from ..helper import GlobalVars


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
        self.datasets = datasets
        
        if "eICU" in self.datasets:
            self.eicu = EICUExtractor(paths, DEMO)
        if "MIMIC3" in self.datasets:
            self.mimic3 = MIMIC3Extractor(paths, DEMO)
        if "MIMIC4" in self.datasets:
            self.mimic4 = MIMIC4Extractor(paths, DEMO)
        if "NWICU" in self.datasets:
            self.nwicu = NWICUExtractor(paths)
        if "SICdb" in self.datasets:
            self.sicdb = SICdbExtractor(paths)
        if "UMCdb" in self.datasets:
            self.umcdb = UMCdbExtractor(paths)

    def harmonize_procedures(self) -> pl.LazyFrame:
        """
        Harmonize procedure data from multiple databases.

        Steps:
            1. Validate non-empty dataset list; raise ValueError if empty.
            2. For each dataset: extract procedures and create global identifiers.
            3. Apply database-specific identifier concatenation via helper methods.
            4. Concatenate all datasets using diagonal-relaxed join.
            5. Select columns in standardized order.
            6. Remove duplicates.

        Returns:
            pl.LazyFrame: Contains columns:
                - {global_person_id_col}: Global person identifier.
                - {global_hospital_stay_id_col}: Global hospital stay identifier.
                - {global_icu_stay_id_col}: Global ICU stay identifier.
                - {procedure_icd_code_col}: ICD procedure code.
                - {procedure_icd_version_col}: ICD version (e.g., "9", "10").
                - {procedure_category_col}: Procedure category grouping.
                - {procedure_start_col}: Procedure start datetime.
                - {procedure_end_col}: Procedure end datetime.
                - {procedure_priority_col}: Procedure priority/sequence.
                - {procedure_discharge_col}: Active at discharge flag.
                - {procedure_description_col}: Procedure description text.
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
                col for col in procedures_cols_list if col in procedures.columns
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
