# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script combines the preprocessed procedures from the differet
# databases into one common table

import polars as pl

from helpers.A_extract.A_extract_eicu import EICUExtractor
from helpers.A_extract.A_extract_mimic3 import MIMIC3Extractor
from helpers.A_extract.A_extract_mimic4 import MIMIC4Extractor
from helpers.helper import GlobalVars


class ProceduresHarmonizer(GlobalVars):
    def __init__(self, paths, datasets: list):
        super().__init__(paths)
        self.eicu = EICUExtractor(paths)
        # self.hirid = HiRIDExtractor(paths)
        self.mimic3 = MIMIC3Extractor(paths)
        self.mimic4 = MIMIC4Extractor(paths)
        # self.sicdb = SICdbExtractor(paths)
        # self.umcdb = UMCdbExtractor(paths)
        self.datasets = datasets

    def harmonize_procedures(self) -> pl.LazyFrame:

        if self.datasets == []:
            raise ValueError("No datasets to harmonize the procedures from.")

        procedures_datasets = []

        if "eICU" in self.datasets:
            procedures_datasets.append(
                self.eicu.extract_treatments().pipe(self._concat_helper1, "eicu-")
            )

        if "MIMIC3" in self.datasets:
            procedures_datasets.append(
                self.mimic3.extract_procedures().pipe(self._concat_helper2, "mimic3-")
            )

        if "MIMIC4" in self.datasets:
            procedures_datasets.append(
                self.mimic4.extract_procedures().pipe(self._concat_helper2, "mimic4-")
            )

        return (
            pl.concat(
                procedures_datasets,
                how="diagonal_relaxed",
            )
            .select(
                [
                    self.global_person_id_col,
                    self.global_hospital_stay_id_col,
                    # self.global_icu_stay_id_col,
                    self.procedure_icd_code_col,
                    self.procedure_icd_version_col,
                    self.procedure_start_col,
                    self.procedure_priority_col,
                    self.procedure_discharge_col,
                    self.procedure_description_col,
                ]
            )
            .unique()
            .drop_nulls(self.procedure_discharge_col)
        )

    # Helper functions
    # Concatenate the IDs with the database name to create a global ID
    def _concat_helper1(self, data: pl.LazyFrame, name: str) -> pl.LazyFrame:
        return data.with_columns(
            pl.concat_str([pl.lit(name), pl.col(self.person_id_col)]).alias(
                self.global_person_id_col
            ),
            pl.concat_str([pl.lit(name), pl.col(self.hospital_stay_id_col)]).alias(
                self.global_hospital_stay_id_col
            ),
            pl.concat_str([pl.lit(name), pl.col(self.icu_stay_id_col)]).alias(
                self.global_icu_stay_id_col
            ),
        )

    def _concat_helper2(self, data: pl.LazyFrame, name: str) -> pl.LazyFrame:
        return data.with_columns(
            pl.concat_str([pl.lit(name), pl.col(self.person_id_col)]).alias(
                self.global_person_id_col
            ),
            pl.concat_str([pl.lit(name), pl.col(self.hospital_stay_id_col)]).alias(
                self.global_hospital_stay_id_col
            ),
        )
