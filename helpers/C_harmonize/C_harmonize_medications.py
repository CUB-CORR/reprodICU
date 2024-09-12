# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script combines the preprocessed medications from the different
# databases into one common table

import polars as pl

from helpers.A_extract.A_extract_eicu import EICUExtractor
from helpers.A_extract.AX_extract_hirid import HiRIDExtractor
from helpers.A_extract.A_extract_mimic3 import MIMIC3Extractor
from helpers.A_extract.A_extract_mimic4 import MIMIC4Extractor
from helpers.A_extract.AX_extract_sicdb import SICdbExtractor
from helpers.A_extract.AX_extract_umcdb import UMCdbExtractor
from helpers.helper import GlobalVars


class MedicationHarmonizer(GlobalVars):
    def __init__(self, paths, datasets: list):
        super().__init__(paths)
        self.eicu = EICUExtractor(paths)
        self.hirid = HiRIDExtractor(paths)
        self.mimic3 = MIMIC3Extractor(paths)
        self.mimic4 = MIMIC4Extractor(paths)
        self.sicdb = SICdbExtractor(paths)
        self.umcdb = UMCdbExtractor(paths)
        self.datasets = datasets

    def harmonize_medications(self) -> pl.LazyFrame:

        if self.datasets == []:
            raise ValueError("No datasets to harmonize the medications from.")

        medications_datasets = []

        if "eICU" in self.datasets:
            medications_datasets.append(
                self.eicu.extract_medications().pipe(self._concat_helper, "eicu-")
            )

        if "HiRID" in self.datasets:
            medications_datasets.append(
                self.hirid.extract_medications().pipe(self._concat_helper, "hirid-")
            )

        if "MIMIC3" in self.datasets:
            medications_datasets.append(
                self.mimic3.extract_medications().pipe(self._concat_helper, "mimic3-")
            )

        if "MIMIC4" in self.datasets:
            medications_datasets.append(
                self.mimic4.extract_medications().pipe(self._concat_helper, "mimic4-")
            )

        if "SICdb" in self.datasets:
            medications_datasets.append(
                self.sicdb.extract_medications().pipe(self._concat_helper, "sicdb-")
            )

        if "UMCdb" in self.datasets:
            medications_datasets.append(
                self.umcdb.extract_medications().pipe(self._concat_helper, "umcdb-")
            )

        return (
            pl.concat(
                medications_datasets,
                how="diagonal_relaxed",
            )
            .select(
                [
                    self.global_icu_stay_id_col,
                    self.drug_ingredient_col,
                    self.drug_name_col,
                    self.drug_amount_col,
                    self.drug_start_col,
                    self.drug_end_col,
                ]
            )
            .unique()
            .sort([self.global_icu_stay_id_col, self.drug_start_col])
        )

    # Helper functions
    # Concatenate the IDs with the database name to create a global ID
    def _concat_helper(self, data: pl.LazyFrame, name: str) -> pl.LazyFrame:
        return data.with_columns(
            pl.concat_str([pl.lit(name), pl.col(self.icu_stay_id_col)]).alias(
                self.global_icu_stay_id_col
            )
        )
