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
    def __init__(self, paths, datasets: list, DEMO=False):
        super().__init__(paths)
        self.eicu = EICUExtractor(paths, DEMO)
        self.hirid = HiRIDExtractor(paths)
        self.mimic3 = MIMIC3Extractor(paths, DEMO)
        self.mimic4 = MIMIC4Extractor(paths, DEMO)
        self.sicdb = SICdbExtractor(paths)
        self.umcdb = UMCdbExtractor(paths)
        self.datasets = datasets

    def harmonize_medications(self) -> pl.LazyFrame:

        if self.datasets == []:
            raise ValueError("No datasets to harmonize the medications from.")

        medications_datasets = []

        if "eICU" in self.datasets:
            eicu_medications = self.eicu.extract_medications().pipe(
                self._concat_helper, "eicu-"
            )
            self._print_unique_cases(
                eicu_medications, "eICU", self.global_icu_stay_id_col
            )
            medications_datasets.append(eicu_medications)

        if "HiRID" in self.datasets:
            hirid_medications = self.hirid.extract_medications().pipe(
                self._concat_helper, "hirid-"
            )
            self._print_unique_cases(
                hirid_medications, "HiRID", self.global_icu_stay_id_col
            )
            medications_datasets.append(hirid_medications)

        if "MIMIC3" in self.datasets:
            mimic3_medications = self.mimic3.extract_medications().pipe(
                self._concat_helper, "mimic3-"
            )
            self._print_unique_cases(
                mimic3_medications, "MIMIC3", self.global_icu_stay_id_col
            )
            medications_datasets.append(mimic3_medications)

        if "MIMIC4" in self.datasets:
            mimic4_medications = self.mimic4.extract_medications().pipe(
                self._concat_helper, "mimic4-"
            )
            self._print_unique_cases(
                mimic4_medications, "MIMIC4", self.global_icu_stay_id_col
            )
            medications_datasets.append(mimic4_medications)

        if "SICdb" in self.datasets:
            sicdb_medications = self.sicdb.extract_medications().pipe(
                self._concat_helper, "sicdb-"
            )
            self._print_unique_cases(
                sicdb_medications, "SICdb", self.global_icu_stay_id_col
            )
            medications_datasets.append(sicdb_medications)

        if "UMCdb" in self.datasets:
            umcdb_medications = self.umcdb.extract_medications().pipe(
                self._concat_helper, "umcdb-"
            )
            self._print_unique_cases(
                umcdb_medications, "UMCdb", self.global_icu_stay_id_col
            )
            medications_datasets.append(umcdb_medications)

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
                    self.drug_class_col,
                    self.drug_admin_route_col,
                    self.drug_amount_col,
                    self.drug_amount_unit_col,
                    self.drug_rate_col,
                    self.drug_rate_unit_col,
                    self.drug_start_col,
                    self.drug_end_col,
                ]
            )
            .cast(
                {
                    self.drug_amount_col: float,
                    self.drug_rate_col: float,
                    self.drug_class_col: str,
                    self.drug_admin_route_col: str,
                },
                strict=False,
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

    # Print the number of unique cases in the timeseries data
    def _print_unique_cases(
        self, data: pl.LazyFrame, name: str, count_col: str
    ) -> None:
        unique_count = (
            data.select(self.global_icu_stay_id_col)
            .unique()
            .count()
            .collect(streaming=True)
            .to_numpy()[0][0]
        )
        print(
            f"reprodICU - {unique_count:6.0f} unique cases with timeseries data in {name}."
        )
