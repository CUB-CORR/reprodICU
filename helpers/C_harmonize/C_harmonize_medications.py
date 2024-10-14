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
from helpers.helper import GlobalHelpers


class MedicationHarmonizer(GlobalVars):
    def __init__(self, paths, datasets: list, DEMO=False):
        super().__init__(paths)
        self.eicu = EICUExtractor(paths, DEMO)
        self.hirid = HiRIDExtractor(paths)
        self.mimic3 = MIMIC3Extractor(paths, DEMO)
        self.mimic4 = MIMIC4Extractor(paths, DEMO)
        self.sicdb = SICdbExtractor(paths)
        self.umcdb = UMCdbExtractor(paths)
        self.helpers = GlobalHelpers()
        self.datasets = datasets

    def harmonize_medications(self) -> pl.LazyFrame:

        if self.datasets == []:
            raise ValueError("No datasets to harmonize the medications from.")

        fluids_class_mapping = self.helpers.load_many_to_one_mapping(
            self.mapping_path + "MEDICATIONS_FLUIDS_CLASSES.yaml"
        )
        drugs_class_mapping = self.helpers.load_mapping(
            self.mapping_path + "MEDICATIONS_DRUGS_CLASSES.yaml"
        )

        medications_datasets = []

        if "eICU" in self.datasets:
            medications_datasets.append(
                self.eicu.extract_medications()
                .pipe(self._concat_helper, "eicu-")
                .pipe(self._print_unique_cases, "eICU")
            )

        if "HiRID" in self.datasets:
            medications_datasets.append(
                self.hirid.extract_medications()
                .pipe(self._concat_helper, "hirid-")
                .pipe(self._print_unique_cases, "HiRID")
            )

        if "MIMIC3" in self.datasets:
            medications_datasets.append(
                self.mimic3.extract_medications()
                .pipe(self._concat_helper, "mimic3-")
                .pipe(self._print_unique_cases, "MIMIC3")
            )

        if "MIMIC4" in self.datasets:
            medications_datasets.append(
                self.mimic4.extract_medications()
                .pipe(self._concat_helper, "mimic4-")
                .pipe(self._print_unique_cases, "MIMIC4")
            )

        if "SICdb" in self.datasets:
            medications_datasets.append(
                self.sicdb.extract_medications()
                .pipe(self._concat_helper, "sicdb-")
                .pipe(self._print_unique_cases, "SICdb")
            )

        if "UMCdb" in self.datasets:
            medications_datasets.append(
                self.umcdb.extract_medications()
                .pipe(self._concat_helper, "umcdb-")
                .pipe(self._print_unique_cases, "UMCdb")
            )

        return (
            pl.concat(
                medications_datasets,
                how="diagonal_relaxed",
            )
            # add missing drug class information
            # NOTE: -> refactor into imputation?
            # NOTE: -> prob yes, since one also needs to deal with boluses
            .with_columns(
                pl.when(
                    pl.col(self.drug_name_col).is_in(
                        fluids_class_mapping.keys()
                    )
                )
                .then(pl.col(self.drug_name_col).replace(fluids_class_mapping))
                .when(
                    pl.col(self.drug_ingredient_col).is_in(
                        drugs_class_mapping.keys()
                    )
                )
                .then(
                    pl.col(self.drug_ingredient_col).replace(
                        drugs_class_mapping
                    )
                )
                .otherwise(pl.col(self.drug_class_col))
                .alias(self.drug_class_col),
                # harmonize units
                pl.col(self.drug_amount_unit_col)
                .str.replace("mL", "ml")
                .str.replace(r"^U$", "units")
                .str.replace("µ", "mc")
                .str.replace("grams", "g")
                .str.replace("mEQ", "mEq")
                .str.replace("mEq\.", "mEq")
                .alias(self.drug_amount_unit_col),
            )
            .select(
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
            .sort(self.global_icu_stay_id_col, self.drug_start_col)
        )

    # Helper functions
    # Concatenate the IDs with the database name to create a global ID
    def _concat_helper(self, data: pl.LazyFrame, name: str) -> pl.LazyFrame:
        return data.with_columns(
            pl.concat_str([pl.lit(name), pl.col(self.icu_stay_id_col)]).alias(
                self.global_icu_stay_id_col
            )
        )

    # Print the number of unique cases in the medication data
    def _print_unique_cases(
        self, data: pl.LazyFrame, name: str
    ) -> pl.LazyFrame:
        unique_count = (
            data.select(self.global_icu_stay_id_col)
            .unique()
            .count()
            .collect(streaming=True)
            .to_numpy()[0][0]
        )
        print(
            f"reprodICU - {unique_count:6.0f} unique cases with medication data in {name}."
        )

        return data
