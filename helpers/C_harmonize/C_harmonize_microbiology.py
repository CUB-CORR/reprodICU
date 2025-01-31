# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script combines the preprocessed microbiology from the different
# databases into one common table

import polars as pl

from helpers.A_extract.A_extract_eicu import EICUExtractor
from helpers.A_extract.A_extract_mimic3 import MIMIC3Extractor
from helpers.A_extract.A_extract_mimic4 import MIMIC4Extractor
from helpers.A_extract.AX_extract_umcdb import UMCdbExtractor
from helpers.helper import GlobalVars
from helpers.helper import GlobalHelpers


class MicrobiologyHarmonizer(GlobalVars):
    def __init__(self, paths, datasets: list, DEMO=False):
        super().__init__(paths)
        self.eicu = EICUExtractor(paths, DEMO)
        self.mimic3 = MIMIC3Extractor(paths, DEMO)
        self.mimic4 = MIMIC4Extractor(paths, DEMO)
        self.umcdb = UMCdbExtractor(paths)
        self.helpers = GlobalHelpers()
        self.datasets = datasets

    def harmonize_microbiology(self) -> pl.LazyFrame:

        if self.datasets == []:
            raise ValueError("No datasets to harmonize the microbiology from.")

        microbiology_datasets = []

        if "eICU" in self.datasets:
            microbiology_datasets.append(
                self.eicu.extract_microbiology()
                .pipe(self._concat_helper, "eicu-")
                .pipe(self._print_unique_cases, "eICU")
            )

        if "MIMIC3" in self.datasets:
            microbiology_datasets.append(
                self.mimic3.extract_microbiology()
                .pipe(self._concat_helper, "mimic3-")
                .pipe(self._print_unique_cases, "MIMIC3")
            )

        if "MIMIC4" in self.datasets:
            microbiology_datasets.append(
                self.mimic4.extract_microbiology()
                .pipe(self._concat_helper, "mimic4-")
                .pipe(self._print_unique_cases, "MIMIC4")
            )

        # if "UMCdb" in self.datasets:
        #     microbiology_datasets.append(
        #         self.umcdb.extract_microbiology()
        #         .pipe(self._concat_helper, "umcdb-")
        #         .pipe(self._print_unique_cases, "UMCdb")
        #     )

        return (
            pl.concat(
                microbiology_datasets,
                how="diagonal_relaxed",
            )
            .select(
                self.global_icu_stay_id_col,
                self.timeseries_time_col,
                self.micro_specimen_col,
                self.micro_test_col,
                self.micro_organism_col,
                self.micro_antibiotic_col,
                self.micro_dilution_col,
                self.micro_sensitivity_col,
            )
            .unique()
            .sort(self.global_icu_stay_id_col, self.timeseries_time_col)
        )

    # Helper functions
    # Concatenate the IDs with the database name to create a global ID
    def _concat_helper(self, data: pl.LazyFrame, name: str) -> pl.LazyFrame:
        return data.with_columns(
            pl.concat_str([pl.lit(name), pl.col(self.icu_stay_id_col)]).alias(
                self.global_icu_stay_id_col
            )
        )

    # Print the number of unique cases in the Microbiology data
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
            f"reprodICU - {unique_count:6.0f} unique cases with Microbiology data in {name}."
        )

        return data
