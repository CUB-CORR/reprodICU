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
        """
        Initializes the MicrobiologyHarmonizer class with the given paths and datasets.

        Args:
            paths (str): The file paths required for data extraction.
            datasets (list): A list of datasets to be harmonized.
            DEMO (bool, optional): A flag indicating whether to use demo data. Defaults to False.
        """
        super().__init__(paths)
        self.eicu = EICUExtractor(paths, DEMO)
        self.mimic3 = MIMIC3Extractor(paths, DEMO)
        self.mimic4 = MIMIC4Extractor(paths, DEMO)
        self.umcdb = UMCdbExtractor(paths)
        self.helpers = GlobalHelpers()
        self.datasets = datasets

    def harmonize_microbiology(self) -> pl.LazyFrame:
        """
        Harmonizes microbiology data from multiple databases into a single LazyFrame.

        This function performs the following steps:
            1. Validates that a non-empty list of datasets is provided; raises ValueError if empty.
            2. Initializes an empty list to accumulate microbiology datasets.
            3. For each dataset in {datasets}:
               - If "eICU" is present: Extracts microbiology data using EICUExtractor, applies _concat_helper to create a global ID and prints the unique cases using _print_unique_cases.
               - If "MIMIC3" is present: Extracts microbiology data using MIMIC3Extractor with similar processing.
               - If "MIMIC4" is present: Extracts microbiology data using MIMIC4Extractor with similar processing.
               - (Note: Extraction for "UMCdb" is commented out.)
            4. Concatenates all accumulated datasets using a "diagonal_relaxed" join.
            5. Selects specific columns, removes duplicate records, and sorts based on {global_icu_stay_id_col} and {timeseries_time_col}.

        The final returned LazyFrame contains the following columns:
            - {global_icu_stay_id_col}: Global ICU stay identifier.
            - {timeseries_time_col}: Timestamp for the time series data.
            - {micro_specimen_col}: Specimen type used in the microbiology test.
            - {micro_test_col}: Identifier or name of the microbiology test.
            - {micro_organism_col}: Identified microorganism in the test.
            - {micro_antibiotic_col}: Antibiotic used or administered.
            - {micro_dilution_col}: Dilution value reported in the test.
            - {micro_sensitivity_col}: Result indicating microorganism sensitivity.

        Returns:
            pl.LazyFrame: A LazyFrame containing harmonized microbiology data with the columns listed above.

        Raises:
            ValueError: If no datasets are provided.
        """
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

        microbiology = pl.concat(microbiology_datasets, how="diagonal_relaxed")
        microbiology_cols_list = [
            self.global_icu_stay_id_col,
            self.timeseries_time_col,
            self.micro_specimen_col,
            self.micro_test_col,
            self.micro_organism_col,
            self.micro_antibiotic_col,
            self.micro_dilution_col,
            self.micro_sensitivity_col,
        ]

        return (
            microbiology.select(
                col
                for col in microbiology_cols_list
                if col in microbiology.columns
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
