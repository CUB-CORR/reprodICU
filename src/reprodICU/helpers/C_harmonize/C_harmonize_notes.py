# Author: Finn Fassbender
# Last modified: 2025-10-30

# Description: This script combines the preprocessed notes from the differet
# databases into one common table

import polars as pl

from ..A_extract.A_extract_mimic3 import MIMIC3Extractor
from ..A_extract.A_extract_mimic4 import MIMIC4Extractor
from ..helper import GlobalVars


class NotesHarmonizer(GlobalVars):
    def __init__(self, paths, datasets: list, DEMO=False):
        """
        Initializes the NotesHarmonizer class with the given paths and datasets.

        Args:
            paths (str): The file paths required for data extraction.
            datasets (list): A list of datasets to be harmonized.
            DEMO (bool, optional): A flag indicating whether to use demo data. Defaults to False.
        """
        super().__init__(paths)
        # self.eicu = EICUExtractor(paths, DEMO)
        # self.hirid = HiRIDExtractor(paths)
        self.mimic3 = MIMIC3Extractor(paths, DEMO)
        self.mimic4 = MIMIC4Extractor(paths, DEMO)
        # self.nwicu = NWICUExtractor(paths)
        # self.sicdb = SICdbExtractor(paths)
        # self.umcdb = UMCdbExtractor(paths)
        self.datasets = datasets

    def harmonize_notes(self) -> pl.LazyFrame:
        """
        Harmonize notes data from multiple databases.

        Steps:
            1. Validate non-empty dataset list; raise ValueError if empty.
            2. For each dataset: extract notes and create global identifiers.
            3. Apply database-specific identifier concatenation via helper methods.
            4. Concatenate all datasets using diagonal-relaxed join.
            5. Select columns in standardized order.
            6. Remove duplicates.

        Returns:
            pl.LazyFrame: Contains columns:
                - {global_person_id_col}: Global person identifier.
                - {global_hospital_stay_id_col}: Global hospital stay identifier.
                - {global_icu_stay_id_col}: Global ICU stay identifier.
                - {note_time_col}: Note timestamp.
                - {note_category_col}: Note category (e.g., Nursing, Physician).
                - {note_description_col}: Note description.
                - {note_text_col}: Note text content.
        """
        if self.datasets == []:
            raise ValueError("No datasets to harmonize the notes from.")

        notes_datasets = []

        # if "eICU" in self.datasets:
        #     notes_datasets.append(
        #         self.eicu.extract_treatments().pipe(
        #             self._concat_helper1, "eicu-"
        #         )
        #     )

        if "MIMIC3" in self.datasets:
            notes_datasets.append(
                self.mimic3.extract_notes().pipe(
                    self._concat_helper1, "mimic3-"
                )
            )

        if "MIMIC4" in self.datasets:
            notes_datasets.append(
                self.mimic4.extract_notes().pipe(
                    self._concat_helper1, "mimic4-"
                )
            )

        notes = pl.concat(notes_datasets, how="diagonal_relaxed")
        notes_cols_list = [
            self.global_person_id_col,
            self.global_hospital_stay_id_col,
            self.global_icu_stay_id_col,
            self.note_time_col,
            self.note_category_col,
            self.note_description_col,
            self.note_text_col,
        ]

        return (
            notes.select(col for col in notes_cols_list if col in notes.columns)
            .unique()
            .sort(self.global_icu_stay_id_col, self.note_time_col)
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
