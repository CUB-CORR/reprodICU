# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script combines the preprocessed patient information from the differet
# databases into one common table

import numpy as np
import pandas as pd
import polars as pl
import os.path

from helpers.A_extract.A_extract_eicu import EICUExtractor
from helpers.A_extract.AX_extract_hirid import HiRIDExtractor
from helpers.A_extract.A_extract_mimic3 import MIMIC3Extractor
from helpers.A_extract.A_extract_mimic4 import MIMIC4Extractor
from helpers.A_extract.AX_extract_sicdb import SICdbExtractor
from helpers.A_extract.AX_extract_umcdb import UMCdbExtractor
from helpers.helper import GlobalVars


class PatientInformationHarmonizer(GlobalVars):
    def __init__(self, paths, datasets: list, DEMO=False):
        super().__init__(paths)
        self.eicu = EICUExtractor(paths, DEMO)
        self.hirid = HiRIDExtractor(paths)
        self.mimic3 = MIMIC3Extractor(paths, DEMO)
        self.mimic4 = MIMIC4Extractor(paths, DEMO)
        self.sicdb = SICdbExtractor(paths)
        self.umcdb = UMCdbExtractor(paths)
        self.datasets = datasets

    def harmonize_patient_information(self) -> pl.LazyFrame:

        if self.datasets == []:
            raise ValueError(
                "No datasets to harmonize the patient information from."
            )

        patient_information_datasets = []

        if "eICU" in self.datasets:
            patient_information_datasets.append(
                self.eicu.extract_patient_information()
                .pipe(self._concat_helper1, "eicu-")
                .with_columns(pl.lit("eICU").alias(self.database_col))
            )

        if "HiRID" in self.datasets:
            patient_information_datasets.append(
                self.hirid.extract_patient_information()
                .pipe(self._concat_helper2, "hirid-")
                .with_columns(pl.lit("HiRID").alias(self.database_col))
            )

        if "MIMIC3" in self.datasets:
            patient_information_datasets.append(
                self.mimic3.extract_patient_information()
                .pipe(self._concat_helper1, "mimic3-")
                .with_columns(pl.lit("MIMIC-III").alias(self.database_col))
            )

        if "MIMIC4" in self.datasets:
            patient_information_datasets.append(
                self.mimic4.extract_patient_information()
                .pipe(self._concat_helper1, "mimic4-")
                .with_columns(pl.lit("MIMIC-IV").alias(self.database_col))
            )

        if "SICdb" in self.datasets:
            patient_information_datasets.append(
                self.sicdb.extract_patient_information()
                .pipe(self._concat_helper1, "sicdb-")
                .with_columns(pl.lit("SICdb").alias(self.database_col))
            )

        if "UMCdb" in self.datasets:
            patient_information_datasets.append(
                self.umcdb.extract_patient_information()
                .pipe(self._concat_helper1, "umcdb-")
                .with_columns(pl.lit("AmsterdamUMCdb").alias(self.database_col))
            )

        return (
            pl.concat(
                patient_information_datasets,
                how="diagonal_relaxed",
            )
            # Define the order of the columns
            .select(
                [
                    self.global_person_id_col,
                    self.global_hospital_stay_id_col,
                    self.global_icu_stay_id_col,
                    self.database_col,
                    self.person_id_col,
                    self.hospital_stay_id_col,
                    self.icu_stay_id_col,
                    self.age_col,
                    self.gender_col,
                    self.height_col,
                    self.weight_col,
                    self.ethnicity_col,
                    self.admission_type_col,
                    self.specialty_col,
                    self.admission_loc_col,
                    self.care_site_col,
                    self.unit_type_col,
                    self.pre_icu_length_of_stay_col,
                    self.icu_length_of_stay_col,
                    self.discharge_loc_col,
                    self.mortality_hosp_col,
                    self.mortality_icu_col,
                    self.mortality_after_col,
                ]
            ).unique()
            # Define the data types of the columns
            .cast(
                {
                    self.global_person_id_col: str,
                    self.global_hospital_stay_id_col: str,
                    self.global_icu_stay_id_col: str,
                    self.person_id_col: str,
                    self.hospital_stay_id_col: str,
                    self.icu_stay_id_col: str,
                    self.age_col: float,
                    self.gender_col: self.gender_dtype,
                    self.height_col: float,
                    self.weight_col: float,
                    self.ethnicity_col: self.ethnicity_dtype,
                    # self.admission_type_col: str,
                    self.admission_loc_col: self.admission_locations_dtype,
                    self.care_site_col: str,
                    self.unit_type_col: self.unit_types_dtype,
                    self.pre_icu_length_of_stay_col: float,
                    self.icu_length_of_stay_col: float,
                    self.discharge_loc_col: self.discharge_locations_dtype,
                    self.mortality_hosp_col: bool,
                    self.mortality_icu_col: bool,
                    self.mortality_after_col: float,
                }
            )
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

    # HiRID does not have a person_id or a hospital_stay_id column
    def _concat_helper2(self, data: pl.LazyFrame, name: str) -> pl.LazyFrame:
        return data.with_columns(
            pl.concat_str([pl.lit(name), pl.col(self.icu_stay_id_col)]).alias(
                self.global_icu_stay_id_col
            ),
        )
