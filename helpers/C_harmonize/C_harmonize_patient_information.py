# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script combines the preprocessed patient information from the differet
# databases into one common table

import polars as pl

from helpers.A_extract.A_extract_eicu import EICUExtractor
from helpers.A_extract.AX_extract_hirid import HiRIDExtractor
from helpers.A_extract.A_extract_mimic3 import MIMIC3Extractor
from helpers.A_extract.A_extract_mimic4 import MIMIC4Extractor
from helpers.A_extract.A_extract_nwicu import NWICUExtractor
from helpers.A_extract.AX_extract_sicdb import SICdbExtractor
from helpers.A_extract.AX_extract_umcdb import UMCdbExtractor
from helpers.helper import GlobalVars


class PatientInformationHarmonizer(GlobalVars):
    def __init__(self, paths, datasets: list, DEMO=False):
        """
        Initializes the PatientInformationHarmonizer class with the given paths and datasets.

        Args:
            paths (str): The file paths required for data extraction.
            datasets (list): A list of datasets to be harmonized.
            DEMO (bool, optional): A flag indicating whether to use demo data. Defaults to False.
        """
        super().__init__(paths)
        self.eicu = EICUExtractor(paths, DEMO)
        self.hirid = HiRIDExtractor(paths)
        self.mimic3 = MIMIC3Extractor(paths, DEMO)
        self.mimic4 = MIMIC4Extractor(paths, DEMO)
        self.nwicu = NWICUExtractor(paths)
        self.sicdb = SICdbExtractor(paths)
        self.umcdb = UMCdbExtractor(paths)
        self.datasets = datasets

    def harmonize_patient_information(self) -> pl.LazyFrame:
        """
        Harmonizes patient information from multiple databases into a single LazyFrame.

        This function performs the following steps:
            1. Validates that a non-empty list of datasets is provided; raises a ValueError if empty.
            2. Initializes an empty list to accumulate patient information datasets.
            3. For each dataset in {datasets}:
               - Extracts patient information using the corresponding extractor.
               - Uses a helper method (_concat_helper1 or _concat_helper2) to concatenate database-specific identifiers with a prefix to form a global identifier.
               - Adds a constant column {dataset_col} with an alias indicating the originating dataset.
            4. Concatenates all the patient information datasets using a "diagonal_relaxed" join.
            5. Selects the desired columns in a pre-defined order.
            6. Applies a cast operation to ensure the correct data types for each column.

        The final returned LazyFrame contains the following columns:
            - {global_person_id_col}: Unique global person identifier.
            - {global_hospital_stay_id_col}: Unique global hospital stay identifier.
            - {global_icu_stay_id_col}: Unique global ICU stay identifier.
            - {icu_stay_seq_num_col}: Sequence number for the ICU stay.
            - {dataset_col}: Identifier for the source dataset.
            - {person_id_col}: Original person identifier (per dataset).
            - {hospital_stay_id_col}: Original hospital stay identifier.
            - {icu_stay_id_col}: Original ICU stay identifier.
            - {age_col}: Patient age.
            - {gender_col}: Patient gender.
            - {height_col}: Patient height (in cm or m as defined).
            - {weight_col}: Patient weight (in kg).
            - {ethnicity_col}: Patient ethnicity.
            - {admission_diagnosis_col}: Diagnosis at admission.
            - {admission_type_col}: Type of admission.
            - {admission_urgency_col}: Urgency level of the admission.
            - {admission_time_col}: Time of admission.
            - {admission_loc_col}: Admission location.
            - {specialty_col}: Medical specialty relevant to the admission.
            - {care_site_col}: Hospital care site.
            - {unit_type_col}: Type of hospital unit.
            - {pre_icu_length_of_stay_col}: Length of stay before ICU (in days).
            - {icu_length_of_stay_col}: ICU length of stay (in days).
            - {hospital_length_of_stay_col}: Total hospital length of stay (in days).
            - {discharge_loc_col}: Location at discharge.
            - {mortality_hosp_col}: Hospital mortality indicator (True/False).
            - {mortality_icu_col}: ICU mortality indicator (True/False).
            - {mortality_after_col}: Post-discharge mortality (in days).

        Returns:
            pl.LazyFrame: A LazyFrame containing harmonized patient information with the columns listed above and the correct data types.

        Raises:
            ValueError: If no datasets are provided.
        """
        if self.datasets == []:
            raise ValueError(
                "No datasets to harmonize the patient information from."
            )

        patient_information_datasets = []

        if "eICU" in self.datasets:
            patient_information_datasets.append(
                self.eicu.extract_patient_information()
                .pipe(self._concat_helper1, "eicu-")
                .with_columns(pl.lit("eICU-CRD").alias(self.dataset_col))
            )

        if "HiRID" in self.datasets:
            patient_information_datasets.append(
                self.hirid.extract_patient_information()
                .pipe(self._concat_helper2, "hirid-")
                .with_columns(pl.lit("HiRID").alias(self.dataset_col))
            )

        if "MIMIC3" in self.datasets:
            patient_information_datasets.append(
                self.mimic3.extract_patient_information()
                .pipe(self._concat_helper1, "mimic3-")
                .with_columns(pl.lit("MIMIC-III").alias(self.dataset_col))
            )

        if "MIMIC4" in self.datasets:
            patient_information_datasets.append(
                self.mimic4.extract_patient_information()
                .pipe(self._concat_helper1, "mimic4-")
                .with_columns(pl.lit("MIMIC-IV").alias(self.dataset_col))
            )

        if "NWICU" in self.datasets:
            patient_information_datasets.append(
                self.nwicu.extract_patient_information()
                .pipe(self._concat_helper1, "nwicu-")
                .with_columns(pl.lit("NWICU").alias(self.dataset_col))
            )

        if "SICdb" in self.datasets:
            patient_information_datasets.append(
                self.sicdb.extract_patient_information()
                .pipe(self._concat_helper1, "sicdb-")
                .with_columns(pl.lit("SICdb").alias(self.dataset_col))
            )

        if "UMCdb" in self.datasets:
            patient_information_datasets.append(
                self.umcdb.extract_patient_information()
                .pipe(self._concat_helper1, "umcdb-")
                .with_columns(pl.lit("AmsterdamUMCdb").alias(self.dataset_col))
            )

        patient_information = pl.concat(
            patient_information_datasets, how="diagonal_relaxed"
        )
        patient_information_cols_list = [
            self.global_person_id_col,
            self.global_hospital_stay_id_col,
            self.global_icu_stay_id_col,
            self.icu_stay_seq_num_col,
            self.icu_time_rel_to_first_col,
            self.dataset_col,
            self.person_id_col,
            self.hospital_stay_id_col,
            self.icu_stay_id_col,
            self.age_col,
            self.gender_col,
            self.height_col,
            self.weight_col,
            self.ethnicity_col,
            self.admission_diagnosis_col,
            self.admission_type_col,
            self.admission_urgency_col,
            self.admission_time_col,
            self.admission_loc_col,
            self.specialty_col,
            self.care_site_col,
            self.unit_type_col,
            self.pre_icu_length_of_stay_col,
            self.icu_length_of_stay_col,
            self.hospital_length_of_stay_col,
            self.discharge_loc_col,
            self.mortality_hosp_col,
            self.mortality_icu_col,
            self.mortality_after_col,
            self.mortality_after_cutoff_col,
        ]

        return (
            patient_information
            # Define the data types of the columns
            .cast(
                {
                    self.global_person_id_col: str,
                    self.global_hospital_stay_id_col: str,
                    self.global_icu_stay_id_col: str,
                    self.icu_stay_seq_num_col: int,
                    self.icu_time_rel_to_first_col: int,
                    self.person_id_col: str,
                    self.hospital_stay_id_col: str,
                    self.icu_stay_id_col: str,
                    self.age_col: float,
                    self.gender_col: self.gender_dtype,
                    self.height_col: float,
                    self.weight_col: float,
                    self.ethnicity_col: self.ethnicity_dtype,
                    self.admission_type_col: self.admission_types_dtype,
                    self.admission_urgency_col: self.admission_urgency_dtype,
                    self.admission_loc_col: self.admission_locations_dtype,
                    self.care_site_col: str,
                    self.unit_type_col: self.unit_types_dtype,
                    self.pre_icu_length_of_stay_col: float,
                    self.icu_length_of_stay_col: float,
                    self.hospital_length_of_stay_col: float,
                    self.discharge_loc_col: self.discharge_locations_dtype,
                    self.mortality_hosp_col: bool,
                    self.mortality_icu_col: bool,
                    self.mortality_after_col: int,
                    self.mortality_after_cutoff_col: int,
                },
                strict=False,
            )
            # Define the order of the columns
            .select(
                col
                for col in patient_information_cols_list
                if col in patient_information.columns
            ).unique()
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
