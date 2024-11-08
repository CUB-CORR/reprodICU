# Author: Finn Fassbender
# Last modified: 2024-09-11

# Description: This script imputes the data to remove missing values.
# It is available as a module for piping in the main script.
# It can be called with command line arguments to specify the source datasets to be imputed. ! NOT IMPLEMENTED YET !

import argparse
import polars as pl
import numpy as np

from helpers.helper import GlobalVars


class PatientInformationCleaner(GlobalVars):
    def __init__(self, paths) -> None:
        super().__init__(paths)
        pass

    def clean_patient_information(self, data) -> pl.LazyFrame:
        """
        Cleans the height, weight, duration data by rounding the values.
        """

        return data.with_columns(
            # Round the height and weight to the nearest integer
            pl.col(self.height_col).round(decimals=0).cast(int),
            pl.col(self.weight_col).round(decimals=0).cast(int),
            # Round the stay durations to the nearest 4 significant digits
            # 4 significant digits are chosen to keep the data at about minute resolution
            pl.col(self.pre_icu_length_of_stay_col).round(decimals=4),
            pl.col(self.icu_length_of_stay_col).round(decimals=4),
            pl.col(self.hospital_length_of_stay_col).round(decimals=4),
            # Round the mortality after X days to the nearest integer
            pl.col(self.mortality_after_col).round(decimals=0).cast(int),
        )

    def remove_bad_patient_information(self, data) -> pl.LazyFrame:
        """
        Removes obviously wrong values from the patient information.
        """

        return (
            data.with_columns(
                # Remove negative / zero values for mortality after ICU discharge
                # when patient died in ICU
                pl.when(pl.col(self.mortality_icu_col))
                .then(None)
                .otherwise(pl.col(self.mortality_after_col))
                .alias(self.mortality_after_col),
            )
            # Define the order of the columns
            .select(
                self.global_person_id_col,
                self.global_hospital_stay_id_col,
                self.global_icu_stay_id_col,
                self.icu_stay_seq_num_col,
                self.database_col,
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
            )
        )

    def add_good_patient_information(self, data) -> pl.LazyFrame:
        """
        Adds information that can easily be derived from the existing data.
        """

        return data.with_columns(
            # Add missing values for the ICU mortality if the patient survived
            pl.when(
                pl.col(self.mortality_icu_col).is_null(),
                pl.col(self.mortality_after_col) > 1,
            )
            .then(False)
            .otherwise(pl.col(self.mortality_icu_col))
            .alias(self.mortality_icu_col),
            # Add missing values for the Hospital mortality if the patient died in the ICU
            pl.when(
                pl.col(self.mortality_hosp_col).is_null(),
                pl.col(self.mortality_icu_col).cast(bool),
            )
            .then(pl.col(self.mortality_icu_col))
            .otherwise(pl.col(self.mortality_hosp_col))
            .alias(self.mortality_hosp_col),
        ).with_columns(
            # Add missing values for the Hospital and ICU mortality
            # if the patient died long after the ICU discharge
            pl.when(
                pl.col(self.mortality_hosp_col).is_null(),
                pl.col(self.mortality_after_col)
                > (
                    pl.col(self.hospital_length_of_stay_col)
                    - pl.col(self.icu_length_of_stay_col)
                ),
            )
            .then(False)
            .otherwise(pl.col(self.mortality_hosp_col))
            .alias(self.mortality_hosp_col),
            pl.when(
                pl.col(self.mortality_icu_col).is_null(),
                pl.col(self.mortality_after_col) > 1,
            )
            .then(False)
            .otherwise(pl.col(self.mortality_icu_col))
            .alias(self.mortality_icu_col),
        )

    def add_data_availability_information(self, data) -> pl.LazyFrame:
        """
        Adds information about the availability of the data in the other
        tables of the dataset.
        """

        return data.with_columns(
            # Add columns for the availability of the data
            # If ICU Stay ID is present in the other tables, the data is available
        )


if __name__ == "__main__":
    raise NotImplementedError(
        "This script is not yet implemented as a command line tool."
    )
