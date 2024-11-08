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

        return data.with_columns(
            # Remove negative / zero values for mortality after ICU discharge
            # when patient died in ICU
            pl.when(pl.col(self.mortality_icu_col))
            .then(None)
            .otherwise(pl.col(self.mortality_after_col))
            .alias(self.mortality_after_col),
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
