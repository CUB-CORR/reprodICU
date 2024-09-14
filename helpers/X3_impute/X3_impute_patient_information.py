# Author: Finn Fassbender
# Last modified: 2024-09-11

# Description: This script imputes the data to remove missing values.
# It is available as a module for piping in the main script.
# It can be called with command line arguments to specify the source datasets to be imputed. ! NOT IMPLEMENTED YET !

import argparse
import polars as pl

from helpers.helper import GlobalVars


class PatientInformationImputer(GlobalVars):
    def __init__(self, paths) -> None:
        super().__init__(paths)
        pass

    def impute_patient_IDs(self, data) -> pl.LazyFrame:
        """
        Imputes missing patient information.
        For missing IDs, new IDs are generated / IDs are assigned from a lower level.
        """

        return data.with_columns(
            # Add missing hospital stay IDs
            pl.when(pl.col(self.hospital_stay_id_col) == None)
            .then(pl.col(self.icu_stay_id_col))
            .otherwise(pl.col(self.hospital_stay_id_col))
            .alias(self.hospital_stay_id_col),
            # Add missing person IDs
            pl.when(pl.col(self.person_id_col) == None)
            .then(pl.col(self.hospital_stay_id_col))
            .otherwise(pl.col(self.person_id_col))
            .alias(self.person_id_col),
        ).unique()


if __name__ == "__main__":
    raise NotImplementedError(
        "This script is not yet implemented as a command line tool."
    )
