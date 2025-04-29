# Author: Finn Fassbender
# Last modified: 2024-09-11

# Description: This script imputes the data to remove missing values.
# It is available as a module for piping in the main script.
# It can be called with command line arguments to specify the source datasets to be imputed. ! NOT IMPLEMENTED YET !

import argparse
import polars as pl

from helpers.helper import GlobalVars


class MedicationImputer(GlobalVars):
    def __init__(self, paths, patient_info_location: str) -> None:
        super().__init__(paths)
        self.patient_info_location = patient_info_location

    def add_common_rate(self, data) -> pl.LazyFrame:
        """
        Adds a common rate column to the medications data with rates in mcg/kg/hr.
        """

        weights = pl.scan_parquet(self.patient_info_location).select(
            self.global_icu_stay_id_col,
            self.weight_col,
        )

        return (
            data.join(weights, on=self.global_icu_stay_id_col, how="left")
            .with_columns(
                pl.when(pl.col(self.drug_rate_unit_col) == "mcg/hr")
                .then(pl.col(self.drug_rate_col) / pl.col(self.weight_col))
                .when(pl.col(self.drug_rate_unit_col) == "mcg/kg/hr")
                .then(pl.col(self.drug_rate_col))
                .when(pl.col(self.drug_rate_unit_col) == "mcg/min")
                .then(pl.col(self.drug_rate_col) / pl.col(self.weight_col) * 60)
                .when(pl.col(self.drug_rate_unit_col) == "mcg/kg/min")
                .then(pl.col(self.drug_rate_col) * 60)
                .when(pl.col(self.drug_rate_unit_col) == "mg/day")
                .then(
                    pl.col(self.drug_rate_col)
                    / pl.col(self.weight_col)
                    / 24
                    * 1000
                )
                .when(pl.col(self.drug_rate_unit_col) == "mg/hr")
                .then(
                    pl.col(self.drug_rate_col) / pl.col(self.weight_col) * 1000
                )
                .when(pl.col(self.drug_rate_unit_col) == "mg/kg/hr")
                .then(pl.col(self.drug_rate_col) * 1000)
                .when(pl.col(self.drug_rate_unit_col) == "mg/min")
                .then(
                    pl.col(self.drug_rate_col)
                    / pl.col(self.weight_col)
                    * 60
                    * 1000
                )
                .otherwise(None)
                .round(2)
                .alias(self.drug_rate_common_col)
            )
            .select(
                self.global_icu_stay_id_col,
                self.drug_ingredient_col,
                self.drug_name_col,
                self.drug_name_OMOP_col,
                self.drug_class_col,
                self.drug_admin_route_col,
                self.drug_amount_col,
                self.drug_amount_unit_col,
                self.drug_rate_col,
                self.drug_rate_unit_col,
                self.drug_rate_common_col,
                self.drug_start_col,
                self.drug_end_col,
            )
            .cast(
                {
                    self.drug_amount_col: float,
                    self.drug_rate_col: float,
                    self.drug_rate_common_col: float,
                    self.drug_class_col: str,
                    self.drug_admin_route_col: str,
                },
                strict=False,
            )
            .unique()
            .sort(self.global_icu_stay_id_col, self.drug_start_col)
        )


if __name__ == "__main__":
    raise NotImplementedError(
        "This script is not yet implemented as a command line tool."
    )
