# Author: Finn Fassbender
# Last modified: 2025-10-30

# Description: This script imputes the data to remove missing values.
# It is available as a module for piping in the main script.
# It can be called with command line arguments to specify the source datasets to be imputed. ! NOT IMPLEMENTED YET !

import polars as pl

from ..helper import GlobalVars


class MedicationImputer(GlobalVars):
    def __init__(self, paths, patient_info_location: str) -> None:
        super().__init__(paths)
        self.patient_info_location = patient_info_location

    def add_common_rate(self, data) -> pl.LazyFrame:
        """
        Standardize medication rates to mcg/kg/hr units.

        Steps:
            1. Load patient weights.
            2. Join weights to medication data.
            3. Convert all rate units to mcg/kg/hr via unit-specific formulas.
            4. Select core medication columns and cast types.
            5. Remove duplicates and sort by ICU stay and start time.

        Returns:
            pl.LazyFrame: Contains columns:
                - {global_icu_stay_id_col}: Global ICU stay identifier.
                - {drug_ingredient_col}: Drug ingredient.
                - {drug_name_col}: Drug name.
                - {drug_name_OMOP_col}: OMOP-mapped drug name.
                - {drug_class_col}: Drug class.
                - {drug_admin_route_col}: Administration route.
                - {drug_amount_col}: Dose amount (float).
                - {drug_amount_unit_col}: Dose unit.
                - {drug_rate_col}: Original rate (float).
                - {drug_rate_unit_col}: Original rate unit.
                - {drug_rate_common_col}: Standardized rate (mcg/kg/hr, float).
                - {drug_start_col}: Medication start time.
                - {drug_end_col}: Medication end time.
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
