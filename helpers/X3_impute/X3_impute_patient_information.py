# Author: Finn Fassbender
# Last modified: 2024-09-11

# Description: This script imputes the data to remove missing values.
# It is available as a module for piping in the main script.
# It can be called with command line arguments to specify the source datasets to be imputed. ! NOT IMPLEMENTED YET !

import argparse
import polars as pl
import numpy as np

from sklearn.impute import KNNImputer

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
            pl.when(pl.col(self.global_hospital_stay_id_col).is_null())
            .then(pl.col(self.global_icu_stay_id_col))
            .otherwise(pl.col(self.global_hospital_stay_id_col))
            .alias(self.global_hospital_stay_id_col),
            # Add missing person IDs
            pl.when(pl.col(self.global_person_id_col).is_null())
            .then(pl.col(self.global_hospital_stay_id_col))
            .otherwise(pl.col(self.global_person_id_col))
            .alias(self.global_person_id_col),
        )

    def impute_patient_anthropometrics(
        self, data: pl.LazyFrame, n_neighbors: int = 2
    ) -> pl.LazyFrame:
        """
        Imputes missing anthropometric data.
        Anthropometric data is imputed using the KNN algorithm.

        Anthropometric data includes:
        - age
        - height
        - weight

        :param data: DataFrame with the data
        """

        # get relevant columns for imputation
        columns_to_impute = [
            self.age_col,
            self.height_col,
            self.weight_col,
        ]

        # get relevant columns for nearest neighbors
        columns_for_neighbors = [
            self.age_col,
            self.height_col,
            self.weight_col,
            # not imputed, but used for nearest neighbors
            self.dataset_col,
            self.gender_col,
            self.ethnicity_col,
            self.care_site_col,
            self.unit_type_col,  # to ensure Newborns are not imputed with adult values
        ]

        # function for replacing categorical values with numerical values
        def replace_categorical_with_numerical(
            column, col_name: str
        ) -> pl.LazyFrame:
            return column.replace(
                data.select(col_name).collect().to_series().unique(),
                np.arange(
                    data.select(col_name).collect().to_series().unique().len()
                ),
            )

        # get data for imputation
        data_for_imputation = (
            data.select(columns_for_neighbors)
            .with_columns(
                pl.col(self.dataset_col)
                .pipe(replace_categorical_with_numerical, self.dataset_col)
                .alias(self.dataset_col),
                pl.col(self.gender_col)
                .cast(str)
                .replace(
                    self.gender_dtype.categories.to_list(),
                    np.arange(self.gender_dtype.categories.len()),
                    return_dtype=int,
                )
                .alias(self.gender_col),
                pl.col(self.ethnicity_col)
                .cast(str)
                .replace(
                    self.ethnicity_dtype.categories.to_list(),
                    np.arange(self.ethnicity_dtype.categories.len()),
                    return_dtype=int,
                )
                .alias(self.ethnicity_col),
                pl.col(self.care_site_col)
                .pipe(replace_categorical_with_numerical, self.care_site_col)
                .alias(self.care_site_col),
                pl.col(self.unit_type_col)
                .cast(str)
                .pipe(replace_categorical_with_numerical, self.unit_type_col)
                .alias(self.unit_type_col),
            )
            .collect()
            .to_pandas()
        )

        # impute missing values
        print("reprodICU - Imputing patient information...")
        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed_data = imputer.fit_transform(data_for_imputation)
        imputed_data = pl.DataFrame(
            imputed_data,
            schema=columns_for_neighbors,
        ).select(*columns_to_impute)

        return data.with_columns(
            pl.when(pl.col(self.age_col).is_null())
            .then(imputed_data[self.age_col])
            .otherwise(pl.col(self.age_col))
            .cast(int)
            .alias(self.age_col),
            pl.when(pl.col(self.height_col).is_null())
            .then(imputed_data[self.height_col])
            .otherwise(pl.col(self.height_col))
            .round(decimals=1)
            .alias(self.height_col),
            pl.when(pl.col(self.weight_col).is_null())
            .then(imputed_data[self.weight_col])
            .otherwise(pl.col(self.weight_col))
            .round(decimals=1)
            .alias(self.weight_col),
        ).with_columns(
            # Drop heights for Newborns since there is no reliable data available
            pl.when(pl.col(self.unit_type_col) == "Neonatal")
            .then(None)
            .otherwise(pl.col(self.height_col))
            .alias(self.height_col),
        )


if __name__ == "__main__":
    raise NotImplementedError(
        "This script is not yet implemented as a command line tool."
    )
