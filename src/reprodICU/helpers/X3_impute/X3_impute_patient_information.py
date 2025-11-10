# Author: Finn Fassbender
# Last modified: 2025-10-30

# Description: This script imputes the data to remove missing values.
# It is available as a module for piping in the main script.
# It can be called with command line arguments to specify the source datasets to be imputed. ! NOT IMPLEMENTED YET !

import numpy as np
import polars as pl
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import OrdinalEncoder

from ..helper import GlobalVars


class PatientInformationImputer(GlobalVars):
    def __init__(self, paths) -> None:
        super().__init__(paths)
        pass

    def impute_patient_IDs(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """
        Impute missing global identifiers from hierarchical levels.

        Steps:
            1. Propagate ICU stay ID to missing hospital stay IDs.
            2. Propagate hospital stay ID to missing person IDs.

        Returns:
            pl.LazyFrame: Patient information with filled ID columns.
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
        self, data: pl.DataFrame, n_neighbors: int = 2
    ) -> pl.DataFrame:
        """
        Impute missing anthropometric data via iterative imputation.

        Steps:
            1. Select columns: age, height, weight (to impute) and demographic/site features.
            2. Encode categorical columns (dataset, gender, ethnicity, care_site, unit_type) numerically.
            3. Apply iterative imputation.
            4. Cast imputed values back to original types.
            5. Drop height for neonatal patients (unreliable).

        Returns:
            pl.LazyFrame: Contains columns:
                - {age_col}: Patient age (years).
                - {height_col}: Patient height (cm).
                - {weight_col}: Patient weight (kg).
                - [All other original columns]
        """

        # columns to impute and their post-processing functions
        post_process = {
            self.age_col: lambda expr: expr.cast(int),
            self.height_col: lambda expr: expr.round(decimals=1),
            self.weight_col: lambda expr: expr.round(decimals=1),
        }
        columns_to_impute = list(post_process.keys())

        # categorical columns
        categorical_cols = [
            self.dataset_col,
            self.gender_col,
            self.ethnicity_col,
            self.care_site_col,
            self.unit_type_col,
        ]

        # get relevant columns for nearest neighbors
        columns_for_neighbors = columns_to_impute + categorical_cols

        # get data for imputation
        imputation_data = data.select(columns_for_neighbors).to_pandas()

        # encode categorical columns
        encoders = {}
        for col in categorical_cols:
            # fill NaN with 'Unknown' to handle missing categoricals
            imputation_data[col] = imputation_data[col].fillna("Unknown")
            encoder = OrdinalEncoder()
            imputation_data[col] = encoder.fit_transform(imputation_data[[col]])
            encoders[col] = encoder

        # impute missing values
        print("reprodICU - Imputing patient information...")
        imputer = IterativeImputer(
            estimator=BayesianRidge(),  # or LinearRegression()
            max_iter=10,
            random_state=42,
            verbose=1,
        )
        imputed_data = imputer.fit_transform(imputation_data)
        imputed_data = pl.DataFrame(
            imputed_data,
            schema=columns_for_neighbors,
        ).select(*columns_to_impute)

        return data.with_columns(
            pl.when(
                pl.col(col).is_null(),
                pl.col(self.unit_type_col) != "Neonatal intensive care unit",
            )
            .then(imputed_data[col])
            .otherwise(pl.col(col))
            .pipe(post_process[col])
            .alias(col)
            for col in columns_to_impute
        )
