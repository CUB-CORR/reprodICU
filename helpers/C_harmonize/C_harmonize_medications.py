# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script combines the preprocessed medications from the different
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
from helpers.helper import GlobalHelpers

SECONDS_IN_1MIN = 60
SECONDS_IN_1H = 3600


class MedicationHarmonizer(GlobalVars):
    def __init__(self, paths, datasets: list, DEMO=False):
        """
        Initializes the MedicationHarmonizer class with the given paths and datasets.

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
        self.helpers = GlobalHelpers()
        self.datasets = datasets

    def harmonize_medications(self) -> pl.LazyFrame:
        """
        Harmonizes medication data from multiple databases into a single table.

        This function performs the following steps:
            1. Validates that datasets are provided; raises a ValueError if empty.
            2. Loads mapping files for fluids and drugs classes:
               - {fluids_class_mapping}: Maps medication names to fluid classes.
               - {drugs_class_mapping}: Maps drug ingredients to drug classes.
            3. Iterates over each dataset provided in {datasets}:
               - Extracts medication data using dataset-specific extractors.
               - Applies helper methods to concatenate and print unique case information.
            4. Concatenates all medication datasets using a relaxed diagonal join.
            5. Adds missing drug class information by replacing columns:
               - Updates {drug_class_col} based on matches from {drug_name_col} and {drug_ingredient_col} using the mappings.
            6. Harmonizes units in {drug_amount_unit_col} by normalizing common differences (e.g., 'mL' to 'ml', 'µ' to 'mc').
            7. Selects and casts the following columns:
               - {global_icu_stay_id_col}: Global ICU stay identifier.
               - {drug_ingredient_col}: Drug ingredient.
               - {drug_name_col}: Original medication name.
               - {drug_name_OMOP_col}: Medications mapped to OMOP convention.
               - {drug_class_col}: Drug/medication classification.
               - {drug_admin_route_col}: Administration route.
               - {drug_amount_col}: Amount of drug administered (float).
               - {drug_amount_unit_col}: Unit for amount (normalized).
               - {drug_rate_col}: Rate of administration (float).
               - {drug_rate_unit_col}: Unit for the rate.
               - {fluid_amount_col}: Fluid amount administered (float).
               - {fluid_rate_col}: Fluid rate administered (float).
               - {drug_start_col}: Start time of medication.
               - {drug_end_col}: End time of medication.
               - {drug_patient_weight_col}: Patient weight used for dosing (float).
            8. Returns a unique and sorted pl.LazyFrame sorted by {global_icu_stay_id_col} and {drug_start_col}.

        Returns:
            pl.LazyFrame: A lazy frame containing the harmonized medication data with columns:
                - {global_icu_stay_id_col}: Global ICU stay identifier.
                - {drug_ingredient_col}: Drug ingredient.
                - {drug_name_col}: Original medication name.
                - {drug_name_OMOP_col}: OMOP mapped medication name.
                - {drug_class_col}: Medication class.
                - {drug_admin_route_col}: Route of administration.
                - {drug_amount_col}: Medication amount.
                - {drug_amount_unit_col}: Normalized unit for amount.
                - {drug_rate_col}: Medication rate.
                - {drug_rate_unit_col}: Unit for rate.
                - {fluid_amount_col}: Fluid amount.
                - {fluid_rate_col}: Fluid rate.
                - {drug_start_col}: Medication start time.
                - {drug_end_col}: Medication end time.
                - {drug_patient_weight_col}: Patient weight.

        Raises:
            ValueError: If no datasets are provided.
        """
        if self.datasets == []:
            raise ValueError("No datasets to harmonize the medications from.")

        fluids_class_mapping = self.helpers.load_many_to_one_mapping(
            self.mapping_path + "MEDICATIONS_FLUIDS_CLASSES.yaml"
        )
        drugs_class_mapping = self.helpers.load_mapping(
            self.mapping_path + "MEDICATIONS_DRUGS_CLASSES.yaml"
        )

        medications_datasets = []

        if "eICU" in self.datasets:
            medications_datasets.append(
                self.eicu.extract_medications()
                .pipe(self._concat_helper, "eicu-")
                .pipe(self._print_unique_cases, "eICU")
            )

        if "HiRID" in self.datasets:
            medications_datasets.append(
                self.hirid.extract_medications()
                .pipe(self._concat_helper, "hirid-")
                .pipe(self._print_unique_cases, "HiRID")
            )

        if "MIMIC3" in self.datasets:
            medications_datasets.append(
                self.mimic3.extract_medications()
                .pipe(self._concat_helper, "mimic3-")
                .pipe(self._print_unique_cases, "MIMIC3")
            )

        if "MIMIC4" in self.datasets:
            medications_datasets.append(
                self.mimic4.extract_medications()
                .pipe(self._concat_helper, "mimic4-")
                .pipe(self._print_unique_cases, "MIMIC4")
            )

        if "NWICU" in self.datasets:
            medications_datasets.append(
                self.nwicu.extract_medications()
                .pipe(self._concat_helper, "nwicu-")
                .pipe(self._print_unique_cases, "NWICU")
            )

        if "SICdb" in self.datasets:
            medications_datasets.append(
                self.sicdb.extract_medications()
                .pipe(self._concat_helper, "sicdb-")
                .pipe(self._print_unique_cases, "SICdb")
            )

        if "UMCdb" in self.datasets:
            medications_datasets.append(
                self.umcdb.extract_medications()
                .pipe(self._concat_helper, "umcdb-")
                .pipe(self._print_unique_cases, "UMCdb")
            )

        medications = pl.concat(
            medications_datasets,
            how="diagonal_relaxed",
        )
        medications_cols_list = [
            self.global_icu_stay_id_col,
            self.drug_mixture_id_col,
            self.drug_mixture_admin_id_col,
            self.drug_ingredient_col,
            self.drug_name_col,
            self.drug_name_OMOP_col,
            self.drug_class_col,
            self.drug_continous_col,
            self.drug_admin_route_col,
            self.drug_amount_col,
            self.drug_amount_unit_col,
            self.drug_rate_col,
            self.drug_rate_unit_col,
            self.fluid_group_col,
            self.fluid_name_col,
            self.fluid_amount_col,
            self.fluid_rate_col,
            self.drug_start_col,
            self.drug_end_col,
            self.drug_patient_weight_col,
        ]

        # Add missing columns as null columns
        medications = medications.with_columns(
            pl.lit(None).alias(col)
            for col in medications_cols_list
            if col not in medications.columns
        )

        return (
            medications.cast(
                {
                    self.drug_name_col: str,
                    self.drug_ingredient_col: str,
                    self.drug_amount_col: float,
                    self.drug_rate_col: float,
                    self.fluid_amount_col: float,
                    self.fluid_rate_col: float,
                    self.drug_patient_weight_col: float,
                    self.drug_class_col: str,
                    self.drug_admin_route_col: str,
                    self.drug_start_col: float,
                    self.drug_end_col: float,
                },
                strict=False,
            )
            # add missing drug rates
            .with_columns(
                pl.when(
                    pl.all_horizontal(
                        pl.col(self.drug_amount_col).is_not_null(),
                        pl.col(self.drug_amount_unit_col).is_in(
                            ["mcg", "mg", "g", "units"]
                        ),
                        pl.col(self.drug_start_col).is_not_null(),
                        pl.col(self.drug_end_col).is_not_null(),
                        pl.col(self.drug_continous_col),
                    )
                    & pl.col(self.drug_rate_col).is_null()
                )
                .then(
                    pl.col(self.drug_amount_col)
                    / pl.when(
                        pl.col(self.drug_patient_weight_col).is_not_null(),
                        pl.col(self.drug_amount_unit_col) != "units",
                    )
                    .then(pl.col(self.drug_patient_weight_col))
                    .otherwise(1)
                    / (pl.col(self.drug_end_col) - pl.col(self.drug_start_col))
                    * SECONDS_IN_1MIN
                )
                .otherwise(pl.col(self.drug_rate_col))
                .alias(self.drug_rate_col),
                pl.when(
                    pl.all_horizontal(
                        pl.col(self.drug_amount_col).is_not_null(),
                        pl.col(self.drug_amount_unit_col).is_in(
                            ["mcg", "mg", "g", "units"]
                        ),
                        pl.col(self.drug_start_col).is_not_null(),
                        pl.col(self.drug_end_col).is_not_null(),
                        pl.col(self.drug_continous_col),
                    )
                    & pl.col(self.drug_rate_col).is_null()
                )
                .then(
                    pl.concat_str(
                        [
                            pl.col(self.drug_amount_unit_col),
                            pl.when(
                                pl.col(
                                    self.drug_patient_weight_col
                                ).is_not_null(),
                                pl.col(self.drug_amount_unit_col) != "units",
                            )
                            .then(pl.lit("/kg"))
                            .otherwise(pl.lit("")),
                            pl.lit("/min"),
                        ]
                    )
                )
                .otherwise(pl.col(self.drug_rate_unit_col))
                .alias(self.drug_rate_unit_col),
            )
            # add missing drug class information
            # NOTE: -> refactor into imputation?
            # NOTE: -> prob yes, since one also needs to deal with boluses
            .with_columns(
                pl.when(
                    pl.col(self.drug_name_col).is_in(
                        fluids_class_mapping.keys()
                    )
                )
                .then(pl.col(self.drug_name_col).replace(fluids_class_mapping))
                .when(
                    pl.col(self.drug_ingredient_col).is_in(
                        drugs_class_mapping.keys()
                    )
                )
                .then(
                    pl.col(self.drug_ingredient_col).replace(
                        drugs_class_mapping
                    )
                )
                .otherwise(pl.col(self.drug_class_col))
                .alias(self.drug_class_col),
                # harmonize units
                pl.col(self.drug_amount_unit_col)
                .str.replace("mL", "ml")
                .str.replace(r"^U$", "units")
                .str.replace("µ", "mc")
                .str.replace("grams", "g")
                .str.replace("mEQ", "mEq")
                .str.replace("mEq\.", "mEq")
                .alias(self.drug_amount_unit_col),
            )
            # remove units if amount is null
            .with_columns(
                pl.when(pl.col(self.drug_amount_col).is_null())
                .then(None)
                .otherwise(pl.col(self.drug_amount_unit_col))
                .alias(self.drug_amount_unit_col),
                pl.when(pl.col(self.drug_rate_col).is_null())
                .then(None)
                .otherwise(pl.col(self.drug_rate_unit_col))
                .alias(self.drug_rate_unit_col),
            )
            .select(medications_cols_list)
            .unique()
            .sort(self.global_icu_stay_id_col, self.drug_start_col)
        )

    # Helper functions
    # Concatenate the IDs with the database name to create a global ID
    def _concat_helper(self, data: pl.LazyFrame, name: str) -> pl.LazyFrame:
        data_cols = data.columns

        if self.drug_mixture_id_col in data_cols:
            data = data.with_columns(
                pl.when(pl.col(self.drug_mixture_id_col).is_not_null())
                .then(
                    pl.concat_str(
                        [pl.lit(name), pl.col(self.drug_mixture_id_col)]
                    )
                )
                .otherwise(None)
                .alias(self.drug_mixture_id_col)
            )

        if self.drug_mixture_admin_id_col in data_cols:
            data = data.with_columns(
                pl.when(pl.col(self.drug_mixture_admin_id_col).is_not_null())
                .then(
                    pl.concat_str(
                        [pl.lit(name), pl.col(self.drug_mixture_admin_id_col)]
                    )
                )
                .otherwise(None)
                .alias(self.drug_mixture_admin_id_col)
            )

        return data.with_columns(
            pl.concat_str([pl.lit(name), pl.col(self.icu_stay_id_col)]).alias(
                self.global_icu_stay_id_col
            )
        )

    # Print the number of unique cases in the medication data
    def _print_unique_cases(
        self, data: pl.LazyFrame, name: str
    ) -> pl.LazyFrame:
        unique_count = (
            data.select(self.global_icu_stay_id_col)
            .unique()
            .count()
            .collect(streaming=True)
            .to_numpy()[0][0]
        )
        print(
            f"reprodICU - {unique_count:6.0f} unique cases with medication data in {name}."
        )

        return data
