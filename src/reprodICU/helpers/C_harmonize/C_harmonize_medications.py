# Author: Finn Fassbender
# Last modified: 2025-10-30

# Description: This script combines the preprocessed medications from the different
# databases into one common table

import polars as pl

from ..A_extract.A_extract_eicu import EICUExtractor
from ..A_extract.A_extract_hirid import HiRIDExtractor
from ..A_extract.A_extract_mimic3 import MIMIC3Extractor
from ..A_extract.A_extract_mimic4 import MIMIC4Extractor
from ..A_extract.A_extract_nwicu import NWICUExtractor
from ..A_extract.A_extract_sicdb import SICdbExtractor
from ..A_extract.A_extract_umcdb import UMCdbExtractor
from ..helper import GlobalHelpers, GlobalVars

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
        self.helpers = GlobalHelpers()
        self.datasets = datasets
        self.medications = None

        if "eICU" in self.datasets:
            self.eicu = EICUExtractor(paths, DEMO)
        if "HiRID" in self.datasets:
            self.hirid = HiRIDExtractor(paths)
        if "MIMIC3" in self.datasets:
            self.mimic3 = MIMIC3Extractor(paths, DEMO)
        if "MIMIC4" in self.datasets:
            self.mimic4 = MIMIC4Extractor(paths, DEMO)
        if "NWICU" in self.datasets:
            self.nwicu = NWICUExtractor(paths)
        if "SICdb" in self.datasets:
            self.sicdb = SICdbExtractor(paths)
        if "UMCdb" in self.datasets:
            self.umcdb = UMCdbExtractor(paths)

        self.medications_cols_list = [
            self.global_icu_stay_id_col,
            self.drug_prescription_id_col,
            self.drug_mixture_id_col,
            self.drug_mixture_admin_id_col,
            self.drug_admin_type_col,
            self.drug_ingredient_col,
            self.drug_name_col,
            self.drug_name_OMOP_col,
            self.drug_code_col,
            self.drug_class_col,
            self.drug_continuous_col,
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

    def harmonize_medications(self) -> pl.LazyFrame:
        """
        Harmonize medication data from multiple databases.

        Steps:
            1. Validate non-empty dataset list; raise ValueError if empty.
            2. Load drug and fluid classification mappings.
            3. For each dataset: extract medications and create global identifiers.
            4. Concatenate all datasets using diagonal-relaxed join.
            5. Enhance drug classifications using mapping tables.
            6. Normalize units (mL→ml, µ→mc).
            7. Select and cast columns to proper types.
            8. Remove duplicates and sort by ICU stay and start time.

        Returns:
            pl.LazyFrame: Contains columns:
                - {global_icu_stay_id_col}: Global ICU stay identifier.
                - {drug_prescription_id_col}: Prescription identifier.
                - {drug_ingredient_col}: Active drug ingredient.
                - {drug_name_col}: Original medication name.
                - {drug_name_OMOP_col}: OMOP-mapped medication name.
                - {drug_class_col}: Drug classification.
                - {drug_admin_route_col}: Administration route.
                - {drug_amount_col}: Dose amount (float).
                - {drug_amount_unit_col}: Dose unit.
                - {drug_rate_col}: Infusion rate (float).
                - {drug_rate_unit_col}: Rate unit.
                - {fluid_name_col}: Fluid name.
                - {fluid_amount_col}: Fluid amount (float).
                - {fluid_rate_col}: Fluid rate (float).
                - {drug_start_col}: Medication start datetime.
                - {drug_end_col}: Medication end datetime.
                - {drug_patient_weight_col}: Patient weight for dosing (float).
        """
        if self.medications is not None:
            print(
                "reprodICU - Medication harmonization already performed. "
                "Skipping harmonization."
            )
            return
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

        medications: pl.LazyFrame = pl.concat(
            medications_datasets,
            how="diagonal_relaxed",
        )

        # Add missing columns as null columns
        medications = medications.with_columns(
            pl.lit(None).alias(col)
            for col in self.medications_cols_list
            if col not in medications.columns
        )

        self.medications = (
            medications.cast(
                {
                    self.drug_name_col: str,
                    self.drug_ingredient_col: str,
                    self.drug_code_col: int,
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
                        pl.col(self.drug_continuous_col),
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
                        pl.col(self.drug_continuous_col),
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
            .select(self.medications_cols_list)
            .unique()
            .sort(self.global_icu_stay_id_col, self.drug_start_col)
        )

        return self.medications

    def harmonize_split_medications(self, table: str) -> pl.LazyFrame:
        """
        Splits medication data into two separate tables:
        - One for administered medications.
        - One for prescribed medications.

        Returns:
            pl.LazyFrame: A lazy frame containing the split medication data.
            - For "administered": Contains medications that were given.
            - For "prescribed": Contains medications that were prescribed.
        """
        assert table in [
            "administered",
            "prescribed",
        ], "Table must be either 'administered' or 'prescribed'."

        _medications_cols_list = self.medications_cols_list.copy()
        _medications_cols_list.remove(self.drug_admin_type_col)

        self.harmonize_medications()
        if table == "administered":
            return (
                self.medications.filter(
                    pl.col(self.drug_admin_type_col) == "given"
                )
                .select(_medications_cols_list)
                .sort(self.global_icu_stay_id_col, self.drug_start_col)
            )
        elif table == "prescribed":
            return (
                self.medications.filter(
                    pl.col(self.drug_admin_type_col) == "prescribed"
                )
                .select(_medications_cols_list)
                .sort(self.global_icu_stay_id_col, self.drug_start_col)
            )

    # Helper functions
    # Concatenate the IDs with the database name to create a global ID
    def _concat_helper(self, data: pl.LazyFrame, name: str) -> pl.LazyFrame:
        data_cols = data.columns

        if self.drug_prescription_id_col in data_cols:
            data = data.with_columns(
                pl.when(pl.col(self.drug_prescription_id_col).is_not_null())
                .then(
                    pl.concat_str(
                        [pl.lit(name), pl.col(self.drug_prescription_id_col)]
                    )
                )
                .otherwise(None)
                .alias(self.drug_prescription_id_col)
            )

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
            .collect()
            .item()
        )
        print(
            f"reprodICU - {unique_count:6.0f} unique cases with medication data in {name}."
        )

        return data
