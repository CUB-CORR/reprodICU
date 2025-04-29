# Author: Finn Fassbender
# Last modified: 2024-09-05

# Enables easily working with OMOP vocabularies.
from typing import Union

import polars as pl
from helpers.helper_filepaths import OMOPPaths


class Vocabulary(OMOPPaths):
    """
    Class for working with OMOP vocabularies.
    """

    def __init__(self, paths):
        super().__init__(paths)
        self.ANCESTOR = pl.scan_parquet(self.CONCEPT_ANCESTOR_path)
        self.CLASS = pl.scan_parquet(self.CONCEPT_CLASS_path)
        self.RELATIONSHIP = pl.scan_parquet(self.CONCEPT_RELATIONSHIP_path)
        self.SYNONYM = pl.scan_parquet(self.CONCEPT_SYNONYM_path)
        self.CONCEPT = pl.scan_parquet(self.CONCEPT_path)
        self.DOMAIN = pl.scan_parquet(self.DOMAIN_path)
        self.DRUG_STRENGTH = pl.scan_parquet(self.DRUG_STRENGTH_path)
        # self.RELATIONSHIP = pl.scan_parquet(self.RELATIONSHIP_path)
        self.VOCABULARY = pl.scan_parquet(self.VOCABULARY_path)

    # region names / ids / codes
    def get_concept_names_from_ids(
        self, concept_ids: list[int], return_dict: bool = True
    ) -> dict:
        """
        Get concept_names from concept_ids.

        Args:
            concept_ids (list[int]): List of concept_ids.
            return_dict (bool, optional): Whether to return the result as a dictionary. Defaults to True.

        Returns:
            dict: Dictionary with concept_id as key and concept_name as value.
        """
        concept_names = (
            self.CONCEPT.filter(pl.col("concept_id").is_in(concept_ids))
            .select("concept_id", "concept_name")
            .collect(streaming=True)
        )

        if not return_dict:
            return concept_names

        return dict(
            zip(
                concept_names["concept_id"].to_numpy(),
                concept_names["concept_name"].to_numpy(),
            )
        )

    def get_concept_ids_from_names(
        self, concept_names: list[str], return_dict: bool = True
    ) -> dict:
        """
        Get concept_ids from concept_names.

        Args:
            concept_names (list[str]): List of concept_names.
            return_dict (bool, optional): Whether to return the result as a dictionary. Defaults to True.

        Returns:
            dict: Dictionary with concept_name as key and concept_id as value.
        """
        concept_ids = (
            self.CONCEPT.filter(pl.col("concept_name").is_in(concept_names))
            .select("concept_id", "concept_name")
            .collect(streaming=True)
        )

        if not return_dict:
            return concept_ids

        return dict(
            zip(
                concept_ids["concept_name"].to_numpy(),
                concept_ids["concept_id"].to_numpy(),
            )
        )

    def get_concept_codes_from_names(
        self, concept_names: list[str], return_dict: bool = True
    ) -> dict:
        """
        Get concept_codes from concept_names.

        Args:
            concept_names (list[str]): List of concept_names.
            return_dict (bool, optional): Whether to return the result as a dictionary. Defaults to True.

        Returns:
            dict: Dictionary with concept_name as key and concept_code as value.
        """
        concept_codes = (
            self.CONCEPT.filter(pl.col("concept_name").is_in(concept_names))
            .select("concept_name", "concept_code")
            .collect(streaming=True)
        )

        if not return_dict:
            return concept_codes

        return dict(
            zip(
                concept_codes["concept_name"].to_numpy(),
                concept_codes["concept_code"].to_numpy(),
            )
        )

    def get_concept_names_from_codes(
        self, concept_codes: list[str], return_dict: bool = True
    ) -> dict:
        """
        Get concept_names from concept_codes.

        Args:
            concept_codes (list[str]): List of concept_codes.
            return_dict (bool, optional): Whether to return the result as a dictionary. Defaults to True.

        Returns:
            dict: Dictionary with concept_code as key and concept_name as value.
        """

        # ensure concept_codes are strings
        concept_codes = [str(concept_code) for concept_code in concept_codes]

        concept_names = (
            self.CONCEPT.filter(pl.col("concept_code").is_in(concept_codes))
            .select("concept_code", "concept_name")
            .collect(streaming=True)
        )

        if not return_dict:
            return concept_names

        return dict(
            zip(
                concept_names["concept_code"].to_numpy(),
                concept_names["concept_name"].to_numpy(),
            )
        )

    def get_concept_name_from_code(self, concept_code: str) -> str:
        """
        Get concept_name from concept_code.

        Args:
            concept_code (str): Concept code.

        Returns:
            str: Concept name.
        """
        return self.get_concept_names_from_codes([concept_code])[concept_code]

    # region ndc
    def get_rxnorm_concept_id_from_ndc(self, ndc: list[str]) -> dict:
        """
        Get RxNorm from NDC.

        Args:
            ndc (list[str]): List of NDC.

        Returns:
            dict: Dictionary with NDC as key and RxNorm concept ID as value.
        """

        ndc_concept_ids_lf = self.CONCEPT.filter(
            pl.col("concept_class_id") == "11-digit NDC",
            pl.col("concept_code").is_in(ndc),
        ).select("concept_id", "concept_code")

        ndc_concept_ids = (
            ndc_concept_ids_lf.select("concept_id")
            .collect()
            .to_series()
            .to_list()
        )

        rxnorm_concept_ids = (
            self.RELATIONSHIP.filter(
                pl.col("concept_id_1").is_in(ndc_concept_ids),
                pl.col("relationship_id") == "Maps to",
            )
            .select("concept_id_1", "concept_id_2")
            .join(
                ndc_concept_ids_lf,
                left_on="concept_id_1",
                right_on="concept_id",
                how="left",
            )
            .rename(
                {
                    "concept_code": "ndc",
                    "concept_id_2": "rxnorm_concept_id",
                }
            )
            .collect()
        )

        return dict(
            zip(
                rxnorm_concept_ids["ndc"].to_numpy(),
                rxnorm_concept_ids["rxnorm_concept_id"].to_numpy(),
            )
        )

    # region ingredient
    def get_ingredient(
        self, drug_concept_ids: list[int], return_dict: bool = True
    ) -> Union[dict, pl.DataFrame]:
        """
        Get ingredient_id from drug concept_ids.
        Based on OMOP-Queries/Drug/D03: Find ingredients of a drug
        https://github.com/OHDSI/OMOP-Queries/blob/master/md/Drug.md#d03-find-ingredients-of-a-drug

        Args:
            drug_concept_ids (list[int]): List of drug concept_ids.
            return_dict (bool, optional): Whether to return the result as a dictionary. Defaults to True.

        Returns:
            dict: Dictionary with drug_concept_id as key and ingredient_name as value.
        """

        ingredients = (
            self.ANCESTOR.filter(
                pl.col("descendant_concept_id").is_in(drug_concept_ids)
            )
            .join(
                self.CONCEPT,
                left_on="ancestor_concept_id",
                right_on="concept_id",
                suffix="_A",
                how="left",
            )
            .join(
                self.CONCEPT,
                left_on="descendant_concept_id",
                right_on="concept_id",
                suffix="_D",
                how="left",
            )
            .filter(pl.col("concept_class_id") == "Ingredient")
            .select(
                "descendant_concept_id",
                "ancestor_concept_id",
                "concept_name",
            )
            .rename(
                {
                    "descendant_concept_id": "drug_concept_id",
                    "ancestor_concept_id": "ingredient_concept_id",
                    "concept_name": "ingredient_name",
                }
            )
            .collect(streaming=True)
        )

        if not return_dict:
            return ingredients

        return dict(
            zip(
                ingredients["drug_concept_id"].to_numpy(),
                ingredients["ingredient_name"].to_numpy(),
            )
        )

    # region lab
    def get_lab_relationship_from_name(
        self, lab_names: list[str], lab_relationship: str
    ) -> dict:
        """
        Get lab properties from lab names.

        Args:
            lab_names (list[str]): List of lab names.
            lab_relationship (str): Relationship to get.

        Returns:
            dict: Dictionary with lab name as key and lab property as value.
        """

        lab_names_to_id = self.get_concept_ids_from_names(lab_names)
        lab_id_to_names = {v: k for k, v in lab_names_to_id.items()}

        lab_id_to_property_id = (
            self.RELATIONSHIP.filter(
                pl.col("concept_id_1").is_in(list(lab_names_to_id.values())),
                pl.col("relationship_id") == lab_relationship,
            )
            .select("concept_id_1", "concept_id_2")
            .collect(streaming=True)
        )
        lab_id_to_property_id = dict(
            zip(
                lab_id_to_property_id["concept_id_1"].to_numpy(),
                lab_id_to_property_id["concept_id_2"].to_numpy(),
            )
        )

        lab_property_id_to_property_name = self.get_concept_names_from_ids(
            lab_id_to_property_id.values()
        )

        # print("LAB_IT_TO_NAMES", lab_id_to_names)
        # print("LAB_ID_TO_PROPERTY_ID", lab_id_to_property_id)
        # print("LAB_PROPERTY_ID_TO_PROPERTY_NAME", lab_property_id_to_property_name)

        return {
            lab_id_to_names[k]: lab_property_id_to_property_name[v]
            for k, v in lab_id_to_property_id.items()
        }

    def get_lab_component_from_name(self, lab_names: list[str]) -> dict:
        """
        Get lab component from lab names.

        Args:
            lab_names (list[str]): List of lab names.

        Returns:
            dict: Dictionary with lab name as key and lab component as value.
        """

        return self.get_lab_relationship_from_name(lab_names, "Has component")

    def get_lab_system_from_name(self, lab_names: list[str]) -> dict:
        """
        Get lab system from lab names.

        Args:
            lab_names (list[str]): List of lab names.

        Returns:
            dict: Dictionary with lab name as key and lab system as value.
        """
        return self.get_lab_relationship_from_name(lab_names, "Has system")

    def get_lab_property_from_name(self, lab_names: list[str]) -> dict:
        """
        Get lab property from lab names.

        Args:
            lab_names (list[str]): List of lab names.

        Returns:
            dict: Dictionary with lab name as key and lab property as value.
        """
        return self.get_lab_relationship_from_name(lab_names, "Has property")

    def get_lab_method_from_name(self, lab_names: list[str]) -> dict:
        """
        Get lab method from lab names.

        Args:
            lab_names (list[str]): List of lab names.

        Returns:
            dict: Dictionary with lab name as key and lab method as value.
        """
        return self.get_lab_relationship_from_name(lab_names, "Has method")

    def get_lab_time_aspect_from_name(self, lab_names: list[str]) -> dict:
        """
        Get lab time aspect from lab names.

        Args:
            lab_names (list[str]): List of lab names.

        Returns:
            dict: Dictionary with lab name as key and lab time aspect as value.
        """
        return self.get_lab_relationship_from_name(lab_names, "Has time aspect")
