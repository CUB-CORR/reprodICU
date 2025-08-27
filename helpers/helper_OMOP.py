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
        self.RELATIONSHIP = pl.scan_parquet(
            self.CONCEPT_RELATIONSHIP_path, parallel="prefiltered"
        )
        self.SYNONYM = pl.scan_parquet(self.CONCEPT_SYNONYM_path)
        self.CONCEPT = pl.scan_parquet(
            self.CONCEPT_path, parallel="prefiltered"
        )
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
            .collect()
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
        self,
        concept_names: list[str],
        vocabulary: str = "LOINC",
        is_lab_test: bool = False,
        return_dict: bool = True,
    ) -> dict:
        """
        Get concept_ids from concept_names.

        Args:
            concept_names (list[str]): List of concept_names.
            vocabulary (str, optional): Vocabulary to filter by. Defaults to "LOINC".
            is_lab_test (bool, optional): Whether to filter for lab tests. Defaults to False.
            return_dict (bool, optional): Whether to return the result as a dictionary. Defaults to True.

        Returns:
            dict: Dictionary with concept_name as key and concept_id as value.
        """
        concept_ids = (
            self.CONCEPT.filter(
                pl.col("concept_name").is_in(concept_names),
                pl.col("vocabulary_id") == vocabulary,
                (
                    # pl.col("concept_class_id") == "Lab Test"
                    pl.col("domain_id") == "Measurement"
                    if is_lab_test
                    else pl.lit(True)
                ),
            )
            .select("concept_id", "concept_name")
            .collect()
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
            .collect()
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
            .collect()
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
            .collect()
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

        lab_names_to_id = self.get_concept_ids_from_names(
            lab_names, is_lab_test=True
        )
        lab_id_to_names = {v: k for k, v in lab_names_to_id.items()}

        lab_id_to_property_id = (
            self.RELATIONSHIP.filter(
                pl.col("concept_id_1").is_in(list(lab_names_to_id.values())),
                pl.col("relationship_id") == lab_relationship,
            )
            .select("concept_id_1", "concept_id_2")
            .collect()
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

    def _load_data_for_get_LOINC_codes_for_attributes(self) -> None:
        """
        Load necessary data for get_LOINC_codes_for_attributes method.
        This method is called automatically when the function is called.
        """
        self.CONCEPT_LOINC_LAB = self.CONCEPT.filter(
            pl.col("vocabulary_id") == "LOINC",
            # pl.col("concept_class_id") == "Lab Test",
            # pl.col("domain_id") == "Measurement",
            ~pl.col("concept_name").str.contains("--"),
        ).collect()
        concept_ids = self.CONCEPT_LOINC_LAB.select("concept_id").to_series()
        self.RELATIONSHIP_LOINC_LAB = self.RELATIONSHIP.filter(
            pl.col("concept_id_1").is_in(concept_ids)
            | pl.col("concept_id_2").is_in(concept_ids),
        ).collect()

    def get_LOINC_codes_for_attributes(
        self,
        queries: list[tuple[str, str, str, str, str]] | None = None,
    ) -> list[str] | list[list[str]]:
        """
        Return all LOINC Lab Test concept_codes matching ALL provided LOINC attributes.
        Any subset of the attributes can be provided; only labs matching every provided
        attribute are returned.
        If no method is provided, only labs WITHOUT a method (no 'Has method' relationship) are returned.

        Args:
            queries (list[tuple[str, str, str, str, str]] | None): List of queries.
                Each query is a tuple of the form:
                (component_name, component_property, component_system, component_method, component_time_aspect).

        Returns:
            list[str]: Matching LOINC concept_codes (sorted, may be empty).

        Raises:
            ValueError: If no attribute provided or an attribute name is unknown.

        Batch queries:
            Pass a list of tuples: (component_name, component_property, component_system, component_method, component_time_aspect)
            The function resolves all queries together and returns a list of lists, preserving input order.
        """
        # Ensure the necessary data is loaded
        if not hasattr(self, "CONCEPT_LOINC_LAB"):
            self._load_data_for_get_LOINC_codes_for_attributes()

        # Validate: at least one attribute per query
        if any(all(v is None for v in q) for q in queries):
            raise ValueError(
                "At least one LOINC attribute must be provided per query."
            )

        # Gather distinct names per attribute across all queries
        comp_names = {q[0] for q in queries}  # can't be None
        prop_names = {q[1] for q in queries if q[1] is not None}
        syst_names = {q[2] for q in queries if q[2] is not None}
        meth_names = {q[3] for q in queries if q[3] is not None}
        time_names = {q[4] for q in queries if q[4] is not None}
        all_names = list(
            comp_names | prop_names | syst_names | meth_names | time_names
        )

        # Resolve all concept_ids for all names in one pass
        attr_concepts_df = (
            self.CONCEPT_LOINC_LAB.filter(
                pl.col("concept_name").is_in(all_names),
                pl.col("concept_class_id").is_in(
                    [
                        "LOINC Component",
                        "LOINC Property",
                        "LOINC System",
                        "LOINC Method",
                        "LOINC Time",
                    ]
                ),
            )
            .select("concept_name", "concept_id")
            .unique()
        )
        # potential 1:N name->id mappings
        name_to_ids = dict(
            attr_concepts_df.group_by("concept_name")
            .all()
            .cast({"concept_id": pl.List(int)})
            .iter_rows()
        )

        missing = [n for n in all_names if n not in name_to_ids]
        if missing:
            raise ValueError(f"Unknown LOINC attribute names: {missing}")

        # Helper: build name -> set(lab_concept_id) map for one attribute
        def labs_by_attr(relation, attribute_names) -> pl.DataFrame:
            """
            Build a mapping: attribute name -> set of lab_concept_id.
            """
            if not attribute_names:
                return pl.DataFrame(
                    schema={"name": pl.String, "lab_concept_id": pl.Int64},
                )

            # Only keep rows for the requested names
            names_df = attr_concepts_df.filter(
                pl.col("concept_name").is_in(list(attribute_names))
            ).select("concept_name", "concept_id")

            # Join with the relationship table to get lab_concept_id
            return (
                self.RELATIONSHIP_LOINC_LAB.filter(
                    pl.col("relationship_id") == relation
                )
                .join(
                    names_df,
                    left_on="concept_id_2",
                    right_on="concept_id",
                    how="inner",
                )
                .select(
                    pl.col("concept_name").alias("name"),
                    pl.col("concept_id_1").cast(int).alias("lab_concept_id"),
                )
            )

        # Build attribute maps in at most one pass per attribute
        # fmt: off
        comp_labs = labs_by_attr("Has component", comp_names).rename({"name": "comp"})
        prop_labs = labs_by_attr("Has property", prop_names).rename({"name": "prop"})
        syst_labs = labs_by_attr("Has system", syst_names).rename({"name": "syst"})
        time_labs = labs_by_attr("Has time aspect", time_names).rename({"name": "time"})
        meth_labs = labs_by_attr("Has method", meth_names).rename({"name": "meth"})
        # fmt: on

        # Labs that have any method (for method=None filtering)
        labs_with_method_set = (
            self.RELATIONSHIP_LOINC_LAB.filter(
                pl.col("relationship_id") == "Has method"
            )
            .select(pl.col("concept_id_1").cast(int).alias("lab_concept_id"))
            .unique()
            .with_columns(pl.lit("other").alias("any_method"))
        )

        # Resolve all queries in parallel using Polars
        base = (
            comp_labs.join(prop_labs, on="lab_concept_id", how="inner")
            .join(syst_labs, on="lab_concept_id", how="left")
            .join(time_labs, on="lab_concept_id", how="left")
            .join(meth_labs, on="lab_concept_id", how="left")
            .join(labs_with_method_set, on="lab_concept_id", how="left")
            .with_columns(pl.coalesce("meth", "any_method").alias("meth"))
            .select("lab_concept_id", "comp", "prop", "syst", "time", "meth")
            .unique()
        )

        # Queries as a small Polars DataFrame (then lazy for joins)
        queries_df = pl.DataFrame(
            {
                "q_idx": list(range(len(queries))),
                "q_comp": [q[0] for q in queries],
                "q_prop": [q[1] for q in queries],
                "q_syst": [q[2] for q in queries],
                "q_meth": [q[3] for q in queries],
                "q_time": [q[4] for q in queries],
            },
            schema={
                "q_idx": pl.Int64,
                "q_comp": pl.Utf8,
                "q_prop": pl.Utf8,
                "q_syst": pl.Utf8,
                "q_meth": pl.Utf8,
                "q_time": pl.Utf8,
            },
        )

        # Join on required keys (comp, prop), then filter optional attributes
        matches = (
            base.join(
                queries_df,
                left_on=["comp", "prop"],
                right_on=["q_comp", "q_prop"],
                how="inner",
                nulls_equal=True,
            )
            .filter(
                pl.col("syst").eq_missing(pl.col("q_syst")),
                pl.col("time").eq_missing(pl.col("q_time")),
                pl.col("meth").eq_missing(pl.col("q_meth")),
            )
            .select("q_idx", "lab_concept_id")
            .unique()
            .group_by("q_idx")
            .agg(pl.col("lab_concept_id").unique().alias("ids"))
        )

        # Build list[set[int]] preserving input order
        result_ids = [set() for _ in range(len(queries))]
        for row in matches.iter_rows(named=True):
            result_ids[row["q_idx"]] = set(row["ids"])

        # Map all candidate lab_ids to concept_codes in one pass
        all_ids = list(set.union(*result_ids))
        if not all_ids:
            return [[] for _ in queries]

        id_to_code = dict(
            self.CONCEPT_LOINC_LAB.filter(pl.col("concept_id").is_in(all_ids))
            .select("concept_id", "concept_code")
            .iter_rows()
        )

        return [
            sorted([id_to_code[i] for i in ids if i in id_to_code])
            for ids in result_ids
        ]
