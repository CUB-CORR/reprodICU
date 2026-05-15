# Author: Finn Fassbender
# Last modified: 2025-10-30

# Enables easily working with OMOP vocabularies.
from typing import Union

import polars as pl

from .helper_filepaths import OMOPPaths


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
        Retrieve concept names from OMOP concept IDs.

        Steps:
            1. Filter CONCEPT table by concept_ids.
            2. Select concept_id and concept_name columns.
            3. Return as dict (concept_id → concept_name) or DataFrame.

        Returns:
            dict: Mapping from concept_id to concept_name if return_dict=True; otherwise pl.DataFrame.
        """
        concept_names = (
            self.CONCEPT.filter(pl.col("concept_id").is_in(concept_ids))
            .select("concept_id", "concept_name")
            .drop_nulls()
            .collect()
        )

        if not return_dict:
            return concept_names

        return dict(
            zip(
                concept_names.get_column("concept_id").to_list(),
                concept_names.get_column("concept_name").to_list(),
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
        Retrieve concept IDs from OMOP concept names.

        Steps:
            1. Filter CONCEPT by concept_names and vocabulary_id.
            2. If is_lab_test=True: filter domain_id to "Measurement".
            3. Select concept_id and concept_name columns.
            4. Return as dict (concept_name → concept_id) or DataFrame.

        Returns:
            dict: Mapping from concept_name to concept_id if return_dict=True; otherwise pl.DataFrame.
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
            .drop_nulls()
            .collect()
        )

        if not return_dict:
            return concept_ids

        # Use Polars to_list instead of to_numpy (avoids issues)
        return dict(
            zip(
                concept_ids.get_column("concept_name").to_list(),
                concept_ids.get_column("concept_id").to_list(),
            )
        )

    def get_concept_codes_from_names(
        self, concept_names: list[str], return_dict: bool = True
    ) -> dict:
        """
        Retrieve concept codes from OMOP concept names.

        Steps:
            1. Filter CONCEPT by concept_names.
            2. Select concept_name and concept_code columns.
            3. Return as dict (concept_name → concept_code) or DataFrame.

        Returns:
            dict: Mapping from concept_name to concept_code if return_dict=True; otherwise pl.DataFrame.
        """
        concept_codes = (
            self.CONCEPT.filter(pl.col("concept_name").is_in(concept_names))
            .select("concept_name", "concept_code")
            .drop_nulls()
            .collect()
        )

        if not return_dict:
            return concept_codes

        return dict(
            zip(
                concept_codes.get_column("concept_name").to_list(),
                concept_codes.get_column("concept_code").to_list(),
            )
        )

    def get_concept_names_from_codes(
        self, concept_codes: list[str], return_dict: bool = True
    ) -> dict:
        """
        Retrieve concept names from OMOP concept codes.

        Steps:
            1. Ensure concept_codes are strings.
            2. Filter CONCEPT by concept_codes.
            3. Select concept_code and concept_name columns.
            4. Return as dict (concept_code → concept_name) or DataFrame.

        Returns:
            dict: Mapping from concept_code to concept_name if return_dict=True; otherwise pl.DataFrame.
        """

        # ensure concept_codes are strings
        concept_codes = [
            str(concept_code)
            for concept_code in concept_codes
            if concept_code is not None
        ]

        concept_names = (
            self.CONCEPT.filter(pl.col("concept_code").is_in(concept_codes))
            .select("concept_code", "concept_name")
            .drop_nulls()
            .collect()
        )

        if not return_dict:
            return concept_names

        return dict(
            zip(
                concept_names.get_column("concept_code").to_list(),
                concept_names.get_column("concept_name").to_list(),
            )
        )

    def get_concept_name_from_code(self, concept_code: str) -> str:
        """
        Retrieve a single concept name from a concept code.

        Steps:
            1. Call get_concept_names_from_codes with single code.
            2. Return the corresponding concept_name value.

        Returns:
            str: The concept name for the given code.
        """
        return self.get_concept_names_from_codes([concept_code])[concept_code]

    # region ndc
    def get_rxnorm_concept_id_from_ndc(self, ndc: list[str]) -> dict:
        """
        Retrieve RxNorm concept IDs mapped from NDC (National Drug Code).

        Steps:
            1. Filter CONCEPT for 11-digit NDC codes matching input list.
            2. Join RELATIONSHIP table on "Maps to" relationship to find RxNorm concept IDs.
            3. Return as dict (ndc → rxnorm_concept_id).

        Returns:
            dict: Mapping from NDC code to RxNorm concept ID.
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
            .drop_nulls()
            .collect()
        )

        return dict(
            zip(
                rxnorm_concept_ids.get_column("ndc").to_list(),
                rxnorm_concept_ids.get_column("rxnorm_concept_id").to_list(),
            )
        )

    # region ingredient
    def get_ingredient(
        self, drug_concept_ids: list[int], return_dict: bool = True
    ) -> Union[dict, pl.DataFrame]:
        """
        Retrieve drug ingredients using OMOP hierarchy (ANCESTOR relationships).
        Based on OMOP-Queries/Drug/D03: Find ingredients of a drug
        https://github.com/OHDSI/OMOP-Queries/blob/master/md/Drug.md#d03-find-ingredients-of-a-drug

        Steps:
            1. Filter ANCESTOR by descendant_concept_id (input drugs).
            2. Join CONCEPT twice to get ancestor (ingredient) and descendant (drug) names.
            3. Filter for concept_class_id == "Ingredient".
            4. Return as dict (drug_concept_id → ingredient_name) or DataFrame.

        Returns:
            dict: Mapping from drug_concept_id to ingredient_name if return_dict=True; otherwise pl.DataFrame.
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
            .drop_nulls()
            .collect()
        )

        if not return_dict:
            return ingredients

        return dict(
            zip(
                ingredients.get_column("drug_concept_id").to_list(),
                ingredients.get_column("ingredient_name").to_list(),
            )
        )

    # region lab
    def get_lab_relationship_from_name(
        self, lab_names: list[str], lab_relationship: str
    ) -> dict:
        """
        Retrieve lab properties from LOINC attributes using specified relationship.

        Steps:
            1. Convert lab_names to OMOP concept IDs.
            2. Filter RELATIONSHIP by relationship_id (e.g., "Has component", "Has property").
            3. Join with concept names to resolve attribute names.
            4. Return as dict (lab_name → attribute_name).

        Returns:
            dict: Mapping from lab name to attribute name for the specified relationship.
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
            .drop_nulls()
            .collect()
        )
        lab_id_to_property_id = dict(
            zip(
                lab_id_to_property_id.get_column("concept_id_1").to_list(),
                lab_id_to_property_id.get_column("concept_id_2").to_list(),
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
        Retrieve lab components from lab names using "Has component" relationship.

        Returns:
            dict: Mapping from lab name to component name.
        """
        return self.get_lab_relationship_from_name(lab_names, "Has component")

    def get_lab_system_from_name(self, lab_names: list[str]) -> dict:
        """
        Retrieve lab systems from lab names using "Has system" relationship.

        Returns:
            dict: Mapping from lab name to system name.
        """
        return self.get_lab_relationship_from_name(lab_names, "Has system")

    def get_lab_property_from_name(self, lab_names: list[str]) -> dict:
        """
        Retrieve lab properties from lab names using "Has property" relationship.

        Returns:
            dict: Mapping from lab name to property name.
        """
        return self.get_lab_relationship_from_name(lab_names, "Has property")

    def get_lab_method_from_name(self, lab_names: list[str]) -> dict:
        """
        Retrieve lab methods from lab names using "Has method" relationship.

        Returns:
            dict: Mapping from lab name to method name.
        """
        return self.get_lab_relationship_from_name(lab_names, "Has method")

    def get_lab_time_aspect_from_name(self, lab_names: list[str]) -> dict:
        """
        Retrieve lab time aspects from lab names using "Has time aspect" relationship.

        Returns:
            dict: Mapping from lab name to time aspect name.
        """
        return self.get_lab_relationship_from_name(lab_names, "Has time aspect")

    def _load_data_for_get_LOINC_codes_for_attributes(self) -> None:
        """
        Load and cache LOINC lab concept and relationship data for attribute queries.

        Steps:
            1. Load all LOINC concepts excluding multi-attribute combinations (containing "--").
            2. Load relationships involving loaded LOINC concepts.

        Returns:
            None: Data stored in self.CONCEPT_LOINC_LAB and self.RELATIONSHIP_LOINC_LAB.
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
        Retrieve LOINC lab test codes matching specified LOINC attributes (component, property, system, method, time).

        Steps:
            1. Load LOINC concept and relationship data if not already cached.
            2. Validate queries: at least one attribute per query; all attribute names must exist.
            3. For each query: collect distinct (component, property, system, method, time) combinations from data.
            4. Resolve attributes to concept IDs; build lab_concept_id → attribute mapping.
            5. Join all attribute mappings on lab_concept_id; filter on exact/null matches for optional attributes.
            6. Apply system fallbacks (Blood arterial→Blood venous→Blood; Blood mixed→Blood venous).
            7. Map lab concept IDs to LOINC codes; return sorted lists.

        Returns:
            list[list[str]]: For batch queries: list of LOINC code lists (one per query), preserving input order; may contain empty lists if no matches.
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
