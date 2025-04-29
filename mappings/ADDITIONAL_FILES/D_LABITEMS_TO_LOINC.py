import polars as pl

# Load the data
d_labitems_mimic3 = pl.read_csv(
    "../../../../raw_data/physionet.org/files/mimiciii/1.4/D_LABITEMS.csv"
)
d_labitems_nwicu = pl.read_csv(
    "../../../../raw_data/physionet.org/files/nwicu-northwestern-icu/0.1.0/data/nw_hosp/d_labitems.csv"
)
d_labitems_to_loinc = pl.read_csv(
    "../mimic4/mimic-code_mapping/d_labitems_to_loinc.csv",
    infer_schema_length=10000,
)
omop_concept = (
    pl.scan_csv(
        "../../../reprodICU_files_OMOP/OMOP_vocabulary/CONCEPT.csv",
        separator="\t",
        quote_char=None,
    )
    .filter(
        pl.col("concept_code").is_in(d_labitems_mimic3.select("LOINC_CODE"))
    )
    .select("concept_code", "concept_name")
    .collect()
)

# ------------------------------------------------------------------------------
# MIMIC-III to LOINC Mapping
# ------------------------------------------------------------------------------
"""
Creates the MIMIC-III to LOINC mapping.

Steps:
    1. Join d_labitems_mimic3 with omop_concept:
       - Left join on {LOINC_CODE} (from d_labitems_mimic3) with {concept_code} (from omop_concept).
       - Retrieve {LOINC_CONCEPT_NAME} from omop_concept (original lab concept name).
    2. Join with d_labitems_to_loinc:
       - Left join on {ITEMID} (from MIMIC-III) with {"itemid (omop_source_code)"} (from mapping).
       - Adds {MAPPED_CONCEPT_NAME} and {MAPPED_CONCEPT_CODE} from MIMIC-IV mappings.
    3. Create a boolean column {LOINC_CODE_MATCH}:
       - Indicates whether {LOINC_CODE} equals {MAPPED_CONCEPT_CODE} (with missing values comparison).
    4. Replace {LOINC_CONCEPT_NAME} with None when {LOINC_CODE_MATCH} is True.
    5. Coalesce between {MAPPED_CONCEPT_NAME} and {LOINC_CONCEPT_NAME}:
       - Prefer {MAPPED_CONCEPT_NAME} if available.
    6. Select and write following columns:
       • ITEMID: Unique identifier for the lab item.
       • COALESCED_CONCEPT_NAME: Final lab concept name after coalescing.
       • LABEL: Lab item label.
       • FLUID: Type of fluid sampled.
       • CATEGORY: Category of the lab test.
       • LOINC_CODE: Original LOINC code.
       • LOINC_CONCEPT_NAME: Original LOINC concept name from OMOP.
       • LOINC_CODE_MATCH: Boolean indicating if {LOINC_CODE} matches {MAPPED_CONCEPT_CODE}.
       • MAPPED_CONCEPT_CODE: Concept code from MIMIC-IV mapping.
       • MAPPED_CONCEPT_NAME: Concept name from MIMIC-IV mapping.
"""

(
    d_labitems_mimic3
    # Join with OMOP vocabulary
    .join(
        omop_concept,
        left_on="LOINC_CODE",
        right_on="concept_code",
        how="left",
    )
    .rename({"concept_name": "LOINC_CONCEPT_NAME"})
    # Join with MIMIC-IV to LOINC mapping
    .join(
        d_labitems_to_loinc,
        left_on="ITEMID",
        right_on="itemid (omop_source_code)",
        how="left",
    )
    .rename(
        {
            "omop_concept_name": "MAPPED_CONCEPT_NAME",
            "omop_concept_code": "MAPPED_CONCEPT_CODE",
        }
    )
    # Check if LOINC_CODE and MAPPED_CONCEPT_CODE match
    .with_columns(
        pl.col("LOINC_CODE")
        .eq_missing(pl.col("MAPPED_CONCEPT_CODE"))
        .alias("LOINC_CODE_MATCH"),
    )
    # Keep only original LOINC_CONCEPT_NAME if LOINC_CODE and MAPPED_CONCEPT_CODE do not match
    .with_columns(
        pl.when("LOINC_CODE_MATCH")
        .then(None)
        .otherwise(pl.col("LOINC_CONCEPT_NAME"))
        .alias("LOINC_CONCEPT_NAME"),
    )
    # Coalesce LOINC_CONCEPT_NAME and MAPPED_CONCEPT_NAME
    # prefer mappings from MIMIC-IV to LOINC 
    .with_columns(
        pl.coalesce(
            pl.col("MAPPED_CONCEPT_NAME"),
            pl.col("LOINC_CONCEPT_NAME"),
        ).alias("COALESCED_CONCEPT_NAME"),
    )
    # Drop unnecessary columns
    .select(
        "ITEMID",  # Unique lab item identifier.
        "COALESCED_CONCEPT_NAME",  # Final lab concept name after coalescing.
        "LABEL",  # Lab item label.
        "FLUID",  # Type of fluid sampled.
        "CATEGORY",  # Category of the lab test.
        "LOINC_CODE",  # Original LOINC code.
        "LOINC_CONCEPT_NAME",  # Revised LOINC concept name.
        "LOINC_CODE_MATCH",  # Boolean indicating if LOINC_CODE matches MAPPED_CONCEPT_CODE.
        "MAPPED_CONCEPT_CODE",  # Concept code from MIMIC-IV mapping.
        "MAPPED_CONCEPT_NAME",  # Concept name from MIMIC-IV mapping.
    )
    .write_csv("../mimic3/mimic-code_mapping/d_labitems_to_loinc_mimic3.csv")
)

# ------------------------------------------------------------------------------
# NWICU to LOINC Mapping
# ------------------------------------------------------------------------------
"""
Creates the NWICU to LOINC mapping.

Steps:
    1. Join d_labitems_nwicu with d_labitems_to_loinc based on:
       - {label}: Lab item label.
       - {fluid}: Fluid type.
       - {category}: Lab test category.
    2. Rename the mapping columns:
       - Rename {omop_concept_name} to {mapped_concept_name}.
       - Rename {omop_concept_code} to {mapped_concept_code}.
    3. Remove duplicate rows and sort the data based on {itemid} (unique lab item identifier).
    4. Select and write the following columns:
       • itemid: Unique identifier for the lab item.
       • label: Lab item label.
       • fluid: Type of fluid.
       • category: Category of the lab test.
       • mapped_concept_code: Mapped lab concept code from MIMIC-IV mapping.
       • mapped_concept_name: Mapped lab concept name from MIMIC-IV mapping.
"""

(
    d_labitems_nwicu
    # Join with MIMIC-IV to LOINC mapping
    .join(
        d_labitems_to_loinc,
        on=["label", "fluid", "category"],
        how="left",
    )
    .rename(
        {
            "omop_concept_name": "mapped_concept_name",
            "omop_concept_code": "mapped_concept_code",
        }
    )
    .unique()
    .sort("itemid")
    # Drop unnecessary columns
    .select(
        "itemid",  # Unique lab item identifier.
        "label",  # Lab item label.
        "fluid",  # Type of fluid.
        "category",  # Category of the lab test.
        "mapped_concept_code",  # Mapped lab concept code.
        "mapped_concept_name",  # Mapped lab concept name.
    )
    .write_csv("../nwicu/d_labitems_to_loinc_nwicu.csv")
)
