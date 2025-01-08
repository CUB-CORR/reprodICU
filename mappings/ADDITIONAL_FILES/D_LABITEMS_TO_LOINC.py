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


# Create the MIMIC-III to LOINC mapping
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
        "ITEMID",
        "COALESCED_CONCEPT_NAME",
        "LABEL",
        "FLUID",
        "CATEGORY",
        "LOINC_CODE",
        "LOINC_CONCEPT_NAME",
        "LOINC_CODE_MATCH",
        "MAPPED_CONCEPT_CODE",
        "MAPPED_CONCEPT_NAME",
    )
    .write_csv("../mimic3/mimic-code_mapping/d_labitems_to_loinc_mimic3.csv")
)


# Create the NWICU to LOINC mapping
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
        "itemid",
        "label",
        "fluid",
        "category",
        "mapped_concept_code",
        "mapped_concept_name",
    )
    .write_csv("../nwicu/d_labitems_to_loinc_nwicu.csv")
)
