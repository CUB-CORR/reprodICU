"""Generate ../eicu_icd9_to_icd10.csv: the ICD-10 code filled in for an ICD-9 code of eICU.

eICU is from 2014 and 2015, so no code can be newer than 2015. The general table of reprodICU is built from the
2018 mapping, which has targets that did not exist then. This file has one ICD-10 code per ICD-9 code in eICU:
  eicu    the ICD-10 code eICU itself pairs with the ICD-9 code (the most frequent one), else
  gem     the first ICD-10 target of the code in the 2015 General Equivalence Mapping (GEM)

Run: python eicu_icd9_to_icd10_generate.py   (eICU location from PATHS.yaml; the 2015 files are in _icd_codes/sources)
"""

from pathlib import Path

import pandas as pd

from reprodICU import reprodICUPaths

MAPPINGS = Path(__file__).resolve().parents[2]
SOURCES = MAPPINGS / "_icd_codes" / "sources"

# generalized equivalence mappings 2015: the first ICD-10 target of an ICD-9 code
gem = pd.read_fwf(
    SOURCES / "e | 2015 ICD-10-CM General Equivalence Mappings (GEMS)/2015_I9gem.txt",
    header=None,
    widths=[6, 8],
    dtype=str,
)
gem.columns = ["icd9", "icd10"]
gem = gem[gem["icd10"] != "NoDx"].drop_duplicates("icd9", keep="first")

# ICD-10-CM codes of 2015
icd10cm_codes = pd.read_fwf(
    SOURCES / "e | 2015 Code Descriptions/icd10cm_order_2015.txt",
    header=None,
    widths=[6, 7],
    dtype=str,
)
icd10cm_codes.columns = ["order", "icd10"]

# eICU codes: a string lists the ICD-9 codes first and then the ICD-10 codes, a code can be repeated
strings = pd.read_csv(
    reprodICUPaths().eicu_source_path + "diagnosis.csv.gz",
    usecols=["icd9code"],
    dtype=str,
)["icd9code"].dropna()
codes = strings.str.replace(".", "", regex=False).str.split(", ")
codes = codes.map(lambda c: list(dict.fromkeys(c)))

# the ICD-10 code eICU pairs with an ICD-9 code: strings with exactly two codes, the second a 2015 ICD-10 code
pairs = pd.DataFrame(codes[codes.str.len() == 2].tolist(), columns=["icd9", "icd10"])
pairs = pairs[pairs["icd10"].isin(icd10cm_codes["icd10"])]
eicu = (
    pairs.value_counts()
    .reset_index(name="count")
    .sort_values(["count", "icd10"], ascending=[False, True])
    .drop_duplicates("icd9", keep="first")
)

eicu_icd9_to_icd10 = pd.merge(gem, eicu, on="icd9", how="left", suffixes=("_gem", "_eicu"))
eicu_icd9_to_icd10["source"] = eicu_icd9_to_icd10["icd10_eicu"].notna().map({True: "eicu", False: "gem"})
eicu_icd9_to_icd10["icd10"] = eicu_icd9_to_icd10["icd10_eicu"].fillna(eicu_icd9_to_icd10["icd10_gem"])

# only the ICD-9 codes that occur in eICU
eicu_icd9_to_icd10 = eicu_icd9_to_icd10[eicu_icd9_to_icd10["icd9"].isin(set(codes.explode()))]
eicu_icd9_to_icd10[["icd9", "icd10", "source"]].sort_values("icd9").to_csv(
    MAPPINGS / "eicu" / "eicu_icd9_to_icd10.csv", index=False, sep=","
)
