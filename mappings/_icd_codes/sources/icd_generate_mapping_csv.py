import numpy as np
import pandas as pd

# DIAGNOSES
# generalized equivalence mappings
cm_gem = pd.read_fwf("2018_I9gem.txt", header=None, widths=[6, 6])
cm_gem.columns = ["icd9", "icd10"]

# ICD-9-CM codes and descriptions
icd9cm_codes = pd.read_fwf(
    "CMS32_DESC_LONG_DX.txt", header=None, widths=[6, 512], encoding="latin-1"
)
icd9cm_codes.columns = ["icd9", "description"]

icd9_diagnoses = pd.merge(cm_gem, icd9cm_codes, on="icd9", how="left")
icd9_diagnoses.to_csv("../icd9_diagnoses.csv", index=False, sep=",")

# ICD-10-CM codes and descriptions
icd10cm_codes = pd.read_fwf("icd10cm_codes_2018.txt", header=None, widths=[7, 512])
icd10cm_codes.columns = ["icd10", "description"]

icd10_diagnoses = pd.merge(cm_gem, icd10cm_codes, on="icd10", how="left")
icd10_diagnoses.to_csv("../icd10_diagnoses.csv", index=False, sep=",")

# PROCEDURES
# generalized equivalence mappings
pcs_gem = pd.read_fwf("gem_i9pcs.txt", header=None, widths=[6, 7])
pcs_gem.columns = ["icd9", "icd10"]

# ICD-9-PCS codes and descriptions
icd9pcs_codes = pd.read_fwf(
    "CMS32_DESC_LONG_SG.txt", header=None, widths=[5, 512], encoding="latin-1"
)
icd9pcs_codes.columns = ["icd9", "description"]

icd9_procedures = pd.merge(pcs_gem, icd9pcs_codes, on="icd9", how="left")
icd9_procedures.to_csv("../icd9_procedures.csv", index=False, sep=",")

# ICD-10-PCS codes and descriptions
icd10pcs_codes = pd.read_fwf("icd10pcs_order_2018.txt", header=None, widths=[6, 8, 1, 60, 512])
icd10pcs_codes.columns = ["order", "icd10", "header", "short_description", "description"]
icd10pcs_codes.drop(columns=["order", "short_description", "header"], inplace=True)

icd10_procedures = pd.merge(pcs_gem, icd10pcs_codes, on="icd10", how="left")
icd10_procedures.to_csv("../icd10_procedures.csv", index=False, sep=",")
