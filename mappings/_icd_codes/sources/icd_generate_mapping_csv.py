import pandas as pd
import xml.etree.ElementTree as ET
import re

# DIAGNOSES
# generalized equivalence mappings
cm_gem = pd.read_fwf(
    "a | 2018 ICD-10-CM General Equivalence Mappings (GEMS)/2018_I9gem.txt",
    header=None,
    widths=[6, 6],
)
cm_gem.columns = ["icd9", "icd10"]

# ICD-9-CM codes and descriptions
icd9cm_codes = pd.read_fwf(
    "b | Version 32 Full and Abbreviated Code Titles/CMS32_DESC_LONG_DX.txt",
    header=None,
    widths=[6, 512],
    encoding="latin-1",
)
icd9cm_codes.columns = ["icd9", "description"]

icd9_diagnoses = pd.merge(cm_gem, icd9cm_codes, on="icd9", how="right")

# Parse RTF file for ICD-9-CM diagnosis codes
rtf_file_path = "d | 2011/Dtab12.rtf"
diagnoses = []

with open(rtf_file_path, "r", encoding="utf-8", errors="ignore") as f:
    rtf_content = f.read()

# Extract procedure codes and descriptions from RTF content
# Look for patterns like "00.01\tab Description" in the RTF
# The pattern matches ICD-9 procedure codes followed by tab and description
pattern = r"((?:|[EV])\d{2,3}(?:|\.\d{1,2}))\\tab\s*([^\\]+?)(?=\\par|\{|$)"
matches = re.findall(pattern, rtf_content, re.MULTILINE | re.DOTALL)

print(matches)

for code, desc in matches:
    if code and desc:
        # Clean up the description - remove RTF formatting
        clean_desc = re.sub(r"\\[a-z]+\d*\s*", " ", desc)  # Remove RTF commands
        clean_desc = re.sub(r"\{[^}]*\}", "", clean_desc)  # Remove RTF groups
        clean_desc = re.sub(r"\s+", " ", clean_desc).strip()  # Normalize whitespace

        if clean_desc:  # Only add if we have a clean description
            # Remove dots from ICD-9 codes for consistency
            icd9_code_clean = code.replace(".", "")
            diagnoses.append(
                {"icd9": icd9_code_clean, "description": clean_desc}
            )

icd9_rtf_df = pd.DataFrame(diagnoses)

# Merge ICD-9 diagnoses with RTF data, keeping all records
icd9_diagnoses = pd.merge(
    icd9_diagnoses,
    icd9_rtf_df,
    on="icd9",
    how="outer",
    suffixes=("", "_rtf"),
)

# Use ICD-9 description where main description is missing
icd9_diagnoses["description"] = icd9_diagnoses["description"].fillna(
    icd9_diagnoses["description_rtf"]
)

# Drop the temporary ICD-9 description column
icd9_diagnoses = icd9_diagnoses.drop("description_rtf", axis=1)

icd9_diagnoses.to_csv("../icd9_diagnoses.csv", index=False, sep=",")

# ICD-10-CM codes and descriptions
icd10cm_codes = pd.read_fwf(
    "a | 2018 Code Descriptions/icd10cm_codes_2018.txt",
    header=None,
    widths=[7, 512],
)
icd10cm_codes.columns = ["icd10", "description"]

icd10_diagnoses = pd.merge(cm_gem, icd10cm_codes, on="icd10", how="right")

# Parse the XML file directly
xml_file_path = (
    "c | ICD-10-CM FY25, April 1, 2025/icd10cm-tabular-April-2025.xml"
)
tree = ET.parse(xml_file_path)
root = tree.getroot()

diagnoses = []

# Find all diag elements recursively
for diag in root.iter("diag"):
    name_elem = diag.find("name")
    desc_elem = diag.find("desc")

    if name_elem is not None and desc_elem is not None:
        icd10_code = name_elem.text
        description = desc_elem.text

        if icd10_code and description:
            # Remove dot from ICD-10 code
            icd10_code_clean = icd10_code.replace(".", "")
            diagnoses.append(
                {"icd10": icd10_code_clean, "description": description}
            )

icd10_xml_df = pd.DataFrame(diagnoses)

# Merge XML data with existing diagnoses, keeping all records
icd10_diagnoses = pd.merge(
    icd10_diagnoses,
    icd10_xml_df,
    on="icd10",
    how="outer",
    suffixes=("", "_xml"),
)

# Use XML description where main description is missing
icd10_diagnoses["description"] = icd10_diagnoses["description"].fillna(
    icd10_diagnoses["description_xml"]
)
icd10_diagnoses = icd10_diagnoses.drop("description_xml", axis=1)

icd10_diagnoses.to_csv("../icd10_diagnoses.csv", index=False, sep=",")

# PROCEDURES
# generalized equivalence mappings
pcs_gem = pd.read_fwf(
    "a | 2018 ICD-10-PCS General Equivalence Mappings (GEMS)/gem_i9pcs.txt",
    header=None,
    widths=[6, 7],
)
pcs_gem.columns = ["icd9", "icd10"]

# ICD-9-PCS codes and descriptions
icd9pcs_codes = pd.read_fwf(
    "b | Version 32 Full and Abbreviated Code Titles/CMS32_DESC_LONG_SG.txt",
    header=None,
    widths=[5, 512],
    encoding="latin-1",
)
icd9pcs_codes.columns = ["icd9", "description"]

icd9_procedures = pd.merge(pcs_gem, icd9pcs_codes, on="icd9", how="right")
icd9_procedures.to_csv("../icd9_procedures.csv", index=False, sep=",")

# ICD-10-PCS codes and descriptions
icd10pcs_codes = pd.read_fwf(
    "a | 2018 ICD-10-PCS Order File (Long and Abbreviated Titles)/icd10pcs_order_2018.txt",
    header=None,
    widths=[6, 8, 1, 60, 512],
)
icd10pcs_codes.columns = [
    "order",
    "icd10",
    "header",
    "short_description",
    "description",
]
icd10pcs_codes.drop(
    columns=["order", "short_description", "header"], inplace=True
)

icd10_procedures = pd.merge(pcs_gem, icd10pcs_codes, on="icd10", how="right")
icd10_procedures.to_csv("../icd10_procedures.csv", index=False, sep=",")
