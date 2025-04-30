# `X3_impute.py`

`X3_impute.py` applies multiple different imputers to the different tables created by `X0_raw_harmonize.py`.

## `DiagnosesImputer`

The `DiagnosesImputer` works on the `diagnoses` table and does the following things:

- ICD codes that are currently saved as ICD code and corresponding ICD version in the columns `diagnosis_icd_code` and `diagnosis_icd_version` respectively are coalesced.
  - ICD9 codes are mapped to ICD10 codes and vice versa based on the ICD-10 CM _2018 General Equivalence Mappings_ by the US _Centers for Medicare & Medicaid Services_. Links to the specific sources used and the script to generate the mapping files from the raw data are provided within `mappings/_icd_codes/sources`.
  - Duplicate rows are dropped (since some Databases include both ICD9 and ICD10 codes for the same cases)
- Since MIMIC-III/-IV provides ICD codes only on hospital stay basis (source: "_[The ICD codes are generated for billing purposes at the end of the hospital stay.](https://mimic.mit.edu/docs/iii/tables/diagnoses_icd/)_"), the table is exploded for each ICU stay that is associated with a hospital stay (i.e., **all ICU stays are assumed to have the same diagnoses**).
  - This should be mostly unbiased with regards to comorbidity calculations (e.g. Elixhauser oder Charlston comorbidity indices).
