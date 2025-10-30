# Datasets

reprodICU harmonizes clinical data from 7 international ICU databases. This section describes each dataset, its characteristics, and what clinical data it contains.

## Overview

### Dataset summary

| Dataset        | Country     |  Timespan | Admissions | Patients | Source                                                   |
| -------------- | ----------- | --------: | ---------: | -------: | -------------------------------------------------------- |
| eICU-CRD       | USA         | 2014–2015 |    200,859 |  139,367 | <https://physionet.org/content/eicu-crd/>                |
| MIMIC-III      | USA         | 2001–2012 |     61,532 |   46,476 | <https://physionet.org/content/mimiciii/>                |
| MIMIC-IV       | USA         | 2008–2019 |     94,458 |   65,366 | <https://physionet.org/content/mimiciv/>                 |
| NWICU          | USA         | 2013–2019 |     28,612 |   23,204 | <https://physionet.org/content/nwicu-northwestern-icu/>  |
| AmsterdamUMCdb | Netherlands | 2008–2016 |     23,106 |   20,109 | <https://amsterdammedicaldatascience.nl/#amsterdamumcdb> |
| HiRID          | Switzerland | 2008–2016 |     33,905 |        0 | <https://physionet.org/content/hirid/1.1.1/>             |
| SICdb          | Austria     | 2006–2018 |     27,350 |   21,566 | <https://physionet.org/content/sicdb/>                   |

**Total**: 469,822 admissions across 4 countries over 2001–2022

## eICU Collaborative Research Database (eICU-CRD)

### Overview

### Data Available

## MIMIC-III: Medical Information Mart for Intensive Care III

## MIMIC-IV: Medical Information Mart for Intensive Care IV

## HiRID: High time-Resolution ICU Dataset

## SICdb: Salzburg Intensive Care Database

## AmsterdamUMCdb

## NWICU: Northwestern ICU Database

## Data Coverage by Dataset

### Timeseries Data Type Availability

| Dataset   | Vitals | Labs    | Respiratory | Intake/Output |
| --------- | ------ | ------- | ----------- | ------------- |
| eICU-CRD  | ✓      | ✓       | ✓           | ✓             |
| MIMIC-III | ✓      | ✓       | ✓           | ✓             |
| MIMIC-IV  | ✓      | ✓       | ✓           | ✓             |
| HiRID     | ✓      | ✓       | ✓           | ✓             |
| SICdb     | ✓      | ✓       | ✓           | ✓             |
| UMCdb     | ✓      | ✓       | ✓           | ✓             |
| NWICU     | ✓      | Partial | X           | X             |

### Clinical Data Availability

| Dataset   | Diagnoses             | Medications | Procedures | Microbiology | Notes                                                                |
| --------- | --------------------- | ----------- | ---------- | ------------ | -------------------------------------------------------------------- |
| eICU-CRD  | ICD-9/10              | ✓           | ✓          | ✓            | X                                                                    |
| MIMIC-III | ICD-9                 | ✓           | ✓          | ✓            | ✓                                                                    |
| MIMIC-IV  | ICD-9/10              | ✓           | ✓          | ✓            | [separate Dataset](https://physionet.org/content/mimic-iv-note/2.2/) |
| HiRID     | X                     | ✓           | X          | X            | X                                                                    |
| SICdb     | ICD-10 (only primary) | ✓           | ✓          | X            | X                                                                    |
| UMCdb     | X (only APACHE)       | ✓           | ✓          | ✓            | X                                                                    |
| NWICU     | ICD-10                | ✓           | ✓          | X            | X                                                                    |

## Combining Datasets

reprodICU automatically handles cross-dataset harmonization. You can analyze all datasets together or select subsets:

```python
from reprodICU import build_all

# Build with all 7 datasets
build_all(datasets=["eICU", "MIMIC3", "MIMIC4", "HiRID", "SICdb", "UMCdb", "NWICU"])

# Build with subset
build_all(datasets=["eICU", "MIMIC3", "MIMIC4"])

# Demo mode uses fast datasets
build_all(datasets="all", demo=True)  # Only eICU, MIMIC3, MIMIC4
```

All harmonized tables automatically identify dataset origin through their `Global ICU Stay ID` column, enabling:

- Subgroup analyses by dataset
- Meta-analyses across datasets
- Dataset-specific quality assessment
- Cross-validation studies

## Citation

When using reprodICU with specific datasets, please cite both this package and the original dataset papers:

- **eICU-CRD**: [Pollard et al., 2018](https://doi.org/10.1038/sdata.2018.178)
- **MIMIC-III**: [Johnson et al., 2016](https://doi.org/10.1038/sdata.2016.35)
- **MIMIC-IV**: [Johnson et al., 2023](https://doi.org/10.1038/s41597-022-01899-x)
- **HiRID**: [Hyland et al., 2020](https://doi.org/10.1038/s41591-020-0789-4)
- **SICdb**: [Rodemund et al., 2024](https://doi.org/10.1038/s41597-024-03164-9)
- **AmsterdamUMCdb**: [Thoral et al., 2021](https://doi.org/10.1097/CCM.0000000000004916)
- **NWICU**:
