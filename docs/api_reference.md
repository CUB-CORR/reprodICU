# API Reference

This section documents the main reprodICU functions for building harmonized datasets.

All functions are in the `reprodICU` module and use Polars LazyFrames for efficient processing.

## Main Build Functions

### build_all()

Build all tables and timeseries from raw data sources.

**Source**: [reprodICU.py](https://github.com/CUB-CORR/reprodICU/blob/main/src/reprodICU/reprodICU.py#L734-L842)

```python
from reprodICU import build_all

files = build_all(
    paths: Optional[reprodICUPaths] = None,
    datasets: Optional[List[str]] = None,
    demo: bool = False,
    create_overview: bool = True
) -> List[str]
```

**Arguments**:

- `paths` – Configuration object. Auto-loads from ConfigManager if None.
- `datasets` – Datasets to process. None/"all" = all configured datasets. Demo restricts to eICU, MIMIC3, MIMIC4.
- `demo` – If True, use demo-sized datasets (subset for testing)
- `create_overview` – If True, generate data availability overview

**Returns**: List of output file paths created

**Example**:

```python
files = build_all(
    datasets=["eICU", "MIMIC3", "MIMIC4"],
)
```

## Patient Information

### build_patient_information()

Build patient demographics, anthropometrics, and admission metadata.

**Source**: [reprodICU.py](https://github.com/CUB-CORR/reprodICU/blob/main/src/reprodICU/reprodICU.py#L156-L210)

```python
from reprodICU import build_patient_information

files = build_patient_information(
    paths: Optional[reprodICUPaths] = None,
    datasets: Optional[List[str]] = None,
    demo: bool = False
) -> List[str]
```

**Output**: `patient_information.parquet`

**Example**:

```python
build_patient_information(datasets=["eICU", "MIMIC3"])
```

## Diagnoses

### build_diagnoses()

Build harmonized diagnosis codes (ICD-9, ICD-10).

**Source**: [reprodICU.py](https://github.com/CUB-CORR/reprodICU/blob/main/src/reprodICU/reprodICU.py#L213-L261)

```python
from reprodICU import build_diagnoses

files = build_diagnoses(
    paths: Optional[reprodICUPaths] = None,
    datasets: Optional[List[str]] = None,
    demo: bool = False
) -> List[str]
```

**Output**: `diagnoses.parquet`

## Procedures

### build_procedures()

Build harmonized procedure codes.

**Source**: [reprodICU.py](https://github.com/CUB-CORR/reprodICU/blob/main/src/reprodICU/reprodICU.py#L264-L310)

```python
from reprodICU import build_procedures(
    paths: Optional[reprodICUPaths] = None,
    datasets: Optional[List[str]] = None,
    demo: bool = False
) -> List[str]
```

**Output**: `procedures.parquet`

## Medications

### build_medications()

Build administered and prescribed medication records.

**Source**: [reprodICU.py](https://github.com/CUB-CORR/reprodICU/blob/main/src/reprodICU/reprodICU.py#L313-L370)

```python
from reprodICU import build_medications(
    paths: Optional[reprodICUPaths] = None,
    datasets: Optional[List[str]] = None,
    demo: bool = False
) -> List[str]
```

**Output**:

- `medications.parquet` (administered)
- `medications_prescribed.parquet` (prescribed)

## Microbiology

### build_microbiology()

Build culture and susceptibility data.

**Source**: [reprodICU.py](https://github.com/CUB-CORR/reprodICU/blob/main/src/reprodICU/reprodICU.py#L373-L421)

```python
from reprodICU import build_microbiology(
    paths: Optional[reprodICUPaths] = None,
    datasets: Optional[List[str]] = None,
    demo: bool = False
) -> List[str]
```

**Output**: `microbiology.parquet`

## Clinical Notes

### build_notes()

Build clinical notes and documentation.

**Source**: [reprodICU.py](https://github.com/CUB-CORR/reprodICU/blob/main/src/reprodICU/reprodICU.py#L424-L472)

```python
from reprodICU import build_notes(
    paths: Optional[reprodICUPaths] = None,
    datasets: Optional[List[str]] = None,
    demo: bool = False
) -> List[str]
```

**Output**: `notes.parquet`

## Timeseries Data

### build_timeseries()

Build vital signs, laboratories, respiratory parameters, and intake/output records with optional imputation and resampling.

**Source**: [reprodICU.py](https://github.com/CUB-CORR/reprodICU/blob/main/src/reprodICU/reprodICU.py#L475-L640)

```python
from reprodICU import build_timeseries

files = build_timeseries(
    paths: Optional[reprodICUPaths] = None,
    datasets: Optional[List[str]] = None,
    timeseries: Optional[List[str]] = None,
    demo: bool = False,
    impute: bool = False,
    resample: Optional[int] = None
) -> List[str]
```

**Arguments**:

- `timeseries` – Types to extract: "vitals", "labs", "respiratory", "inout". None/"all" = all types
- `impute` – Impute missing vitals
- `resample` – Resample to specified seconds (e.g., 3600 for hourly). Also generates `_winsorized` (labs) and `_imputed`/`_resampled` variants

**Output**:

- `timeseries_vitals.parquet`
- `timeseries_labs.parquet` (+ `_winsorized.parquet`)
- `timeseries_respiratory.parquet`
- `timeseries_intakeoutput.parquet` (+ `_balanced.parquet`)
- Optional: `_imputed.parquet`, `_resampled.parquet` for vitals

**Example**:

```python
# Full timeseries with imputation and hourly resampling
build_timeseries(
    datasets=["eICU", "MIMIC3"],
    timeseries=["vitals", "labs", "respiratory", "inout"],
    impute=True,
    resample=3600
)
```

## MAGIC_CONCEPTS

### build_magic_concepts()

Extract pre-defined clinical concepts from raw data.

**Source**: [MAGIC_CONCEPTS.py](https://github.com/CUB-CORR/reprodICU/blob/main/src/reprodICU/helpers/MAGIC_CONCEPTS.py)

```python
from reprodICU.helpers.MAGIC_CONCEPTS import build_magic_concepts

result = build_magic_concepts(
    paths: Optional[reprodICUPaths] = None,
    datasets: Optional[List[str]] = None,
    concepts: Optional[List[str]] = None,
    demo: bool = False
) -> Dict[str, List[str]]
```

**Arguments**:

- `paths` – Configuration object. Auto-loads from ConfigManager if None.
- `datasets` – Datasets to process. None/"all" = all configured datasets. Supported: eICU, HiRID, MIMIC3, MIMIC4, SICdb, UMCdb (demo restricts to eICU, MIMIC3, MIMIC4).
- `concepts` – Concepts to extract: "CODE_STATUS", "RECEIVED_ANY_ANTIBIOTICS", "VENTILATION_DURATION", "RENAL_REPLACEMENT_THERAPY_DURATION", "SEVERITY_SCORES". None/"all" = all concepts
- `demo` – If True, use demo-sized datasets

**Returns**: Dict mapping concept names to output file paths

**Output Files**:

- `MAGIC_CONCEPTS/CODE_STATUS.parquet`
- `MAGIC_CONCEPTS/RECEIVED_ANY_ANTIBIOTICS.parquet`
- `MAGIC_CONCEPTS/VENTILATION_DURATION.parquet`
- `MAGIC_CONCEPTS/RENAL_REPLACEMENT_THERAPY_DURATION.parquet`
- `MAGIC_CONCEPTS/SEVERITY_SCORES.parquet`

**Example**:

```python
# Extract all MAGIC_CONCEPTS
result = build_magic_concepts(datasets="all", concepts="all")

# Extract specific concepts
result = build_magic_concepts(
    datasets=["eICU", "MIMIC3"],
    concepts=["VENTILATION_DURATION", "RECEIVED_ANY_ANTIBIOTICS"]
)
```

See [MAGIC_CONCEPTS documentation](/magic_concepts/) for details on each concept.

## Utilities

### Mortality Measures

```python
from reprodICU.utils.mortality import COMMON_MORTALITY_MEASURES

mortality_df = COMMON_MORTALITY_MEASURES(
    patient_information: Optional[pl.LazyFrame] = None
) -> pl.LazyFrame
```

Computes mortality at timepoints: 7d, 28d, 30d, 90d, 180d, 360d, 365d.

See [Usage & Utilities: Mortality](/usage_utilities/#mortality-prediction).

### Clinical Scores

```python
from reprodICU.utils.scores import SOFA, SOFA2, OASIS, VIS

sofa = SOFA()
sofa2 = SOFA2()
apache = OASIS()
vis = VIS()
```

See [Usage & Utilities: Clinical Scoring Systems](/usage_utilities/#clinical-scoring-systems).

### Comorbidity

Comorbidity indices using diagnosis codes.

```python
from reprodICU.utils.comorbidity import CHARLSON, ELIXHAUSER, GAGNE

# Charlson Comorbidity Index
charlson = CHARLSON(
    diagnoses: Optional[pl.LazyFrame] = None,
    patient_information: Optional[pl.LazyFrame] = None,
    return_categories: bool = False
) -> pl.DataFrame

# Elixhauser Comorbidity Index (with van Walraven weights)
elixhauser = ELIXHAUSER(
    diagnoses: Optional[pl.LazyFrame] = None,
    patient_information: Optional[pl.LazyFrame] = None,
    return_categories: bool = False
) -> pl.DataFrame

# Gagne Comorbidity Index
gagne = GAGNE(
    diagnoses: Optional[pl.LazyFrame] = None,
    patient_information: Optional[pl.LazyFrame] = None,
    return_categories: bool = False
) -> pl.DataFrame
```

See [Usage & Utilities: Comorbidity](/usage_utilities/#comorbidity-assessment).

### Sepsis Detection

Sepsis definitions using Sepsis-3 and alternative criteria.

```python
from reprodICU.utils.sepsis import SEPSIS, ANGUS_SEPSIS, MARTIN_SEPSIS

# Sepsis-3 (Seymour, Shah, Rhee definitions)
sepsis = SEPSIS() -> pl.LazyFrame

# Angus sepsis criteria
angus = ANGUS_SEPSIS() -> pl.LazyFrame

# Martin sepsis criteria
martin = MARTIN_SEPSIS() -> pl.LazyFrame
```

**Note**: Sepsis functions return long-format data with timeframe-level granularity (multiple rows per admission).

See [Usage & Utilities: Sepsis](/usage_utilities/#sepsis-detection).

## Configuration

### ConfigManager

Manages YAML configuration files with user overrides.

**Source**: [config.py](https://github.com/CUB-CORR/reprodICU/blob/main/src/reprodICU/config.py#L20-L120)

```python
from reprodICU.config import ConfigManager

config = ConfigManager()

# Load config
paths_config = config.load_config("PATHS.yaml", user_override=True)

# Get config file path
path = config.get_config_path("COLUMN_NAMES.yaml")
```

### reprodICUPaths

Convenient access to configured paths.

**Source**: [config.py](https://github.com/CUB-CORR/reprodICU/blob/main/src/reprodICU/config.py#L150-L230)

```python
from reprodICU.config import reprodICUPaths, get_config_manager

config = get_config_manager()
paths = reprodICUPaths(config)

# Access paths
print(paths.reprodICU_files_path)
print(paths.reprodICU_demo_files_path)
```

## Data Access

### Loading Data

```python
import polars as pl
import pandas as pd

# Polars (recommended for large files)
patient_info = pl.read_parquet("reprodICU/patient_information.parquet")
labs = pl.scan_parquet("reprodICU/timeseries_labs.parquet")  # Lazy loading

# Pandas
df = pd.read_parquet("reprodICU/patient_information.parquet")
```

### Querying Data

```python
import polars as pl

# Filter and select
labs = pl.scan_parquet("reprodICU/timeseries_labs.parquet").filter(
    pl.col("Lab Name") == "Hemoglobin"
).select(
    ["Global ICU Stay ID", "Lab Time", "Value", "Unit"]
).collect()

# Group and aggregate
summary = labs.group_by("Lab Name").agg([
    pl.col("Value").mean().alias("mean"),
    pl.col("Value").median().alias("median"),
    pl.col("Value").std().alias("std"),
])
```

## Module Structure

```
reprodICU/
├── reprodICU.py              # Main build functions
├── config.py                 # Configuration management
├── helpers/
│   ├── A_extract/            # Dataset extraction
│   ├── B_process/            # Data processing
│   ├── C_harmonize/          # Harmonization
│   ├── X1_clean/             # Cleaning
│   ├── X2_winsorize/         # Winsorization
│   ├── X3_impute/            # Imputation
│   ├── X4_resample/          # Resampling
│   └── Y_MAGIC_CONCEPTS/     # Concept extraction
├── utils/
│   ├── mortality.py          # Mortality measures
│   ├── comorbidity.py        # Comorbidity scores
│   ├── sepsis.py             # Sepsis detection
│   ├── scores/               # Clinical scoring systems
│   └── clinical/             # Other clinical utilities
├── interfaces/               # Format conversions (OMOP, CLIF, MEDS)
└── mappings/                 # Variable and concept mappings
```

## Error Handling

Common errors and solutions:

**ImportError: Cannot import 'build_all'**

```python
# Install package first
# pip install reprodICU

from reprodICU import build_all
```

**FileNotFoundError: Config file not found**

```bash
# Edit ~/.reprodICU/PATHS.yaml
nano ~/.reprodICU/PATHS.yaml
```

**ValueError: Invalid dataset selection**

```python
# Valid datasets
valid = ["eICU", "MIMIC3", "MIMIC4", "HiRID", "SICdb", "UMCdb", "NWICU"]
build_all(datasets=valid)
```

## See Also

- [First Start Guide](/first_start/) – Installation and setup
- [Usage & Utilities](/usage_utilities/) – Computing scores and outcomes
- [Data Structure](/data_structure/) – Output schemas
- [MAGIC_CONCEPTS](/magic_concepts/) – Derived clinical variables
- [Datasets](/datasets/) – Available data sources

## GitHub Repository

For source code, issues, and contributions:

[github.com/CUB-CORR/reprodICU](https://github.com/CUB-CORR/reprodICU)
