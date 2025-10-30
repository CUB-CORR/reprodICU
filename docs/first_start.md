# First Start

This guide walks you through installing reprodICU, configuring paths to your ICU datasets, and building your first harmonized database.

## Installation

### Prerequisites

- Python 3.10 or later
- pip (Python package installer)
- Access to at least one ICU dataset (eICU, HiRID, MIMIC-III, MIMIC-IV, NWICU, SICdb, or AmsterdamUMCdb)

### Install reprodICU

Install the package with documentation and development dependencies:

```bash
pip install reprodICU
```

Or, if you want to include documentation tools:

```bash
pip install "reprodICU[docs]"
```

## Initial Configuration

### First Run

On your first import or function call, reprodICU automatically creates a user configuration directory:

```text
~/.reprodICU/
└── PATHS.yaml # ← You must edit this
```

### Editing PATHS.yaml

The `PATHS.yaml` file tells reprodICU where your raw datasets are located. Open it and configure paths to your datasets:

```bash
nano ~/.reprodICU/PATHS.yaml
```

Here's what it should look like:

```yaml
eicu_source_path: "/path/to/eICU/" # Path to eICU-CRD dataset
hirid_source_path: "/path/to/hirid/" # Path to HiRID dataset
mimic3_source_path: "/path/to/mimic3/" # Path to MIMIC-III dataset
mimic4_source_path: "/path/to/mimic4/" # Path to MIMIC-IV dataset
nwicu_source_path: "/path/to/nwicu/" # Path to NWICU dataset
sicdb_source_path: "/path/to/sicdb/" # Path to SICdb dataset
umcdb_source_path: "/path/to/umcdb/" # Path to UMCdb dataset

# Output directory where harmonized data will be saved
reprodICU_files_path: "/path/to/output/reprodICU/"
reprodICU_demo_files_path: "/path/to/output/reprodICU_demo/"

# OMOP vocabularies path
OMOP_vocab_path: "/path/to/omop_vocabularies/"
```

**Important**: Only include paths for datasets you have access to. Missing dataset paths will be skipped during processing.

## Obtaining Datasets

### PhysioNet Datasets

All included datasets are publicly available from PhysioNet (https://physionet.org/):

| Dataset   | PhysioNet Link                                                | Size     |
| --------- | ------------------------------------------------------------- | -------- |
| eICU-CRD  | [Link](https://physionet.org/content/eicu-crd/)               | ~5.1 GB  |
| MIMIC-III | [Link](https://physionet.org/content/mimiciii/)               | ~6.2 GB  |
| MIMIC-IV  | [Link](https://physionet.org/content/mimiciv/)                | ~9.9 GB  |
| HiRID     | [Link](https://physionet.org/content/hirid/)                  | ~16.8 GB |
| SICdb     | [Link](https://physionet.org/content/sicdb/)                  | ~2.4 GB  |
| NWICU     | [Link](https://physionet.org/content/nwicu-northwestern-icu/) | ~0.6 GB  |

### Dataset-Specific Access

- **UMCdb**: Contact [Amsterdam UMC](https://amsterdammedicaldatascience.nl/#amsterdamumcdb) (~12.1 GB)

## Building Your First Database

### Quick Start: Demo Mode

To test reprodICU with small datasets, use demo mode:

```python
from reprodICU import build_all

# Build demo database (limited to eICU, MIMIC-III, MIMIC-IV)
files = build_all(
    datasets="all",  # Use all configured datasets in demo mode
    demo=True,
    create_overview=True
)

print(f"Built {len(files)} tables")
for f in files:
    print(f"  ✓ {f}")
```

This creates output in your `reprodICU_demo_output_path`.

### Full Build: All Datasets

```python
from reprodICU import build_all

# Build full database with all datasets
files = build_all(
    datasets=["eICU", "MIMIC3", "MIMIC4"],  # Specify which datasets
    demo=False,
)
```

### Building Specific Tables

Instead of building everything, you can build individual tables:

```python
from reprodICU import (
    build_patient_information,
    build_diagnoses,
    build_procedures,
    build_medications,
    build_microbiology,
    build_notes,
    build_timeseries,
    build_magic_concepts
)

# Patient demographics
build_patient_information(datasets=["eICU", "MIMIC3"])

# Diagnostic codes
build_diagnoses(datasets=["eICU", "MIMIC3"])

# Procedures and interventions
build_procedures(datasets=["eICU", "MIMIC3"])

# Medication records
build_medications(datasets=["eICU", "MIMIC3"])

# Microbiology cultures
build_microbiology(datasets=["eICU", "MIMIC3"])

# Clinical notes
build_notes(datasets=["eICU", "MIMIC3"])

# Timeseries data (vitals, labs, respiratory, intake/output)
build_timeseries(
    datasets=["eICU", "MIMIC3"],
    timeseries=["vitals", "labs", "respiratory", "inout"],
)

# Derived clinical concepts
build_magic_concepts(
    datasets=["eICU", "MIMIC3"],
    concepts=["CODE_STATUS", "RECEIVED_ANY_ANTIBIOTICS", "VENTILATION_DURATION"]
)
```

## Understanding the Processing Pipeline

reprodICU processes data through sequential stages:

### Stage 1: Extraction (A_extract)

Extracts raw data from each dataset's source files (CSVs, parquets, etc.).

### Stage 2: Processing (B_process)

Cleans and validates extracted data, handling missing values and inconsistencies.

### Stage 3: Harmonization (C_harmonize)

Maps variables and units to common definitions across all datasets.

### Stage 4: Quality Control (X1_clean → X4_resample)

- **X1_clean**: Remove implausible values, map diagnoses to standard codes
- **X2_winsorize**: Cap extreme values
- **X3_impute**: Fill missing values using statistical methods
- **X4_resample**: Standardize temporal resolution (e.g., hourly)

### Stage 5: Derivation (Y_MAGIC_CONCEPTS)

Compute high-level clinical concepts (e.g., sepsis status, code status) from raw data.

## Output Files

By default, all harmonized tables are saved as **Parquet files** for efficient storage and access:

```
/path/to/output/reprodICU/
├── diagnoses.parquet                     # ICD codes
├── medications_prescribed.parquet        # Prescribed
├── medications.parquet                   # Administered
├── microbiology.parquet                  # Cultures
├── notes.parquet                         # Clinical notes
├── patient_information.parquet           # Demographics
├── procedures.parquet                    # Procedures
├── timeseries_intakeoutput.parquet       # I/O records
├── timeseries_labs_winsorized.parquet    # Cleaned labs
├── timeseries_labs.parquet               # Lab values
├── timeseries_respiratory.parquet        # Ventilation params
├── timeseries_vitals.parquet             # Heart rate, BP, etc.
└── MAGIC_CONCEPTS
    ├── CODE_STATUS.parquet
    ├── RECEIVED_ANY_ANTIBIOTICS.parquet
    └── ...  # Other concepts
```

## Reading and Exploring Data

### Load Parquet Files

Use Polars (native) or Pandas for analysis:

```python
import polars as pl
import reprodICU

# Load using Polars (recommended for large files)
patient_info = reprodICU.patient_information

# Inspect schema
print(patient_info.schema)

# Check size
print(f"Total rows: {patient_info.height}")
print(f"Total columns: {patient_info.width}")

# View sample
print(patient_info.head())
```

For working with Pandas, you need to convert the polars DataFrame:

```python
import pandas as pd
import reprodICU

# Load using Polars and convert to Pandas
patient_info = reprodICU.patient_information.collect().to_pandas()
```

## Common Issues and Solutions

### Issue: "Config file not found"

**Cause**: `PATHS.yaml` not configured

**Solution**:

```bash
cat ~/.reprodICU/PATHS.yaml  # Check file exists
nano ~/.reprodICU/PATHS.yaml  # Edit paths
```

### Issue: "Dataset path does not exist"

**Cause**: Path in `PATHS.yaml` is incorrect or inaccessible

**Solution**:

```bash
# Verify dataset exists
ls -la /path/to/eICU-CRD/

# Update PATHS.yaml with correct path
```

### Issue: Out of Memory

**Cause**: Dataset too large for available RAM

**Solution**: Use demo mode first, or process one dataset at a time:

```python
build_timeseries(datasets=["eICU"], demo=True)
```

### Issue: "Missing required dataset file"

**Cause**: Dataset incomplete or in wrong format

**Solution**: Verify dataset was downloaded completely and follows expected directory structure.

## Next Steps

- **Learn about utilities**: Head to [Usage & Utilities](/usage_utilities/) to compute clinical scores and predict mortality
- **Understand your data**: Read [Data Structure](/data_structure/) for detailed schema information
- **Explore MAGIC_CONCEPTS**: See [MAGIC_CONCEPTS](/magic_concepts/) to learn about derived clinical variables
- **Browse API**: Visit [API Reference](/api_reference/) for complete function documentation

## Troubleshooting

For additional help:

- **GitHub Issues**: [CUB-CORR/reprodICU/issues](https://github.com/CUB-CORR/reprodICU/issues)
- **Documentation**: Full docs at [reprodICU Documentation](https://reprodicu.notion.site)
