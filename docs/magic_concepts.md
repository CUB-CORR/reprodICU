# MAGIC_CONCEPTS: Derived Clinical Variables

MAGIC_CONCEPTS are pre-defined clinical concepts derived from raw data. Rather than computing these yourself, reprodICU extracts them directly from each dataset's source data.

## What are MAGIC_CONCEPTS?

MAGIC_CONCEPTS are **derived variables** computed from **axioms** (raw measurements) using standardized clinical definitions. Each MAGIC_CONCEPT:

1. **Extracts** raw data from each dataset's source files
2. **Transforms** values using dataset-specific rules
3. **Harmonizes** across datasets to produce a unified definition
4. **Validates** against clinical plausibility

For example, `VENTILATION_DURATION` extracts mechanical ventilation flags from each dataset, maps them to a common "ventilated/not ventilated" status, and calculates continuous duration.

## Available MAGIC_CONCEPTS

reprodICU includes 5 main MAGIC_CONCEPTS:

1. **CODE_STATUS** – Resuscitation status limitations
2. **RECEIVED_ANY_ANTIBIOTICS** – Whether antibiotics were given
3. **VENTILATION_DURATION** – Duration of mechanical ventilation
4. **RENAL_REPLACEMENT_THERAPY_DURATION** – Duration of dialysis
5. **SEVERITY_SCORES** – Pre-extracted severity scores (APACHE, SOFA, SAPS III)

## CODE_STATUS

Indicates limitations on resuscitation efforts.

### Definition

One of:

- **Full Code** – Full resuscitation if cardiopulmonary arrest
- **DNR** – Do Not Resuscitate
- **DNI** – Do Not Intubate
- **DNCPR** – Do Not Attempt Cardiopulmonary Resuscitation
- **CMO** – Comfort Measures Only
- **Unknown** – Not recorded

### Dataset Extraction

- **eICU**: From `carePlanGeneral` table, "Care Limitation" group
- **MIMIC-III/IV**: From `chartevents` chart items related to code status
- **HiRID**: From assessment records
- **SICdb**: From case records
- **UMCdb**: From assessment protocols

### Usage

```python
from reprodICU.helpers.MAGIC_CONCEPTS import build_magic_concepts

# Extract CODE_STATUS
build_magic_concepts(
    datasets=["eICU", "MIMIC3", "MIMIC4"],
    concepts=["CODE_STATUS"]
)
```

### Output Schema

| Column               | Type      | Description                 |
| -------------------- | --------- | --------------------------- |
| `Global ICU Stay ID` | String    | Unique admission identifier |
| `Code Status`        | String    | One of above values         |
| `Code Status Time`   | Timestamp | When status changed         |
| `Source Dataset`     | String    | Dataset of origin           |

## RECEIVED_ANY_ANTIBIOTICS

Binary indicator of antibiotic exposure during ICU stay.

### Definition

- **True** – One or more antibiotics administered
- **False** – No antibiotics administered

Broad antibiotic spectrum includes: beta-lactams, aminoglycosides, fluoroquinolones, macrolides, glycopeptides, anti-fungals, antivirals, antiparasitics.

### Dataset Extraction

- **eICU**: From `medication` table, drug name/class mapping
- **MIMIC-III/IV**: From `prescriptions` and `inputevents` tables
- **HiRID**: From medication administration records
- **SICdb**: From case medication records
- **UMCdb**: From administration protocols

### Usage

```python
from reprodICU.helpers.MAGIC_CONCEPTS import build_magic_concepts

build_magic_concepts(
    datasets=["eICU", "MIMIC3"],
    concepts=["RECEIVED_ANY_ANTIBIOTICS"]
)
```

### Output Schema

| Column                  | Type      | Description                        |
| ----------------------- | --------- | ---------------------------------- |
| `Global ICU Stay ID`    | String    | Unique admission identifier        |
| `Received Antibiotics`  | Boolean   | True if any antibiotic given       |
| `First Antibiotic Time` | Timestamp | When first antibiotic administered |
| `Source Dataset`        | String    | Dataset of origin                  |

## VENTILATION_DURATION

Duration and timing of mechanical ventilation support.

### Definition

Calculates:

- Whether patient was mechanically ventilated
- Start time of ventilation
- End time of ventilation
- Total duration (hours)
- Number of separate ventilation episodes

### Dataset Extraction

- **eICU**: From `respiratoryCharting` table, mechanical ventilation flags
- **MIMIC-III/IV**: From `chartevents`, ventilation mode items
- **HiRID**: From ventilation parameter recordings
- **SICdb**: From mechanical ventilation records
- **UMCdb**: From ventilation waveform and parameter data

### Usage

```python
from reprodICU.helpers.MAGIC_CONCEPTS import build_magic_concepts

build_magic_concepts(
    datasets=["eICU", "MIMIC3", "MIMIC4"],
    concepts=["VENTILATION_DURATION"]
)
```

### Output Schema

| Column                               | Type      | Description                    |
| ------------------------------------ | --------- | ------------------------------ |
| `Global ICU Stay ID`                 | String    | Unique admission identifier    |
| `Ventilated`                         | Boolean   | Whether ventilated during stay |
| `Ventilation Start Time`             | Timestamp | First ventilation start        |
| `Ventilation End Time`               | Timestamp | Last ventilation end           |
| `Total Ventilation Duration (hours)` | Float     | Sum of all ventilation periods |
| `Number of Ventilation Episodes`     | Integer   | How many separate periods      |
| `Source Dataset`                     | String    | Dataset of origin              |

## RENAL_REPLACEMENT_THERAPY_DURATION

Duration of renal replacement therapy (dialysis/CRRT).

### Definition

Calculates:

- Whether patient received RRT
- Start and end times of RRT
- Total duration (hours)
- Number of separate RRT episodes
- Modality (CRRT, IHD, SLED, etc.) if available

### Dataset Extraction

- **eICU**: From procedures and medication administration
- **MIMIC-III/IV**: From `procedures_icd` and `inputevents`
- **HiRID**: From procedure records
- **SICdb**: From case procedures and therapies
- **UMCdb**: From therapy administration records

### Usage

```python
from reprodICU.helpers.MAGIC_CONCEPTS import build_magic_concepts

build_magic_concepts(
    datasets=["eICU", "MIMIC3", "MIMIC4"],
    concepts=["RENAL_REPLACEMENT_THERAPY_DURATION"]
)
```

### Output Schema

| Column                       | Type      | Description                     |
| ---------------------------- | --------- | ------------------------------- |
| `Global ICU Stay ID`         | String    | Unique admission identifier     |
| `Received RRT`               | Boolean   | Whether RRT was provided        |
| `RRT Start Time`             | Timestamp | First RRT start                 |
| `RRT End Time`               | Timestamp | Last RRT end                    |
| `Total RRT Duration (hours)` | Float     | Sum of all RRT periods          |
| `Number of RRT Episodes`     | Integer   | How many separate periods       |
| `RRT Modality`               | String    | "CRRT", "IHD", "SLED", or other |
| `Source Dataset`             | String    | Dataset of origin               |

## SEVERITY_SCORES

Pre-extracted severity scores directly from source datasets. These are not recalculated by reprodICU; they are extracted as-is from each dataset.

### Available Scores

**By Dataset**:

| Score      | eICU | MIMIC-III | MIMIC-IV | HiRID | SICdb | UMCdb |
| ---------- | ---- | --------- | -------- | ----- | ----- | ----- |
| APACHE II  | ✓    | ✓         | ✓        | —     | —     | ✓     |
| APACHE III | ✓    | ✓         | ✓        | —     | —     | ✓     |
| APACHE IV  | ✓    | —         | —        | —     | —     | ✓     |
| APS III    | ✓    | ✓         | ✓        | —     | —     | —     |
| SOFA       | —    | ✓         | ✓        | —     | —     | —     |
| SAPS III   | —    | —         | —        | —     | ✓     | —     |

### Extraction Details

**eICU** (`apachePatientResult.csv`):

- APACHE IV: `apachescore` (when `apacheversion == "IV"`)
- APS III: `acutephysiologyscore`

**MIMIC-III/IV** (`chartevents.csv`):

- APACHE II: itemid 226743
- APACHE III: itemid 226991
- APS III: itemid 226996
- SOFA: itemid 227428

**SICdb** (`cases.csv`):

- SAPS III: `saps3` field

**UMCdb** (`numericitems.parquet`):

- APACHE II: itemid 19499
- APACHE III: itemid 19750
- APACHE IV: itemid 19500
- SAPS II: itemid 19503

### Usage

```python
from reprodICU.helpers.MAGIC_CONCEPTS import build_magic_concepts

build_magic_concepts(
    datasets=["eICU", "MIMIC3", "MIMIC4"],
    concepts=["SEVERITY_SCORES"]
)
```

### Output Schema

| Column               | Type      | Description                        |
| -------------------- | --------- | ---------------------------------- |
| `Global ICU Stay ID` | String    | Unique admission identifier        |
| `Score Name`         | String    | Score type (APACHE II, SOFA, etc.) |
| `Score Value`        | Float     | Numeric score value                |
| `Score Time`         | Timestamp | When score was calculated/recorded |
| `Source Dataset`     | String    | Dataset of origin                  |

## Building MAGIC_CONCEPTS

### All Concepts

```python
from reprodICU.helpers.MAGIC_CONCEPTS import build_magic_concepts

# Extract all MAGIC_CONCEPTS
build_magic_concepts(
    datasets=["eICU", "MIMIC3", "MIMIC4"],
    concepts="all"
)
```

### Specific Concepts

```python
# Extract only CODE_STATUS and VENTILATION_DURATION
build_magic_concepts(
    datasets=["eICU", "MIMIC3"],
    concepts=["CODE_STATUS", "VENTILATION_DURATION"]
)
```

### Demo Mode

```python
# Use smaller demo datasets
build_magic_concepts(
    datasets="all",
    concepts="all",
    demo=True  # Only eICU, MIMIC3, MIMIC4
)
```

## Output Files

MAGIC_CONCEPTS are saved as separate parquet files:

```
reprodICU/
├── MAGIC_CONCEPTS/
│   ├── CODE_STATUS.parquet
│   ├── RECEIVED_ANY_ANTIBIOTICS.parquet
│   ├── VENTILATION_DURATION.parquet
│   ├── RENAL_REPLACEMENT_THERAPY_DURATION.parquet
│   └── SEVERITY_SCORES.parquet
```

## Accessing MAGIC_CONCEPTS

```python
import polars as pl

# Load CODE_STATUS
code_status = pl.read_parquet("reprodICU/MAGIC_CONCEPTS/CODE_STATUS.parquet")

# View
print(code_status.head())

# Filter for specific dataset
eicu_antibiotics = pl.read_parquet(
    "reprodICU/MAGIC_CONCEPTS/RECEIVED_ANY_ANTIBIOTICS.parquet"
).filter(pl.col("Source Dataset") == "eICU")

# Join with patient info for analysis
patient_info = pl.read_parquet("reprodICU/patient_information.parquet")
combined = (
    patient_info
    .join(code_status, on="Global ICU Stay ID")
)

print(combined)
```

## Rationale: Why MAGIC_CONCEPTS?

MAGIC_CONCEPTS address critical challenges:

1. **Definition Heterogeneity**: Different datasets define the same concept differently. MAGIC_CONCEPTS provide standard definitions across all datasets.

2. **Data Location**: These concepts are recorded in different places across datasets. MAGIC_CONCEPTS automatically locate and extract them.

3. **Validation**: Extracted concepts are validated against clinical plausibility rules.

4. **Reproducibility**: Same code produces identical results across datasets, enabling valid cross-dataset comparisons.

5. **Efficiency**: Pre-extracted concepts avoid redundant computation during analysis.

## Related: MAGIC_VARIABLES (Future)

MAGIC_VARIABLES are a planned extension providing:

- **Derived physiologic metrics** (e.g., MAP from systolic/diastolic)
- **Calculated severity indices** (recalculated APACHE, SOFA)
- **Normalized lab values** (z-scores, reference ranges)
- **Temporal aggregates** (daily minimums, maximums, averages)

Currently these can be computed from the base data using the utilities in [Usage & Utilities](/usage_utilities/).