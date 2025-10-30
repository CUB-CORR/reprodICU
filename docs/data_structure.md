# Data Structure & Schemas

This section describes the structure and schema of output files produced by reprodICU.

All output files are Parquet format for efficient storage and cross-platform compatibility.

## Patient Information Table

**File**: `patient_information.parquet`

**Description**: One row per ICU admission with demographics, anthropometrics, and outcomes.

**Schema**:

| Column                           | Type      | Description                                           |
| -------------------------------- | --------- | ----------------------------------------------------- |
| `Global ICU Stay ID`             | String    | Unique identifier across all datasets                 |
| `Source Dataset`                 | String    | Which dataset (eICU, MIMIC3, etc.)                    |
| `Patient ID`                     | String    | Patient identifier within dataset                     |
| `Admission Time`                 | Timestamp | ICU admission datetime                                |
| `Discharge Time`                 | Timestamp | ICU discharge datetime                                |
| `ICU Length of Stay (days)`      | Float     | Duration from admission to discharge                  |
| `Age (years)`                    | Float     | Patient age at admission                              |
| `Sex`                            | String    | M/F                                                   |
| `Weight (kg)`                    | Float     | Body weight, winsorized to clinically plausible range |
| `Height (cm)`                    | Float     | Body height, winsorized to clinically plausible range |
| `BMI`                            | Float     | Calculated body mass index                            |
| `Admission Type`                 | String    | Emergency/Scheduled/Urgent/Other                      |
| `Admission Location`             | String    | ER/Ward/Operating Room/Other                          |
| `Discharge Location`             | String    | Home/Skilled Facility/Death/Other                     |
| `Mortality in ICU`               | Boolean   | In-hospital death during ICU stay                     |
| `Hospital Length of Stay (days)` | Float     | Total hospital admission duration                     |
| `Mortality in Hospital`          | Boolean   | In-hospital mortality at any time                     |

## Diagnoses Table

**File**: `diagnoses.parquet`

**Description**: ICD diagnosis codes mapped to standard coding systems.

**Schema**:

| Column                  | Type      | Description                               |
| ----------------------- | --------- | ----------------------------------------- |
| `Global ICU Stay ID`    | String    | Link to ICU admission                     |
| `Diagnosis ID`          | String    | Unique diagnosis record identifier        |
| `ICD Code`              | String    | Original ICD-9 or ICD-10 code             |
| `ICD Version`           | String    | "9" or "10"                               |
| `Diagnosis Description` | String    | Human-readable diagnosis name             |
| `Diagnosis Time`        | Timestamp | When diagnosis was recorded               |
| `Seq Num`               | Integer   | Sequence number (1=primary, 2+=secondary) |
| `Source Dataset`        | String    | Dataset of origin                         |

## Procedures Table

**File**: `procedures.parquet`

**Description**: Procedure codes performed during ICU stay.

**Schema**:

| Column                  | Type      | Description                    |
| ----------------------- | --------- | ------------------------------ |
| `Global ICU Stay ID`    | String    | Link to ICU admission          |
| `Procedure ID`          | String    | Unique procedure identifier    |
| `Procedure Code`        | String    | ICD-9 or ICD-10 procedure code |
| `ICD Version`           | String    | "9" or "10"                    |
| `Procedure Description` | String    | Human-readable name            |
| `Procedure Time`        | Timestamp | When procedure was performed   |
| `Source Dataset`        | String    | Dataset of origin              |

## Medications Table

**File**: `medications.parquet` (administered) and `medications_prescribed.parquet` (prescribed)

**Description**: Medication administrations with doses, routes, and timing.

**Schema**:

| Column               | Type      | Description                                |
| -------------------- | --------- | ------------------------------------------ |
| `Global ICU Stay ID` | String    | Link to ICU admission                      |
| `Medication ID`      | String    | Unique medication record identifier        |
| `Drug Name`          | String    | Medication name                            |
| `Drug Code`          | String    | Standardized drug code (RxNorm, ATC, etc.) |
| `Dose`               | Float     | Dose quantity                              |
| `Dose Unit`          | String    | Unit (mg, mcg, mL, units, etc.)            |
| `Route`              | String    | Administration route (IV, PO, IM, etc.)    |
| `Start Time`         | Timestamp | When administration began                  |
| `Stop Time`          | Timestamp | When administration ended                  |
| `Rate`               | Float     | Infusion rate (if applicable)              |
| `Rate Unit`          | String    | Rate unit (mL/hr, mcg/kg/min, etc.)        |
| `Source Dataset`     | String    | Dataset of origin                          |

## Microbiology Table

**File**: `microbiology.parquet`

**Description**: Culture results and antibiotic susceptibility data.

**Schema**:

| Column               | Type      | Description                                      |
| -------------------- | --------- | ------------------------------------------------ |
| `Global ICU Stay ID` | String    | Link to ICU admission                            |
| `Culture ID`         | String    | Unique culture record identifier                 |
| `Culture Type`       | String    | Specimen type (blood, urine, sputum, etc.)       |
| `Culture Site`       | String    | Sample source location                           |
| `Culture Time`       | Timestamp | Time sample was obtained                         |
| `Organism`           | String    | Identified organism name                         |
| `Organism Code`      | String    | Standardized organism code                       |
| `Growth`             | String    | Growth result (positive, negative, mixed, etc.)  |
| `Antibiotic`         | String    | Antibiotic tested                                |
| `Susceptibility`     | String    | S (susceptible), I (intermediate), R (resistant) |
| `Source Dataset`     | String    | Dataset of origin                                |

## Clinical Notes Table

**File**: `notes.parquet`

**Description**: Clinical documentation and assessments.

**Schema**:

| Column               | Type      | Description                                           |
| -------------------- | --------- | ----------------------------------------------------- |
| `Global ICU Stay ID` | String    | Link to ICU admission                                 |
| `Note ID`            | String    | Unique note identifier                                |
| `Note Type`          | String    | Nursing note, physician note, discharge summary, etc. |
| `Note Time`          | Timestamp | When note was written                                 |
| `Note Text`          | String    | Full note content                                     |
| `Source Dataset`     | String    | Dataset of origin                                     |

## Timeseries: Vital Signs

**File**: `timeseries_vitals.parquet` (and variants: `_imputed.parquet`, `_resampled.parquet`)

**Description**: Continuous or periodic vital sign measurements over ICU stay.

**Schema**:

| Column                            | Type      | Description                   |
| --------------------------------- | --------- | ----------------------------- |
| `Global ICU Stay ID`              | String    | Link to ICU admission         |
| `Measurement Time`                | Timestamp | When measurement was taken    |
| `Heart Rate (bpm)`                | Struct    | Contains value and unit       |
| `Systolic Blood Pressure (mmHg)`  | Struct    |                               |
| `Diastolic Blood Pressure (mmHg)` | Struct    |                               |
| `Mean Arterial Pressure (mmHg)`   | Struct    |                               |
| `Body Temperature (°C)`           | Struct    |                               |
| `Respiratory Rate (breaths/min)`  | Struct    |                               |
| `Oxygen Saturation (%)`           | Struct    | SpO2 percentage               |
| `Glasgow Coma Scale Score`        | Struct    | GCS 3-15                      |
| ... (other vitals)                | Struct    | Additional vital measurements |
| `Source Dataset`                  | String    | Dataset of origin             |

**Struct Format**: Each measurement is a struct with:

```python
{
    "value": <Float>,
    "unit": <String>
}
```

## Timeseries: Laboratory Values

**File**: `timeseries_labs.parquet` (and variant: `_winsorized.parquet`)

**Description**: Laboratory test results at multiple timepoints. **Long format** – one row per test per timepoint.

### Structure

**Row Format**: Each row represents one lab test at one timepoint.

| Column               | Type      | Description                                   |
| -------------------- | --------- | --------------------------------------------- |
| `Global ICU Stay ID` | String    | Link to ICU admission                         |
| `Lab Time`           | Timestamp | When lab was drawn/resulted                   |
| `Lab Name`           | String    | Test name (e.g., "Hemoglobin", "Creatinine")  |
| `Lab Code`           | String    | LOINC code or dataset-specific lab identifier |
| `Value`              | Float     | Numeric result                                |
| `Value Min`          | Float     | Reference range minimum                       |
| `Value Max`          | Float     | Reference range maximum                       |
| `Unit`               | String    | Unit of measurement                           |
| `Flag`               | String    | H (high), L (low), N (normal)                 |
| `Source Dataset`     | String    | Dataset of origin                             |

### Available Laboratory Tests

Laboratory tests are harmonized using LOINC (Logical Observation Identifiers Names and Codes) where possible. Common labs include:

**Hematology**:

- Hemoglobin (g/dL)
- Hematocrit (%)
- White Blood Cell Count (K/uL)
- Platelet Count (K/uL)
- Red Blood Cell Count (M/uL)

**Chemistry**:

- Sodium (mEq/L)
- Potassium (mEq/L)
- Chloride (mEq/L)
- Bicarbonate/CO2 (mEq/L)
- Glucose (mg/dL)
- Blood Urea Nitrogen (mg/dL)
- Creatinine (mg/dL)
- Calcium (mg/dL)
- Magnesium (mg/dL)
- Phosphate (mg/dL)

**Hepatic Function**:

- Alanine Aminotransferase/ALT (U/L)
- Aspartate Aminotransferase/AST (U/L)
- Alkaline Phosphatase (U/L)
- Bilirubin Total (mg/dL)
- Albumin (g/dL)

**Coagulation**:

- Prothrombin Time/PT (seconds)
- Partial Thromboplastin Time/aPTT (seconds)
- International Normalized Ratio/INR
- Fibrinogen (mg/dL)
- D-Dimer (mcg/mL)

**Arterial/Venous Blood Gas**:

- pH
- pCO2 (mmHg)
- pO2 (mmHg)
- HCO3 (mEq/L)
- Base Excess (mEq/L)
- Lactate (mmol/L)

**Cardiac Markers**:

- Troponin I (ng/mL)
- Troponin T (ng/mL)
- B-type Natriuretic Peptide/BNP (pg/mL)

**Infection Markers**:

- C-Reactive Protein (mg/L)
- Procalcitonin (ng/mL)
- White Blood Cell Count (K/uL) ← also listed above

**Lipids**:

- Cholesterol (mg/dL)
- Triglycerides (mg/dL)

**Other**:

- Lactate (mmol/L or mg/dL)
- Ammonia (mcg/dL)
- Anion Gap (mEq/L)

For a complete list of mapped labs, see: `src/reprodICU/configs/RELEVANT_VALUES/RELEVANT_LABS_LOINC.yaml`

## Timeseries: Respiratory Parameters

**File**: `timeseries_respiratory.parquet`

**Description**: Mechanical ventilation settings and parameters.

**Schema**:

| Column                           | Type      | Description                               |
| -------------------------------- | --------- | ----------------------------------------- |
| `Global ICU Stay ID`             | String    | Link to ICU admission                     |
| `Measurement Time`               | Timestamp | When parameters were recorded             |
| `Ventilated`                     | Boolean   | Whether patient on mechanical ventilation |
| `Ventilator Mode`                | String    | Mode (AC, SIMV, CPAP, etc.)               |
| `Tidal Volume (mL)`              | Struct    | {value, unit}                             |
| `Respiratory Rate (breaths/min)` | Struct    |                                           |
| `PEEP (cmH2O)`                   | Struct    | Positive end-expiratory pressure          |
| `FiO2 (%)`                       | Struct    | Fraction of inspired oxygen               |
| `Peak Pressure (cmH2O)`          | Struct    |                                           |
| `Plateau Pressure (cmH2O)`       | Struct    |                                           |
| `Minute Ventilation (L/min)`     | Struct    |                                           |
| ... (other params)               | Struct    | Additional respiratory parameters         |
| `Source Dataset`                 | String    | Dataset of origin                         |

## Timeseries: Intake & Output

**File**: `timeseries_intakeoutput.parquet` (and variant: `_balanced.parquet`)

**Description**: Fluid intake (IV, oral, tube feeding) and output (urine, drain, stool) measurements.

**Schema**:

| Column                   | Type      | Description                                              |
| ------------------------ | --------- | -------------------------------------------------------- |
| `Global ICU Stay ID`     | String    | Link to ICU admission                                    |
| `Measurement Time`       | Timestamp | When recorded                                            |
| `Intake Type`            | String    | "IV", "Oral", "TubeFeeding", "Blood", "Dialysis", etc.   |
| `Intake Volume (mL)`     | Float     | Volume of intake                                         |
| `Output Type`            | String    | "Urine", "Drain", "Stool", "Emesis", "Nasogastric", etc. |
| `Output Volume (mL)`     | Float     | Volume of output                                         |
| `Net Fluid Balance (mL)` | Float     | Intake - Output                                          |
| `Source Dataset`         | String    | Dataset of origin                                        |

## MAGIC_CONCEPTS Tables

**Files**: Located in `MAGIC_CONCEPTS/` subdirectory: `CODE_STATUS.parquet`, `RECEIVED_ANY_ANTIBIOTICS.parquet`, etc.

**Description**: Derived clinical concepts extracted from raw data. Output structure varies by concept (see below).

### CODE_STATUS

**File**: `MAGIC_CONCEPTS/CODE_STATUS.parquet`

Code status indicates limitations on resuscitation efforts.

| Column               | Type      | Description                                     |
| -------------------- | --------- | ----------------------------------------------- |
| `Global ICU Stay ID` | String    | Link to ICU admission                           |
| `Code Status`        | String    | "Full Code", "DNR", "DNI", "DNCPR", "CMO", etc. |
| `Code Status Time`   | Timestamp | When code status changed                        |
| `Source Dataset`     | String    | Dataset of origin                               |

### RECEIVED_ANY_ANTIBIOTICS

**File**: `MAGIC_CONCEPTS/RECEIVED_ANY_ANTIBIOTICS.parquet`

Whether patient received any antibiotic during ICU stay.

| Column                  | Type      | Description                     |
| ----------------------- | --------- | ------------------------------- |
| `Global ICU Stay ID`    | String    | Link to ICU admission           |
| `Received Antibiotics`  | Boolean   | True if any antibiotic given    |
| `First Antibiotic Time` | Timestamp | When first antibiotic was given |
| `Source Dataset`        | String    | Dataset of origin               |

### VENTILATION_DURATION

**File**: `MAGIC_CONCEPTS/VENTILATION_DURATION.parquet`

Duration and details of mechanical ventilation.

| Column                               | Type      | Description                           |
| ------------------------------------ | --------- | ------------------------------------- |
| `Global ICU Stay ID`                 | String    | Link to ICU admission                 |
| `Ventilated`                         | Boolean   | Whether patient was ventilated        |
| `Ventilation Start Time`             | Timestamp | When ventilation began                |
| `Ventilation End Time`               | Timestamp | When ventilation ended                |
| `Total Ventilation Duration (hours)` | Float     | Total hours ventilated                |
| `Number of Ventilation Episodes`     | Integer   | How many separate ventilation periods |
| `Source Dataset`                     | String    | Dataset of origin                     |

### RENAL_REPLACEMENT_THERAPY_DURATION

**File**: `MAGIC_CONCEPTS/RENAL_REPLACEMENT_THERAPY_DURATION.parquet`

Renal replacement therapy (dialysis) duration and timing.

| Column                       | Type      | Description                   |
| ---------------------------- | --------- | ----------------------------- |
| `Global ICU Stay ID`         | String    | Link to ICU admission         |
| `Received RRT`               | Boolean   | Whether patient received RRT  |
| `RRT Start Time`             | Timestamp | When RRT began                |
| `RRT End Time`               | Timestamp | When RRT ended                |
| `Total RRT Duration (hours)` | Float     | Total hours on RRT            |
| `Number of RRT Episodes`     | Integer   | How many separate RRT periods |
| `RRT Modality`               | String    | "CRRT", "IHD", "SLED", etc.   |
| `Source Dataset`             | String    | Dataset of origin             |

### SEVERITY_SCORES

**File**: `MAGIC_CONCEPTS/SEVERITY_SCORES.parquet`

Pre-extracted severity scores from source datasets (APACHE II, SOFA, SAPS III, etc.).

| Column               | Type      | Description                                                    |
| -------------------- | --------- | -------------------------------------------------------------- |
| `Global ICU Stay ID` | String    | Link to ICU admission                                          |
| `Score Name`         | String    | "APACHE II", "APACHE III", "APS III", "SOFA", "SAPS III", etc. |
| `Score Value`        | Float     | Score value                                                    |
| `Score Time`         | Timestamp | When score was calculated                                      |
| `Source Dataset`     | String    | Dataset of origin                                              |

## Accessing Data

### Polars (Recommended)

```python
import polars as pl

# Read entire table
patient_info = pl.read_parquet("reprodICU/patient_information.parquet")

# Read specific columns
labs = pl.read_parquet(
    "reprodICU/timeseries_labs.parquet",
    columns=["Global ICU Stay ID", "Lab Time", "Lab Name", "Value"]
)

# Read with filter (pushdown optimization)
recent_labs = pl.scan_parquet("reprodICU/timeseries_labs.parquet").filter(
    pl.col("Lab Time") > "2015-01-01"
).collect()
```

### Pandas

```python
import pandas as pd

patient_info = pd.read_parquet("reprodICU/patient_information.parquet")
labs = pd.read_parquet("reprodICU/timeseries_labs.parquet")
```

### DuckDB

```python
import duckdb

result = duckdb.query("""
    SELECT
        p.Global_ICU_Stay_ID,
        p.Age,
        COUNT(DISTINCT l.Lab_Name) as unique_labs
    FROM 'reprodICU/patient_information.parquet' p
    LEFT JOIN 'reprodICU/timeseries_labs.parquet' l
        ON p.Global_ICU_Stay_ID = l.Global_ICU_Stay_ID
    GROUP BY p.Global_ICU_Stay_ID, p.Age
""").to_df()
```

## Data Quality Notes

- **Winsorization**: Lab and vital values are winsorized to clinically plausible ranges (typically 0.5th to 99.5th percentile)
- **Imputation**: Missing vitals can be imputed using `impute=True` in `build_timeseries()`
- **Resampling**: Timeseries can be resampled to regular intervals (e.g., hourly) using `resample=3600` in `build_timeseries()`
- **Missing Data**: Missing values are represented as `None` in Polars / `NaN` in Pandas
- **Temporal Alignment**: All timestamps are in UTC and aligned to ICU admission time as reference (0)
