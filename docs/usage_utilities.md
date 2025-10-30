# Usage & Utilities

reprodICU includes pre-built utilities for common clinical analyses on harmonized data.

## Clinical Scoring Systems

### SOFA Score

The Sequential Organ Failure Assessment (SOFA) score quantifies multi-organ failure severity.

**Function**: `reprodICU.utils.scores.SOFA`

**Source**: [SOFA scoring system](https://en.wikipedia.org/wiki/SOFA_score)

**Output**:

- SOFA total score per timepoint
- Component scores: Respiratory, Cardiovascular, Hepatic, Renal, Neurologic, Coagulation

```python
from reprodICU.utils.scores import SOFA

# Auto-loads all required data (vitals, labs, respiratory, etc.)
sofa = SOFA()
print(sofa.collect())  # DataFrame with SOFA scores over time
```

Or, to use custom data:

```python
import polars as pl
from reprodICU.utils.scores import SOFA

# Load timeseries data
timeseries_vitals = pl.scan_parquet("reprodICU/timeseries_vitals.parquet")
timeseries_labs = pl.scan_parquet("reprodICU/timeseries_labs.parquet")

# Compute SOFA scores
sofa = SOFA(timeseries_vitals=timeseries_vitals, timeseries_labs=timeseries_labs)
print(sofa.collect())
```

### SOFA-2 Score

Simplified version of SOFA using fewer variables.

**Function**: `reprodICU.utils.scores.SOFA2`

```python
from reprodICU.utils.scores import SOFA2

# Auto-loads all required data
sofa2 = SOFA2()
print(sofa2.collect())
```

### APACHE IV Score

Acute Physiology And Chronic Health Evaluation – severity score combining acute physiologic data with chronic health status.

**Function**: `reprodICU.utils.scores.OASIS`

**Source**: [APACHE scoring](https://en.wikipedia.org/wiki/Acute_Physiology_and_Chronic_Health_Evaluation)

Note: APACHE II, III, IV scores may be pre-extracted from datasets (see MAGIC_CONCEPTS).

### VIS (Vasoactive-Inotropic Score)

Quantifies inotropic/vasopressor support intensity.

**Function**: `reprodICU.utils.scores.VIS`

```python
from reprodICU.utils.scores import VIS

# Auto-loads medication data
vis = VIS()
print(vis.collect())  # DataFrame with VIS scores per stay and timepoint
```

## Mortality Prediction

### Common Mortality Measures

Calculate mortality at standard clinical timepoints.

**Function**: `reprodICU.utils.mortality.COMMON_MORTALITY_MEASURES`

**Output**: Mortality flags at standard timepoints

```python
from reprodICU.utils.mortality import COMMON_MORTALITY_MEASURES

# Auto-loads patient information (contains mortality data)
mortality = COMMON_MORTALITY_MEASURES()
print(mortality.collect())  # DataFrame with mortality flags at each timepoint

# Example output columns:
# Global ICU Stay ID | Mortality After ICU Admission (days) | Mortality 7 Days After ICU Admission | ...
```

**Columns**:

- `Global ICU Stay ID`: Unique identifier for each ICU admission
- `Mortality After ICU Admission (days)`: Time from admission to death (days, or None if censored)
- `Mortality 7 Days After ICU Admission`: Boolean flag for death within 7 days
- `Mortality 28 Days After ICU Admission`: Boolean flag for death within 28 days
- `Mortality 30 Days After ICU Admission`: Boolean flag for death within 30 days
- `Mortality 90 Days After ICU Admission`: Boolean flag for death within 90 days
- `Mortality 180 Days After ICU Admission`: Boolean flag for death within 180 days
- `Mortality 360 Days After ICU Admission`: Boolean flag for death within 360 days
- `Mortality 1 Year After ICU Admission`: Boolean flag for death within 365 days

## Comorbidity Assessment

### Charlson Comorbidity Index

Calculates comorbidity using the Charlson index (Quan implementation).

**Function**: `reprodICU.utils.comorbidity.CHARLSON`

**Backend**: [PyComOrb](https://github.com/USCbiostats/pycomorbidity)

```python
from reprodICU.utils.comorbidity import CHARLSON

# Auto-loads diagnosis and patient information data
charlson = CHARLSON()
print(charlson.collect())  # DataFrame with Charlson scores per admission
```

### Elixhauser Comorbidity Index

Calculates comorbidity using the Elixhauser index (Quan implementation with van Walraven weights).

**Function**: `reprodICU.utils.comorbidity.ELIXHAUSER`

```python
from reprodICU.utils.comorbidity import ELIXHAUSER

# Auto-loads diagnosis and patient information data
elixhauser = ELIXHAUSER()
print(elixhauser.collect())
```

### Gagne Comorbidity Index

Calculates combined Gagne comorbidity index.

**Function**: `reprodICU.utils.comorbidity.GAGNE`

```python
from reprodICU.utils.comorbidity import GAGNE

# Auto-loads diagnosis and patient information data
gagne = GAGNE()
print(gagne.collect())
```

**Note**: All comorbidity functions support both ICD-9 and ICD-10 codes. Data is automatically loaded from the reprodICU database if not provided explicitly.

## Sepsis Detection

### Sepsis-3 Definition (Seymour)

Detect sepsis using Sepsis-3 criteria anchored on culture and antibiotic pairs.

**Function**: `reprodICU.utils.sepsis.SEPSIS`

**Output**: Produces three independent sepsis definitions:

- **SEPSIS**: Seymour-anchored (culture + antibiotic pair)
- **SEPSIS_ABX**: Shah-anchored (antibiotic escalation)
- **SEPSIS_RHEE**: Rhee EHR surveillance definition

**References**:

- Seymour CW, et al. Sepsis-3. JAMA 2016
- Shah AD, et al. Descriptors of Sepsis Using Sepsis-3. Crit Care Med 2021
- Rhee C, et al. Objective Sepsis Surveillance Using EHR. Infect Control Hosp Epidemiol 2016

```python
from reprodICU.utils.sepsis import SEPSIS

# Auto-loads all required data (vitals, labs, medications, cultures, etc.)
# SEPSIS returns long-format output with timeframe-level granularity
# (not just binary yes/no per admission)
sepsis = SEPSIS()
print(sepsis.collect())  # DataFrame with columns: Global ICU Stay ID, timeframe, T_0, SEPSIS, SEPSIS_ABX, SEPSIS_RHEE
```

### Angus Sepsis Definition

Alternative sepsis definition using Angus criteria (older standard).

**Function**: `reprodICU.utils.sepsis.ANGUS_SEPSIS`

```python
from reprodICU.utils.sepsis import ANGUS_SEPSIS

# Auto-loads all required data
angus = ANGUS_SEPSIS()
print(angus.collect())
```

### Martin Sepsis Definition

Martin's modification of sepsis criteria.

**Function**: `reprodICU.utils.sepsis.MARTIN_SEPSIS`

```python
from reprodICU.utils.sepsis import MARTIN_SEPSIS

# Auto-loads all required data
martin = MARTIN_SEPSIS()
print(martin.collect())
```

**Note**: Sepsis functions return long-format data with timeframe-level granularity (multiple rows per admission), unlike other utilities which return one row per admission.

## Data Quality and Transformations

### Fix Window Borders

Correct data at ICU admission and discharge boundaries (often erroneous).

**Function**: `reprodICU.utils.FIX_WINDOW_BORDERS`

```python
from reprodICU.utils import FIX_WINDOW_BORDERS
import polars as pl

# Load timeseries data
timeseries = pl.read_parquet("reprodICU/timeseries_vitals.parquet")

# Apply boundary correction
fixed = FIX_WINDOW_BORDERS(timeseries)
```

## Clinical Concepts & Derived Variables

### Organ Support Free Days

Calculate the number of days without requiring specific life-sustaining interventions.

#### Ventilator-Free Days

**Function**: `reprodICU.utils.clinical.VENTILATOR_FREE_DAYS`

Number of calendar days where patient was alive and not receiving mechanical ventilation.

```python
from reprodICU.utils.clinical import VENTILATOR_FREE_DAYS

# Auto-loads patient information and ventilation data
vfd = VENTILATOR_FREE_DAYS(timeframe_days=28)
print(vfd.collect())  # Columns: Global ICU Stay ID, Ventilator Free Days (28d), ...
```

#### RRT-Free Days

**Function**: `reprodICU.utils.clinical.RENAL_REPLACEMENT_THERAPY_FREE_DAYS`

Number of calendar days where patient was alive and not receiving renal replacement therapy (dialysis).

```python
from reprodICU.utils.clinical import RENAL_REPLACEMENT_THERAPY_FREE_DAYS

# Auto-loads all required data
rrt_free = RENAL_REPLACEMENT_THERAPY_FREE_DAYS(timeframe_days=28)
print(rrt_free.collect())
```

#### Vasopressor-Free Days

**Function**: `reprodICU.utils.clinical.VASOPRESSOR_FREE_DAYS`

Number of calendar days where patient was alive and not receiving vasopressor/inotropic support.

```python
from reprodICU.utils.clinical import VASOPRESSOR_FREE_DAYS

# Auto-loads all required data
vpd = VASOPRESSOR_FREE_DAYS(timeframe_days=28)
print(vpd.collect())
```

### Anthropometric Calculations

Body composition metrics for patient assessment and drug dosing.

#### Ideal Body Weight

**Function**: `reprodICU.utils.clinical.IDEAL_BODY_WEIGHT_DEVINE` or `IDEAL_BODY_WEIGHT_LORENTZ`

Calculate ideal body weight using Devine or Lorentz formula.

```python
from reprodICU.utils.clinical import IDEAL_BODY_WEIGHT_DEVINE, IDEAL_BODY_WEIGHT_LORENTZ

# Devine formula (widely used)
ibw_devine = IDEAL_BODY_WEIGHT_DEVINE()
print(ibw_devine.collect())

# Lorentz formula (alternative)
ibw_lorentz = IDEAL_BODY_WEIGHT_LORENTZ()
print(ibw_lorentz.collect())
```

#### Adjusted Body Weight

**Function**: `reprodICU.utils.clinical.ADJUSTED_BODY_WEIGHT`

Calculate adjusted body weight for dosing in obese patients.

```python
from reprodICU.utils.clinical import ADJUSTED_BODY_WEIGHT

# Auto-loads patient information
abw = ADJUSTED_BODY_WEIGHT()
print(abw.collect())
```

#### Body Surface Area

**Function**: `reprodICU.utils.clinical.BODY_SURFACE_AREA`

Calculate body surface area (BSA) using Mosteller formula (m²).

```python
from reprodICU.utils.clinical import BODY_SURFACE_AREA

# Auto-loads patient information
bsa = BODY_SURFACE_AREA()
print(bsa.collect())
```

#### BMI Classification

**Function**: `reprodICU.utils.clinical.CLASSIFY_BODY_MASS_INDEX` or `BODY_MASS_INDEX`

Classify BMI according to WHO categories.

```python
from reprodICU.utils.clinical import CLASSIFY_BODY_MASS_INDEX, BODY_MASS_INDEX

# Get raw BMI values
bmi = BODY_MASS_INDEX()
print(bmi.collect())

# Get WHO BMI classification
bmi_classified = CLASSIFY_BODY_MASS_INDEX()
print(bmi_classified.collect())  # Columns: Global ICU Stay ID, BMI, BMI Classification
```

### Respiratory Mechanics

Assess oxygenation and lung compliance.

#### PaO2/FiO2 Ratio

**Function**: `reprodICU.utils.clinical.PAO2_FIO2_RATIO`

Calculate arterial oxygen partial pressure to fraction of inspired oxygen ratio.

```python
from reprodICU.utils.clinical import PAO2_FIO2_RATIO

# Auto-loads respiratory and lab timeseries
pf_ratio = PAO2_FIO2_RATIO()
print(pf_ratio.collect())  # Timeseries with PaO2/FiO2 Ratio
```

**Reference**: PaO2/FiO2 < 300 indicates ARDS, < 100 indicates severe ARDS.

#### SpO2/FiO2 Ratio

**Function**: `reprodICU.utils.clinical.SPO2_FIO2_RATIO`

Calculate peripheral oxygen saturation to fraction of inspired oxygen ratio (alternative oxygenation assessment).

```python
from reprodICU.utils.clinical import SPO2_FIO2_RATIO

# Auto-loads vital signs and respiratory data
sf_ratio = SPO2_FIO2_RATIO()
print(sf_ratio.collect())
```

#### Dynamic & Static Compliance

**Function**: `reprodICU.utils.clinical.DYNAMIC_COMPLIANCE` or `STATIC_COMPLIANCE`

Calculate lung compliance (volume change per unit pressure change).

```python
from reprodICU.utils.clinical import DYNAMIC_COMPLIANCE, STATIC_COMPLIANCE

# Dynamic compliance (during ventilation)
dyn_compliance = DYNAMIC_COMPLIANCE()
print(dyn_compliance.collect())

# Static compliance (at inspiratory hold)
stat_compliance = STATIC_COMPLIANCE()
print(stat_compliance.collect())
```

#### Mechanical Power

**Function**: `reprodICU.utils.clinical.MECHANICAL_POWER`

Calculate ventilator-delivered mechanical power (J/min).

```python
from reprodICU.utils.clinical import MECHANICAL_POWER

# Auto-loads respiratory data
mech_power = MECHANICAL_POWER()
print(mech_power.collect())
```

**Reference**: Mechanical power > 12 J/min associated with ventilator-induced lung injury.

### Renal Function

#### Estimated GFR

**Function**: `reprodICU.utils.clinical.ESTIMATED_GFR`

Calculate estimated glomerular filtration rate using MDRD or CKD-EPI equations.

```python
from reprodICU.utils.clinical import ESTIMATED_GFR

# Auto-loads patient info and lab data
egfr = ESTIMATED_GFR()
print(egfr.collect())
```

#### Urine Output

**Function**: `reprodICU.utils.clinical.URINE_OUTPUT`

Calculate urine output in mL/kg/hour or absolute volume.

```python
from reprodICU.utils.clinical import URINE_OUTPUT

# Auto-loads intake/output and patient data
urine = URINE_OUTPUT()
print(urine.collect())  # Timeseries with urine output rates
```

### Vasopressor Equivalence

#### Norepinephrine Equivalent Dosage

**Function**: `reprodICU.utils.clinical.NOREPINEPHRINE_EQUIVALENT_DOSAGE`

Convert all vasopressors/inotropes to equivalent norepinephrine dose (mcg/kg/min) for standardized comparison.

```python
from reprodICU.utils.clinical import NOREPINEPHRINE_EQUIVALENT_DOSAGE

# Auto-loads medication data
norepi_eq = NOREPINEPHRINE_EQUIVALENT_DOSAGE()
print(norepi_eq.collect())  # Timeseries with norepinephrine-equivalent dosages
```

**References**: Used in conjunction with VIS or other intensity-of-support scores.

## Composition: Building Custom Pipelines

Chain utilities to build custom analyses:

```python
from reprodICU.utils.mortality import COMMON_MORTALITY_MEASURES
from reprodICU.utils.scores import SOFA

# All utilities auto-load data - no need to manually load parquet files!
mortality = COMMON_MORTALITY_MEASURES()
sofa = SOFA()

# Collect results
mortality_df = mortality.collect()
sofa_df = sofa.collect()

# Join for analysis
analysis_df = (
    mortality_df
    .join(sofa_df, on="Global ICU Stay ID")
)

# Simple analysis: SOFA vs mortality
analysis_df.with_columns(
    sofa_quartile=pl.col("SOFA Score").qcut(4)
).group_by("sofa_quartile").agg([
    pl.col("Mortality 28 Days After ICU Admission").mean().alias("28d_mortality_rate")
])
```

## Reference: Available Utilities

| Module               | Function                              | Description                                                            |
| -------------------- | ------------------------------------- | ---------------------------------------------------------------------- |
| `scores`             | `SOFA`                                | Sequential Organ Failure Assessment (SOFA)                             |
| `scores`             | `SOFA2`                               | Simplified SOFA score                                                  |
| `scores`             | `OASIS`                               | Acute Physiology and Chronic Health Evaluation                         |
| `scores`             | `OASIS_icu_mortality`                 | OASIS ICU mortality predictor                                          |
| `scores`             | `OASIS_hospital_mortality`            | OASIS hospital mortality predictor                                     |
| `scores`             | `VIS`                                 | Vasoactive-Inotropic Score                                             |
| `mortality`          | `COMMON_MORTALITY_MEASURES`           | Mortality at standard timepoints (7d, 28d, 30d, 90d, 180d, 360d, 365d) |
| `comorbidity`        | `CHARLSON`                            | Charlson comorbidity index (Quan implementation)                       |
| `comorbidity`        | `ELIXHAUSER`                          | Elixhauser comorbidity index (Quan + van Walraven weights)             |
| `comorbidity`        | `GAGNE`                               | Gagne combined comorbidity index                                       |
| `sepsis`             | `SEPSIS`                              | Sepsis-3 (Seymour, Shah, Rhee definitions)                             |
| `sepsis`             | `ANGUS_SEPSIS`                        | Angus sepsis definition                                                |
| `sepsis`             | `MARTIN_SEPSIS`                       | Martin sepsis definition                                               |
| `clinical`           | `VENTILATOR_FREE_DAYS`                | Ventilator-free days                                                   |
| `clinical`           | `RENAL_REPLACEMENT_THERAPY_FREE_DAYS` | RRT-free days                                                          |
| `clinical`           | `VASOPRESSOR_FREE_DAYS`               | Vasopressor-free days                                                  |
| `clinical`           | `IDEAL_BODY_WEIGHT_DEVINE`            | Ideal body weight (Devine formula)                                     |
| `clinical`           | `IDEAL_BODY_WEIGHT_LORENTZ`           | Ideal body weight (Lorentz formula)                                    |
| `clinical`           | `ADJUSTED_BODY_WEIGHT`                | Adjusted body weight for dosing                                        |
| `clinical`           | `BODY_SURFACE_AREA`                   | Body surface area (Mosteller formula)                                  |
| `clinical`           | `BODY_MASS_INDEX`                     | Body mass index (kg/m²)                                                |
| `clinical`           | `CLASSIFY_BODY_MASS_INDEX`            | WHO BMI classification                                                 |
| `clinical`           | `PAO2_FIO2_RATIO`                     | Arterial oxygen/FiO2 ratio (P/F ratio)                                 |
| `clinical`           | `SPO2_FIO2_RATIO`                     | Peripheral oxygen saturation/FiO2 ratio (S/F ratio)                    |
| `clinical`           | `DYNAMIC_COMPLIANCE`                  | Dynamic lung compliance                                                |
| `clinical`           | `STATIC_COMPLIANCE`                   | Static lung compliance                                                 |
| `clinical`           | `MECHANICAL_POWER`                    | Ventilator-delivered mechanical power                                  |
| `clinical`           | `ESTIMATED_GFR`                       | Estimated glomerular filtration rate                                   |
| `clinical`           | `URINE_OUTPUT`                        | Urine output (mL/kg/hour)                                              |
| `clinical`           | `NOREPINEPHRINE_EQUIVALENT_DOSAGE`    | Vasopressor/inotrope as norepinephrine equivalent                      |
| `FIX_WINDOW_BORDERS` | `FIX_WINDOW_BORDERS`                  | Correct admission/discharge boundary data                              |

For API signatures and advanced usage, see [API Reference](/api_reference/).
