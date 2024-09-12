# reprodICU

A comprehensive database harmonization and interaction package for multiple ICU databases.

### Based on previous work by

- Bennett et al. (2023) [ricu: R’s interface to intensive care data](https://doi.org/10.1093/gigascience/giad041) and the associated git-Repository [ricu](https://github.com/eth-mds/ricu/tree/main)
- Oliver et al. (2023) [Introducing the BlendedICU dataset, the first harmonized, international intensive care dataset](https://doi.org/10.1016/j.jbi.2023.104502) and the associated git-Repository [BlendedICU](https://github.com/USM-CHU-FGuyon/BlendedICU)

## AXIOMS

**`Axioms`** are datapoints that are completely underivable — for example: the `heart_rate` of a patient is not calculable from his lab values.
**Anything else(!)** that can be calculated, however complicated that may be, is not(!) an axiom. Anything that can be calculated, should be calculated. Calculable variables are called **`Concepts`**.
Concepts should be defined as python functions depending on their respective axiomatic inputs. Concepts do not need to be defined on the basis of axioms, concepts may also be derived from other concepts. At the end, where there is no more derivation possible, there there are the axioms.

## Important Design Choices

- `time` is **relative to ICU admission** in **`seconds`** (exact datetimes are never available, due to de-identification)
- **calculate once, use twice!**
  → any concepts that are used for covariates / exposure should be coded in such a way that they allow for pre-computation on the complete dataset! - should be able to handle `NaN` values if using a non-imputed version (i.e. return `None` for these ICU stay IDs)

## How to build

- There are multiple build scripts available, each differing in the amount of preprocessing they do. Depending on the amount of preprocessing, the output tables might differ in structure (however, they are then clearly named as such)
  - `reprodICU.py` runs a **recommended** set and order of operations
  - `reprodICU_DEMO.py` runs a **recommended** set and order of operations on the publicly available DEMO-Datasets of `MIMIC-III` / `-IV` and `eICU` (**not yet implemented**)
  1. `X0_raw_harmonize.py` only harmonizes the data and neither cleans nor preprocesses anything
  2. `X1_clean.py` manually removes some identified outliers that are deemed false values
  3. `X2_winsorize.py` winsorizes laboratory values and anthropometric data
  4. `X3_impute.py` imputes missing values using the preferred imputation method (**not yet fully implemented**)
  5. `X4_resample.py` resamples and imputes the timeseries to a common frequency (**not yet implemented**)

## Planned Workflow

1. **Include** / **exclude** patients
   1. Create boolean masks on the `global_icu_stay_id` table to create a set of included patients
   2. Boolean masks may in practice be created in a step-down procedure (i.e. evaluate only the patients that successfully passed the previous selection criteria for computational efficiency), however the underlying code should be independent of the order of inclusion/exclusion operations
2. Determine **exposure**
   1. Define a concept for the relevant exposure for the study
      (relevant code should be written in a way that allows for pre-computation on the complete dataset!)
3. Determine **covariates**
   1. Define / use established concepts for relevant covariates for the study
      (common covariates such as Elixhauser comorbidity index should be available as precalculated dataframes for the complete dataset)

## Basic structure of the reprodICU(bility) database

0. labels

   - age, height, weight on admission
   - location

1. timeseries (sparse)
   Each measurement is associated with an ICU stay ID and a timepoint - vitals - laboratory values

2. medications

   - ICU stay ID
   - drug name (as in source database)
   - harmonized ingredient
   - medication start time
   - medication end time
   - medication dose
   - medication unit

3. diagnoses
