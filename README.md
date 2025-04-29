# reprodICU

## [DOCUMENTATION](http://wiki.reprodicu.org/)

## INTRODUCTION
**reprodICU** is a freely accessible critical care dataset harmonizing data relating to more than 470k admissions to multiple healthcare centers across the US and Europe. The dataset was developed at the **Institute of Medical Informatics** (IMI) at **Charité - Universitätsmedizin Berlin**.

The dataset contains de-identified demographic information and a total of 136 routinely collected physiological variables, diagnostic test results and treatment parameters from almost 350k patients during the period from 2001 to 2022.

## INCLUDED DATASETS

- [AmsterdamUMCdb v1.0.2](https://amsterdammedicaldatascience.nl)
- [eICU Collaborative Research Database v2.0](https://doi.org/10.13026/C2WM1R)
- [HiRID, a high time-resolution ICU dataset v1.1.1](https://doi.org/10.13026/nkwc-js72)
- [MIMIC-III Clinical Database v1.4](https://doi.org/10.13026/C2XW26)
- [MIMIC-IV v3.1](https://doi.org/10.13026/kpb9-mt58)
- [Northwestern ICU (NWICU) database v0.1.0](https://doi.org/10.13026/s84w-1829)
- [Salzburg Intensive Care database (SICdb) v1.0.8](https://doi.org/10.13026/8m72-6j83)

## AXIOMS

**`Axioms`** are datapoints that are completely underivable — for example: the `heart_rate` of a patient is not calculable from his lab values.
**Anything else(!)** that can be calculated, however complicated that may be, is not(!) an axiom. Anything that can be calculated, should be calculated. Calculable variables are called **`Concepts`**.
Concepts should be defined as python functions depending on their respective axiomatic inputs. Concepts do not need to be defined on the basis of axioms, concepts may also be derived from other concepts. At the end, where there is no more derivation possible, there there are the axioms.

# HIGHLIGHTS

## **Unprecedented Scale and Scope**

reprodICU integrates **469,822 ICU admissions** from **seven** major public datasets across **four countries**, creating the **largest harmonized ICU dataset** publicly available. This breadth enables cross-institutional and cross-national studies that were previously impractical due to data incompatibility.

## **Standardization Without Overprocessing**

reprodICU is harmonized using **established clinical vocabularies** (e.g., SNOMED, LOINC, RxNorm) and broadly follows the structure of the **German Medical Informatics Initiative modules** to ensure interoperability. Crucially, the project applies **minimal preprocessing** to preserve source fidelity and maintain compatibility with the original datasets.

## **Fast, Reproducible Research at Scale**

With the built-in **replication pipeline**, researchers can recreate complex study cohorts across all datasets **in just a few minutes** on a standard machine. This dramatically reduces the time and effort needed for **external validation** of clinical models.

## **Rich Library of Pre-Defined Clinical Concepts**

The project includes a ***massive, curated* catalog** of clinical variables, ranging from advanced ventilator metrics to dozens of mortality and severity scoring systems (e.g., SOFA, APACHE, MODS, NEWS, SAPS). These **ready-to-use components** eliminate the need for researchers and developers to manually redefine or look up formulas, making it easier and faster to build robust analyses or models.

## Inspired by previous work by

- Bennett et al. (2023) [ricu: R’s interface to intensive care data](https://doi.org/10.1093/gigascience/giad041) and the associated git-Repository [ricu](https://github.com/eth-mds/ricu/tree/main)
- Oliver et al. (2023) [Introducing the BlendedICU dataset, the first harmonized, international intensive care dataset](https://doi.org/10.1016/j.jbi.2023.104502) and the associated git-Repository [BlendedICU](https://github.com/USM-CHU-FGuyon/BlendedICU)