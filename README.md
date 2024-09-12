# reprodICU
A comprehensive database harmonization and interaction package for multiple ICU databases.

## Based on previous work by
- Bennett et al. (2023) [ricu: R’s interface to intensive care data](https://doi.org/10.1093/gigascience/giad041) and the associated git-Repository [ricu](https://github.com/eth-mds/ricu/tree/main)
- Oliver et al. (2023) [Introducing the BlendedICU dataset, the first harmonized, international intensive care dataset](https://doi.org/10.1016/j.jbi.2023.104502) and the associated git-Repository [BlendedICU](https://github.com/USM-CHU-FGuyon/BlendedICU)


## Basic structure of the reprodICU(bility) database

0. labels
    - age, height, weight on admission
    - location

1. timeseries (sparse)
Each measurement is associated with an ICU stay ID and a timepoint
    - vitals
    - laboratory values

2. medications
    - ICU stay ID
    - drug name (as in source database)
    - harmonized ingredient
    - medication start time
    - medication end time
    - medication dose
    - medication unit

3. diagnoses