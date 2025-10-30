# reprodICU

**reprodICU** is a freely accessible pipeline, streamlining the creation of a harmonized critical care dataset, including data from up to 470k ICU admissions from multiple healthcare centers across the US and Europe. In this pipeline, **reprodICU** harmonizes data from the following publicly available ICU datasets, which were previously published by others: _AmsterdamUMCdb, eICU-CRD, HiRID, MIMIC-III, MIMIC-IV, NWICU, SICb_.

As part of the **Charité Outcomes Research Repository (CORR)**, the pipeline was developed by the **Institute of Medical Informatics** (IMI) at **Charité - Universitätsmedizin Berlin**.

The dataset created by running the pipeline contains de-identified demographic information and a total of 136 routinely collected physiological variables, diagnostic test results and treatment parameters from almost 350k patients during the period from 2001 to 2022.

---

reprodICU harmonizes clinical data from 7 international ICU datasets into a unified, analysis-ready format:

- **469,822** ICU admissions
- **7 datasets** across **4 countries**
- **2001–2022** timespan
- **136** harmonized clinical variables
- **150+** harmonized drugs
- almost **3 billion** data points in total

## What is reprodICU?

reprodICU is a Python package that:

1. **Extracts** raw clinical data from multiple ICU databases. Currently included are
   - [eICU Collaborative Research Database](https://physionet.org/content/eicu-crd/2.0/)
   - [HiRID, a high time-resolution ICU dataset](https://physionet.org/content/hirid/1.1.1/)
   - [MIMIC-III](https://physionet.org/content/mimiciii/1.4/)
   - [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/)
   - [Northwestern ICU (NWICU)](https://physionet.org/content/nwicu-northwestern-icu/0.1.0/)
   - [Salzburg Intensive Care database (SICdb)](https://physionet.org/content/sicdb/1.0.8/)
   - [AmsterdamUMCdb](https://amsterdammedicaldatascience.nl/#amsterdamumcdb)
2. **Harmonizes** variable definitions and units across datasets
3. **Validates** data for clinical plausibility
4. **Transforms** raw measurements into analysis-ready parquet files
5. **Derives** high-level clinical concepts (MAGIC_CONCEPTS) from raw data
6. **Provides** utilities for clinical scoring, mortality prediction, comorbidity assessment, and more

All output is **reproducible** – the same code produces identical results across datasets and analyses.

## Key Concepts

- **Axioms**: Raw, irreducible datapoints from measurements (e.g., a single heart rate reading)
- **Concepts**: Derived variables calculated from axioms or other concepts (e.g., sepsis status, code status)
- **MAGIC_CONCEPTS**: Pre-defined clinical concepts based on previous literature and clinical definitions, facilitating standardized analyses across datasets. Magic concepts are adapted from
  - the [MIMIC Code Repository](https://github.com/MIT-LCP/mimic-code)
  - the [eICU Code Repository](https://github.com/MIT-LCP/eicu-code)
  - the [AmsterdamUMCdb repo](https://github.com/AmsterdamUMC/AmsterdamUMCdb)
  - and the [ricu](https://physionet.org/content/ricu/1.0.0/) R package

## Getting Started

- **New users**: Start with [First Start](/first_start/) for installation and your first data build
- **Exploring utilities**: See [Usage & Utilities](/usage_utilities/) for clinical scoring, mortality prediction, and other tools
- **Understanding data**: Check [Data Structure](/data_structure/) for schema details and output formats
- **Deep dive**: Read about [MAGIC_CONCEPTS](/magic_concepts/) to understand derived variables
- **API Reference**: Browse [API Reference](/api_reference/) for complete function documentation

## Quick Start

```python
from reprodICU import build_all

# Build entire database from raw datasets
files = build_all(
    datasets=["eICU", "MIMIC3", "MIMIC4"],  # Specify datasets
    demo=False,  # Set True for smaller demo datasets
)
```

This creates 8 harmonized tables:

- `patient_information.parquet` – Demographics and admission data
- `diagnoses.parquet` – ICD codes and diagnoses
- `procedures.parquet` – Procedure codes
- `medications.parquet` – Administered medications
- `medications_prescribed.parquet` – Prescribed medications
- `microbiology.parquet` – Culture and susceptibility data
- `notes.parquet` – Clinical notes
- `timeseries_vitals.parquet` – Vital signs (HR, BP, etc.)
- `timeseries_labs.parquet` – Laboratory values
- `timeseries_respiratory.parquet` – Ventilation parameters
- `timeseries_intakeoutput.parquet` – Intake and output records

## Repository

**GitHub**: [CUB-CORR/reprodICU](https://github.com/CUB-CORR/reprodICU)

## Citation

If you use reprodICU in your research, please cite:

> [To be filled with actual publication information]

## License

This project is licensed under the [LICENSE](https://github.com/CUB-CORR/reprodICU/blob/main/LICENSE) included in the repository.
