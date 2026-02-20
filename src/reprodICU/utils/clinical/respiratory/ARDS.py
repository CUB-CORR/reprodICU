"""
Acute Respiratory Distress Syndrome (ARDS): identify ARDS cases using the New Global Definition.

This module implements ARDS identification based on the 2024 New Global Definition of ARDS,
which requires:
- Oxygenation criteria: PaO2/FiO2 ≤ 300 or SpO2/FiO2 ≤ 315
- Respiratory support: PEEP ≥ 5 cm H2O
- Bilateral infiltrates on chest imaging
- Exclusion of acute heart failure

References:
- Serpa Neto A, Deliberato RO, Johnson AEW, Bos LD, Amorim P, Pereira SM, Cazati DC, Cordioli RL, Correa TD, Pollard TJ, Schettino GPP, Timenetsky KT, Celi LA, Pelosi P, Gama de Abreu M, Schultz MJ; PROVE Network Investigators.
  Mechanical power of ventilation is associated with mortality in critically ill patients: an analysis of patients in two observational cohorts.
  Intensive Care Med. 2018 Nov;44(11):1914-1922. doi: 10.1007/s00134-018-5375-6. Epub 2018 Oct 5. PMID: 30291378.
- Qian F, van den Boom W, See KC.
  The new global definition of acute respiratory distress syndrome: insights from the MIMIC-IV database.
  Intensive Care Med. 2024 Apr;50(4):608-609. doi: 10.1007/s00134-024-07383-x. Epub 2024 Mar 14. PMID: 38483560.
- Pensier J, Fosset M, Paschold BS, von Wedel D, Redaelli S, Braeuer BLP, Novack V, Balzer F, Jung B, Amato MBP, Jaber S, Talmor D, Baedorf-Kassis E, Schaefer MS.
  Temporal stability of phenotypes of acute respiratory distress syndrome: clinical implications for early corticosteroid therapy and mortality.
  Intensive Care Med. 2025 Oct;51(10):1784-1796. doi: 10.1007/s00134-025-08089-4. Epub 2025 Aug 21. PMID: 40839098.

- NEW GLOBAL DEFINITION OF ACUTE RESPIRATORY DISTRESS SYNDROME
  Matthay MA, Arabi Y, Arroliga AC, Bernard G, Bersten AD, Brochard LJ, Calfee CS, Combes A, Daniel BM, Ferguson ND,
  Gong MN, Gotts JE, Herridge MS, Laffey JG, Liu KD, Machado FR, Martin TR, McAuley DF, Mercat A, Moss M,
  Mularski RA, Pesenti A, Qiu H, Ramakrishnan N, Ranieri VM, Riviello ED, Rubin E, Slutsky AS, Thompson BT,
  Twagirumugabe T, Ware LB, Wick KD.
  A New Global Definition of Acute Respiratory Distress Syndrome.
  Am J Respir Crit Care Med. 2024 Jan 1;209(1):37-47. doi: 10.1164/rccm.202303-0558WS. PMID: 37487152; PMCID: PMC10870872.
- BERLIN DEFINITION OF ACUTE RESPIRATORY DISTRESS SYNDROME
  ARDS Definition Task Force; Ranieri VM, Rubenfeld GD, Thompson BT, Ferguson ND, Caldwell E, Fan E, Camporota L, Slutsky AS.
  Acute respiratory distress syndrome: the Berlin Definition.
  JAMA. 2012 Jun 20;307(23):2526-33. doi: 10.1001/jama.2012.5669. PMID: 22797452.
"""

from typing import Literal, Optional

import polars as pl

from ...common import (
    _build_t0,
    _to_lazy,
    get_diagnoses,
    get_notes,
    get_patient_information,
    get_timeseries_labs,
    get_timeseries_respiratory,
    get_timeseries_vitals,
    get_ventilation,
)
from .PF_RATIO import PaO2_FiO2_RATIO, SpO2_FiO2_RATIO

SECONDS_IN_1H = 60 * 60
SECONDS_IN_6H = 6 * SECONDS_IN_1H
SECONDS_IN_1D = 24 * SECONDS_IN_1H

STAY_KEY = "Global ICU Stay ID"
TIME_KEY = "Time Relative to Admission (seconds)"


# region IMAGING_CRITERION
def IMAGING_CRITERION(
    patient_information: Optional[pl.LazyFrame] = None,
    notes: Optional[pl.LazyFrame] = None,
    *,
    t_0: Optional[int] = 0,
    t_0_per_stay: Optional[pl.LazyFrame] = None,
    definition_source: Literal["Qian", "Neto", "Pensier", "any"] = "Qian",
) -> pl.LazyFrame:
    """
    Detect bilateral infiltrates in chest imaging reports.

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information. Loaded automatically if None.
        notes : pl.LazyFrame, optional
            Clinical notes data. Loaded automatically if None.
        t_0 : int, optional
            Scalar reference time (seconds from admission). Defaults to 0 (admission).
            Ignored when t_0_per_stay is provided.
        t_0_per_stay : pl.LazyFrame, optional
            Per-stay T_0 overrides with columns [Global ICU Stay ID, T_0].
        definition_source : str, optional
            Choice of definition for detecting bilateral infiltrates in chest
            imaging reports. Can be 'Qian', 'Neto', 'Pensier', or 'any'.
            Defaults to 'Qian'.

    Returns
    -------
        pl.LazyFrame with columns:
            - Global ICU Stay ID
            - Note Written Relative to Admission (seconds)
            - Bilateral Infiltrates
    """

    # Load defaults if not provided
    if patient_information is None:
        patient_information = get_patient_information()
    if notes is None:
        notes = get_notes()

    # Validate all required data is available
    required = {
        "patient_information": patient_information,
        "notes": notes,
    }

    missing = [name for name, data in required.items() if data is None]
    if missing:
        raise ValueError(
            f"Cannot compute IMAGING_CRITERION: Missing required datasets: {', '.join(missing)}. "
            f"Ensure they are configured in ~/.reprodICU/PATHS.yaml or provide them explicitly."
        )

    notes = _to_lazy(notes)

    t_0_per_stay = _to_lazy(t_0_per_stay) if t_0_per_stay is not None else None

    all_stays = patient_information.select(STAY_KEY)
    all_stays_t0 = _build_t0(all_stays, t_0_per_stay, t_0)

    # Neto et al. 2018
    NETO_PATTERN = (
        pl.col("Note Text (fixed)")
        .str.contains(r"bilateral (\w)* ?(\w)* ?(opaci|infil|haziness)") 
        | pl.col("Note Text (fixed)")
        .str.contains(r"\.?([\w ]*)^(no )(opaci|infil|hazy|haziness)([\w ]+)bilaterally")
    )  # fmt: skip

    # Qian et al. 2024
    QIAN_PATTERN = (
        pl.col("Note Text (fixed)")
        .str.contains(r"bilateral (\w)* ?(\w)* ?(opaci|infil|haziness)")
        | pl.col("Note Text (fixed)")
        .str.contains(r"(opaci|infil|hazy|haziness)([\w ]+)bilaterally")
        | pl.col("Note Text (fixed)").str.contains(r"(edema)")
    ) & (
        pl.col("Note Text (fixed)")
        .str.contains(r"\b(no|without)\b[\w\s]*(bilateral (\w)* ?(\w)* ?(opaci|infil|haziness)|edema|(opaci|infil|hazy|haziness)([\w ]+)bilaterally)\b")
        | pl.col("Note Text (fixed)")
        .str.contains(r"\bthere (is no|is no evidence of)\b[\w\s]*(bilateral (\w)* ?(\w)* ?(opaci|infil|haziness)|edema|(opaci|infil|hazy|haziness)([\w ]+)bilaterally)\b")
    ).not_()  # fmt: skip

    # Pensier et al. 2025
    PENSIER_PATTERN = (
        pl.col("Note Text (fixed)")
        .str.contains(r"(edem|infiltra|condensation|pneumoni)") 
        & pl.col("Note Text (fixed)").str.contains("bilateral")
    )  # fmt: skip

    if definition_source == "Neto":
        imaging_expr = NETO_PATTERN.alias("Bilateral Infiltrates")
    elif definition_source == "Qian":
        imaging_expr = QIAN_PATTERN.alias("Bilateral Infiltrates")
    elif definition_source == "Pensier":
        imaging_expr = PENSIER_PATTERN.alias("Bilateral Infiltrates")
    else:
        imaging_expr = (NETO_PATTERN | QIAN_PATTERN | PENSIER_PATTERN).alias("Bilateral Infiltrates")  # fmt: skip

    BILATERAL_INFILTRATES = (
        notes.filter(
            pl.col("Note Category") == "Radiology",
            pl.col("Note Text").str.to_lowercase().str.contains("chest"),
        )
        .with_columns(
            pl.col("Note Text")
            .str.replace_all(r"\n", " ")
            .str.replace_all(r"\r", "")
            .str.to_lowercase()
            .alias("Note Text (fixed)")
        )
        .with_columns(imaging_expr)
        .filter(pl.col("Bilateral Infiltrates"))
        .select(
            STAY_KEY,
            pl.col("Note Written Relative to Admission (seconds)").alias(TIME_KEY),
            "Bilateral Infiltrates",
        ) # fmt: skip
    )

    if (t_0 != 0) or (t_0_per_stay is not None):
        BILATERAL_INFILTRATES = (
            BILATERAL_INFILTRATES.join(all_stays_t0, on=STAY_KEY, how="inner")
            .with_columns(
                pl.col(TIME_KEY)
                .sub(pl.col("T_0"))
                .alias("Time Relative to T_0 (seconds)")
            )
            .drop("T_0")
        )

    return BILATERAL_INFILTRATES


# region RESPIRATORY_CRITERION
def RESPIRATORY_CRITERION(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_vitals: Optional[pl.LazyFrame] = None,
    timeseries_labs: Optional[pl.LazyFrame] = None,
    timeseries_resp: Optional[pl.LazyFrame] = None,
    vent: Optional[pl.LazyFrame] = None,
    *,
    t_0: Optional[int] = 0,
    t_0_per_stay: Optional[pl.LazyFrame] = None,
) -> pl.LazyFrame:
    """
    Identify periods meeting respiratory criteria:
    - P/F ratio <= 300 or S/F ratio <= 315
      AND
    - PEEP >= 5 cm H2O

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information. Loaded automatically if None.
        timeseries_vitals : pl.LazyFrame, optional
            Vital signs timeseries data. Loaded automatically if None.
        timeseries_labs : pl.LazyFrame, optional
            Laboratory timeseries data. Loaded automatically if None.
        timeseries_resp : pl.LazyFrame, optional
            Respiratory timeseries data. Loaded automatically if None.
        vent : pl.LazyFrame, optional
            Ventilation timeseries data. Loaded automatically if None.
        t_0 : int, optional
            Scalar reference time (seconds from admission). Defaults to 0 (admission).
            Ignored when t_0_per_stay is provided.
        t_0_per_stay : pl.LazyFrame, optional
            Per-stay T_0 overrides with columns [Global ICU Stay ID, T_0].

    Returns
    -------
        pl.LazyFrame with columns:
            - Global ICU Stay ID
            - Time Relative to Admission (seconds)
            - PaO2/FiO2 Ratio
            - SpO2/FiO2 Ratio
            - PEEP
            - Ventilation Start
            - Ventilation End
            - Ventilation Type
    """

    # Load defaults if not provided
    if patient_information is None:
        patient_information = get_patient_information()
    if timeseries_vitals is None:
        timeseries_vitals = get_timeseries_vitals()
    if timeseries_labs is None:
        timeseries_labs = get_timeseries_labs()
    if timeseries_resp is None:
        timeseries_resp = get_timeseries_respiratory()
    if vent is None:
        vent = get_ventilation()

    # Validate all required data is available
    required = {
        "patient_information": patient_information,
        "timeseries_vitals": timeseries_vitals,
        "timeseries_labs": timeseries_labs,
        "timeseries_resp": timeseries_resp,
        "vent": vent,
    }

    missing = [name for name, data in required.items() if data is None]
    if missing:
        raise ValueError(
            f"Cannot compute ARDS: Missing required datasets: {', '.join(missing)}. "
            f"Ensure they are configured in ~/.reprodICU/PATHS.yaml or provide them explicitly."
        )

    patient_information = _to_lazy(patient_information)
    timeseries_vitals = _to_lazy(timeseries_vitals)
    timeseries_labs = _to_lazy(timeseries_labs)
    timeseries_resp = _to_lazy(timeseries_resp)
    vent = _to_lazy(vent)
    t_0_per_stay = _to_lazy(t_0_per_stay) if t_0_per_stay is not None else None

    all_stays = patient_information.select(STAY_KEY)
    all_stays_t0 = _build_t0(all_stays, t_0_per_stay, t_0)

    # region ventilator parameters
    # --------------------------------------------------------------------------
    PEEP = timeseries_resp.select(
        STAY_KEY,
        TIME_KEY,
        pl.max_horizontal(
            "Positive end expiratory pressure setting Ventilator",
            "PEEP Respiratory system",
        ).alias("PEEP"),
    )

    # region O2 ratios
    # --------------------------------------------------------------------------
    def O2_RATIOS(fio2_type: str) -> pl.LazyFrame:
        """Calculate PaO2/FiO2 and SpO2/FiO2 ratios."""
        pf = PaO2_FiO2_RATIO(
            patient_information=patient_information,
            timeseries_resp=timeseries_resp,
            timeseries_labs=timeseries_labs,
            t_0=None,
            tolerance=SECONDS_IN_6H,
            fio2_type=fio2_type,
        )
        sf = SpO2_FiO2_RATIO(
            patient_information=patient_information,
            timeseries_resp=timeseries_resp,
            timeseries_vitals=timeseries_vitals,
            t_0=None,
            tolerance=SECONDS_IN_6H,
            fio2_type=fio2_type,
        )

        return pf.join(sf, on=[STAY_KEY, TIME_KEY], how="full", coalesce=True)

    INVASIVE_O2_RATIOS = O2_RATIOS(fio2_type="invasive")
    NONINVASIVE_O2_RATIOS = O2_RATIOS(fio2_type="non-invasive")


    # region ventilation
    # --------------------------------------------------------------------------
    VENTILATION = (
        vent.rename(
            {
                "Ventilation Start Relative to Admission (seconds)": "Ventilation Start",
                "Ventilation End Relative to Admission (seconds)": "Ventilation End",
            }  # fmt: skip
        )
        .filter(pl.col("Ventilation Start") > 0)
        .with_columns(
            pl.col("Ventilation Start", "Ventilation End").cast(float),
            pl.when(pl.col(STAY_KEY).str.starts_with("eicu"))
            .then(True)  # assume invasive ventilation for eICU
            .otherwise(
                pl.col("Ventilation Type").is_in(
                    ["invasive ventilation", "tracheostomy"]
                )
            )
            .alias("Invasive ventilation"),
            pl.col("Ventilation Type")
            .eq_missing("non-invasive ventilation")
            .alias("Non-invasive ventilation"),
        )
        .filter(
            pl.col("Invasive ventilation") | pl.col("Non-invasive ventilation")
        )
    )

    # region O2 ratios during ventilation
    # --------------------------------------------------------------------------
    INVASIVE_VENT_O2_RATIOS = INVASIVE_O2_RATIOS.join(
        VENTILATION, on=STAY_KEY, how="left"
    ).filter(
        pl.col("Invasive ventilation"),
        pl.col(TIME_KEY).is_between("Ventilation Start", "Ventilation End"),
    )

    NONINVASIVE_VENT_O2_RATIOS = NONINVASIVE_O2_RATIOS.join(
        VENTILATION, on=STAY_KEY, how="left"
    ).filter(
        pl.col("Non-invasive ventilation"),
        pl.col(TIME_KEY).is_between("Ventilation Start", "Ventilation End"),
    )

    HFNO_O2_RATIOS = (
        NONINVASIVE_O2_RATIOS.join(VENTILATION, on=STAY_KEY, how="left")
        .filter(
            pl.col(TIME_KEY)
            .is_between("Ventilation Start", "Ventilation End")
            .not_()
        )
        .select(*NONINVASIVE_O2_RATIOS.collect_schema().names())
    )

    CRITERION = (
        pl.concat(
            [
                INVASIVE_VENT_O2_RATIOS,
                NONINVASIVE_VENT_O2_RATIOS,
                HFNO_O2_RATIOS,
            ],
            how="diagonal_relaxed",
        )
        .unique()
        .group_by(STAY_KEY, TIME_KEY)
        .agg(
            pl.col("PaO2/FiO2 Ratio", "SpO2/FiO2 Ratio").min(),
            pl.exclude("PaO2/FiO2 Ratio", "SpO2/FiO2 Ratio").max(),
        )
        .join(PEEP, on=[STAY_KEY, TIME_KEY], how="left")
        .filter(
            (pl.col("PaO2/FiO2 Ratio") <= 300)
            | (pl.col("SpO2/FiO2 Ratio") <= 315),
            pl.col("PEEP") >= 5,
        )
    )

    if (t_0 != 0) or (t_0_per_stay is not None):
        CRITERION = (
            CRITERION.join(all_stays_t0, on=STAY_KEY, how="inner")
            .with_columns(
                pl.col(TIME_KEY)
                .sub(pl.col("T_0"))
                .alias("Time Relative to T_0 (seconds)")
            )
            .drop("T_0")
        )

    return CRITERION


# region ARDS
def ARDS(
    patient_information: Optional[pl.LazyFrame] = None,
    diagnoses: Optional[pl.LazyFrame] = None,
    timeseries_vitals: Optional[pl.LazyFrame] = None,
    timeseries_labs: Optional[pl.LazyFrame] = None,
    timeseries_resp: Optional[pl.LazyFrame] = None,
    notes: Optional[pl.LazyFrame] = None,
    vent: Optional[pl.LazyFrame] = None,
    *,
    t_0: Optional[int] = 0,
    t_0_per_stay: Optional[pl.LazyFrame] = None,
    definition_source: Literal["Qian", "Neto", "Pensier", "any"] = "Qian",
) -> pl.LazyFrame:
    """
    Identify ARDS cases using the New Global Definition of ARDS.

    Applies New Global Definition criteria: oxygenation ratio thresholds (PaO2/FiO2 ≤ 300
    or SpO2/FiO2 ≤ 315), PEEP ≥ 5 cm H2O, bilateral infiltrates on chest imaging,
    and exclusion of acute heart failure.

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information. Loaded automatically if None.
        diagnoses : pl.LazyFrame, optional
            Diagnosis data with ICD codes. Loaded automatically if None.
        timeseries_vitals : pl.LazyFrame, optional
            Vital signs timeseries data. Loaded automatically if None.
        timeseries_labs : pl.LazyFrame, optional
            Laboratory timeseries data. Loaded automatically if None.
        timeseries_resp : pl.LazyFrame, optional
            Respiratory timeseries data. Loaded automatically if None.
        notes : pl.LazyFrame, optional
            Clinical notes data. Loaded automatically if None.
        vent : pl.LazyFrame, optional
            Ventilation timeseries data. Loaded automatically if None.
        t_0 : int, optional
            Scalar reference time (seconds from admission). Defaults to 0 (admission).
            Ignored when t_0_per_stay is provided.
        t_0_per_stay : pl.LazyFrame, optional
            Per-stay T_0 overrides with columns [Global ICU Stay ID, T_0].
        definition_source : str, optional
            Choice of definition for detecting bilateral infiltrates in chest
            imaging reports. Can be 'Qian', 'Neto', 'Pensier', or 'any'.
            Defaults to 'Qian'.

    Returns
    -------
        pl.LazyFrame
            ARDS cases with columns:
            - Global ICU Stay ID
            - ARDS onset (time in seconds from admission)
            - Ventilation Start
            - Ventilation End
            - Ventilation Type
    """

    # region load datasets
    # --------------------------------------------------------------------------
    # Load defaults if not provided
    if patient_information is None:
        patient_information = get_patient_information()
    if diagnoses is None:
        diagnoses = get_diagnoses()
    if timeseries_vitals is None:
        timeseries_vitals = get_timeseries_vitals()
    if timeseries_labs is None:
        timeseries_labs = get_timeseries_labs()
    if timeseries_resp is None:
        timeseries_resp = get_timeseries_respiratory()
    if notes is None:
        notes = get_notes()
    if vent is None:
        vent = get_ventilation()

    # Validate all required data is available
    required = {
        "patient_information": patient_information,
        "diagnoses": diagnoses,
        "timeseries_vitals": timeseries_vitals,
        "timeseries_labs": timeseries_labs,
        "timeseries_resp": timeseries_resp,
        "notes": notes,
        "vent": vent,
    }

    missing = [name for name, data in required.items() if data is None]
    if missing:
        raise ValueError(
            f"Cannot compute ARDS: Missing required datasets: {', '.join(missing)}. "
            f"Ensure they are configured in ~/.reprodICU/PATHS.yaml or provide them explicitly."
        )

    patient_information = _to_lazy(patient_information)
    diagnoses = _to_lazy(diagnoses)
    timeseries_vitals = _to_lazy(timeseries_vitals)
    timeseries_labs = _to_lazy(timeseries_labs)
    timeseries_resp = _to_lazy(timeseries_resp)
    notes = _to_lazy(notes)
    vent = _to_lazy(vent)
    t_0_per_stay = _to_lazy(t_0_per_stay) if t_0_per_stay is not None else None

    all_stays = patient_information.select(STAY_KEY).unique()
    all_stays_t0 = _build_t0(all_stays, t_0_per_stay, t_0)

    # region ICD / APACHE
    # --------------------------------------------------------------------------
    # ICD-9 - 518.5 is included due to
    # CBER Surveillance Program Biologics Effectiveness and Safety Initiative
    # A Structured Review of Electronic Coding Algorithms for Acute Respiratory Distress Syndrome (ARDS) Using Administrative Claims and Electronic Health Records - Final Report
    # https://bestinitiative.org/wp-content/uploads/2022/05/ARDS_Algorithm_Final_Report_-2021.pdf

    ARDS_ICD_CODES = ["51882", "5185", "J80"]
    AHF_ICD_CODES = ["428", "I50"]

    ARDS_AHF_BY_ICD = (
        diagnoses.with_columns(
            pl.when(pl.col("Diagnosis ICD Code Version (source)") == "ICD-9")
            .then(pl.col("Diagnosis ICD-9 Code"))
            .otherwise(pl.col("Diagnosis ICD-10 Code"))
            .alias("Diagnosis ICD Code")
        )
        .with_columns(
            pl.any_horizontal(
                pl.col("Diagnosis ICD Code")
                .str.strip_chars()
                .str.starts_with(ICD)
                for ICD in ARDS_ICD_CODES
            ).alias("ARDS by ICD"),
            pl.any_horizontal(
                pl.col("Diagnosis ICD Code")
                .str.strip_chars()
                .str.starts_with(ICD)
                for ICD in AHF_ICD_CODES
            ).alias("AHF by ICD"),
        )
        .select(STAY_KEY, "ARDS by ICD", "AHF by ICD")
        .group_by(STAY_KEY)
        .max()
    )

    ARDS_BY_APACHE = (
        patient_information.select(STAY_KEY, "Admission Diagnosis (APACHE)")
        .with_columns(
            pl.col("Admission Diagnosis (APACHE)")
            .str.to_lowercase()
            .str.contains("ards")
            .alias("ARDS by APACHE")
        )
        .select(STAY_KEY, "ARDS by APACHE")
        .group_by(STAY_KEY)
        .max()
    )

    # region Criteria
    # --------------------------------------------------------------------------
    BILATERAL_INFILTRATES = IMAGING_CRITERION(
        patient_information=patient_information,
        notes=notes,
        t_0=t_0,
        t_0_per_stay=t_0_per_stay,
        definition_source=definition_source,
    )

    RESPIRATORY_DATA = RESPIRATORY_CRITERION(
        patient_information=patient_information,
        timeseries_vitals=timeseries_vitals,
        timeseries_labs=timeseries_labs,
        timeseries_resp=timeseries_resp,
        vent=vent,
        t_0=t_0,
        t_0_per_stay=t_0_per_stay,
    )

    # region ARDS cohort
    # --------------------------------------------------------------------------
    ARDS_COHORT = (
        RESPIRATORY_DATA.join_asof(
            BILATERAL_INFILTRATES,
            left_on=TIME_KEY,
            right_on=TIME_KEY,
            by=STAY_KEY,
            strategy="nearest",
            tolerance=SECONDS_IN_1D,
            coalesce=True,
            allow_exact_matches=True,
        )
        .join(ARDS_AHF_BY_ICD, on=STAY_KEY, how="left")
        .join(ARDS_BY_APACHE, on=STAY_KEY, how="left")
        .filter(
            pl.col("Bilateral Infiltrates").fill_null(False)
            | pl.col("ARDS by ICD").fill_null(False)
            | pl.col("ARDS by APACHE").fill_null(False),
            pl.col("AHF by ICD").fill_null(False).not_(),
        )
        .sort(STAY_KEY, TIME_KEY)
        .group_by(STAY_KEY)
        .agg(
            pl.min(TIME_KEY).alias("ARDS onset"),
            pl.col(
                "Ventilation Start",
                "Ventilation End",
                "Ventilation Type",
            ).min(),
        )
    )

    if (t_0 != 0) or (t_0_per_stay is not None):
        ARDS_COHORT = ARDS_COHORT.join(
            all_stays_t0, on=STAY_KEY, how="inner"
        ).select(
            STAY_KEY,
            pl.col("ARDS onset")
            .sub(pl.col("T_0"))
            .alias("ARDS onset Relative to T_0 (seconds)"),
            pl.col("Ventilation Start")
            .sub(pl.col("T_0"))
            .alias("Ventilation Start Relative to T_0 (seconds)"),
            pl.col("Ventilation End")
            .sub(pl.col("T_0"))
            .alias("Ventilation End Relative to T_0 (seconds)"),
            "Ventilation Type",
        )

    return ARDS_COHORT


__all__ = ["ARDS"]
