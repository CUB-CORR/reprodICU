"""
Acute Respiratory Distress Syndrome (ARDS): identify ARDS cases using the New Global Definition.

This module implements ARDS identification based on the 2024 New Global Definition of ARDS,
which requires:
- Oxygenation criteria: PaO2/FiO2 ≤ 300 or SpO2/FiO2 ≤ 315
- Respiratory support: PEEP ≥ 5 cm H2O
- Bilateral infiltrates on chest imaging
- Exclusion of acute heart failure

References:
- Qian F, van den Boom W, See KC.
  The new global definition of acute respiratory distress syndrome: insights from the MIMIC-IV database.
  Intensive Care Med. 2024 Apr;50(4):608-609. doi: 10.1007/s00134-024-07383-x. Epub 2024 Mar 14. PMID: 38483560.

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

from typing import Optional

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

SECONDS_IN_1H = 60 * 60
SECONDS_IN_6H = 6 * SECONDS_IN_1H
SECONDS_IN_1D = 24 * SECONDS_IN_1H


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
    # ──────────────────────────────────────────────────────────────────────────
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

    STAY_KEY = "Global ICU Stay ID"
    TIME_KEY = "Time Relative to Admission (seconds)"

    patient_information = _to_lazy(patient_information)
    diagnoses = _to_lazy(diagnoses)
    timeseries_vitals = _to_lazy(timeseries_vitals)
    timeseries_labs = _to_lazy(timeseries_labs)
    timeseries_resp = _to_lazy(timeseries_resp)
    notes = _to_lazy(notes)
    vent = _to_lazy(vent)
    t_0_per_stay = _to_lazy(t_0_per_stay) if t_0_per_stay is not None else None

    all_stays = patient_information.select(STAY_KEY)
    all_stays_t0 = _build_t0(all_stays, t_0_per_stay, t_0)

    IDs = all_stays.unique()

    # region ICD / APACHE
    # ──────────────────────────────────────────────────────────────────────────
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
                pl.col("Diagnosis ICD Code").str.strip_chars().str.starts_with(ICD)
                for ICD in ARDS_ICD_CODES
            ).alias("ARDS by ICD"),
            pl.any_horizontal(
                pl.col("Diagnosis ICD Code").str.strip_chars().str.starts_with(ICD)
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

    # region chest imaging
    # ──────────────────────────────────────────────────────────────────────────
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
        .with_columns(
            (
                pl.col("Note Text (fixed)")
                .str.contains(r"bilateral (\w)* ?(\w)* ?(opaci|infil|haziness)")
                | pl.col("Note Text (fixed)")
                .str.contains(r"\.?([\w ]*)^(no )(opaci|infil|hazy|haziness)([\w ]+)bilaterally")
            ).alias("Bilateral infiltrates by chest imaging (Neto et al.)"),
            (
                (
                    pl.col("Note Text (fixed)")
                    .str.contains(r"bilateral (\w)* ?(\w)* ?(opaci|infil|haziness)") # fmt: skip
                    | pl.col("Note Text (fixed)")
                    .str.contains(r"(opaci|infil|hazy|haziness)([\w ]+)bilaterally") # fmt: skip
                    | pl.col("Note Text (fixed)")
                    .str.contains(r"(edema)")
                )
                & (
                    pl.col("Note Text (fixed)")
                    .str.contains(r"\b(no|without)\b[\w\s]*(bilateral (\w)* ?(\w)* ?(opaci|infil|haziness)|edema|(opaci|infil|hazy|haziness)([\w ]+)bilaterally)\b") # fmt: skip
                    | pl.col("Note Text (fixed)")
                    .str.contains(r"\bthere (is no|is no evidence of)\b[\w\s]*(bilateral (\w)* ?(\w)* ?(opaci|infil|haziness)|edema|(opaci|infil|hazy|haziness)([\w ]+)bilaterally)\b") # fmt: skip
                ).not_()
            ).alias("Bilateral infiltrates by chest imaging (Qian et al.)"),
        )
        .filter(
            pl.col("Bilateral infiltrates by chest imaging (Neto et al.)")
            | pl.col("Bilateral infiltrates by chest imaging (Qian et al.)")
        )
        .select(
            STAY_KEY,
            "Note Written Relative to Admission (seconds)",
            "Bilateral infiltrates by chest imaging (Neto et al.)",
            "Bilateral infiltrates by chest imaging (Qian et al.)",
        )
    )

    # region ventilator parameters
    # ──────────────────────────────────────────────────────────────────────────
    O2FLOW = timeseries_resp.select(
        STAY_KEY,
        TIME_KEY,
        "Oxygen delivery system",
        pl.col("Oxygen gas flow Oxygen delivery system").alias("O2 flow"),
    )

    PEEP = timeseries_resp.select(
        STAY_KEY,
        TIME_KEY,
        pl.max_horizontal(
            "Positive end expiratory pressure setting Ventilator",
            "PEEP Respiratory system",
        ).alias("PEEP"),
    )

    FiO2 = (
        timeseries_resp.select(
            STAY_KEY,
            TIME_KEY,
            pl.max_horizontal(
                "Oxygen/Total gas setting [Volume Fraction] Ventilator",
                "Oxygen/Gas total [Pure volume fraction] Inhaled gas",
            ).alias("FiO2"),
        )
        .with_columns(
            pl.when(pl.col("FiO2").is_between(0, 1))
            .then(pl.col("FiO2") * 100)
            .when(pl.col("FiO2").is_between(1, 100))
            .then(pl.col("FiO2"))
            .otherwise(None)
            .round(2)
            .alias("FiO2"),
        )
        .drop_nulls("FiO2")
    )

    # region O2 ratios
    # ──────────────────────────────────────────────────────────────────────────
    SPO2 = (
        timeseries_vitals.select(
            STAY_KEY,
            TIME_KEY,
            pl.col("Peripheral oxygen saturation").alias("SpO2"),
        ).drop_nulls("SpO2")
        # in New Global Definition of ARDS the cutoff is:
        # SpO2:FiO2 <= 315 (if SpO2  <= 97%)
        .filter(pl.col("SpO2").is_between(0, 97))
    )

    PAO2 = (
        timeseries_labs.select(
            STAY_KEY,
            TIME_KEY,
            "Oxygen",
        )
        .with_columns(
            pl.when(
                pl.col("Oxygen")
                .struct.field("system")
                .is_in(["Blood arterial", "Blood"])
                | pl.col("Oxygen").struct.field("system").is_null()
            )
            .then(pl.col("Oxygen").struct.field("value"))
            .otherwise(None)
            .alias("paO2")
        )
        .drop_nulls("paO2")
        .filter(pl.col("paO2") > 0)
        .select(
            STAY_KEY, TIME_KEY, "paO2"
        )
    )

    def O2_RATIOS(FiO2: pl.LazyFrame) -> pl.LazyFrame:
        """Calculate PaO2/FiO2 and SpO2/FiO2 ratios."""
        return (
            SPO2.join(
                PAO2,
                on=[STAY_KEY, TIME_KEY],
                how="outer",
                coalesce=True,
            )
            .join_asof(
                FiO2,
                on=TIME_KEY,
                by=STAY_KEY,
                strategy="backward",
                tolerance=SECONDS_IN_6H,
                coalesce=True,
            )
            .filter(pl.col("FiO2").is_not_null())
            .with_columns(
                pl.col("paO2")
                .truediv(pl.col("FiO2").truediv(100))
                .alias("PaO2/FiO2 ratio"),
                pl.col("SpO2")
                .truediv(pl.col("FiO2").truediv(100))
                .alias("SpO2/FiO2 ratio"),
            )
            .with_columns(
                pl.when(pl.col("PaO2/FiO2 ratio").is_finite())
                .then(pl.col("PaO2/FiO2 ratio"))
                .otherwise(None)
                .alias("PaO2/FiO2 ratio"),
                pl.when(pl.col("SpO2/FiO2 ratio").is_finite())
                .then(pl.col("SpO2/FiO2 ratio"))
                .otherwise(None)
                .alias("SpO2/FiO2 ratio"),
            )
        )

    INVASIVE_O2_RATIOS = O2_RATIOS(FiO2)

    NONINVASIVE_O2_RATIOS = O2_RATIOS(
        O2FLOW.filter(
            pl.col("Oxygen delivery system").str.contains_any(
                ["High flow", "Continuous positive"]
            )
        )
        # Global Definition ONLY:
        # Estimated FiO2  =  ambient FiO2 (e.g., 0.21)  +  0.03  ×  O2 flow rate (L/min)
        .with_columns(
            (0.21 + 0.03 * pl.col("O2 flow"))
            .mul(100)
            .clip(upper_bound=100)
            .alias("FiO2")
        )
    )

    # region ventilation
    # ──────────────────────────────────────────────────────────────────────────
    VENTILATION = (
        vent.rename(
            {
                "Ventilation Start Relative to Admission (seconds)": "Ventilation Start",
                "Ventilation End Relative to Admission (seconds)": "Ventilation End",
            }
        )
        .filter(pl.col("Ventilation Start") > 0)
        .with_columns(
            pl.col("Ventilation Start", "Ventilation End").cast(float),
            pl.when(pl.col(STAY_KEY).str.starts_with("eicu"))
            .then(True)
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
        .filter(pl.col("Invasive ventilation") | pl.col("Non-invasive ventilation"))
    )

    # region O2 ratios during ventilation
    # ──────────────────────────────────────────────────────────────────────────
    INVASIVE_VENT_O2_RATIOS = INVASIVE_O2_RATIOS.join(
        VENTILATION, on=STAY_KEY, how="left"
    ).filter(
        pl.col("Invasive ventilation"),
        pl.col(TIME_KEY).is_between(
            "Ventilation Start", "Ventilation End"
        ),
    )

    NONINVASIVE_VENT_O2_RATIOS = NONINVASIVE_O2_RATIOS.join(
        VENTILATION, on=STAY_KEY, how="left"
    ).filter(
        pl.col("Non-invasive ventilation"),
        pl.col(TIME_KEY).is_between(
            "Ventilation Start", "Ventilation End"
        ),
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

    O2_RATIO = (
        pl.concat(
            [INVASIVE_VENT_O2_RATIOS, NONINVASIVE_VENT_O2_RATIOS, HFNO_O2_RATIOS],
            how="diagonal_relaxed",
        )
        .unique()
        .group_by(STAY_KEY, TIME_KEY)
        .agg(
            pl.col("PaO2/FiO2 ratio", "SpO2/FiO2 ratio").min(),
            pl.exclude("PaO2/FiO2 ratio", "SpO2/FiO2 ratio").max(),
        )
    )

    # region ARDS criteria
    # ──────────────────────────────────────────────────────────────────────────
    ards_cohort = (
        IDs.join(ARDS_AHF_BY_ICD, on=STAY_KEY, how="left")
        .join(ARDS_BY_APACHE, on=STAY_KEY, how="left")
        .join(O2_RATIO, on=STAY_KEY, how="left")
        .join(
            PEEP,
            on=[STAY_KEY, TIME_KEY],
            how="left",
        )
        .join_asof(
            BILATERAL_INFILTRATES,
            left_on="Ventilation Start",
            right_on="Note Written Relative to Admission (seconds)",
            by=STAY_KEY,
            strategy="nearest",
            tolerance=SECONDS_IN_1D,
            coalesce=True,
        )
        .filter(
            (pl.col("PaO2/FiO2 ratio") <= 300) | (pl.col("SpO2/FiO2 ratio") <= 315),
            pl.col("PEEP") >= 5,
            # pl.col("Bilateral infiltrates by chest imaging (Neto et al.)")
            pl.col("Bilateral infiltrates by chest imaging (Qian et al.)")
            | pl.col("ARDS by ICD")
            | pl.col("ARDS by APACHE"),
            pl.col("AHF by ICD").fill_null(False).not_(),
        )
        .sort(STAY_KEY, TIME_KEY)
        .group_by(STAY_KEY)
        .agg(
            pl.min(TIME_KEY).alias("ARDS onset"),
            pl.col(
                "Ventilation Start",
                "Ventilation End",
                "Ventilation Type"
            ).min(),
        )
    )

    # Always join with T_0 and compute relative times
    ards_cohort = (
        ards_cohort.join(all_stays_t0, on=STAY_KEY, how="inner")
        .select(
            STAY_KEY,
            pl.col("ARDS onset")
            .sub(pl.col("T_0"))
            .alias("ARDS onset (relative to T_0)"),
            pl.col("Ventilation Start")
            .sub(pl.col("T_0"))
            .alias("Ventilation Start (relative to T_0)"),
            pl.col("Ventilation End")
            .sub(pl.col("T_0"))
            .alias("Ventilation End (relative to T_0)"),
        )
    )

    return ards_cohort
    
__all__ = ["ARDS"]