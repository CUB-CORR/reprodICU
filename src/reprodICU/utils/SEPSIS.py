"""
SEPSIS: compute Sepsis-3 in long format directly from raw inputs.

Output columns per row:
- Global ICU Stay ID
- timeframe (0-indexed integer window)
- SEPSIS      (Seymour-anchored on culture+antibiotic pair) -> "SEPSIS" | "SHOCK"
- SEPSIS_ABX  (Shah-anchored on antibiotic escalation)      -> "SEPSIS" | "SHOCK"
- SEPSIS_RHEE (Rhee EHR surveillance definition)            -> "SEPSIS"
- T_0 (seconds from admission used as reference)

Time is in seconds. Window index = floor((time - T_0)/window_size).
Worst-within-window aggregation is applied for SOFA via SOFA.
Three independent suspicion anchors are produced and kept separate.

Sources
-------
- Singer M, Deutschman CS, Seymour CW, Shankar-Hari M, Annane D, Bauer M, Bellomo R, Bernard GR, Chiche JD, Coopersmith CM, Hotchkiss RS, Levy MM, Marshall JC, Martin GS, Opal SM, Rubenfeld GD, van der Poll T, Vincent JL, Angus DC.
  The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3).
  JAMA. 2016 Feb 23;315(8):801-10. doi: 10.1001/jama.2016.0287. PMID: 26903338; PMCID: PMC4968574.
- Shah AD, MacCallum NS, Harris S, Brealey DA, Palmer E, Hetherington J, Shi S, Perez-Suarez D, Ercole A, Watkinson PJ, Jones A, Ashworth S, Beale R, Brett SJ, Singer M.
  Descriptors of Sepsis Using the Sepsis-3 Criteria: A Cohort Study in Critical Care Units Within the U.K. National Institute for Health Research Critical Care Health Informatics Collaborative.
  Crit Care Med. 2021 Nov 1;49(11):1883-1894. doi: 10.1097/CCM.0000000000005169. PMID: 34259454; PMCID: PMC8508729.
- Rhee C, Kadri S, Huang SS, Murphy MV, Li L, Platt R, Klompas M.
  Objective Sepsis Surveillance Using Electronic Clinical Data.
  Infect Control Hosp Epidemiol. 2016 Feb;37(2):163-71. doi: 10.1017/ice.2015.264. Epub 2015 Nov 3. PMID: 26526737; PMCID: PMC4743875.
"""

# author: Finn Fassbender
# version: 31.08.2025
from pathlib import Path
from typing import Optional

import polars as pl
import yaml

from .common import (
    _assign_timeframe,
    _build_t0,
    _optional_time_bounds_filter,
    get_medications,
    get_microbiology,
    get_patient_information,
    get_procedures,
    get_timeseries_intakeoutput,
    get_timeseries_labs,
    get_timeseries_respiratory,
    get_timeseries_vitals,
    get_ventilation
)
from .SOFA import SOFA

# seconds constants
SECONDS_PER_HOUR = 60 * 60
SECONDS_PER_DAY = 24 * SECONDS_PER_HOUR
TWELVE_HOURS = 12 * SECONDS_PER_HOUR

# strict time column name used across helpers
STAY_COL = "Global ICU Stay ID"
TIME_COL = "Time Relative to Admission (seconds)"


# region helpers
def _read_antibiotic_ranks() -> dict[str, int]:
    ranks_path = Path(__file__).parent / "ANTIBIOTIC_RANKS.yaml"
    with open(ranks_path, "r") as f:
        data = yaml.safe_load(f) or {}
    # Expect mapping { ingredient (str): rank (int) }
    return {str(k): int(v) for k, v in data.items()}


def _admission_time_to_seconds(data: pl.LazyFrame) -> pl.LazyFrame:
    """Extract seconds since midnight from a pl.Time column.

    Returns:
        Expression yielding seconds since midnight (0-86399)

    Example: 14:30:00 -> 14*3600 + 30*60 = 52200 seconds
    """
    return data.with_columns(
        pl.when(pl.col("Admission Time (24h)").is_null())
        .then(pl.lit(0))  # Default to midnight if missing
        .otherwise(
            pl.col("Admission Time (24h)")
            .dt.hour()
            .mul(3600)
            .add(pl.col("Admission Time (24h)").dt.minute().mul(60))
            .add(pl.col("Admission Time (24h)").dt.second())  # seconds
        )
        .cast(pl.Int64)
        .alias("Admission Time (seconds)")
    )


def _calendar_day_from_admission(
    time_seconds: pl.Expr, admission_seconds_into_day: pl.Expr
) -> pl.Expr:
    """Compute calendar day number from admission given admission time.

    Calendar day 0 is the day of admission (from midnight to next midnight).
    Takes into account the admission time within that day.

    Args:
        time_seconds: Time relative to admission in seconds
        admission_seconds_into_day: Seconds from midnight to admission time (0-86399)

    Returns:
        Calendar day index (0 = admission day, 1 = next day, -1 = day before, etc.)

    Example:
        Admission at 14:00 (50400 seconds into day)
        - time=-3600 (1h before admission = 13:00) -> day 0 (same calendar day)
        - time=-39600 (11h before admission = 03:00) -> day -1 (previous calendar day)
        - time=36000 (10h after admission = 00:00 next day) -> day 1
    """
    # Seconds from the start of admission day
    seconds_from_day_start = time_seconds + admission_seconds_into_day
    # Integer division by SECONDS_PER_DAY gives calendar day
    return seconds_from_day_start.floordiv(SECONDS_PER_DAY)


def rhee_compute_lab_baselines(
    all_stays: pl.LazyFrame,
    patient_information: pl.LazyFrame,
    timeseries_labs: pl.LazyFrame,
    lab_columns: list[str],
    baseline_mode: dict[str, str],  # {lab_name: "min" | "max"}
) -> pl.LazyFrame:
    """Compute baseline lab values from day -30 to end of stay per Rhee criteria.

    Args:
        all_stays: DataFrame with Global ICU Stay ID
        patient_information: Contains admission time and LOS
        timeseries_labs: Time series lab data
        lab_columns: List of lab column names to compute baselines for
        baseline_mode: Dict mapping lab name to aggregation mode ("min" or "max")

    Returns:
        LazyFrame with columns:
        - Global ICU Stay ID
        - baseline_{lab_name} for each lab in lab_columns
        - admission_seconds_into_day (for downstream calendar day calculations)
    """
    all_stays = all_stays.lazy()
    patient_info = patient_information.lazy()
    labs = timeseries_labs.lazy()

    # Get admission time and LOS
    stays_with_time = all_stays.join(
        patient_info.select(
            STAY_COL, "Admission Time (24h)", "ICU Length of Stay (days)"
        ),
        on=STAY_COL,
        how="left",
    ).pipe(_admission_time_to_seconds)

    # Define baseline period: day -30 to day of discharge
    # Need to calculate calendar day for each lab measurement
    labs_with_day = (
        labs.select(STAY_COL, TIME_COL, *lab_columns)
        .join(stays_with_time, on=STAY_COL, how="inner")
        .with_columns(
            calendar_day=_calendar_day_from_admission(
                pl.col(TIME_COL),
                pl.col("Admission Time (seconds)"),
            ),
            discharge_day=pl.col("ICU Length of Stay (days)").cast(pl.Int32),
        )
    )

    # Filter to baseline period: day -30 to discharge_day (inclusive)
    baseline_data = labs_with_day.filter(
        pl.col("calendar_day") >= -30,
        pl.col("discharge_day").is_null()
        | (pl.col("calendar_day") <= pl.col("discharge_day")),
    )

    # Extract values from struct fields and compute baselines
    aggregations = []
    for lab_name in lab_columns:
        mode = baseline_mode.get(lab_name, "min")

        # Extract value from struct (assuming struct has "value" field)
        lab_value = (
            pl.when(pl.col(lab_name).is_null())
            .then(pl.lit(None))
            .otherwise(pl.col(lab_name).struct.field("value"))
        )

        if mode == "min":
            aggregations.append(lab_value.min().alias(f"baseline_{lab_name}"))
        elif mode == "max":
            aggregations.append(lab_value.max().alias(f"baseline_{lab_name}"))
        else:
            raise ValueError(f"Invalid baseline_mode for {lab_name}: {mode}")

    # Compute baselines per stay
    return (
        baseline_data.group_by(STAY_COL)
        .agg(aggregations)
        .join(
            stays_with_time.select(STAY_COL, "Admission Time (seconds)"),
            on=STAY_COL,
            how="left",
        )
    )


# endregion helpers


# region cultures
def cultures(
    microbiology: pl.LazyFrame,
    procedures: pl.LazyFrame,
) -> pl.LazyFrame:
    """Return cultures with event times (reported or requested).

    Output columns per row:
    - Global ICU Stay ID
    - culture_time (seconds relative to admission)
    - positive_culture (bool or null)
    - type ("reported" | "requested")
    """

    microbiology = microbiology.lazy()
    procedures = procedures.lazy()

    reported_cultures = (
        microbiology.group_by(STAY_COL, TIME_COL)
        .agg(
            pl.lit(True).alias("culture_taken"),
            pl.col("Organism")
            .is_in(["No living organism identified", "No growth", None])
            .not_()
            .max()
            .alias("positive_culture"),
        )
        .rename({TIME_COL: "culture_time"})
        .with_columns(pl.lit("reported").alias("type"))
    )

    requested_cultures = (
        procedures.filter(
            pl.col("Procedure Description").is_in(
                [
                    "Blood culture",
                    "Microbial culture of cerebrospinal fluid",
                    "Microbial culture of sputum",
                    "Microscopic examination of specimen from operative wound, culture",
                    "Microscopic examination of specimen from skin and other integument, culture and sensitivity",
                    "Stool culture",
                    "Urine culture",
                    "Wound microscopy, culture and sensitivities",
                    "Bacteria identified in Anal by Culture",
                    "Bacteria identified in Catheter tip by Culture",
                    "Bacteria identified in Drain by Aerobe culture",
                    "Bacteria identified in Peritoneal fluid by Culture",
                    "Bacteria identified in Wound by Culture",
                    "Cerebrospinal fluid collection",
                    "Cerebrospinal fluid culture",
                    "Collection of catheter tip as specimen",
                    "Legionella rapid microagglutination test",
                    "Microbial culture of sputum",
                    "Microbial culture",
                    "Nasal culture for bacteria",
                    "Staph aureus and MRSA screening panel - Specimen by Organism specific culture",
                    "Throat culture",
                ]
            )
            | pl.col("Procedure Description").str.starts_with(
                "Infectious Diseases - Cultures / Immuno-Assays - Cultures"
            )
            | pl.col("Procedure Description").str.starts_with(
                "Surgery - Infection - Cultures"
            )
        )
        .group_by(
            STAY_COL,
            "Procedure Start Relative to Admission (seconds)",
        )
        .agg(
            pl.lit(True).alias("culture_requested"),
            pl.lit(None).alias("positive_culture"),
        )
        .rename(
            {"Procedure Start Relative to Admission (seconds)": "culture_time"}
        )
        .with_columns(pl.lit("requested").alias("type"))
    )

    return pl.concat(
        [reported_cultures, requested_cultures], how="diagonal_relaxed"
    )


# endregion cultures


# region antibiotics
def antibiotics(
    medications: pl.LazyFrame, prescriptions: pl.LazyFrame = None
) -> pl.LazyFrame:
    """Filter to antibiotic administrations and annotate with rank and intravenous flag.

    Output columns per row:
    - Global ICU Stay ID
    - Drug Ingredient
    - antibiotic_rank (int)
    - intravenous (0/1)
    - Drug Start Relative to Admission (seconds)
    - Drug End Relative to Admission (seconds)
    """

    medications = medications.lazy()
    antibiotic_ranks = _read_antibiotic_ranks()
    if prescriptions is not None:
        medications = pl.concat([medications, prescriptions], how="vertical")

    abx = (
        medications.filter(
            pl.col("Drug Ingredient").is_in(list(antibiotic_ranks.keys())),
            ~pl.col("Drug Name").is_in(
                [
                    "Chlooramfenicol (Globenicol)",
                    "Dexamethason/gentamicine oogzalf (Dexamytrex)",
                ]
            ),
        )
        .with_columns(
            pl.col("Drug Ingredient")
            .replace(antibiotic_ranks)
            .alias("antibiotic_rank"),
            (pl.col("Drug Administration Route") == "intravenous")
            .cast(pl.Int8)
            .alias("intravenous"),
        )
        .select(
            STAY_COL,
            "Drug Ingredient",
            "antibiotic_rank",
            "intravenous",
            "Drug Start Relative to Admission (seconds)",
            "Drug End Relative to Admission (seconds)",
        )
    )

    # AmsterdamUMCdb: exclude common prophylaxis patterns
    abx = abx.filter(
        ~(
            (pl.col(STAY_COL).str.starts_with("umcdb-"))
            & (pl.col("Drug Ingredient") == "cefotaxime")
            & (
                pl.col("Drug End Relative to Admission (seconds)")
                < (4 * SECONDS_PER_DAY)
            )
        ),
        ~(
            (pl.col(STAY_COL).str.starts_with("umcdb-"))
            & (pl.col("Drug Ingredient") == "vancomycin")
            & (
                pl.col("Drug End Relative to Admission (seconds)")
                < (1 * SECONDS_PER_DAY)
            )
        ),
    )

    return abx


# endregion antibiotics


# region suspected infection
def suspected_infection(
    all_stays_t0: pl.LazyFrame,
    medications: pl.LazyFrame,
    microbiology: pl.LazyFrame,
    procedures: pl.LazyFrame,
    *,
    window_size: int,
    t_0: Optional[int] = None,
    t_1: Optional[int] = None,
) -> pl.LazyFrame:
    """Suspected infection per timeframe (Seymour et al.).

    We identify pairs of cultures and antibiotics within windows:
    - culture first -> antibiotic within 72h
    - antibiotic first -> culture within 24h

    We anchor suspected_time to the first of the pair and map to timeframe.
    """

    all_stays_t0 = all_stays_t0.lazy()
    CULT = cultures(microbiology, procedures)
    ABX = antibiotics(medications)

    # Attach T_0
    cult = CULT.join(all_stays_t0, on=STAY_COL, how="inner")
    abx = ABX.join(all_stays_t0, on=STAY_COL, how="inner")

    # Do not apply timeframe-based bounds on raw seconds columns here; bounds are applied
    # after mapping to timeframe (consistent with SOFA helper semantics).

    # Culture-first: antibiotic within 72h after culture
    # Use asof forward join per stay.
    culture_first = (
        cult.sort(STAY_COL, "culture_time")  # required by asof
        .join_asof(
            abx.sort(
                STAY_COL,
                "Drug Start Relative to Admission (seconds)",
            ),
            left_on="culture_time",
            right_on="Drug Start Relative to Admission (seconds)",
            by=STAY_COL,
            strategy="forward",
            tolerance=72 * SECONDS_PER_HOUR,
        )
        .drop_nulls(
            ["Drug Start Relative to Admission (seconds)", "culture_time"]
        )
        .select(
            STAY_COL,
            pl.col("culture_time").alias("suspected_time"),
            "T_0",
        )
    )

    # Antibiotic-first: culture within 24h after antibiotic
    antibiotic_first = (
        abx.sort(STAY_COL, "Drug Start Relative to Admission (seconds)")
        .join_asof(
            cult.sort(STAY_COL, "culture_time"),
            left_on="Drug Start Relative to Admission (seconds)",
            right_on="culture_time",
            by=STAY_COL,
            strategy="forward",
            tolerance=24 * SECONDS_PER_HOUR,
        )
        .drop_nulls(
            ["Drug Start Relative to Admission (seconds)", "culture_time"]
        )
        .select(
            STAY_COL,
            pl.col("Drug Start Relative to Admission (seconds)").alias(
                "suspected_time"
            ),
            "T_0",
        )
    )

    suspected = pl.concat(
        [culture_first, antibiotic_first], how="diagonal_relaxed"
    )

    # Map to timeframe and mark presence
    return (
        suspected.with_columns(
            timeframe=_assign_timeframe("suspected_time", window_size)
        )
        .filter(pl.col("timeframe") >= 0)
        .filter(
            *_optional_time_bounds_filter("timeframe", window_size, t_0, t_1)
        )
        .group_by(STAY_COL, "timeframe")
        .agg(pl.lit(True).alias("suspected_infection"))
    )


# endregion suspected infection


# region antibiotic escalation
def antibiotic_escalation(
    all_stays_t0: pl.LazyFrame,
    medications: pl.LazyFrame,
    *,
    window_size: int,
    t_0: Optional[int] = None,
    t_1: Optional[int] = None,
) -> pl.LazyFrame:
    """Antibiotic escalation per timeframe (Shah et al.).

    Rules:
    - Consider an antibiotic "on" in timeframe k if drug_start + 12h falls into k.
    - For each timeframe, compute:
        rank_max = max antibiotic_rank
        rank_max_n = number of antibiotics at rank_max
        intravenous_n = count of intravenous administrations
    - Escalation at k if:
        (rank_max_k > rank_max_{k-1}) OR
        (rank_max_k == rank_max_{k-1} AND rank_max_n_k > rank_max_n_{k-1})
      and intravenous_n_k > 0.
    - For the first timeframe with any antibiotics, mark escalation True.
    """

    all_stays_t0 = all_stays_t0.lazy()
    ABX = antibiotics(medications).join(all_stays_t0, on=STAY_COL, how="inner")

    # Attribution timeframe using (start + 12h)
    abx_tf = (
        ABX.with_columns(
            considered_start=(
                pl.col("Drug Start Relative to Admission (seconds)")
                + TWELVE_HOURS
            )
        )
        .with_columns(
            timeframe=_assign_timeframe("considered_start", window_size)
        )
        .filter(pl.col("timeframe") >= 0)
        .filter(
            *_optional_time_bounds_filter("timeframe", window_size, t_0, t_1)
        )
        .group_by(STAY_COL, "timeframe", "antibiotic_rank")
        .agg(
            rank_n=pl.len(),
            intravenous=pl.sum("intravenous"),
        )
    )

    # Collapse to top rank per timeframe
    max_rank = abx_tf.group_by(STAY_COL, "timeframe").agg(
        rank_max=pl.max("antibiotic_rank")
    )
    top = (
        abx_tf.join(max_rank, on=[STAY_COL, "timeframe"], how="inner")
        .filter(pl.col("antibiotic_rank") == pl.col("rank_max"))
        .group_by(STAY_COL, "timeframe")
        .agg(
            antibiotic_rank=pl.max("antibiotic_rank"),
            antibiotic_rank_n=pl.sum("rank_n"),
            intravenous=pl.sum("intravenous"),
        )
        .with_columns(present=pl.lit(True))
    )

    # Escalation via shift over stay ordered by timeframe
    return (
        top.with_columns(
            prev_rank=pl.col("antibiotic_rank")
            .shift(1)
            .over(STAY_COL, order_by="timeframe"),
            prev_rank_n=pl.col("antibiotic_rank_n")
            .shift(1)
            .over(STAY_COL, order_by="timeframe"),
        )
        .with_columns(
            pl.when(pl.col("present") & pl.col("prev_rank").is_null())
            .then(True)
            .otherwise(
                (
                    (pl.col("antibiotic_rank") > pl.col("prev_rank"))
                    | (
                        (pl.col("antibiotic_rank") == pl.col("prev_rank"))
                        & (pl.col("antibiotic_rank_n") > pl.col("prev_rank_n"))
                    )
                )
                & (pl.col("intravenous") > 0)
            )
            .alias("antibiotic_escalation")
        )
        .select(STAY_COL, "timeframe", "antibiotic_escalation")
    )


# endregion antibiotic escalation


# endregion helpers


# region lactate >= 2mmol/L
def lactate_long(
    all_stays_t0: pl.LazyFrame,
    timeseries_labs: pl.LazyFrame,
    *,
    window_size: int,
    t_0: Optional[int] = None,
    t_1: Optional[int] = None,
) -> pl.LazyFrame:
    """Compute per timeframe whether lactate >= 2 mmol/L."""

    all_stays_t0 = all_stays_t0.lazy()
    labs = timeseries_labs.lazy()

    lact = (
        labs.select(STAY_COL, TIME_COL, "Lactate")
        .filter(pl.col("Lactate").struct.field("system").str.contains("Blood"))
        .with_columns(pl.col("Lactate").struct.field("value").alias("Lactate"))
        .drop_nulls("Lactate")
        .join(all_stays_t0, on=STAY_COL, how="inner")
        .with_columns(timeframe=_assign_timeframe(TIME_COL, window_size))
        .filter(pl.col("timeframe") >= 0)
        .filter(
            *_optional_time_bounds_filter("timeframe", window_size, t_0, t_1)
        )
        .group_by(STAY_COL, "timeframe")
        .agg(max_lactate=pl.max("Lactate").cast(pl.Float64))
        .with_columns((pl.col("max_lactate") >= 2.0).alias("lactate_ge2"))
        .select(STAY_COL, "timeframe", "lactate_ge2")
    )
    return lact


# endregion lactate


# region rhee helpers
def rhee_consecutive_antibiotics(
    all_stays_t0: pl.LazyFrame,
    patient_information: pl.LazyFrame,
    medications: pl.LazyFrame,
    *,
    window_size: int,
    t_0: Optional[int] = None,
    t_1: Optional[int] = None,
) -> pl.LazyFrame:
    """Detect timeframes with ≥4 consecutive antibiotic days per Rhee criteria.

    Rhee Rule: New IV antibiotic followed by any systemic antibiotic daily for
    total ≥4 days (or until 1 day prior to discharge/death).

    Returns timeframe of the 4th day when criterion is met.

    Args:
        all_stays_t0: Stays with T_0 reference times
        patient_information: Contains "Admission Time (24h)" and "ICU Length of Stay (days)"
        medications: Medication administrations
        window_size: Window size in seconds
        t_0, t_1: Optional time bounds

    Returns:
        LazyFrame with columns:
        - Global ICU Stay ID
        - timeframe: The timeframe when 4th consecutive day is reached
        - abx_day_4_reached: Boolean marker
    """
    all_stays_t0 = all_stays_t0.lazy()
    patient_information = patient_information.lazy()

    # Attach T_0 and admission time to antibiotics
    abx = (
        antibiotics(medications)
        .join(all_stays_t0, on=STAY_COL, how="inner")
        .join(
            patient_information.select(STAY_COL, "Admission Time (24h)"),
            on=STAY_COL,
            how="left",
        )
        .pipe(_admission_time_to_seconds)
        .filter(
            pl.col("Drug Start Relative to Admission (seconds)")
            >= pl.col("T_0")
        )
        # Calendar day from admission for each antibiotic start
        .with_columns(
            calendar_day=_calendar_day_from_admission(
                pl.col("Drug Start Relative to Admission (seconds)"),
                pl.col("Admission Time (seconds)"),
            )
        )
        .filter(pl.col("calendar_day") >= 0)
    )

    # Daily coverage: IV on day 0, any antibiotic on subsequent days
    daily = (
        abx.group_by(STAY_COL, "calendar_day")
        .agg(
            pl.first("Admission Time (seconds)"),
            iv_count=pl.sum("intravenous"),
            total_count=pl.len(),
            T_0=pl.first("T_0"),
        )
        .with_columns(
            has_coverage=pl.when(pl.col("calendar_day") == 0)
            .then(pl.col("iv_count") > 0)
            .otherwise(pl.col("total_count") > 0)
        )
        .filter(pl.col("has_coverage"))
        .sort(STAY_COL, "calendar_day")
    )

    # Check for 4 consecutive days via shift
    consec = (
        daily.with_columns(
            day_plus_1=pl.col("calendar_day")
            .shift(-1)
            .over(STAY_COL, order_by="calendar_day"),
            day_plus_2=pl.col("calendar_day")
            .shift(-2)
            .over(STAY_COL, order_by="calendar_day"),
            day_plus_3=pl.col("calendar_day")
            .shift(-3)
            .over(STAY_COL, order_by="calendar_day"),
        )
        .with_columns(
            four_consecutive=(
                (pl.col("day_plus_1") == (pl.col("calendar_day") + 1))
                & (pl.col("day_plus_2") == (pl.col("calendar_day") + 2))
                & (pl.col("day_plus_3") == (pl.col("calendar_day") + 3))
            )
        )
        .filter(pl.col("four_consecutive"))
    )

    # First qualifying sequence per stay, map day 4 to timeframe
    result = (
        consec.group_by(STAY_COL)
        .agg(
            pl.col("Admission Time (seconds)").sort_by("calendar_day").first(),
            first_day=pl.min("calendar_day"),
            T_0=pl.first("T_0"),
        )
        .with_columns(
            timeframe=(
                (pl.col("first_day") + 3) * pl.lit(SECONDS_PER_DAY)
                - pl.col("Admission Time (seconds)")
                - pl.col("T_0")
            ).floordiv(window_size),
        )
        .filter(pl.col("timeframe") >= 0)
    )

    if t_0 is not None or t_1 is not None:
        result = result.filter(
            *_optional_time_bounds_filter("timeframe", window_size, t_0, t_1)
        )

    return result.select(
        STAY_COL,
        "timeframe",
        pl.lit(True).alias("abx_day_4_reached"),
    )


def rhee_organ_dysfunction(
    all_stays_t0: pl.LazyFrame,
    patient_information: pl.LazyFrame,
    timeseries_labs: pl.LazyFrame,
    ventilation: pl.LazyFrame,
    cv_ge3_timeline: pl.LazyFrame,
    lactate_timeline: pl.LazyFrame,
    *,
    window_size: int,
    t_0: Optional[int] = None,
    t_1: Optional[int] = None,
) -> pl.LazyFrame:
    """Detect organ dysfunction markers per Rhee criteria.

    Returns timeframe with any of:
    - Vasopressor (cv_ge3)
    - Mechanical ventilation ≥2 continuous calendar days
    - Creatinine increase ≥0.5 from baseline (excluding ESRD)
    - Bilirubin ≥2.0 and 100% increase from baseline
    - Platelet <100 and ≥50% decline from baseline
    - INR >1.5 and ≥0.5 increase from baseline (excluding warfarin)
    - Lactate ≥2.0 mmol/L

    Args:
        all_stays_t0: Stays with T_0
        patient_information: Contains admission time, LOS, diagnoses
        timeseries_labs: Lab timeseries
        ventilation: Ventilation intervals
        cv_ge3_timeline: Pre-computed vasopressor timeline
        lactate_timeline: Pre-computed lactate timeline
        window_size: Window size in seconds
        t_0, t_1: Optional time bounds

    Returns:
        LazyFrame with Global ICU Stay ID, timeframe, and organ_dysfunction flag
    """
    all_stays_t0 = all_stays_t0.lazy()
    patient_info = patient_information.lazy()
    labs = timeseries_labs.lazy()

    # Vasopressor (pre-computed)
    vaso_tf = cv_ge3_timeline.filter(pl.col("cv_ge3")).select(
        STAY_COL, "timeframe", pl.lit(True).alias("dysfunction")
    )

    # Lactate ≥2.0 (pre-computed)
    lact_tf = lactate_timeline.filter(pl.col("lactate_ge2")).select(
        STAY_COL, "timeframe", pl.lit(True).alias("dysfunction")
    )

    # Mechanical ventilation ≥2 continuous calendar days
    vent = (
        ventilation.lazy()
        .join(all_stays_t0, on=STAY_COL, how="inner")
        .join(
            patient_info.select(STAY_COL, "Admission Time (24h)"),
            on=STAY_COL,
            how="left",
        )
        .pipe(_admission_time_to_seconds)
        .filter(
            pl.col("Ventilation Start Relative to Admission (seconds)")
            >= pl.col("T_0")
        )
    )

    vent_tf = (
        vent.with_columns(
            start_day=_calendar_day_from_admission(
                pl.col("Ventilation Start Relative to Admission (seconds)"),
                pl.col("Admission Time (seconds)"),
            ),
            end_day=_calendar_day_from_admission(
                pl.col("Ventilation End Relative to Admission (seconds)"),
                pl.col("Admission Time (seconds)"),
            ),
        )
        .with_columns(
            duration_days=pl.col("end_day").sub(pl.col("start_day")).add(1)
        )
        .filter(pl.col("duration_days") >= 2)
        .with_columns(
            timeframe=_assign_timeframe(
                "Ventilation Start Relative to Admission (seconds)",
                window_size,
            )
        )
        .filter(pl.col("timeframe") >= 0)
        .select(STAY_COL, "timeframe", pl.lit(True).alias("dysfunction"))
    )

    # Lab-based criteria with baselines
    baselines = rhee_compute_lab_baselines(
        all_stays_t0.select(STAY_COL),
        patient_info,
        labs,
        lab_columns=["Creatinine", "Bilirubin", "Platelets", "INR"],
        baseline_mode={
            "Creatinine": "min",
            "Bilirubin": "min",
            "INR": "min",
            "Platelets": "max",
        },
    )

    # Attach T_0 and baselines, extract struct values, filter to analysis window
    LABS = ["Creatinine", "Bilirubin", "Platelets", "INR"]
    labs_base = (
        labs.select(STAY_COL, TIME_COL, *LABS)
        .join(all_stays_t0, on=STAY_COL, how="inner")
        .filter(pl.col(TIME_COL) >= pl.col("T_0"))
        .join(baselines, on=STAY_COL, how="inner")
        .with_columns(
            pl.when(pl.col(col).is_null())
            .then(pl.lit(None))
            .otherwise(pl.col(col).struct.field("value"))
            .alias(col)
            for col in LABS
        )
        .with_columns(timeframe=_assign_timeframe(TIME_COL, window_size))
        .filter(pl.col("timeframe") >= 0)
    )

    # Creatinine: increase ≥0.5 from baseline (TODO: exclude ESRD)
    creat_tf = labs_base.filter(
        pl.col("Creatinine").is_not_null(),
        pl.col("baseline_Creatinine").is_not_null(),
        ((pl.col("Creatinine") - pl.col("baseline_Creatinine")) >= 0.5),
    ).select(STAY_COL, "timeframe", pl.lit(True).alias("dysfunction"))

    # Bilirubin: ≥2.0 AND 100% increase from baseline
    bili_tf = labs_base.filter(
        pl.col("Bilirubin").is_not_null(),
        pl.col("baseline_Bilirubin").is_not_null(),
        pl.col("Bilirubin") >= 2.0,
        (pl.col("Bilirubin") - pl.col("baseline_Bilirubin"))
        >= pl.col("baseline_Bilirubin"),
    ).select(STAY_COL, "timeframe", pl.lit(True).alias("dysfunction"))

    # Platelets: <100 AND ≥50% decline from baseline
    plt_tf = labs_base.filter(
        pl.col("Platelets").is_not_null(),
        pl.col("baseline_Platelets").is_not_null(),
        pl.col("Platelets") < 100,
        (pl.col("baseline_Platelets") - pl.col("Platelets"))
        >= (pl.col("baseline_Platelets") * 0.5),
    ).select(STAY_COL, "timeframe", pl.lit(True).alias("dysfunction"))

    # INR: >1.5 AND ≥0.5 increase from baseline (TODO: exclude warfarin)
    inr_tf = labs_base.filter(
        pl.col("INR").is_not_null(),
        pl.col("baseline_INR").is_not_null(),
        pl.col("INR") > 1.5,
        (pl.col("INR") - pl.col("baseline_INR")) >= 0.5,
    ).select(STAY_COL, "timeframe", pl.lit(True).alias("dysfunction"))

    # Union all markers
    result = pl.concat(
        [vaso_tf, lact_tf, vent_tf, creat_tf, bili_tf, plt_tf, inr_tf],
        how="diagonal_relaxed",
    ).unique()

    if t_0 is not None or t_1 is not None:
        result = result.filter(
            *_optional_time_bounds_filter("timeframe", window_size, t_0, t_1)
        )

    return result.select(
        STAY_COL, "timeframe", pl.lit(True).alias("organ_dysfunction")
    ).unique()


# endregion rhee helpers


# region main sepsis long
def SEPSIS(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_vitals: Optional[pl.LazyFrame] = None,
    timeseries_labs: Optional[pl.LazyFrame] = None,
    timeseries_resp: Optional[pl.LazyFrame] = None,
    timeseries_inout: Optional[pl.LazyFrame] = None,
    medications: Optional[pl.LazyFrame] = None,
    prescriptions: Optional[pl.LazyFrame] = None,
    microbiology: Optional[pl.LazyFrame] = None,
    procedures: Optional[pl.LazyFrame] = None,
    ventilation: Optional[pl.LazyFrame] = None,
    *,
    t_0: Optional[int] = 0,
    t_0_per_stay: Optional[pl.LazyFrame] = None,
    t_1: Optional[int] = None,
    window_size: int = SECONDS_PER_HOUR,
    timeframe_unit: str = "Hours",  # semantics only; output timeframe is numeric
) -> pl.LazyFrame:
    """Compute Sepsis in long format from raw inputs.

    Produces three separate labels per timeframe:
    - SEPSIS: Seymour-anchored on earliest culture+antibiotic pair, requiring an
      acute SOFA increase (>=2) within [-48h, +24h] of the suspicion time and
      classifying SHOCK when cardiovascular_points>=3 and lactate>=2 at onset.
    - SEPSIS_ABX: Shah-anchored on earliest antibiotic escalation (with IV),
      applying the same SOFA delta and shock rules.
    - SEPSIS_RHEE: Rhee-anchored on blood culture + ≥4 consecutive antibiotic days
      + ≥1 organ dysfunction marker within ±2 days of culture.

    Arguments
    ---------
        patient_information: pl.LazyFrame
            Patient/stay-level information; must contain Global ICU Stay ID and any
            fields needed downstream by SOFA (e.g., weight).
        timeseries_vitals: pl.LazyFrame
            Raw vitals timeseries used by SOFA (MAP, GCS, etc.).
        timeseries_labs: pl.LazyFrame
            Raw laboratory results used by SOFA and lactate detection.
        timeseries_resp: pl.LazyFrame
            Raw respiratory timeseries (FiO2, etc.) used by SOFA.
        timeseries_intakeoutput: pl.LazyFrame
            Raw intake/output timeseries used by SOFA (urine output).
        medications: pl.LazyFrame
            Medication administrations; used for antibiotic detection and SOFA meds.
        prescriptions: pl.LazyFrame, optional
            Medication prescriptions; concatenated to medications if provided.
        microbiology: pl.LazyFrame
            Microbiology results for reported cultures.
        procedures: pl.LazyFrame
            Procedures for requested cultures.
        ventilation: pl.LazyFrame
            Mechanical ventilation intervals (optional for SOFA behavior).
        t_0: int, optional
            Scalar reference time (seconds from admission). Defaults to 0. Ignored
            when t_0_per_stay is provided.
        t_1: int, optional
            Optional upper time bound (seconds from admission) for filtering inputs.
        window_size: int, optional
            Timeframe width in seconds (default: 3600). Window index is
            floor((time - T_0)/window_size).
        timeframe_unit: str, optional
            Semantic only; output column remains a numeric timeframe.
        t_0_per_stay: pl.LazyFrame, optional
            Per-stay T_0 overrides with columns [Global ICU Stay ID, T_0].
        lactate_item_names: list[str], optional
            Explicit lactate item names to detect in labs when auto-detection is
            insufficient.

    Returns
    -------
        pl.LazyFrame: one row per (stay, timeframe) where onset occurred in at
        least one mode, with columns
        - Global ICU Stay ID
        - timeframe (0-indexed integer window)
        - SEPSIS      -> "SEPSIS" | "SHOCK" | null
        - SEPSIS_ABX  -> "SEPSIS" | "SHOCK" | null
        - SEPSIS_RHEE -> "SEPSIS" | null
        - T_0 (seconds)
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
    if timeseries_inout is None:
        timeseries_inout = get_timeseries_intakeoutput()
    if medications is None:
        medications = get_medications()
    if microbiology is None:
        microbiology = get_microbiology()
    if procedures is None:
        procedures = get_procedures()
    if ventilation is None:
        ventilation = get_ventilation()
    

    # Validate all required data is available
    required = {
        "patient_information": patient_information,
        "timeseries_vitals": timeseries_vitals,
        "timeseries_labs": timeseries_labs,
        "timeseries_resp": timeseries_resp,
        "timeseries_inout": timeseries_inout,
        "medications": medications,
        "microbiology": microbiology,
        "procedures": procedures,
        "ventilation": ventilation,
    }
    
    missing = [name for name, data in required.items() if data is None]
    if missing:
        raise ValueError(
            f"Cannot compute SEPSIS: Missing required datasets: {', '.join(missing)}. "
            f"Ensure they are configured in ~/.reprodICU/PATHS.yaml or provide them explicitly."
        )

    # Inputs to Lazy
    patient_information = patient_information.lazy()
    medications = medications.lazy()
    if prescriptions is not None:
        prescriptions = prescriptions.lazy()
        medications = pl.concat([medications, prescriptions], how="vertical")

    ALL_STAYS = patient_information.select("Global ICU Stay ID")
    ALL_STAYS_T0 = _build_t0(ALL_STAYS, t_0_per_stay=t_0_per_stay, t_0=t_0)
    los_col = "ICU Length of Stay (days)"
    ALL_STAYS_T0_LOS = ALL_STAYS_T0.join(
        patient_information.select(STAY_COL, los_col),
        on=STAY_COL,
        how="left",
    )

    # region SOFA
    # SOFA from raw inputs (long format)
    SOFA_SCORE = SOFA(
        patient_information,
        timeseries_vitals,
        timeseries_labs,
        timeseries_resp,
        timeseries_inout,
        medications,
        ventilation,
        t_0=t_0,
        t_1=t_1,
        window_size=window_size,
        t_0_per_stay=t_0_per_stay,
        timeframe_name="timeframe",
    )
    # Aggregate per timeframe (worst-within-window)
    sofa_base = (
        SOFA_SCORE.select(
            STAY_COL,
            "timeframe",
            pl.col("SOFA Score").alias("sofa_score"),
            pl.col("Cardiovascular").alias("cardiovascular_points"),
        )
        .collect()
        .lazy()
    )
    cv_ge3_timeline = (
        sofa_base.select(
            STAY_COL,
            "timeframe",
            (pl.col("cardiovascular_points") >= 3).alias("cv_ge3"),
        )
        .with_columns(
            prev_cv=pl.col("cv_ge3")
            .shift(1)
            .over(STAY_COL, order_by="timeframe")
        )
        .with_columns(
            pl.when(
                pl.col("cv_ge3") & pl.col("prev_cv").fill_null(False).not_()
            )
            .then(True)
            .otherwise(False)
            .alias("cv_start"),
        )
        .select(STAY_COL, "timeframe", "cv_ge3", "cv_start")
    )
    lactate_timeline = lactate_long(
        ALL_STAYS_T0, timeseries_labs, window_size=window_size, t_0=t_0, t_1=t_1
    )

    # region Sepsis-3
    # Earliest suspicion per Seymour (earliest culture-antibiotic pair)
    suspicion_by_timeframe = suspected_infection(
        ALL_STAYS_T0,
        medications,
        microbiology,
        procedures,
        window_size=window_size,
        t_0=t_0,
        t_1=t_1,
    )

    suspicion_numbered = (
        suspicion_by_timeframe.filter(pl.col("suspected_infection"))
        # no new suspicion if [-48h, +24h] includes the previous suspicion time
        #   -> timeframe less than 48h after previous suspicion time
        .with_columns(
            pl.col("timeframe")
            .le(
                pl.col("timeframe")
                .shift(1)
                .over(partition_by="Global ICU Stay ID", order_by="timeframe")
                .fill_null(-999)
                .add(48)
            )
            .not_()
            .alias("new_suspicion")
        )
        .with_columns(
            pl.col("new_suspicion")
            .cum_sum()
            .over(partition_by="Global ICU Stay ID", order_by="timeframe")
            .alias("suspicion_number")
        )
        .group_by(STAY_COL, "suspicion_number")
        .agg(suspicion_timeframe=pl.min("timeframe"))
        .join(ALL_STAYS_T0_LOS, on=STAY_COL, how="inner")
        .with_columns(
            max_tf_by_los=(
                (pl.col(los_col) * pl.lit(SECONDS_PER_DAY))
                .sub(pl.col("T_0"))
                .floordiv(window_size)
            )
        )
        # suspicion must fall within ICU stay bounds when LOS is available
        .filter(
            pl.col(los_col).is_null()
            | (pl.col("suspicion_timeframe") <= pl.col("max_tf_by_los"))
        )
        .with_columns(
            baseline_timeframe=pl.max_horizontal(
                pl.lit(0, dtype=pl.Int64),
                pl.col("suspicion_timeframe")
                - pl.lit(
                    int(48 * SECONDS_PER_HOUR // window_size), dtype=pl.Int64
                ),
            ),
            end_timeframe=pl.col("suspicion_timeframe")
            + pl.lit(
                int(24 * SECONDS_PER_HOUR // window_size),
                dtype=pl.Int64,
            ),
        )
    )

    # Baseline SOFA score at the baseline_timeframe
    baseline_scores = (
        sofa_base.join(suspicion_numbered, on=STAY_COL, how="inner")
        .filter(pl.col("timeframe") == pl.col("baseline_timeframe"))
        .group_by(STAY_COL, "suspicion_number")
        .agg(baseline_sofa_score=pl.max("sofa_score"))
    )

    # First timeframe within [baseline_timeframe, end_timeframe] where SOFA >= baseline + 2
    onset_candidates = (
        sofa_base.join(suspicion_numbered, on=STAY_COL, how="inner")
        .join(baseline_scores, on=[STAY_COL, "suspicion_number"], how="inner")
        .filter(
            (pl.col("timeframe") >= pl.col("baseline_timeframe"))
            & (pl.col("timeframe") <= pl.col("end_timeframe"))
        )
        .with_columns(
            sofa_increase_delta=pl.col("sofa_score")
            - pl.col("baseline_sofa_score")
        )
        .filter(pl.col("sofa_increase_delta") >= 2)
        .group_by(STAY_COL, "suspicion_number")
        .agg(onset_timeframe=pl.min("timeframe"))
    )

    # Label SEPSIS/SHOCK at onset timeframe
    cv_at_onset = (
        cv_ge3_timeline.join(onset_candidates, on=STAY_COL, how="inner")
        .filter(pl.col("timeframe") == pl.col("onset_timeframe"))
        .select(
            STAY_COL,
            pl.col("timeframe").alias("onset_timeframe"),
            pl.col("cv_ge3").alias("cv_ge3_onset"),
        )
    )
    lactate_at_onset = (
        lactate_timeline.join(onset_candidates, on=STAY_COL, how="inner")
        .filter(pl.col("timeframe") == pl.col("onset_timeframe"))
        .select(
            STAY_COL,
            pl.col("timeframe").alias("onset_timeframe"),
            pl.col("lactate_ge2").alias("lactate_ge2_onset"),
        )
    )
    onset_labels = (
        onset_candidates.join(
            cv_at_onset,
            on=[STAY_COL, "onset_timeframe"],
            how="left",
        )
        .join(
            lactate_at_onset,
            on=[STAY_COL, "onset_timeframe"],
            how="left",
        )
        .with_columns(
            shock_onset=pl.coalesce(pl.col("cv_ge3_onset"), pl.lit(False))
            & pl.coalesce(pl.col("lactate_ge2_onset"), pl.lit(False))
        )
        .with_columns(
            SEPSIS=pl.when(pl.col("shock_onset"))
            .then(pl.lit("SHOCK"))
            .otherwise(pl.lit("SEPSIS"))
        )
        .select(
            STAY_COL,
            pl.col("onset_timeframe").alias("timeframe"),
            "SEPSIS",
        )
    )

    # region Shah
    # Antibiotic-anchored Sepsis (SEPSIS_ABX per Shah et al.)
    abx_escalation_tf = antibiotic_escalation(
        ALL_STAYS_T0,
        medications,
        window_size=window_size,
        t_0=t_0,
        t_1=t_1,
    )

    # Earliest timeframe with antibiotic escalation
    abx_suspicion_numbered = (
        abx_escalation_tf.filter(pl.col("antibiotic_escalation"))
        # no new suspicion if [-48h, +24h] includes the previous suspicion time
        #   -> timeframe less than 72h after previous suspicion time
        .with_columns(
            pl.col("timeframe")
            .le(
                pl.col("timeframe")
                .shift(1)
                .over(partition_by="Global ICU Stay ID", order_by="timeframe")
                .fill_null(-999)
                .add(72)
            )
            .not_()
            .alias("new_suspicion")
        )
        .with_columns(
            pl.col("new_suspicion")
            .cum_sum()
            .over(partition_by="Global ICU Stay ID", order_by="timeframe")
            .alias("suspicion_number")
        )
        .group_by(STAY_COL, "suspicion_number")
        .agg(abx_suspicion_timeframe=pl.min("timeframe"))
        .join(ALL_STAYS_T0_LOS, on=STAY_COL, how="inner")
        .with_columns(
            max_tf_by_los=(
                (pl.col(los_col) * pl.lit(SECONDS_PER_DAY))
                .sub(pl.col("T_0"))
                .floordiv(window_size)
                .cast(pl.Int64)
            )
        )
        .filter(
            pl.col(los_col).is_null()
            | (pl.col("abx_suspicion_timeframe") <= pl.col("max_tf_by_los"))
        )
    )

    win_back = int(48 * SECONDS_PER_HOUR // window_size)
    win_fwd = int(24 * SECONDS_PER_HOUR // window_size)

    # Baseline and end windows relative to antibiotic-only suspicion
    abx_suspicion_numbered = abx_suspicion_numbered.with_columns(
        baseline_timeframe_abx=pl.max_horizontal(
            pl.lit(0, dtype=pl.Int64),
            pl.col("abx_suspicion_timeframe")
            - pl.lit(win_back, dtype=pl.Int64),
        ),
        end_timeframe_abx=pl.col("abx_suspicion_timeframe")
        + pl.lit(win_fwd, dtype=pl.Int64),
    )

    # Baseline SOFA at antibiotic-anchored baseline timeframe
    baseline_scores_abx = (
        sofa_base.join(abx_suspicion_numbered, on=STAY_COL, how="inner")
        .filter(pl.col("timeframe") == pl.col("baseline_timeframe_abx"))
        .group_by(STAY_COL, "suspicion_number")
        .agg(baseline_sofa_score_abx=pl.max("sofa_score"))
    )

    # First timeframe in [baseline, end] where SOFA >= baseline + 2
    onset_candidates_abx = (
        sofa_base.join(abx_suspicion_numbered, on=STAY_COL, how="inner")
        .join(
            baseline_scores_abx, on=[STAY_COL, "suspicion_number"], how="inner"
        )
        .filter(
            (pl.col("timeframe") >= pl.col("baseline_timeframe_abx"))
            & (pl.col("timeframe") <= pl.col("end_timeframe_abx"))
        )
        .with_columns(
            sofa_increase_delta_abx=pl.col("sofa_score")
            - pl.col("baseline_sofa_score_abx")
        )
        .filter(pl.col("sofa_increase_delta_abx") >= 2)
        .group_by(STAY_COL, "suspicion_number")
        .agg(onset_timeframe_abx=pl.min("timeframe"))
    )

    # Determine SEPSIS_ABX vs SHOCK using same shock criteria
    cv_at_onset_abx = (
        cv_ge3_timeline.join(onset_candidates_abx, on=STAY_COL, how="inner")
        .filter(pl.col("timeframe") == pl.col("onset_timeframe_abx"))
        .select(
            STAY_COL,
            pl.col("timeframe").alias("onset_timeframe_abx"),
            pl.col("cv_ge3").alias("cv_ge3_onset_abx"),
        )
    )
    lactate_at_onset_abx = (
        lactate_timeline.join(onset_candidates_abx, on=STAY_COL, how="inner")
        .filter(pl.col("timeframe") == pl.col("onset_timeframe_abx"))
        .select(
            STAY_COL,
            pl.col("timeframe").alias("onset_timeframe_abx"),
            pl.col("lactate_ge2").alias("lactate_ge2_onset_abx"),
        )
    )
    onset_labels_abx = (
        onset_candidates_abx.join(
            cv_at_onset_abx,
            on=[STAY_COL, "onset_timeframe_abx"],
            how="left",
        )
        .join(
            lactate_at_onset_abx,
            on=[STAY_COL, "onset_timeframe_abx"],
            how="left",
        )
        .with_columns(
            shock_onset_abx=pl.coalesce(
                [pl.col("cv_ge3_onset_abx"), pl.lit(False)]
            )
            & pl.coalesce([pl.col("lactate_ge2_onset_abx"), pl.lit(False)])
        )
        .with_columns(
            SEPSIS_ABX=pl.when(pl.col("shock_onset_abx"))
            .then(pl.lit("SHOCK"))
            .otherwise(pl.lit("SEPSIS"))
        )
        .select(
            STAY_COL,
            pl.col("onset_timeframe_abx").alias("timeframe"),
            "SEPSIS_ABX",
        )
    )

    # region SHOCK
    # Septic shock (independent per anchor)
    vaso_starts = (
        cv_ge3_timeline.join(suspicion_numbered, on=STAY_COL, how="inner")
        .filter(
            (pl.col("timeframe") >= pl.col("baseline_timeframe"))
            & (pl.col("timeframe") <= pl.col("end_timeframe"))
            & pl.col("cv_start")
        )
        .group_by(STAY_COL, "suspicion_number")
        .agg(vasopressor_start_timeframe=pl.min("timeframe"))
    )
    lactate_first = (
        lactate_timeline.filter(pl.col("lactate_ge2"))
        .join(suspicion_numbered, on=STAY_COL, how="inner")
        .filter(
            (pl.col("timeframe") >= pl.col("baseline_timeframe"))
            & (pl.col("timeframe") <= pl.col("end_timeframe"))
        )
        .group_by(STAY_COL, "suspicion_number")
        .agg(lactate_timeframe=pl.min("timeframe"))
    )
    septic_shock = (
        suspicion_numbered.join(vaso_starts, on=STAY_COL, how="inner")
        .join(lactate_first, on=[STAY_COL, "suspicion_number"], how="inner")
        .with_columns(
            shock_timeframe=pl.max_horizontal(
                pl.col("vasopressor_start_timeframe"),
                pl.col("lactate_timeframe"),
            )
        )
        .select(
            STAY_COL,
            pl.col("shock_timeframe").alias("timeframe"),
            pl.lit("SHOCK").alias("SEPSIS_shock"),
        )
    )

    vaso_starts_abx = (
        cv_ge3_timeline.join(abx_suspicion_numbered, on=STAY_COL, how="inner")
        .filter(
            (pl.col("timeframe") >= pl.col("baseline_timeframe_abx"))
            & (pl.col("timeframe") <= pl.col("end_timeframe_abx"))
            & pl.col("cv_start")
        )
        .group_by(STAY_COL, "suspicion_number")
        .agg(vasopressor_start_timeframe_abx=pl.min("timeframe"))
    )
    lactate_first_abx = (
        lactate_timeline.filter(pl.col("lactate_ge2"))
        .join(abx_suspicion_numbered, on=STAY_COL, how="inner")
        .filter(
            (pl.col("timeframe") >= pl.col("baseline_timeframe_abx"))
            & (pl.col("timeframe") <= pl.col("end_timeframe_abx"))
        )
        .group_by(STAY_COL, "suspicion_number")
        .agg(lactate_timeframe_abx=pl.min("timeframe"))
    )
    septic_shock_abx = (
        abx_suspicion_numbered.join(vaso_starts_abx, on=STAY_COL, how="inner")
        .join(lactate_first_abx, on=[STAY_COL, "suspicion_number"], how="inner")
        .with_columns(
            shock_timeframe_abx=pl.max_horizontal(
                pl.col("vasopressor_start_timeframe_abx"),
                pl.col("lactate_timeframe_abx"),
            )
        )
        .select(
            STAY_COL,
            pl.col("shock_timeframe_abx").alias("timeframe"),
            pl.lit("SHOCK").alias("SEPSIS_ABX_shock"),
        )
    )

    # region Rhee
    # EHR Clinical Surveillance Definition (Rhee et al.)
    # Blood culture + ≥4 consecutive antibiotic days + organ dysfunction within ±2 calendar days

    # Get blood culture timeframes with admission time for calendar day calculation
    blood_cultures_with_time = (
        cultures(microbiology, procedures)
        .join(ALL_STAYS_T0, on=STAY_COL, how="inner")
        .join(
            patient_information.select(STAY_COL, "Admission Time (24h)"),
            on=STAY_COL,
            how="left",
        )
        .pipe(_admission_time_to_seconds)
        .with_columns(
            timeframe=_assign_timeframe("culture_time", window_size),
            culture_calendar_day=_calendar_day_from_admission(
                pl.col("culture_time"),
                pl.col("Admission Time (seconds)"),
            ),
        )
        .filter(
            pl.col("timeframe") >= 0,
            *_optional_time_bounds_filter("timeframe", window_size, t_0, t_1),
        )
        .select(
            STAY_COL,
            "timeframe",
            "culture_calendar_day",
            "Admission Time (seconds)",
            "T_0",
        )
    )

    # Get antibiotics that meet 4-day consecutive criterion
    rhee_abx_tf = rhee_consecutive_antibiotics(
        ALL_STAYS_T0,
        patient_information,
        medications,
        window_size=window_size,
        t_0=t_0,
        t_1=t_1,
    )

    # Get organ dysfunction markers
    rhee_organ_dysfunction_tf = rhee_organ_dysfunction(
        ALL_STAYS_T0,
        patient_information,
        timeseries_labs,
        ventilation,
        cv_ge3_timeline,
        lactate_timeline,
        window_size=window_size,
        t_0=t_0,
        t_1=t_1,
    )

    # Calculate calendar days for antibiotics and organ dysfunction
    rhee_abx_with_day = (
        rhee_abx_tf.join(
            patient_information.select(STAY_COL, "Admission Time (24h)"),
            on=STAY_COL,
            how="left",
        )
        .join(ALL_STAYS_T0.select(STAY_COL, "T_0"), on=STAY_COL, how="inner")
        .pipe(_admission_time_to_seconds)
        .with_columns(
            # Convert timeframe back to seconds, then to calendar day
            abx_calendar_day=_calendar_day_from_admission(
                (pl.col("timeframe") * window_size) + pl.col("T_0"),
                pl.col("Admission Time (seconds)"),
            )
        )
        .select(STAY_COL, "timeframe", "abx_calendar_day")
    )

    rhee_organ_with_day = (
        rhee_organ_dysfunction_tf.join(
            patient_information.select(STAY_COL, "Admission Time (24h)"),
            on=STAY_COL,
            how="left",
        )
        .join(ALL_STAYS_T0.select(STAY_COL, "T_0"), on=STAY_COL, how="inner")
        .pipe(_admission_time_to_seconds)
        .with_columns(
            organ_calendar_day=_calendar_day_from_admission(
                (pl.col("timeframe") * window_size) + pl.col("T_0"),
                pl.col("Admission Time (seconds)"),
            )
        )
        .select(STAY_COL, "timeframe", "organ_calendar_day")
    )

    # Join all three components and check ±2 calendar day constraint
    onset_labels_rhee = (
        blood_cultures_with_time.join(
            rhee_abx_with_day, on=STAY_COL, how="inner", suffix="_abx"
        )
        .join(rhee_organ_with_day, on=STAY_COL, how="inner", suffix="_organ")
        .filter(
            # Antibiotics within ±2 calendar days of culture
            (pl.col("abx_calendar_day") - pl.col("culture_calendar_day"))
            .abs()
            .le(2),
            # Organ dysfunction within ±2 calendar days of culture
            (pl.col("organ_calendar_day") - pl.col("culture_calendar_day"))
            .abs()
            .le(2),
        )
        .group_by(STAY_COL)
        # Use culture timeframe as onset
        .agg(onset_timeframe_rhee=pl.min("timeframe"))
        .select(
            STAY_COL,
            pl.col("onset_timeframe_rhee").alias("timeframe"),
            pl.lit("SEPSIS").alias("SEPSIS_RHEE"),
        )
    )
    # endregion Rhee

    tf_union = pl.concat(
        [
            onset_labels.select(STAY_COL, "timeframe"),
            onset_labels_abx.select(STAY_COL, "timeframe"),
            septic_shock.select(STAY_COL, "timeframe"),
            septic_shock_abx.select(STAY_COL, "timeframe"),
            onset_labels_rhee.select(STAY_COL, "timeframe"),
        ],
        how="diagonal_relaxed",
    ).unique()

    base = ALL_STAYS_T0.join(tf_union, on=STAY_COL, how="inner")

    return (
        base.join(onset_labels, on=[STAY_COL, "timeframe"], how="left")
        .join(onset_labels_abx, on=[STAY_COL, "timeframe"], how="left")
        .join(septic_shock, on=[STAY_COL, "timeframe"], how="left")
        .join(septic_shock_abx, on=[STAY_COL, "timeframe"], how="left")
        .join(onset_labels_rhee, on=[STAY_COL, "timeframe"], how="left")
        .select(
            STAY_COL,
            "timeframe",
            pl.coalesce("SEPSIS_shock", "SEPSIS").alias("SEPSIS"),
            pl.coalesce("SEPSIS_ABX_shock", "SEPSIS_ABX").alias("SEPSIS_ABX"),
            "SEPSIS_RHEE",
            "T_0",
        )
        .filter(
            *_optional_time_bounds_filter("timeframe", window_size, t_0, t_1)
        )
        .sort(STAY_COL, "timeframe")  # stable order
        .unique()
    )


# endregion main sepsis long


__all__ = [
    "SEPSIS",
    "cultures",
    "antibiotics",
    "antibiotic_escalation",
    "suspected_infection",
    "lactate_long",
    "rhee_consecutive_antibiotics",
    "rhee_organ_dysfunction",
    "rhee_compute_lab_baselines",
]
