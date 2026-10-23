"""
SEPSIS: compute Sepsis-3 in long format directly from raw inputs.

Sepsis is defined as:
- suspected infection (culture+antibiotic pair / antibiotic escalation) AND
- organ dysfunction (SOFA ≥2 / SOFA increase ≥2 from baseline / SIRS criteria / Rhee organ dysfunction criteria)

Septic shock is defined as:
- sepsis AND
- hypotension requiring vasopressors to maintain MAP ≥65 mmHg AND
- serum lactate >2 mmol/L despite adequate fluid resuscitation

To operationalize this definition, hypotension requiring vasopressor support is
approximated by vasopressor infusion (SOFA CV ≥2). Adequate fluid resuscitation
is assumed for ICU patients.

Output columns per row:
- Global ICU Stay ID
- timeframe (integer window index counted from T_0)
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
  JAMA. 2016 Feb 23;315(8):801-10.
  doi: 10.1001/jama.2016.0287. PMID: 26903338; PMCID: PMC4968574.
- Seymour CW, Liu VX, Iwashyna TJ, Brunkhorst FM, Rea TD, Scherag A, Rubenfeld G, Kahn JM, Shankar-Hari M, Singer M, Deutschman CS, Escobar GJ, Angus DC.
  Assessment of Clinical Criteria for Sepsis: For the Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3).
  JAMA. 2016 Feb 23;315(8):762-74.
  doi: 10.1001/jama.2016.0288. PMID: 26903335; PMCID: PMC5433435.
- Shah AD, MacCallum NS, Harris S, Brealey DA, Palmer E, Hetherington J, Shi S, Perez-Suarez D, Ercole A, Watkinson PJ, Jones A, Ashworth S, Beale R, Brett SJ, Singer M.
  Descriptors of Sepsis Using the Sepsis-3 Criteria: A Cohort Study in Critical Care Units Within the U.K. National Institute for Health Research Critical Care Health Informatics Collaborative.
  Crit Care Med. 2021 Nov 1;49(11):1883-1894.
  doi: 10.1097/CCM.0000000000005169. PMID: 34259454; PMCID: PMC8508729.
- Rhee C, Kadri S, Huang SS, Murphy MV, Li L, Platt R, Klompas M.
  Objective Sepsis Surveillance Using Electronic Clinical Data.
  Infect Control Hosp Epidemiol. 2016 Feb;37(2):163-71.
  doi: 10.1017/ice.2015.264. Epub 2015 Nov 3. PMID: 26526737; PMCID: PMC4743875.
"""

from pathlib import Path
from typing import Optional

import polars as pl
import yaml

from ..base import (
    SECONDS_IN_1D,
    SECONDS_IN_1H,
    STAY_KEY,
    TIME_KEY,
    Window,
    _assign_timeframe,
    _window_context,
    with_windows,
)
from ..common import (
    _validate_required_data,
    get_medications,
    get_microbiology,
    get_patient_information,
    get_prescriptions,
    get_procedures,
    get_timeseries_intakeoutput,
    get_timeseries_labs,
    get_timeseries_respiratory,
    get_timeseries_vitals,
    get_ventilation,
)
from ..scores.SOFA import SOFA
from .SIRS import SIRS

# seconds constants
TWELVE_HOURS = 12 * SECONDS_IN_1H

# strict time column name used across helpers


# region helpers
def _read_antibiotic_ranks() -> dict[str, int]:
    ranks_path = Path(__file__).parent / "ANTIBIOTIC_RANKS.yaml"
    with open(ranks_path, "r") as f:
        data = yaml.safe_load(f) or {}
    # Expect mapping { ingredient (str): rank (int) }
    return {str(k): int(v) for k, v in data.items()}


def _admission_time_to_seconds(data: pl.LazyFrame) -> pl.LazyFrame:
    """Extract seconds since midnight from a pl.Time column.

    Returns
    -------
        Expression yielding seconds since midnight (0-86399)

    Example: 14:30:00 -> 14*3600 + 30*60 = 52200 seconds
    """
    return data.with_columns(
        (pl.col("Admission Time (24h)") - pl.time(0, 0, 0))
        .dt.total_seconds()
        .fill_null(0)  # Default to midnight if missing
        .cast(pl.Int64)
        .alias("Admission Time (seconds)")
    )


def _calendar_day_from_admission(
    time_seconds: pl.Expr, admission_seconds_into_day: pl.Expr
) -> pl.Expr:
    """Compute calendar day number from admission given admission time.

    Calendar day 0 is the day of admission (from midnight to next midnight).
    Takes into account the admission time within that day.

    Arguments
    ---------
        time_seconds : pl.Expr
            Time relative to admission in seconds
        admission_seconds_into_day : pl.Expr
            Seconds from midnight to admission time (0-86399)

    Returns
    -------
        Calendar day index (0 = admission day, 1 = next day, -1 = day before, etc.)

    Examples
    --------
        Admission at 14:00 (50400 seconds into day)
        - time=-3600 (1h before admission = 13:00) -> day 0 (same calendar day)
        - time=-39600 (11h before admission = 03:00) -> day -1 (previous calendar day)
        - time=36000 (10h after admission = 00:00 next day) -> day 1
    """
    # Seconds from the start of admission day
    seconds_from_day_start = time_seconds + admission_seconds_into_day
    # Integer division by SECONDS_IN_1D gives calendar day
    return seconds_from_day_start.floordiv(SECONDS_IN_1D).cast(int)


def rhee_compute_lab_baselines(
    all_stays: pl.LazyFrame,
    patient_information: pl.LazyFrame,
    timeseries_labs: pl.LazyFrame,
    lab_columns: list[str],
    baseline_mode: dict[str, str],  # {lab_name: "min" | "max"}
) -> pl.LazyFrame:
    """Compute baseline lab values from day -30 to end of stay per Rhee criteria.

    Arguments
    ---------
        all_stays : pl.LazyFrame
            DataFrame with Global ICU Stay ID
        patient_information : pl.LazyFrame
            Contains admission time and LOS
        timeseries_labs : pl.LazyFrame
            Time series lab data
        lab_columns : list[str]
            List of lab column names to compute baselines for
        baseline_mode : dict[str, str]
            Dict mapping lab name to aggregation mode ("min" or "max")

    Returns
    -------
        LazyFrame with columns:
        - Global ICU Stay ID
        - baseline_{lab_name} for each lab in lab_columns
        - admission_seconds_into_day (for downstream calendar day calculations)
    """

    # Get admission time and LOS
    stays_with_time = all_stays.join(
        patient_information.select(
            STAY_KEY, "Admission Time (24h)", "ICU Length of Stay (days)"
        ),
        on=STAY_KEY,
        how="left",
    ).pipe(_admission_time_to_seconds)

    # Define baseline period: day -30 to day of discharge
    # Need to calculate calendar day for each lab measurement
    labs_with_day = (
        timeseries_labs.select(STAY_KEY, TIME_KEY, *lab_columns)
        .join(stays_with_time, on=STAY_KEY, how="inner")
        .with_columns(
            calendar_day=_calendar_day_from_admission(
                pl.col(TIME_KEY),
                pl.col("Admission Time (seconds)"),
            ),
            discharge_day=pl.col("ICU Length of Stay (days)").cast(int),
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
    return baseline_data.group_by(STAY_KEY).agg(
        *aggregations,
        pl.first("Admission Time (seconds)"),
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

    reported_cultures = (
        microbiology.group_by(STAY_KEY, TIME_KEY)
        .agg(
            pl.lit(True).alias("culture_taken"),
            pl.col("Organism")
            .is_in(["No living organism identified", "No growth", None])
            .not_()
            .max()
            .alias("positive_culture"),
        )
        .rename({TIME_KEY: "culture_time"})
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
            STAY_KEY,
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
    ).sort(STAY_KEY, "culture_time")


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

    antibiotic_ranks = _read_antibiotic_ranks()
    if prescriptions is not None:
        medications = pl.concat([medications, prescriptions], how="vertical")

    return (
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
            .replace_strict(antibiotic_ranks, return_dtype=pl.Int64)
            .alias("antibiotic_rank"),
            # unknown route (e.g. SICdb, NWICU) is assumed intravenous
            (pl.col("Drug Administration Route") == "intravenous")
            .fill_null(True)
            .cast(int)
            .alias("intravenous"),
        )
        .select(
            STAY_KEY,
            "Drug Ingredient",
            "antibiotic_rank",
            "intravenous",
            "Drug Start Relative to Admission (seconds)",
            "Drug End Relative to Admission (seconds)",
        )
        # AmsterdamUMCdb: exclude common prophylaxis patterns
        .filter(
            ~(
                (pl.col(STAY_KEY).str.starts_with("umcdb-"))
                & (pl.col("Drug Ingredient") == "cefotaxime")
                & (
                    pl.col("Drug End Relative to Admission (seconds)")
                    < (4 * SECONDS_IN_1D)
                )
            ),
            ~(
                (pl.col(STAY_KEY).str.starts_with("umcdb-"))
                & (pl.col("Drug Ingredient") == "vancomycin")
                & (
                    pl.col("Drug End Relative to Admission (seconds)")
                    < (1 * SECONDS_IN_1D)
                )
            ),
        )
        .sort(STAY_KEY, "Drug Start Relative to Admission (seconds)")
    )


# endregion antibiotics


# region suspected infection
def suspected_infection(
    all_stays_t0: pl.LazyFrame,
    medications: pl.LazyFrame,
    microbiology: pl.LazyFrame,
    procedures: pl.LazyFrame,
    window_size: int,
) -> pl.LazyFrame:
    """Suspected infection per timeframe (Seymour et al.).

    We identify pairs of cultures and antibiotics within windows:
    - culture first -> antibiotic within 72h
    - antibiotic first -> culture within 24h

    We anchor suspected_time to the first of the pair and map to timeframe.
    """

    # re-sort after the join: join_asof needs sorted input, and a join can reorder rows
    CULT = (
        cultures(microbiology, procedures)
        .join(all_stays_t0, on=STAY_KEY, how="inner")
        .sort(STAY_KEY, "culture_time")
    )  # fmt: skip
    ABX = (
        antibiotics(medications)
        .join(all_stays_t0, on=STAY_KEY, how="inner")
        .sort(STAY_KEY, "Drug Start Relative to Admission (seconds)")
    )

    # Do not apply timeframe-based bounds on raw seconds columns here; bounds are applied
    # after mapping to timeframe (consistent with SOFA helper semantics).

    # Culture-first: antibiotic within 72h after culture
    # Use asof forward join per stay.
    culture_first = (
        CULT.join_asof(
            ABX,
            left_on="culture_time",
            right_on="Drug Start Relative to Admission (seconds)",
            by=STAY_KEY,
            strategy="forward",
            tolerance=72 * SECONDS_IN_1H,
        )
        .filter(
            pl.col("Drug Start Relative to Admission (seconds)").is_not_null(),
            pl.col("culture_time").is_not_null(),
        )
        .select(
            STAY_KEY,
            pl.col("culture_time").alias("suspected_time"),
            "T_0",
        )
    )

    # Antibiotic-first: culture within 24h after antibiotic
    antibiotic_first = (
        ABX.join_asof(
            CULT,
            left_on="Drug Start Relative to Admission (seconds)",
            right_on="culture_time",
            by=STAY_KEY,
            strategy="forward",
            tolerance=24 * SECONDS_IN_1H,
        )
        .filter(
            pl.col("Drug Start Relative to Admission (seconds)").is_not_null(),
            pl.col("culture_time").is_not_null(),
        )
        .select(
            STAY_KEY,
            pl.col("Drug Start Relative to Admission (seconds)").alias("suspected_time"),
            "T_0",
        )
    ) # fmt: skip

    # Map to timeframe and mark presence
    return (
        pl.concat([culture_first, antibiotic_first], how="diagonal_relaxed")
        .with_columns(
            timeframe=_assign_timeframe("suspected_time", window_size)
        )
        .group_by(STAY_KEY, "timeframe")
        .agg(pl.lit(True).alias("suspected_infection"))
        .sort(STAY_KEY, "timeframe")
    )


# endregion suspected infection


# region antibiotic escalation
def antibiotic_escalation(
    all_stays_t0: pl.LazyFrame,
    medications: pl.LazyFrame,
    window_size: int,
) -> pl.LazyFrame:
    """Antibiotic escalation per timeframe (Shah et al.).

    Rules:
    - An antibiotic is "on" in every timeframe from its start to its end + 12h
      (point doses without a usable end: start + 12h + 12h).
    - For each timeframe, compute:
        antibiotic_rank = max antibiotic_rank
        antibiotic_rank_n = number of distinct antibiotics at antibiotic_rank
        intravenous = count of intravenous administrations
    - Escalation at k if intravenous_k > 0 and
        (nothing was on at k-1 OR
         antibiotic_rank_k > antibiotic_rank_{k-1} OR
         (antibiotic_rank_k == antibiotic_rank_{k-1} AND
          antibiotic_rank_n_k > antibiotic_rank_n_{k-1})).
    """
    start = "Drug Start Relative to Admission (seconds)"
    end = "Drug End Relative to Admission (seconds)"

    return (
        antibiotics(medications)
        .join(all_stays_t0, on=STAY_KEY, how="inner")
        # Attribution timeframes: from start to (end + 12h)
        .with_columns(
            considered_end=pl.when(pl.col(end) > pl.col(start))
            .then(pl.col(end))
            .otherwise(pl.col(start) + TWELVE_HOURS)
            + TWELVE_HOURS
        )
        .with_columns(
            timeframe=pl.int_ranges(
                _assign_timeframe(start, window_size),
                _assign_timeframe("considered_end", window_size) + 1,
            )
        )
        .explode("timeframe")
        .group_by(STAY_KEY, "timeframe", "antibiotic_rank")
        .agg(
            rank_n=pl.n_unique("Drug Ingredient"),
            intravenous=pl.sum("intravenous"),
        )
        # Collapse to top rank per timeframe
        .with_columns(
            rank_max=pl.max("antibiotic_rank").over(STAY_KEY, "timeframe")
        )
        .filter(pl.col("antibiotic_rank") == pl.col("rank_max"))
        .group_by(STAY_KEY, "timeframe")
        .agg(
            antibiotic_rank=pl.max("antibiotic_rank"),
            antibiotic_rank_n=pl.sum("rank_n"),
            intravenous=pl.sum("intravenous"),
        )
        # Escalation via shift over stay ordered by timeframe
        .with_columns(
            prev_rank=pl.col("antibiotic_rank")
            .shift(1)
            .over(STAY_KEY, order_by="timeframe"),
            prev_rank_n=pl.col("antibiotic_rank_n")
            .shift(1)
            .over(STAY_KEY, order_by="timeframe"),
            prev_timeframe=pl.col("timeframe")
            .shift(1)
            .over(STAY_KEY, order_by="timeframe"),
        )
        .with_columns(
            (
                (
                    pl.col("prev_rank").is_null()
                    | (pl.col("timeframe") - pl.col("prev_timeframe") > 1)
                    | (pl.col("antibiotic_rank") > pl.col("prev_rank"))
                    | (
                        (pl.col("antibiotic_rank") == pl.col("prev_rank"))
                        & (pl.col("antibiotic_rank_n") > pl.col("prev_rank_n"))
                    )
                )
                & (pl.col("intravenous") > 0)
            ).alias("antibiotic_escalation")
        )
        .select(STAY_KEY, "timeframe", "antibiotic_escalation")
        .sort(STAY_KEY, "timeframe")
    )


# endregion antibiotic escalation


# endregion helpers


# region lactate > 2mmol/L
def lactate_long(
    all_stays_t0: pl.LazyFrame,
    timeseries_labs: pl.LazyFrame,
    window_size: int,
) -> pl.LazyFrame:
    """Compute per timeframe whether lactate > 2 mmol/L (Sepsis-3 septic shock)."""
    return (
        timeseries_labs.select(STAY_KEY, TIME_KEY, "Lactate")
        .filter(pl.col("Lactate").struct.field("system").str.contains("Blood"))
        .with_columns(pl.col("Lactate").struct.field("value").alias("Lactate"))
        .drop_nulls("Lactate")
        .join(all_stays_t0, on=STAY_KEY, how="inner")
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window_size))
        .group_by(STAY_KEY, "timeframe")
        .agg(max_lactate=pl.max("Lactate").cast(pl.Float64))
        .with_columns((pl.col("max_lactate") > 2.0).alias("lactate_gt2"))
        .select(STAY_KEY, "timeframe", "lactate_gt2")
    )


# endregion lactate


# region rhee helpers
def rhee_consecutive_antibiotics(
    all_stays_t0: pl.LazyFrame,
    patient_information: pl.LazyFrame,
    medications: pl.LazyFrame,
    window_size: int,
) -> pl.LazyFrame:
    """Detect timeframes with ≥4 consecutive antibiotic days per Rhee criteria.

    Rhee Rule: New IV antibiotic followed by any systemic antibiotic daily for
    total ≥4 days (or until 1 day prior to discharge/death).

    Returns timeframe of the 4th day when criterion is met.

    Arguments
    ---------
        all_stays_t0 : pl.LazyFrame
            Stays with T_0 reference times
        patient_information : pl.LazyFrame
            Contains "Admission Time (24h)" and "ICU Length of Stay (days)"
        medications : pl.LazyFrame
            Medication administrations
        window_size : int
            Window size in seconds

    Returns
    -------
        LazyFrame with columns:
        - Global ICU Stay ID
        - timeframe: The timeframe when 4th consecutive day is reached
        - abx_first_day: Calendar day on which the 4-day course starts
        - abx_day_4_reached: Boolean marker
    """

    return (
        antibiotics(medications)
        .join(all_stays_t0, on=STAY_KEY, how="inner")
        .join(
            patient_information.select(STAY_KEY, "Admission Time (24h)"),
            on=STAY_KEY,
            how="left",
        )
        .pipe(_admission_time_to_seconds)
        .filter(
            pl.col("Drug Start Relative to Admission (seconds)")
            >= 0
        )
        # Calendar days from admission covered by each antibiotic administration
        .with_columns(
            calendar_day=pl.int_ranges(
                _calendar_day_from_admission(
                    pl.col("Drug Start Relative to Admission (seconds)"),
                    pl.col("Admission Time (seconds)"),
                ),
                _calendar_day_from_admission(
                    pl.max_horizontal(
                        pl.col("Drug Start Relative to Admission (seconds)"),
                        pl.col("Drug End Relative to Admission (seconds)"),
                    ),
                    pl.col("Admission Time (seconds)"),
                )
                + 1,
            )
        )
        .explode("calendar_day")
        .filter(pl.col("calendar_day") >= 0)
        # Daily coverage: IV on day 0, any antibiotic on subsequent days
        .group_by(STAY_KEY, "calendar_day")
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
        .sort(STAY_KEY, "calendar_day")
        # Check for 4 consecutive days via shift
        .with_columns(
            day_plus_1=pl.col("calendar_day")
            .shift(-1)
            .over(STAY_KEY, order_by="calendar_day"),
            day_plus_2=pl.col("calendar_day")
            .shift(-2)
            .over(STAY_KEY, order_by="calendar_day"),
            day_plus_3=pl.col("calendar_day")
            .shift(-3)
            .over(STAY_KEY, order_by="calendar_day"),
        )
        .with_columns(
            four_consecutive=(
                (pl.col("day_plus_1") == (pl.col("calendar_day") + 1))
                & (pl.col("day_plus_2") == (pl.col("calendar_day") + 2))
                & (pl.col("day_plus_3") == (pl.col("calendar_day") + 3))
            )
        )
        .filter(pl.col("four_consecutive"))
        # First qualifying sequence per stay, map day 4 to timeframe
        .group_by(STAY_KEY)
        .agg(
            pl.col("Admission Time (seconds)").sort_by("calendar_day").first(),
            first_day=pl.min("calendar_day"),
            T_0=pl.first("T_0"),
        )
        .with_columns(
            timeframe=(
                (pl.col("first_day") + 3) * pl.lit(SECONDS_IN_1D)
                - pl.col("Admission Time (seconds)")
                - pl.col("T_0")
            ).floordiv(window_size),
        )
        .select(
            STAY_KEY,
            "timeframe",
            pl.col("first_day").alias("abx_first_day"),
            pl.lit(True).alias("abx_day_4_reached"),
        )
    )


def rhee_organ_dysfunction(
    all_stays_t0: pl.LazyFrame,
    patient_information: pl.LazyFrame,
    timeseries_labs: pl.LazyFrame,
    ventilation: pl.LazyFrame,
    cv_ge2_timeline: pl.LazyFrame,
    lactate_timeline: pl.LazyFrame,
    *,
    window_size: int,
) -> pl.LazyFrame:
    """Detect organ dysfunction markers per Rhee criteria.

    Returns timeframe with any of:
    - Vasopressor (cv_ge2) - vasopressor infusion is assumed to indicate hypotension requiring support
    - Mechanical ventilation ≥2 continuous calendar days
    - Creatinine increase ≥0.5 from baseline (excluding ESRD)
    - Bilirubin ≥2.0 and 100% increase from baseline
    - Platelet <100 and ≥50% decline from baseline
    - INR >1.5 and ≥0.5 increase from baseline (excluding warfarin)
    - Lactate ≥2.0 mmol/L

    Arguments
    ---------
        all_stays_t0 : pl.LazyFrame
            Stays with T_0
        patient_information : pl.LazyFrame
            Contains admission time, LOS, diagnoses
        timeseries_labs : pl.LazyFrame
            Lab timeseries
        ventilation : pl.LazyFrame
            Ventilation intervals
        cv_ge2_timeline : pl.LazyFrame
            Pre-computed vasopressor timeline
        lactate_timeline : pl.LazyFrame
            Pre-computed lactate timeline
        window_size : int
            Window size in seconds

    Returns
    -------
        LazyFrame with Global ICU Stay ID, timeframe, and organ_dysfunction flag
    """

    # Vasopressor (pre-computed)
    vaso_tf = cv_ge2_timeline.filter(pl.col("cv_ge2")).select(
        STAY_KEY, "timeframe", pl.lit(True).alias("dysfunction")
    )

    # Lactate >2.0 (pre-computed)
    lact_tf = lactate_timeline.filter(pl.col("lactate_gt2")).select(
        STAY_KEY, "timeframe", pl.lit(True).alias("dysfunction")
    )

    # Mechanical ventilation ≥2 continuous calendar days
    vent = (
        ventilation.join(all_stays_t0, on=STAY_KEY, how="inner")
        .join(
            patient_information.select(STAY_KEY, "Admission Time (24h)"),
            on=STAY_KEY,
            how="left",
        )
        .pipe(_admission_time_to_seconds)
        .filter(
            pl.col("Ventilation Start Relative to Admission (seconds)")
            >= 0
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
        .select(STAY_KEY, "timeframe", pl.lit(True).alias("dysfunction"))
    )

    # Lab-based criteria with baselines
    baselines = rhee_compute_lab_baselines(
        all_stays_t0.select(STAY_KEY),
        patient_information,
        timeseries_labs,
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
        timeseries_labs.select(STAY_KEY, TIME_KEY, *LABS)
        .join(all_stays_t0, on=STAY_KEY, how="inner")
        .filter(pl.col(TIME_KEY) >= 0)
        .join(baselines, on=STAY_KEY, how="inner")
        .with_columns(
            pl.col(col).struct.field("value").alias(col) for col in LABS
        )
        .with_columns(timeframe=_assign_timeframe(TIME_KEY, window_size))
    )

    # Creatinine: increase ≥0.5 from baseline (TODO: exclude ESRD)
    creat_tf = labs_base.filter(
        pl.col("Creatinine").is_not_null(),
        pl.col("baseline_Creatinine").is_not_null(),
        ((pl.col("Creatinine") - pl.col("baseline_Creatinine")) >= 0.5),
    ).select(STAY_KEY, "timeframe", pl.lit(True).alias("dysfunction"))

    # Bilirubin: ≥2.0 AND 100% increase from baseline
    bili_tf = labs_base.filter(
        pl.col("Bilirubin").is_not_null(),
        pl.col("baseline_Bilirubin").is_not_null(),
        pl.col("Bilirubin") >= 2.0,
        (pl.col("Bilirubin") - pl.col("baseline_Bilirubin"))
        >= pl.col("baseline_Bilirubin"),
    ).select(STAY_KEY, "timeframe", pl.lit(True).alias("dysfunction"))

    # Platelets: <100 AND ≥50% decline from baseline
    plt_tf = labs_base.filter(
        pl.col("Platelets").is_not_null(),
        pl.col("baseline_Platelets").is_not_null(),
        pl.col("Platelets") < 100,
        (pl.col("baseline_Platelets") - pl.col("Platelets"))
        >= (pl.col("baseline_Platelets") * 0.5),
    ).select(STAY_KEY, "timeframe", pl.lit(True).alias("dysfunction"))

    # INR: >1.5 AND ≥0.5 increase from baseline (TODO: exclude warfarin)
    inr_tf = labs_base.filter(
        pl.col("INR").is_not_null(),
        pl.col("baseline_INR").is_not_null(),
        pl.col("INR") > 1.5,
        (pl.col("INR") - pl.col("baseline_INR")) >= 0.5,
    ).select(STAY_KEY, "timeframe", pl.lit(True).alias("dysfunction"))

    # Union all markers
    return (
        pl.concat(
            [vaso_tf, lact_tf, vent_tf, creat_tf, bili_tf, plt_tf, inr_tf],
            how="diagonal_relaxed",
        )
        .select(STAY_KEY, "timeframe", pl.lit(True).alias("organ_dysfunction"))
        .unique()
    )


# endregion rhee helpers


# region main sepsis long
@with_windows(window_size=SECONDS_IN_1H)
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
    organ_failure_criterion: str = "delta",
    baseline_source: str = "icu",
    anchors: set[str] | None = None,
    window: Window,
) -> pl.LazyFrame:
    """Compute Sepsis-3 labels in long format from raw inputs.

    Produces separate labels per timeframe:
    - SEPSIS: Seymour-anchored on earliest culture+antibiotic pair, requiring an
      acute SOFA increase (≥2) within [-48h, +24h] of suspicion time.
    - SEPSIS_ABX: Shah-anchored on earliest antibiotic escalation (with IV),
      applying same SOFA delta and shock rules.
    - SEPSIS_RHEE: Rhee-anchored on blood culture + ≥4 consecutive antibiotic days
      + ≥1 organ dysfunction marker within ±2 days of culture.

    Computation is selective: only requested anchors are computed, and intermediate
    tables remain lazy until final collection. This prevents wasted work on unused labels.

    All data parameters are optional and will be automatically loaded from the
    package datasets if not provided. This makes it convenient for quick analysis
    while maintaining flexibility for custom data.

    Arguments
    ---------
        patient_information : pl.LazyFrame, optional
            Patient information dataset. Loaded automatically if None.
        timeseries_vitals : pl.LazyFrame, optional
            Timeseries vitals data. Loaded automatically if None.
        timeseries_labs : pl.LazyFrame, optional
            Timeseries labs data. Loaded automatically if None.
        timeseries_resp : pl.LazyFrame, optional
            Timeseries respiratory data. Loaded automatically if None.
        timeseries_inout : pl.LazyFrame, optional
            Timeseries intake/output data. Loaded automatically if None.
        medications : pl.LazyFrame, optional
            Medications data. Loaded automatically if None.
        prescriptions : pl.LazyFrame, optional
            Medication prescriptions; concatenated to medications. Loaded automatically if None.
        microbiology : pl.LazyFrame, optional
            Microbiology results. Loaded automatically if None.
        procedures : pl.LazyFrame, optional
            Procedures data. Loaded automatically if None.
        ventilation : pl.LazyFrame, optional
            Ventilation data. Loaded automatically if None.
        organ_failure_criterion : str, optional
            "absolute" applies SOFA criteria to raw scores; "delta" (default) applies
            to change from baseline. Baseline is the lowest score in the 24h before
            the window [-48h, +24h] around suspicion time (Seymour) or antibiotic
            escalation (Shah), or 0 if the window starts at the beginning of the stay.
            "sirs" applies SIRS criteria instead of SOFA.
        baseline_source : str, optional
            Where the "delta" baseline SOFA may come from. "icu" (default) uses only
            the ICU stay and counts values measured before ICU admission as missing
            (as mimic-code). "pre_icu" also takes the lowest SOFA before ICU
            admission (back to t_minus1) as baseline for the whole stay, so chronic
            dysfunction is not counted as new. Only used with organ_failure_criterion="delta".
        anchors : set[str], optional
            Which onset anchors to compute: {"seymour", "shah", "rhee"}.
            Defaults to all three. Specify a subset to skip unnecessary computation.
            Example: anchors={"seymour"} skips Shah and Rhee logic, and skips SOFA
            computation if organ_failure_criterion="sirs".

    Returns
    -------
        pl.LazyFrame
            One row per (stay, timeframe) with columns:
            - Global ICU Stay ID
            - T_0 (seconds)
            - timeframe (integer window index counted from T_0)
            - SEPSIS (Seymour onset): "SEPSIS" | "SHOCK" | null
            - SEPSIS_ABX (Shah onset): "SEPSIS" | "SHOCK" | null
            - SEPSIS_RHEE (Rhee onset): "SEPSIS" | null
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
    if prescriptions is None:
        prescriptions = get_prescriptions()
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
    _validate_required_data("SEPSIS", required)

    if organ_failure_criterion not in ("absolute", "delta", "sirs"):
        raise ValueError(f"organ_failure_criterion must be 'absolute', 'delta', or 'sirs', got {organ_failure_criterion!r}") # fmt: skip

    if baseline_source not in ("icu", "pre_icu"):
        raise ValueError(f"baseline_source must be 'icu' or 'pre_icu', got {baseline_source!r}") # fmt: skip

    if anchors is None:
        anchors = {"seymour", "shah", "rhee"}
    invalid = anchors - {"seymour", "shah", "rhee"}
    if invalid:
        raise ValueError(f"anchors must be subset of {{'seymour', 'shah', 'rhee'}}, got invalid: {invalid}")  # fmt: skip

    compute_seymour = "seymour" in anchors
    compute_shah = "shah" in anchors
    compute_rhee = "rhee" in anchors

    if prescriptions is not None:
        medications = pl.concat([medications, prescriptions], how="vertical")

    ctx = _window_context(patient_information, window)
    los_col = "ICU Length of Stay (days)"
    ALL_STAYS_T0_LOS = ctx.all_stays_t0.join(
        patient_information.select(STAY_KEY, los_col),
        on=STAY_KEY,
        how="left",
    )

    # region SOFA
    ORGAN_FAILURE_CRITERION = {
        "absolute": pl.col("sofa_score") >= 2,
        "delta": pl.col("sofa_increase_delta") >= 2,
        "sirs": pl.col("sirs_criteria_fulfilled"),
    }[organ_failure_criterion]

    sofa_base = SOFA(
        patient_information,
        timeseries_vitals,
        timeseries_labs,
        timeseries_resp,
        timeseries_inout,
        medications,
        ventilation,
        t_0=window.t_0,
        t_1=window.t_1,
        t_minus1=0,  # dense from ICU admission; the onset logic rolls over rows
        _fill_from_pre_icu=False,
        window_size=window.window_size,
        t_0_per_stay=window.t_0_per_stay,
        timeframe_name="timeframe",
    ).select(
        STAY_KEY,
        "timeframe",
        pl.col("SOFA Score").alias("sofa_score"),
        pl.col("Cardiovascular").alias("cardiovascular_points"),
    ).collect().lazy()  # materialize once: referenced many times below

    sirs_base = SIRS(
        patient_information,
        timeseries_vitals,
        timeseries_labs,
        t_0=window.t_0,
        t_1=window.t_1,
        t_minus1=0,  # dense from ICU admission; the onset logic rolls over rows
        window_size=window.window_size,
        t_0_per_stay=window.t_0_per_stay,
        timeframe_name="timeframe",
    ).select(
        STAY_KEY,
        "timeframe",
        pl.col("SIRS Criteria fulfilled").alias("sirs_criteria_fulfilled"),
    ).collect().lazy()  # materialize once: referenced many times below

    lactate_timeline = lactate_long(
        ctx.all_stays_t0, timeseries_labs, window_size=window.window_size
    ).collect().lazy()  # materialize once: referenced many times below

    cv_ge2_timeline = (
        sofa_base.select(
            STAY_KEY,
            "timeframe",
            (pl.col("cardiovascular_points") >= 2).alias("cv_ge2"),
        )
        .with_columns(
            prev_cv=pl.col("cv_ge2")
            .shift(1)
            .over(STAY_KEY, order_by="timeframe")
            .fill_null(False),
        )
        .select(
            STAY_KEY,
            "timeframe",
            "cv_ge2",
            pl.when(pl.col("cv_ge2") & ~pl.col("prev_cv"))
            .then(True)
            .otherwise(False)
            .alias("cv_start"),
        )
    ).collect().lazy()  # materialize once: referenced many times below

    # region Sepsis-3
    # Earliest suspicion per Seymour (earliest culture-antibiotic pair)
    suspicion_numbered = (
        suspected_infection(
            ctx.all_stays_t0,
            medications,
            microbiology,
            procedures,
            window_size=window.window_size,
        )
        .filter(pl.col("suspected_infection"))
        # no new suspicion if [-48h, +24h] includes the previous suspicion time
        #   -> timeframe less than 48h after previous suspicion time
        .with_columns(
            pl.struct(
                pl.col(STAY_KEY),
                pl.col("timeframe").le(
                    pl.col("timeframe")
                    .shift(1)
                    .over(
                        partition_by=STAY_KEY,
                        order_by="timeframe",
                    )
                    .fill_null(-999)
                    .add(48)
                ),
            )
            .rle_id()
            .alias("suspicion_number")
        )
        .group_by(STAY_KEY, "suspicion_number")
        .agg(suspicion_timeframe=pl.min("timeframe"))
        .join(ALL_STAYS_T0_LOS, on=STAY_KEY, how="inner")
        .with_columns(
            max_tf_by_los=(
                (pl.col(los_col) * pl.lit(SECONDS_IN_1D))
                .sub(pl.col("T_0"))
                .floordiv(window.window_size)
            )
        )
        # suspicion must fall within ICU stay bounds when LOS is available
        .filter(
            pl.col(los_col).is_null()
            | (pl.col("suspicion_timeframe") <= pl.col("max_tf_by_los"))
        )
        .with_columns(
            baseline_timeframe=pl.max_horizontal(
                pl.lit(0),
                pl.col("suspicion_timeframe")
                - pl.lit(int((48 * SECONDS_IN_1H) // window.window_size)),
            ),
            end_timeframe=pl.col("suspicion_timeframe")
            + pl.lit(int((24 * SECONDS_IN_1H) // window.window_size)),
        )
    )

    # region Shah
    # Antibiotic-anchored Sepsis (SEPSIS_ABX per Shah et al.)
    abx_suspicion_numbered = (
        antibiotic_escalation(
            ctx.all_stays_t0,
            medications,
            window_size=window.window_size,
        )
        .sort(STAY_KEY, "timeframe")
        .filter(pl.col("antibiotic_escalation"))
        # no new suspicion if [-48h, +24h] includes the previous suspicion time
        #   -> timeframe less than 72h after previous suspicion time
        .with_columns(
            pl.struct(
                pl.col(STAY_KEY),
                pl.col("timeframe").le(
                    pl.col("timeframe")
                    .shift(1)
                    .over(
                        partition_by=STAY_KEY,
                        order_by="timeframe",
                    )
                    .fill_null(-999)
                    .add(72)
                ),
            )
            .rle_id()
            .alias("suspicion_number")
        )
        .group_by(STAY_KEY, "suspicion_number")
        .agg(abx_suspicion_timeframe=pl.min("timeframe"))
        .join(ALL_STAYS_T0_LOS, on=STAY_KEY, how="inner")
        .with_columns(
            max_tf_by_los=(
                (pl.col(los_col) * pl.lit(SECONDS_IN_1D))
                .sub(pl.col("T_0"))
                .floordiv(window.window_size)
                .cast(int)
            )
        )
        .filter(
            pl.col(los_col).is_null()
            | (pl.col("abx_suspicion_timeframe") <= pl.col("max_tf_by_los"))
        )
    )

    sirs_true = sirs_base.filter(pl.col("sirs_criteria_fulfilled")).select(
        STAY_KEY, "timeframe"
    )
    onset_candidates_sirs = (
        suspicion_numbered.sort(STAY_KEY, "baseline_timeframe")
        .join_asof(
            sirs_true.sort(STAY_KEY, "timeframe"),
            left_on="baseline_timeframe",
            right_on="timeframe",
            by=STAY_KEY,
            strategy="forward",
        )
        .filter(pl.col("timeframe") <= pl.col("end_timeframe"))
        .select(STAY_KEY, "suspicion_number", "timeframe")
        .group_by(STAY_KEY, "suspicion_number")
        .agg(onset_timeframe=pl.min("timeframe"))
    )

    # Baseline SOFA: lowest score in the 24h before a timeframe (null at stay start -> 0)
    sofa_pre_24h_min = (
        pl.col("sofa_score")
        .rolling_min(window_size=max(1, SECONDS_IN_1D // window.window_size), min_samples=1)
        .shift(1)
        .over(STAY_KEY, order_by="timeframe")
    )
    if baseline_source == "pre_icu":
        # ... and the lowest SOFA before ICU admission
        sofa_pre_icu_min = (
            SOFA(
                patient_information,
                timeseries_vitals,
                timeseries_labs,
                timeseries_resp,
                timeseries_inout,
                medications,
                ventilation,
                t_0=window.t_0,
                t_minus1=window.t_minus1,
                t_1=0,
                window_size=window.window_size,
                t_0_per_stay=window.t_0_per_stay,
                timeframe_name="timeframe",
            )
            .join(ctx.all_stays_t0, on=STAY_KEY)
            .filter(pl.col("timeframe") < (-pl.col("T_0")).floordiv(window.window_size))
            .group_by(STAY_KEY)
            .agg(pl.col("SOFA Score").min().alias("pre_icu_sofa_min"))
        )
        sofa_base = sofa_base.join(sofa_pre_icu_min, on=STAY_KEY, how="left")
        sofa_pre_24h_min = pl.min_horizontal(
            sofa_pre_24h_min, pl.col("pre_icu_sofa_min")
        )
    sofa_pre_24h_min = sofa_pre_24h_min.alias("baseline_sofa_score")

    # Seymour: SOFA-based onset (bounded explode to avoid stay-only join blowup)
    # Explode to bounded candidate range (end_timeframe - baseline_timeframe, typically ~72-97 hours at 1h window.window_size)
    # instead of joining full stay × all timeframes
    candidates = (
        suspicion_numbered.join(
            sofa_base.select(STAY_KEY, "timeframe", sofa_pre_24h_min),
            left_on=[STAY_KEY, "baseline_timeframe"],
            right_on=[STAY_KEY, "timeframe"],
            how="left",
        )
        .with_columns(
            candidate_timeframe=pl.int_ranges(
                pl.col("baseline_timeframe"), pl.col("end_timeframe") + 1
            )
        )
        .explode("candidate_timeframe")
        .join(
            sofa_base.select(STAY_KEY, "timeframe", "sofa_score"),
            left_on=[STAY_KEY, "candidate_timeframe"],
            right_on=[STAY_KEY, "timeframe"],
            how="left",
        )
        .with_columns(
            sofa_increase_delta=pl.col("sofa_score")
            - pl.col("baseline_sofa_score").fill_null(0)
        )
        .filter(ORGAN_FAILURE_CRITERION)
    )

    onset_candidates_sofa = candidates.group_by(
        STAY_KEY, "suspicion_number"
    ).agg(onset_timeframe=pl.min("candidate_timeframe"))

    # Select based on criterion
    onset_candidates = (
        onset_candidates_sirs
        if organ_failure_criterion == "sirs"
        else onset_candidates_sofa
    )

    # Label SEPSIS/SHOCK at onset timeframe
    cv_at_onset = cv_ge2_timeline.select(
        STAY_KEY,
        "timeframe",
        pl.col("cv_ge2").alias("cv_ge2_onset"),
    )
    lactate_at_onset = lactate_timeline.select(
        STAY_KEY,
        "timeframe",
        pl.col("lactate_gt2").alias("lactate_gt2_onset"),
    )
    onset_labels = (
        onset_candidates.join(
            cv_at_onset,
            left_on=[STAY_KEY, "onset_timeframe"],
            right_on=[STAY_KEY, "timeframe"],
            how="left",
        )
        .join(
            lactate_at_onset,
            left_on=[STAY_KEY, "onset_timeframe"],
            right_on=[STAY_KEY, "timeframe"],
            how="left",
        )
        .with_columns(
            (
                pl.col("cv_ge2_onset").fill_null(False)
                & pl.col("lactate_gt2_onset").fill_null(False)
            ).alias("shock_onset")
        )
        .select(
            STAY_KEY,
            pl.col("onset_timeframe").alias("timeframe"),
            pl.when(pl.col("shock_onset"))
            .then(pl.lit("SHOCK"))
            .otherwise(pl.lit("SEPSIS"))
            .alias("SEPSIS"),
        )
    )

    win_back = int((48 * SECONDS_IN_1H) // window.window_size)
    win_fwd = int((24 * SECONDS_IN_1H) // window.window_size)

    # Baseline and end windows relative to antibiotic-only suspicion
    abx_suspicion_numbered = abx_suspicion_numbered.with_columns(
        baseline_timeframe_abx=pl.max_horizontal(
            pl.lit(0),
            pl.col("abx_suspicion_timeframe") - pl.lit(win_back),
        ),
        end_timeframe_abx=pl.col("abx_suspicion_timeframe") + pl.lit(win_fwd),
    )

    # Shah: SIRS-based onset (join_asof to avoid stay-only join blowup)
    onset_candidates_abx_sirs = (
        abx_suspicion_numbered.sort(STAY_KEY, "baseline_timeframe_abx")
        .join_asof(
            sirs_true.sort(STAY_KEY, "timeframe"),
            left_on="baseline_timeframe_abx",
            right_on="timeframe",
            by=STAY_KEY,
            strategy="forward",
        )
        .filter(pl.col("timeframe") <= pl.col("end_timeframe_abx"))
        .select(STAY_KEY, "suspicion_number", "timeframe")
        .group_by(STAY_KEY, "suspicion_number")
        .agg(onset_timeframe_abx=pl.min("timeframe"))
    )

    # Shah: SOFA-based onset (bounded explode to avoid stay-only join blowup)
    # Explode to bounded candidate range
    candidates_abx = (
        abx_suspicion_numbered.join(
            sofa_base.select(STAY_KEY, "timeframe", sofa_pre_24h_min),
            left_on=[STAY_KEY, "baseline_timeframe_abx"],
            right_on=[STAY_KEY, "timeframe"],
            how="left",
        )
        .with_columns(
            candidate_timeframe=pl.int_ranges(
                pl.col("baseline_timeframe_abx"),
                pl.col("end_timeframe_abx") + 1,
            )
        )
        .explode("candidate_timeframe")
        .join(
            sofa_base.select(STAY_KEY, "timeframe", "sofa_score"),
            left_on=[STAY_KEY, "candidate_timeframe"],
            right_on=[STAY_KEY, "timeframe"],
            how="left",
        )
        .with_columns(
            sofa_increase_delta=pl.col("sofa_score")
            - pl.col("baseline_sofa_score").fill_null(0)
        )
        .filter(ORGAN_FAILURE_CRITERION)
    )

    onset_candidates_abx_sofa = candidates_abx.group_by(
        STAY_KEY, "suspicion_number"
    ).agg(onset_timeframe_abx=pl.min("candidate_timeframe"))

    # Select based on criterion
    onset_candidates_abx = (
        onset_candidates_abx_sirs
        if organ_failure_criterion == "sirs"
        else onset_candidates_abx_sofa
    )

    # Determine SEPSIS_ABX vs SHOCK using same shock criteria
    cv_at_onset_abx = cv_ge2_timeline.select(
        STAY_KEY,
        "timeframe",
        pl.col("cv_ge2").alias("cv_ge2_onset_abx"),
    )
    lactate_at_onset_abx = lactate_timeline.select(
        STAY_KEY,
        "timeframe",
        pl.col("lactate_gt2").alias("lactate_gt2_onset_abx"),
    )
    onset_labels_abx = (
        onset_candidates_abx.join(
            cv_at_onset_abx,
            left_on=[STAY_KEY, "onset_timeframe_abx"],
            right_on=[STAY_KEY, "timeframe"],
            how="left",
        )
        .join(
            lactate_at_onset_abx,
            left_on=[STAY_KEY, "onset_timeframe_abx"],
            right_on=[STAY_KEY, "timeframe"],
            how="left",
        )
        .with_columns(
            shock_onset_abx=(
                pl.col("cv_ge2_onset_abx").fill_null(False)
                & pl.col("lactate_gt2_onset_abx").fill_null(False)
            )
        )
        .select(
            STAY_KEY,
            pl.col("onset_timeframe_abx").alias("timeframe"),
            pl.when(pl.col("shock_onset_abx"))
            .then(pl.lit("SHOCK"))
            .otherwise(pl.lit("SEPSIS"))
            .alias("SEPSIS_ABX"),
        )
    )

    # region SHOCK
    # Septic shock (independent per anchor)
    suspicion_windows = suspicion_numbered.select(
        STAY_KEY,
        "suspicion_number",
        "baseline_timeframe",
        "end_timeframe",
    )

    # Seymour shock: find first vasopressor start in [baseline, end] (join_asof to avoid blowup)
    cv_starts = cv_ge2_timeline.filter(pl.col("cv_start")).select(
        STAY_KEY, "timeframe"
    )
    vaso_starts = (
        suspicion_windows.sort(STAY_KEY, "baseline_timeframe")
        .join_asof(
            cv_starts.sort(STAY_KEY, "timeframe"),
            left_on="baseline_timeframe",
            right_on="timeframe",
            by=STAY_KEY,
            strategy="forward",
        )
        .filter(pl.col("timeframe") <= pl.col("end_timeframe"))
        .select(STAY_KEY, "suspicion_number", "timeframe")
        .group_by(STAY_KEY, "suspicion_number")
        .agg(vasopressor_start_timeframe=pl.min("timeframe"))
    )

    # Seymour shock: find first lactate >=2 in [baseline, end] (join_asof to avoid blowup)
    lactate_high = lactate_timeline.filter(pl.col("lactate_gt2")).select(STAY_KEY, "timeframe")
    lactate_first = (
        suspicion_windows.sort(STAY_KEY, "baseline_timeframe")
        .join_asof(
            lactate_high.sort(STAY_KEY, "timeframe"),
            left_on="baseline_timeframe",
            right_on="timeframe",
            by=STAY_KEY,
            strategy="forward",
        )
        .filter(pl.col("timeframe") <= pl.col("end_timeframe"))
        .select(STAY_KEY, "suspicion_number", "timeframe")
        .group_by(STAY_KEY, "suspicion_number")
        .agg(lactate_timeframe=pl.min("timeframe"))
    )
    septic_shock = (
        suspicion_windows.join(
            vaso_starts, on=[STAY_KEY, "suspicion_number"], how="inner"
        )
        .join(lactate_first, on=[STAY_KEY, "suspicion_number"], how="inner")
        .with_columns(
            shock_timeframe=pl.max_horizontal(
                pl.col("vasopressor_start_timeframe"),
                pl.col("lactate_timeframe"),
            )
        )
        .select(
            STAY_KEY,
            pl.col("shock_timeframe").alias("timeframe"),
            pl.lit("SHOCK").alias("SEPSIS_shock"),
        )
    )

    abx_suspicion_windows = abx_suspicion_numbered.select(
        STAY_KEY,
        "suspicion_number",
        "baseline_timeframe_abx",
        "end_timeframe_abx",
    )

    # Shah shock: find first vasopressor start in [baseline, end] (join_asof to avoid blowup)
    vaso_starts_abx = (
        abx_suspicion_windows.sort(STAY_KEY, "baseline_timeframe_abx")
        .join_asof(
            cv_starts.sort(STAY_KEY, "timeframe"),
            left_on="baseline_timeframe_abx",
            right_on="timeframe",
            by=STAY_KEY,
            strategy="forward",
        )
        .filter(pl.col("timeframe") <= pl.col("end_timeframe_abx"))
        .select(STAY_KEY, "suspicion_number", "timeframe")
        .group_by(STAY_KEY, "suspicion_number")
        .agg(vasopressor_start_timeframe_abx=pl.min("timeframe"))
    )

    # Shah shock: find first lactate >=2 in [baseline, end] (join_asof to avoid blowup)
    lactate_first_abx = (
        abx_suspicion_windows.sort(STAY_KEY, "baseline_timeframe_abx")
        .join_asof(
            lactate_high.sort(STAY_KEY, "timeframe"),
            left_on="baseline_timeframe_abx",
            right_on="timeframe",
            by=STAY_KEY,
            strategy="forward",
        )
        .filter(pl.col("timeframe") <= pl.col("end_timeframe_abx"))
        .select(STAY_KEY, "suspicion_number", "timeframe")
        .group_by(STAY_KEY, "suspicion_number")
        .agg(lactate_timeframe_abx=pl.min("timeframe"))
    )
    septic_shock_abx = (
        abx_suspicion_windows.join(
            vaso_starts_abx, on=[STAY_KEY, "suspicion_number"], how="inner"
        )
        .join(lactate_first_abx, on=[STAY_KEY, "suspicion_number"], how="inner")
        .with_columns(
            shock_timeframe_abx=pl.max_horizontal(
                pl.col("vasopressor_start_timeframe_abx"),
                pl.col("lactate_timeframe_abx"),
            )
        )
        .select(
            STAY_KEY,
            pl.col("shock_timeframe_abx").alias("timeframe"),
            pl.lit("SHOCK").alias("SEPSIS_ABX_shock"),
        )
    )

    # region Rhee
    # EHR Clinical Surveillance Definition (Rhee et al.)
    # Blood culture + ≥4 consecutive antibiotic days + organ dysfunction within ±2 calendar days
    stay_time_ref = (
        ctx.all_stays_t0.join(
            patient_information.select(STAY_KEY, "Admission Time (24h)"),
            on=STAY_KEY,
            how="left",
        )
        .pipe(_admission_time_to_seconds)
        .select(STAY_KEY, "T_0", "Admission Time (seconds)")
    )

    # Get blood culture timeframes with admission time for calendar day calculation
    blood_cultures_with_time = (
        cultures(microbiology, procedures)
        .join(stay_time_ref, on=STAY_KEY, how="inner")
        .with_columns(
            timeframe=_assign_timeframe("culture_time", window.window_size),
            culture_calendar_day=_calendar_day_from_admission(
                pl.col("culture_time"),
                pl.col("Admission Time (seconds)"),
            ),
        )
        .select(
            STAY_KEY,
            "timeframe",
            "culture_calendar_day",
            "Admission Time (seconds)",
            "T_0",
        )
    )

    # Get antibiotics that meet 4-day consecutive criterion
    rhee_abx_tf = rhee_consecutive_antibiotics(
        ctx.all_stays_t0,
        patient_information,
        medications,
        window_size=window.window_size,
    )

    # Get organ dysfunction markers
    rhee_organ_dysfunction_tf = rhee_organ_dysfunction(
        ctx.all_stays_t0,
        patient_information,
        timeseries_labs,
        ventilation,
        cv_ge2_timeline,
        lactate_timeline,
        window_size=window.window_size,
    )

    # Calculate calendar days for organ dysfunction
    rhee_organ_with_day = (
        rhee_organ_dysfunction_tf.join(stay_time_ref, on=STAY_KEY, how="inner")
        .with_columns(
            organ_calendar_day=_calendar_day_from_admission(
                (pl.col("timeframe") * window.window_size) + pl.col("T_0"),
                pl.col("Admission Time (seconds)"),
            )
        )
        .select(STAY_KEY, "timeframe", "organ_calendar_day")
    )

    # Join all three components and check ±2 calendar day constraint
    cult_days = (
        blood_cultures_with_time.select(
            STAY_KEY,
            "timeframe",
            "culture_calendar_day",
        )
        .unique()
        .sort(STAY_KEY, "culture_calendar_day")
    )
    abx_days = (
        # compared with the culture: the first day of the course
        rhee_abx_tf.select(STAY_KEY, pl.col("abx_first_day").alias("abx_calendar_day"))
        .unique()
        .sort(STAY_KEY, "abx_calendar_day")
    )
    organ_days = (
        rhee_organ_with_day.select(STAY_KEY, "organ_calendar_day")
        .unique()
        .sort(STAY_KEY, "organ_calendar_day")
    )

    # Existence checks within +/- 2 calendar days using nearest asof matches.
    onset_labels_rhee = (
        cult_days.sort(STAY_KEY, "culture_calendar_day")
        .join_asof(
            abx_days,
            left_on="culture_calendar_day",
            right_on="abx_calendar_day",
            by=STAY_KEY,
            strategy="nearest",
            tolerance=2,
        )
        .sort(STAY_KEY, "culture_calendar_day")
        .join_asof(
            organ_days,
            left_on="culture_calendar_day",
            right_on="organ_calendar_day",
            by=STAY_KEY,
            strategy="nearest",
            tolerance=2,
        )
        .filter(
            pl.col("abx_calendar_day").is_not_null(),
            pl.col("organ_calendar_day").is_not_null(),
        )
        .group_by(STAY_KEY)
        # Use culture timeframe as onset
        .agg(onset_timeframe_rhee=pl.min("timeframe"))
        .select(
            STAY_KEY,
            pl.col("onset_timeframe_rhee").alias("timeframe"),
            pl.lit("SEPSIS").alias("SEPSIS_RHEE"),
        )
    )
    # endregion Rhee

    sepsis_events = (
        pl.concat(
            [
                onset_labels.select(STAY_KEY, "timeframe", "SEPSIS"),
                septic_shock.select(
                    STAY_KEY,
                    "timeframe",
                    pl.lit("SHOCK").alias("SEPSIS"),
                ),
            ],
            how="diagonal_relaxed",
        )
        .group_by(STAY_KEY, "timeframe")
        .agg(
            pl.when((pl.col("SEPSIS") == "SHOCK").any())
            .then(pl.lit("SHOCK"))
            .otherwise(pl.lit("SEPSIS"))
            .alias("SEPSIS")
        )
    )

    sepsis_abx_events = (
        pl.concat(
            [
                onset_labels_abx.select(STAY_KEY, "timeframe", "SEPSIS_ABX"),
                septic_shock_abx.select(
                    STAY_KEY,
                    "timeframe",
                    pl.lit("SHOCK").alias("SEPSIS_ABX"),
                ),
            ],
            how="diagonal_relaxed",
        )
        .group_by(STAY_KEY, "timeframe")
        .agg(
            pl.when((pl.col("SEPSIS_ABX") == "SHOCK").any())
            .then(pl.lit("SHOCK"))
            .otherwise(pl.lit("SEPSIS"))
            .alias("SEPSIS_ABX")
        )
    )

    tf_union = pl.concat(
        [
            sepsis_events.select(STAY_KEY, "timeframe"),
            sepsis_abx_events.select(STAY_KEY, "timeframe"),
            onset_labels_rhee.select(STAY_KEY, "timeframe"),
        ],
        how="diagonal_relaxed",
    ).unique()

    cols = [STAY_KEY, "timeframe", "T_0"]
    if compute_seymour:
        cols.append("SEPSIS")
    if compute_shah:
        cols.append("SEPSIS_ABX")
    if compute_rhee:
        cols.append("SEPSIS_RHEE")

    # bound first: polars 2.0 pushes a filter on T_0 and timeframe below the first join if it comes after the later ones
    return (
        tf_union.join(ctx.all_stays_t0, on=STAY_KEY, how="left")
        .filter(*ctx.bounds)
        .join(sepsis_events,     on=[STAY_KEY, "timeframe"], how="left")
        .join(sepsis_abx_events, on=[STAY_KEY, "timeframe"], how="left")
        .join(onset_labels_rhee, on=[STAY_KEY, "timeframe"], how="left")
        .select(*cols)
        .unique()
        .rename({"timeframe": ctx.name})
        .sort(STAY_KEY, ctx.name)
    )


__all__ = ["SEPSIS"]
