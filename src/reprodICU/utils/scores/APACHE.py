"""
APACHE: compute Acute Physiology And Chronic Health Evaluation (II & III) in long format.

Output columns per row:
- Global ICU Stay ID
- T_0 (seconds from admission used as reference)
- APACHE II or APACHE III Score (sum of component points)
- Component points (APS/APS3 Score, age_points, history_points/comorbidity_points)

Time is in seconds. Windows determined by floor((time - T_0)/window_size).
APACHE scores are calculated for the first 24 hours of ICU admission.

SOURCES
-------
- Knaus WA, Draper EA, Wagner DP, Zimmerman JE.
  APACHE II: a severity of disease classification system.
  Crit Care Med. 1985 Oct;13(10):818-29. PMID: 3928249.
- Knaus WA, Wagner DP, Draper EA, Zimmerman JE, Bergner M, Bastos PG, Sirio CA, Murphy DJ, Lotring T, Damiano A, et al.
  The APACHE III prognostic system. Risk prediction of hospital mortality for critically ill hospitalized adults.
  Chest. 1991 Dec;100(6):1619-36.
  doi: 10.1378/chest.100.6.1619. PMID: 1959406.
- Kao E, Gulbis B.
  Validation of ICD-9-CM/ICD-10-CM Codes for Automated Electronic Scoring of APACHE II, APACHE III, and SAPS II.
  online publication (2016 Nov).
  https://rstudio-pubs-static.s3.amazonaws.com/231351_940f14aa51a6427a9e92d5a04daefc3e.html
"""

from pathlib import Path
from typing import Optional

import polars as pl
from pycomorb import CustomComorbidityIndex

from ..common import (
    _to_lazy,
    get_diagnoses,
    get_patient_information,
    get_timeseries_intakeoutput,
    get_timeseries_labs,
    get_timeseries_respiratory,
    get_timeseries_vitals,
    get_ventilation,
)
from ..comorbidity import _load_and_process_diagnoses
from .APS import APS, APS3

STAY_KEY = "Global ICU Stay ID"
TIME_KEY = "Time Relative to Admission (seconds)"
ICD_COL = "Diagnosis ICD-10 Code"

SECONDS_IN_1D = 86400


# region age points
def _apache2_age_points(age_col: str = "Admission Age (years)") -> pl.Expr:
    """
    APACHE II points for age.

    Age (years)
    <44      0
     45-54  +2
     55-64  +3
     65-74  +5
    >75     +6
    """
    return (
        pl.when(pl.col(age_col) <= 44)
        .then(0)
        .when(pl.col(age_col).is_between(45, 54))
        .then(2)
        .when(pl.col(age_col).is_between(55, 64))
        .then(3)
        .when(pl.col(age_col).is_between(65, 74))
        .then(5)
        .when(pl.col(age_col) >= 75)
        .then(6)
        .otherwise(None)
    )


def _apache3_age_points(age_col: str = "Admission Age (years)") -> pl.Expr:
    """
    APACHE III points for age.

    Age (years)
    <44       0
     45-59   +5
     60-64  +11
     65-69  +13
     70-74  +16
     75-84  +17
    >85     +24
    """
    return (
        pl.when(pl.col(age_col) <= 44)
        .then(0)
        .when(pl.col(age_col).is_between(45, 59))
        .then(5)
        .when(pl.col(age_col).is_between(60, 64))
        .then(11)
        .when(pl.col(age_col).is_between(65, 69))
        .then(13)
        .when(pl.col(age_col).is_between(70, 74))
        .then(16)
        .when(pl.col(age_col).is_between(75, 84))
        .then(17)
        .when(pl.col(age_col) >= 85)
        .then(24)
        .otherwise(None)
    )


# endregion


# region history points
def _apache2_history_points(
    patient_information: pl.LazyFrame, diagnoses: pl.LazyFrame
) -> pl.LazyFrame:
    """
    APACHE II points for chronic health history.

    Points:
        - 5 points for non-operative or emergency postoperative patients.
        - 2 points for elective postoperative patients.

    Criteria:
        - Liver: Biopsy proven cirrhosis, portal hypertension, episodes of past upper GI bleeding attributed to portal hypertension, or prior episodes of hepatic failure/encephalopathy/coma.
        - Cardiovascular: New York Heart Association (NYHA) Class IV.
        - Respiratory: Chronic restrictive, obstructive, or vascular disease resulting in severe exercise restriction, i.e., unable to climb stairs or perform household duties; or documented chronic hypoxia, hypercapnia, secondary polycythemia, severe pulmonary hypertension (>40 mmHg), or respirator dependency.
        - Renal: Receiving chronic dialysis.
        - Immunocompromised: The patient has received therapy that suppresses resistance to infection, e.g., immuno-suppression, chemotherapy, radiation, long term or recent high dose steroids, or has a disease that is sufficiently advanced to suppress resistance to infection, e.g., leukemia, lymphoma, AIDS.
    """
    # ICD prefixes for APACHE II history categories
    APACHE2_history_data = pl.DataFrame(
        {
            "category": [
                "Biopsy proven cirrhosis and documented portal hypertension",
                "Past upper GI bleed attributed to portal hypertension",
                "Portal hypertension",
                "Prior hepatic failure, encephalopathy, or coma",
                "Heart failure",
                "Chronic restrictive, obstructive, or vascular disease",
                "Chronic hypoxia",
                "Chronic hypercapnia",
                "Secondary polycythemia",
                "Severe pulmonary hypertension",
                "Respirator dependence",
                "Receiving chronic dialysis",
                "Immuno-suppression therapy",
                "Chemotherapy",
                "Radiation",
                "Long-term or high-dose steroids",
                "Leukemia",
                "Lymphoma",
                "HIV infection",
                "AIDS defining disease",
            ],
            "icd9_codes": [
                "456|56723|5712|5715|5722|5724|78959",
                "53082|5789",
                "5723",
                "0702|0703|07041|07044|07051|07054|0707|155|275|571|5733|5722",
                "428",
                "490|491|4920|4921|4922|4923|4924|4925|4926|4927|4928|4930|4931|4932|4933|4934|4935|4936|4937|4938|49390|49391|494|495|496|497|498|499|500|501|502|503|504|505|5064|440|4412|4414|4417|4419|4431|4432|4433|4434|4435|4436|4437|4438|4439|4471|5571|5579|V434",
                "51883|79902",
                "78609",
                "2890",
                "416",
                "V4611",
                "5856",
                "V8746",
                "V581|V8741|E9331",
                "E926",
                "V5865",
                "204|205|206|207|208",
                "200|201|2020|2021|2022|20230|20231|20232|20233|20234|20235|20236|20237|20238|2025|2026|2027|2028|2029|20300|20301|20380|20381|2386|2733|V1071|V1072|V1079",
                "042",
                "176|1363|130|010|011|012|013|014|015|016|017|018",
            ],
            "icd10_codes": [
                "I8500|I8501|I8510|I8511|K652|K7030|K7290|K7291|K740|K7460|K7469|K767|R188",
                "K228|K922",
                "K766",
                "B160|B161|B162|B169|B1710|B1711|B180|B180|B181|B181|B182|B182|B1910|B1911|B1920|B1921|C220|C221|C222|C227|C228|K700|K7010|K7030|K709|K716|K730|K732|K738|K739|K740|K741|K743|K744|K745|K7460|K7469|K754|K759|K760|K7689|K769|K7290|K7291",
                "I501|I5020|I5021|I5022|I5023|I5030|I5031|I5032|I5033|I5040|I5041|I5042|I5043|I509|I509",
                "I278|I279|J40|J41|J42|J43|J44|J45|J46|J47|J60|J61|J62|J63|J64|J65|J66|J67|J684|J701|J703|I70|I71|I731|I738|I739|I771|I790|I792|K551|K558|K559|Z958|Z959",
                "J9610|R0902",
                "R0600|R0609|R063|R0683|R0689",
                "D751",
                "I270",
                "Z9911",
                "N186",
                "Z9225",
                "Z5111|Z5112|Z9221",
                "W900X",
                "Z7951|Z7952",
                "C9500|C9501|C9502|C9510|C9511|C9512|C9590|C9590|C9590|C9591|C9592",
                "C81|C82|C83|C84|C85|C88|C900|C902|C96",
                "B20",
                "C460|B59|B5801|B5809|B581|B582|B583|B5881|B5889|B5889|B589|A150|A154|A155|A156|A157|A158|A170|A171|A1781|A1782|A1789|A179|A1801|A1802|A1803|A1810|A1811|A1812|A1813|A1814|A1815|A1816|A1817|A1818|A182|A1831|A1832|A1839|A184|A1850|A1851|A1852|A1853|A1854|A1859|A186|A187|A1881|A1884|A1885|A1889|A192|A198|A199",
            ],
            "weights": [1] * 20,
        }
    )

    APACHE2_history = (
        CustomComorbidityIndex(
            df=_load_and_process_diagnoses(),
            id_col=STAY_KEY,
            code_col="Diagnosis ICD Code",
            icd_version_col="Diagnosis ICD Code Version (source)",
            definition_data=APACHE2_history_data,
            weight_col_name="weights",
            score_col_name="score",
            return_categories=True,
        )
        .with_columns(
            pl.when(
                pl.col("HIV infection").cast(bool)
                & pl.col("AIDS defining disease").cast(bool)
            )
            .then(1)
            .otherwise(0)
            .alias("AIDS")
        )
        .drop("score", "HIV infection", "AIDS defining disease")
        .with_columns(
            pl.any_horizontal(pl.exclude(STAY_KEY)).alias("has_history")
        )
        .lazy()
    )

    return (
        patient_information.join(APACHE2_history, on=STAY_KEY, how="left")
        .group_by(STAY_KEY)
        .agg(
            pl.col("has_history").any(),
            (
                (pl.col("Admission Type") == "Surgical")
                & (pl.col("Admission Urgency") == "Elective")
            )
            .any()
            .alias("is_elective_surgical"),
        )
        .select(
            STAY_KEY,
            pl.when(pl.col("has_history"))
            .then(pl.when("is_elective_surgical").then(2).otherwise(5))
            .otherwise(0)
            .alias("history_points"),
        )
    )


def _apache3_comorbidity_points(
    patient_information: pl.LazyFrame, diagnoses: pl.LazyFrame
) -> pl.LazyFrame:
    """
    APACHE III points for comorbidity history.

    Comorbidity
    AIDS                          +23
    Hepatic failure               +16
    Lymphoma                      +13
    Metastatic cancer             +11
    Immuno-suppression therapy    +10
    Leukemia or multiple myeloma  +10
    Cirrhosis                      +4
    """
    # ICD-10 prefixes for APACHE III comorbidity categories
    # ICD prefixes for APACHE II history categories
    APACHE3_history_data = pl.DataFrame(
        {
            "category": [
                "HIV infection",
                "AIDS defining disease",
                "Hepatic failure",
                "Lymphoma",
                "Metastatic cancer",
                "Leukemia or multiple myeloma",
                "Immuno-suppression therapy",
                "Cirrhosis",
            ],
            "icd9_codes": [
                "042",
                "176|1363|130|010|011|012|013|014|015|016|017|018",
                "0702|0703|07041|07044|07051|07054|0707|155|275|571|5733",
                "200|201|2020|2021|2022|20230|20231|20232|20233|20234|20235|20236|20237|20238|2025|2026|2027|2028|2029|20300|20301|20380|20381|2386|2733|V1071|V1072|V1079",
                "196|197|198|1990|1991",
                "203|204|205|206|207|208",
                "V8746",
                "456|56723|5712|5715|5722|5724|78959",
            ],
            "icd10_codes": [
                "B20",
                "C460|B59|B5801|B5809|B581|B582|B583|B5881|B5889|B5889|B589|A150|A154|A155|A156|A157|A158|A170|A171|A1781|A1782|A1789|A179|A1801|A1802|A1803|A1810|A1811|A1812|A1813|A1814|A1815|A1816|A1817|A1818|A182|A1831|A1832|A1839|A184|A1850|A1851|A1852|A1853|A1854|A1859|A186|A187|A1881|A1884|A1885|A1889|A192|A198|A199",
                "B160|B161|B162|B169|B1710|B1711|B180|B180|B181|B181|B182|B182|B1910|B1911|B1920|B1921|C220|C221|C222|C227|C228|K700|K7010|K7030|K709|K716|K730|K732|K738|K739|K740|K741|K743|K744|K745|K7460|K7469|K754|K759|K760|K7689|K769",
                "C81|C82|C83|C84|C85|C88|C900|C902|C96",
                "C77|C78|C79|C80",
                "C9500|C9501|C9502|C9510|C9511|C9512|C9590|C9590|C9590|C9591|C9591|C9591|C9592|C9592|C9592|C9000|C9000|C9001|C9002",
                "Z9225",
                "I8500|I8501|I8510|I8511|K652|K7030|K7290|K7291|K740|K7460|K7469|K767|R188",
            ],
            "weights": [1] * 8,
        }
    )

    APACHE3_history = (
        CustomComorbidityIndex(
            df=_load_and_process_diagnoses(
                diagnoses=diagnoses,
                patient_information=patient_information,
            ),
            id_col=STAY_KEY,
            code_col="Diagnosis ICD Code",
            icd_version_col="Diagnosis ICD Code Version (source)",
            definition_data=APACHE3_history_data,
            weight_col_name="weights",
            score_col_name="score",
            return_categories=True,
        )
        .with_columns(
            pl.when(
                pl.col("HIV infection").cast(bool)
                & pl.col("AIDS defining disease").cast(bool)
            )
            .then(1)
            .otherwise(0)
            .alias("AIDS")
        )
        .drop("score", "HIV infection", "AIDS defining disease")
        .lazy()
    )

    comorb_weights = {
        "Cirrhosis": 4,
        "Leukemia or multiple myeloma": 10,
        "Immuno-suppression therapy": 10,
        "Metastatic cancer": 11,
        "Lymphoma": 13,
        "Hepatic failure": 16,
        "AIDS": 23,
    }

    points_expr = pl.lit(0)
    name_expr = pl.lit(None)
    for name, weight in comorb_weights.items():
        points_expr = (
            pl.when(pl.col(name).cast(bool)).then(weight).otherwise(points_expr)
        )
        name_expr = (
            pl.when(pl.col(name).cast(bool))
            .then(pl.lit(name))
            .otherwise(name_expr)
        )

    return APACHE3_history.select(
        STAY_KEY,
        points_expr.alias("comorbidity_points"),
        name_expr.alias("comorbidity_name"),
    )


# endregion


################################################################################
################################################################################
# region APACHE2
def APACHE2(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_vitals: Optional[pl.LazyFrame] = None,
    timeseries_labs: Optional[pl.LazyFrame] = None,
    timeseries_resp: Optional[pl.LazyFrame] = None,
    diagnoses: Optional[pl.LazyFrame] = None,
    *,
    t_0: int = 0,
    t_0_per_stay: Optional[pl.LazyFrame] = None,
) -> pl.LazyFrame:
    """
    Acute Physiology And Chronic Health Evaluation II (APACHE II).

    APACHE II = APS + Age Points + Chronic Health Points.

    Steps:
        1. Calculate Acute Physiology Score (APS).
        2. Calculate Age Points based on "Admission Age (years)".
        3. Calculate Chronic Health Points based on diagnoses and "Admission Type".
        4. Assemble components and sum to get the final APACHE II score.

    Returns:
        pl.LazyFrame: Contains columns:
            - {stay_id_col}: Unique identifier for the ICU stay.
            - T_0: Reference time for the stay.
            - APACHE II Score.
            - Component points (APS Score, age_points, history_points)
    """
    # region data loading
    if patient_information is None:
        patient_information = get_patient_information()
    if timeseries_vitals is None:
        timeseries_vitals = get_timeseries_vitals()
    if timeseries_labs is None:
        timeseries_labs = get_timeseries_labs()
    if timeseries_resp is None:
        timeseries_resp = get_timeseries_respiratory()
    if diagnoses is None:
        diagnoses = get_diagnoses()

    # Validate all required data is available
    required = {
        "patient_information": patient_information,
        "timeseries_vitals": timeseries_vitals,
        "timeseries_labs": timeseries_labs,
        "timeseries_resp": timeseries_resp,
        "diagnoses": diagnoses,
    }

    missing = [name for name, data in required.items() if data is None]
    if missing:
        raise ValueError(
            f"Cannot compute APACHE II: Missing required datasets: {', '.join(missing)}. "
            f"Ensure they are configured in ~/.reprodICU/PATHS.yaml or provide them explicitly."
        )

    info = _to_lazy(patient_information)
    diag = _to_lazy(diagnoses)

    # region component scoring
    # 1. APS
    aps_tf = (
        APS(
            patient_information=patient_information,
            timeseries_vitals=timeseries_vitals,
            timeseries_labs=timeseries_labs,
            timeseries_resp=timeseries_resp,
            t_0=t_0,
            t_0_per_stay=t_0_per_stay,
            window_size=SECONDS_IN_1D,
        )
        .filter(pl.col("Days Relative to Admission") == 1)
        .select(STAY_KEY, "T_0", "APS Score")
    )

    # 2. Age Points
    age_tf = info.select(STAY_KEY, _apache2_age_points().alias("age_points"))

    # 3. History Points
    history_tf = _apache2_history_points(info, diag)
    # endregion

    # region assemble
    return (
        aps_tf.join(age_tf, on=STAY_KEY, how="left")
        .join(history_tf, on=STAY_KEY, how="left")
        .with_columns(
            pl.sum_horizontal(
                "APS Score", "age_points", "history_points"
            ).alias("APACHE II Score")
        )
        .select(
            STAY_KEY,
            "T_0",
            "APACHE II Score",
            "APS Score",
            "age_points",
            "history_points",
        )
    )


# endregion


################################################################################
################################################################################
# region APACHE3
def APACHE3(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_vitals: Optional[pl.LazyFrame] = None,
    timeseries_labs: Optional[pl.LazyFrame] = None,
    timeseries_resp: Optional[pl.LazyFrame] = None,
    timeseries_inout: Optional[pl.LazyFrame] = None,
    diagnoses: Optional[pl.LazyFrame] = None,
    ventilation: Optional[pl.LazyFrame] = None,
    *,
    t_0: int = 0,
    t_0_per_stay: Optional[pl.LazyFrame] = None,
) -> pl.LazyFrame:
    """
    Acute Physiology And Chronic Health Evaluation III (APACHE III).

    APACHE III = APS III + Age Points + Comorbidity Points.

    Steps:
        1. Calculate Acute Physiology Score III (APS III).
        2. Calculate Age Points based on Admission Age (years).
        3. Calculate Comorbidity Points based on diagnoses.
        4. Assemble components and sum to get the final APACHE III score.

    Returns:
        pl.LazyFrame: Contains columns:
            - {stay_id_col}: Unique identifier for the ICU stay.
            - T_0: Reference time for the stay.
            - APACHE III Score.
            - Component points (APS3 Score, age_points, comorbidity_points)
    """
    # region data loading
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
    if diagnoses is None:
        diagnoses = get_diagnoses()
    if ventilation is None:
        ventilation = get_ventilation()

    # Validate all required data is available
    required = {
        "patient_information": patient_information,
        "timeseries_vitals": timeseries_vitals,
        "timeseries_labs": timeseries_labs,
        "timeseries_resp": timeseries_resp,
        "diagnoses": diagnoses,
        "ventilation": ventilation,
    }

    missing = [name for name, data in required.items() if data is None]
    if missing:
        raise ValueError(
            f"Cannot compute APACHE III: Missing required datasets: {', '.join(missing)}. "
            f"Ensure they are configured in ~/.reprodICU/PATHS.yaml or provide them explicitly."
        )

    info = _to_lazy(patient_information)
    diag = _to_lazy(diagnoses)

    # region component scoring
    # 1. APS3
    aps3_tf = (
        APS3(
            patient_information=patient_information,
            timeseries_vitals=timeseries_vitals,
            timeseries_labs=timeseries_labs,
            timeseries_resp=timeseries_resp,
            timeseries_inout=timeseries_inout,
            ventilation=ventilation,
            t_0=t_0,
            t_0_per_stay=t_0_per_stay,
            window_size=SECONDS_IN_1D,
        )
        .filter(pl.col("Days Relative to Admission") == 1)
        .select(STAY_KEY, "T_0", "APS3 Score")
    )

    # 2. Age Points
    age_tf = info.select(
        STAY_KEY,
        _apache3_age_points().alias("age_points"),
    )

    # 3. Comorbidity Points
    comorbidity_tf = _apache3_comorbidity_points(info, diag)
    # endregion

    # region assemble
    return (
        aps3_tf.join(age_tf, on=STAY_KEY, how="left")
        .join(comorbidity_tf, on=STAY_KEY, how="left")
        .with_columns(
            pl.sum_horizontal(
                "APS3 Score", "age_points", "comorbidity_points"
            ).alias("APACHE III Score")
        )
        .select(
            STAY_KEY,
            "T_0",
            "APACHE III Score",
            "APS3 Score",
            "age_points",
            "comorbidity_points",
        )
    )


# endregion


################################################################################
################################################################################
# region APACHE II mortality
def _APACHE2_mortality(
    apache2_score: pl.Expr,
    is_emergency_surgery: pl.Expr,
    diagnostic_category_weight: pl.Expr = pl.lit(0),
) -> pl.Expr:
    """
    Calculate predicted ICU mortality rate from APACHE II score.

    ln(R/1-R)= -3.517 + (APACHE II score x 0.146)
                      + (0.603, only if postemergency surgery)
                      + (diagnostic category weight)
    """

    logit = (
        -3.517
        + (apache2_score * 0.146).fill_null(0)
        + (pl.when(is_emergency_surgery).then(0.603).otherwise(0))
        + diagnostic_category_weight.fill_null(0)
    )
    odds = logit.exp()
    return odds / (1 + odds)


def APACHE2_mortality(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_vitals: Optional[pl.LazyFrame] = None,
    timeseries_labs: Optional[pl.LazyFrame] = None,
    timeseries_resp: Optional[pl.LazyFrame] = None,
    diagnoses: Optional[pl.LazyFrame] = None,
    *,
    t_0: int = 0,
    t_0_per_stay: Optional[pl.LazyFrame] = None,
) -> pl.LazyFrame:
    """
    Calculate predicted mortality rate from APACHE II score.
    """
    if patient_information is None:
        patient_information = get_patient_information()

    return (
        APACHE2(
            patient_information=patient_information,
            timeseries_vitals=timeseries_vitals,
            timeseries_labs=timeseries_labs,
            timeseries_resp=timeseries_resp,
            diagnoses=diagnoses,
            t_0=t_0,
            t_0_per_stay=t_0_per_stay,
        )
        .join(
            patient_information.select(
                STAY_KEY,
                "Admission Type",
                "Admission Urgency",
                "Admission Diagnosis (APACHE)",
            ),
            on=STAY_KEY,
            how="left",
        )
        .select(
            STAY_KEY,
            "APACHE II Score",
            _APACHE2_mortality(
                pl.col("APACHE II Score"),
                (pl.col("Admission Type") == "Surgical")
                & (pl.col("Admission Urgency").is_in(["Urgent", "Emergency"])),
                pl.col("Admission Diagnosis (APACHE)")
                .pipe(_APACHE_diagnostic_category_weight, "II")
                .alias("APACHE diagnostic category weight"),
            ).alias("Predicted ICU Mortality Rate"),
        )
    )


# region APACHE IV mortality
def _APACHE4_hospital_mortality(
    age: pl.Expr,
    chronic_health: pl.Expr,
    aps3_score: pl.Expr,
    pafi_ratio: pl.Expr,
    ventilated: pl.Expr,
    icu_admission_source: pl.Expr,
    is_emergency_surgery: pl.Expr,
    pre_icu_length_of_stay: pl.Expr,
    diagnostic_category_weight: pl.Expr = pl.lit(0),
    worst_gcs: pl.Expr = pl.lit(15),
) -> pl.Expr:
    """
    Calculate predicted ICU mortality rate from APACHE II score.

    ln(R/1-R)= -5.950 + (age spline)
                      + (chronic health weight)
                      + (APS3 score spline)
                      + (ICU admission source weight)
                      + (0.249, only if postemergency surgery)
                      + (pre-ICU length of stay spline)
                      + (diagnostic category weight)
                      + (gcs)
    """

    age_spline = (
        0.0241774552028671 * age
        + pl.when(age > 27).then(-4.38861616925701e-6 * (age - 27) ** 3)
        + pl.when(age > 51).then( 5.01422332950743e-5 * (age - 51) ** 3)
        + pl.when(age > 64).then(-1.27787345016656e-4 * (age - 64) ** 3)
        + pl.when(age > 74).then( 1.09605981585416e-4 * (age - 74) ** 3)
        + pl.when(age > 86).then(-2.75722536945775e-5 * (age - 86) ** 3)
    ) # fmt: skip

    chronic_health_weight = (
        pl.when(chronic_health == "AIDS")
        .then(0.958100515711858)
        .when(chronic_health == "Hepatic failure")
        .then(1.03737992527122)
        .when(chronic_health == "Lymphoma")
        .then(0.743471747986033)
        .when(chronic_health == "Metastatic cancer")
        .then(1.08642375217988)
        .when(chronic_health == "Leukemia or multiple myeloma")
        .then(0.969308298882914)
        .when(chronic_health == "Immuno-suppression therapy")
        .then(0.435581082699776)
        .when(chronic_health == "Cirrhosis")
        .then(0.814665087687609)
        .otherwise(0)
    )

    aps3_spline = (
        0.0556349159410042 * aps3_score
        + pl.when(aps3_score > 10).then( 8.71852213571101e-6 * (aps3_score - 10) ** 3)
        + pl.when(aps3_score > 22).then(-4.51101465395454e-5 * (aps3_score - 22) ** 3)
        + pl.when(aps3_score > 32).then( 5.03800407345824e-5 * (aps3_score - 32) ** 3)
        + pl.when(aps3_score > 48).then(-1.31230671327519e-5 * (aps3_score - 48) ** 3)
        + pl.when(aps3_score > 89).then(-8.65349197996149e-7 * (aps3_score - 89) ** 3)
    ) # fmt: skip

    admission_source_weight = (
        pl.when(
            icu_admission_source.is_in(
                [
                    "Other Hospital",
                    "Nursing Facility",
                    "Psychiatric Facility",
                    "Other ICU",
                    "High-Dependency Unit",
                ]
            )
        )
        .then(0.0221062655202521)
        .when(
            icu_admission_source.is_in(
                [
                    "Operating Room",
                    "Recovery Room",
                ]
            )
        )
        .then(-0.5838281212914)
        .otherwise(0.0171491928103093)
    )

    los_sqrt = pre_icu_length_of_stay.sqrt().fill_null(0)
    los_spline = (
        -0.310487496000706 * los_sqrt
        + pl.when(los_sqrt > 0.121).then( 1.47467251149713   * (los_sqrt - 0.121) ** 3)
        + pl.when(los_sqrt > 0.423).then(-2.86188569954954   * (los_sqrt - 0.423) ** 3)
        + pl.when(los_sqrt > 0.794).then( 1.42165901026679   * (los_sqrt - 0.794) ** 3)
        + pl.when(los_sqrt > 2.806).then(-0.0344458222143702 * (los_sqrt - 2.806) ** 3)
    ) # fmt: skip

    logit = (
        -5.95047195162616
        + age_spline
        + chronic_health_weight
        + aps3_spline
        + admission_source_weight
        + (pafi_ratio * -0.000397068178650386)
        + pl.when(ventilated).then(0.271760035621294).otherwise(0)
        + pl.when(is_emergency_surgery).then(0.249073458479819).otherwise(0)
        + los_spline
        + diagnostic_category_weight
        + ((15 - worst_gcs) * 0.0391175318213502)
    )
    odds = logit.exp()
    return odds / (1 + odds)


def APACHE4_hospital_mortality(
    patient_information: Optional[pl.LazyFrame] = None,
    timeseries_vitals: Optional[pl.LazyFrame] = None,
    timeseries_labs: Optional[pl.LazyFrame] = None,
    timeseries_resp: Optional[pl.LazyFrame] = None,
    timeseries_inout: Optional[pl.LazyFrame] = None,
    diagnoses: Optional[pl.LazyFrame] = None,
    ventilation: Optional[pl.LazyFrame] = None,
    *,
    t_0: int = 0,
    t_0_per_stay: Optional[pl.LazyFrame] = None,
) -> pl.LazyFrame:
    """
    Calculate predicted mortality rate from APACHE IV score.
    """
    if patient_information is None:
        patient_information = get_patient_information()
    if timeseries_vitals is None:
        timeseries_vitals = get_timeseries_vitals()
    if diagnoses is None:
        diagnoses = get_diagnoses()

    return (
        APS3(
            patient_information=patient_information,
            timeseries_vitals=timeseries_vitals,
            timeseries_labs=timeseries_labs,
            timeseries_resp=timeseries_resp,
            timeseries_inout=timeseries_inout,
            ventilation=ventilation,
            t_0=t_0,
            t_0_per_stay=t_0_per_stay,
        )
        .join(
            patient_information.select(
                STAY_KEY,
                "Admission Age (years)",
                "Admission Origin",
                "Admission Type",
                "Admission Urgency",
                "Admission Diagnosis (APACHE)",
                "Pre-ICU Length of Stay (days)",
            ),
            on=STAY_KEY,
            how="left",
        )
        .join(
            _apache3_comorbidity_points(patient_information, diagnoses).select(
                STAY_KEY,
                pl.col("comorbidity_name").alias("Chronic Health"),
            ),
            on=STAY_KEY,
            how="left",
        )
        .with_columns(
            pl.col("Admission Diagnosis (APACHE)")
            .pipe(_APACHE_diagnostic_category_weight, "IV")
            .alias("APACHE diagnostic category weight"),
        )
        .sink_parquet("apache4_mortality_debug.parquet")
        .select(
            STAY_KEY,
            "APS3 Score",
            _APACHE4_hospital_mortality(
                pl.col("Admission Age (years)"),
                pl.col("Chronic Health"),
                pl.col("APS3 Score"),
                pl.lit(300),  # P/F ratio placeholder
                pl.lit(False),  # Ventilated placeholder
                pl.col("Admission Origin"),
                (pl.col("Admission Type") == "Surgical")
                & (pl.col("Admission Urgency").is_in(["Urgent", "Emergency"])),
                pl.col("Pre-ICU Length of Stay (days)"),
                pl.col("Admission Diagnosis (APACHE)")
                .pipe(_APACHE_diagnostic_category_weight, "IV")
                .alias("APACHE diagnostic category weight"),
                pl.lit(15),  # Worst GCS placeholder
            ).alias("Predicted ICU Mortality Rate"),
        )
    )


# endregion


################################################################################
################################################################################
# region diagnostic category weights
def _APACHE_diagnostic_category_weight(
    diagnostic_category: pl.Expr, version: str
) -> pl.Expr:
    """
    Get APACHE diagnostic category weight based on admission diagnosis.
    """
    weights_path = Path(__file__).parent / "APACHE.tsv"

    diagnostic_category_to_coefficient = dict(
        pl.read_csv(weights_path, separator="\t")
        .filter(pl.col("APACHE version") == version)
        .select("APACHE diagnostic category", pl.col("coefficient").cast(float))
        .iter_rows()
    )

    return diagnostic_category.replace_strict(
        diagnostic_category_to_coefficient, default=None
    )


__all__ = ["APACHE2", "APACHE3"]
