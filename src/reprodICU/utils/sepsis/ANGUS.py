# based on: https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/concepts/sepsis/angus.sql
# based on: https://github.com/alistairewj/mimic-iv-aline-study/blob/main/sql/angus_sepsis.sql

# Angus et al, 2001. Epidemiology of severe sepsis in the United States
# http://www.ncbi.nlm.nih.gov/pubmed/11445675

# Case selection and definitions
# To identify cases with severe sepsis, we selected all acute care
# hospitalizations with ICD-9-CM codes for both:
# (a) a bacterial or fungal infectious process AND
# (b) a diagnosis of acute organ dysfunction (Appendix 2).

from typing import Optional

import polars as pl

from ..common import (
    _to_lazy,
    _validate_required_data,
    get_diagnoses,
    get_patient_information,
    get_procedures,
    get_ventilation,
)

# ICD-9 codes for infection - as sourced from Appendix 1
ANGUS_ICD9_3_DIGIT_INFECTION_CODES = [
           "001", "002", "003", "004", "005",               "008", "009",
    "010", "011", "012", "013", "014", "015", "016", "017", "018",
    "020", "021", "022", "023", "024", "025", "026", "027",
    "030", "031", "032", "033", "034", "035", "036", "037", "038", "039",
    "040", "041",
    "090", "091", "092", "093", "094", "095", "096", "097", "098",
    "100", "101", "102", "103", "104",
    "110", "111", "112",        "114", "115", "116", "117", "118",
    "320",        "322",        "324", "325",
    "420", "421",
           "451",
           "461", "462", "463", "464", "465",
           "481", "482",               "485", "486",
                                "494",
    "510",               "513",
    "540", "541", "542",
                                              "566", "567",
    "590",                                           "597",
           "601",
                                "614", "615", "616",
           "681", "682", "683",               "686",
    "730",
] # fmt: skip
ANGUS_ICD9_4_DIGIT_INFECTION_CODES = ["5695", "5720", "5721", "5750", "5990", "7110", "7907", "9966", "9985", "9993"] # fmt: skip
ANGUS_ICD9_5_DIGIT_INFECTION_CODES = ["49121", "56201", "56203", "56211", "56213", "56983"] # fmt: skip

# ICD-9 codes for organ dysfunction - as sourced from Appendix 2
ANGUS_ICD9_3_DIGIT_DYSFUNCTION_CODES = ["458", "293", "570", "584"]
ANGUS_ICD9_4_DIGIT_DYSFUNCTION_CODES = ["7855", "3483", "3481", "2874", "2875", "2869", "2866", "5734"] # fmt: skip

# ICD-9 codes for explicit sepsis
ANGUS_ICD9_EXPLICIT_SEPSIS = ["99592", "78552"]

# ICD-9 codes for mechanical ventilation
MECHVENT_ICD9_CODES = ["9670", "9671", "9672"]

# ICD-10 codes for infection
ANGUS_ICD10_INFECTION_CODES = [
    "A000", "A001", "A009", "A0100", "A011", "A012", "A013", "A014", "A020", "A021", "A0220", "A0221", "A0222", "A0223", "A0224", "A0229", "A028", "A029", "A030", "A031", "A032", "A033", "A038", "A039", "A050", "A051", "A052", "A058", "A053", "A055", "A054", "A059", "A044", "A040", "A041", "A042", "A043", "A048", "A045", "A046", "A047", "A049", "A080", "A082", "A0811", "A0819", "A0831", "A0832", "A0839", "A088", "A09", "A157", "A156", "A150", "A155", "A154", "A158", "A170", "A171", "A1781", "A1782", "A1789", "A179", "A1831", "A1832", "A1839", "A1801", "A1802", "A1803", "A1811", "A1812", "A1813", "A1815", "A1814", "A1817", "A1816", "A1818", "A1810", "A184", "A182", "A1850", "A1851", "A1852", "A1853", "A1854", "A1859", "A186", "A1881", "A187", "A1885", "A1889", "A1884", "A192", "A198", "A199", "A200", "A201", "A207", "A202", "A208", "A209", "A210", "A213", "A212", "A211", "A217", "A218", "A219", "A220", "A221", "A222", "A227", "A228", "A229", "A230", "A231", "A232", "A233", "A238", "A239", "A240", "A243", "A249", "A250", "A251", "A259", "A3211", "A3212", "A327", "A3281", "A3289", "A329", "A267", "A268", "A269", "A280", "A288", "A289", "A305", "A301", "A300", "A303", "A308", "A309", "A310", "A311", "A312", "A318", "A319", "A360", "A361", "A3689", "A362", "A3686", "A3681", "A3685", "A363", "A3682", "A3683", "A3684", "A369", "A3700", "A3710", "A3780", "A3790", "J020", "J0300", "A389", "A46", "A390", "A3981", "A394", "A391", "A3950", "A3953", "A3951", "A3952", "A3982", "A3983", "A3989", "A399", "A35", "A409", "A412", "A4101", "A4102", "A411", "A403", "A414", "A4150", "A413", "A4151", "A4152", "A4153", "A4159", "A4189", "A419", "L081", "A420", "A421", "A422", "B479", "A4281", "A4282", "A4289", "A438", "A429", "A439", "B471", "A480", "A488", "K9081", "A4851", "A4852", "M60009", "A483", "B955", "B950", "B951", "B954", "B952", "B958", "B9561", "B9562", "B957", "B953", "B961", "B9621", "B9622", "B9623", "B9620", "B9629", "B963", "B964", "B965", "A493", "B960", "B966", "B967", "B9689", "B9681", "A5009", "A501", "A502", "A5031", "A5040", "A5045", "A5042", "A5041", "A5049", "A5052", "A5057", "A5059", "A506", "A507", "A509", "A510", "A511", "A512", "A5131", "A5139", "A5149", "A5143", "A5146", "A5145", "A5141", "A5132", "A515", "A5201", "A5202", "A5203", "A5206", "A5209", "A5200", "A5211", "A5217", "A5213", "A522", "A5214", "A5219", "A5215", "A523", "A5271", "A5272", "A5274", "A5275", "A5277", "A5278", "A5273", "A5276", "A5279", "A528", "A529", "A530", "A539", "A5400", "A5429", "A5401", "A5422", "A5423", "A5403", "A5424", "A5421", "A5431", "A5432", "A5439", "A5433", "A5442", "A5449", "A5441", "A5440", "A545", "A546", "A5489", "A5481", "A5483", "A5485", "A5486", "A270", "A2781", "A2789", "A279", "A690", "A691", "A660", "A661", "A662", "A663", "A664", "A665", "A666", "A667", "A668", "A669", "A670", "A671", "A672", "A673", "A679", "A65", "A698", "A699", "B350", "B351", "B352", "B356", "B353", "B355", "B358", "B359", "B360", "B361", "B362", "B363", "B368", "B369", "B370", "B3783", "B373", "B3742", "B3749", "B372", "B371", "B377", "B376", "B3784", "B375", "B3781", "B3782", "B3789", "B379", "B380", "B383", "B384", "B3889", "B381", "B382", "B389", "B394", "G02", "H32", "I32", "I39", "B392", "B393", "B395", "J17", "B399", "B409", "B410", "B419", "B480", "B481", "B420", "B421", "B427", "B429", "B439", "B449", "B470", "B450", "B457", "B459", "B482", "B469", "B488", "B49", "G000", "G001", "G002", "G003", "G01", "G008", "G009", "G042", "G030", "G038", "G031", "G039", "G060", "G061", "G062", "G08", "I309", "I300", "I308", "I330", "I339", "I8000", "I8010", "I80209", "I803", "I80219", "I808", "I809", "J0100", "J0110", "J0120", "J0130", "J0140", "J0190", "J029", "J0390", "J040", "J050", "J0410", "J0411", "J042", "J0510", "J0511", "J0430", "J0431", "J060", "J069", "J13", "J181", "J150", "J151", "J14", "J154", "J153", "J1520", "J15211", "J15212", "J1529", "J158", "J155", "J156", "A481", "J159", "J180", "J189", "J479", "J471", "J860", "J869", "J850", "J851", "J852", "J853", "K352", "K353", "K3580", "K3589", "K37", "K36", "K610", "K611", "K613", "K67", "K658", "K650", "K651", "K652", "K6812", "K6819", "K689", "K653", "K654", "K659", "N110", "N118", "N10", "N151", "N2884", "N2885", "N2886", "N12", "N16", "N159", "N340", "N341", "N342", "N343", "N410", "N411", "N412", "N413", "N51", "N414", "N418", "N419", "N7001", "N7002", "N7003", "N7011", "N7012", "N7013", "N7091", "N7092", "N7093", "N730", "N731", "N732", "N733", "N736", "N734", "N738", "N739", "N710", "N711", "N719", "N72", "N760", "N761", "N762", "N763", "N771", "N750", "N751", "N764", "N766", "N770", "N7681", "N759", "N765", "N7689", "L03019", "L03029", "L03039", "L03049", "K122", "L03211", "L03212", "L03221", "L03222", "L03319", "L03329", "L03119", "L03129", "L03317", "L03811", "L03818", "L03891", "L03898", "L0390", "L0391", "L049", "L080", "L88", "L0889", "L980", "E832", "L089", "M8610", "M8620", "M86119", "M86219", "M86129", "M86229", "M86139", "M86239", "M86149", "M86249", "M86159", "M86259", "M86169", "M86269", "M86179", "M86279", "M8618", "M8628", "M8619", "M8629", "M8660", "M86619", "M86629", "M86639", "M86642", "M86659", "M86669", "M86679", "M8668", "M8669", "M869", "M4620", "M8960", "M89619", "M89629", "M89639", "M89649", "M89659", "M89669", "M89679", "M8968", "M8969", "M9080", "M90819", "M90829", "M90839", "M90849", "M90859", "M90869", "M90879", "M9088", "M9089", "M4630",
    "K630", "K750", "K751", "K810", "N390", "M0010", "M009", "M00019", "M00119", "M00219", "M00819", "M00029", "M00129", "M00229", "M00829", "M00039", "M00139", "M00239", "M00839", "M00049", "M00149", "M00249", "M00849", "M00059", "M00159", "M00259", "M00859", "M00069", "M00169", "M00269", "M00869", "M00079", "M00179", "M00279", "M00879", "M0008", "M0018", "M0028", "M0088", "M0009", "M0019", "M0029", "M0089", "R7881", "T8579XA", "T826XXA", "T827XXA", "T8351XA", "T8359XA", "T836XXA", "T8450XA", "T8460XA", "T847XXA", "T8571XA", "T814XXA", "K6811", "T80219A", "T80211A", "T80212A", "T8022XA", "T8029XA", "T880XXA",
    "J441", "K5712", "K5713", "K5732", "K5733", "K631"
] # fmt: skip

# ICD-10 codes for organ dysfunction
ANGUS_ICD10_DYSFUNCTION_CODES = [
    "F05", "F062", "F060", "F0630", "F064", "F061", "F53", "F068", "I951", "I9589", "I953", "I952", "I9581", "I959", "K7200", "K762", "N170", "N171", "N172", "N178", "N179",
    "D65", "D688", "D689", "D6951", "D6959", "D696", "G931", "G9340", "G9341", "G9349", "I6783", "K763", "R579", "R570", "R6521", "R571", "R578"
] # fmt: skip

# ICD-10 codes for explicit sepsis
ANGUS_ICD10_EXPLICIT_SEPSIS = ["R6521", "R6520"]

# ICD-10 codes for mechanical ventilation
MECHVENT_ICD10_CODES = ["T423X1A", "T423X2A", "T423X3A", "T423X4A", "T426X1A"] # fmt: skip


# region Angus Sepsis
def ANGUS_SEPSIS(
    diagnoses: Optional[pl.LazyFrame] = None,
    procedures: Optional[pl.LazyFrame] = None,
    patient_information: Optional[pl.LazyFrame] = None,
    mechanical_ventilation: Optional[pl.LazyFrame] = None,
) -> pl.LazyFrame:
    """
    Calculate Angus sepsis classification from diagnosis and procedure codes.

    Implements the Angus et al. (2001) definition of sepsis, which requires:
    - Infection: ICD codes indicating bacterial or fungal infection
    - Organ Dysfunction: ICD codes indicating acute organ dysfunction
    - Severe Sepsis: Infection + Organ Dysfunction OR Infection + Mechanical Ventilation
    - Explicit Sepsis: Sepsis diagnosis codes

    Arguments
    ---------
        diagnoses : pl.LazyFrame, optional
            Diagnosis data. Loaded automatically if None.
        procedures : pl.LazyFrame, optional
            Procedure data. Loaded automatically if None.
        patient_information : pl.LazyFrame, optional
            Patient/stay-level information. Loaded automatically if None.
        mechanical_ventilation : pl.LazyFrame, optional
            Mechanical ventilation concept data. Loaded automatically if None.

    Returns
    -------
        pl.LazyFrame
            Angus sepsis classification with columns:
            - Global ICU Stay ID
            - Mechanical Ventilation: Patient received invasive ventilation
            - Infection: Evidence of infection
            - Organ Dysfunction: Evidence of organ dysfunction
            - Explicit Sepsis: ICD codes for sepsis
            - Angus Sepsis: Meets Angus criteria for sepsis
    """
    if diagnoses is None:
        diagnoses = get_diagnoses()
    if procedures is None:
        procedures = get_procedures()
    if patient_information is None:
        patient_information = get_patient_information()
    if mechanical_ventilation is None:
        mechanical_ventilation = get_ventilation()

    # Validate all required data is available
    required = {
        "diagnoses": diagnoses,
        "procedures": procedures,
        "patient_information": patient_information,
        "mechanical_ventilation": mechanical_ventilation,
    }
    _validate_required_data("Angus sepsis", required)

    diagnoses = _to_lazy(diagnoses)
    procedures = _to_lazy(procedures)
    patient_information = _to_lazy(patient_information)
    mechanical_ventilation = _to_lazy(mechanical_ventilation)

    IDs = patient_information.select(
        "Global Person ID", "Global Hospital Stay ID", "Global ICU Stay ID"
    )

    # Merge with patient information to fill in missing IDs
    procedures = pl.concat(
        [
            procedures.filter(pl.col("Global ICU Stay ID").is_null())
            .drop("Global ICU Stay ID")
            .join(
                IDs,
                on=["Global Person ID", "Global Hospital Stay ID"],
                how="left",
                coalesce=True,
            ),
            procedures.filter(pl.col("Global ICU Stay ID").is_not_null()),
        ],
        how="diagonal_relaxed",
    )

    # Get whether the patient received any mechanical ventilation
    MECHVENT_ICD = (
        procedures.filter(
            pl.when(pl.col("Procedure ICD Code Version") == 9)
            .then(pl.col("Procedure ICD Code").is_in(MECHVENT_ICD9_CODES))
            .otherwise(pl.col("Procedure ICD Code").is_in(MECHVENT_ICD10_CODES))
        )
        .select("Global ICU Stay ID")
        .unique()
        .with_columns(pl.lit(True).alias("Mechanical Ventilation"))
    )

    MECHANICAL_VENTILATION = (
        mechanical_ventilation.filter(
            pl.col("Ventilation Type").is_in(
                ["invasive ventilation", "tracheostomy"]
            ),
            pl.col("Ventilation End Relative to Admission (seconds)") > 0,
            # ICD-coded procedures are only available for MIMIC-III and MIMIC-IV
            ~pl.col("Global ICU Stay ID").str.starts_with("mimic"),
        )
        .select("Global ICU Stay ID")
        .unique()
        .with_columns(pl.lit(True).alias("Mechanical Ventilation"))
    )

    return (
        diagnoses.join(
            pl.concat(
                [MECHVENT_ICD, MECHANICAL_VENTILATION],
                how="vertical",
            ),
            on="Global ICU Stay ID",
            how="left",
            coalesce=True,
        )
        .with_columns(
            # Check for infection
            pl.when(pl.col("Diagnosis ICD Code Version (source)") == "ICD-9")
            .then(
                pl.when(
                    pl.col("Diagnosis ICD-9 Code")
                    .str.slice(0, 3)
                    .is_in(ANGUS_ICD9_3_DIGIT_INFECTION_CODES)
                )
                .then(1)
                .when(
                    pl.col("Diagnosis ICD-9 Code")
                    .str.slice(0, 4)
                    .is_in(ANGUS_ICD9_4_DIGIT_INFECTION_CODES)
                )
                .then(1)
                .when(
                    pl.col("Diagnosis ICD-9 Code")
                    .str.slice(0, 5)
                    .is_in(ANGUS_ICD9_5_DIGIT_INFECTION_CODES)
                )
                .then(1)
            )
            .otherwise(
                pl.col("Diagnosis ICD-10 Code").is_in(
                    ANGUS_ICD10_INFECTION_CODES
                )
            )
            .alias("Infection"),
            # Check for organ dysfunction
            pl.when(pl.col("Diagnosis ICD Code Version (source)") == "ICD-9")
            .then(
                pl.when(
                    pl.col("Diagnosis ICD-9 Code")
                    .str.slice(0, 3)
                    .is_in(ANGUS_ICD9_3_DIGIT_DYSFUNCTION_CODES)
                )
                .then(1)
                .when(
                    pl.col("Diagnosis ICD-9 Code")
                    .str.slice(0, 4)
                    .is_in(ANGUS_ICD9_4_DIGIT_DYSFUNCTION_CODES)
                )
                .then(1)
            )
            .otherwise(
                pl.col("Diagnosis ICD-10 Code").is_in(
                    ANGUS_ICD10_DYSFUNCTION_CODES
                )
            )
            .alias("Organ Dysfunction"),
            # Check for explicit sepsis
            pl.when(pl.col("Diagnosis ICD Code Version (source)") == "ICD-9")
            .then(
                pl.when(
                    pl.col("Diagnosis ICD-9 Code").is_in(
                        ANGUS_ICD9_EXPLICIT_SEPSIS
                    )
                ).then(1)
            )
            .otherwise(
                pl.col("Diagnosis ICD-10 Code").is_in(
                    ANGUS_ICD10_EXPLICIT_SEPSIS
                )
            )
            .alias("Explicit Sepsis"),
        )
        .group_by("Global ICU Stay ID")
        .agg(pl.all().sum().cast(bool))
        .with_columns(
            pl.when(pl.col("Explicit Sepsis"))
            .then(True)
            .when(
                pl.all_horizontal(
                    pl.col("Infection"), pl.col("Organ Dysfunction")
                )
            )
            .then(True)
            .when(
                pl.all_horizontal(
                    pl.col("Infection"),
                    pl.col("Mechanical Ventilation"),
                )
            )
            .then(True)
            .otherwise(False)
            .alias("Angus Sepsis"),
        )
        .select(
            "Global ICU Stay ID",
            "Mechanical Ventilation",
            "Infection",
            "Organ Dysfunction",
            "Explicit Sepsis",
            "Angus Sepsis",
        )
    )


# endregion
__all__ = ["ANGUS_SEPSIS"]
