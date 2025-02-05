# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script extracts the so called MAGIC CONCEPT "Ventilation Duration" directly from the source datasets.
# The MAGIC CONCEPTS are a set of concepts that are based on the concept dict used in the ricu R package and/or
# available prewritten code snippets where indicated.

import os
import polars as pl

from helpers.MAGIC_CONCEPTS.MAGIC_CONCEPTS import MAGIC_CONCEPTS
from helpers.A_extract.A_extract_eicu import EICUExtractor


class VENTILATION_DURATION(MAGIC_CONCEPTS):
    def __init__(self, paths, datasets):
        super().__init__(paths, datasets)

    def VENTILATION_DURATION(self) -> pl.DataFrame:
        """
        Returns the magic concept VENTILATION_DURATION

        Description:
        This concept is used to determine whether a patient received any antibiotics during the ICU stay.

        Returns a DataFrame with the following columns:
        - ICU stay ID
        - Ventilation Type, one of
            - tracheostomy
            - invasive ventilation
            - non-invasive ventilation
            - weaning
            - other
            - unknown
        - Ventilation Start Relative to Admission (seconds)
        - Ventilation End Relative to Admission (seconds)
        - Ventilation Duration (hours)

        :return: DataFrame
        :rtype: pl.DataFrame
        """

        MAX_VENTILATION_PAUSE_HOURS = 8

        SECONDS_IN_1H = 60 * 60
        SECONDS_IN_1D = 24 * 60 * 60

        # region eICU
        print("MAGIC_CONCEPTS: Ventilation Duration - eICU")
        eicu_extractor = EICUExtractor(self.paths, DEMO=False)
        eicu_RESPIRATORY_CARE = pl.scan_csv(
            self.eicu_paths.respiratoryCare_path,
            null_values=[
                "",
                # see https://github.com/MIT-LCP/eicu-code/issues/49 for why 0 is NULL
                "0",
            ],
        ).cast(
            {
                "ventstartoffset": int,
                "ventendoffset": int,
                "priorventstartoffset": int,
                "priorventendoffset": int,
            }
        )
        eicu_RESPIRATORY_CARE_VENT = eicu_RESPIRATORY_CARE.filter(
            # can't set prior end before the time
            pl.col("ventendoffset").le(pl.col("respcarestatusoffset"))
        ).select(
            "patientunitstayid",
            "airwaytype",
            "ventstartoffset",
            "ventendoffset",
        )
        eicu_RESPIRATORY_CARE_PRIOR = (
            eicu_RESPIRATORY_CARE.filter(
                # can't set prior end before the time
                pl.col("priorventendoffset").le(pl.col("respcarestatusoffset"))
            )
            .select(
                "patientunitstayid",
                "airwaytype",
                "priorventstartoffset",
                "priorventendoffset",
            )
            .rename(
                {
                    "priorventstartoffset": "ventstartoffset",
                    "priorventendoffset": "ventendoffset",
                }
            )
        )
        eicu_RESPIRATORY_CARE = (
            pl.concat(
                [eicu_RESPIRATORY_CARE_VENT, eicu_RESPIRATORY_CARE_PRIOR],
                how="vertical",
            )
            .with_columns(
                pl.col("airwaytype")
                .replace_strict(
                    self.global_helpers.load_mapping(
                        self.eicu_paths.resp_airwaytype_mapping_path
                    ),
                    default=None,
                )
                .alias("Ventilation Type"),
                # reltimes in eICU are in minutes
                (pl.col("ventstartoffset") * 60).alias(
                    "Ventilation Start Relative to Admission (seconds)"
                ),
                (pl.col("ventendoffset") * 60).alias(
                    "Ventilation End Relative to Admission (seconds)"
                ),
            )
            .select(
                "patientunitstayid",
                "Ventilation Type",
                "Ventilation Start Relative to Admission (seconds)",
                "Ventilation End Relative to Admission (seconds)",
            )
        )

        # based on https://github.com/nus-mornin-lab/oxygenation_kc/blob/master/data-extraction/eICU/eicu_oxygen_therapy.sql
        eicu_RESPIRATORY_CHARTING = (
            pl.scan_csv(self.eicu_paths.respiratoryCharting_path)
            .select(
                "patientunitstayid",
                "respchartoffset",
                "respcharttypecat",
                "respchartvaluelabel",
                "respchartvalue",
            )
            .with_columns(
                # oxygen device from respchart
                pl.when(
                    pl.col("respchartvaluelabel")
                    .str.to_lowercase()
                    .is_in(
                        [
                            "o2 device",
                            "respiratory device",
                            "ventilator type",
                            "oxygen delivery method",
                        ]
                    )
                )
                .then(pl.col("respchartvalue").str.to_lowercase())
                .otherwise(pl.col("respchartvaluelabel").str.to_lowercase())
                .alias("string")
            )
            .with_columns(
                pl.when(
                    pl.col("string").is_in(
                        [
                            "plateau pressure",
                            "postion at lip",
                            "position at lip",
                            "pressure control",
                        ]
                    )
                    | pl.col("string").str.contains_any(
                        # fmt: off
                        [
                            "set vt", "sputum", "rsbi", "tube", "ett",
                            "endotracheal", "tracheal suctioning",
                            "tracheostomy", "reintubation",
                            "assist controlled", "volume controlled",
                            "pressure controlled", "trach collar"
                        ]
                        # fmt: on
                    )
                )
                .then(4)
                .when(
                    pl.col("string").is_in(["bi-pap", "ambubag"])
                    | pl.col("string").str.contains_any(
                        # fmt: off
                        [
                            "ipap", "niv", "epap", "mask leak",
                            "volume assured", "non-invasive ventilation",
                            "cpap"
                        ]
                        # fmt: on
                    )
                )
                .then(3)
                .when(
                    pl.col("string").is_in(
                        # fmt: off
                        [
                            "flowtrigger", "peep", "tv/kg ibw",
                            "mean airway pressure", "peak insp. pressure",
                            "exhaled mv", "exhaled tv (machine)",
                            "exhaled tv (patient)", "flow sensitivity",
                            "peak flow", "f total", "pressure to trigger ps",
                            "adult con setting set rr", "adult con setting set vt",
                            "vti", "exhaled vt", "adult con alarms hi press alarm",
                            "mve", "respiratory phase", "inspiratory pressure, set",
                            "a1: high exhaled vt",
                            "set fraction of inspired oxygen (fio2)",
                            "insp flow (l/min)", "adult con setting spont exp vt",
                            "spont tv", "pulse ox results vt",
                            "vt spontaneous (ml)", "peak pressure", "ltv1200",
                            "tc"
                        ]
                        # fmt: on
                    )
                    | (
                        pl.col("string").str.contains("vent")
                        & pl.col("string").str.contains("hyperventilat").not_()
                    )
                    | pl.col("string").str.contains_any(
                        # fmt: off
                        [
                            "tidal", "flow rate", "minute volume",
                            "leak", "pressure support", "peep",
                            "tidal volume"
                        ]
                        # fmt: on
                    )
                )
                .then(2)
                .when(
                    pl.col("string").is_in(
                        # fmt: off
                        [
                            "t-piece", "blow-by", "oxyhood", "nc",
                            "oxymizer", "hfnc", "oximizer", "high flow",
                            "oxymask", "nch", "hi flow", "hiflow", "hhfnc",
                            "nasal canula", "face tent", "high flow mask",
                            "aerosol mask", "venturi mask", "cool aerosol mask",
                            "simple mask", "face mask"
                        ]
                        # fmt: on
                    )
                    | pl.col("string").str.contains_any(
                        # fmt: off
                        [
                            "nasal cannula", "non-rebreather",
                            "nasal mask", "face tent"
                        ]
                        # fmt: on
                    )
                )
                .then(1)
                .when(
                    pl.col("string").is_in(
                        # fmt: off
                        [
                            "pressure support", "rr spont", "ps",
                            "insp cycle off (%)", "trach mask/collar"
                        ]
                        # fmt: on
                    )
                    | pl.col("string").str.contains_any(
                        ["spontaneous", "oxygen therapy"]
                    )
                )
                .then(0)
                .when(pl.col("string").is_in(["lpm o2"]))
                .then(-1)
                # fraction of inspired oxygen (fiO2) outside of [.2, .22] and [20, 22] indicates oxygen therapy
                .when(pl.col("string").is_in(["fio2", "fio2 (%)"]))
                .then(
                    pl.when(
                        pl.col("respchartvalue")
                        .cast(float, strict=False)
                        .is_between(0.22, 1, closed="right")
                    )
                    .then(-1)
                    .when(
                        pl.col("respchartvalue")
                        .cast(float, strict=False)
                        .gt(22)
                    )
                    .then(-1)
                    .otherwise(0)
                )
                .otherwise(None)
                .alias("oxygen_therapy_type"),
            )
            # if oxygen_therapy_type is NULL, then the record does not correspond with oxygen therapy
            .filter(pl.col("oxygen_therapy_type").is_not_null())
            # ensure charttime is unique
            .group_by("patientunitstayid", "respchartoffset")
            .agg(pl.max("oxygen_therapy_type"))
            .with_columns(
                # this carries over the previous charttime which had an oxygen therapy event
                pl.col("respchartoffset")
                .shift(1)
                .over("patientunitstayid")
                .sort_by("respchartoffset")
                .alias("respchartoffset_lag"),
            )
            # If the time since the last oxygen therapy event is more than MAX_VENTILATION_PAUSE_HOURS hours,
            # we consider that ventilation had ended in between.
            # That is, the next ventilation record corresponds to a new ventilation session.
            # MAX_VENTILATION_PAUSE_HOURS is set to 24 hours in the original code.
            .with_columns(
                pl.when(
                    pl.col("respchartoffset")
                    .sub(pl.col("respchartoffset_lag"))
                    .gt(pl.duration(hours=MAX_VENTILATION_PAUSE_HOURS))
                )
                .then(1)
                .when(pl.col("respchartoffset_lag").is_null())
                .then(None)
                .otherwise(0)
                .alias("newvent"),
            )
            # create a cumulative sum of the instances of new ventilation
            # this results in a monotonic integer assigned to each instance of ventilation
            .with_columns(
                pl.sum("newvent")
                .over("patientunitstayid")
                .sort_by("respchartoffset")
                .alias("ventnum")
            )
            # now we convert CHARTTIME of ventilator settings into durations
            # create the durations for each oxygen therapy instance
            # we only keep the first oxygen therapy instance
            .group_by("patientunitstayid", "ventnum")
            .agg(
                pl.min("respchartoffset").alias("vent_start"),
                pl.max("respchartoffset").alias("vent_end"),
                pl.max("oxygen_therapy_type"),
            )
            .with_columns(
                pl.col("oxygen_therapy_type")
                .replace_strict(
                    {
                        4: "invasive ventilation",
                        3: "non-invasive ventilation",
                        2: "unknown",
                    },
                    default=None,
                )
                .alias("Ventilation Type"),
                # reltimes in eICU are in minutes
                (pl.col("vent_start") * 60).alias(
                    "Ventilation Start Relative to Admission (seconds)"
                ),
                (pl.col("vent_end") * 60).alias(
                    "Ventilation End Relative to Admission (seconds)"
                ),
            )
            .filter(pl.col("oxygen_therapy_type").is_not_null())
            .select(
                "patientunitstayid",
                "Ventilation Type",
                "Ventilation Start Relative to Admission (seconds)",
                "Ventilation End Relative to Admission (seconds)",
            )
        )

        eicu_TREATMENT = (
            pl.scan_csv(self.eicu_paths.treatment_path)
            .with_columns(
                pl.when(
                    pl.col("treatmentstring").str.starts_with(
                        "pulmonary|ventilation and oxygenation|"
                    )
                    | pl.col("treatmentstring").str.starts_with(
                        "surgery|pulmonary therapies|"
                    )
                    | pl.col("treatmentstring").str.starts_with(
                        "toxicology|drug overdose|"
                    )
                )
                .then(
                    pl.when(
                        pl.col("treatmentstring").str.contains_any(
                            ["CPAP/PEEP therapy", "non-invasive ventilation"]
                        )
                    )
                    .then(pl.lit("non-invasive ventilation"))
                    .when(
                        pl.col("treatmentstring").str.contains_any(
                            ["mechanical ventilation", "ventilator weaning"]
                        )
                    )
                    .then(pl.lit("invasive ventilation"))
                    .when(
                        pl.col("treatmentstring").str.contains(
                            "ventilator weaning"
                        )
                    )
                    .then(pl.lit("weaning"))
                )
                .alias("treatmentstring"),
            )
            .pipe(eicu_extractor._extract_treatments_helper)
            .rename(
                {
                    self.column_names[
                        "procedure_start_col"
                    ]: "Ventilation Start Relative to Admission (seconds)",
                    self.column_names[
                        "procedure_end_col"
                    ]: "Ventilation End Relative to Admission (seconds)",
                    self.column_names[
                        "procedure_description_col"
                    ]: "Ventilation Type",
                }
            )
            .filter(
                pl.col("Ventilation Type").is_in(
                    [
                        "invasive ventilation",
                        "non-invasive ventilation",
                        "weaning",
                    ]
                )
            )
            .select(
                pl.col("ICU Stay ID").alias("patientunitstayid"),
                "Ventilation Type",
                "Ventilation Start Relative to Admission (seconds)",
                "Ventilation End Relative to Admission (seconds)",
            )
        )

        eicu_VENTILATION_DURATION = pl.concat(
            [
                eicu_RESPIRATORY_CARE.collect(streaming=True),
                eicu_RESPIRATORY_CHARTING.collect(streaming=True),
                eicu_TREATMENT.collect(streaming=True),
            ],
            how="vertical_relaxed",
        ).pipe(self._add_global_id_stay_id, "eicu-", "patientunitstayid")

        # region HiRID
        print("MAGIC_CONCEPTS: Ventilation Duration - HiRID")

        # get admission times for HiRID
        hirid_ADMISSIONTIMES = (
            pl.scan_csv(self.hirid_paths.general_table_path)
            .select("patientid", "admissiontime")
            .with_columns(
                pl.col("patientid").cast(str),
                pl.col("admissiontime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
        )

        # Scan all files in the timeseries folder
        hirid_VENTILATION_DURATION = pl.LazyFrame()

        for file in os.listdir(self.hirid_paths.timeseries_path):
            hirid_timeseries_AIRWAYTYPE = (
                pl.scan_parquet(self.hirid_paths.timeseries_path + file)
                .select("datetime", "patientid", "value", "variableid")
                .cast({"datetime": str, "patientid": str})
                # Filter for ventilation IDs
                .filter(
                    pl.col("variableid")
                    == self.ricu_mappings.ricu_concept_dict["mech_vent"][
                        "sources"
                    ]["hirid"][0]["ids"]
                )
                .drop("variableid")
                # replace ventilation concepts
                .with_columns(
                    pl.col("value")
                    .cast(int)
                    .cast(str)
                    .replace(
                        {
                            "1": "invasive ventilation",
                            "2": "tracheostomy",
                            "3": "non-invasive ventilation",
                            "4": "non-invasive ventilation",
                            "5": "non-invasive ventilation",
                            "6": "other",  # TODO: check if this is correct
                        }
                    )
                    .alias("Ventilation Type"),
                    pl.col("datetime").str.to_datetime("%Y-%m-%d %H:%M:%S%.9f"),
                )
                # Make datetime relative to admission in seconds
                .join(hirid_ADMISSIONTIMES, on="patientid", how="left")
                .with_columns(
                    (pl.col("datetime") - pl.col("admissiontime"))
                    .dt.total_seconds()
                    .alias("Ventilation Start Relative to Admission (seconds)")
                )
                .drop("admissiontime", "datetime", "value")
                # Rename columns
            )

            hirid_timeseries_VENTMODE = (
                pl.scan_parquet(self.hirid_paths.timeseries_path + file)
                .select("datetime", "patientid", "value", "variableid")
                .cast({"datetime": str, "patientid": str})
                # Filter for ventilation IDs
                .filter(pl.col("variableid") == 3845)  # Ventilator mode
                .drop("variableid")
                # replace ventilation concepts
                .with_columns(
                    pl.when(pl.col("value").gt(1))
                    .then(pl.lit("active"))
                    .otherwise(pl.lit("inactive"))
                    .alias("Ventilator Mode"),
                    pl.col("datetime").str.to_datetime("%Y-%m-%d %H:%M:%S%.9f"),
                )
                # Make datetime relative to admission in seconds
                .join(hirid_ADMISSIONTIMES, on="patientid", how="left")
                .with_columns(
                    (pl.col("datetime") - pl.col("admissiontime"))
                    .dt.total_seconds()
                    .alias("Ventilation Start Relative to Admission (seconds)")
                )
                .drop("admissiontime", "datetime", "value")
                # Rename columns
            )

            hirid_timeseries = (
                pl.concat(
                    [hirid_timeseries_AIRWAYTYPE, hirid_timeseries_VENTMODE],
                    how="align",
                )
                .select(pl.all().forward_fill())
                .sort(
                    "patientid",
                    "Ventilation Start Relative to Admission (seconds)",
                )
                # drop rows where both columns are staying the same
                .filter(
                    pl.col("Ventilator Mode").ne_missing(
                        pl.col("Ventilator Mode")
                        .shift(1)
                        .over("patientid", "Ventilation Type")
                    )
                    | pl.col("Ventilator Mode").ne_missing(
                        pl.col("Ventilator Mode")
                        .shift(-1)
                        .over("patientid", "Ventilation Type")
                    )
                    | pl.col("Ventilation Type")
                    .shift(1)
                    .over("patientid")
                    .is_null()
                    | pl.col("Ventilation Type")
                    .shift(-1)
                    .over("patientid")
                    .is_null()
                )
                .with_columns(
                    pl.col("Ventilation Start Relative to Admission (seconds)")
                    .shift(-1)
                    .over("patientid")
                    .alias("Ventilation End Relative to Admission (seconds)")
                )
                .drop_nulls("Ventilation End Relative to Admission (seconds)")
                .filter(pl.col("Ventilator Mode") == "active")
            )

            hirid_VENTILATION_DURATION = pl.concat(
                [hirid_VENTILATION_DURATION, hirid_timeseries],
                how="diagonal_relaxed",
            )

        hirid_VENTILATION_DURATION = hirid_VENTILATION_DURATION.pipe(
            self._add_global_id_stay_id, "hirid-", "patientid"
        ).collect(streaming=True)
        # endregion

        # region MIMIC-III
        print("MAGIC_CONCEPTS: Ventilation Duration - MIMIC3")
        # based on https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/concepts/durations/ventilation_classification.sql
        # and https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/concepts/durations/ventilation_durations.sql

        # fmt: off
        mimic3_chartevents_ventilation_ids = [
            720, 223849,  # vent mode
            223848,  # vent type
            445, 448, 449, 450, 1340, 1486, 1600, 224687,  # minute volume
            639, 654, 681, 682, 683, 684, 224685, 224684, 224686,  # tidal volume
            218, 436, 535, 444, 224697, 224695, 224696, 224746, 224747,  # High/Low/Peak/Mean ("RespPressure")
            221, 1, 1211, 1655, 2000, 226873, 224738, 224419, 224750, 227187,  # Insp pressure
            543,  # PlateauPressure
            5865, 5866, 224707, 224709, 224705, 224706,  # APRV pressure
            60, 437, 505, 506, 686, 220339, 224700,  # PEEP
            3459,  # high pressure relief
            501, 502, 503, 224702,  # PCV
            223, 667, 668, 669, 670, 671, 672,  # TCPCV
            224701,  # PSVlevel
        ]
        mimic3_chartevents_extubation_ids = [
            640 # extubated
        ]
        mimic3_chartevents_niv_ids = [
            468,  # O2 Delivery Device#2
            469,  # O2 Delivery Mode
            470,  # O2 Flow (lpm)
            471,  # O2 Flow (lpm) #2
            227287,  # O2 Flow (additional cannula)
            226732,  # O2 Delivery Device(s)
            223834,  # O2 Flow
        ]
        mimic3_chartevents_more_ids = [
            467 # O2 Delivery Device
        ]
        mimic3_chartevents_all_ids = (
            mimic3_chartevents_ventilation_ids
            + mimic3_chartevents_extubation_ids
            + mimic3_chartevents_niv_ids
            + mimic3_chartevents_more_ids
        )

        mimic3_id226732_oxygen_therapy = [
            "Nasal cannula", "Face tent", "Aerosol-cool", "Trach mask ",
            "High flow neb", "Non-rebreather", "Venti mask ", "Medium conc mask ",
            "T-piece", "High flow nasal cannula", "Ultrasonic neb", "Vapomist",
        ]
        mimic3_id467_oxygen_therapy = [
            "Cannula", "Nasal Cannula", "Face Tent", "Aerosol-Cool",
            "Trach Mask", "Hi Flow Neb", "Non-Rebreather", "Venti Mask",
            "Medium Conc Mask", "Vapotherm", "T-Piece", "Hood", "Hut",
            "TranstrachealCat", "Heated Neb", "Ultrasonic Neb",
        ]
        # fmt: on

        mimic3_procedureevents_mv_extubation_ids = [
            227194,  # Extubation
            225468,  # Unplanned Extubation (patient-initiated)
            225477,  # Unplanned Extubation (non-patient initiated)
        ]

        # get admission times for MIMIC-III
        mimic3_ADMISSIONTIMES = (
            pl.scan_csv(self.mimic3_paths.icustays_path)
            .select("ICUSTAY_ID", "INTIME")
            .with_columns(
                pl.col("INTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
        )

        # ventilation_classification.sql
        mimic3_CHARTEVENTS_VENTILATION_CLASSIFICATION = (
            pl.scan_csv(
                self.mimic3_paths.chartevents_path,
                schema_overrides={"VALUE": str},
            )
            .filter(
                pl.col("ITEMID").is_in(mimic3_chartevents_all_ids),
                pl.col("ERROR").ne_missing(1),
                pl.col("VALUE").is_not_null(),
            )
            .select("ICUSTAY_ID", "CHARTTIME", "ITEMID", "VALUE")
            .with_columns(
                pl.col("CHARTTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                # case statement determining whether it is an instance of mech vent
                pl.when(
                    pl.col("ITEMID") == 720,
                    pl.col("VALUE") != ("Other/Remarks"),
                )
                .then(1)
                .when(pl.col("ITEMID") == 223848, pl.col("VALUE") != "Other")
                .then(1)
                .when(pl.col("ITEMID") == 223849)
                .then(1)
                .when(pl.col("ITEMID") == 467, pl.col("VALUE") == "Ventilator")
                .then(1)
                .when(
                    pl.col("ITEMID").is_in(mimic3_chartevents_ventilation_ids)
                )
                .then(1)
                .otherwise(0)
                .alias("MechVent"),
                # initiation of oxygen therapy indicates the ventilation has ended
                pl.when(
                    pl.col("ITEMID") == 226732,
                    pl.col("VALUE").is_in(mimic3_id226732_oxygen_therapy),
                )
                .then(1)
                .when(
                    pl.col("ITEMID") == 467,
                    pl.col("VALUE").is_in(mimic3_id467_oxygen_therapy),
                )
                .then(1)
                .otherwise(0)
                .alias("OxygenTherapy"),
                # extubated indicates ventilation event has ended
                pl.when(
                    pl.col("ITEMID") == 640,
                    pl.col("VALUE").is_in(["Extubated", "Self Extubation"]),
                )
                .then(1)
                .otherwise(0)
                .alias("Extubated"),
                pl.when(
                    pl.col("ITEMID") == 640,
                    pl.col("VALUE") == "Self Extubation",
                )
                .then(1)
                .otherwise(0)
                .alias("SelfExtubated"),
            )
            .group_by("ICUSTAY_ID", "CHARTTIME")
            .agg(
                pl.max("MechVent"),
                pl.max("OxygenTherapy"),
                pl.max("Extubated"),
                pl.max("SelfExtubated"),
            )
        )

        mimic3_PROCEDUREEVENTS_MV_VENTILATION_CLASSIFICATION = (
            pl.scan_csv(self.mimic3_paths.procedureevents_mv_path)
            .select("ICUSTAY_ID", "STARTTIME", "ITEMID")
            .rename({"STARTTIME": "CHARTTIME"})
            # Filter for ventilation IDs
            .filter(
                pl.col("ITEMID").is_in(mimic3_procedureevents_mv_extubation_ids)
            )
            .with_columns(
                pl.col("CHARTTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.lit(0).alias("MechVent"),
                pl.lit(0).alias("OxygenTherapy"),
                pl.lit(1).alias("Extubated"),
                pl.when(pl.col("ITEMID") == 225468)
                .then(1)
                .otherwise(0)
                .alias("SelfExtubated"),
            )
            .drop("ITEMID")
        )

        # ventilation_durations.sql
        mimic3_VENTILATION_EVENTS = (
            pl.concat(
                [
                    mimic3_CHARTEVENTS_VENTILATION_CLASSIFICATION.collect(
                        streaming=True
                    ),
                    mimic3_PROCEDUREEVENTS_MV_VENTILATION_CLASSIFICATION.collect(
                        streaming=True
                    ),
                ],
                how="vertical",
            )
            .lazy()
            .sort("ICUSTAY_ID", "CHARTTIME")
            .unique()
            .with_columns(
                pl.col("CHARTTIME")
                .shift(1)
                .over("ICUSTAY_ID", "MechVent")
                .alias("CHARTTIME_LAG"),
            )
            # if this is a mechanical ventilation event, we calculate the time since the last event
            .with_columns(
                pl.col("Extubated")
                .shift(1)
                .over(
                    pl.col("ICUSTAY_ID"),
                    pl.when(
                        (pl.col("MechVent") == 1) | (pl.col("Extubated") == 1)
                    )
                    .then(1)
                    .otherwise(0),
                )
                .alias("ExtubatedLag"),
            )
            # now we determine if the current mech vent event is a "new", i.e. they've just been intubated
            .with_columns(
                pl.when(pl.col("ExtubatedLag") == 1)
                .then(1)
                .when(pl.col("MechVent") == 0, pl.col("OxygenTherapy") == 1)
                .then(1)
                .when(
                    pl.col("CHARTTIME")
                    > pl.col("CHARTTIME_LAG").add(
                        # is 8 hours in original code
                        pl.duration(hours=MAX_VENTILATION_PAUSE_HOURS)
                    )
                )
                .then(1)
                .otherwise(0)
                .alias("NewVent")
            )
            # create a cumulative sum of the instances of new ventilation
            .with_columns(
                pl.when((pl.col("NewVent") == 1) | (pl.col("Extubated") == 1))
                .then(pl.sum("NewVent").over("ICUSTAY_ID"))
                .otherwise(None)
                .alias("VentNum")
            )
            # create the durations for each mechanical ventilation instance
            .group_by("ICUSTAY_ID", "VentNum")
            .agg(
                pl.min("CHARTTIME").alias("STARTTIME"),
                pl.max("CHARTTIME").alias("ENDTIME"),
            )
            # Make datetime relative to admission in seconds
            .join(mimic3_ADMISSIONTIMES, on="ICUSTAY_ID", how="left")
            .with_columns(
                (pl.col("STARTTIME") - pl.col("INTIME"))
                .dt.total_seconds()
                .alias("Ventilation Start Relative to Admission (seconds)"),
                (pl.col("ENDTIME") - pl.col("INTIME"))
                .dt.total_seconds()
                .alias("Ventilation End Relative to Admission (seconds)"),
                pl.lit("invasive ventilation").alias("Ventilation Type"),
            )
            .drop("INTIME", "STARTTIME", "ENDTIME", "VentNum")
        )

        mimic3_VENTILATION_PROCEDURES = (
            pl.scan_csv(self.mimic3_paths.procedureevents_mv_path)
            .select("ICUSTAY_ID", "STARTTIME", "ENDTIME", "ITEMID")
            .join(mimic3_ADMISSIONTIMES, on="ICUSTAY_ID", how="left")
            # Filter for ventilation IDs
            .filter(
                pl.col("ITEMID").is_in(
                    self.ricu_mappings.ricu_concept_dict["mech_vent"][
                        "sources"
                    ]["miiv"][0]["ids"]
                )
            )
            .cast({"ITEMID": str})
            # replace ventilation concepts
            .with_columns(
                pl.col("ITEMID")
                .replace(
                    {
                        225792: "invasive ventilation",
                        225794: "non-invasive ventilation",
                    }
                )
                .cast(str)
                .alias("Ventilation Type")
            )
            # Make datetime relative to admission in seconds
            .with_columns(
                pl.col("STARTTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("ENDTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
            .with_columns(
                (pl.col("STARTTIME") - pl.col("INTIME"))
                .truediv(pl.duration(seconds=1))
                .alias("Ventilation Start Relative to Admission (seconds)"),
                (pl.col("ENDTIME") - pl.col("INTIME"))
                .truediv(pl.duration(seconds=1))
                .alias("Ventilation End Relative to Admission (seconds)"),
            )
            .filter(pl.col("STARTTIME").ne(pl.col("ENDTIME")))
            .drop("INTIME", "STARTTIME", "ENDTIME", "ITEMID")
        )

        mimic3_VENTILATION_DURATION = (
            pl.concat(
                [
                    mimic3_VENTILATION_EVENTS.collect(streaming=True),
                    mimic3_VENTILATION_PROCEDURES.collect(streaming=True),
                ],
                how="diagonal_relaxed",
            )
            .unique()
            .pipe(self._add_global_id_stay_id, "mimic3-", "ICUSTAY_ID")
        )

        # endregion

        # region MIMIC-IV
        print("MAGIC_CONCEPTS: Ventilation Duration - MIMIC4")
        # based on https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/measurement/ventilator_setting.sql
        # and https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/measurement/oxygen_delivery.sql
        # and https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/treatment/ventilation.sql

        # fmt: off
        mimic4_vent_setting_chartevents_ids = [
            224688,  # Respiratory Rate (Set)
            224689,  # Respiratory Rate (spontaneous)
            224690,  # Respiratory Rate (Total)
            224687,  # minute volume
            224685, 224684, 224686,  # tidal volume
            224696,  # PlateauPressure
            220339, 224700,  # PEEP
            223835,  # fio2
            223849,  # vent mode
            229314,  # vent mode (Hamilton)
            223848,  # vent type
            224691,  # Flow Rate (L)
        ]
        mimic4_o2_flow_chartevents_ids = [
            223834, # o2 flow
            227582, # bipap o2 flow
            227287, # additional o2 flow
        ]
        mimic4_o2_delivery_chartevents_ids = [
            226732,  # oxygen delivery device(s)
        ]
        # fmt: on

        # get admission times for MIMIC-IV
        mimic4_ADMISSIONTIMES = (
            pl.scan_csv(self.mimic4_paths.icustays_path)
            .select("stay_id", "intime")
            .with_columns(
                pl.col("intime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
        )

        # common chartevents scan
        mimic4_chartevents = pl.scan_csv(
            self.mimic4_paths.chartevents_path,
            schema_overrides={"value": str},
        ).filter(
            pl.col("itemid").is_in(
                mimic4_vent_setting_chartevents_ids
                + mimic4_o2_flow_chartevents_ids
                + mimic4_o2_delivery_chartevents_ids
            )
        )

        # oxygen_therapy.sql
        mimic4_CHARTEVENTS_OXYGEN_FLOW = (
            mimic4_chartevents.select(
                "subject_id",
                "stay_id",
                "charttime",
                "storetime",
                "itemid",
                "value",
                "valuenum",
            )
            .sort("subject_id", "charttime")
            .filter(pl.col("itemid").is_in(mimic4_o2_flow_chartevents_ids))
            .with_columns(
                pl.col("charttime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("storetime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                # merge o2 flows into a single row
                pl.when(pl.col("itemid") == 226732)
                .then(223834)
                .otherwise(pl.col("itemid"))
                .alias("itemid"),
            )
            .with_columns(
                pl.int_range(pl.len())
                .over("subject_id", "charttime", "itemid")
                .sort_by("storetime")
                .alias("rn"),
            )
        )
        mimic4_CHARTEVENTS_OXYGEN_DELIVERY = (
            mimic4_chartevents.select(
                "subject_id",
                "stay_id",
                "charttime",
                "storetime",
                "itemid",
                "value",
            )
            .sort("subject_id", "charttime")
            .filter(pl.col("itemid").is_in(mimic4_o2_delivery_chartevents_ids))
            .rename({"value": "o2_device"})
            .with_columns(
                pl.col("charttime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("storetime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
            .with_columns(
                pl.int_range(pl.len())
                .over("subject_id", "charttime", "itemid")
                .sort_by("storetime")
                .alias("rn"),
            )
        )
        mimic4_OXYGEN_THERAPY = (
            mimic4_CHARTEVENTS_OXYGEN_DELIVERY.join(
                mimic4_CHARTEVENTS_OXYGEN_FLOW.filter(pl.col("rn") == 1),
                on=["subject_id", "stay_id", "charttime", "itemid"],
                how="outer",
                coalesce=True,
            )
            .group_by("subject_id", "charttime")
            .agg(
                pl.max("stay_id"),
                pl.when(pl.col("itemid") == 223834)
                .then(pl.col("valuenum"))
                .otherwise(None)
                .max()
                .alias("o2_flow"),
                pl.when(pl.col("itemid") == 227287)
                .then(pl.col("valuenum"))
                .otherwise(None)
                .max()
                .alias("o2_flow_additional"),
                # contrary to mimic-code template we only need the first o2_device
                pl.when(pl.col("rn") == 1)
                .then(pl.col("o2_device"))
                .otherwise(None)
                .max()
                .alias("o2_device"),
            )
            .collect(streaming=True)
        )

        # ventilator_setting.sql
        mimic4_VENTILATOR_SETTINGS = (
            mimic4_chartevents.select(
                "subject_id",
                "stay_id",
                "charttime",
                "itemid",
                "value",
                "valuenum",
            )
            .filter(
                pl.col("itemid").is_in(mimic4_vent_setting_chartevents_ids),
                pl.col("value").is_not_null(),
            )
            .with_columns(
                pl.col("charttime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                # fio2 cleaning
                pl.when(pl.col("itemid") == 223835).then(
                    pl.when(pl.col("valuenum") >= 0.20, pl.col("valuenum") <= 1)
                    .then(pl.col("valuenum") * 100)
                    .when(pl.col("valuenum") > 1, pl.col("valuenum") < 20)
                    .then(None)
                    .when(pl.col("valuenum") >= 20, pl.col("valuenum") <= 100)
                    .then(pl.col("valuenum"))
                    .otherwise(None)
                )
                # peep cleaning
                .when(pl.col("itemid").is_in([220339, 224700]))
                .then(
                    pl.when(pl.col("valuenum") > 100)
                    .then(None)
                    .when(pl.col("valuenum") < 0)
                    .then(None)
                    .otherwise(pl.col("valuenum"))
                )
                .otherwise(pl.col("valuenum"))
                .alias("valuenum"),
            )
            .group_by("subject_id", "charttime")
            .agg(
                pl.max("stay_id"),
                pl.when(pl.col("itemid") == 224688)
                .then(pl.col("valuenum"))
                .otherwise(None)
                .max()
                .alias("respiratory_rate_set"),
                pl.when(pl.col("itemid") == 224690)
                .then(pl.col("valuenum"))
                .otherwise(None)
                .max()
                .alias("respiratory_rate_total"),
                pl.when(pl.col("itemid") == 224689)
                .then(pl.col("valuenum"))
                .otherwise(None)
                .max()
                .alias("respiratory_rate_spontaneous"),
                pl.when(pl.col("itemid") == 224687)
                .then(pl.col("valuenum"))
                .otherwise(None)
                .max()
                .alias("minute_volume"),
                pl.when(pl.col("itemid") == 224684)
                .then(pl.col("valuenum"))
                .otherwise(None)
                .max()
                .alias("tidal_volume_set"),
                pl.when(pl.col("itemid") == 224685)
                .then(pl.col("valuenum"))
                .otherwise(None)
                .max()
                .alias("tidal_volume_observed"),
                pl.when(pl.col("itemid") == 224686)
                .then(pl.col("valuenum"))
                .otherwise(None)
                .max()
                .alias("tidal_volume_spontaneous"),
                pl.when(pl.col("itemid") == 224696)
                .then(pl.col("valuenum"))
                .otherwise(None)
                .max()
                .alias("plateau_pressure"),
                pl.when(pl.col("itemid").is_in([220339, 224700]))
                .then(pl.col("valuenum"))
                .otherwise(None)
                .max()
                .alias("peep"),
                pl.when(pl.col("itemid") == 223835)
                .then(pl.col("valuenum"))
                .otherwise(None)
                .max()
                .alias("fio2"),
                pl.when(pl.col("itemid") == 224691)
                .then(pl.col("valuenum"))
                .otherwise(None)
                .max()
                .alias("flow_rate"),
                pl.when(pl.col("itemid") == 223849)
                .then(pl.col("value"))
                .otherwise(None)
                .max()
                .alias("ventilator_mode"),
                pl.when(pl.col("itemid") == 229314)
                .then(pl.col("value"))
                .otherwise(None)
                .max()
                .alias("ventilator_mode_hamilton"),
                pl.when(pl.col("itemid") == 223848)
                .then(pl.col("value"))
                .otherwise(None)
                .max()
                .alias("ventilator_type"),
            )
            .collect(streaming=True)
        )

        # ventilation.sql
        mimic4_VENT_IDS = (
            pl.concat(
                [
                    mimic4_OXYGEN_THERAPY.select("stay_id", "charttime"),
                    mimic4_VENTILATOR_SETTINGS.select("stay_id", "charttime"),
                ],
                how="vertical",
            )
            .unique()
            .lazy()
        )

        mimic4_CHARTEVENTS_VENTILATION = (
            mimic4_VENT_IDS.join(
                mimic4_OXYGEN_THERAPY.lazy(),
                on=["stay_id", "charttime"],
                how="left",
                coalesce=True,
            )
            .join(
                mimic4_VENTILATOR_SETTINGS.lazy(),
                on=["stay_id", "charttime"],
                how="left",
                coalesce=True,
            )
            .with_columns(
                # case statement determining the type of intervention
                # done in order of priority: trach > mech vent > NIV > o2 / hiflow
                pl.coalesce(
                    pl.col("ventilator_mode"),
                    pl.col("ventilator_mode_hamilton"),
                ).alias("vent_mode"),
                pl.when(
                    pl.col("o2_device").is_in(
                        ["Tracheostomy tube", "Trach mask"]
                    )
                )
                .then(pl.lit("tracheostomy"))
                .when(
                    (pl.col("o2_device") == "Endotracheal tube")
                    | pl.col("ventilator_mode").is_in(
                        # fmt: off
                        [
                            "(S) CMV", "APRV", "APRV/Biphasic+ApnPress",
                            "APRV/Biphasic+ApnVol", "APV (cmv)", "Ambient",
                            "Apnea Ventilation", "CMV", "CMV/ASSIST",
                            "CMV/ASSIST/AutoFlow", "CMV/AutoFlow", "CPAP/PPS",
                            "CPAP/PSV", "CPAP/PSV+Apn TCPL",
                            "CPAP/PSV+ApnPres", "CPAP/PSV+ApnVol", "MMV",
                            "MMV/AutoFlow", "MMV/PSV", "MMV/PSV/AutoFlow",
                            "P-CMV", "PCV+", "PCV+/PSV", "PCV+Assist",
                            "PRES/AC", "PRVC/AC", "PRVC/SIMV", "PSV/SBT",
                            "SIMV", "SIMV/AutoFlow", "SIMV/PRES", "SIMV/PSV",
                            "SIMV/PSV/AutoFlow", "SIMV/VOL", "SYNCHRON MASTER",
                            "SYNCHRON SLAVE", "VOL/AC", 
                        ]
                        # fmt: on
                    )
                    | pl.col("ventilator_mode_hamilton").is_in(
                        # fmt: off
                        [
                            "APRV", "APV (cmv)", "Ambient", "(S) CMV", "P-CMV",
                            "SIMV", "APV (simv)", "P-SIMV", "VS", "ASV"
                        ]
                        # fmt: on
                    )
                )
                .then(pl.lit("invasive ventilation"))
                .when(
                    pl.col("o2_device").is_in(["Bipap mask", "CPAP mask"])
                    | pl.col("ventilator_mode_hamilton").is_in(
                        ["DuoPaP", "NIV", "NIV-ST"]
                    )
                )
                .then(pl.lit("non-invasive ventilation"))
                .when(
                    pl.col("o2_device").is_in(
                        # fmt: off
                        [
                            "High flow nasal cannula", # HFNC not extra
                            "Non-rebreather", "Face tent", "Aerosol-cool",
                            "Venti mask ", "Medium conc mask ",
                            "Ultrasonic neb", "Vapomist", "Oxymizer",
                            "High flow neb", "Nasal cannula",
                        ]
                        # fmt: on
                    )
                )
                .then(pl.lit("supplemental oxygen"))
                .when(pl.col("o2_device") == "None")
                .then(pl.lit("None"))
                .otherwise(None)
                .alias("ventilation_status"),
            )
            .with_columns(
                # carry over the previous charttime which had the same state
                pl.col("charttime")
                .shift(1)
                .over("stay_id", "ventilation_status")
                .sort_by("charttime")
                .alias("charttime_lag"),
                # bring back the next charttime, regardless of the state
                # this will be used as the end time for state transitions
                pl.col("charttime")
                .shift(-1)
                .over("stay_id")
                .sort_by("charttime")
                .alias("charttime_lead"),
                pl.col("ventilation_status")
                .shift(1)
                .over("stay_id")
                .sort_by("charttime")
                .alias("ventilation_status_lag"),
            )
            .with_columns(
                # now we determine if the current ventilation status is "new",
                # or continuing the previous event
                pl.when(pl.col("ventilation_status_lag").is_null())
                .then(1)
                .when(
                    (pl.col("charttime") - pl.col("charttime_lag")).gt(
                        # is 14 hours in original code
                        pl.duration(hours=MAX_VENTILATION_PAUSE_HOURS)
                    )
                )
                .then(1)
                .when(
                    pl.col("ventilation_status_lag")
                    != pl.col("ventilation_status")
                )
                .then(1)
                .otherwise(0)
                .alias("new_ventilation_event"),
            )
            .with_columns(
                pl.sum("new_ventilation_event")
                .over("stay_id")
                .sort_by("charttime")
                .alias("vent_seq")
            )
            .group_by("stay_id", "vent_seq")
            .agg(
                pl.min("charttime").alias("starttime"),
                pl.when(
                    pl.col("charttime_lead").is_null()
                    | (pl.col("charttime") - pl.col("charttime_lag")).gt(
                        # is 14 hours in original code
                        pl.duration(hours=MAX_VENTILATION_PAUSE_HOURS)
                    )
                )
                .then(pl.col("charttime"))
                .otherwise(pl.col("charttime_lead"))
                .max()
                .alias("endtime"),
                pl.col("ventilation_status").max(),
            )
            # Make datetime relative to admission in seconds
            .join(mimic4_ADMISSIONTIMES, on="stay_id", how="left")
            .with_columns(
                (pl.col("starttime") - pl.col("intime"))
                .dt.total_seconds()
                .alias("Ventilation Start Relative to Admission (seconds)"),
                (pl.col("endtime") - pl.col("intime"))
                .dt.total_seconds()
                .alias("Ventilation End Relative to Admission (seconds)"),
                pl.col("ventilation_status").alias("Ventilation Type"),
            )
            .drop("intime", "starttime", "endtime", "vent_seq")
        )

        mimic4_PROCEDUREEVENTS_VENTILATION = (
            pl.scan_csv(self.mimic4_paths.procedureevents_path)
            .select("stay_id", "starttime", "endtime", "itemid")
            .join(mimic4_ADMISSIONTIMES, on="stay_id", how="left")
            # Filter for ventilation IDs
            .filter(
                pl.col("itemid").is_in(
                    self.ricu_mappings.ricu_concept_dict["mech_vent"][
                        "sources"
                    ]["miiv"][0]["ids"]
                )
            )
            .cast({"itemid": str})
            # replace ventilation concepts
            .with_columns(
                pl.col("itemid")
                .replace(
                    {
                        225792: "invasive ventilation",
                        225794: "non-invasive ventilation",
                    }
                )
                .cast(str)
                .alias("Ventilation Type")
            )
            # Make datetime relative to admission in seconds
            .with_columns(
                pl.col("starttime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("endtime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
            .with_columns(
                (pl.col("starttime") - pl.col("intime"))
                .truediv(pl.duration(seconds=1))
                .alias("Ventilation Start Relative to Admission (seconds)"),
                (pl.col("endtime") - pl.col("intime"))
                .truediv(pl.duration(seconds=1))
                .alias("Ventilation End Relative to Admission (seconds)"),
            )
            .drop("intime", "starttime", "endtime", "itemid")
        )

        mimic4_VENTILATION_DURATION = (
            pl.concat(
                [
                    mimic4_CHARTEVENTS_VENTILATION.collect(streaming=True),
                    mimic4_PROCEDUREEVENTS_VENTILATION.collect(streaming=True),
                ],
                how="diagonal_relaxed",
            )
            .unique()
            .pipe(self._add_global_id_stay_id, "mimic4-", "stay_id")
        )

        # endregion

        # region SICdb
        print("MAGIC_CONCEPTS: Ventilation Duration - SICdb")

        sicdb_ADMISSION_TIMES = pl.scan_csv(self.sicdb_paths.cases_path).select(
            "CaseID", "ICUOffset"
        )

        sicdb_VENTILATION_DURATION = (
            pl.scan_csv(self.sicdb_paths.data_range_path)
            .join(sicdb_ADMISSION_TIMES, on="CaseID", how="left")
            # Filter for ventilation IDs
            .filter(pl.col("DataID").is_in([720, 3043]))
            .with_columns(
                pl.col("DataID")
                .cast(str)
                .replace(
                    {"720": "invasive ventilation", "3043": "tracheostomy"}
                )
                .alias("Ventilation Type"),
                pl.col("Offset").alias(
                    "Ventilation Start Relative to Admission (seconds)"
                ),
                pl.col("OffsetEnd").alias(
                    "Ventilation End Relative to Admission (seconds)"
                ),
            )
            .drop("DataID", "Offset", "OffsetEnd")
            .pipe(self._add_global_id_stay_id, "sicdb-", "CaseID")
            .collect(streaming=True)
        )

        # endregion

        # region UMCdb
        print("MAGIC_CONCEPTS: Ventilation Duration - UMCdb")

        umcdb_ADMISSION_TIMES = pl.scan_parquet(
            self.umcdb_paths.admissions_path
        ).select("admissionid", "admittedat")

        umcdb_VENTILATION_DURATION = (
            pl.scan_parquet(self.umcdb_paths.processitems_path)
            .join(umcdb_ADMISSION_TIMES, on="admissionid", how="left")
            # Filter for ventilation IDs
            .filter(
                pl.col("itemid").is_in(
                    self.ricu_mappings.ricu_concept_dict["mech_vent"][
                        "sources"
                    ]["aumc"][0]["ids"]
                    + [9671]  # CPAP
                )
            )
            .drop("itemid")
            # replace ventilation concepts
            .with_columns(
                pl.col("item")
                .replace(
                    {
                        "Beademen": "invasive ventilation",
                        "Beademen non-invasief": "non-invasive ventilation",
                        "CPAP": "non-invasive ventilation",
                        "Tracheostoma": "tracheostomy",
                    }
                )
                .cast(str)
                .alias("item")
            )
            # Make datetime relative to admission in seconds
            .with_columns(
                pl.duration(
                    milliseconds=(pl.col("start") - pl.col("admittedat"))
                )
                .dt.total_seconds()
                .alias("start"),
                pl.duration(
                    milliseconds=(pl.col("stop") - pl.col("admittedat"))
                )
                .dt.total_seconds()
                .alias("stop"),
            )
            .drop("admittedat")
            # Rename columns
            .rename(
                {
                    "item": "Ventilation Type",
                    "start": "Ventilation Start Relative to Admission (seconds)",
                    "stop": "Ventilation End Relative to Admission (seconds)",
                }
            )
            .pipe(self._add_global_id_stay_id, "umcdb-", "admissionid")
            .collect(streaming=True)
        )
        # endregion

        # region ALL
        print("MAGIC_CONCEPTS: Ventilation Duration")

        VENTILATION_TYPE_ENUM = pl.Enum(
            [
                "tracheostomy",
                "invasive ventilation",
                "non-invasive ventilation",
                "weaning",
                "supplemental oxygen",
                "unknown",
                "other",
            ]
        )
        VENTILATION_DURATION = (
            pl.concat(
                [
                    eicu_VENTILATION_DURATION,
                    hirid_VENTILATION_DURATION,
                    mimic3_VENTILATION_DURATION,
                    mimic4_VENTILATION_DURATION,
                    sicdb_VENTILATION_DURATION,
                    umcdb_VENTILATION_DURATION,
                ],
                how="diagonal_relaxed",
            )
            .filter(
                pl.col("Ventilation Start Relative to Admission (seconds)").lt(
                    pl.col("Ventilation End Relative to Admission (seconds)")
                ),
                pl.col("Ventilation End Relative to Admission (seconds)").gt(
                    -self.global_vars.PRE_ICU_TIMESERIES_DAYS_CUTOFF
                    * (SECONDS_IN_1D)
                ),
            )
            .unique()
            .select(
                "Global ICU Stay ID",
                "Ventilation Type",
                "Ventilation Start Relative to Admission (seconds)",
                "Ventilation End Relative to Admission (seconds)",
            )
            # .cast({"Ventilation Type": VENTILATION_TYPE_ENUM})
            .group_by(
                "Global ICU Stay ID",
                "Ventilation Start Relative to Admission (seconds)",
                "Ventilation End Relative to Admission (seconds)",
            )
            .agg(pl.col("Ventilation Type").max())
            .with_columns(
                (
                    pl.col("Ventilation End Relative to Admission (seconds)")
                    - pl.col(
                        "Ventilation Start Relative to Admission (seconds)"
                    )
                )
                .truediv(SECONDS_IN_1H)
                .round(2)
                .alias("Ventilation Duration (hours)")
            )
            .lazy()
        )

        return VENTILATION_DURATION

    # region helpers
    def _add_global_id_stay_id(
        self, data, source_dataset, stay_id_col
    ) -> pl.DataFrame:
        return data.with_columns(
            # add global ICU stay ID
            pl.concat_str([pl.lit(source_dataset), pl.col(stay_id_col)]).alias(
                self.column_names["global_icu_stay_id_col"]
            )
        ).drop(stay_id_col)

    # endregion
