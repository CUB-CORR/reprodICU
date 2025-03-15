# based on https://github.com/nus-mornin-lab/oxygenation_kc/blob/master/data-extraction/eICU/eicu_oxygen_therapy.sql

import polars as pl
from helpers.A_extract.A_extract_eicu import EICUExtractor
from helpers.MAGIC_CONCEPTS.MAGIC_CONCEPTS import MAGIC_CONCEPTS


class VENTILATION_DURATION_eICU(MAGIC_CONCEPTS):
    def __init__(self, paths, datasets, MAX_VENTILATION_PAUSE_HOURS):
        super().__init__(paths, datasets)
        self.MAX_VENTILATION_PAUSE_HOURS = MAX_VENTILATION_PAUSE_HOURS

    def VENTILATION_DURATION(self) -> pl.DataFrame:
        print("MAGIC_CONCEPTS: Ventilation Duration - eICU")

        # # region ricu
        # ##############################
        # # Respiratory Care
        # # based on ricu code
        # ##############################

        # RESPIRATORY_CARE = pl.scan_csv(
        #     self.eicu_paths.respiratoryCare_path,
        #     null_values=[
        #         "",
        #         # see https://github.com/MIT-LCP/eicu-code/issues/49 for why 0 is NULL
        #         "0",
        #     ],
        # ).cast(
        #     {
        #         "ventstartoffset": int,
        #         "ventendoffset": int,
        #         "priorventstartoffset": int,
        #         "priorventendoffset": int,
        #     }
        # )
        # RESPIRATORY_CARE_VENT = RESPIRATORY_CARE.filter(
        #     # can't set prior end before the time
        #     pl.col("ventendoffset").le(pl.col("respcarestatusoffset"))
        # ).select(
        #     "patientunitstayid",
        #     "airwaytype",
        #     "ventstartoffset",
        #     "ventendoffset",
        # )
        # RESPIRATORY_CARE_PRIOR = (
        #     RESPIRATORY_CARE.filter(
        #         # can't set prior end before the time
        #         pl.col("priorventendoffset").le(pl.col("respcarestatusoffset"))
        #     )
        #     .select(
        #         "patientunitstayid",
        #         "airwaytype",
        #         "priorventstartoffset",
        #         "priorventendoffset",
        #     )
        #     .rename(
        #         {
        #             "priorventstartoffset": "ventstartoffset",
        #             "priorventendoffset": "ventendoffset",
        #         }
        #     )
        # )
        # RESPIRATORY_CARE = (
        #     pl.concat(
        #         [RESPIRATORY_CARE_VENT, RESPIRATORY_CARE_PRIOR],
        #         how="vertical",
        #     )
        #     .with_columns(
        #         pl.col("airwaytype")
        #         .replace_strict(
        #             self.global_helpers.load_mapping(
        #                 self.eicu_paths.resp_airwaytype_mapping_path
        #             ),
        #             default=None,
        #         )
        #         .alias("Ventilation Type"),
        #         # reltimes in eICU are in minutes
        #         (pl.col("ventstartoffset") * 60).alias(
        #             "Ventilation Start Relative to Admission (seconds)"
        #         ),
        #         (pl.col("ventendoffset") * 60).alias(
        #             "Ventilation End Relative to Admission (seconds)"
        #         ),
        #     )
        #     .select(
        #         "patientunitstayid",
        #         "Ventilation Type",
        #         "Ventilation Start Relative to Admission (seconds)",
        #         "Ventilation End Relative to Admission (seconds)",
        #     )
        #     .collect(streaming=True)
        # )

        # # region treatment
        # ##############################
        # # Treatment
        # # based on nothing else
        # ##############################

        # eicu_extractor = EICUExtractor(self.paths, DEMO=False)
        # TREATMENT = (
        #     pl.scan_csv(self.eicu_paths.treatment_path)
        #     .with_columns(
        #         pl.when(
        #             pl.col("treatmentstring").str.starts_with(
        #                 "pulmonary|ventilation and oxygenation|"
        #             )
        #             | pl.col("treatmentstring").str.starts_with(
        #                 "surgery|pulmonary therapies|"
        #             )
        #             | pl.col("treatmentstring").str.starts_with(
        #                 "toxicology|drug overdose|"
        #             )
        #         )
        #         .then(
        #             pl.when(
        #                 pl.col("treatmentstring").str.contains_any(
        #                     ["CPAP/PEEP therapy", "non-invasive ventilation"]
        #                 )
        #             )
        #             .then(pl.lit("non-invasive ventilation"))
        #             .when(
        #                 pl.col("treatmentstring").str.contains_any(
        #                     ["mechanical ventilation", "ventilator weaning"]
        #                 )
        #             )
        #             .then(pl.lit("invasive ventilation"))
        #             .when(
        #                 pl.col("treatmentstring").str.contains(
        #                     "ventilator weaning"
        #                 )
        #             )
        #             .then(pl.lit("weaning"))
        #         )
        #         .alias("treatmentstring"),
        #     )
        #     .pipe(eicu_extractor._extract_treatments_helper)
        #     .rename(
        #         {
        #             self.column_names[
        #                 "procedure_start_col"
        #             ]: "Ventilation Start Relative to Admission (seconds)",
        #             self.column_names[
        #                 "procedure_end_col"
        #             ]: "Ventilation End Relative to Admission (seconds)",
        #             self.column_names[
        #                 "procedure_description_col"
        #             ]: "Ventilation Type",
        #         }
        #     )
        #     .filter(
        #         pl.col("Ventilation Type").is_in(
        #             [
        #                 "invasive ventilation",
        #                 "non-invasive ventilation",
        #                 "weaning",
        #             ]
        #         )
        #     )
        #     .select(
        #         pl.col("ICU Stay ID").alias("patientunitstayid"),
        #         "Ventilation Type",
        #         "Ventilation Start Relative to Admission (seconds)",
        #         "Ventilation End Relative to Admission (seconds)",
        #     )
        #     .collect(streaming=True)
        # )

        # region NUS Mornin Lab
        ##############################
        # Respiratory Charting
        # based on linked SQL script
        ##############################

        PATIENT = pl.scan_csv(self.eicu_paths.patient_path).select(
            "patientunitstayid", "unitdischargeoffset"
        )

        NURSECHARTING = pl.scan_csv(self.eicu_paths.nurseCharting_path).select(
            pl.col("patientunitstayid"),
            pl.col("nursingchartoffset").alias("charttime"),
            pl.col("nursingchartvalue").str.to_lowercase().alias("string"),
            pl.lit(None).alias("activeupondischarge"),
        )
        RESPIRATORYCHARTING = pl.scan_csv(
            self.eicu_paths.respiratoryCharting_path
        ).select(
            pl.col("patientunitstayid"),
            pl.col("respchartoffset").alias("charttime"),
            pl.col("respchartvaluelabel").str.to_lowercase().alias("string"),
            pl.col("respchartvalue"),
            pl.lit(None).alias("activeupondischarge"),
        )
        RESPIRATORYCHARTING_OXYGEN_DEVICE = (
            pl.scan_csv(self.eicu_paths.respiratoryCharting_path)
            .select(
                pl.col("patientunitstayid"),
                pl.col("respchartoffset").alias("charttime"),
                pl.col("respchartvalue").str.to_lowercase().alias("string"),
                pl.lit(None).alias("activeupondischarge"),
            )
            .filter(
                pl.col("string").is_in(
                    [
                        "o2 device",
                        "respiratory device",
                        "ventilator type",
                        "oxygen delivery method",
                    ]
                )
            )
        )
        TREATMENT = pl.scan_csv(self.eicu_paths.treatment_path).select(
            pl.col("patientunitstayid"),
            pl.col("treatmentoffset").alias("charttime"),
            pl.col("treatmentstring").str.to_lowercase().alias("string"),
            pl.col("activeupondischarge"),
        )

        # Extract the type of oxygen therapy.
        # The categories are invasive ventilation,
        # noninvasive ventilation, and supplemental oxygen.
        # `oxygen_therapy_type = -1` indicates oxygen therapy,
        # i.e. more oxygen than in room air is administered.
        OXYGEN_THERAPY_KNOWN_TYPE = (
            pl.concat(
                [
                    NURSECHARTING,
                    RESPIRATORYCHARTING,
                    RESPIRATORYCHARTING_OXYGEN_DEVICE,
                    TREATMENT,
                ],
                how="diagonal_relaxed",
            )
            .filter(pl.col("charttime") > -60)
            .with_columns(
                # Invasive ventilation
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
                        [
                            "set vt", "sputum", "rsbi", "tube", "ett",
                            "endotracheal", "tracheal suctioning",
                            "tracheostomy", "reintubation",
                            "assist controlled", "volume controlled",
                            "pressure controlled", "trach collar"
                        ] # fmt: skip
                    )
                )
                .then(4)
                # Noninvasive ventilation
                .when(
                    pl.col("string").is_in(["bi-pap", "ambubag"])
                    | pl.col("string").str.contains_any(
                        [
                            "ipap", "niv", "epap", "mask leak",
                            "volume assured", "non-invasive ventilation",
                            "cpap"
                        ] # fmt: skip
                    )
                )
                .then(3)
                # Either invasive or noninvasive ventilation
                .when(
                    pl.col("string").is_in(
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
                        ] # fmt: skip
                    )
                    | (
                        pl.col("string").str.contains("vent")
                        & pl.col("string").str.contains("hyperventilat").not_()
                    )
                    | pl.col("string").str.contains_any(
                        [
                            "tidal", "flow rate", "minute volume",
                            "leak", "pressure support", "peep",
                            "tidal volume"
                        ] # fmt: skip
                    )
                )
                .then(2)
                # Supplemental oxygen
                .when(
                    pl.col("string").is_in(
                        [
                            "t-piece", "blow-by", "oxyhood", "nc",
                            "oxymizer", "hfnc", "oximizer", "high flow",
                            "oxymask", "nch", "hi flow", "hiflow", "hhfnc",
                            "nasal canula", "face tent", "high flow mask",
                            "aerosol mask", "venturi mask", "cool aerosol mask",
                            "simple mask", "face mask"
                        ] # fmt: skip
                    )
                    | pl.col("string").str.contains_any(
                        [
                            "nasal cannula", "non-rebreather",
                            "nasal mask", "face tent"
                        ] # fmt: skip
                    )
                )
                .then(1)
                # Oxygen therapy but unknown what type
                .when(
                    pl.col("string").is_in(
                        [
                            "pressure support", "rr spont", "ps",
                            "insp cycle off (%)", "trach mask/collar"
                        ] # fmt: skip
                    )
                    | pl.col("string").str.contains_any(
                        ["spontaneous", "oxygen therapy"]
                    )
                )
                .then(0)
                # Supplemental oxygen therapy
                # i.e. more oxygen than in room air is administered.
                .when(pl.col("string").is_in(["lpm o2"]))
                .then(-1)
                # fraction of inspired oxygen (fiO2) outside of [.2, .22] and [20, 22]
                # indicates oxygen therapy
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
            .select(
                "patientunitstayid",
                "charttime",
                "oxygen_therapy_type",
                "activeupondischarge",
            )
        )

        OXYGEN_THERAPY_UNKNOWN_TYPE = (
            pl.scan_csv(self.eicu_paths.nurseCharting_path)
            .filter(
                pl.col("nursingchartoffset") > -60,
                pl.col("nursingchartcelltypevallabel").str.contains("O2 L/%"),
                pl.col("nursingchartvalue")
                .cast(int, strict=False)
                .is_between(0, 100, closed="right"),
            )
            .select(
                pl.col("patientunitstayid"),
                pl.col("nursingchartoffset").alias("charttime"),
                pl.lit(-1).alias("oxygen_therapy_type"),
                pl.lit(None).alias("activeupondischarge"),
            )
        )

        OXYGEN_THERAPY = (
            pl.concat(
                [
                    OXYGEN_THERAPY_KNOWN_TYPE,
                    OXYGEN_THERAPY_UNKNOWN_TYPE,
                ],
                how="vertical_relaxed",
            )
            # if oxygen_therapy_type is NULL, then the record does not correspond with oxygen therapy
            .filter(pl.col("oxygen_therapy_type").is_not_null())
            # ensure charttime is unique
            .group_by("patientunitstayid", "charttime")
            .agg(
                pl.col("oxygen_therapy_type").max(),
                pl.col("activeupondischarge").max(),
                # pl.when(pl.col("oxygen_therapy_type") == -1)
                # .then(1)
                # .otherwise(0)
                # .sum()
                # .alias("supp_oxygen"),
            )
            .with_columns(
                # this carries over the previous charttime which had an oxygen therapy event
                pl.col("charttime")
                .shift(1)
                .over(
                    partition_by=["patientunitstayid"],
                    order_by=["charttime"],
                )
                .alias("charttime_lag"),
            )
            # If the time since the last oxygen therapy event is more than MAX_VENTILATION_PAUSE_HOURS hours,
            # we consider that ventilation had ended in between.
            # That is, the next ventilation record corresponds to a new ventilation session.
            # MAX_VENTILATION_PAUSE_HOURS is set to 24 hours in the original code.
            .with_columns(
                pl.when(
                    pl.col("charttime")
                    .sub(pl.col("charttime_lag"))
                    .gt(pl.duration(hours=self.MAX_VENTILATION_PAUSE_HOURS))
                )
                .then(1)
                # No lag can be computed for the very first record
                .when(pl.col("charttime_lag").is_null())
                .then(None)
                .otherwise(0)
                .alias("newvent"),
            )
            # create a cumulative sum of the instances of new ventilation
            # this results in a monotonic integer assigned to each instance of ventilation
            .with_columns(
                pl.col("newvent")
                .cum_sum()
                .over(
                    partition_by=["patientunitstayid"],
                    order_by=["charttime"],
                )
                .alias("ventnum")
            )
            # now we convert CHARTTIME of ventilator settings into durations
            # create the durations for each oxygen therapy instance
            # we only keep the first oxygen therapy instance
            .join(PATIENT, how="left", on="patientunitstayid", coalesce=True)
            .group_by("patientunitstayid", "ventnum")
            .agg(
                pl.col("charttime").min().alias("vent_start"),
                # If activeupondischarge, then the unit discharge time is vent_end
                pl.when(
                    pl.col("activeupondischarge").max().cast(bool),
                    # vent_end cannot be later than the unit discharge time.
                    # However, unitdischargeoffset often seems too low.
                    # So, we only use it if it yields and extension of the
                    # ventilation time from ventsettings.
                    (pl.col("charttime").max() + 60)
                    < pl.col("unitdischargeoffset").max(),
                ).then(pl.col("unitdischargeoffset").max())
                # End time is currently a charting time
                # Since these are usually recorded hourly, ventilation is actually longer.
                # We therefore add 60 minutes to the last time.
                .otherwise(pl.col("charttime").max() + 60).alias("vent_end"),
                pl.col("oxygen_therapy_type").max(),
            )
            .with_columns(
                pl.col("oxygen_therapy_type")
                .replace_strict(
                    {
                        4: "invasive ventilation",
                        3: "non-invasive ventilation",
                        2: "unknown",
                        1: "supplemental oxygen",
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
            .collect(streaming=True)
        )

        return (
            pl.concat(
                [
                    # RESPIRATORY_CARE,
                    # TREATMENT,
                    OXYGEN_THERAPY,
                ],
                how="vertical_relaxed",
            )
            .unique()
            .pipe(self._add_global_id_stay_id, "eicu-", "patientunitstayid")
            .lazy()
        )

    # region helpers
    def _add_global_id_stay_id(
        self, data, source_dataset, stay_id_col
    ) -> pl.LazyFrame:
        return data.with_columns(
            # add global ICU stay ID
            pl.concat_str([pl.lit(source_dataset), pl.col(stay_id_col)]).alias(
                self.column_names["global_icu_stay_id_col"]
            )
        ).drop(stay_id_col)

    # endregion
