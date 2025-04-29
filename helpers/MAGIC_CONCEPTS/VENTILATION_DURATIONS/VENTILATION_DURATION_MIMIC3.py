# based on https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/concepts/durations/ventilation_classification.sql
# and https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/concepts/durations/ventilation_durations.sql

import polars as pl
from helpers.MAGIC_CONCEPTS.MAGIC_CONCEPTS import MAGIC_CONCEPTS


class VENTILATION_DURATION_MIMIC3(MAGIC_CONCEPTS):
    def __init__(self, paths, datasets, MAX_VENTILATION_PAUSE_HOURS):
        super().__init__(paths, datasets)
        self.MAX_VENTILATION_PAUSE_HOURS = MAX_VENTILATION_PAUSE_HOURS

    def VENTILATION_DURATION(self) -> pl.DataFrame:
        print("MAGIC_CONCEPTS: Ventilation Duration - MIMIC3")

        # get admission times for MIMIC-III
        ADMISSIONTIMES = (
            pl.scan_csv(self.mimic3_paths.icustays_path)
            .select("ICUSTAY_ID", "INTIME")
            .with_columns(
                pl.col("INTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
            .collect()
        )

        ##############################################################################
        # ventilation_classification.sql
        ##############################################################################
        # the below are settings used to indicate ventilation
        chartevents_ventilation_ids = [
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
        ] # fmt: skip

        # the below are settings used to indicate extubation
        chartevents_extubation_ids = [640]  # extubated

        # the below indicate oxygen/NIV, i.e. the end of a mechanical vent event
        chartevents_niv_ids = [
            468,  # O2 Delivery Device#2
            469,  # O2 Delivery Mode
            470,  # O2 Flow (lpm)
            471,  # O2 Flow (lpm) #2
            227287,  # O2 Flow (additional cannula)
            226732,  # O2 Delivery Device(s)
            223834,  # O2 Flow
        ]

        # used in both oxygen + vent calculation
        chartevents_more_ids = [467]  # O2 Delivery Device

        chartevents_all_ids = (
            chartevents_ventilation_ids
            + chartevents_extubation_ids
            + chartevents_niv_ids
            + chartevents_more_ids
        )

        # fmt: off
        id226732_oxygen_therapy = [
            "Nasal cannula", "Face tent", "Aerosol-cool", "Trach mask ",
            "High flow neb", "Non-rebreather", "Venti mask ", "Medium conc mask ",
            "T-piece", "High flow nasal cannula", "Ultrasonic neb", "Vapomist",
        ]
        id467_oxygen_therapy = [
            "Cannula", "Nasal Cannula", "Face Tent", "Aerosol-Cool",
            "Trach Mask", "Hi Flow Neb", "Non-Rebreather", "Venti Mask",
            "Medium Conc Mask", "Vapotherm", "T-Piece", "Hood", "Hut",
            "TranstrachealCat", "Heated Neb", "Ultrasonic Neb",
        ]
        # fmt: on

        procedureevents_mv_extubation_ids = [
            227194,  # Extubation
            225468,  # Unplanned Extubation (patient-initiated)
            225477,  # Unplanned Extubation (non-patient initiated)
        ]

        # Identify the presence of a mechanical ventilation using settings
        CHARTEVENTS_VENTILATION_CLASSIFICATION = (
            pl.scan_csv(
                self.mimic3_paths.chartevents_path,
                schema_overrides={"VALUE": str},
            )
            .filter(
                pl.col("ITEMID").is_in(chartevents_all_ids),
                pl.col("ERROR").ne_missing(1),
                pl.col("VALUE").is_not_null(),
            )
            .select("ICUSTAY_ID", "CHARTTIME", "ITEMID", "VALUE")
            .with_columns(
                pl.col("CHARTTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                # case statement determining whether it is an instance of mech vent
                # vent type recorded
                pl.when(
                    pl.col("ITEMID") == 720,
                    pl.col("VALUE") != ("Other/Remarks"),
                )
                .then(1)
                .when(pl.col("ITEMID") == 223848, pl.col("VALUE") != "Other")
                .then(1)
                # ventilator mode
                .when(pl.col("ITEMID") == 223849).then(1)
                # O2 delivery device == ventilator
                .when(pl.col("ITEMID") == 467, pl.col("VALUE") == "Ventilator")
                .then(1)
                .when(pl.col("ITEMID").is_in(chartevents_ventilation_ids))
                .then(1)
                .otherwise(0)
                .alias("MechVent"),
                # initiation of oxygen therapy indicates the ventilation has ended
                pl.when(
                    pl.col("ITEMID") == 226732,
                    pl.col("VALUE").is_in(id226732_oxygen_therapy),
                )
                .then(1)
                .when(
                    pl.col("ITEMID") == 467,
                    pl.col("VALUE").is_in(id467_oxygen_therapy),
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
                pl.col("MechVent").max(),
                pl.col("OxygenTherapy").max(),
                pl.col("Extubated").max(),
                pl.col("SelfExtubated").max(),
            )
            .collect(streaming=True)
        )

        # add in the extubation flags from procedureevents_mv
        # note that we only need the start time for the extubation
        # (extubation is always charted as ending 1 minute after it started)
        PROCEDUREEVENTS_MV_VENTILATION_CLASSIFICATION = (
            pl.scan_csv(self.mimic3_paths.procedureevents_mv_path)
            .select("ICUSTAY_ID", "STARTTIME", "ITEMID")
            .rename({"STARTTIME": "CHARTTIME"})
            # Filter for ventilation IDs
            .filter(pl.col("ITEMID").is_in(procedureevents_mv_extubation_ids))
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
            .collect(streaming=True)
        )

        ##############################################################################
        # ventilation_durations.sql
        ##############################################################################
        # This query extracts the duration of mechanical ventilation
        # The main goal of the query is to aggregate sequential ventilator settings
        # into single mechanical ventilation "events". The start and end time of these
        # events can then be used for various purposes: calculating the total duration
        # of mechanical ventilation, cross-checking values (e.g. PaO2:FiO2 on vent), etc

        # The query's logic is roughly:
        #    1) The presence of a mechanical ventilation setting starts a new ventilation event
        #    2) Any instance of a setting in the next 8 hours continues the event
        #    3) Certain elements end the current ventilation event
        #        a) documented extubation ends the current ventilation
        #        b) initiation of non-invasive vent and/or oxygen ends the current vent

        # See the ventilation_classification.sql query for step 1 of the above.
        # This query has the logic for converting events into durations.

        VENTILATION_DURATIONS = (
            pl.concat(
                [
                    CHARTEVENTS_VENTILATION_CLASSIFICATION,
                    PROCEDUREEVENTS_MV_VENTILATION_CLASSIFICATION,
                ],
                how="vertical",
            )
            .unique()
            # this carries over the previous charttime which had a mechanical ventilation event
            .with_columns(
                pl.when(pl.col("MechVent") == 1)
                .then(
                    pl.col("CHARTTIME")
                    .shift(1)
                    .over(
                        partition_by=["ICUSTAY_ID", "MechVent"],
                        order_by="CHARTTIME",
                    )
                )
                .otherwise(None)
                .alias("CHARTTIME_LAG"),
            )
            .with_columns(
                pl.col("Extubated")
                .shift(1)
                .over(
                    partition_by=[
                        pl.col("ICUSTAY_ID"),
                        pl.when(
                            (pl.col("MechVent") == 1)
                            | (pl.col("Extubated") == 1)
                        )
                        .then(1)
                        .otherwise(0),
                    ],
                    order_by=["CHARTTIME"],
                )
                .alias("ExtubatedLag"),
            )
            # now we determine if the current mech vent event is a "new", i.e. they've just been intubated
            .with_columns(
                # if there is an extubation flag, we mark any subsequent ventilation as a new ventilation event
                # when Extubated = 1 then 0 -- extubation is *not* a new ventilation event, the *subsequent* row is
                pl.when(pl.col("ExtubatedLag") == 1)
                .then(1)
                # if patient has initiated oxygen therapy, and is not currently vented, start a newvent
                .when(pl.col("MechVent") == 0, pl.col("OxygenTherapy") == 1)
                .then(1)
                # if there is less than MAX_VENTILATION_PAUSE_HOURS hours between vent settings, we do not treat this as a new ventilation event
                .when(
                    pl.col("CHARTTIME")
                    > pl.col("CHARTTIME_LAG").add(
                        # is 8 hours in original code
                        pl.duration(hours=self.MAX_VENTILATION_PAUSE_HOURS)
                    )
                )
                .then(1)
                .otherwise(0)
                .alias("NewVent")
            )
            # create a cumulative sum of the instances of new ventilation
            # this results in a monotonic integer assigned to each instance of ventilation
            .with_columns(
                pl.when((pl.col("MechVent") == 1) | (pl.col("Extubated") == 1))
                .then(
                    pl.col("NewVent")
                    .cum_sum()
                    .over(partition_by=["ICUSTAY_ID"], order_by=["CHARTTIME"])
                )
                .otherwise(None)
                .alias("VentNum")
            )
            .group_by("ICUSTAY_ID", "VentNum")
            .agg(
                pl.col("CHARTTIME").min().alias("STARTTIME"),
                pl.col("CHARTTIME").max().alias("ENDTIME"),
                pl.col("MechVent").max().alias("MechVent"),
                pl.col("OxygenTherapy").max().alias("OxygenTherapy"),
            )
            # Make datetime relative to admission in seconds
            .join(ADMISSIONTIMES, on="ICUSTAY_ID", how="left")
            .with_columns(
                (pl.col("STARTTIME") - pl.col("INTIME"))
                .dt.total_seconds()
                .alias("Ventilation Start Relative to Admission (seconds)"),
                (pl.col("ENDTIME") - pl.col("INTIME"))
                .dt.total_seconds()
                .alias("Ventilation End Relative to Admission (seconds)"),
                pl.when(pl.col("MechVent") == 1)
                .then(pl.lit("invasive ventilation"))
                .when(pl.col("OxygenTherapy") == 1)
                .then(pl.lit("supplemental oxygen"))
                .alias("Ventilation Type"),
            )
            .select(
                "ICUSTAY_ID",
                "Ventilation Type",
                "Ventilation Start Relative to Admission (seconds)",
                "Ventilation End Relative to Admission (seconds)",
            )
        )

        return (
            VENTILATION_DURATIONS.unique()
            .pipe(self._add_global_id_stay_id, "mimic3-", "ICUSTAY_ID")
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
