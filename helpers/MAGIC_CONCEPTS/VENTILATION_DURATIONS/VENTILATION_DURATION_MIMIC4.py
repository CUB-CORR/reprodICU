# based on https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/measurement/ventilator_setting.sql
# and https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/measurement/oxygen_delivery.sql
# and https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/treatment/ventilation.sql

import polars as pl
from helpers.MAGIC_CONCEPTS.MAGIC_CONCEPTS import MAGIC_CONCEPTS


class VENTILATION_DURATION_MIMIC4(MAGIC_CONCEPTS):
    def __init__(self, paths, datasets, MAX_VENTILATION_PAUSE_HOURS):
        super().__init__(paths, datasets)
        self.MAX_VENTILATION_PAUSE_HOURS = MAX_VENTILATION_PAUSE_HOURS

    def VENTILATION_DURATION(self) -> pl.DataFrame:
        print("MAGIC_CONCEPTS: Ventilation Duration - MIMIC4")
        # fmt: off
        vent_setting_chartevents_ids = [
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
        o2_flow_chartevents_ids = [
            223834, # o2 flow
            227582, # bipap o2 flow
            227287, # additional o2 flow
        ]
        o2_delivery_chartevents_ids = [
            226732,  # oxygen delivery device(s)
        ]
        # fmt: on

        # get admission times for MIMIC-IV
        ADMISSIONTIMES = (
            pl.scan_csv(self.mimic4_paths.icustays_path)
            .select("stay_id", "intime")
            .with_columns(
                pl.col("intime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
            .collect()
        )

        # common chartevents scan
        CHARTEVENTS = pl.scan_csv(
            self.mimic4_paths.chartevents_path,
            schema_overrides={"value": str},
        ).filter(
            pl.col("itemid").is_in(
                vent_setting_chartevents_ids
                + o2_flow_chartevents_ids
                + o2_delivery_chartevents_ids
            )
        )

        ##############################################################################
        # oxygen_delivery.sql
        ##############################################################################
        CHARTEVENTS_OXYGEN_FLOW = (
            # ce_stg1
            CHARTEVENTS.select(
                "subject_id",
                "stay_id",
                "charttime",
                "storetime",
                "itemid",
                "value",
                "valuenum",
            )
            .sort("subject_id", "charttime")
            .filter(pl.col("itemid").is_in(o2_flow_chartevents_ids))
            .with_columns(
                pl.col("charttime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("storetime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                # merge o2 flows into a single row
                pl.when(pl.col("itemid") == 226732)
                .then(223834)
                .otherwise(pl.col("itemid"))
                .alias("itemid"),
            )
            # ce_stg2
            # retain only 1 row per charttime
            # prioritizing the last documented value
            # primarily used to subselect o2 flows
            .with_columns(
                pl.col("storetime")
                .rank("ordinal", descending=True)
                .over(partition_by=["subject_id", "charttime", "itemid"])
                .alias("rn"),
            )
            .drop("storetime")
        )

        CHARTEVENTS_OXYGEN_DELIVERY = (
            CHARTEVENTS.select(
                "subject_id",
                "stay_id",
                "charttime",
                "storetime",
                "itemid",
                "value",
            )
            .sort("subject_id", "charttime")
            .filter(pl.col("itemid").is_in(o2_delivery_chartevents_ids))
            .rename({"value": "o2_device"})
            .with_columns(
                pl.col("charttime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("storetime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
            .with_columns(
                pl.col("o2_device")
                .rank("ordinal")
                .over(partition_by=["subject_id", "charttime", "itemid"])
                .alias("rn"),
            )
            .drop("storetime")
        )

        OXYGEN_DELIVERY_IDS = (
            CHARTEVENTS_OXYGEN_FLOW.select("subject_id", "charttime")
            .join(
                CHARTEVENTS_OXYGEN_DELIVERY.select("subject_id", "charttime"),
                on=["subject_id", "charttime"],
                how="outer",
                coalesce=True,
            )
            .unique()
        )

        OXYGEN_DELIVERY = (
            OXYGEN_DELIVERY_IDS.join(
                CHARTEVENTS_OXYGEN_DELIVERY.join(
                    # limit to 1 row per subject_id/charttime/itemid
                    CHARTEVENTS_OXYGEN_FLOW.filter(pl.col("rn") == 1),
                    on=["subject_id", "stay_id", "charttime", "itemid"],
                    how="full",
                    coalesce=True,
                ),
                on=["subject_id", "charttime"],
                how="left",
                coalesce=True,
            )
            .group_by("subject_id", "charttime")
            .agg(
                pl.col("stay_id").max(),
                # contrary to mimic-code template we only need the first o2_device
                pl.when(pl.col("rn") == 1)
                .then(pl.col("o2_device"))
                .otherwise(None)
                .drop_nulls()
                .first()
                .alias("o2_device"),
            )
            .collect(streaming=True)
        )

        ##############################################################################
        # ventilator_setting.sql
        ##############################################################################
        VENTILATOR_SETTINGS = (
            CHARTEVENTS.select(
                "subject_id",
                "stay_id",
                "charttime",
                "itemid",
                "value",
                "valuenum",
            )
            .filter(
                pl.col("stay_id").is_not_null(),
                pl.col("value").is_not_null(),
                pl.col("itemid").is_in(vent_setting_chartevents_ids),
            )
            .with_columns(
                pl.col("charttime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                # fio2 cleaning
                pl.when(pl.col("itemid") == 223835).then(
                    pl.when(pl.col("valuenum") >= 0.20, pl.col("valuenum") <= 1)
                    .then(pl.col("valuenum") * 100)
                    # improperly input data - looks like O2 flow in litres
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
            # only including the columns that are relevant downstream
            .agg(
                pl.col("stay_id").max(),
                pl.when(pl.col("itemid") == 223849)
                .then(pl.col("value"))
                .otherwise(None)
                .first()
                .alias("ventilator_mode"),
                pl.when(pl.col("itemid") == 229314)
                .then(pl.col("value"))
                .otherwise(None)
                .first()
                .alias("ventilator_mode_hamilton"),
            )
            .collect(streaming=True)
        )

        ##############################################################################
        # ventilation.sql
        ##############################################################################
        # Classify oxygen devices and ventilator modes into six clinical categories.

        # Categories include..
        #  Invasive oxygen delivery types:
        #      Tracheostomy (with or without positive pressure ventilation)
        #      InvasiveVent (positive pressure ventilation via endotracheal tube,
        #          could be oro/nasotracheal or tracheostomy)
        #  Non invasive oxygen delivery types (ref doi:10.1001/jama.2020.9524):
        #      NonInvasiveVent (non-invasive positive pressure ventilation)
        #      HFNC (high flow nasal oxygen / cannula)
        #      SupplementalOxygen (all other non-rebreather,
        #          facemask, face tent, nasal prongs...)
        #  No oxygen device:
        #      None

        # When conflicting settings occur (rare), the priority is:
        #  trach > mech vent > NIV > high flow > o2

        # Some useful cases for debugging:
        #  stay_id = 30019660 has a tracheostomy placed in the ICU
        #  stay_id = 30000117 has explicit documentation of extubation

        # first we collect all times which have relevant documentation
        VENT_IDS = (
            pl.concat(
                [
                    OXYGEN_DELIVERY.select("stay_id", "charttime"),
                    VENTILATOR_SETTINGS.select("stay_id", "charttime"),
                ],
                how="vertical",
            )
            .unique()
            .sort("stay_id", "charttime")
        )

        VENTILATION = (
            # vs
            VENT_IDS.join(
                OXYGEN_DELIVERY,
                on=["stay_id", "charttime"],
                how="left",
                coalesce=True,
            )
            .join(
                VENTILATOR_SETTINGS,
                on=["stay_id", "charttime"],
                how="left",
                coalesce=True,
            )
            .with_columns(
                pl.coalesce(
                    pl.col("ventilator_mode"),
                    pl.col("ventilator_mode_hamilton"),
                ).alias("vent_mode"),
                # case statement determining the type of intervention
                # done in order of priority: trach > mech vent > NIV > hiflow / o2
                pl.when(
                    pl.col("o2_device").is_in(
                        ["Tracheostomy tube", "Trach mask"]
                    )
                )
                .then(pl.lit("tracheostomy"))
                .when(
                    (pl.col("o2_device") == "Endotracheal tube")
                    | pl.col("ventilator_mode").is_in(
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
                        ] # fmt: skip
                    )
                    | pl.col("ventilator_mode_hamilton").is_in(
                        [
                            "APRV", "APV (cmv)", "Ambient", "(S) CMV", "P-CMV",
                            "SIMV", "APV (simv)", "P-SIMV", "VS", "ASV"
                        ] # fmt: skip
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
                        [
                            "High flow nasal cannula", # HFNC not extra
                            "Non-rebreather", "Face tent", "Aerosol-cool",
                            "Venti mask ", "Medium conc mask ",
                            "Ultrasonic neb", "Vapomist", "Oxymizer",
                            "High flow neb", "Nasal cannula",
                        ] # fmt: skip
                    )
                )
                .then(pl.lit("supplemental oxygen"))
                .when(pl.col("o2_device") == "None")
                .then(pl.lit("None"))
                .otherwise(None)
                .alias("ventilation_status"),
            )
            # vd0
            .filter(pl.col("ventilation_status").is_not_null())
            .with_columns(
                # carry over the previous charttime which had the same state
                pl.col("charttime")
                .shift(1)
                .over(
                    partition_by=["stay_id", "ventilation_status"],
                    order_by=["charttime"],
                )
                .alias("charttime_lag"),
                # bring back the next charttime, regardless of the state
                # this will be used as the end time for state transitions
                pl.col("charttime")
                .shift(-1)
                .over(partition_by=["stay_id"], order_by=["charttime"])
                .alias("charttime_lead"),
                pl.col("ventilation_status")
                .shift(1)
                .over(partition_by=["stay_id"], order_by=["charttime"])
                .alias("ventilation_status_lag"),
            )
            # vd1
            .with_columns(
                # now we determine if the current ventilation status is "new",
                # or continuing the previous event
                # if lag is null, this is the first event for the patient
                pl.when(pl.col("ventilation_status_lag").is_null())
                .then(1)
                # a X hour gap always initiates a new event
                .when(
                    (pl.col("charttime") - pl.col("charttime_lag")).gt(
                        # is 14 hours in original code
                        pl.duration(hours=self.MAX_VENTILATION_PAUSE_HOURS)
                    )
                )
                .then(1)
                # not a new event if identical to the last row
                .when(
                    pl.col("ventilation_status_lag")
                    != pl.col("ventilation_status")
                )
                .then(1)
                .otherwise(0)
                .alias("new_ventilation_event"),
            )
            # vd2
            .with_columns(
                # create a cumulative sum of the instances of new ventilation
                # this results in a monotonically increasing integer assigned
                # to each instance of ventilation
                pl.col("new_ventilation_event")
                .cum_sum()
                .over(partition_by=["stay_id"], order_by=["charttime"])
                .alias("vent_seq")
            )
            # create the durations for each ventilation instance
            .group_by("stay_id", "vent_seq")
            .agg(
                pl.col("charttime").min().alias("starttime"),
                # for the end time of the ventilation event, the time of the *next* setting
                # i.e. if we go NIV -> O2, the end time of NIV is the first row
                # with a documented O2 device
                # ... unless it's been over X hours,
                # in which case it's the last row with a documented NIV.
                pl.when(
                    pl.col("charttime_lead").is_null()
                    | (pl.col("charttime") - pl.col("charttime_lag")).gt(
                        # is 14 hours in original code
                        pl.duration(hours=self.MAX_VENTILATION_PAUSE_HOURS)
                    )
                )
                .then(pl.col("charttime"))
                .otherwise(pl.col("charttime_lead"))
                .max()
                .alias("endtime"),
                # all rows with the same vent_num will have the same ventilation_status
                # for efficiency, we use an aggregate here,
                # but we could equally well group by this column
                pl.col("ventilation_status").drop_nulls().first(),
            )
            .filter(pl.col("starttime").ne(pl.col("endtime")))
            # Make datetime relative to admission in seconds
            .join(ADMISSIONTIMES, on="stay_id", how="left")
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

        return (
            VENTILATION.unique()
            .pipe(self._add_global_id_stay_id, "mimic4-", "stay_id")
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
