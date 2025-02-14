# based on https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/concepts/durations/crrt_durations.sql

import polars as pl
from helpers.MAGIC_CONCEPTS.MAGIC_CONCEPTS import MAGIC_CONCEPTS


class RENAL_REPLACEMENT_THERAPY_DURATION_MIMIC3(MAGIC_CONCEPTS):
    def __init__(self, paths, datasets):
        super().__init__(paths, datasets)

    def RENAL_REPLACEMENT_THERAPY_DURATION(self) -> pl.DataFrame:
        print("MAGIC_CONCEPTS: Renal Replacement Therapy Duration - MIMIC3")

        # get admission times for MIMIC-III
        ADMISSIONTIMES = (
            pl.scan_csv(self.mimic3_paths.icustays_path)
            .select("ICUSTAY_ID", "INTIME")
            .with_columns(
                pl.col("INTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
        )

        # region ITEMIDS
        # fmt: off
        chartevents_metavision = [
            224144,  # Blood Flow (ml/min)
            224145,  # Heparin Dose (per hour)
            224149,  # Access Pressure
            224150,  # Filter Pressure
            224151,  # Effluent Pressure
            224152,  # Return Pressure
            224153,  # Replacement Rate
            224154,  # Dialysate Rate
            224191,  # Hourly Patient Fluid Removal
            224404,  # ART Lumen Volume
            224406,  # VEN Lumen Volume
            225183,  # Current Goal
            225806,  # Volume In (PD)
            225807,  # Volume Out (PD)
            225810,  # Dwell Time (Peritoneal Dialysis)
            225959,  # Medication Added Amount  #1 (Peritoneal Dialysis)
            225958,  # Heparin Concentration (units/mL)
            225976,  # Replacement Fluid
            225977,  # Dialysate Fluid
            226457,  # Ultrafiltrate Output
            226499,  # Hemodialysis Output
            227438,  # Volume not removed
            227639,  # Medication Added Amount  #2 (Peritoneal Dialysis)
            228004,  # Citrate (ACD-A)
            228005,  # PBP (Prefilter) Replacement Rate
            228006,  # Post Filter Replacement Rate
        ]
        chartevents_metavision_addional = [
            224146,  # System Integrity (MetaVision)
            225956,  # Reason for CRRT Filter Change (MetaVision)
            227290,  # CRRT mode
        ]

        chartevents_carevue = [
            29,    # Access mmHg (CareVue)
            79,    # Blood Flow ml/min (CareVue)
            142,   # Current Goal (CareVue)
            146,   # Dialysate Flow ml/hr (CareVue)
            173,   # Effluent Press mmHg (CareVue)
            192,   # Filter Pressure mmHg (CareVue)
            611,   # Replace Rate ml/hr (CareVue)
            624,   # Return Pressure mmHg (CareVue)
            5683,  # Hourly PFR (CareVue)
        ]
        chartevents_carevue_additional = [
            147,  # Dialysate Infusing (CareVue)
            152,  # Dialysis Type
            582,  # Procedures
            665,  # System integrity (CareVue)
        ]
        # fmt: on

        ##############################################################################
        # pivoted_rrt.sql
        ##############################################################################
        # region CE
        RENAL_REPLACEMENT_THERAPY_DURATION = (
            # Load chartevents and filter for CRRT settings (see crrt.sql: crrt_settings)
            pl.scan_csv(
                self.mimic3_paths.chartevents_path,
                schema_overrides={"VALUE": str},
            )
            .filter(
                pl.col("ITEMID").is_in(
                    chartevents_metavision
                    + chartevents_metavision_addional
                    + chartevents_carevue
                    + chartevents_carevue_additional
                )
            )
            .filter(
                pl.col("VALUE").is_not_null()
                & ((pl.col("VALUENUM").fill_null(1)) != 0)
            )
            .select("ICUSTAY_ID", "CHARTTIME", "ITEMID", "VALUE")
            .with_columns(
                pl.col("CHARTTIME").str.to_datetime("%Y-%m-%d %H:%M:%S")
            )
            # Create flag columns matching the SQL CASE logic
            .with_columns(
                pl.when(
                    pl.col("ITEMID").is_in(
                        chartevents_metavision + chartevents_carevue
                    )
                )
                .then(1)
                .when(
                    pl.col("ITEMID") == 665,
                    pl.col("VALUE").is_in(
                        [
                            "Active",
                            "Clot Increasing",
                            "Clots Present",
                            "No Clot Present",
                        ]
                    ),
                )
                .then(1)
                .when(pl.col("ITEMID") == 147, pl.col("VALUE") == "Yes")
                .then(1)
                .otherwise(0)
                .alias("RRT"),
                # Below indicates that a new instance of CRRT has started
                pl.when(
                    pl.col("ITEMID") == 224146,
                    pl.col("VALUE").is_in(["New Filter", "Reinitiated"]),
                )
                .then(1)
                .when(pl.col("ITEMID") == 665, pl.col("VALUE") == "Initiated")
                .then(1)
                # taken from pivoted_rrt.sql
                .when(
                    pl.col("ITEMID") == 582,
                    pl.col("VALUE").is_in(
                        [
                            "CAVH Start",
                            "CVVHD Start",
                            "Hemodialysis st",
                            "Peritoneal Dial",
                        ]
                    ),
                )
                .then(1)
                .otherwise(0)
                .alias("RRT_start"),
                # Below indicates that the current instance of CRRT has ended
                pl.when(
                    pl.col("ITEMID") == 224146,
                    pl.col("VALUE").is_in(["Discontinued", "Recirculating"]),
                )
                .then(1)
                .when(
                    pl.col("ITEMID") == 665,
                    (pl.col("VALUE") == "Clotted")
                    | pl.col("VALUE").str.starts_with("DC"),
                )
                .then(1)
                .when(pl.col("ITEMID") == 225956)
                .then(1)
                # taken from pivoted_rrt.sql
                .when(
                    pl.col("ITEMID") == 582,
                    pl.col("VALUE").is_in(
                        ["CAVH D/C", "CVVHD D/C", "Hemodialysis end"]
                    ),
                )
                .then(1)
                .otherwise(0)
                .alias("RRT_end"),
                # Below indicates the type of CRRT (taken from pivoted_rrt.sql)
                pl.when(pl.col("ITEMID").is_in([227290, 152]))
                .then(pl.col("VALUE"))
                .when(
                    pl.col("ITEMID").is_in(
                        # IDs for peritoneal dialysis
                        [225806, 225807, 225810, 225959, 227639]
                    )
                )
                .then(pl.lit("Peritoneal dialysis"))
                .when(pl.col("ITEMID") == 226499)
                .then(pl.lit("IHD"))
                .when(pl.col("ITEMID") == 582)
                .then(
                    pl.when(pl.col("VALUE").is_in(["CAVH Start", "CAVH D/C"]))
                    .then(pl.lit("CAVH"))
                    .when(pl.col("VALUE").is_in(["CVVHD Start", "CVVHD D/C"]))
                    .then(pl.lit("CVVHD"))
                    .otherwise(None)
                )
                .otherwise(None)
                .alias("RRT_type"),
            )
            .group_by("ICUSTAY_ID", "CHARTTIME")
            .agg(
                pl.col("RRT").max(),
                pl.col("RRT_start").max(),
                pl.col("RRT_end").max(),
                pl.col("RRT_type").max(),
            )
            .sort("ICUSTAY_ID", "CHARTTIME")
            # create various lagged variables for future query
            .with_columns(
                pl.col("CHARTTIME")
                .shift(1)
                .over(
                    "ICUSTAY_ID",
                    pl.when((pl.col("RRT") == 1) | pl.col("RRT_end") == 1)
                    .then(1)
                    .otherwise(0),
                    order_by="CHARTTIME",
                )
                .alias("charttime_prev_row"),
                pl.col("RRT_end")
                .shift(1)
                .over(
                    "ICUSTAY_ID",
                    pl.when((pl.col("RRT") == 1) | pl.col("RRT_end") == 1)
                    .then(1)
                    .otherwise(0),
                    order_by="CHARTTIME",
                )
                .alias("rrt_ended_prev_row"),
            )
            # now we determine if the current event is a new instantiation
            .with_columns(
                pl.when(pl.col("RRT_start") == 1)
                .then(1)
                # if there is an end flag, we mark any subsequent event as new
                # note the end is *not* a new event, the *subsequent* row is
                # so here we output 0
                .when(pl.col("RRT_end") == 1)
                .then(0)
                .when(pl.col("rrt_ended_prev_row") == 1)
                .then(1)
                # if there is less than 2 hours between CRRT settings, we do not treat this as a new CRRT event
                .when(
                    (pl.col("CHARTTIME") - pl.col("charttime_prev_row")).lt(
                        pl.duration(hours=2)
                    )
                )
                .then(0)
                .otherwise(1)
                .alias("NewCRRT")
            )
            # create a cumulative sum of the instances of new CRRT
            # this results in a monotonically increasing integer assigned to each CRRT
            .with_columns(
                pl.when(
                    (pl.col("RRT_start") == 1)
                    | (pl.col("RRT") == 1)
                    | (pl.col("RRT_end") == 1)
                )
                .then(
                    pl.col("NewCRRT")
                    .cum_sum()
                    .over("ICUSTAY_ID", order_by="CHARTTIME")
                )
                .otherwise(None)
                .alias("num")
            )
            # # now we can isolate to just rows with settings
            # # (before we had rows with start/end flags)
            # # this removes any null values for NewCRRT
            # .filter(
            #     pl.col("RRT_start") == 1,
            #     pl.col("RRT") == 1,
            #     pl.col("RRT_end") == 1,
            # )
            .group_by("ICUSTAY_ID", "num")
            .agg(
                pl.col("CHARTTIME").min().alias("STARTTIME"),
                pl.col("CHARTTIME").max().alias("ENDTIME"),
                pl.col("RRT_type")
                .max()
                .alias("Renal Replacement Therapy Type"),
            )
            # Make datetime relative to admission in seconds
            .join(ADMISSIONTIMES, on="ICUSTAY_ID", how="left")
            .with_columns(
                (pl.col("STARTTIME") - pl.col("INTIME"))
                .dt.total_seconds()
                .alias(
                    "Renal Replacement Therapy Start Relative to Admission (seconds)"
                ),
                (pl.col("ENDTIME") - pl.col("INTIME"))
                .dt.total_seconds()
                .alias(
                    "Renal Replacement Therapy End Relative to Admission (seconds)"
                ),
            )
            .select(
                "ICUSTAY_ID",
                "Renal Replacement Therapy Type",
                "Renal Replacement Therapy Start Relative to Admission (seconds)",
                "Renal Replacement Therapy End Relative to Admission (seconds)",
            )
            .collect(streaming=True)
        )

        return (
            RENAL_REPLACEMENT_THERAPY_DURATION.select(
                "ICUSTAY_ID",
                "Renal Replacement Therapy Type",
                "Renal Replacement Therapy Start Relative to Admission (seconds)",
                "Renal Replacement Therapy End Relative to Admission (seconds)",
            ).unique()
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
