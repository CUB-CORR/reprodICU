# based on https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/concepts/durations/crrt_durations.sql
# and https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/concepts/pivot/pivoted_rrt.sql

import polars as pl

from ..MAGIC_CONCEPTS import MAGIC_CONCEPTS

START = "Renal Replacement Therapy Start Relative to Admission (seconds)"
END = "Renal Replacement Therapy End Relative to Admission (seconds)"
TYPE = "Renal Replacement Therapy Type"


class RENAL_REPLACEMENT_THERAPY_DURATION_MIMIC3(MAGIC_CONCEPTS):
    def __init__(self, paths, datasets):
        super().__init__(paths, datasets)

    def RENAL_REPLACEMENT_THERAPY_DURATION(self) -> pl.DataFrame:
        """
        Extract renal replacement therapy episodes from MIMIC-III procedures.

        Steps:
            1. Identify RRT from procedure/event records (CRRT codes).
            2. Extract RRT start/end times.
            3. Calculate episode duration.
            4. Classify RRT type (CVVH, CVVHD, IHD, etc.).
            5. Compute time relative to admission.

        Returns:
            pl.DataFrame: Contains columns:
                - ICUSTAY_ID: ICU stay identifier.
                - {timeseries_time_col}: RRT start time (seconds from admission).
                - Renal Replacement Therapy Type: RRT modality.
                - Renal Replacement Therapy Duration (hours): Episode duration.
        """
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
            225183,  # Current Goal
            225958,  # Heparin Concentration (units/mL)
            225976,  # Replacement Fluid
            225977,  # Dialysate Fluid
            226457,  # Ultrafiltrate Output
            228004,  # Citrate (ACD-A)
            228005,  # PBP (Prefilter) Replacement Rate
            228006,  # Post Filter Replacement Rate
        ]
        chartevents_metavision_addional = [
            224146,  # System Integrity (MetaVision)
            225956,  # Reason for CRRT Filter Change (MetaVision)
            227290,  # CRRT mode
            # not in crrt_durations.sql (RRT = 0), kept for the type of pivoted_rrt.sql
            224404,  # ART Lumen Volume
            224406,  # VEN Lumen Volume
            225806,  # Volume In (PD)
            225807,  # Volume Out (PD)
            225810,  # Dwell Time (Peritoneal Dialysis)
            225959,  # Medication Added Amount  #1 (Peritoneal Dialysis)
            226499,  # Hemodialysis Output
            227438,  # Volume not removed
            227639,  # Medication Added Amount  #2 (Peritoneal Dialysis)
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
            612,  # Replace.Fluid Infuse (CareVue)
            147,  # Dialysate Infusing (CareVue)
            152,  # Dialysis Type
            582,  # Procedures
            665,  # System integrity (CareVue)
        ]
        # pivoted_rrt.sql, mv_ranges
        inputevents = [
            227536,  # KCl (CRRT)
            227525,  # Calcium Gluconate (CRRT)
        ]
        procedureevents = [
            225441,  # Hemodialysis
            225802,  # Dialysis - CRRT
            225803,  # Dialysis - CVVHD
            225805,  # Peritoneal Dialysis
            224270,  # Dialysis Catheter
            225809,  # Dialysis - CVVHDF
            225955,  # Dialysis - SCUF
            225436,  # CRRT Filter Change
        ]
        # fmt: on

        if "parquet" in self.mimic3_paths.chartevents_path:
            CHARTEVENTS = pl.scan_parquet(self.mimic3_paths.chartevents_path)
        else:
            CHARTEVENTS = pl.scan_csv(
                self.mimic3_paths.chartevents_path,
                schema_overrides={"VALUE": str},
            )

        if "parquet" in self.mimic3_paths.inputevents_mv_path:
            INPUTEVENTS = pl.scan_parquet(self.mimic3_paths.inputevents_mv_path)
        else:
            INPUTEVENTS = pl.scan_csv(self.mimic3_paths.inputevents_mv_path)

        ##############################################################################
        # crrt_durations.sql
        ##############################################################################
        # region CE
        RENAL_REPLACEMENT_THERAPY_DURATION = (
            # Load chartevents and filter for CRRT settings (see crrt.sql: crrt_settings)
            CHARTEVENTS.filter(
                pl.col("ITEMID").is_in(
                    chartevents_metavision
                    + chartevents_metavision_addional
                    + chartevents_carevue
                    + chartevents_carevue_additional
                )
            )
            .filter(
                pl.col("VALUE").is_not_null(),
                pl.col("VALUENUM").ne_missing(0),  # non-zero rates/values
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
                .otherwise(0)
                .alias("RRT_start"),
                # Below indicates that the current instance of CRRT has ended
                pl.when(
                    pl.col("ITEMID") == 224146,
                    pl.col("VALUE").is_in(["Discontinued", "Recirculating"]),
                ).then(1)
                # the only value like DC is "DC'D", use starts_with to handle pattern
                .when(
                    pl.col("ITEMID") == 665,
                    (pl.col("VALUE") == "Clotted")
                    | pl.col("VALUE").str.starts_with("DC"),
                ).then(1)
                # Reason for CRRT filter change
                .when(pl.col("ITEMID") == 225956).then(1)
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
                    pl.when((pl.col("RRT") == 1) | (pl.col("RRT_end") == 1))
                    .then(1)
                    .otherwise(0),
                    order_by="CHARTTIME",
                )
                .alias("charttime_prev_row"),
                pl.col("RRT_end")
                .shift(1)
                .over(
                    "ICUSTAY_ID",
                    pl.when((pl.col("RRT") == 1) | (pl.col("RRT_end") == 1))
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
                .when(pl.col("RRT_end") == 1)
                # note the end is *not* a new event, the *subsequent* row is
                # so here we output 0
                .then(0)
                .when(pl.col("rrt_ended_prev_row") == 1)
                .then(1)
                # if there are at most 2 hours between CRRT settings, we do not treat this as a new CRRT event
                # (DATETIME_DIFF(charttime, charttime_prev_row, HOUR) counts the hour boundaries)
                .when(
                    (
                        pl.col("CHARTTIME").dt.truncate("1h")
                        - pl.col("charttime_prev_row").dt.truncate("1h")
                    )
                    <= pl.duration(hours=2)
                )
                .then(0)
                .otherwise(1)
                .alias("NewCRRT")
            )
            # now we can isolate to just rows with settings
            # (before we had rows with start/end flags)
            # this removes any null values for NewCRRT
            .filter(
                (pl.col("RRT_start") == 1)
                | (pl.col("RRT") == 1)
                | (pl.col("RRT_end") == 1)
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
            .group_by("ICUSTAY_ID", "num")
            .agg(
                pl.col("CHARTTIME").min().alias("STARTTIME"),
                pl.col("CHARTTIME").max().alias("ENDTIME"),
                pl.col("RRT_type")
                .max()
                .alias(TYPE),
            )
            .filter(pl.col("STARTTIME") != pl.col("ENDTIME"))
            # Make datetime relative to admission in seconds
            .join(ADMISSIONTIMES, on="ICUSTAY_ID", how="left")
            .with_columns(
                (pl.col("STARTTIME") - pl.col("INTIME"))
                .dt.total_seconds()
                .alias(START),
                (pl.col("ENDTIME") - pl.col("INTIME"))
                .dt.total_seconds()
                .alias(END),
            )
            .select(
                "ICUSTAY_ID",
                TYPE,
                START,
                END,
            )
            .collect()
        )

        ##############################################################################
        # pivoted_rrt.sql, mv_ranges
        ##############################################################################
        RENAL_REPLACEMENT_THERAPY_INPUTEVENTS = (
            INPUTEVENTS.select("ICUSTAY_ID", "STARTTIME", "ENDTIME", "ITEMID", "AMOUNT")
            .filter(
                pl.col("ITEMID").is_in(inputevents),
                pl.col("AMOUNT") > 0,
            )
            .with_columns(
                pl.lit(True).alias("dialysis_present"),
                pl.lit(True).alias("dialysis_active"),
                pl.lit("CRRT").alias("dialysis_type"),
            )
            .select(
                "ICUSTAY_ID",
                "STARTTIME",
                "ENDTIME",
                "dialysis_present",
                "dialysis_active",
                "dialysis_type",
            )
        )

        RENAL_REPLACEMENT_THERAPY_PROCEDUREEVENTS = (
            pl.scan_csv(
                self.mimic3_paths.procedureevents_mv_path,
                schema_overrides={"VALUE": pl.String},
            )
            .select("ICUSTAY_ID", "STARTTIME", "ENDTIME", "ITEMID", "VALUE")
            .filter(
                pl.col("ITEMID").is_in(procedureevents),
                pl.col("VALUE").is_not_null(),
            )
            .with_columns(
                pl.lit(True).alias("dialysis_present"),
                pl.when(pl.col("ITEMID").is_in([224270, 225436]))
                .then(False)
                .otherwise(True)
                .alias("dialysis_active"),
                pl.when(pl.col("ITEMID") == 225441)
                .then(pl.lit("IHD"))
                .when(pl.col("ITEMID") == 225802)
                .then(pl.lit("CRRT"))
                .when(pl.col("ITEMID") == 225803)
                .then(pl.lit("CVVHD"))
                .when(pl.col("ITEMID") == 225805)
                .then(pl.lit("Peritoneal dialysis"))
                .when(pl.col("ITEMID") == 225809)
                .then(pl.lit("CVVHDF"))
                .when(pl.col("ITEMID") == 225955)
                .then(pl.lit("SCUF"))
                .otherwise(None)
                .alias("dialysis_type"),
            )
            .select(
                "ICUSTAY_ID",
                "STARTTIME",
                "ENDTIME",
                "dialysis_present",
                "dialysis_active",
                "dialysis_type",
            )
        )

        RENAL_REPLACEMENT_THERAPY_RANGES = (
            pl.concat(
                [
                    RENAL_REPLACEMENT_THERAPY_INPUTEVENTS,
                    RENAL_REPLACEMENT_THERAPY_PROCEDUREEVENTS,
                ],
                how="vertical",
            )
            .unique()
            # dialysis_active = 0 (catheter, filter change) is no therapy
            .filter(pl.col("dialysis_active"), pl.col("ICUSTAY_ID").is_not_null())
            .join(ADMISSIONTIMES, on="ICUSTAY_ID", how="left")
            # Make datetime relative to admission in seconds
            .select(
                "ICUSTAY_ID",
                pl.col("dialysis_type").alias(TYPE),
                (
                    pl.col("STARTTIME").str.to_datetime("%Y-%m-%d %H:%M:%S")
                    - pl.col("INTIME")
                )
                .dt.total_seconds()
                .alias(START),
                (
                    pl.col("ENDTIME").str.to_datetime("%Y-%m-%d %H:%M:%S")
                    - pl.col("INTIME")
                )
                .dt.total_seconds()
                .alias(END),
            )
            .collect()
        )

        # merge overlapping and touching episodes of a stay; a specific type wins over "CRRT"
        RENAL_REPLACEMENT_THERAPY_DURATION = (
            pl.concat([RENAL_REPLACEMENT_THERAPY_DURATION, RENAL_REPLACEMENT_THERAPY_RANGES])
            .with_columns(pl.col(TYPE).fill_null("CRRT"))
            .sort("ICUSTAY_ID", START)
            .with_columns(
                (pl.col(START) > pl.col(END).cum_max().shift(1).over("ICUSTAY_ID"))
                .fill_null(True)
                .cum_sum()
                .over("ICUSTAY_ID")
                .alias("episode")
            )
            .group_by("ICUSTAY_ID", "episode")
            .agg(
                pl.col(TYPE).filter(pl.col(TYPE) != "CRRT").mode().sort().first(),
                pl.col(START).min(),
                pl.col(END).max(),
            )
            .with_columns(pl.col(TYPE).fill_null("CRRT"))
        )

        return (
            RENAL_REPLACEMENT_THERAPY_DURATION.select(
                "ICUSTAY_ID",
                TYPE,
                START,
                END,
            )
            .unique()
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
