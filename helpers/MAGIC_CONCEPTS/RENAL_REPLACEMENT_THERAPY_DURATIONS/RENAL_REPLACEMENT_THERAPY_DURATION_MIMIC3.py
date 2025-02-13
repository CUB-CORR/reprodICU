# based on https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/concepts/pivot/pivoted_rrt.sql

import polars as pl
from helpers.MAGIC_CONCEPTS.MAGIC_CONCEPTS import MAGIC_CONCEPTS


class RENAL_REPLACEMENT_THERAPY_DURATION_MIMIC3(MAGIC_CONCEPTS):
    def __init__(self, paths, datasets, MAX_VENTILATION_PAUSE_HOURS):
        super().__init__(paths, datasets)
        self.MAX_VENTILATION_PAUSE_HOURS = MAX_VENTILATION_PAUSE_HOURS

    def RENAL_REPLACEMENT_THERAPY_DURATION(self) -> pl.DataFrame:
        print("MAGIC_CONCEPTS: Renal Replacement Therapy Duration - MIMIC3")

        # get admission times for MIMIC-III
        ADMISSIONTIMES = (
            pl.scan_csv(self.mimic3_paths.icustays_path)
            .select("ICUSTAY_ID", "INTIME")
            .with_columns(
                pl.col("INTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
            .collect()
        )

        # region ITEMIDS
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
        ]

        chartevents_carevue = [
            29,  # Access mmHg (CareVue)
            173,  # Effluent Press mmHg (CareVue)
            192,  # Filter Pressure mmHg (CareVue)
            624,  # Return Pressure mmHg (CareVue)
            79,  # Blood Flow ml/min (CareVue)
            142,  # Current Goal (CareVue)
            146,  # Dialysate Flow ml/hr (CareVue)
            611,  # Replace Rate ml/hr (CareVue)
            5683,  # Hourly PFR (CareVue)
        ]
        chartevents_carevue_additional = [
            665,  # System integrity (CareVue)
            147,  # Dialysate Infusing (CareVue)
        ]

        ##############################################################################
        # pivoted_rrt.sql
        ##############################################################################
        # region CE
        RENAL_REPLACEMENT_THERAPY_DURATION = (
            # Load chartevents and filter for CRRT settings (see crrt.sql: crrt_settings)
            pl.scan_csv(self.mimic3_paths.chartevents_path)
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
                        "Active",
                        "Clot Increasing",
                        "Clots Present",
                        "No Clot Present",
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
                    pl.col("VALUE").is_in("New Filter", "Reinitiated"),
                )
                .then(1)
                .when(pl.col("ITEMID") == 665, pl.col("VALUE") == "Initiated")
                .then(1)
                .otherwise(0)
                .alias("RRT_start"),
                # Below indicates that the current instance of CRRT has ended
                pl.when(
                    pl.col("ITEMID") == 224146,
                    pl.col("VALUE").is_in("Discontinued", "Recirculating"),
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
                .otherwise(0)
                .alias("RRT_end"),
            )
            .group_by("ICUSTAY_ID", "CHARTTIME")
            .agg(
                pl.col("RRT").max(),
                pl.col("RRT_start").max(),
                pl.col("RRT_end").max(),
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
                .alias("num")
            )
            # now we can isolate to just rows with settings
            # (before we had rows with start/end flags)
            # this removes any null values for NewCRRT
            .filter(
                pl.col("RRT_start") == 1,
                pl.col("RRT") == 1,
                pl.col("RRT_end") == 1,
            )
            .group_by("ICUSTAY_ID", "num")
            .agg(
                pl.col("CHARTTIME").min().alias("STARTTIME"),
                pl.col("CHARTTIME").max().alias("ENDTIME"),
            )
            .filter(pl.col("STARTTIME") != pl.col("ENDTIME"))
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
        )

        return (
            RENAL_REPLACEMENT_THERAPY_DURATION.unique()
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
