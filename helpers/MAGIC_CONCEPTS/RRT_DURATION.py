# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script extracts the so called MAGIC CONCEPT "Renal Replacement Therapy Duration" directly from the source datasets.
# The MAGIC CONCEPTS are a set of concepts that are based on the concept dict used in the ricu R package and/or
# available prewritten code snippets where indicated.

import os
import polars as pl

from helpers.MAGIC_CONCEPTS.MAGIC_CONCEPTS import MAGIC_CONCEPTS
from helpers.A_extract.A_extract_eicu import EICUExtractor


class RENAL_REPLACEMENT_THERAPY_DURATION(MAGIC_CONCEPTS):
    def __init__(self, paths, datasets):
        super().__init__(paths, datasets)

    def RENAL_REPLACEMENT_THERAPY_DURATION(self):
        """
        Returns the magic concept RENAL_REPLACEMENT_THERAPY_DURATION

        Description:
        This concept is used to determine whether a patient received any antibiotics during the ICU stay.

        Returns a DataFrame with the following columns:
        - ICU stay ID
        - renal replacement therapy type "Renal Replacement Therapy Type", one of
            - "CVVH" (Continuous venovenous hemofiltration),
            - "CAVHD" (Continuous arteriovenous hemodialysis),
            - "CVVHD" (Continuous venovenous hemodialysis),
            - "CVVHDF" (Continuous venovenous hemodiafiltration)
            - "IHD" (Intermittent hemodialysis)
            - "Peritoneal dialysis"
            - "SCUF" (Slow continuous ultra filtration)
            - "SLED" (Sustained low-efficiency dialysis)
            - None (if the type could not be determined)
        - renal replacement therapy start "Renal Replacement Therapy Start Relative to Admission (seconds)"
        - renal replacement therapy end "Renal Replacement Therapy End Relative to Admission (seconds)"
        - renal replacement therapy duration "Renal Replacement Therapy Duration (hours)"

        :return: DataFrame
        :rtype: pl.DataFrame
        """

        # region eICU
        # print("MAGIC_CONCEPTS: Renal Replacement Therapy Duration - eICU")
        eicu_RENAL_REPLACEMENT_THERAPY_DURATION = (
            EICUExtractor(self.paths, self.datasets)
            .extract_treatments(verbose=False)
            .rename(
                {
                    self.column_names[
                        "procedure_start_col"
                    ]: "Renal Replacement Therapy Start Relative to Admission (seconds)",
                    self.column_names[
                        "procedure_end_col"
                    ]: "Renal Replacement Therapy End Relative to Admission (seconds)",
                    self.column_names["procedure_description_col"]: "RRT Type",
                }
            )
            .filter(
                pl.col("RRT Type").str.contains("Renal - Dialysis"),
                pl.col("RRT Type")
                .str.contains_any(["Arteriovenous Shunt", "Venous Catheter"])
                .not_(),
            )
            .with_columns(
                pl.when(pl.col("RRT Type").str.contains("C A V H D"))
                .then(pl.lit("CAVHD"))
                .when(pl.col("RRT Type").str.contains("C V V H"))
                .then(pl.lit("CVVH"))
                .when(pl.col("RRT Type").str.contains("C V V H D"))
                .then(pl.lit("CVVHD"))
                .when(pl.col("RRT Type").str.contains("Hemodialysis"))
                .then(pl.lit("CVVHDF"))
                .when(pl.col("RRT Type").str.contains("Peritoneal Dialysis"))
                .then(pl.lit("Peritoneal dialysis"))
                .when(pl.col("RRT Type").str.contains("Ultrafiltration"))
                .then(pl.lit("SCUF"))
                .when(pl.col("RRT Type").str.contains("SLED"))
                .then(pl.lit("SLED"))
                .otherwise(None)
                .alias("Renal Replacement Therapy Type"),
                (
                    pl.col(
                        "Renal Replacement Therapy End Relative to Admission (seconds)"
                    )
                    - pl.col(
                        "Renal Replacement Therapy Start Relative to Admission (seconds)"
                    )
                )
                .truediv(pl.duration(hours=1).dt.total_seconds())
                .alias("Renal Replacement Therapy Duration (hours)"),
            )
            .pipe(
                self._add_global_id_stay_id,
                "eicu-",
                self.column_names["icu_stay_id_col"],
            )
            .select(
                "Global ICU Stay ID",
                "Renal Replacement Therapy Type",
                "Renal Replacement Therapy Start Relative to Admission (seconds)",
                "Renal Replacement Therapy End Relative to Admission (seconds)",
                "Renal Replacement Therapy Duration (hours)",
            )
        )

        eicu_RENAL_REPLACEMENT_THERAPY_DURATION.collect().write_parquet(
            "eicu_RENAL_REPLACEMENT_THERAPY_DURATION.parquet"
        )

        # endregion

        # region MIMIC-III
        # based on https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/concepts/durations/crrt_durations.sql
        print("MAGIC_CONCEPTS: Renal Replacement Therapy Duration - MIMIC3")

        # get admission times for MIMIC-III
        mimic3_ADMISSIONTIMES = (
            pl.scan_csv(self.mimic3_paths.icustays_path)
            .select("ICUSTAY_ID", "INTIME")
            .with_columns(
                pl.col("INTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
        )

        mimic3_chartevents_metavision_special = [
            # MetaVision ITEMIDs
            # Below require special handling
            224146,  # System Integrity
            225956,  # Reason for CRRT Filter Change
        ]
        mimic3_chartevents_metavision = [
            # Below are settings which indicate CRRT is started/continuing
            224149,  # Access Pressure
            224144,  # Blood Flow (ml/min)
            228004,  # Citrate (ACD-A)
            225183,  # Current Goal
            225977,  # Dialysate Fluid
            224154,  # Dialysate Rate
            224151,  # Effluent Pressure
            224150,  # Filter Pressure
            225958,  # Heparin Concentration (units/mL)
            224145,  # Heparin Dose (per hour)
            224191,  # Hourly Patient Fluid Removal
            228005,  # PBP (Prefilter) Replacement Rate
            228006,  # Post Filter Replacement Rate
            225976,  # Replacement Fluid
            224153,  # Replacement Rate
            224152,  # Return Pressure
            226457,  # Ultrafiltrate Output
        ]
        mimic3_chartevents_carevue_special = [
            # CareVue ITEMIDs
            # Below require special handling
            665,  # System integrity
            147,  # Dialysate Infusing
            612,  # Replace.Fluid Infuse
        ]
        mimic3_chartevents_carevue = [
            # Below are settings which indicate CRRT is started/continuing
            29,  # Access mmHg
            173,  # Effluent Press mmHg
            192,  # Filter Pressure mmHg
            624,  # Return Pressure mmHg
            79,  # Blood Flow ml/min
            142,  # Current Goal
            146,  # Dialysate Flow ml/hr
            611,  # Replace Rate ml/hr
            5683,  # Hourly PFR
        ]
        mimic3_chartevents = (
            mimic3_chartevents_metavision_special
            + mimic3_chartevents_metavision
            + mimic3_chartevents_carevue_special
            + mimic3_chartevents_carevue
        )

        mimic3_CRRT_SETTINGS = (
            pl.scan_csv(self.mimic3_paths.chartevents_path, schema_overrides={"VALUE": str})
            .filter(
                pl.col("ITEMID").is_in(mimic3_chartevents),
                pl.col("VALUE").is_not_null(),
            )
            .with_columns(
                pl.coalesce(pl.col("VALUENUM"), pl.lit(1)).alias("VALUENUM")
            )
            .with_columns(
                pl.col("CHARTTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.when(
                    pl.col("ITEMID").is_in(
                        mimic3_chartevents_metavision
                        + mimic3_chartevents_carevue
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
                pl.when(
                    pl.col("ITEMID") == 224146,
                    pl.col("VALUE").is_in(["New Filter", "Reinitiated"]),
                )
                .then(1)
                .when(pl.col("ITEMID") == 665, pl.col("VALUE") == "Initiated")
                .then(1)
                .otherwise(0)
                .alias("RRT_start"),
                pl.when(
                    pl.col("ITEMID") == 224146,
                    pl.col("VALUE").is_in(["Discontinued", "Recirculating"]),
                )
                .then(1)
                .when(
                    pl.col("ITEMID") == 665,
                    (pl.col("VALUE") == "Clotted")
                    | (pl.col("VALUE") == "DC'D"),
                )
                .then(1)
                .when(pl.col("ITEMID") == 225956)
                .then(1)
                .otherwise(0)
                .alias("RRT_end"),
            )
            .group_by("ICUSTAY_ID", "CHARTTIME")
            .max()
        )

        mimic3_VD_LAG = (
            mimic3_CRRT_SETTINGS.with_columns(
                pl.when((pl.col("RRT") == 1) | (pl.col("RRT_end") == 1))
                .then(1)
                .otherwise(0)
                .alias("CASE"),
            )
            .sort("ICUSTAY_ID", "CHARTTIME")
            .with_columns(
                pl.col("CHARTTIME")
                .shift(1)
                .over("ICUSTAY_ID", "CASE")
                .alias("CHARTTIME_PREV_ROW"),
                pl.col("RRT_end")
                .shift(1)
                .over("ICUSTAY_ID", "CASE")
                .alias("RRT_ENDED_PREV_ROW"),
            )
            .drop("CASE")
        )

        mimic3_VD1 = mimic3_VD_LAG.with_columns(
            # now we determine if the current event is a new instantiation
            pl.when(pl.col("RRT_start") == 1)
            .then(1)
            # if there is an end flag, we mark any subsequent event as new
            # note the end is *not* a new event, the *subsequent* row is so here we output 0
            .when(pl.col("RRT_end") == 1)
            .then(0)
            .when(pl.col("RRT_ENDED_PREV_ROW") == 1)
            .then(1)
            # if there is less than 2 hours between CRRT settings, we do not treat this as a new CRRT event
            .when(
                (pl.col("CHARTTIME") - pl.col("CHARTTIME_PREV_ROW")).le(
                    pl.duration(hours=2)
                )
            )
            .then(0)
            .otherwise(1)
            .alias("NewCRRT")
        )

        mimic3_VD2 = mimic3_VD1.with_columns(
            # create a cumulative sum of the instances of new CRRT
            # this results in a monotonically increasing integer assigned to each CRRT
            pl.when(
                (pl.col("RRT_start") == 1)
                | (pl.col("RRT") == 1)
                | (pl.col("RRT_end") == 1)
            )
            .then(
                pl.col("NewCRRT").sort_by("CHARTTIME").sum().over("ICUSTAY_ID")
            )
            .otherwise(None)
            .alias("NUM")
        )# .drop_nulls("NUM")

        mimic3_FIN = (
            mimic3_VD2.group_by("ICUSTAY_ID", "NUM")
            .agg(
                pl.col("CHARTTIME").min().alias("STARTTIME"),
                pl.col("CHARTTIME").max().alias("ENDTIME"),
            )
            .with_columns(
                (pl.col("ENDTIME") - pl.col("STARTTIME"))
                .truediv(pl.duration(hours=1))
                .alias("DURATION_HOURS")
            )
            # .filter(pl.col("DURATION_HOURS") > 0)
        )

        mimic3_RENAL_REPLACEMENT_THERAPY_DURATION = (
            mimic3_FIN.join(mimic3_ADMISSIONTIMES, on="ICUSTAY_ID", how="left")
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
                pl.col("DURATION_HOURS").alias(
                    "Renal Replacement Therapy Duration (hours)"
                ),
            )
            # .select(
            #     "ICUSTAY_ID",
            #     "Renal Replacement Therapy Start Relative to Admission (seconds)",
            #     "Renal Replacement Therapy End Relative to Admission (seconds)",
            #     "Renal Replacement Therapy Duration (hours)",
            # )
            .pipe(self._add_global_id_stay_id, "mimic3-", "ICUSTAY_ID")
        )

        mimic3_RENAL_REPLACEMENT_THERAPY_DURATION.collect().write_parquet(
            "mimic3_RENAL_REPLACEMENT_THERAPY_DURATION.parquet"
        )

        # endregion

        # region MIMIC-IV
        # based on https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/treatment/rrt.sql
        print("MAGIC_CONCEPTS: Renal Replacement Therapy Duration - MIMIC4")

        # get admission times for MIMIC-IV
        mimic4_ADMISSIONTIMES = (
            pl.scan_csv(self.mimic4_paths.icustays_path)
            .select("stay_id", "intime")
            .with_columns(
                pl.col("intime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
        )

        mimic4_chartevents_dialysis_present = [
            # checkboxes
            226118,  # Dialysis Catheter placed in outside facility
            227357,  # Dialysis Catheter Dressing Occlusive
            225725,  # Dialysis Catheter Tip Cultured
            # numeric data
            226499,  # Hemodialysis Output
            224154,  # Dialysate Rate
            225810,  # Dwell Time (Peritoneal Dialysis)
            225959,  # Medication Added Amount  #1 (Peritoneal Dialysis)
            227639,  # Medication Added Amount  #2 (Peritoneal Dialysis)
            225183,  # Current Goal
            227438,  # Volume not removed
            224191,  # Hourly Patient Fluid Removal
            225806,  # Volume In (PD)
            225807,  # Volume Out (PD)
            228004,  # Citrate (ACD-A)
            228005,  # PBP (Prefilter) Replacement Rate
            228006,  # Post Filter Replacement Rate
            224144,  # Blood Flow (ml/min)
            224145,  # Heparin Dose (per hour)
            224149,  # Access Pressure
            224150,  # Filter Pressure
            224151,  # Effluent Pressure
            224152,  # Return Pressure
            224153,  # Replacement Rate
            224404,  # ART Lumen Volume
            224406,  # VEN Lumen Volume
            226457,  # Ultrafiltrate Output
            # text fields
            224135,  # Dialysis Access Site
            224139,  # Dialysis Site Appearance
            224146,  # System Integrity
            225323,  # Dialysis Catheter Site Appear
            225740,  # Dialysis Catheter Discontinued
            225776,  # Dialysis Catheter Dressing Type
            225951,  # Peritoneal Dialysis Fluid Appearance
            225952,  # Medication Added #1 (Peritoneal Dialysis)
            225953,  # Solution (Peritoneal Dialysis)
            225954,  # Dialysis Access Type
            225956,  # Reason for CRRT Filter Change
            225958,  # Heparin Concentration (units/mL)
            225961,  # Medication Added Units #1 (Peritoneal Dialysis)
            225963,  # Peritoneal Dialysis Catheter Type
            225965,  # Peritoneal Dialysis Catheter Status
            225976,  # Replacement Fluid
            225977,  # Dialysate Fluid
            227124,  # Dialysis Catheter Type | Access Lines - Invasive
            227290,  # CRRT mode
            227638,  # Medication Added #2 (Peritoneal Dialysis)
            227640,  # Medication Added Units #2 (Peritoneal Dialysis)
            227753,  # Dialysis Catheter Placement Confirmed by X-ray
        ]
        mimic4_chartevents_dialysis_active = [
            226499,  # Hemodialysis Output
            224154,  # Dialysate Rate
            225183,  # Current Goal
            227438,  # Volume not removed
            224191,  # Hourly Patient Fluid Removal
            225806,  # Volume In (PD)
            225807,  # Volume Out (PD)
            228004,  # Citrate (ACD-A)
            228005,  # PBP (Prefilter) Replacement Rat
            228006,  # Post Filter Replacement Rate
            224144,  # Blood Flow (ml/min)
            224145,  # Heparin Dose (per hour)
            224153,  # Replacement Rate
            226457,  # Ultrafiltrate Output
        ]
        mimic4_chartevents_dialysis_mode = [227290]
        mimic4_chartevents_dialysis_mode_peritoneal = [
            225810,  # Dwell Time (Peritoneal Dialysis)
            225806,  # Volume In (PD)
            225807,  # Volume Out (PD)
            225810,  # Dwell Time (Peritoneal Dialysis)
            227639,  # Medication Added Amount  #2 (Peritoneal Dialysis)
            225959,  # Medication Added Amount  #1 (Peritoneal Dialysis)
            225951,  # Peritoneal Dialysis Fluid Appearance
            225952,  # Medication Added #1 (Peritoneal Dialysis)
            225961,  # Medication Added Units #1 (Peritoneal Dialysis)
            225953,  # Solution (Peritoneal Dialysis)
            225963,  # Peritoneal Dialysis Catheter Type
            225965,  # Peritoneal Dialysis Catheter Status
            227638,  # Medication Added #2 (Peritoneal Dialysis)
            227640,  # Medication Added Units #2 (Peritoneal Dialysis)
        ]
        mimic4_chartevents_dialysis_mode_ihd = [226499]
        mimic4_procedureevents = [
            225441,  # Hemodialysis
            225802,  # Dialysis - CRRT
            225803,  # Dialysis - CVVHD
            225805,  # Peritoneal Dialysis
            224270,  # Dialysis Catheter
            225809,  # Dialysis - CVVHDF
            225955,  # Dialysis - SCUF
            225436,  # CRRT Filter Change
        ]

        mimic4_RENAL_REPLACEMENT_THERAPY_PRESENCE = (
            pl.scan_csv(self.mimic4_paths.chartevents_path)
            .select("stay_id", "charttime", "itemid", "value")
            # Filter for renal replacement therapy IDs
            .filter(
                pl.col("itemid").is_in(
                    mimic4_chartevents_dialysis_present
                    + mimic4_chartevents_dialysis_active
                    + mimic4_chartevents_dialysis_mode
                    + mimic4_chartevents_dialysis_mode_peritoneal
                    + mimic4_chartevents_dialysis_mode_ihd
                )
            )
            # .cast({"itemid": str})
            # replace renal replacement therapy concepts
            .with_columns(
                (
                    pl.col("itemid").is_in(mimic4_chartevents_dialysis_present)
                    & pl.col("value").is_not_null()
                )
                .fill_null(False)
                .alias("dialysis_present"),
                (
                    pl.col("itemid").is_in(mimic4_chartevents_dialysis_active)
                    & pl.col("value").is_not_null()
                )
                .fill_null(False)
                .alias("dialysis_active"),
                pl.when(
                    pl.col("itemid").is_in(mimic4_chartevents_dialysis_mode)
                )
                .then(pl.col("value"))
                .when(
                    pl.col("itemid").is_in(
                        mimic4_chartevents_dialysis_mode_peritoneal
                    )
                )
                .then(pl.lit("Peritoneal Dialysis"))
                .when(
                    pl.col("itemid").is_in(mimic4_chartevents_dialysis_mode_ihd)
                )
                .then(pl.lit("IHD"))
                .otherwise(None)
                .alias("dialysis_type"),
            )
            .select(
                "stay_id",
                "charttime",
                "dialysis_present",
                "dialysis_active",
                "dialysis_type",
            )
        )

        mimic4_RENAL_REPLACEMENT_THERAPY_DURATION_INPUTEVENTS = (
            pl.scan_csv(self.mimic4_paths.inputevents_path)
            .select("stay_id", "starttime", "endtime", "itemid", "amount")
            .filter(
                pl.col("itemid").is_in(
                    [
                        227536,  # KCl (CRRT) Medications	inputevents_mv	Solution
                        227525,  # Calcium Gluconate (CRRT)	Medications	inputevents_mv	Solutio
                    ]
                ),
                pl.col("amount") > 0,
            )
            .with_columns(
                pl.lit(True).alias("dialysis_present"),
                pl.lit(True).alias("dialysis_active"),
                pl.lit("CRRT").alias("dialysis_type"),
            )
            .select(
                "stay_id",
                "starttime",
                "endtime",
                "dialysis_present",
                "dialysis_active",
                "dialysis_type",
            )
        )

        mimic4_RENAL_REPLACEMENT_THERAPY_DURATION_PROCEDUREEVENTS = (
            pl.scan_csv(self.mimic4_paths.procedureevents_path)
            .select("stay_id", "starttime", "endtime", "itemid", "value")
            .filter(
                pl.col("itemid").is_in(mimic4_procedureevents),
                pl.col("value").is_not_null(),
            )
            .with_columns(
                pl.lit(True).alias("dialysis_present"),
                pl.when(pl.col("itemid").is_in([224270, 225436]))
                .then(False)
                .otherwise(True)
                .alias("dialysis_active"),
                pl.when(pl.col("itemid") == 225441)
                .then(pl.lit("IHD"))
                .when(pl.col("itemid") == 225802)
                .then(pl.lit("CRRT"))
                .when(pl.col("itemid") == 225803)
                .then(pl.lit("CVVHD"))
                .when(pl.col("itemid") == 225805)
                .then(pl.lit("Peritoneal Dialysis"))
                .when(pl.col("itemid") == 225809)
                .then(pl.lit("CVVHDF"))
                .when(pl.col("itemid") == 225955)
                .then(pl.lit("SCUF"))
                .otherwise(pl.lit("other"))
                .alias("dialysis_type"),
            )
            .select(
                "stay_id",
                "starttime",
                "endtime",
                "dialysis_present",
                "dialysis_active",
                "dialysis_type",
            )
        )

        mimic4_RENAL_REPLACEMENT_THERAPY_RANGES = pl.concat(
            [
                mimic4_RENAL_REPLACEMENT_THERAPY_DURATION_INPUTEVENTS,
                mimic4_RENAL_REPLACEMENT_THERAPY_DURATION_PROCEDUREEVENTS,
            ],
            how="vertical",
        ).unique()

        def print_schema(data: pl.LazyFrame) -> pl.LazyFrame:
            print(data.collect_schema())
            return data

        mimic4_RENAL_REPLACEMENT_THERAPY_DURATION = (
            pl.concat(
                [
                    mimic4_RENAL_REPLACEMENT_THERAPY_PRESENCE.filter(
                        pl.col("dialysis_present") == 1
                    ),
                    mimic4_RENAL_REPLACEMENT_THERAPY_RANGES.drop(
                        "endtime"
                    ).rename({"starttime": "charttime"}),
                ]
            )
            .unique()
            .join(
                mimic4_RENAL_REPLACEMENT_THERAPY_RANGES,
                on="stay_id",
                suffix="_",
                how="left",
            )
            .with_columns(
                pl.coalesce(
                    pl.col("dialysis_present"), pl.col("dialysis_present_")
                ),
                pl.coalesce(
                    pl.col("dialysis_active"), pl.col("dialysis_active_")
                ),
                pl.coalesce(pl.col("dialysis_type"), pl.col("dialysis_type_")),
            )
            .join(mimic4_ADMISSIONTIMES, on="stay_id", how="left")
            .pipe(print_schema)
            # Make datetime relative to admission in seconds
            .with_columns(
                pl.col("charttime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("starttime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("endtime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
            .filter(
                pl.col("charttime") > pl.col("starttime"),
                pl.col("charttime") < pl.col("endtime"),
            )
            .with_columns(
                (pl.col("starttime") - pl.col("intime"))
                .truediv(pl.duration(seconds=1))
                .alias(
                    "Renal Replacement Therapy Start Relative to Admission (seconds)"
                ),
                (pl.col("endtime") - pl.col("intime"))
                .truediv(pl.duration(seconds=1))
                .alias(
                    "Renal Replacement Therapy End Relative to Admission (seconds)"
                ),
                (pl.col("endtime") - pl.col("starttime"))
                .truediv(pl.duration(hours=1))
                .alias("Renal Replacement Therapy Duration (hours)"),
                pl.coalesce(
                    pl.col("dialysis_present"), pl.col("dialysis_present_")
                ).alias("Renal Replacement Therapy Present"),
                pl.coalesce(
                    pl.col("dialysis_active"), pl.col("dialysis_active_")
                ).alias("Renal Replacement Therapy Active"),
                pl.coalesce(
                    pl.col("dialysis_type"), pl.col("dialysis_type_")
                ).alias("Renal Replacement Therapy Type"),
            )
            .drop("intime", "starttime", "endtime")
            .pipe(self._add_global_id_stay_id, "mimic4-", "stay_id")
        )

        # mimic4_RENAL_REPLACEMENT_THERAPY_DURATION.sink_parquet(
        #     "mimic4_RENAL_REPLACEMENT_THERAPY_DURATION.parquet"
        # )

        # endregion

        # region UMCdb
        # print("MAGIC_CONCEPTS: Renal Replacement Therapy Duration - UMCdb")

        # umcdb_ADMISSION_TIMES = pl.scan_parquet(
        #     self.umcdb_paths.admissions_path
        # ).select("admissionid", "admittedat")

        # print("umcdb_RENAL_REPLACEMENT_THERAPY_DURATION")

        # umcdb_RENAL_REPLACEMENT_THERAPY_DURATION = (
        #     pl.scan_parquet(self.umcdb_paths.processitems_path)
        #     .join(umcdb_ADMISSION_TIMES, on="admissionid", how="left")
        #     # Filter for renal replacement therapy IDs
        #     .filter(
        #         pl.col("itemid").is_in(
        #             self.ricu_mappings.ricu_concept_dict["mech_vent"][
        #                 "sources"
        #             ]["aumc"][0]["ids"]
        #         )
        #     )
        #     .drop("itemid")
        #     # replace renal replacement therapy concepts
        #     .with_columns(
        #         pl.col("item")
        #         .replace(
        #             {
        #                 "Beademen": "invasive renal replacement therapy",
        #                 "Beademen non-invasief": "non-invasive renal replacement therapy",
        #                 "Tracheostoma": "tracheostomy",
        #             }
        #         )
        #         .cast(str)
        #         .alias("item")
        #     )
        #     # Make datetime relative to admission in seconds
        #     .with_columns(
        #         pl.duration(
        #             milliseconds=(pl.col("start") - pl.col("admittedat"))
        #         )
        #         .dt.total_seconds()
        #         .alias("start"),
        #         pl.duration(
        #             milliseconds=(pl.col("stop") - pl.col("admittedat"))
        #         )
        #         .dt.total_seconds()
        #         .alias("stop"),
        #         pl.duration(milliseconds=pl.col("stop") - pl.col("start"))
        #         .truediv(pl.duration(hours=1))
        #         .alias("duration"),
        #     )
        #     .drop("admittedat")
        #     # Rename columns
        #     .rename(
        #         {
        #             "item": "dialysis_type",
        #             "start": "Renal Replacement Therapy Start Relative to Admission (seconds)",
        #             "stop": "Renal Replacement Therapy End Relative to Admission (seconds)",
        #             "duration": "Renal Replacement Therapy Duration (hours)",
        #         }
        #     )
        #     .pipe(self._add_global_id_stay_id, "umcdb-", "admissionid")
        # )
        # endregion

        # region ALL
        print("MAGIC_CONCEPTS: Renal Replacement Therapy Duration")

        RENAL_REPLACEMENT_THERAPY_DURATION = (
            pl.concat(
                [
                    eicu_RENAL_REPLACEMENT_THERAPY_DURATION,
                    # hirid_RENAL_REPLACEMENT_THERAPY_DURATION,
                    mimic3_RENAL_REPLACEMENT_THERAPY_DURATION,
                    mimic4_RENAL_REPLACEMENT_THERAPY_DURATION,
                    # sicdb_RENAL_REPLACEMENT_THERAPY_DURATION,
                    # umcdb_RENAL_REPLACEMENT_THERAPY_DURATION,
                ],
                how="diagonal_relaxed",
            )
            # .filter(pl.col("Renal Replacement Therapy Duration (hours)") > 0)
            .unique()
            .select(
                "Global ICU Stay ID",
                "dialysis_type",
                "Renal Replacement Therapy Start Relative to Admission (seconds)",
                "Renal Replacement Therapy End Relative to Admission (seconds)",
                "Renal Replacement Therapy Duration (hours)",
            )
            .with_columns(
                pl.col("Renal Replacement Therapy Duration (hours)").round(2)
            )
        )
        # endregion

        return RENAL_REPLACEMENT_THERAPY_DURATION

    # region helpers
    def _add_global_id_stay_id(
        self, data: pl.LazyFrame, source_dataset: str, stay_id_col: str
    ) -> pl.LazyFrame:
        return data.with_columns(
            # add global ICU stay ID
            pl.concat_str([pl.lit(source_dataset), pl.col(stay_id_col)]).alias(
                self.column_names["global_icu_stay_id_col"]
            )
        ).drop(stay_id_col)

    # endregion
