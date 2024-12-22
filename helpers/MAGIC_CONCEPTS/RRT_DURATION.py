# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script extracts the so called MAGIC CONCEPT "Renal Replacement Therapy Duration" directly from the source datasets.
# The MAGIC CONCEPTS are a set of concepts that are based on the concept dict used in the ricu R package and/or
# available prewritten code snippets where indicated.

import os
import polars as pl

from helpers.MAGIC_CONCEPTS.MAGIC_CONCEPTS import MAGIC_CONCEPTS


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
        - renal replacement therapy type "Renal Replacement Therapy Type" (one of
            - "CVVH" (Continuous venovenous hemofiltration),
            - "CVVHD" (Continuous venovenous hemodialysis),
            - "CVVHDF" (Continuous venovenous hemodiafiltration)
            - "IHD" (Intermittent hemodialysis)
            - "peritoneal" (Peritoneal dialysis)
            - "SCUF" (Slow continuous ultra filtration)
            - "other"
          )
        - renal replacement therapy start "Renal Replacement Therapy Start Relative to Admission (seconds)"
        - renal replacement therapy end "Renal Replacement Therapy End Relative to Admission (seconds)"
        - renal replacement therapy duration "Renal Replacement Therapy Duration (hours)"

        :return: DataFrame
        :rtype: pl.DataFrame
        """

        # region eICU
        # # print("MAGIC_CONCEPTS: Renal Replacement Therapy Duration - eICU")
        # eicu_RENAL_REPLACEMENT_THERAPY_DURATION = (
        #     pl.scan_csv(self.eicu_paths.respiratoryCare_path)
        #     # ventstartoffset and ventendoffset seem not include full renal replacement therapy duration
        #     .select(
        #         "patientunitstayid",
        #         "priorventstartoffset",
        #         "priorventendoffset",
        #     )
        #     .with_columns(
        #         # reltimes in eICU are in minutes
        #         (pl.col("priorventstartoffset") * 60).alias(
        #             "Renal Replacement Therapy Start Relative to Admission (seconds)"
        #         ),
        #         (pl.col("priorventendoffset") * 60).alias(
        #             "Renal Replacement Therapy End Relative to Admission (seconds)"
        #         ),
        #     )
        #     .drop("priorventstartoffset", "priorventendoffset")
        #     .with_columns(
        #         # add duration
        #         pl.duration(
        #             seconds=(
        #                 pl.col(
        #                     "Renal Replacement Therapy End Relative to Admission (seconds)"
        #                 )
        #                 - pl.col(
        #                     "Renal Replacement Therapy Start Relative to Admission (seconds)"
        #                 )
        #             )
        #         )
        #         .truediv(pl.duration(hours=1))
        #         .alias("Renal Replacement Therapy Duration (hours)")
        #     )
        # ).pipe(self._add_global_id_stay_id, "eicu-", "patientunitstayid")

        # region HiRID
        # # print("MAGIC_CONCEPTS: Renal Replacement Therapy Duration - HiRID")

        # # get admission times for HiRID
        # hirid_ADMISSIONTIMES = (
        #     pl.scan_csv(self.hirid_paths.general_table_path)
        #     .select("patientid", "admissiontime")
        #     .with_columns(
        #         pl.col("patientid").cast(str),
        #         pl.col("admissiontime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
        #     )
        # )

        # # Scan all files in the timeseries folder
        # hirid_RENAL_REPLACEMENT_THERAPY_DURATION = pl.LazyFrame()

        # for file in os.listdir(self.hirid_paths.timeseries_path):
        #     hirid_timeseries = (
        #         pl.scan_parquet(self.hirid_paths.timeseries_path + file)
        #         .select("datetime", "patientid", "value", "variableid")
        #         .cast({"datetime": str, "patientid": str})
        #         # Filter for renal replacement therapy IDs
        #         .filter(
        #             pl.col("variableid")
        #             == self.ricu_mappings.ricu_concept_dict["mech_vent"][
        #                 "sources"
        #             ]["hirid"][0]["ids"]
        #         )
        #         .drop("variableid")
        #         # replace renal replacement therapy concepts
        #         .with_columns(
        #             pl.col("value")
        #             .cast(int)
        #             .cast(str)
        #             .replace(
        #                 {
        #                     "1": "invasive renal replacement therapy",
        #                     "2": "tracheostomy",
        #                     "3": "non-invasive renal replacement therapy",
        #                     "4": "non-invasive renal replacement therapy",
        #                     "5": "non-invasive renal replacement therapy",
        #                     "6": "other",  # TODO: check if this is correct
        #                 }
        #             )
        #             .alias("dialysis_type"),
        #             pl.col("datetime").str.to_datetime("%Y-%m-%d %H:%M:%S%.9f"),
        #         )
        #         # Make datetime relative to admission in seconds
        #         .join(hirid_ADMISSIONTIMES, on="patientid", how="left")
        #         .with_columns(
        #             (pl.col("datetime") - pl.col("admissiontime"))
        #             .dt.total_seconds()
        #             .alias(
        #                 "Renal Replacement Therapy Start Relative to Admission (seconds)"
        #             )
        #         )
        #         .drop("admissiontime", "datetime", "value")
        #         # Rename columns
        #     )

        #     hirid_RENAL_REPLACEMENT_THERAPY_DURATION = pl.concat(
        #         [hirid_RENAL_REPLACEMENT_THERAPY_DURATION, hirid_timeseries],
        #         how="diagonal_relaxed",
        #     )

        # hirid_RENAL_REPLACEMENT_THERAPY_DURATION = (
        #     hirid_RENAL_REPLACEMENT_THERAPY_DURATION.pipe(
        #         self._add_global_id_stay_id, "hirid-", "patientid"
        #     )
        # )
        # # endregion

        # region MIMIC-III
        # print("MAGIC_CONCEPTS: Renal Replacement Therapy Duration - MIMIC3")

        # get admission times for MIMIC-III
        mimic3_ADMISSIONTIMES = (
            pl.scan_csv(self.mimic3_paths.icustays_path)
            .select("ICUSTAY_ID", "INTIME")
            .with_columns(
                pl.col("INTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
        )

        print("mimic3_RENAL_REPLACEMENT_THERAPY_DURATION")

        mimic3_RENAL_REPLACEMENT_THERAPY_DURATION = (
            pl.scan_csv(self.mimic3_paths.procedureevents_mv_path)
            .select("ICUSTAY_ID", "STARTTIME", "ENDTIME", "ITEMID")
            .join(mimic3_ADMISSIONTIMES, on="ICUSTAY_ID", how="left")
            # Filter for renal replacement therapy IDs
            .filter(
                pl.col("ITEMID").is_in(
                    self.ricu_mappings.ricu_concept_dict["mech_vent"][
                        "sources"
                    ]["miiv"][0]["ids"]
                )
            )
            # .cast({"ITEMID": str})
            # replace renal replacement therapy concepts
            .with_columns(
                pl.col("ITEMID")
                .replace(
                    {
                        225792: "invasive renal replacement therapy",
                        225794: "non-invasive renal replacement therapy",
                    }
                )
                .cast(str)
                .alias("dialysis_type")
            )
            # Make datetime relative to admission in seconds
            .with_columns(
                pl.col("STARTTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("ENDTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
            .with_columns(
                (pl.col("STARTTIME") - pl.col("INTIME"))
                .truediv(pl.duration(seconds=1))
                .alias(
                    "Renal Replacement Therapy Start Relative to Admission (seconds)"
                ),
                (pl.col("ENDTIME") - pl.col("INTIME"))
                .truediv(pl.duration(seconds=1))
                .alias(
                    "Renal Replacement Therapy End Relative to Admission (seconds)"
                ),
                (pl.col("ENDTIME") - pl.col("STARTTIME"))
                .truediv(pl.duration(hours=1))
                .alias("Renal Replacement Therapy Duration (hours)"),
            )
            .drop("INTIME", "STARTTIME", "ENDTIME", "ITEMID")
            .pipe(self._add_global_id_stay_id, "mimic3-", "ICUSTAY_ID")
        )
        # endregion

        # region MIMIC-IV
        # based on https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/treatment/rrt.sql
        # print("MAGIC_CONCEPTS: Renal Replacement Therapy Duration - MIMIC4")

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
                .then(pl.lit("peritoneal"))
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

        print("mimic4_RENAL_REPLACEMENT_THERAPY_DURATION_INPUTEVENTS")

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

        print("mimic4_RENAL_REPLACEMENT_THERAPY_RANGES")

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
            )
            .drop("intime", "starttime", "endtime")
            .pipe(self._add_global_id_stay_id, "mimic4-", "stay_id")
        )

        print("mimic4_RENAL_REPLACEMENT_THERAPY_DURATION collect")

        mimic4_RENAL_REPLACEMENT_THERAPY_DURATION.sink_parquet(
            "mimic4_RENAL_REPLACEMENT_THERAPY_DURATION.parquet"
        )

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
                    # eicu_RENAL_REPLACEMENT_THERAPY_DURATION,
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
