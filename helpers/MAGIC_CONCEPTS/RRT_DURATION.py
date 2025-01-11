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

        # eicu_RENAL_REPLACEMENT_THERAPY_DURATION.collect().write_parquet(
        #     "eicu_RENAL_REPLACEMENT_THERAPY_DURATION.parquet"
        # )

        # endregion

        # region MIMIC
        # metavision itemids for both MIMIC-III and MIMIC-IV
        mimic_chartevents_dialysis_present = [
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
        mimic_chartevents_dialysis_active = [
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
        mimic_chartevents_dialysis_mode = [227290]
        mimic_chartevents_dialysis_mode_peritoneal = [
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
        mimic_chartevents_dialysis_mode_ihd = [226499]
        mimic_inputevents = [
            227536,  # KCl (CRRT) Medications	inputevents_mv	Solution
            227525,  # Calcium Gluconate (CRRT)	Medications	inputevents_mv	Solutio
        ]
        mimic_procedureevents = [
            225441,  # Hemodialysis
            225802,  # Dialysis - CRRT
            225803,  # Dialysis - CVVHD
            225805,  # Peritoneal Dialysis
            224270,  # Dialysis Catheter
            225809,  # Dialysis - CVVHDF
            225955,  # Dialysis - SCUF
            225436,  # CRRT Filter Change
        ]

        # region MIMIC-III
        # based on https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/concepts/pivot/pivoted_rrt.sql
        # print("MAGIC_CONCEPTS: Renal Replacement Therapy Duration - MIMIC3")

        # get admission times for MIMIC-III
        mimic3_ADMISSIONTIMES = (
            pl.scan_csv(self.mimic3_paths.icustays_path)
            .select("ICUSTAY_ID", "INTIME")
            .with_columns(
                pl.col("INTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
        )

        mimic3_chartevents_dialysis_present = [
            146,  # Dialysate Flow ml/hr
            147,  # Dialysate Infusing
            148,  # Dialysis Access Site
            149,  # Dialysis Access Type
            150,  # Dialysis Machine
            151,  # Dialysis Site Appear
            152,  # Dialysis Type
            # below require special handling
            582,  # Procedures
            # below indicate existence of a dialysis line
            229,  # INV Line#1 [Type]
            235,  # INV Line#2 [Type]
            241,  # INV Line#3 [Type]
            247,  # INV Line#4 [Type]
            253,  # INV Line#5 [Type]
            259,  # INV Line#6 [Type]
            265,  # INV Line#7 [Type]
            271,  # INV Line#8 [Type]
        ]
        mimic3_chartevents_dialysis_active = [
            146,  # Dialysate Flow ml/hr
            # below require special handling
            582,  # Procedures
            147,  # Dialysate Infusing
            225965,  # Peritoneal Dialysis Catheter Status # NOTE: this is not handled in MIMIC-IV
        ]
        mimic3_chartevents_dialysis_mode1 = [152]  # Dialysis Type
        mimic3_chartevents_dialysis_mode2 = [582]  # Procedures
        mimic3_chartevents_dialysis_mode = (
            mimic3_chartevents_dialysis_mode1
            + mimic3_chartevents_dialysis_mode2
        )
        mimic3_inputevents_cv_dialysis_active = [
            44954,  # CVVHDF
        ]
        mimic3_inputevents_cv_dialysis_mode_peritoneal = [
            40788,  # PD dialysate in
            41063,  # PD Dialysate Intake
            41307,  # Peritoneal Dialysate
            43829,  # PERITONEAL DIALYSATE
            44698,  # peritoneal dialysate
            46720,  # PD Dialysate
        ]
        mimic3_inputevents_cv_dialysis_mode_cvvh = [
            45352,  # CA GLUC for CVVH
            45353,  # KCL for CVVH
        ]
        mimic3_inputevents_cv_dialysis_mode_cvvhd = [
            45268,  # CALCIUM FOR CVVHD
            46769,  # cvvdh rescue line
            46773,  # CVVHD NS line flush
        ]
        mimic3_inputevents_cv_dialysis_mode_cvvhdf = [
            46012,  # CA GLUC CVVHDF
            46013,  # KCL CVVHDF
            46172,  # CVVHDF CA GLUC
            46173,  # CVVHDF KCL
        ]
        mimic3_inputevents_cv_other = [
            40907,  # dialysate
            41147,  # Dialysate instilled
            41460,  # capd dialysate
            41620,  # dialysate in
            41711,  # CAPD dialysate dwell
            41791,  # 2.5% dialysate in
            41792,  # 1.5% dialysate
            42562,  # pos. dialysate intak
            44037,  # Dialysate Instilled
            44188,  # rep.+dialysate
            44526,  # dialysate 1.5% dex
            44527,  # dialysate 2.5%
            44584,  # Dialysate IN
            44591,  # dialysate 4.25%
            44927,  # CRRT HEPARIN
            45157,  # ca+ gtt for cvvh
            46250,  # EBL  CVVH
            46262,  # dialysate 2.5% in
            46292,  # CRRT Irrigation
            46293,  # CRRT Citrate
            46311,  # crrt irrigation
            46389,  # CRRT FLUSH
            46574,  # CRRT rescue line NS
            46681,  # CRRT Rescue Flush
        ]
        mimic3_outputevents_dialysis_active = [41897]  # CVVH OUTPUT FROM OR
        mimic3_outputevents_dialysis_type = [
            40789,  # PD dialysate out
            40910,  # PERITONEAL DIALYSIS
            41069,  # PD Dialysate Output
            44843,  # peritoneal dialysis
            46394,  # Peritoneal dialysis
        ]
        mimic3_outputevents_other = [
            40386,  # hemodialysis
            40425,  # dialysis output
            40426,  # dialysis out
            40507,  # Dialysis out
            40613,  # DIALYSIS OUT
            40624,  # dialysis
            40690,  # DIALYSIS
            40745,  # Dialysis
            40881,  # Hemodialysis
            41016,  # hemodialysis out
            41034,  # dialysis in
            41112,  # Dialysys out
            41250,  # HEMODIALYSIS OUT
            41374,  # Dialysis Out
            41417,  # Hemodialysis Out
            41500,  # hemodialysis output
            41527,  # HEMODIALYSIS
            41623,  # dialysate out
            41635,  # Hemodialysis removal
            41713,  # dialyslate out
            41750,  # dialysis  out
            41829,  # HEMODIALYSIS OUTPUT
            41842,  # Dialysis Output.
            42289,  # dialysis off
            42388,  # DIALYSIS OUTPUT
            42464,  # hemodialysis ultrafe
            42524,  # HemoDialysis
            42536,  # Dialysis output
            42868,  # hemodialysis off
            42928,  # HEMODIALYSIS.
            42972,  # HEMODIALYSIS OFF
            43016,  # DIALYSIS TOTAL OUT
            43052,  # DIALYSIS REMOVED
            43098,  # hemodialysis crystal
            43115,  # dialysis net
            43687,  # crystalloid/dialysis
            43941,  # dialysis/intake
            44027,  # dialysis fluid off
            44085,  # DIALYSIS OFF
            44193,  # Dialysis.
            44199,  # HEMODIALYSIS O/P
            44216,  # Hemodialysis out
            44286,  # Dialysis indwelling
            44567,  # Hemodialysis.
            44845,  # Dialysis fluids
            44857,  # dialysis- fluid off
            44901,  # Dialysis Removed
            44943,  # fluid removed dialys
            45479,  # Dialysis In
            45828,  # Hemo dialysis out
            46230,  # Dialysis 1.5% IN
            46232,  # dialysis flush
            46464,  # Hemodialysis OUT
            46712,  # CALCIUM-DIALYSIS
            46713,  # KCL-10 MEQ-DIALYSIS
            46715,  # Citrate - dialysis
            46741,  # dialysis removed
        ]

        mimic3_RENAL_REPLACEMENT_THERAPY_CHARTEVENTS = (
            pl.scan_csv(
                self.mimic3_paths.chartevents_path,
                schema_overrides={"VALUE": str},
            )
            .select("ICUSTAY_ID", "CHARTTIME", "ITEMID", "VALUE", "ERROR")
            # Filter for renal replacement therapy IDs
            .filter(
                pl.col("ITEMID").is_in(
                    mimic_chartevents_dialysis_present
                    + mimic_chartevents_dialysis_active
                    + mimic_chartevents_dialysis_mode
                    + mimic_chartevents_dialysis_mode_peritoneal
                    + mimic_chartevents_dialysis_mode_ihd
                    + mimic3_chartevents_dialysis_present
                    + mimic3_chartevents_dialysis_active
                    + mimic3_chartevents_dialysis_mode
                ),
                pl.col("VALUE").is_not_null(),
                pl.col("ICUSTAY_ID").is_not_null(),
                pl.col("ERROR") == 0,  # exclude rows marked as error
            )
            .drop("ERROR")
            # replace renal replacement therapy concepts
            .with_columns(
                (
                    pl.col("ITEMID").is_in(
                        mimic_chartevents_dialysis_present
                        + mimic3_chartevents_dialysis_present
                    )
                    & pl.col("VALUE").is_not_null()
                )
                # .fill_null(False)
                .alias("dialysis_present"),
                (
                    pl.col("ITEMID").is_in(
                        mimic_chartevents_dialysis_active
                        + mimic3_chartevents_dialysis_active
                    )
                    & pl.col("VALUE").is_not_null()
                )
                # .fill_null(False)
                .alias("dialysis_active"),
                pl.when(
                    pl.col("ITEMID").is_in(
                        mimic_chartevents_dialysis_mode
                        + mimic3_chartevents_dialysis_mode1
                    )
                )
                .then(pl.col("VALUE"))
                .when(
                    pl.col("ITEMID").is_in(
                        mimic_chartevents_dialysis_mode_peritoneal
                    )
                )
                .then(pl.lit("Peritoneal dialysis"))
                .when(
                    pl.col("ITEMID").is_in(mimic_chartevents_dialysis_mode_ihd)
                )
                .then(pl.lit("IHD"))
                .when(pl.col("ITEMID").is_in(mimic3_chartevents_dialysis_mode2))
                .then(
                    pl.when(pl.col("VALUE").is_in(["CAVH Start", "CAVH D/C"]))
                    .then(pl.lit("CAVH"))
                    .when(pl.col("VALUE").is_in(["CVVHD Start", "CVVHD D/C"]))
                    .then(pl.lit("CVVHD"))
                    .otherwise(None)
                )
                .otherwise(None)
                .alias("dialysis_type"),
            )
            # do special handling for MIMIC-III
            .with_columns(
                pl.when(
                    pl.col("ITEMID").is_in(mimic3_chartevents_dialysis_present)
                )
                .then(
                    pl.when(
                        pl.col("ITEMID") == 582,
                        pl.col("VALUE").is_in(
                            [
                                "CAVH Start",
                                "CVVHD Start",
                                "Hemodialysis st",
                                "CAVH D/C",
                                "CVVHD D/C",
                                "Hemodialysis end",
                                "Peritoneal Dial",
                            ]
                        ),
                    )
                    .then(True)
                    .when(
                        pl.col("ITEMID").is_in(
                            [229, 235, 241, 247, 253, 259, 265, 271]
                        ),
                        pl.col("VALUE") == "Dialysis Line",
                    )
                    .then(True)
                    .otherwise(False)
                )
                .otherwise(pl.col("dialysis_present"))
                .alias("dialysis_present"),
                pl.when(
                    pl.col("ITEMID").is_in(mimic3_chartevents_dialysis_active)
                )
                .then(
                    pl.when(
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
                    .then(True)
                    .when(
                        pl.col("ITEMID") == 582,
                        pl.col("VALUE").is_in(
                            ["CAVH D/C", "CVVHD D/C", "Hemodialysis end"]
                        ),
                    )
                    .then(False)
                    .when(pl.col("ITEMID") == 147, pl.col("VALUE") == "Yes")
                    .then(True)
                    .when(
                        pl.col("ITEMID") == 225965, pl.col("VALUE") == "In use"
                    )
                    .then(True)
                    .otherwise(False)
                )
                .otherwise(pl.col("dialysis_active"))
                .alias("dialysis_active"),
            )
            .select(
                "ICUSTAY_ID",
                "CHARTTIME",
                "dialysis_present",
                "dialysis_active",
                "dialysis_type",
            )
        )

        mimic3_RENAL_REPLACEMENT_THERAPY_INPUTEVENTS_CV = (
            pl.scan_csv(
                self.mimic3_paths.inputevents_cv_path,
                schema_overrides={"AMOUNT": float},
            )
            .select("ICUSTAY_ID", "CHARTTIME", "ITEMID", "AMOUNT")
            .filter(
                pl.col("ITEMID").is_in(
                    mimic3_inputevents_cv_dialysis_active
                    + mimic3_inputevents_cv_dialysis_mode_peritoneal
                    + mimic3_inputevents_cv_dialysis_mode_cvvh
                    + mimic3_inputevents_cv_dialysis_mode_cvvhd
                    + mimic3_inputevents_cv_dialysis_mode_cvvhdf
                    + mimic3_inputevents_cv_other
                ),
                pl.col("AMOUNT") > 0,
            )
            .with_columns(
                pl.lit(True).alias("dialysis_present"),
                pl.col("ITEMID")
                .is_in(mimic3_inputevents_cv_dialysis_active)
                .not_()
                .alias("dialysis_active"),
                pl.when(
                    pl.col("ITEMID").is_in(
                        mimic3_inputevents_cv_dialysis_mode_peritoneal
                    )
                )
                .then(pl.lit("Peritoneal dialysis"))
                .when(
                    pl.col("ITEMID").is_in(
                        mimic3_inputevents_cv_dialysis_mode_cvvh
                    )
                )
                .then(pl.lit("CVVH"))
                .when(
                    pl.col("ITEMID").is_in(
                        mimic3_inputevents_cv_dialysis_mode_cvvhd
                    )
                )
                .then(pl.lit("CVVHD"))
                .when(
                    pl.col("ITEMID").is_in(
                        mimic3_inputevents_cv_dialysis_mode_cvvhdf
                    )
                )
                .then(pl.lit("CVVHDF"))
                .otherwise(None)
                .alias("dialysis_type"),
            )
            .select(
                "ICUSTAY_ID",
                "CHARTTIME",
                "dialysis_present",
                "dialysis_active",
                "dialysis_type",
            )
        )

        mimic3_RENAL_REPLACEMENT_THERAPY_INPUTEVENTS_MV = (
            pl.scan_csv(
                self.mimic3_paths.inputevents_mv_path,
                schema_overrides={"AMOUNT": float},
            )
            .select("ICUSTAY_ID", "STARTTIME", "ENDTIME", "ITEMID", "AMOUNT")
            .filter(
                pl.col("ITEMID").is_in(mimic_inputevents),
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

        mimic3_RENAL_REPLACEMENT_THERAPY_OUTPUTEVENTS = (
            pl.scan_csv(
                self.mimic3_paths.outputevents_path,
                schema_overrides={"VALUE": float},
            )
            .select("ICUSTAY_ID", "CHARTTIME", "ITEMID", "VALUE")
            .filter(
                pl.col("ITEMID").is_in(
                    mimic3_outputevents_dialysis_active
                    + mimic3_outputevents_dialysis_type
                    + mimic3_outputevents_other
                ),
                pl.col("VALUE") > 0,
            )
            .with_columns(
                pl.lit(True).alias("dialysis_present"),
                pl.col("ITEMID")
                .is_in(mimic3_inputevents_cv_dialysis_active)
                .not_()
                .alias("dialysis_active"),
                pl.lit("CRRT").alias("dialysis_type"),
            )
            .select(
                "ICUSTAY_ID",
                "CHARTTIME",
                "dialysis_present",
                "dialysis_active",
                "dialysis_type",
            )
        )

        mimic3_RENAL_REPLACEMENT_THERAPY_PROCEDUREEVENTS_MV = (
            pl.scan_csv(
                self.mimic3_paths.procedureevents_mv_path,
                schema_overrides={"VALUE": str},
            )
            .select("ICUSTAY_ID", "STARTTIME", "ENDTIME", "ITEMID", "VALUE")
            .filter(
                pl.col("ITEMID").is_in(mimic_procedureevents),
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

        mimic3_RENAL_REPLACEMENT_THERAPY_MV_RANGES = pl.concat(
            [
                mimic3_RENAL_REPLACEMENT_THERAPY_INPUTEVENTS_MV,
                mimic3_RENAL_REPLACEMENT_THERAPY_PROCEDUREEVENTS_MV,
            ],
            how="vertical",
        ).unique()

        mimic3_RENAL_REPLACEMENT_THERAPY_DURATION = (
            pl.concat(
                [
                    mimic3_RENAL_REPLACEMENT_THERAPY_CHARTEVENTS.filter(
                        pl.col("dialysis_present") == 1
                    ),
                    mimic3_RENAL_REPLACEMENT_THERAPY_INPUTEVENTS_CV.filter(
                        pl.col("dialysis_present") == 1
                    ),
                    mimic3_RENAL_REPLACEMENT_THERAPY_OUTPUTEVENTS.filter(
                        pl.col("dialysis_present") == 1
                    ),
                    mimic3_RENAL_REPLACEMENT_THERAPY_MV_RANGES.drop(
                        "ENDTIME"
                    ).rename({"STARTTIME": "CHARTTIME"}),
                    mimic3_RENAL_REPLACEMENT_THERAPY_MV_RANGES.drop(
                        "STARTTIME"
                    ).rename({"ENDTIME": "CHARTTIME"}),
                ],
                how="vertical",
            )
            .unique()
            .join(
                mimic3_RENAL_REPLACEMENT_THERAPY_MV_RANGES,
                on="ICUSTAY_ID",
                suffix="_mv",
                how="left",
            )
            .join(mimic3_ADMISSIONTIMES, on="ICUSTAY_ID", how="left")
            # Make datetime relative to admission in seconds
            .with_columns(
                pl.col("CHARTTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("STARTTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("ENDTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
            .filter(
                pl.col("CHARTTIME") > pl.col("STARTTIME"),
                pl.col("CHARTTIME") < pl.col("ENDTIME"),
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
                pl.coalesce(
                    pl.col("dialysis_present_mv"), pl.col("dialysis_present")
                ).alias("Renal Replacement Therapy Present"),
                pl.coalesce(
                    pl.col("dialysis_active_mv"), pl.col("dialysis_active")
                ).alias("Renal Replacement Therapy Active"),
                pl.coalesce(
                    pl.col("dialysis_type_mv"), pl.col("dialysis_type")
                ).alias("Renal Replacement Therapy Type"),
            )
            .drop("INTIME", "STARTTIME", "ENDTIME")
            .pipe(self._add_global_id_stay_id, "mimic3-", "ICUSTAY_ID")
        )

        mimic3_RENAL_REPLACEMENT_THERAPY_DURATION

        # mimic3_RENAL_REPLACEMENT_THERAPY_DURATION.collect().write_parquet(
        #     "mimic3_RENAL_REPLACEMENT_THERAPY_DURATION.parquet"
        # )

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

        mimic4_RENAL_REPLACEMENT_THERAPY_CHARTEVENTS = (
            pl.scan_csv(
                self.mimic4_paths.chartevents_path,
                schema_overrides={"value": str},
            )
            .select("stay_id", "charttime", "itemid", "value")
            # Filter for renal replacement therapy IDs
            .filter(
                pl.col("itemid").is_in(
                    mimic_chartevents_dialysis_present
                    + mimic_chartevents_dialysis_active
                    + mimic_chartevents_dialysis_mode
                    + mimic_chartevents_dialysis_mode_peritoneal
                    + mimic_chartevents_dialysis_mode_ihd
                )
            )
            # replace renal replacement therapy concepts
            .with_columns(
                (
                    pl.col("itemid").is_in(mimic_chartevents_dialysis_present)
                    & pl.col("value").is_not_null()
                )
                # .fill_null(False)
                .alias("dialysis_present"),
                (
                    pl.col("itemid").is_in(mimic_chartevents_dialysis_active)
                    & pl.col("value").is_not_null()
                )
                # .fill_null(False)
                .alias("dialysis_active"),
                pl.when(pl.col("itemid").is_in(mimic_chartevents_dialysis_mode))
                .then(pl.col("value"))
                .when(
                    pl.col("itemid").is_in(
                        mimic_chartevents_dialysis_mode_peritoneal
                    )
                )
                .then(pl.lit("Peritoneal dialysis"))
                .when(
                    pl.col("itemid").is_in(mimic_chartevents_dialysis_mode_ihd)
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

        mimic4_RENAL_REPLACEMENT_THERAPY_INPUTEVENTS = (
            pl.scan_csv(self.mimic4_paths.inputevents_path)
            .select("stay_id", "starttime", "endtime", "itemid", "amount")
            .filter(
                pl.col("itemid").is_in(mimic_inputevents),
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

        mimic4_RENAL_REPLACEMENT_THERAPY_PROCEDUREEVENTS = (
            pl.scan_csv(self.mimic4_paths.procedureevents_path)
            .select("stay_id", "starttime", "endtime", "itemid", "value")
            .filter(
                pl.col("itemid").is_in(mimic_procedureevents),
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
                .then(pl.lit("Peritoneal dialysis"))
                .when(pl.col("itemid") == 225809)
                .then(pl.lit("CVVHDF"))
                .when(pl.col("itemid") == 225955)
                .then(pl.lit("SCUF"))
                .otherwise(None)
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
                mimic4_RENAL_REPLACEMENT_THERAPY_INPUTEVENTS,
                mimic4_RENAL_REPLACEMENT_THERAPY_PROCEDUREEVENTS,
            ],
            how="vertical",
        ).unique()

        mimic4_RENAL_REPLACEMENT_THERAPY_DURATION = (
            pl.concat(
                [
                    mimic4_RENAL_REPLACEMENT_THERAPY_CHARTEVENTS.filter(
                        pl.col("dialysis_present") == 1
                    ),
                    mimic4_RENAL_REPLACEMENT_THERAPY_RANGES.drop(
                        "endtime"
                    ).rename({"starttime": "charttime"}),
                ],
                how="vertical",
            )
            .unique()
            .join(
                mimic4_RENAL_REPLACEMENT_THERAPY_RANGES,
                on="stay_id",
                suffix="_mv",
                how="left",
            )
            .join(mimic4_ADMISSIONTIMES, on="stay_id", how="left")
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
                    pl.col("dialysis_present_mv"), pl.col("dialysis_present")
                ).alias("Renal Replacement Therapy Present"),
                pl.coalesce(
                    pl.col("dialysis_active_mv"), pl.col("dialysis_active")
                ).alias("Renal Replacement Therapy Active"),
                pl.coalesce(
                    pl.col("dialysis_type_mv"), pl.col("dialysis_type")
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
