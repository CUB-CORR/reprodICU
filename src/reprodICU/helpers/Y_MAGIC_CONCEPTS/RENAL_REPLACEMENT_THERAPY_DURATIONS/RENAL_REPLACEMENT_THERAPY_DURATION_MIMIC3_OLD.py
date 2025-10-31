# based on https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/concepts/pivot/pivoted_rrt.sql

import polars as pl

from ..MAGIC_CONCEPTS import MAGIC_CONCEPTS


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
            .collect()
        )

        # region ITEMIDS
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

        # region MIMIC-III ITEMIDS
        chartevents_dialysis_present = [
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
        chartevents_dialysis_active = [
            146,  # Dialysate Flow ml/hr
            # below require special handling
            582,  # Procedures
            147,  # Dialysate Infusing
            225965,  # Peritoneal Dialysis Catheter Status # NOTE: this is not handled in MIMIC-IV
        ]
        chartevents_dialysis_mode1 = [152]  # Dialysis Type
        chartevents_dialysis_mode2 = [582]  # Procedures
        chartevents_dialysis_mode = (
            chartevents_dialysis_mode1 + chartevents_dialysis_mode2
        )
        inputevents_cv_dialysis_active = [
            44954,  # CVVHDF
        ]
        inputevents_cv_dialysis_mode_peritoneal = [
            40788,  # PD dialysate in
            41063,  # PD Dialysate Intake
            41307,  # Peritoneal Dialysate
            43829,  # PERITONEAL DIALYSATE
            44698,  # peritoneal dialysate
            46720,  # PD Dialysate
        ]
        inputevents_cv_dialysis_mode_cvvh = [
            45352,  # CA GLUC for CVVH
            45353,  # KCL for CVVH
        ]
        inputevents_cv_dialysis_mode_cvvhd = [
            45268,  # CALCIUM FOR CVVHD
            46769,  # cvvdh rescue line
            46773,  # CVVHD NS line flush
        ]
        inputevents_cv_dialysis_mode_cvvhdf = [
            46012,  # CA GLUC CVVHDF
            46013,  # KCL CVVHDF
            46172,  # CVVHDF CA GLUC
            46173,  # CVVHDF KCL
        ]
        inputevents_cv_other = [
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
        outputevents_dialysis_active = [41897]  # CVVH OUTPUT FROM OR
        outputevents_dialysis_type = [
            40789,  # PD dialysate out
            40910,  # PERITONEAL DIALYSIS
            41069,  # PD Dialysate Output
            44843,  # peritoneal dialysis
            46394,  # Peritoneal dialysis
        ]
        outputevents_other = [
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

        ##############################################################################
        # pivoted_rrt.sql
        ##############################################################################
        # region CE
        CHARTEVENTS = (
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
                    + chartevents_dialysis_present
                    + chartevents_dialysis_active
                    + chartevents_dialysis_mode
                ),
                pl.col("VALUE").is_not_null(),
                pl.col("ICUSTAY_ID").is_not_null(),
                pl.col("ERROR") == 0,  # exclude rows marked as error
            )
            .drop("ERROR")
            # replace renal replacement therapy concepts
            # first handle metavision itemids
            .with_columns(
                pl.when(
                    pl.col("ITEMID").is_in(
                        mimic_chartevents_dialysis_present
                        + chartevents_dialysis_present
                    ),
                    pl.col("VALUE").is_not_null(),
                )
                .then(1)
                .otherwise(0)
                .alias("dialysis_present"),
                pl.when(
                    pl.col("ITEMID").is_in(
                        mimic_chartevents_dialysis_active
                        + chartevents_dialysis_active
                    ),
                    pl.col("VALUE").is_not_null(),
                )
                .then(1)
                .otherwise(0)
                .alias("dialysis_active"),
                pl.when(
                    pl.col("ITEMID").is_in(
                        mimic_chartevents_dialysis_mode
                        + chartevents_dialysis_mode1
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
                .when(pl.col("ITEMID").is_in(chartevents_dialysis_mode2))
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
            # do special handling for MIMIC-III specific itemids
            .with_columns(
                pl.when(pl.col("ITEMID").is_in(chartevents_dialysis_present))
                .then(
                    pl.when(
                        pl.col("ITEMID") == 582,
                        pl.col("VALUE").is_in(
                            # fmt: off
                            [
                                "CAVH Start", "CVVHD Start", "Hemodialysis st",
                                "CAVH D/C", "CVVHD D/C", "Hemodialysis end",
                                "Peritoneal Dial"
                            ]
                            # fmt: on
                        ),
                    )
                    .then(1)
                    .when(
                        pl.col("ITEMID").is_in(
                            [229, 235, 241, 247, 253, 259, 265, 271]
                        ),
                        pl.col("VALUE") == "Dialysis Line",
                    )
                    .then(1)
                    .otherwise(0)
                )
                .otherwise(pl.col("dialysis_present"))
                .alias("dialysis_present"),
                pl.when(pl.col("ITEMID").is_in(chartevents_dialysis_active))
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
                    .then(1)
                    .when(
                        pl.col("ITEMID") == 582,
                        pl.col("VALUE").is_in(
                            ["CAVH D/C", "CVVHD D/C", "Hemodialysis end"]
                        ),
                    )
                    .then(0)
                    .when(pl.col("ITEMID") == 147, pl.col("VALUE") == "Yes")
                    .then(1)
                    .when(
                        pl.col("ITEMID") == 225965, pl.col("VALUE") == "In use"
                    )
                    .then(1)
                    .otherwise(0)
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

        # region IE_CV
        INPUTEVENTS_CV = (
            pl.scan_csv(
                self.mimic3_paths.inputevents_cv_path,
                schema_overrides={"AMOUNT": float},
            )
            .select("ICUSTAY_ID", "CHARTTIME", "ITEMID", "AMOUNT")
            .filter(
                pl.col("ITEMID").is_in(
                    inputevents_cv_dialysis_active
                    + inputevents_cv_dialysis_mode_peritoneal
                    + inputevents_cv_dialysis_mode_cvvh
                    + inputevents_cv_dialysis_mode_cvvhd
                    + inputevents_cv_dialysis_mode_cvvhdf
                    + inputevents_cv_other
                ),
                pl.col("AMOUNT") > 0,
            )
            .with_columns(
                pl.lit(1).alias("dialysis_present"),
                pl.col("ITEMID")
                .is_in(inputevents_cv_dialysis_active)
                .not_()
                .cast(int)
                .alias("dialysis_active"),
                pl.when(
                    pl.col("ITEMID").is_in(
                        inputevents_cv_dialysis_mode_peritoneal
                    )
                )
                .then(pl.lit("Peritoneal dialysis"))
                .when(pl.col("ITEMID").is_in(inputevents_cv_dialysis_mode_cvvh))
                .then(pl.lit("CVVH"))
                .when(
                    pl.col("ITEMID").is_in(inputevents_cv_dialysis_mode_cvvhd)
                )
                .then(pl.lit("CVVHD"))
                .when(
                    pl.col("ITEMID").is_in(inputevents_cv_dialysis_mode_cvvhdf)
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

        # region OE
        OUTPUTEVENTS = (
            pl.scan_csv(
                self.mimic3_paths.outputevents_path,
                schema_overrides={"VALUE": float},
            )
            .select("ICUSTAY_ID", "CHARTTIME", "ITEMID", "VALUE")
            .filter(
                pl.col("ITEMID").is_in(
                    outputevents_dialysis_active
                    + outputevents_dialysis_type
                    + outputevents_other
                ),
                pl.col("VALUE") > 0,
            )
            .with_columns(
                pl.lit(1).alias("dialysis_present"),
                pl.col("ITEMID")
                .is_in(outputevents_dialysis_active)
                .not_()
                .cast(int)
                .alias("dialysis_active"),
                pl.lit("Peritoneal dialysis").alias("dialysis_type"),
            )
            .select(
                "ICUSTAY_ID",
                "CHARTTIME",
                "dialysis_present",
                "dialysis_active",
                "dialysis_type",
            )
        )

        # region IE_MV
        INPUTEVENTS_MV = (
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
                pl.lit(1).alias("dialysis_present"),
                pl.lit(1).alias("dialysis_active"),
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

        # region PE_MV
        PROCEDUREEVENTS_MV = (
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
                pl.lit(1).alias("dialysis_present"),
                pl.when(pl.col("ITEMID").is_in([224270, 225436]))
                .then(0)
                .otherwise(1)
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

        MV_RANGES = pl.concat(
            [INPUTEVENTS_MV, PROCEDUREEVENTS_MV],
            how="vertical",
        ).unique()

        RENAL_REPLACEMENT_THERAPY_DURATION = (
            pl.concat(
                [
                    CHARTEVENTS.filter(pl.col("dialysis_present") == 1),
                    INPUTEVENTS_CV.filter(pl.col("dialysis_present") == 1),
                    OUTPUTEVENTS.filter(pl.col("dialysis_present") == 1),
                    MV_RANGES.drop("ENDTIME").rename(
                        {"STARTTIME": "CHARTTIME"}
                    ),
                    MV_RANGES.drop("STARTTIME").rename(
                        {"ENDTIME": "CHARTTIME"}
                    ),
                ],
                how="vertical",
            )
            .unique()
            .join(
                MV_RANGES,
                on="ICUSTAY_ID",
                suffix="_mv",
                how="left",
            )
            .join(ADMISSIONTIMES, on="ICUSTAY_ID", how="left")
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
                .dt.total_seconds()
                .alias(
                    "Renal Replacement Therapy Start Relative to Admission (seconds)"
                ),
                (pl.col("ENDTIME") - pl.col("INTIME"))
                .dt.total_seconds()
                .alias(
                    "Renal Replacement Therapy End Relative to Admission (seconds)"
                ),
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
