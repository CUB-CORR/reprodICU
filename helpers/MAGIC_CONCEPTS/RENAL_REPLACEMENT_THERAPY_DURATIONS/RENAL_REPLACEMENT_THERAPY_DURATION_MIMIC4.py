# based on https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/treatment/rrt.sql

import polars as pl
from helpers.MAGIC_CONCEPTS.MAGIC_CONCEPTS import MAGIC_CONCEPTS


class RENAL_REPLACEMENT_THERAPY_DURATION_MIMIC4(MAGIC_CONCEPTS):
    def __init__(self, paths, datasets):
        super().__init__(paths, datasets)

    def RENAL_REPLACEMENT_THERAPY_DURATION(self) -> pl.DataFrame:
        print("MAGIC_CONCEPTS: Renal Replacement Therapy Duration - MIMIC4")

        # get admission times for MIMIC-IV
        ADMISSIONTIMES = (
            pl.scan_csv(self.mimic4_paths.icustays_path)
            .select("stay_id", "intime")
            .with_columns(
                pl.col("intime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
        )

        # region ITEMIDS
        # metavision itemids for both MIMIC-III and MIMIC-IV
        chartevents_dialysis_present = [
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
        chartevents_dialysis_active = [
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
        chartevents_dialysis_mode = [227290]
        chartevents_dialysis_mode_peritoneal = [
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
        chartevents_dialysis_mode_ihd = [226499]
        inputevents = [
            227536,  # KCl (CRRT) Medications	inputevents_mv	Solution
            227525,  # Calcium Gluconate (CRRT)	Medications	inputevents_mv	Solutio
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

        # region MIMIC-IV

        # print("MAGIC_CONCEPTS: Renal Replacement Therapy Duration - MIMIC4")

        RENAL_REPLACEMENT_THERAPY_CHARTEVENTS = (
            pl.scan_csv(
                self.mimic4_paths.chartevents_path,
                schema_overrides={"value": str},
            )
            .select("stay_id", "charttime", "itemid", "value")
            # Filter for renal replacement therapy IDs
            .filter(
                pl.col("itemid").is_in(
                    chartevents_dialysis_present
                    + chartevents_dialysis_active
                    + chartevents_dialysis_mode
                    + chartevents_dialysis_mode_peritoneal
                    + chartevents_dialysis_mode_ihd
                )
            )
            # replace renal replacement therapy concepts
            .with_columns(
                (
                    pl.col("itemid").is_in(chartevents_dialysis_present)
                    & pl.col("value").is_not_null()
                )
                # .fill_null(False)
                .alias("dialysis_present"),
                (
                    pl.col("itemid").is_in(chartevents_dialysis_active)
                    & pl.col("value").is_not_null()
                )
                # .fill_null(False)
                .alias("dialysis_active"),
                pl.when(pl.col("itemid").is_in(chartevents_dialysis_mode))
                .then(pl.col("value"))
                .when(
                    pl.col("itemid").is_in(chartevents_dialysis_mode_peritoneal)
                )
                .then(pl.lit("Peritoneal dialysis"))
                .when(pl.col("itemid").is_in(chartevents_dialysis_mode_ihd))
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

        RENAL_REPLACEMENT_THERAPY_INPUTEVENTS = (
            pl.scan_csv(self.mimic4_paths.inputevents_path)
            .select("stay_id", "starttime", "endtime", "itemid", "amount")
            .filter(
                pl.col("itemid").is_in(inputevents),
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

        RENAL_REPLACEMENT_THERAPY_PROCEDUREEVENTS = (
            pl.scan_csv(self.mimic4_paths.procedureevents_path)
            .select("stay_id", "starttime", "endtime", "itemid", "value")
            .filter(
                pl.col("itemid").is_in(procedureevents),
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

        RENAL_REPLACEMENT_THERAPY_RANGES = pl.concat(
            [
                RENAL_REPLACEMENT_THERAPY_INPUTEVENTS,
                RENAL_REPLACEMENT_THERAPY_PROCEDUREEVENTS,
            ],
            how="vertical",
        ).unique()

        RENAL_REPLACEMENT_THERAPY_DURATION = (
            pl.concat(
                [
                    RENAL_REPLACEMENT_THERAPY_CHARTEVENTS.filter(
                        pl.col("dialysis_present") == 1
                    ),
                    RENAL_REPLACEMENT_THERAPY_RANGES.drop("endtime").rename(
                        {"starttime": "charttime"}
                    ),
                ],
                how="vertical",
            )
            .unique()
            .join(
                RENAL_REPLACEMENT_THERAPY_RANGES,
                on="stay_id",
                suffix="_mv",
                how="left",
            )
            .join(ADMISSIONTIMES, on="stay_id", how="left")
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
                .dt.total_seconds()
                .alias(
                    "Renal Replacement Therapy Start Relative to Admission (seconds)"
                ),
                (pl.col("endtime") - pl.col("intime"))
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
            .drop("intime", "starttime", "endtime")
            .collect(streaming=True)
        )

        return (
            RENAL_REPLACEMENT_THERAPY_DURATION.unique()
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
