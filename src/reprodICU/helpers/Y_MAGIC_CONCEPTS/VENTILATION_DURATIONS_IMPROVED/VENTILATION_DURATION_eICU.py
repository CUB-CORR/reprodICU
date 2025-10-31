import polars as pl

from ..MAGIC_CONCEPTS import MAGIC_CONCEPTS


class VENTILATION_DURATION_eICUv1(MAGIC_CONCEPTS):
    def __init__(self, paths, datasets):
        super().__init__(paths, datasets)

    def VENTILATION_DURATION(self) -> pl.LazyFrame:
        # Tables
        careplan = pl.scan_csv(self.eicu_paths.carePlanGeneral_path).select(
            "cplgeneralid",
            "patientunitstayid",
            pl.col("activeupondischarge")
            .cast(pl.Boolean)
            .alias("activeupondischarge"),
            "cplitemoffset",
            "cplgroup",
            "cplitemvalue",
        )
        patient = pl.scan_csv(self.eicu_paths.patient_path).select(
            "patientunitstayid",
            "unitdischargeoffset",
            "hospitaldischargeoffset",
        )

        concat_cols = [
            "cplgeneralid",
            "patientunitstayid",
            "activeupondischarge",
            "cplitemoffset",
            "cplgroup",
            "cplitemvalue",
            "vent_flag",
            "vent_type",
        ]

        # ----------------------------------------------------------------------
        # region part1
        # pivoted_vent_part1_eicu
        # ----------------------------------------------------------------------

        # -- Intubated/nasal ETT                | 335
        # -- Intubated/nasal ETT - difficult    | 52
        # -- Intubated/oral ETT                 | 59566
        # -- Intubated/oral ETT - difficult     | 798
        # -- Intubated/trach-acute              | 4829
        # -- Intubated/trach-chronic            | 4993
        # -- Ventilated - chronic dependency                | 3105
        # -- Ventilated - with daily extubation evaluation  | 51862
        # -- Ventilated - with no daily extubation trial    | 14907
        # -- Non-invasive ventilation                       | 26836

        # -- Not intubated/normal airway                | 206795
        # -- Not intubated/partial airway obstruction   | 1543
        # -- Ventilated - rapid wean/extubation         | 5705
        # -- Spontaneous - adequate                     | 190809
        # -- Spontaneous - tenuous                      | 32587
        # --                                            | 14896

        ventilation_info = (
            careplan.filter(
                pl.col("cplgroup").is_in(["Airway", "Ventilation"]),
                pl.col("cplitemvalue").is_not_null(),
                pl.col("cplitemvalue") != "",
            ).with_columns(
                pl.when(
                    pl.col("cplitemvalue").str.starts_with("Intubated")
                    | pl.col("cplitemvalue").is_in(
                        [
                            "Ventilated - chronic dependency",
                            "Ventilated - with daily extubation evaluation",
                            "Ventilated - with no daily extubation trial",
                            "Non-invasive ventilation",
                        ]
                    )
                )
                .then(1)
                .otherwise(0)
                .alias("vent_flag"),
            )
            # START NOTE: not in SQL
            .with_columns(
                pl.when(
                    (pl.col("vent_flag") == 1)
                    & (pl.col("cplitemvalue") == "Non-invasive ventilation")
                )
                .then(pl.lit("non-invasive ventilation"))
                .when(pl.col("vent_flag") == 1)
                .then(pl.lit("invasive ventilation"))
                .otherwise(pl.lit("none"))
                .alias("vent_type")
            )
            # END NOTE: not in SQL
        )

        ventilation_00 = ventilation_info.group_by("patientunitstayid").agg(
            pl.col("vent_flag").sum().alias("num")
        )

        # -- drop patientunitstayid didn't have ventaliation
        ventilation_01 = (
            ventilation_info.join(
                ventilation_00.filter(pl.col("num") > 0).select(
                    "patientunitstayid"
                ),
                on="patientunitstayid",
                how="inner",
            )
            .group_by("patientunitstayid", "cplitemoffset")
            .agg(pl.col("vent_flag").sum().alias("num"))
        )

        ventilation_02 = (
            ventilation_info.join(
                ventilation_01.filter(pl.col("num") >= 1),
                on=["patientunitstayid", "cplitemoffset"],
                how="inner",
            )
            .filter(pl.col("vent_flag") == 0)
            .with_columns(
                pl.col("patientunitstayid")
                .rank("ordinal")
                .over(
                    partition_by=["patientunitstayid", "cplitemoffset"],
                    order_by="vent_flag",
                    descending=True,
                )
                .alias("flag")
            )
            .select("cplgeneralid", "flag")  # only those are needed later
        )

        # -- drop the same cplitemoffset rows of non-ventiliation, existing ventiliation and non-ventiliation
        ventilation_0 = ventilation_info.filter(
            ~pl.col("cplgeneralid").is_in(
                ventilation_02.filter(pl.col("flag") == 1)
                .select("cplgeneralid")
                .collect()
                .to_series()
            )
        )

        # -- solving the same cplitemoffset rows of more than two different ventiliation
        # -- remain one rows
        ventilation_10 = (
            ventilation_0.filter(pl.col("vent_flag") == 1)
            .with_columns(
                pl.col("patientunitstayid")
                .rank("ordinal")
                .over(
                    partition_by=["patientunitstayid", "cplitemoffset"],
                    order_by="cplitemvalue",
                )
                .alias("rn")
            )
            .select("cplgeneralid", "rn")
        )
        PIVOTED_VENT_PART1 = ventilation_0.filter(
            ~pl.col("cplgeneralid").is_in(
                ventilation_10.filter(pl.col("rn") > 1)
                .select("cplgeneralid")
                .collect()
                .to_series()
            )
        ).sort("patientunitstayid", "cplitemoffset")

        # ----------------------------------------------------------------------
        # region part2
        # pivoted_vent_part2_eicu
        # ----------------------------------------------------------------------

        ventilation_20 = PIVOTED_VENT_PART1.with_columns(
            pl.col("patientunitstayid")
            .rank("ordinal")
            .over(partition_by="patientunitstayid", order_by="cplitemoffset")
            .alias("rn")
        )

        ventilation_21 = ventilation_20.join(
            ventilation_20.filter(pl.col("rn") == 1, pl.col("vent_flag") == 0)
            .select("patientunitstayid")
            .unique(),
            on="patientunitstayid",
            how="inner",
        )

        ventilation_22 = ventilation_20.filter(
            ~pl.col("cplgeneralid").is_in(
                ventilation_21.select("cplgeneralid").collect().to_series()
            )
        )

        # -- ventilation_21: delete the first rows of non-ventiliation
        ventilation_21_ = (
            # ventilation_210
            ventilation_21.with_columns(
                pl.col("vent_flag")
                .shift(1)
                .over(
                    partition_by="patientunitstayid", order_by="cplitemoffset"
                )
                .alias("vent_flag_new")
            )
            # ventilation_211
            .with_columns(
                (pl.col("vent_flag_new") - pl.col("vent_flag")).alias(
                    "del_flag"
                )
            )
            .filter(pl.col("del_flag") == -1)
            .with_columns(
                pl.col("patientunitstayid")
                .rank("ordinal")
                .over(
                    partition_by="patientunitstayid",
                    order_by="cplitemoffset",
                )
                .alias("rn")
            )
            # ventilation_212
            .filter(pl.col("rn") == 1)
            .select(
                "patientunitstayid",
                pl.col("cplitemoffset").alias("cplitemoffset_cutoff"),
            )
        )

        ventilation_213 = (
            ventilation_21.join(
                ventilation_21_, on="patientunitstayid", how="inner"
            )
            .filter(pl.col("cplitemoffset") >= pl.col("cplitemoffset_cutoff"))
            .drop("cplitemoffset_cutoff")
        )

        PIVOTED_VENT_PART2 = pl.concat(
            [
                ventilation_213.select(concat_cols),
                ventilation_22.select(concat_cols),
            ],
            how="vertical",
        ).unique()

        # ----------------------------------------------------------------------
        # region part34
        # pivoted_vent_part34_eicu
        # ----------------------------------------------------------------------

        # -- delete the same cplitemoffset with different types of non-ventilation, remain one row
        ventilation_30 = PIVOTED_VENT_PART2.filter(
            pl.col("vent_flag") == 0
        ).with_columns(
            pl.col("patientunitstayid")
            .rank("ordinal")
            .over(
                partition_by=["patientunitstayid", "cplitemoffset"],
                order_by="cplitemvalue",
                descending=True,
            )
            .alias("rn")
        )

        ventilation_31 = PIVOTED_VENT_PART2.filter(pl.col("vent_flag") != 0)

        ventilation_3 = pl.concat(
            [
                ventilation_30.filter(pl.col("rn") == 1).select(concat_cols),
                ventilation_31.select(concat_cols),
            ],
            how="vertical",
        ).unique()

        print("ventilation_3", ventilation_3.collect_schema())

        # -- existing some patientunitstayid didn't know the endtime
        # -- Assume that it ends after 1h
        ventilation_40 = ventilation_3.unique().with_columns(
            pl.col("patientunitstayid")
            .rank("ordinal")
            .over(
                partition_by="patientunitstayid",
                order_by="cplitemoffset",
                descending=True,
            )
            .alias("rn")
        )

        ventilation_41 = ventilation_40.filter(
            pl.col("rn") == 1,
            pl.col("vent_flag") == 1,
            ~pl.col("activeupondischarge"),
        )

        ventilation_411 = pl.concat(
            [
                ventilation_41.select(concat_cols),
                ventilation_41.select(
                    "cplgeneralid",
                    "patientunitstayid",
                    "activeupondischarge",
                    (pl.col("cplitemoffset") + 60).alias("cplitemoffset"),
                    "cplgroup",
                    pl.lit("Spontaneous - adequate").alias("cplitemvalue"),
                    pl.lit(0).alias("vent_flag"),
                    pl.lit("none").alias("vent_type"),
                ).select(concat_cols),
            ],
            how="vertical",
        )

        ventilation_42 = ventilation_40.filter(
            ~pl.col("cplgeneralid").is_in(
                ventilation_41.select("cplgeneralid").collect().to_series()
            )
        )

        PIVOTED_VENT_PART34 = pl.concat(
            [
                ventilation_411.select(concat_cols),
                ventilation_42.select(concat_cols),
            ],
            how="vertical",
        ).unique()

        # ----------------------------------------------------------------------
        # region part56
        # pivoted_vent_part56_eicu
        # ----------------------------------------------------------------------

        # -- existing some patients : the last two rows were active ventilation and active non-ventilation
        # -- we will handle this situation : assume that patient finish ventilation before start non-ventilation
        ventilation_50 = PIVOTED_VENT_PART34.with_columns(
            pl.col("patientunitstayid")
            .rank("ordinal")
            .over(
                partition_by="patientunitstayid",
                order_by="cplitemoffset",
                descending=True,
            )
            .alias("rn")
        )

        ventilation_500 = (
            ventilation_50.with_columns(
                (pl.col("rn") == 1)
                .and_(pl.col("vent_flag") == 0)
                .cast(pl.Int8)
                .alias("flag")
            )
            # only those are needed later
            .filter(pl.col("flag") == 1).select("patientunitstayid")
        )
        ventilation_501 = (
            ventilation_50.with_columns(
                (
                    (pl.col("rn") == 2)
                    & (pl.col("vent_flag") == 1)
                    & ~pl.col("activeupondischarge")
                )
                .cast(int)
                .alias("flag")
            )
            # only those are needed later
            .filter(pl.col("flag") == 1).select("patientunitstayid")
        )
        ventilation_502 = ventilation_500.join(
            ventilation_501,
            on="patientunitstayid",
            how="inner",
        ).unique()

        ventilation_510 = ventilation_50.join(
            ventilation_502, on="patientunitstayid", how="inner"
        )
        ventilation_51 = pl.concat(
            [
                ventilation_510.filter(pl.col("rn") > 1).select(concat_cols),
                ventilation_510.filter(pl.col("rn") == 1)
                .select(
                    "cplgeneralid",
                    "patientunitstayid",
                    pl.lit(False).alias("activeupondischarge"),
                    "cplitemoffset",
                    "cplgroup",
                    pl.lit("Spontaneous - adequate").alias("cplitemvalue"),
                    pl.lit(0).alias("vent_flag"),
                    pl.lit("none").alias("vent_type"),
                )
                .select(concat_cols),
            ],
            how="vertical",
        )

        ventilation_52 = ventilation_50.filter(
            ~pl.col("patientunitstayid").is_in(
                ventilation_502.select("patientunitstayid")
                .collect()
                .to_series()
            )
        )

        ventilation_5 = pl.concat(
            [
                ventilation_51.select(concat_cols),
                ventilation_52.select(concat_cols),
            ],
            how="vertical",
        ).unique()

        # -- handling with tha last row is activeupondischarge = True and vent_flag = 1
        ventilation_60 = ventilation_5.with_columns(
            pl.col("patientunitstayid")
            .rank("ordinal")
            .over(
                partition_by="patientunitstayid",
                order_by="cplitemoffset",
                descending=True,
            )
            .alias("rn")
        )
        ventilation_610 = ventilation_60.filter(
            pl.col("rn") == 1,
            pl.col("vent_flag") == 1,
            pl.col("activeupondischarge"),
        ).unique()

        ventilation_61 = pl.concat(
            [
                ventilation_610.select(concat_cols),
                ventilation_610.join(
                    patient, on="patientunitstayid", how="left"
                )
                .select(
                    "cplgeneralid",
                    "patientunitstayid",
                    pl.lit(False).alias("activeupondischarge"),
                    pl.col("unitdischargeoffset").alias("cplitemoffset"),
                    pl.lit("Airway").alias("cplgroup"),
                    pl.lit("Spontaneous - adequate").alias("cplitemvalue"),
                    pl.lit(0).alias("vent_flag"),
                    pl.lit("none").alias("vent_type"),
                )
                .select(concat_cols),
            ],
            how="vertical",
        )

        ventilation_62 = ventilation_60.filter(
            ~pl.col("cplgeneralid").is_in(
                ventilation_610.select("cplgeneralid").collect().to_series()
            )
        )

        PIVOTED_VENT_PART56 = pl.concat(
            [
                ventilation_61.select(concat_cols),
                ventilation_62.select(concat_cols),
            ],
            how="vertical",
        ).unique()

        # ----------------------------------------------------------------------
        # region part7
        # pivoted_vent_eicu
        # ----------------------------------------------------------------------

        # # -- get start and end time of ventilation
        # ventilation_70 = PIVOTED_VENT_PART56.with_columns(
        #     pl.col("cplitemoffset").alias("starttime"),
        #     pl.col("cplitemoffset")
        #     .shift(-1)
        #     .over(partition_by="patientunitstayid", order_by="cplitemoffset")
        #     .alias("endtime"),
        #     pl.col("vent_flag")
        #     .shift(-1)
        #     .over(partition_by="patientunitstayid", order_by="cplitemoffset")
        #     .alias("vent_flag_new"),
        # )

        # ventilation_701 = (
        #     ventilation_70.filter(pl.col("vent_flag") == 1)
        #     .with_columns(
        #         (pl.col("vent_flag") - pl.col("vent_flag_new")).alias("flag")
        #     )
        #     .filter(pl.col("flag") != -1)
        # )

        # ventilation_71 = ventilation_701.filter(pl.col("flag") == 1).select(
        #     "patientunitstayid",
        #     "starttime",
        #     "endtime",
        # )

        # ventilation_72 = (
        #     # ventilation_720
        #     pl.concat(
        #         [
        #             ventilation_701.filter(pl.col("flag") == 0).select(
        #                 "patientunitstayid",
        #                 pl.col("starttime").alias("cplitemoffset"),
        #             ),
        #             ventilation_701.filter(pl.col("flag") == 0).select(
        #                 "patientunitstayid",
        #                 pl.col("endtime").alias("cplitemoffset"),
        #             ),
        #         ],
        #         how="vertical",
        #     )
        #     .unique()
        #     .group_by("patientunitstayid", "cplitemoffset")
        #     .agg(pl.count().alias("num"))
        #     # ventilation_721
        #     .filter(pl.col("num") == 1)
        #     .select("patientunitstayid", "cplitemoffset")
        # )

        # result = (
        #     # ventilation_730
        #     pl.concat(
        #         [
        #             ventilation_71.select(
        #                 "patientunitstayid",
        #                 pl.col("starttime").alias("cplitemoffset"),
        #             ),
        #             ventilation_71.select(
        #                 "patientunitstayid",
        #                 pl.col("endtime").alias("cplitemoffset"),
        #             ),
        #             ventilation_72,
        #         ],
        #         how="vertical",
        #     )
        #     .unique()
        #     # ventilation_731
        #     .group_by("patientunitstayid", "cplitemoffset")
        #     .agg(pl.len().alias("num"))
        #     # ventilation_732
        #     .filter(pl.col("num") == 1)
        #     .select("patientunitstayid", "cplitemoffset")
        #     # ventilation_733
        #     .sort("patientunitstayid", "cplitemoffset")
        #     .with_columns(
        #         pl.col("cplitemoffset")
        #         .shift(-1)
        #         .over(
        #             partition_by="patientunitstayid",
        #             order_by="cplitemoffset",
        #         )
        #         .alias("endtime")
        #     )
        #     .rename({"cplitemoffset": "starttime"})
        #     # ventilation_734
        #     .filter(pl.col("endtime").is_not_null())
        #     .with_columns(
        #         pl.col("patientunitstayid")
        #         .rank("ordinal")
        #         .over(partition_by="patientunitstayid", order_by="starttime")
        #         .alias("rn")
        #     )
        #     # result
        #     .filter((pl.col("rn") % 2) == 1)
        #     .select("patientunitstayid", "starttime", "endtime")
        #     .sort("patientunitstayid", "starttime")
        # )

        # Build state timeline and segment on state changes to get per-vent_type intervals
        result = (
            PIVOTED_VENT_PART56.sort("patientunitstayid", "cplitemoffset")
            .with_columns(
                pl.when(pl.col("vent_flag") == 1)
                .then(pl.col("vent_type"))
                .otherwise(pl.lit("off"))
                .alias("state"),
            )
            .with_columns(
                (
                    pl.col("state")
                    != pl.col("state")
                    .shift(1)
                    .over(
                        partition_by="patientunitstayid",
                        order_by="cplitemoffset",
                    )
                )
                .fill_null(True)
                .alias("is_state_start")
            )
            .filter(pl.col("is_state_start"))
            .with_columns(
                pl.col("cplitemoffset")
                .shift(-1)
                .over("patientunitstayid")
                .alias("endtime")
            )
            .filter(
                pl.col("endtime").is_not_null(),
                pl.col("state").is_in(
                    ["invasive ventilation", "non-invasive ventilation"]
                ),
            )
            .select(
                "patientunitstayid",
                pl.col("cplitemoffset").alias("starttime"),
                "endtime",
                pl.col("state").alias("vent_type"),
            )
            .sort("patientunitstayid", "starttime")
        )

        return (
            result.with_columns(
                # reltimes in eICU are in minutes
                (pl.col("starttime") * 60).alias(
                    "Ventilation Start Relative to Admission (seconds)"
                ),
                (pl.col("endtime") * 60).alias(
                    "Ventilation End Relative to Admission (seconds)"
                ),
                pl.col("vent_type").alias("Ventilation Type"),
            )
            .select(
                "patientunitstayid",
                "Ventilation Type",
                "Ventilation Start Relative to Admission (seconds)",
                "Ventilation End Relative to Admission (seconds)",
            )
            .unique()
            .pipe(self._add_global_id_stay_id, "eicu-", "patientunitstayid")
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
