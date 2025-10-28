import polars as pl
from helpers.MAGIC_CONCEPTS.MAGIC_CONCEPTS import MAGIC_CONCEPTS


class VENTILATION_DURATION_eICUv2(MAGIC_CONCEPTS):
    def __init__(self, paths, datasets):
        super().__init__(paths, datasets)

    def VENTILATION_DURATION(self) -> pl.LazyFrame:
        # Inputs
        careplan = pl.scan_csv(self.eicu_paths.carePlanGeneral_path).select(
            "patientunitstayid",
            pl.col("activeupondischarge")
            .cast(pl.Boolean)
            .alias("activeupondischarge"),
            "cplitemoffset",
            "cplgroup",
            "cplitemvalue",
        )
        icudetail = pl.scan_csv(self.eicu_paths.patient_path).select(
            "patientunitstayid", "hospitaldischargeoffset"
        )

        # ----------------------------------------------------------------------
        # region ventaliation_info_time_0_eicu
        # ----------------------------------------------------------------------

        # ventaliation_info_initial (bounded by hospitaldischargeoffset)
        valid_vals = [
            "Intubated/nasal ETT",
            "Intubated/oral ETT",
            "Intubated/trach-acute",
            "Intubated/trach-chronic",
            "Not intubated/normal airway",
            "Not intubated/partial airway obstruction",
            "Non-invasive ventilation",
            "Spontaneous - adequate",
            "Spontaneous - tenuous",
            "Ventilated - chronic dependency",
            "Ventilated - with daily extubation evaluation",
            "Ventilated - with no daily extubation trial",
        ]
        vi_init = (
            careplan.filter(
                pl.col("cplgroup").is_in(["Airway", "Ventilation"])
                & pl.col("cplitemvalue").is_in(valid_vals)
            )
            .join(icudetail, on="patientunitstayid", how="inner")
            .filter(
                pl.col("cplitemoffset") <= pl.col("hospitaldischargeoffset")
            )
            .with_columns(
                pl.when(
                    pl.col("cplitemvalue").is_in(
                        [
                            "Intubated/nasal ETT",
                            "Intubated/oral ETT",
                            "Intubated/trach-acute",
                            "Intubated/trach-chronic",
                            "Ventilated - chronic dependency",
                            "Ventilated - with daily extubation evaluation",
                            "Ventilated - with no daily extubation trial",
                        ]
                    )
                )
                .then(pl.lit("invasive"))
                .when(
                    pl.col("cplitemvalue").is_in(
                        [
                            "Not intubated/normal airway",
                            "Not intubated/partial airway obstruction",
                            "Non-invasive ventilation",
                            # "Spontaneous - tenuous",
                        ]
                    )
                )
                .then(pl.lit("non-invasive"))
                .otherwise(None)
                .alias("vent_type"),
                pl.when(pl.col("cplitemvalue") == "Spontaneous - adequate")
                .then(0)
                .otherwise(1)
                .alias("vent_start_flag"),
            )
        )

        # ventaliation_info_time_00
        v00 = (
            vi_init.group_by(
                [
                    "patientunitstayid",
                    "cplitemoffset",
                    "hospitaldischargeoffset",
                ]
            )
            .agg(
                pl.when(pl.col("vent_type") == "invasive")
                .then(1)
                .when(pl.col("vent_type") == "non-invasive")
                .then(0)
                .otherwise(None)
                .max()
                .alias("vent_type_num"),
                pl.col("activeupondischarge").cast(int).fill_null(0).max(),
                pl.col("vent_start_flag").max(),
            )
            .with_columns(
                pl.when(pl.col("vent_type_num") == 1)
                .then(pl.lit("invasive"))
                .when(pl.col("vent_type_num") == 0)
                .then(pl.lit("non-invasive"))
                .otherwise(None)
                .alias("vent_type")
            )
            .select(
                "patientunitstayid",
                "hospitaldischargeoffset",
                "cplitemoffset",
                "vent_type",
                "vent_start_flag",
                "activeupondischarge",
            )
        )

        # ventaliation_info_time_01
        # -- drop: the first row with 'Spontaneous - adequate' and activeupondischarge = 'False'
        # -- later existing ventilation records
        v01 = (
            v00.with_columns(
                pl.col("patientunitstayid")
                .rank("ordinal")
                .over("patientunitstayid", order_by="cplitemoffset")
                .alias("rn")
            )
            .with_columns(
                pl.when(
                    pl.col("rn") == 1,
                    pl.col("vent_start_flag") == 0,
                    pl.col("activeupondischarge") == 0,
                )
                .then(1)
                .otherwise(0)
                .alias("drop_flag")
            )
            .filter(pl.col("drop_flag") == 0)
            .select(
                "patientunitstayid",
                "hospitaldischargeoffset",
                "cplitemoffset",
                "vent_type",
                "vent_start_flag",
                "activeupondischarge",
            )
        )

        # drop patients
        # -- drop patients: the first row with 'Spontaneous - adequate' and activeupondischarge = 'True' and no existing 'invasive'
        # -- we thought they didn't receive ventilation
        drop_patients = (
            # drop_info_part_0
            v01.with_columns(
                pl.col("patientunitstayid")
                .rank("ordinal")
                .over("patientunitstayid", order_by="cplitemoffset")
                .alias("rn")
            )
            .join(
                v01.group_by("patientunitstayid").agg(
                    (pl.col("vent_type") == "invasive")
                    .max()
                    .alias("has_invasive")
                ),
                on="patientunitstayid",
                how="inner",
            )
            .filter(pl.col("has_invasive") == 0)
            # drop_info_part
            .filter(pl.col("rn") == 1, pl.col("activeupondischarge") == 1)
            .select("patientunitstayid")
            .unique()
            .collect()
            .to_series()
        )

        # ventaliation_info_time_0
        v0 = v01.filter(~pl.col("patientunitstayid").is_in(drop_patients))

        # ----------------------------------------------------------------------
        # region ventaliation_info_use_4_eicu
        # ----------------------------------------------------------------------

        # change_info_part
        # -- identify the 'non-invasive' dischargestatus true, while existing 'invasive' type
        # -- which should be changed to end before the 'invasive' start
        change_info_part = (
            v0.filter(
                pl.col("vent_type") == "non-invasive",
                pl.col("activeupondischarge") == 1,
            )
            .join_where(
                v0.filter(pl.col("vent_type") == "invasive"),
                pl.col("patientunitstayid") == pl.col("patientunitstayid_inv"),
                pl.col("cplitemoffset") < pl.col("cplitemoffset_inv"),
                suffix="_inv",
            )
            .select("patientunitstayid", "cplitemoffset")
            .unique()
        )

        # ventaliation_info_time_1
        v1 = (
            v0.join(
                change_info_part.with_columns(pl.lit(0).alias("change")),
                on=["patientunitstayid", "cplitemoffset"],
                how="left",
            )
            .with_columns(
                pl.coalesce(
                    pl.col("change"), pl.col("activeupondischarge")
                ).alias("activeupondischarge")
            )
            .select(
                "patientunitstayid",
                "cplitemoffset",
                "hospitaldischargeoffset",
                "activeupondischarge",
                "vent_type",
                "vent_start_flag",
            )
        )

        # ventaliation_info_time
        vi_time = pl.concat(
            [
                v1.select(
                    "patientunitstayid",
                    "cplitemoffset",
                    "vent_type",
                    "vent_start_flag",
                ),
                v1.filter(pl.col("activeupondischarge") == 1).select(
                    "patientunitstayid",
                    pl.col("hospitaldischargeoffset").alias("cplitemoffset"),
                    "vent_type",
                    pl.lit(0).alias("vent_start_flag"),
                ),
                # -- the last record was vent with false status, since we didn't know the end time, we set adding 60min as endtime
                v1.with_columns(
                    pl.col("patientunitstayid")
                    .rank("ordinal")
                    .over(
                        "patientunitstayid",
                        order_by="cplitemoffset",
                        descending=True,
                    )
                    .alias("rn")
                )
                .filter(
                    pl.col("rn") == 1,
                    pl.col("activeupondischarge") == 0,
                    pl.col("vent_start_flag") == 1,
                )
                .select(
                    "patientunitstayid",
                    (pl.col("cplitemoffset") + 60).alias("cplitemoffset"),
                    "vent_type",
                    pl.lit(0).alias("vent_start_flag"),
                ),
            ],
            how="vertical_relaxed",
        ).sort("patientunitstayid", "cplitemoffset")

        # ventaliation_info_use_*: compress toggles into groups and pick boundary offsets
        # ventaliation_info_use
        use = (
            # ventaliation_info_use_0
            vi_time.with_columns(
                pl.col("vent_start_flag")
                .shift(1)
                .over(
                    partition_by="patientunitstayid",
                    order_by="cplitemoffset",
                )
                .alias("lag_flag")
            )
            # ventaliation_info_use_1
            .with_columns(
                pl.when(pl.col("lag_flag") == pl.col("vent_start_flag"))
                .then(None)
                .otherwise(1)
                .alias("r")
            )
            .with_columns(
                pl.col("r")
                .sort_by("patientunitstayid", "cplitemoffset")
                .cum_sum()
                # .over(order_by=["patientunitstayid", "cplitemoffset"])
                .alias("grp")
            )
            # ventaliation_info_use_2
            .group_by("grp")
            .agg(
                pl.col("patientunitstayid").min().alias("patientunitstayid"),
                pl.col("vent_start_flag").min().alias("vent_start_flag"),
                pl.col("cplitemoffset").min().alias("sort_first"),
                pl.col("cplitemoffset").max().alias("sort_last"),
            )
            # ventaliation_info_use_3
            # -- get the start and end time of each patient
            .select(
                "patientunitstayid",
                "vent_start_flag",
                pl.when(pl.col("vent_start_flag") == 1)
                .then(pl.col("sort_first"))
                .otherwise(pl.col("sort_last"))
                .alias("cplitemoffset"),
            )
        )

        # ventaliation_info_use_4
        # -- here we check the abnormal types:
        # -- non-invasive cplitemoffset = hospitaldischargeoffset
        # -- only existing 'Spontaneous - adequate' records
        # Remove patients with only 'adequate' (sum flags = 0)
        only_adequate_patients = (
            use.group_by("patientunitstayid")
            .agg(pl.col("vent_start_flag").sum().alias("num"))
            .filter(pl.col("num") == 0)
            .select("patientunitstayid")
            .collect()
            .to_series()
        )
        use4 = use.filter(
            ~pl.col("patientunitstayid").is_in(only_adequate_patients)
        ).sort("patientunitstayid", "cplitemoffset")
        
        # ----------------------------------------------------------------------
        # region pivoted_vent_eicu
        # ----------------------------------------------------------------------

        # pivoted_vent_eICU with manual overrides
        specials = [1571346, 1571446, 1589211]
        base = use4.filter(
            ~pl.col("patientunitstayid").is_in([1565479] + specials)
        )
        manual_start = pl.LazyFrame(
            {
                "patientunitstayid": specials,
                "vent_start_flag": [1, 1, 1],
                "cplitemoffset": [2091, 1430, 2372],
            }
        )
        manual_end = pl.LazyFrame(
            {
                "patientunitstayid": specials,
                "vent_start_flag": [0, 0, 0],
                "cplitemoffset": [5170, 1430 + 60, 4903],
            }
        )

        pivot0 = pl.concat(
            [
                base.select(
                    "patientunitstayid", "vent_start_flag", "cplitemoffset"
                ),
                manual_start.select(
                    "patientunitstayid", "vent_start_flag", "cplitemoffset"
                ),
                manual_end.select(
                    "patientunitstayid", "vent_start_flag", "cplitemoffset"
                ),
            ],
            how="vertical_relaxed",
        )

        pivot1 = pivot0.sort(
            "patientunitstayid", "cplitemoffset", "vent_start_flag"
        )

        starts = (
            pivot0.filter(pl.col("vent_start_flag") == 1)
            .with_columns(
                pl.col("patientunitstayid")
                .rank("ordinal")
                .over(
                    "patientunitstayid",
                    order_by="cplitemoffset",
                    descending=True,
                )
                .alias("rn")
            )
            .select(
                "patientunitstayid",
                pl.col("cplitemoffset").alias("starttime"),
                "rn",
            )
        )
        ends = (
            pivot1.filter(pl.col("vent_start_flag") == 0)
            .with_columns(
                pl.col("patientunitstayid")
                .rank("ordinal")
                .over(
                    "patientunitstayid",
                    order_by="cplitemoffset",
                    descending=True,
                )
                .alias("rn")
            )
            .select(
                "patientunitstayid",
                pl.col("cplitemoffset").alias("endtime"),
                "rn",
            )
        )

        result = (
            starts.join(ends, on=["patientunitstayid", "rn"], how="inner")
            .select("patientunitstayid", "starttime", "endtime")
            .sort("patientunitstayid", "starttime")
        )
        # CHANGED: materialize to avoid streaming sink issues with anti-joins
        return result.collect().lazy()
