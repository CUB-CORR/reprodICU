# Author: Finn Fassbender
# Last modified: 2024-09-05

# Enables the easy conversion of the data.
# Conversion constants were taken from: https://www.labcorp.com/resource/si-unit-conversion-table

import polars as pl


def _struct_with_all_null_to_null(
    frame: pl.DataFrame, struct_col: str
) -> pl.DataFrame:
    """
    Set any structs to null that have all null fields.

    WARNING
    -------
    The function only checks for null in the current struct fields. It doesn't
    do recursive checks on structs inside the struct that could also have all
    null fields.

    Parameters
    ----------
    frame: pl.DataFrame
        The frame to modify.
    struct_col: str
        The name of the struct column to modify.

    Returns
    -------
    pl.DataFrame
        Modified DataFrame.
    """

    # If any struct field is non-null, then keep the struct, otherwise replace it by null.
    return frame.with_columns(
        pl.when(
            pl.any_horizontal(
                pl.col(struct_col).struct.field("*").is_not_null()
            )
        )
        .then(pl.col(struct_col))
        .otherwise(None)
        .alias(struct_col)
    )


# Enables the easy combination of Glasgow Coma Scale (GCS) components.
# ASSUMPTION: data is in wide format, after pivoting.
class GCSCombiner:
    def __init__(self):
        pass

    def combine_gcs_components(
        self,
        data: pl.LazyFrame,
        eye_subscore: str = "glasgow_coma_score_eye",
        motor_subscore: str = "glasgow_coma_score_motor",
        verbal_subscore: str = "glasgow_coma_score_verbal",
        total_score: str = "glasgow_coma_score",
    ) -> pl.LazyFrame:
        """
        Combine the GCS components to the GCS total score.
        """
        return data.with_columns(
            pl.when(pl.col(total_score) == None)
            .then(
                pl.col(eye_subscore)
                + pl.col(motor_subscore)
                + pl.col(verbal_subscore)
            )
            .otherwise(pl.col(total_score))
            .alias(total_score)
        )


# Enables the easy conversion of the data.
# ASSUMPTION: data is in long format, before pivoting.
class UnitConversions:
    def __init__(self):
        pass

    # CAVE: THIS ASSUMES WIDE FORMAT
    def convert_absolute_count_to_relative(
        self,
        data: pl.LazyFrame,
        itemcol: str,
        total_itemcol: str,
        goal_itemcol: str = None,
        structfield: str = None,
        structstring: bool = False,
    ) -> pl.LazyFrame:
        """
        Convert absolute counts to relative counts.
        """

        if goal_itemcol is None:
            goal_itemcol = itemcol

        labstructdtype = pl.Struct(
            [
                pl.Field("value", pl.Float64),
                pl.Field("system", pl.String),
                pl.Field("method", pl.String),
                pl.Field("time", pl.String),
                pl.Field("LOINC", pl.String),
            ]
        )

        if structfield is not None:
            if structstring:
                data = data.with_columns(
                    pl.col(itemcol)
                    .str.json_decode(labstructdtype)
                    .alias(goal_itemcol),
                    pl.col(total_itemcol).str.json_decode(labstructdtype),
                )

            data = (
                data.with_columns(
                    # Rename the columns for the unnest
                    pl.col(goal_itemcol).struct.rename_fields(
                        [
                            "goal_itemcol_value",
                            "goal_itemcol_system",
                            "goal_itemcol_method",
                            "goal_itemcol_time",
                            "goal_itemcol_LOINC",
                        ]
                    ),
                    pl.col(total_itemcol).struct.rename_fields(
                        [
                            "total_itemcol_value",
                            "total_itemcol_system",
                            "total_itemcol_method",
                            "total_itemcol_time",
                            "total_itemcol_LOINC",
                        ]
                    ),
                )
                .unnest(goal_itemcol)
                .unnest(total_itemcol)
                .with_columns(
                    pl.when(
                        pl.col("goal_itemcol_value").is_not_null()
                        & pl.col("total_itemcol_value").is_not_null()
                    )
                    .then(
                        pl.col("goal_itemcol_value").truediv(
                            pl.col("total_itemcol_value")
                        )
                    )
                    .otherwise(None)
                    .alias("goal_itemcol_value")
                )
                # Combine the columns back into a struct again
                .select(
                    pl.exclude(
                        "goal_itemcol_value",
                        "goal_itemcol_system",
                        "goal_itemcol_method",
                        "goal_itemcol_time",
                        "goal_itemcol_LOINC",
                        "total_itemcol_value",
                        "total_itemcol_system",
                        "total_itemcol_method",
                        "total_itemcol_time",
                        "total_itemcol_LOINC",
                    ),
                    pl.struct(
                        value="goal_itemcol_value",
                        system="goal_itemcol_system",
                        method="goal_itemcol_method",
                        time="goal_itemcol_time",
                        LOINC=pl.lit(None),
                    ).alias(goal_itemcol),
                    pl.struct(
                        value="total_itemcol_value",
                        source="total_itemcol_system",
                        method="total_itemcol_method",
                        time="total_itemcol_time",
                        LOINC="total_itemcol_LOINC",
                    ).alias(total_itemcol),
                )
                .pipe(_struct_with_all_null_to_null, struct_col=goal_itemcol)
                .pipe(_struct_with_all_null_to_null, struct_col=total_itemcol)
            )

            if structstring:
                data = data.with_columns(
                    pl.col(goal_itemcol)
                    .struct.json_encode()
                    .replace("null", None),
                    pl.col(total_itemcol)
                    .struct.json_encode()
                    .replace("null", None),
                )

        else:
            data = data.with_columns(
                pl.when(
                    pl.col(goal_itemcol).is_not_null()
                    & pl.col(total_itemcol).is_not_null()
                )
                .then(pl.col(goal_itemcol).truediv(pl.col(total_itemcol)))
                .otherwise(None)
                .alias(goal_itemcol)
            )

        return data

    def convert_temperature_F_to_C(
        self,
        data: pl.LazyFrame,
        itemid_F: str,
        itemid_C: str,
        labelcol: str = "LABEL",
        valuecol: str = "VALUENUM",
    ) -> pl.LazyFrame:
        """
        Convert temperature values to Celsius.
        """
        return data.with_columns(
            pl.when(pl.col(labelcol) == itemid_F)
            .then((pl.col(valuecol) - 32) * 5 / 9)
            .otherwise(pl.col(valuecol))
            .alias(valuecol)
        ).with_columns(
            pl.when(pl.col(labelcol) == itemid_F)
            .then(pl.lit(itemid_C))
            .otherwise(pl.col(labelcol))
        )

    def GENERIC_CONVERTER(
        self,
        data: pl.LazyFrame,
        itemid: str,
        labelcol: str = "LABEL",
        valuecol: str = "VALUENUM",
        structfield: str = None,
        factor: float = 1,
    ) -> pl.LazyFrame:
        """
        Convert values from one unit to another.
        """

        if structfield is not None:
            return (
                data.unnest(valuecol)
                .with_columns(
                    pl.when(pl.col(labelcol) == itemid)
                    .then(pl.col("value") * factor)
                    .otherwise(pl.col("value"))
                    .alias("value")
                )
                .select(
                    pl.exclude("value", "system", "method", "time", "LOINC"),
                    pl.struct(
                        value="value",
                        system="system",
                        method="method",
                        time="time",
                        LOINC="LOINC",
                    ).alias(valuecol),
                )
            )

        return data.with_columns(
            pl.when(pl.col(labelcol) == itemid)
            .then(pl.col(valuecol) * factor)
            .otherwise(pl.col(valuecol))
            .alias(valuecol)
        )

    def convert_ammonia_ug_dL_to_umol_L(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """
        Convert ammonia values from µg/dL to µmol/L.
        """
        return data.pipe(self.GENERIC_CONVERTER, factor=0.59, **kwargs)

    def convert_bilirubin_mg_dL_to_umol_L(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """
        Convert bilirubin total values from mg/dL to µmol/L.
        """
        return data.pipe(self.GENERIC_CONVERTER, factor=17.1, **kwargs)

    def convert_bilirubin_umol_L_to_mg_dL(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """
        Convert bilirubin total values from µmol/L to mg/dL.
        """
        return data.pipe(self.GENERIC_CONVERTER, factor=1 / 0.0585, **kwargs)

    def convert_blood_urea_nitrogen_mg_dL_to_mmol_L(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """
        Convert blood urea nitrogen values from mg/dL to mmol/L.
        """
        return data.pipe(self.GENERIC_CONVERTER, factor=0.357, **kwargs)

    def convert_blood_urea_nitrogen_mmol_L_to_mg_dL(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """
        Convert blood urea nitrogen values from mmol/L to mg/dL.
        """
        return data.pipe(self.GENERIC_CONVERTER, factor=1 / 0.357, **kwargs)

    def convert_urea_nitrogen_from_urea(
        self,
        data: pl.LazyFrame,
        itemid_urea: str = "urea",
        itemid_BUN: str = "urea_nitrogen",
        labelcol: str = "LABEL",
        valuecol: str = "VALUENUM",
        structfield: str = None,
    ) -> pl.LazyFrame:
        """
        Convert urea nitrogen values from urea.
        """

        if structfield is not None:
            return (
                data.unnest(valuecol)
                .with_columns(
                    pl.when(pl.col(labelcol) == itemid_urea)
                    .then(pl.col("value") * 0.467)
                    .otherwise(pl.col("value"))
                    .alias("value"),
                    pl.when(pl.col(labelcol) == itemid_urea)
                    .then(pl.lit(itemid_BUN))
                    .otherwise(pl.col(labelcol))
                    .alias(labelcol),
                )
                .select(
                    pl.exclude("value", "system", "method", "time", "LOINC"),
                    pl.struct(
                        value="value",
                        system="system",
                        method="method",
                        time="time",
                        LOINC="LOINC",
                    ).alias(valuecol),
                )
            )

        return data.with_columns(
            pl.when(pl.col(labelcol) == itemid_urea)
            .then(pl.col(valuecol) * 0.467)
            .otherwise(pl.col(valuecol))
            .alias(valuecol),
            pl.when(pl.col(labelcol) == itemid_urea)
            .then(pl.lit(itemid_BUN))
            .otherwise(pl.col(labelcol))
            .alias(labelcol),
        )

    def convert_calcium_mg_dL_to_mmol_L(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """
        Convert calcium values from mg/dL to mmol/L.
        """
        return data.pipe(self.GENERIC_CONVERTER, factor=0.2495, **kwargs)

    def convert_CKMB_ng_mL_to_U_L(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """
        Convert CKMB values from ng/mL to U/L.
        Does nothing, but is used for consistency.

        1 ng/mL = 1 µg/L
        1 µg/L  = 0.01667 µkat/L
        1 µkat/L = 60 U/L

        1 ng/mL = 1 * 0.01667 * 60 U/L = 1 U/L
        """
        return data.pipe(self.GENERIC_CONVERTER, factor=1, **kwargs)

    def convert_creatinine_mg_dL_to_umol_L(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """
        Convert creatinine values from mg/dL to µmol/L.
        """
        return data.pipe(self.GENERIC_CONVERTER, factor=88.4, **kwargs)

    def convert_creatinine_umol_L_to_mg_dL(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """
        Convert creatinine values from µmol/L to mg/dL.
        """
        return data.pipe(self.GENERIC_CONVERTER, factor=1 / 88.4, **kwargs)

    def convert_creatinine_mmol_L_to_mg_dL(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """
        Convert creatinine values from mmol/L to mg/dL.
        """
        return data.pipe(self.GENERIC_CONVERTER, factor=11.312, **kwargs)

    def convert_cholesterol_mmol_L_to_mg_dL(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """
        Convert total cholesterol values from mmol/L to mg/dL.
        """
        return data.pipe(self.GENERIC_CONVERTER, factor=38.665, **kwargs)

    def convert_cortisol_nmol_L_to_ug_dL(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """
        Convert cortisol values from nmol/L to µg/dL.
        """
        return data.pipe(self.GENERIC_CONVERTER, factor=0.0363, **kwargs)

    def convert_FEU_to_DDU(
        self,
        data: pl.LazyFrame,
        itemid: str,
        labelcol: str = "LABEL",
        valuecol: str = "VALUENUM",
        structfield: str = None,
    ) -> pl.LazyFrame:
        """
        Convert D-Dimer values from FEU to DDU.
        """

        if structfield is not None:
            return (
                data.unnest(valuecol)
                .with_columns(
                    pl.when(pl.col(labelcol) == itemid)
                    .then(pl.col("value") / 2)
                    .otherwise(pl.col("value"))
                    .alias("value"),
                    pl.when(pl.col(labelcol) == itemid)
                    .then(pl.col(labelcol).str.replace("FEU", "DDU"))
                    .otherwise(pl.col(labelcol))
                    .alias(labelcol),
                )
                .select(
                    pl.exclude("value", "system", "method", "time", "LOINC"),
                    pl.struct(
                        value="value",
                        system="system",
                        method="method",
                        time="time",
                        LOINC="LOINC",
                    ).alias(valuecol),
                )
            )

        return data.with_columns(
            pl.when(pl.col(labelcol) == itemid)
            .then(pl.col(valuecol) / 2)
            .otherwise(pl.col(valuecol))
            .alias(valuecol),
            # Replace FEU with DDU in the label
            pl.when(pl.col(labelcol) == itemid)
            .then(pl.col(labelcol).str.replace("FEU", "DDU"))
            .otherwise(pl.col(labelcol))
            .alias(labelcol),
        )

    def convert_folate_nmol_L_to_ng_mL(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """
        Convert folate values from nmol/L to ng/mL.
        """
        return data.pipe(self.GENERIC_CONVERTER, factor=2.265, **kwargs)

    def convert_glucose_mg_dL_to_mmol_L(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """
        Convert glucose values from mg/dL to mmol/L.
        """
        return data.pipe(self.GENERIC_CONVERTER, factor=0.0555, **kwargs)

    def convert_glucose_mmol_L_to_mg_dL(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """
        Convert glucose values from mmol/L to mg/dL.
        """
        return data.pipe(self.GENERIC_CONVERTER, factor=1 / 0.0555, **kwargs)

    def convert_hemoglobin_mmol_L_to_g_dL(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """
        Convert hemoglobin values from mmol/L to g/dL.
        """
        return data.pipe(self.GENERIC_CONVERTER, factor=1.61, **kwargs)

    def convert_iron_ug_dL_to_umol_L(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """
        Convert iron values from µg/dL to µmol/L.
        """
        return data.pipe(self.GENERIC_CONVERTER, factor=0.179, **kwargs)

    def convert_magnesium_mg_dL_to_mmol_L(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """
        Convert magnesium values from mg/dL to mmol/L.
        """
        return data.pipe(self.GENERIC_CONVERTER, factor=0.4114, **kwargs)

    def convert_phosphate_mg_dL_to_mmol_L(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """
        Convert phosphate values from mg/dL to mmol/L.
        """
        return data.pipe(self.GENERIC_CONVERTER, factor=0.323, **kwargs)

    def convert_T3_ng_dL_to_nmol_L(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """
        Convert T3 values from ng/dL to nmol/L.
        """
        return data.pipe(self.GENERIC_CONVERTER, factor=0.0154, **kwargs)

    def convert_T4_ug_dL_to_nmol_L_or_ng_dL_to_pmol_L(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """
        Convert T4 values from µg/dL to nmol/L or from ng/dL to pmol/L.
        """
        return data.pipe(self.GENERIC_CONVERTER, factor=12.9, **kwargs)

    def convert_triglycerides_mmol_L_to_mg_dL(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """
        Convert triglycerides values from mmol/L to mg/dL.
        """
        return data.pipe(self.GENERIC_CONVERTER, factor=88.5, **kwargs)

    def convert_urate_umol_L_to_mg_dL(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """
        Convert urate values from µmol/L to mg/dL.
        """
        return data.pipe(self.GENERIC_CONVERTER, factor=16.9, **kwargs)

    def convert_VitB12_pg_mL_to_pmol_L(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """
        Convert Vitamin B12 values from pg/mL to pmol/L.
        """
        return data.pipe(self.GENERIC_CONVERTER, factor=0.738, **kwargs)

    def convert_g_dL_to_g_L(self, data: pl.LazyFrame, **kwargs) -> pl.LazyFrame:
        """
        Convert values from g/dL to g/L.
        """
        return data.pipe(self.GENERIC_CONVERTER, factor=10, **kwargs)

    def convert_g_L_to_g_dL(self, data: pl.LazyFrame, **kwargs) -> pl.LazyFrame:
        """
        Convert values from g/L to g/dL.
        """
        return data.pipe(self.GENERIC_CONVERTER, factor=1 / 10, **kwargs)

    def convert_g_L_to_mg_dL(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """
        Convert values from g/L to mg/dL.
        """
        return data.pipe(self.GENERIC_CONVERTER, factor=100, **kwargs)

    def convert_mg_dL_to_mg_L(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """
        Convert values from mg/dL to mg/L.
        """
        return data.pipe(self.GENERIC_CONVERTER, factor=10, **kwargs)

    def convert_mg_L_to_mg_dL(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """
        Convert values from mg/dL to mg/L.
        """
        return data.pipe(self.GENERIC_CONVERTER, factor=1 / 10, **kwargs)

    def convert_ng_L_to_ug_L(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """
        Convert values from ng/L to µg/L.
        """
        return data.pipe(self.GENERIC_CONVERTER, factor=1 / 1000, **kwargs)

    def convert_ug_L_to_ng_L(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """
        Convert values from µg/L to ng/L.
        """
        return data.pipe(self.GENERIC_CONVERTER, factor=1000, **kwargs)

    def convert_ng_mL_to_ug_L(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """
        Convert values from ng/mL to µg/L.
        Does nothing, but is used for consistency.
        """
        return data.pipe(self.GENERIC_CONVERTER, factor=1, **kwargs)

    def convert_ng_mL_to_mg_L(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """
        Convert values from ng/mL to mg/L.
        """
        return data.pipe(self.GENERIC_CONVERTER, factor=1 / 1000, **kwargs)

    def convert_ng_mL_to_ng_L(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """
        Convert values from ng/mL to ng/L.
        """
        return data.pipe(self.GENERIC_CONVERTER, factor=1000, **kwargs)

    def convert_mEq_L_to_mmol_L(
        self, data: pl.LazyFrame, ions: int = 1, **kwargs
    ) -> pl.LazyFrame:
        """
        Convert values from mEq/L to mmol/L, e.g. for sodium and potassium.
        """
        return data.pipe(self.GENERIC_CONVERTER, factor=ions, **kwargs)

    def convert_ratio_to_percentage(
        self,
        data: pl.LazyFrame,
        itemid: str,
        labelcol: str = "LABEL",
        valuecol: str = "VALUENUM",
        structfield: str = None,
    ) -> pl.LazyFrame:
        """
        Convert ratios to percentages (i.e., 0.23 to 23%).
        """

        if structfield is not None:
            return (
                data.unnest(valuecol)
                .with_columns(
                    pl.when(
                        (pl.col(labelcol) == itemid) & (pl.col("value") <= 2)
                    )
                    .then(pl.col("value") * 100)
                    .otherwise(pl.col("value"))
                    .alias("value"),
                )
                .select(
                    pl.exclude("value", "system", "method", "time", "LOINC"),
                    pl.struct(
                        value="value",
                        system="system",
                        method="method",
                        time="time",
                        LOINC="LOINC",
                    ).alias(valuecol),
                )
            )

        return data.with_columns(
            pl.when((pl.col(labelcol) == itemid) & (pl.col(valuecol) <= 2))
            .then(pl.col(valuecol) * 100)
            .otherwise(pl.col(valuecol))
            .alias(valuecol)
        )


class UnitConverter(UnitConversions):
    def __init__(self):
        super().__init__()
