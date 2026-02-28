# Author: Finn Fassbender
# Last modified: 2025-10-30

# Enables the easy conversion of the data.
# Conversion constants were taken from: https://www.labcorp.com/resource/si-unit-conversion-table

import polars as pl

from .helper import GlobalVars
from .helper_OMOP import Vocabulary


def _struct_with_all_null_to_null(
    frame: pl.DataFrame,
    struct_cols: list[str],
) -> pl.DataFrame:
    """
    Replace struct columns with null when all struct fields are null.

    Steps:
        1. Check if any field within each struct column is non-null.
        2. Keep struct if any field is non-null; replace with null otherwise.

    Returns:
        pl.DataFrame: Modified DataFrame with null-all-fields structs replaced by null.
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
        for struct_col in struct_cols
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
        Combine Glasgow Coma Scale (GCS) component subscores into total score.

        Steps:
            1. If total score is null, sum eye + motor + verbal subscores.
            2. Otherwise keep existing total score value.

        Returns:
            pl.LazyFrame: Data with calculated {total_score_col} column (or existing value preserved).
        """
        return data.with_columns(
            pl.when(pl.col(total_score).is_null())
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
class UnitConversions(GlobalVars):
    def __init__(self):
        super().__init__()

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
        """Convert absolute counts to relative counts (as percentage).

        Steps:
            1. Ensure goal column exists; create with null struct if missing.
            2. For structfield: decode JSON, calculate relative percentage, rebuild struct.
            3. For non-struct: divide itemcol by total_itemcol and multiply by 100.

        Returns:
            pl.LazyFrame: Data with relative count values replacing absolute counts.
        """
        if goal_itemcol is None:
            goal_itemcol = itemcol

        if goal_itemcol not in data.collect_schema().names():
            data = data.with_columns(
                pl.struct(
                    value=pl.lit(None),
                    system=pl.lit(None),
                    method=pl.lit(None),
                    time=pl.lit(None),
                    LOINC=pl.lit(None),
                )
                .struct.json_encode()
                .alias(goal_itemcol)
            )

        temp_itemcol = f"temp_{itemcol}"

        if structfield is not None:
            if structstring:
                data = data.with_columns(
                    pl.col(itemcol)
                    .str.json_decode(self.labstructdtype)
                    .alias(temp_itemcol),
                    pl.col(goal_itemcol).str.json_decode(self.labstructdtype),
                    pl.col(total_itemcol).str.json_decode(self.labstructdtype),
                )

            data = (
                data.unnest(temp_itemcol)
                .with_columns(
                    pl.when(
                        pl.col("value").is_not_null(),
                        pl.col(total_itemcol)
                        .struct.field("value")
                        .is_not_null(),
                    )
                    .then(
                        pl.col("value").truediv(
                            pl.col(total_itemcol).struct.field("value")
                        )
                    )
                    .otherwise(None)
                    .alias("value")
                )
                # Combine the columns back into a struct again
                .select(
                    pl.exclude(
                        "value",
                        "system",
                        "method",
                        "time",
                        "LOINC",
                        goal_itemcol,  # avoid name clash
                    ),
                    pl.struct(
                        value=pl.coalesce(
                            pl.when(pl.col(goal_itemcol).is_not_null())
                            .then(pl.col(goal_itemcol).struct.field("value"))
                            .otherwise(None),
                            pl.col("value"),
                        ),
                        system=pl.coalesce(
                            pl.when(pl.col(goal_itemcol).is_not_null())
                            .then(pl.col(goal_itemcol).struct.field("system"))
                            .otherwise(None),
                            pl.col("system"),
                        ),
                        method=pl.coalesce(
                            pl.when(pl.col(goal_itemcol).is_not_null())
                            .then(pl.col(goal_itemcol).struct.field("method"))
                            .otherwise(None),
                            pl.col("method"),
                        ),
                        time=pl.coalesce(
                            pl.when(pl.col(goal_itemcol).is_not_null())
                            .then(pl.col(goal_itemcol).struct.field("time"))
                            .otherwise(None),
                            pl.col("time"),
                        ),
                        LOINC=pl.coalesce(
                            pl.when(pl.col(goal_itemcol).is_not_null())
                            .then(pl.col(goal_itemcol).struct.field("LOINC"))
                            .otherwise(None),
                            pl.lit(None),
                        ),
                    ).alias(goal_itemcol),
                )
                .pipe(
                    _struct_with_all_null_to_null,
                    struct_cols=[goal_itemcol, total_itemcol],
                )
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
        """Convert temperature values from Fahrenheit to Celsius.

        Steps:
            1. Filter rows where label matches itemid_F.
            2. Apply conversion formula: (value - 32) * 5/9.
            3. Update label from itemid_F to itemid_C.

        Returns:
            pl.LazyFrame: Data with converted temperature values and updated labels.
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
        changes_LOINC_property: bool = True,
    ) -> pl.LazyFrame:
        """Convert unit values by applying a multiplicative conversion factor.

        Steps:
            1. Filter rows where label matches itemid.
            2. Apply conversion: multiply value column by factor.
            3. For structfield: decode struct, apply conversion, update LOINC if changes_LOINC_property=True.

        Returns:
            pl.LazyFrame: Data with converted values; LOINC set to null if changes_LOINC_property=True.
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
                        value=pl.col("value"),
                        system=pl.col("system"),
                        method=pl.col("method"),
                        time=pl.col("time"),
                        # If changes_LOINC_property is True, set LOINC to None
                        LOINC=(
                            pl.lit(None)
                            if changes_LOINC_property
                            else pl.col("LOINC")
                        ),
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
        """Convert ammonia values from µg/dL to µmol/L (factor: 0.59)."""
        return data.pipe(self.GENERIC_CONVERTER, factor=0.59, **kwargs)

    def convert_bilirubin_mg_dL_to_umol_L(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """Convert bilirubin values from mg/dL to µmol/L (factor: 17.1)."""
        return data.pipe(self.GENERIC_CONVERTER, factor=17.1, **kwargs)

    def convert_bilirubin_umol_L_to_mg_dL(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """Convert bilirubin values from µmol/L to mg/dL (factor: 1/17.1)."""
        return data.pipe(self.GENERIC_CONVERTER, factor=1 / 17.1, **kwargs)

    def convert_blood_urea_nitrogen_mg_dL_to_mmol_L(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """Convert blood urea nitrogen values from mg/dL to mmol/L (factor: 0.357)."""
        return data.pipe(self.GENERIC_CONVERTER, factor=0.357, **kwargs)

    def convert_blood_urea_nitrogen_mmol_L_to_mg_dL(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """Convert blood urea nitrogen values from mmol/L to mg/dL (factor: 1/0.357)."""
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
        """Convert urea nitrogen values from urea by multiplying by 0.467.

        Steps:
            1. Filter rows where label matches itemid_urea.
            2. Apply conversion: multiply value by 0.467.
            3. Update label from itemid_urea to itemid_BUN (preserves LOINC).

        Returns:
            pl.LazyFrame: Data with converted urea nitrogen values and updated labels.
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
                        LOINC="LOINC",  # does not change LOINC property
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
        """Convert calcium values from mg/dL to mmol/L (factor: 0.2495)."""
        return data.pipe(self.GENERIC_CONVERTER, factor=0.2495, **kwargs)

    def convert_CKMB_ng_mL_to_U_L(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """Convert CKMB values from ng/mL to U/L (factor: 1, identity conversion)."""
        return data.pipe(self.GENERIC_CONVERTER, factor=1, **kwargs)

    def convert_creatinine_mg_dL_to_umol_L(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """Convert creatinine values from mg/dL to µmol/L (factor: 88.4)."""
        return data.pipe(self.GENERIC_CONVERTER, factor=88.4, **kwargs)

    def convert_creatinine_umol_L_to_mg_dL(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """Convert creatinine values from µmol/L to mg/dL (factor: 1/88.4)."""
        return data.pipe(self.GENERIC_CONVERTER, factor=1 / 88.4, **kwargs)

    def convert_creatinine_mmol_L_to_mg_dL(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """Convert creatinine values from mmol/L to mg/dL (factor: 11.312)."""
        return data.pipe(self.GENERIC_CONVERTER, factor=11.312, **kwargs)

    def convert_cholesterol_mmol_L_to_mg_dL(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """Convert total cholesterol values from mmol/L to mg/dL (factor: 38.665)."""
        return data.pipe(self.GENERIC_CONVERTER, factor=38.665, **kwargs)

    def convert_cortisol_nmol_L_to_ug_dL(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """Convert cortisol values from nmol/L to µg/dL (factor: 0.0363)."""
        return data.pipe(self.GENERIC_CONVERTER, factor=0.0363, **kwargs)

    def convert_FEU_to_DDU(
        self,
        data: pl.LazyFrame,
        itemid: str,
        labelcol: str = "LABEL",
        valuecol: str = "VALUENUM",
        structfield: str = None,
    ) -> pl.LazyFrame:
        """Convert D-Dimer values from FEU (Fibrinogen Equivalent Unit) to DDU (D-Dimer Unit).

        Steps:
            1. Filter rows where label matches itemid.
            2. Apply conversion: divide value by 2.
            3. Update label: replace "FEU" with "DDU" in label string.

        Returns:
            pl.LazyFrame: Data with converted D-Dimer values and updated labels.
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
        """Convert folate values from nmol/L to ng/mL (factor: 2.265)."""
        return data.pipe(self.GENERIC_CONVERTER, factor=2.265, **kwargs)

    def convert_glucose_mg_dL_to_mmol_L(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """Convert glucose values from mg/dL to mmol/L (factor: 0.0555)."""
        return data.pipe(self.GENERIC_CONVERTER, factor=0.0555, **kwargs)

    def convert_glucose_mmol_L_to_mg_dL(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """Convert glucose values from mmol/L to mg/dL (factor: 1/0.0555)."""
        return data.pipe(self.GENERIC_CONVERTER, factor=1 / 0.0555, **kwargs)

    def convert_hemoglobin_mmol_L_to_g_dL(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """Convert hemoglobin values from mmol/L to g/dL (factor: 1.61)."""
        return data.pipe(self.GENERIC_CONVERTER, factor=1.61, **kwargs)

    def convert_iron_ug_dL_to_umol_L(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """Convert iron values from µg/dL to µmol/L (factor: 0.179)."""
        return data.pipe(self.GENERIC_CONVERTER, factor=0.179, **kwargs)

    def convert_magnesium_mg_dL_to_mmol_L(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """Convert magnesium values from mg/dL to mmol/L (factor: 0.4114)."""
        return data.pipe(self.GENERIC_CONVERTER, factor=0.4114, **kwargs)

    def convert_phosphate_mg_dL_to_mmol_L(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """Convert phosphate values from mg/dL to mmol/L (factor: 0.323)."""
        return data.pipe(self.GENERIC_CONVERTER, factor=0.323, **kwargs)

    def convert_T3_ng_dL_to_nmol_L(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """Convert T3 values from ng/dL to nmol/L (factor: 0.0154)."""
        return data.pipe(self.GENERIC_CONVERTER, factor=0.0154, **kwargs)

    def convert_T4_ug_dL_to_nmol_L_or_ng_dL_to_pmol_L(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """Convert T4 values from µg/dL to nmol/L or from ng/dL to pmol/L (factor: 12.9)."""
        return data.pipe(self.GENERIC_CONVERTER, factor=12.9, **kwargs)

    def convert_triglycerides_mmol_L_to_mg_dL(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """Convert triglycerides values from mmol/L to mg/dL (factor: 88.5)."""
        return data.pipe(self.GENERIC_CONVERTER, factor=88.5, **kwargs)

    def convert_urate_umol_L_to_mg_dL(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """Convert urate values from µmol/L to mg/dL (factor: 16.9)."""
        return data.pipe(self.GENERIC_CONVERTER, factor=16.9, **kwargs)

    def convert_VitB12_pg_mL_to_pmol_L(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """Convert Vitamin B12 values from pg/mL to pmol/L (factor: 0.738)."""
        return data.pipe(self.GENERIC_CONVERTER, factor=0.738, **kwargs)

    def convert_g_dL_to_g_L(self, data: pl.LazyFrame, **kwargs) -> pl.LazyFrame:
        """Convert values from g/dL to g/L (factor: 10, LOINC unchanged)."""
        return data.pipe(
            self.GENERIC_CONVERTER,
            factor=10,
            changes_LOINC_property=False,
            **kwargs,
        )

    def convert_g_L_to_g_dL(self, data: pl.LazyFrame, **kwargs) -> pl.LazyFrame:
        """Convert values from g/L to g/dL (factor: 1/10, LOINC unchanged)."""
        return data.pipe(
            self.GENERIC_CONVERTER,
            factor=1 / 10,
            changes_LOINC_property=False,
            **kwargs,
        )

    def convert_g_L_to_mg_dL(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """Convert values from g/L to mg/dL (factor: 100, LOINC unchanged)."""
        return data.pipe(
            self.GENERIC_CONVERTER,
            factor=100,
            changes_LOINC_property=False,
            **kwargs,
        )

    def convert_mg_dL_to_mg_L(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """Convert values from mg/dL to mg/L (factor: 10, LOINC unchanged)."""
        return data.pipe(
            self.GENERIC_CONVERTER,
            factor=10,
            changes_LOINC_property=False,
            **kwargs,
        )

    def convert_mg_L_to_mg_dL(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """Convert values from mg/L to mg/dL (factor: 1/10, LOINC unchanged)."""
        return data.pipe(
            self.GENERIC_CONVERTER,
            factor=1 / 10,
            changes_LOINC_property=False,
            **kwargs,
        )

    def convert_ng_L_to_ug_L(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """Convert values from ng/L to µg/L (factor: 1/1000, LOINC unchanged)."""
        return data.pipe(
            self.GENERIC_CONVERTER,
            factor=1 / 1000,
            changes_LOINC_property=False,
            **kwargs,
        )

    def convert_ug_L_to_ng_L(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """Convert values from µg/L to ng/L (factor: 1000, LOINC unchanged)."""
        return data.pipe(
            self.GENERIC_CONVERTER,
            factor=1000,
            changes_LOINC_property=False,
            **kwargs,
        )

    def convert_ng_mL_to_ug_L(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """Convert values from ng/mL to µg/L (factor: 1, identity, LOINC unchanged)."""
        return data.pipe(
            self.GENERIC_CONVERTER,
            factor=1,
            changes_LOINC_property=False,
            **kwargs,
        )

    def convert_ng_mL_to_mg_L(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """Convert values from ng/mL to mg/L (factor: 1/1000, LOINC unchanged)."""
        return data.pipe(
            self.GENERIC_CONVERTER,
            factor=1 / 1000,
            changes_LOINC_property=False,
            **kwargs,
        )

    def convert_ng_mL_to_ng_L(
        self, data: pl.LazyFrame, **kwargs
    ) -> pl.LazyFrame:
        """Convert values from ng/mL to ng/L (factor: 1000, LOINC unchanged)."""
        return data.pipe(
            self.GENERIC_CONVERTER,
            factor=1000,
            changes_LOINC_property=False,
            **kwargs,
        )

    def convert_mEq_L_to_mmol_L(
        self, data: pl.LazyFrame, ions: int = 1, **kwargs
    ) -> pl.LazyFrame:
        """Convert electrolyte values from mEq/L to mmol/L (factor: ions parameter).

        Steps:
            1. Apply multiplication by ions factor (default: 1 for monovalent ions like Na/K).

        Returns:
            pl.LazyFrame: Data with converted electrolyte values.
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
        """Convert ratio values to percentages (e.g., 0.23 to 23%).

        Steps:
            1. Filter rows where label matches itemid and value ≤ 2.
            2. Multiply value by 100 to convert ratio to percentage.

        Returns:
            pl.LazyFrame: Data with ratio values converted to percentages.
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

    def rename_anion_gap(
        self,
        data: pl.LazyFrame,
        itemid: str = "Anion gap 4",
        labelcol: str = "LABEL",
        valuecol: str = "VALUENUM",
        structfield: str = None,
    ) -> pl.LazyFrame:
        """
        Rename "Anion gap 4" to "Anion gap" label for consistency.

        Steps:
            1. Filter rows where label matches itemid ("Anion gap 4").
            2. Replace label with "Anion gap".
            3. For structfield: set LOINC to null (property changes).

        Returns:
            pl.LazyFrame: Data with standardized "Anion gap" label; LOINC nulled for structfield.
        """
        if structfield is not None:
            return (
                data.unnest(valuecol)
                .with_columns(
                    pl.when(pl.col(labelcol) == itemid)
                    .then(pl.lit("Anion gap"))
                    .otherwise(pl.col(labelcol))
                    .alias(labelcol)
                )
                .select(
                    pl.exclude("value", "system", "method", "time", "LOINC"),
                    pl.struct(
                        value=pl.col("value"),
                        system=pl.col("system"),
                        method=pl.col("method"),
                        time=pl.col("time"),
                        LOINC=pl.lit(None),  # Anion gap LOINC property changes
                    ).alias(valuecol),
                )
            )

        return data.with_columns(
            pl.when(pl.col(labelcol) == itemid)
            .then(pl.lit("Anion gap"))
            .otherwise(pl.col(labelcol))
            .alias(labelcol)
        )


class UnitConverter(UnitConversions):
    def __init__(self):
        super().__init__()

    def _decode_lab_structs(
        self,
        lf: pl.LazyFrame,
        cols_to_exclude: list[str] | None = None,
        cols_to_include: list[str] | None = None,
    ) -> pl.LazyFrame:
        """
        Decode JSON-encoded lab value structs into typed struct format.

        Steps:
            1. Filter columns: exclude specified columns, optionally include only specified columns.
            2. For each remaining column: decode JSON string to struct with fields (value, system, method, time, LOINC).

        Returns:
            pl.LazyFrame: Data with JSON lab values decoded to struct format.
        """

        def decode_lab_struct(lab_value):
            return pl.col(lab_value).str.json_decode(self.labstructdtype)

        columns = lf.collect_schema().names()
        exclude = set(cols_to_exclude or [])
        if cols_to_include is not None:
            include = set(cols_to_include)
            value_cols = [
                c for c in columns if c in include and c not in exclude
            ]
        else:
            value_cols = [c for c in columns if c not in exclude]

        if not value_cols:
            return lf

        return lf.with_columns(*map(decode_lab_struct, value_cols))

    def _assign_LOINC_codes(
        self,
        data: pl.LazyFrame,
        vocab: Vocabulary,
        cols_to_exclude: list[str] = [],
        struct_cols: list[str] | None = None,
        component_col: str | None = None,
    ) -> pl.LazyFrame:
        """
        Assign missing LOINC codes using vocabulary attribute queries (component, property, system, method, time).

        Steps:
            1. Decode JSON lab struct columns if not already decoded.
            2. Extract distinct (component, property, system, method, time) tuples from data.
            3. Query vocabulary; retain pairs matching exactly one LOINC code.
            4. For system/component mismatches, query fallback systems (arterial→venous→blood).
            5. For each struct column: join mapping, fill LOINC with precedence (exact > fallback1 > fallback2), rebuild struct.
            6. Replace all-null structs with null; re-encode to JSON if needed.

        Returns:
            pl.LazyFrame: Data with LOINC codes filled where uniquely determinable from attributes.
        """
        # Assert only one struct column is provided if component_col is not None
        if component_col is not None and struct_cols is not None:
            assert (
                len(struct_cols) == 1
            ), "If component_col is provided, struct_cols must be a single column list."

        # Decode struct columns if requested
        if struct_cols[0] != "labstruct":
            data = self._decode_lab_structs(data, cols_to_include=struct_cols)
        # Make data lazy if not already
        data = data.lazy()

        schema = data.collect_schema()
        inferred_struct_cols = [
            c
            for c, dt in schema.items()
            if c not in cols_to_exclude and dt == pl.Struct
        ]
        # Use manual struct columns when provided
        # Use manual struct columns when provided
        struct_cols = (
            struct_cols if struct_cols is not None else inferred_struct_cols
        )

        # Collect unique (component, property, system, method, time) tuples needing LOINC
        unique_pairs: list[tuple[str, str, str, str, str]] = []
        if component_col is not None:
            # Pre-pivot: component comes from component_col, values live in provided struct columns
            pairs_df = (
                data.select(component_col, struct_cols[0])
                .unnest(struct_cols[0])
                .drop("value")
                .filter(pl.col("LOINC").is_null())
                # Fill null time with "Point in time (spot)" for mapping
                .with_columns(pl.col("time").fill_null("Point in time (spot)"))
                .select(component_col, "system", "method", "time")
                .unique()
                .collect()
            )
            for row in pairs_df.rows(named=True):
                comp = row[component_col]
                prop = self.relevant_lab_LOINC_properties.get(comp, None)
                unique_pairs.append(
                    (comp, prop, row["system"], row["method"], row["time"])
                )
        else:
            # Post-pivot: component equals column name
            for col in sorted(struct_cols):
                component = col
                prop = self.relevant_lab_LOINC_properties.get(col, None)
                pairs_df = (
                    data.select(col)
                    .unnest(col)
                    .drop("value")
                    .filter(pl.col("LOINC").is_null())
                    # Fill null time with "Point in time (spot)" for mapping
                    .with_columns(
                        pl.col("time").fill_null("Point in time (spot)")
                    )
                    .unique()
                    .sort("system", "method", "time")
                    .collect()
                )
                for row in pairs_df.rows(named=True):
                    unique_pairs.append(
                        (
                            component,
                            prop,
                            row["system"],
                            row["method"],
                            row["time"],
                        )
                    )

        # Build mapping via batched vocabulary lookup, keeping fallbacks
        mapping_records = []
        tried = set()

        def enqueue_fallbacks(comp, prop, system, method, time):
            fb = []
            if system in ("Blood arterial", "Blood venous"):
                fb.append((comp, prop, "Blood", method, time))
            if system in ("Blood mixed venous", "Blood central venous"):
                fb.append((comp, prop, "Blood venous", method, time))
            return fb

        # First pass: resolve all unique pairs
        queries = [p for p in unique_pairs if p not in tried]
        tried.update(queries)
        results = vocab.get_LOINC_codes_for_attributes(queries)
        fallback = []

        for (comp, prop, system, method, time), codes in zip(queries, results):
            if len(codes) == 1:
                mapping_records.append(
                    {
                        "component": comp,
                        "property": prop,
                        "system": system,
                        "method": method,
                        "time": time,
                        "LOINC_mapped": codes[0],
                    }
                )
            elif len(codes) == 0:
                fallback.extend(
                    [
                        fb
                        for fb in enqueue_fallbacks(
                            comp, prop, system, method, time
                        )
                        if fb not in tried
                    ]
                )

        # Second pass: resolve system fallbacks in batch
        if fallback:
            tried.update(fallback)
            results = vocab.get_LOINC_codes_for_attributes(fallback)
            for (comp, prop, system, method, time), codes in zip(
                fallback, results
            ):
                if len(codes) == 1:
                    mapping_records.append(
                        {
                            "component": comp,
                            "property": prop,
                            "system": system,
                            "method": method,
                            "time": time,
                            "LOINC_mapped": codes[0],
                        }
                    )

        # Create once; filter or key-join per column below
        mapping_lf = pl.LazyFrame(
            mapping_records,
            schema={
                "component": str,
                "property": str,
                "system": str,
                "method": str,
                "time": str,
                "LOINC_mapped": str,
            },
        )

        if not mapping_records:
            if struct_cols[0] != "labstruct":
                data = data.pipe(
                    _struct_with_all_null_to_null, struct_cols
                ).with_columns(
                    pl.col(c).struct.json_encode().replace("null", None)
                    for c in struct_cols
                )
            return data

        # Update each struct column
        for col in sorted(struct_cols):
            # Per-column mapping view:
            if component_col is None:
                # Wide mode: pre-filter by component == column name
                map_df = mapping_lf.filter(pl.col("component") == col).select(
                    "system", "method", "time", "LOINC_mapped"
                )
            else:
                # Long mode: include component in join keys
                map_df = mapping_lf.select(
                    "component", "system", "method", "time", "LOINC_mapped"
                )

            # Helper: try mapping for a given system column (single join using time_norm)
            def _map_once(lf: pl.LazyFrame, sys_col: str) -> pl.LazyFrame:
                if component_col is None:
                    # Join without component in key
                    # Join without component in key
                    return lf.join(
                        map_df,
                        left_on=[sys_col, "method", "time_norm"],
                        right_on=["system", "method", "time"],
                        how="left",
                        nulls_equal=True,
                        coalesce=True,
                    ).rename({"LOINC_mapped": f"LOINC_{sys_col}"})
                else:
                    # Join with component column in key
                    return lf.join(
                        map_df,
                        left_on=[component_col, sys_col, "method", "time_norm"],
                        right_on=["component", "system", "method", "time"],
                        how="left",
                        nulls_equal=True,
                        coalesce=True,
                    ).rename({"LOINC_mapped": f"LOINC_{sys_col}"})

            # Apply precedence mapping with only three joins (system, fallback1, fallback2)
            data = (
                data.unnest(col)
                .with_columns(
                    pl.coalesce(
                        pl.col("time"), pl.lit("Point in time (spot)")
                    ).alias("time_norm"),
                    pl.when(pl.col("system").str.ends_with("venous"))
                    .then(pl.lit("Blood venous"))
                    .otherwise(pl.lit(None))
                    .alias("fallback1"),
                    pl.when(pl.col("system").str.starts_with("Blood"))
                    .then(pl.lit("Blood"))
                    .otherwise(pl.lit(None))
                    .alias("fallback2"),
                )
                # Stage 1: exact system
                .pipe(_map_once, "system")
                # Stage 2: fallback 1 (mixed/central -> venous)
                .pipe(_map_once, "fallback1")
                # Stage 3: fallback 2 (venous/arterial -> blood; mixed/central -> blood if needed)
                .pipe(_map_once, "fallback2")
                # Reconstruct the struct with precedence-coalesced LOINC
                .select(
                    pl.exclude(
                        "value",
                        "system",
                        "method",
                        "time",
                        "time_norm",
                        "fallback1",
                        "fallback2",
                        "LOINC",
                        "LOINC_system",
                        "LOINC_fallback1",
                        "LOINC_fallback2",
                    ),
                    pl.struct(
                        value="value",
                        system="system",
                        method="method",
                        time=pl.when(pl.col("time") == "Point in time (spot)")
                        .then(None)
                        .otherwise(pl.col("time")),
                        LOINC=pl.coalesce(
                            pl.col("LOINC"),
                            pl.col("LOINC_system"),
                            pl.col("LOINC_fallback1"),
                            pl.col("LOINC_fallback2"),
                        ),
                    ).alias(col),
                )
                .with_columns(
                    pl.when(pl.col(col).struct.field("value").is_null())
                    .then(pl.lit(None))
                    .otherwise(pl.col(col))
                    .alias(col)
                )
            )

        if struct_cols[0] != "labstruct":
            data = data.pipe(
                _struct_with_all_null_to_null, struct_cols
            ).with_columns(
                pl.col(c).struct.json_encode().replace("null", None)
                for c in struct_cols
            )

        return data  # .filter(pl.any_horizontal(pl.col(struct_cols).is_not_null()))
