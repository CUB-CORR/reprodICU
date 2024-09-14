# Author: Finn Fassbender
# Last modified: 2024-09-05

# Enables the easy conversion of the data.
# Conversion constants were taken from: https://www.labcorp.com/resource/si-unit-conversion-table

import polars as pl


# Enables the easy combination of Glasgow Coma Scale (GCS) components.
# ASSUMPTION: data is in wide format, after pivoting.
class GCSCombiner:
    def __init__(self):
        pass

    def combine_gcs_components(
        self,
        data,
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
            .then(pl.col(eye_subscore) + pl.col(motor_subscore) + pl.col(verbal_subscore))
            .otherwise(pl.col(total_score))
            .alias(total_score)
        )


# Enables the easy conversion of the data.
# ASSUMPTION: data is in long format, before pivoting.
class UnitConversions:
    def __init__(self):
        pass

    def convert_absolute_count_to_relative(
        self, data, itemid, total_itemid, labelcol: str = "LABEL", valuecol: str = "VALUENUM"
    ) -> pl.LazyFrame:
        """
        Convert absolute counts to relative counts.
        """
        return data.with_columns(
            pl.when(pl.col(labelcol) == itemid)
            .then(pl.col(valuecol).truediv(pl.col(total_itemid)))
            .otherwise(pl.col(valuecol))
            .alias(valuecol)
        )

    def convert_temperature_F_to_C(
        self, data, itemid_F, itemid_C, labelcol: str = "LABEL", valuecol: str = "VALUENUM"
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
            pl.when(pl.col(labelcol) == itemid_F).then(pl.lit(itemid_C)).otherwise(pl.col(labelcol))
        )

    def convert_ammonia_ug_dL_to_umol_L(
        self, data, itemid, labelcol: str = "LABEL", valuecol: str = "VALUENUM"
    ) -> pl.LazyFrame:
        """
        Convert ammonia values from µg/dL to µmol/L.
        """
        return data.with_columns(
            pl.when(pl.col(labelcol) == itemid)
            .then(pl.col(valuecol) * 0.59)
            .otherwise(pl.col(valuecol))
            .alias(valuecol)
        )

    def convert_bilirubin_mg_dL_to_umol_L(
        self, data, itemid, labelcol: str = "LABEL", valuecol: str = "VALUENUM"
    ) -> pl.LazyFrame:
        """
        Convert bilirubin total values from mg/dL to µmol/L.
        """
        return data.with_columns(
            pl.when(pl.col(labelcol) == itemid)
            .then(pl.col(valuecol) * 17.1)
            .otherwise(pl.col(valuecol))
            .alias(valuecol)
        )

    def convert_blood_urea_nitrogen_mg_dL_to_mmol_L(
        self, data, itemid, labelcol: str = "LABEL", valuecol: str = "VALUENUM"
    ) -> pl.LazyFrame:
        """
        Convert blood urea nitrogen values from mg/dL to mmol/L.
        """
        return data.with_columns(
            pl.when(pl.col(labelcol) == itemid)
            .then(pl.col(valuecol) * 0.357)
            .otherwise(pl.col(valuecol))
            .alias(valuecol)
        )

    def convert_blood_urea_nitrogen_from_urea(
        self,
        data,
        itemid_urea="urea",
        itemid_BUN: str = "blood_urea_nitrogen",
        labelcol: str = "LABEL",
        valuecol: str = "VALUENUM",
    ) -> pl.LazyFrame:
        """
        Convert blood urea nitrogen values from urea.
        """
        return data.with_columns(
            pl.when(pl.col(labelcol) == itemid_urea)
            .then(pl.col(valuecol) * 0.357)
            .otherwise(pl.col(valuecol))
            .alias(valuecol),
            pl.when(pl.col(labelcol) == itemid_urea)
            .then(pl.lit(itemid_BUN))
            .otherwise(pl.col(labelcol))
            .alias(labelcol),
        )

    def convert_calcium_mg_dL_to_mmol_L(
        self, data, itemid, labelcol: str = "LABEL", valuecol: str = "VALUENUM"
    ) -> pl.LazyFrame:
        """
        Convert calcium values from mg/dL to mmol/L.
        """
        return data.with_columns(
            pl.when(pl.col(labelcol) == itemid)
            .then(pl.col(valuecol) * 0.25)
            .otherwise(pl.col(valuecol))
            .alias(valuecol)
        )

    def convert_CKMB_ng_mL_to_U_L(
        self, data, itemid, labelcol: str = "LABEL", valuecol: str = "VALUENUM"
    ) -> pl.LazyFrame:
        """
        Convert CKMB values from ng/mL to U/L.
        Does nothing, but is used for consistency.

        1 ng/mL = 1 µg/L
        1 µg/L  = 0.01667 µkat/L
        1 µkat/L = 60 U/L

        1 ng/mL = 1 * 0.01667 * 60 U/L = 1 U/L
        """
        return data.with_columns(
            pl.when(pl.col(labelcol) == itemid)
            .then(pl.col(valuecol) * 1)
            .otherwise(pl.col(valuecol))
            .alias(valuecol)
        )

    def convert_creatinine_mg_dL_to_umol_L(
        self, data, itemid, labelcol: str = "LABEL", valuecol: str = "VALUENUM"
    ) -> pl.LazyFrame:
        """
        Convert creatinine values from mg/dL to µmol/L.
        """
        return data.with_columns(
            pl.when(pl.col(labelcol) == itemid)
            .then(pl.col(valuecol) * 88.4)
            .otherwise(pl.col(valuecol))
            .alias(valuecol)
        )

    def convert_creatinine_umol_L_to_mg_dL(
        self, data, itemid, labelcol: str = "LABEL", valuecol: str = "VALUENUM"
    ) -> pl.LazyFrame:
        """
        Convert creatinine values from µmol/L to mg/dL.
        """
        return data.with_columns(
            pl.when(pl.col(labelcol) == itemid)
            .then(pl.col(valuecol) / 88.4)
            .otherwise(pl.col(valuecol))
            .alias(valuecol)
        )

    def convert_creatinine_mmol_L_to_mg_dL(
        self, data, itemid, labelcol: str = "LABEL", valuecol: str = "VALUENUM"
    ) -> pl.LazyFrame:
        """
        Convert creatinine values from mmol/L to mg/dL.
        """
        return data.with_columns(
            pl.when(pl.col(labelcol) == itemid)
            .then(pl.col(valuecol) * 11.312)
            .otherwise(pl.col(valuecol))
            .alias(valuecol)
        )

    def convert_cholesterol_mmol_L_to_mg_dL(
        self, data, itemid, labelcol: str = "LABEL", valuecol: str = "VALUENUM"
    ) -> pl.LazyFrame:
        """
        Convert total cholesterol values from mmol/L to mg/dL.
        """
        return data.with_columns(
            pl.when(pl.col(labelcol) == itemid)
            .then(pl.col(valuecol) * 38.665)
            .otherwise(pl.col(valuecol))
            .alias(valuecol)
        )

    def convert_cortisol_nmol_L_to_ug_dL(
        self, data, itemid, labelcol: str = "LABEL", valuecol: str = "VALUENUM"
    ) -> pl.LazyFrame:
        """
        Convert cortisol values from nmol/L to µg/dL.
        """
        return data.with_columns(
            pl.when(pl.col(labelcol) == itemid)
            .then(pl.col(valuecol) * 0.0363)
            .otherwise(pl.col(valuecol))
            .alias(valuecol)
        )

    def convert_hemoglobin_mmol_L_to_g_dL(
        self, data, itemid, labelcol: str = "LABEL", valuecol: str = "VALUENUM"
    ) -> pl.LazyFrame:
        """
        Convert hemoglobin values from mmol/L to g/dL.
        """
        return data.with_columns(
            pl.when(pl.col(labelcol) == itemid)
            .then(pl.col(valuecol) * 1.61)
            .otherwise(pl.col(valuecol))
            .alias(valuecol)
        )

    def convert_glucose_mg_dL_to_mmol_L(
        self, data, itemid, labelcol: str = "LABEL", valuecol: str = "VALUENUM"
    ) -> pl.LazyFrame:
        """
        Convert glucose values from mg/dL to mmol/L.
        """
        return data.with_columns(
            pl.when(pl.col(labelcol) == itemid)
            .then(pl.col(valuecol) * 0.0555)
            .otherwise(pl.col(valuecol))
            .alias(valuecol)
        )

    def convert_glucose_mmol_L_to_mg_dL(
        self, data, itemid, labelcol: str = "LABEL", valuecol: str = "VALUENUM"
    ) -> pl.LazyFrame:
        """
        Convert glucose values from mmol/L to mg/dL.
        """
        return data.with_columns(
            pl.when(pl.col(labelcol) == itemid)
            .then(pl.col(valuecol) / 0.0555)
            .otherwise(pl.col(valuecol))
            .alias(valuecol)
        )

    def convert_iron_ug_dL_to_umol_L(
        self, data, itemid, labelcol: str = "LABEL", valuecol: str = "VALUENUM"
    ) -> pl.LazyFrame:
        """
        Convert iron values from µg/dL to µmol/L.
        """
        return data.with_columns(
            pl.when(pl.col(labelcol) == itemid)
            .then(pl.col(valuecol) * 0.179)
            .otherwise(pl.col(valuecol))
            .alias(valuecol)
        )

    def convert_magnesium_mg_dL_to_mmol_L(
        self, data, itemid, labelcol: str = "LABEL", valuecol: str = "VALUENUM"
    ) -> pl.LazyFrame:
        """
        Convert magnesium values from mg/dL to mmol/L.
        """
        return data.with_columns(
            pl.when(pl.col(labelcol) == itemid)
            .then(pl.col(valuecol) * 0.4114)
            .otherwise(pl.col(valuecol))
            .alias(valuecol)
        )

    def convert_phosphate_mg_dL_to_mmol_L(
        self, data, itemid, labelcol: str = "LABEL", valuecol: str = "VALUENUM"
    ) -> pl.LazyFrame:
        """
        Convert phosphate values from mg/dL to mmol/L.
        """
        return data.with_columns(
            pl.when(pl.col(labelcol) == itemid)
            .then(pl.col(valuecol) * 0.323)
            .otherwise(pl.col(valuecol))
            .alias(valuecol)
        )

    def convert_T3_ng_dL_to_nmol_L(
        self, data, itemid, labelcol: str = "LABEL", valuecol: str = "VALUENUM"
    ) -> pl.LazyFrame:
        """
        Convert T3 values from ng/dL to nmol/L.
        """
        return data.with_columns(
            pl.when(pl.col(labelcol) == itemid)
            .then(pl.col(valuecol) * 0.0154)
            .otherwise(pl.col(valuecol))
            .alias(valuecol)
        )

    def convert_T4_ug_dL_to_nmol_L_or_ng_dL_to_pmol_L(
        self, data, itemid, labelcol: str = "LABEL", valuecol: str = "VALUENUM"
    ) -> pl.LazyFrame:
        """
        Convert T4 values from µg/dL to nmol/L or from ng/dL to pmol/L.
        """
        return data.with_columns(
            pl.when(pl.col(labelcol) == itemid)
            .then(pl.col(valuecol) * 12.9)
            .otherwise(pl.col(valuecol))
            .alias(valuecol)
        )

    def convert_triglycerides_mmol_L_to_mg_dL(
        self, data, itemid, labelcol: str = "LABEL", valuecol: str = "VALUENUM"
    ) -> pl.LazyFrame:
        """
        Convert triglycerides values from mmol/L to mg/dL.
        """
        return data.with_columns(
            pl.when(pl.col(labelcol) == itemid)
            .then(pl.col(valuecol) * 88.5)
            .otherwise(pl.col(valuecol))
            .alias(valuecol)
        )

    def convert_VitB12_pg_mL_to_pmol_L(
        self, data, itemid, labelcol: str = "LABEL", valuecol: str = "VALUENUM"
    ) -> pl.LazyFrame:
        """
        Convert Vitamin B12 values from pg/mL to pmol/L.
        """
        return data.with_columns(
            pl.when(pl.col(labelcol) == itemid)
            .then(pl.col(valuecol) * 0.738)
            .otherwise(pl.col(valuecol))
            .alias(valuecol)
        )

    def convert_g_dL_to_g_L(
        self, data, itemid, labelcol: str = "LABEL", valuecol: str = "VALUENUM"
    ) -> pl.LazyFrame:
        """
        Convert values from g/dL to g/L.
        """
        return data.with_columns(
            pl.when(pl.col(labelcol) == itemid)
            .then(pl.col(valuecol) * 10)
            .otherwise(pl.col(valuecol))
            .alias(valuecol)
        )

    def convert_g_L_to_g_dL(
        self, data, itemid, labelcol: str = "LABEL", valuecol: str = "VALUENUM"
    ) -> pl.LazyFrame:
        """
        Convert values from g/L to g/dL.
        """
        return data.with_columns(
            pl.when(pl.col(labelcol) == itemid)
            .then(pl.col(valuecol) / 10)
            .otherwise(pl.col(valuecol))
            .alias(valuecol)
        )

    def convert_g_L_to_mg_dL(
        self, data, itemid, labelcol: str = "LABEL", valuecol: str = "VALUENUM"
    ) -> pl.LazyFrame:
        """
        Convert values from g/L to mg/dL.
        """
        return data.with_columns(
            pl.when(pl.col(labelcol) == itemid)
            .then(pl.col(valuecol) * 100)
            .otherwise(pl.col(valuecol))
            .alias(valuecol)
        )

    def convert_mg_dL_to_mg_L(
        self, data, itemid, labelcol: str = "LABEL", valuecol: str = "VALUENUM"
    ) -> pl.LazyFrame:
        """
        Convert values from mg/dL to mg/L.
        """
        return data.with_columns(
            pl.when(pl.col(labelcol) == itemid)
            .then(pl.col(valuecol) * 10)
            .otherwise(pl.col(valuecol))
            .alias(valuecol)
        )

    def convert_ng_L_to_ug_L(
        self, data, itemid, labelcol: str = "LABEL", valuecol: str = "VALUENUM"
    ) -> pl.LazyFrame:
        """
        Convert values from ng/L to µg/L.
        Does nothing, but is used for consistency.
        """
        return data.with_columns(
            pl.when(pl.col(labelcol) == itemid)
            .then(pl.col(valuecol) / 1000)
            .otherwise(pl.col(valuecol))
            .alias(valuecol)
        )

    def convert_ug_L_to_ng_L(
        self, data, itemid, labelcol: str = "LABEL", valuecol: str = "VALUENUM"
    ) -> pl.LazyFrame:
        """
        Convert values from µg/L to ng/L.
        """
        return data.with_columns(
            pl.when(pl.col(labelcol) == itemid)
            .then(pl.col(valuecol) * 1000)
            .otherwise(pl.col(valuecol))
            .alias(valuecol)
        )

    def convert_ng_mL_to_ug_L(
        self, data, itemid, labelcol: str = "LABEL", valuecol: str = "VALUENUM"
    ) -> pl.LazyFrame:
        """
        Convert values from ng/mL to µg/L.
        Does nothing, but is used for consistency.
        """
        return data.with_columns(
            pl.when(pl.col(labelcol) == itemid)
            .then(pl.col(valuecol))
            .otherwise(pl.col(valuecol))
            .alias(valuecol)
        )

    def convert_ng_mL_to_mg_L(
        self, data, itemid, labelcol: str = "LABEL", valuecol: str = "VALUENUM"
    ) -> pl.LazyFrame:
        """
        Convert values from ng/mL to mg/L.
        """
        return data.with_columns(
            pl.when(pl.col(labelcol) == itemid)
            .then(pl.col(valuecol) / 1000)
            .otherwise(pl.col(valuecol))
            .alias(valuecol)
        )

    def convert_ng_mL_to_ng_L(
        self, data, itemid, labelcol: str = "LABEL", valuecol: str = "VALUENUM"
    ) -> pl.LazyFrame:
        """
        Convert values from ng/mL to ng/L.
        """
        return data.with_columns(
            pl.when(pl.col(labelcol) == itemid)
            .then(pl.col(valuecol) * 1000)
            .otherwise(pl.col(valuecol))
            .alias(valuecol)
        )

    def convert_mEq_L_to_mmol_L(
        self, data, itemid, ions: int = 1, labelcol: str = "LABEL", valuecol: str = "VALUENUM"
    ) -> pl.LazyFrame:
        """
        Convert values from mEq/L to mmol/L, e.g. for sodium and potassium.
        """
        return data.with_columns(
            pl.when(pl.col(labelcol) == itemid)
            .then(pl.col(valuecol) * ions)
            .otherwise(pl.col(valuecol))
            .alias(valuecol)
        )

    def convert_ratio_to_percentage(
        self, data, itemid, labelcol: str = "LABEL", valuecol: str = "VALUENUM"
    ) -> pl.LazyFrame:
        """
        Convert ratios to percentages (i.e., 0.23 to 23%).
        """
        return data.with_columns(
            pl.when((pl.col(labelcol) == itemid) & (pl.col(valuecol) <= 2))
            .then(pl.col(valuecol) * 100)
            .otherwise(pl.col(valuecol))
            .alias(valuecol)
        )


class UnitConverter(UnitConversions):
    def __init__(self):
        super().__init__()
