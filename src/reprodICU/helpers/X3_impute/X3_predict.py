import polars as pl


def predict_arterial_venous_blood_gas(
    labs: pl.LazyFrame,
    vitals: pl.LazyFrame,
    resp: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Predicts whether a blood gas sample is arterial or venous using logistic regression.

    Based on https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/concepts/pivot/pivoted_bg_art.sql
    
    See other/ABG_VBG_MODEL.py for model training details.

    Adds columns:
        - specimen_prob: Probability of being arterial (0-1)
        - specimen_type: "arterial" if specimen_prob > 0.5, else "venous"

    Only predicts if Oxygen value is present.
    """

    STAY_KEY = "Global ICU Stay ID"
    TIME_KEY = "Time Relative to Admission (seconds)"
    SECONDS_IN_4H = 4 * 60 * 60

    LAB_COLUMNS = [
        "Oxygen",
        "Oxygen saturation",
        "Carbon dioxide",
        "pH",
        "Lactate",
        "Bicarbonate",
        "Hemoglobin",
    ]

    def sigmoid(c: pl.Expr) -> pl.Expr:
        return 1 / ((-c).exp() + 1)

    SPECIMEN = (
        labs.select(STAY_KEY, TIME_KEY, *LAB_COLUMNS)
        .with_columns(
            pl.when(
                pl.col(col)
                .struct.field("system")
                .str.contains_any(["Blood", "Serum"])
            )
            .then(pl.col(col).struct.field("value"))
            .otherwise(None)
            .alias(col)
            for col in LAB_COLUMNS
        )
        .join_asof(
            vitals.select(
                STAY_KEY,
                TIME_KEY,
                pl.col("Peripheral oxygen saturation").alias("SpO2"),
            ),
            by=STAY_KEY,
            on=TIME_KEY,
            strategy="backward",
            tolerance=SECONDS_IN_4H,
        )
        .join_asof(
            resp.with_columns(
                pl.max_horizontal(
                    "Oxygen/Total gas setting [Volume Fraction] Ventilator",
                    "Oxygen/Gas total [Pure volume fraction] Inhaled gas",
                ).alias("FiO2")
            )
            .select(
                STAY_KEY,
                TIME_KEY,
                pl.when(pl.col("FiO2").is_between(0, 1))
                .then(pl.col("FiO2") * 100)
                .when(pl.col("FiO2").is_between(1, 100))
                .then(pl.col("FiO2"))
                .otherwise(None)
                .alias("FiO2"),
            )
            .drop_nulls("FiO2"),
            by=STAY_KEY,
            on=TIME_KEY,
            strategy="backward",
            tolerance=SECONDS_IN_4H,
        )
        .filter(pl.col("Oxygen").is_not_null())
        .with_columns(
            (
                0.02576
                + 0.04814 * pl.col("Oxygen")
                + pl.coalesce(
                    0.13269 * pl.col("Oxygen saturation"),
                    0.13269 * 96.60000 + -2.41699,
                )
                + pl.coalesce(
                    -0.00505 * pl.col("Carbon dioxide"),
                    -0.00505 * 36.00000 + -0.02115,
                )
                + pl.coalesce(
                    0.41277 * pl.col("pH"),
                    0.41277 * 7.40400 + -0.00002,
                )
                + pl.coalesce(
                    0.07432 * pl.col("Lactate"),
                    0.07432 * 1.40000 + 0.31148,
                )
                + pl.coalesce(
                    0.07749 * pl.col("Bicarbonate"),
                    0.07749 * 24.50000 + -1.43942,
                )
                + pl.coalesce(
                    -0.00194 * pl.col("Hemoglobin"),
                    -0.00194 * 9.60000 + -1.49415,
                )
                + pl.coalesce(
                    -0.14933 * pl.col("SpO2"),
                    -0.14933 * 98.00000 + -0.99259,
                )
                + pl.coalesce(
                    0.01014 * pl.col("FiO2"),
                    0.01014 * 50.00000 + -0.91143,
                )
            )
            .pipe(sigmoid)
            .alias("specimen_prob")
        )
        .select(
            STAY_KEY,
            TIME_KEY,
            pl.when(pl.col("specimen_prob") > 0.5)
            .then(pl.lit("arterial"))
            .otherwise(pl.lit("venous"))
            .alias("specimen"),
        )
    )

    return labs.join(
        SPECIMEN, on=[STAY_KEY, TIME_KEY], how="left"
    ).with_columns(
        pl.when(pl.col(col).struct.field("system") == "Blood")
        .then(
            pl.col(col).struct.with_fields(
                system=pl.concat_str(pl.lit("Blood "), pl.col("specimen"))
            )
        )
        .otherwise(pl.col(col))
        .alias(col)
        for col in [
            "Oxygen",
            "Oxygen saturation",
            "Carbon dioxide",
            "pH",
            "Bicarbonate",
        ]
    )
