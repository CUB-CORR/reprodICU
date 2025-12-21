"""
This script fits a logistic regression model to estimate the source of a blood
gas measurement, whether it is arterial (ABG - Arterial Blood Gas) or venous
(VBG - Venous Blood Gas).

The script performs the following steps:
1. Determines the blood system (arterial/venous) from existing lab data.
2. Prepares prediction data by joining lab, vital, and respiratory features.
3. Trains a logistic regression model with median imputation for missing values.
4. Outputs the model coefficients and a prediction formula for use in SQL or
   other contexts.

Note: The target variable is encoded as 1 for arterial and 0 for venous.
"""

import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

import reprodICU

# Get clinically plausible value ranges for data validation
config_manager = reprodICU.get_config_manager()
PLAUSIBLE_VALUES = config_manager.get_clinically_plausible_values()

# Constants for column names and thresholds
STAY_KEY = "Global ICU Stay ID"
TIME_KEY = "Time Relative to Admission (seconds)"
SECONDS_IN_4H = 4 * 60 * 60

COLUMNS = ["Oxygen", "Oxygen saturation", "Carbon dioxide", "pH", "Bicarbonate"]
DATASETS = ["HiRID", "MIMIC-III", "MIMIC-IV", "SICdb"]
SYSTEMS = ["Blood arterial", "Blood venous"]

# Load dataframes
labs: pl.LazyFrame = reprodICU.labs
resp: pl.LazyFrame = reprodICU.respiratory
vitals: pl.LazyFrame = reprodICU.vitals
info: pl.LazyFrame = reprodICU.patient_information

# Determine the blood system (arterial or venous) based on lab measurements
SYSTEM = (
    labs.select(
        STAY_KEY,
        TIME_KEY,
        *[pl.col(col).struct.field("system").alias(col) for col in COLUMNS],
    )
    .join(
        info.filter(pl.col("Source Dataset").is_in(DATASETS)),
        on=STAY_KEY,
        how="semi",
    )
    .filter(
        pl.any_horizontal(pl.col(col).is_not_null() for col in COLUMNS),
        pl.col("pH") != "Urine",
    )
    .with_columns(
        pl.concat_list([pl.col(col) for col in COLUMNS])
        .list.eval(pl.element().filter(pl.element().is_in(SYSTEMS)))
        .alias("filtered_systems")
    )
    .with_columns(
        pl.col("filtered_systems")
        .list.count_matches("Blood arterial")
        .alias("arterial_count"),
        pl.col("filtered_systems")
        .list.count_matches("Blood venous")
        .alias("venous_count"),
    )
    .select(
        STAY_KEY,
        TIME_KEY,
        pl.when(pl.col("arterial_count") > pl.col("venous_count"))
        .then(pl.lit("Blood arterial"))
        .when(pl.col("venous_count") > pl.col("arterial_count"))
        .then(pl.lit("Blood venous"))
        .when(pl.col("arterial_count") > 0)
        .then(pl.lit("Blood arterial"))
        .otherwise(None)
        .alias("system"),
    )
)

# Columns for prediction features
LAB_COLUMNS = [
    "Oxygen",
    "Oxygen saturation",
    "Carbon dioxide",
    "pH",
    "Lactate",
    "Bicarbonate",
    "Hemoglobin",
]

# Prepare prediction data by selecting and joining relevant columns
PREDICTION_DATA = (
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
    .join(SYSTEM, on=[STAY_KEY, TIME_KEY], how="right")
)

print(f"Collected {PREDICTION_DATA.select(pl.count()).collect().item()} rows of prediction data") # fmt: skip

# Massively simplified filtering: set implausible values to None
cleaning_vars = [
    "Oxygen",
    "Carbon dioxide",
    "pH",
    "Lactate",
    "Bicarbonate",
    "Hemoglobin",
    "SpO2",
    "FiO2",
]

key_map = {"SpO2": "Peripheral oxygen saturation"}
lower = [PLAUSIBLE_VALUES[key_map.get(var, var)]["min"] for var in cleaning_vars] # fmt: skip
upper = [PLAUSIBLE_VALUES[key_map.get(var, var)]["max"] for var in cleaning_vars] # fmt: skip

PREDICTION_DATA = PREDICTION_DATA.with_columns(
    pl.when(pl.col(var).is_between(_lower, _upper))
    .then(pl.col(var))
    .otherwise(None)
    .alias(var)
    for var, _lower, _upper in zip(cleaning_vars, lower, upper)
)

# Filter for any oxygen values after plausible value filtering
PREDICTION_DATA = (
    PREDICTION_DATA.filter(pl.col("Oxygen").is_not_null()).collect().to_pandas()
)
print(f"Filtered for oxygen values - data has {len(PREDICTION_DATA)} rows")

PREDICTION_DATA.to_parquet("ABG_VBG_prediction_data.parquet")

# Encode system column: 1 for arterial, 0 for venous
PREDICTION_DATA["system"] = PREDICTION_DATA["system"].map(
    {"Blood arterial": 1, "Blood venous": 0}
)

# ------------------------------------------------------------------------------
# Train logistic regression model
# ------------------------------------------------------------------------------
print("Training logistic regression model for specimen type prediction...")

# Define features and target for the model
features = [
    "Oxygen",
    "Oxygen saturation",
    "Carbon dioxide",
    "pH",
    "Lactate",
    "Bicarbonate",
    "Hemoglobin",
    "SpO2",
    "FiO2",
]
target = "system"

X = PREDICTION_DATA[features]
y = PREDICTION_DATA[target]

# Drop rows where target is NaN
mask = y.notna()
X = X[mask]
y = y[mask]

# Impute missing values with median and add indicator columns for missingness
imputer = SimpleImputer(strategy="median", add_indicator=True)
X_imputed = imputer.fit_transform(X)

# Fit logistic regression model
model = LogisticRegression(random_state=42).fit(X_imputed, y)

# Initialize formula parts
formula_parts = []
constant = model.intercept_[0]
indicator_features = imputer.indicator_.features_
indicator_map = {
    feat_idx: idx for idx, feat_idx in enumerate(indicator_features)
}

# Build formula for each feature
for i, feature in enumerate(features):
    coeff = model.coef_[0][i]  # Coefficient for the feature
    imputed = imputer.statistics_[i]  # Median imputed value

    # Polars expression part: pl.coalesce(coeff * pl.col(feature), coeff * imputed + indicator_coeff)
    if i in indicator_map:
        indicator_idx = len(features) + indicator_map[i]
        indicator_coeff = model.coef_[0][indicator_idx]
        formula_parts.append(
            f"pl.coalesce({coeff:.5f} * pl.col('{feature}'), "
            f"{coeff:.5f} * {imputed:.5f} + {indicator_coeff:.5f},)"
        )
    else:
        formula_parts.append(f"{coeff:.5f} * pl.col('{feature}')")


# Construct and print the prediction formula
print()
print(f"inner_sum = {constant:.5f} + " + " + ".join(formula_parts))
print("specimen_prob = (1 / (1 + (-(inner_sum)).exp()))")
print("Model trained successfully.")
