# Belletti, A., Lerose, C. C., Zangrillo, A., & Landoni, G. (2021).
# Vasoactive-inotropic score: Evolution, clinical utility, and pitfalls.
# Journal of Cardiothoracic and Vascular Anesthesia, 35(10), 3067–3077. doi:10.1053/j.jvca.2020.09.117
# ------------------------------------------------------------------------------
# Implements the Vasoactive-Inotropic Score (VIS)

import argparse
from pathlib import Path

import polars as pl
from numpy import all, any

SECONDS_IN_1H = 60 * 60


def FIX_WINDOW_BORDERS(
    DATA: pl.LazyFrame, TIMEWINDOW_IN_SECONDS: int = SECONDS_IN_1H
) -> pl.LazyFrame:
    return (
        DATA.with_columns(
            pl.int_ranges(
                # ceil the start
                (
                    pl.col("Drug Start Relative to T_0 (seconds)")
                    // TIMEWINDOW_IN_SECONDS
                )
                * TIMEWINDOW_IN_SECONDS,
                pl.col("Drug End Relative to T_0 (seconds)"),
                TIMEWINDOW_IN_SECONDS,
            ).alias("Drug Window Borders")
        )
        .with_columns(
            # concatenate the ranges of start times
            pl.concat_list(
                pl.col("Drug Start Relative to T_0 (seconds)"),
                pl.col("Drug Window Borders"),
            ).alias("Drug Window Borders Start"),
            # concatenate the ranges of end times
            pl.concat_list(
                pl.col("Drug Window Borders"),
                pl.col("Drug End Relative to T_0 (seconds)"),
            ).alias("Drug Window Borders End"),
        )
        .drop("Drug Window Borders")
        # explode the dataframe to long format by exploding the given columns
        .explode("Drug Window Borders Start", "Drug Window Borders End")
        .with_columns(
            # calculate the duration of each drug administration
            (
                pl.col("Drug Window Borders End")
                - pl.col("Drug Window Borders Start")
            ).alias("Drug Duration (seconds)"),
            (
                pl.col("Drug Window Borders End")
                - pl.col("Drug Window Borders Start")
            )
            .truediv(TIMEWINDOW_IN_SECONDS)
            .alias("Drug Duration (windows)"),
            # calculate the hour the drug administration started
            pl.col("Drug Window Borders Start")
            .floordiv(TIMEWINDOW_IN_SECONDS)
            .alias("Window Relative to T_0"),
        )
        # drop unnecessary columns
        .drop("Drug Window Borders Start", "Drug Window Borders End")
    )


# region VIS
def VIS(
    patient_information: pl.LazyFrame,
    medications: pl.LazyFrame = None,
    t_0: int | pl.LazyFrame = 0,
) -> pl.LazyFrame:
    """
    Vasoactive-Inotropic Score (VIS)

    Calculate the vasoactive-inotropic score based on the provided parameters for each ICU stay.

    :param patient_information: DataFrame with patient information
    :param medications: DataFrame with medications
    :param column_names: Optional dictionary to map default column names to custom names

    :return: DataFrame with vasoactive-inotropic score(s) for each ICU stay. Includes timestamp if `calculate_at_each_timestep` is True.
    """

    # Check if all required data is provided
    if any([patient_information is None, medications is None]):
        raise ValueError("All required data must be provided.")

    # Check that "Global ICU Stay ID" is in both dataframes
    if not all(
        [
            "Global ICU Stay ID" in patient_information.collect_schema().names(),
            "Global ICU Stay ID" in medications.collect_schema().names(),
        ] # fmt: skip
    ):
        raise ValueError("'Global ICU Stay ID' must be present in both dataframes.") # fmt: skip

    # Check that t_0 is either an integer or a dataframe with "Global ICU Stay ID" and "T_0 (seconds)"
    if not (
        isinstance(t_0, int)
        or (
            isinstance(t_0, pl.LazyFrame)
            and all(
                [
                    "Global ICU Stay ID" in t_0.collect_schema().names(),
                    "T_0 (seconds)" in t_0.collect_schema().names(),
                ]
            )
        )
    ):
        raise ValueError("'t_0' must be either an integer or a dataframe with 'Global ICU Stay ID' and 'T_0 (seconds)'.") # fmt: skip

    VASOPRESSORS_INOTROPES = [
        "angiotensin II",  # 0.25 * dose in ng/kg/min
        "dobutamine",  # dose in mcg/kg/min
        "dopamine",  # dose in mcg/kg/min
        "enoximone",  # dose in mcg/kg/min
        "epinephrine",  # 100 * dose in mcg/kg/min
        "levosimendan",  # 50 * dose in mcg/kg/min
        "methylene blue",  # 20 * dose in mg/kg/h
        "milrinone",  # 10 * dose in mcg/kg/min
        "norepinephrine",  # 100 * dose in mcg/kg/min
        "olprinone",  # 25 * dose in mcg/kg/min
        "phenylephrine",  # 100 * dose in mcg/kg/min
        "terlipressin",  # 10 * dose in mcg/h
        "vasopressin (USP)",  # 10000 * dose in units/kg/min
    ]

    ### CALCULATIONS ###
    # Select relevant columns
    t_0 = (
        patient_information.select(
            "Global ICU Stay ID", pl.lit(t_0).alias("T_0 (seconds)")
        )
        if isinstance(t_0, int)
        else t_0.select("Global ICU Stay ID", pl.col("T_0 (seconds)"))
    )
    weights = patient_information.select(
        "Global ICU Stay ID", "Admission Weight (kg)"
    )
    medications = (
        medications.filter(
            pl.col("Drug Ingredient").is_in(VASOPRESSORS_INOTROPES)
        )
        .join(t_0, on="Global ICU Stay ID", how="left")
        .with_columns(
            # Calculate start and end relative to T_0
            (
                pl.col("Drug Start Relative to Admission (seconds)")
                - pl.col("T_0 (seconds)")
            ).alias("Drug Start Relative to T_0 (seconds)"),
            (
                pl.col("Drug End Relative to Admission (seconds)")
                - pl.col("T_0 (seconds)")
            ).alias("Drug End Relative to T_0 (seconds)"),
        )
    )

    # Fix rates
    PREDICATES = (
        pl.col("Drug Rate").is_null(),
        pl.col("Drug Rate Unit").is_null(),
        pl.col("Drug Amount").is_not_null(),
        pl.col("Drug Amount Unit").is_in(
            ["g", "mg", "mcg", "U", "IE", "units"]
        ),
    )
    medications = medications.with_columns(
        pl.when(*PREDICATES)
        .then(
            pl.col("Drug Amount")
            / (
                pl.col("Drug End Relative to Admission (seconds)")
                - pl.col("Drug Start Relative to Admission (seconds)")
            ).truediv(60)
        )
        .otherwise(pl.col("Drug Rate"))
        .alias("Drug Rate"),
        pl.when(*PREDICATES)
        .then(pl.concat_str(pl.col("Drug Amount Unit"), pl.lit("/min")))
        .otherwise(pl.col("Drug Rate Unit"))
        .alias("Drug Rate Unit"),
    )

    # Fix units
    medications = (
        medications.join(weights, on="Global ICU Stay ID", how="left")
        .with_columns(
            # CONVERTING UNITS
            # Convert mcg / mg / g to mcg/kg/min
            pl.when(pl.col("Drug Rate Unit") == "mcg/min")
            .then(pl.col("Drug Rate") / pl.col("Admission Weight (kg)"))
            .when(pl.col("Drug Rate Unit") == "mcg/hr")
            .then(pl.col("Drug Rate") / pl.col("Admission Weight (kg)") / 60)
            .when(pl.col("Drug Rate Unit") == "mcg/kg/hr")
            .then(pl.col("Drug Rate") / 60)
            .when(pl.col("Drug Rate Unit") == "mg/hr")
            .then(
                pl.col("Drug Rate")
                * 1000
                / pl.col("Admission Weight (kg)")
                / 60
            )
            .when(pl.col("Drug Rate Unit") == "mg/min")
            .then(pl.col("Drug Rate") * 1000 / pl.col("Admission Weight (kg)"))
            .when(pl.col("Drug Rate Unit") == "mg/kg/min")
            .then(pl.col("Drug Rate") * 1000)
            .when(pl.col("Drug Rate Unit") == "g/hr")
            .then(
                pl.col("Drug Rate")
                * 1_000_000
                / pl.col("Admission Weight (kg)")
                / 60
            )
            .when(pl.col("Drug Rate Unit") == "g/min")
            .then(
                pl.col("Drug Rate")
                * 1_000_000
                / pl.col("Admission Weight (kg)")
            )
            .when(pl.col("Drug Rate Unit") == "g/kg/hr")
            .then(pl.col("Drug Rate") * 1_000_000 / 60)
            .when(pl.col("Drug Rate Unit") == "g/kg/min")
            .then(pl.col("Drug Rate") * 1_000_000)
            # Convert Units
            .when(pl.col("Drug Rate Unit").is_in(["U/hr", "units/hr"]))
            .then(pl.col("Drug Rate") / pl.col("Admission Weight (kg)") / 60)
            .when(
                pl.col("Drug Rate Unit").is_in(["U/min", "units/min", "IE/min"])
            )
            .then(pl.col("Drug Rate") / pl.col("Admission Weight (kg)"))
            # Keep unchanged
            .when(
                pl.col("Drug Rate Unit").is_in(
                    ["mcg/kg/min", "U/min", "units/min", "IE/min"]
                )
            )
            .then(pl.col("Drug Rate"))
            .otherwise(None)
            .alias("Drug Rate (fixed units)"),
            # RENAMING UNITS
            pl.when(
                pl.col("Drug Rate Unit").is_in(
                    [
                        "mcg/kg/min",
                        "mcg/min",
                        "mcg/hr",
                        "mcg/kg/hr",
                        "mg/hr",
                        "mg/min",
                        "mg/kg/min",
                        "g/hr",
                        "g/min",
                        "g/kg/hr",
                        "g/kg/min",
                    ]
                )
            ).then(pl.lit("mcg/kg/min"))
            # TODO: check this again (stupid me, forgot proper documentation)
            .when(
                pl.col("Drug Rate Unit").is_in(
                    ["U/min", "U/hr", "units/hr", "units/min", "IE/min"]
                )
            )
            .then(pl.lit("U/kg/min"))
            .otherwise(None)
            .alias("Drug Rate Unit (fixed units)"),
        )
        .drop_nulls(["Drug Rate (fixed units)", "Drug Rate Unit (fixed units)"])
    )

    # Convert to VIS components
    medications = medications.with_columns(
        pl.when(
            pl.col("Drug Ingredient") == "angiotensin II",
            pl.col("Drug Rate Unit (fixed units)") == "mcg/kg/min",
        )
        .then(pl.col("Drug Rate (fixed units)") * 0.25 * 1000) # mcg -> ng
        .when(
            pl.col("Drug Ingredient").is_in(["dopamine", "dobutamine", "enoximone"]),
            pl.col("Drug Rate Unit (fixed units)") == "mcg/kg/min",
        )
        .then(pl.col("Drug Rate (fixed units)"))
        .when(
            pl.col("Drug Ingredient").is_in(["milrinone", "phenylephrine"]),
            pl.col("Drug Rate Unit (fixed units)") == "mcg/kg/min",
        )
        .then(pl.col("Drug Rate (fixed units)") * 10)
        .when(
            pl.col("Drug Ingredient") == "terlipressin",
            pl.col("Drug Rate Unit (fixed units)") == "mcg/kg/min",
        )
        .then(pl.col("Drug Rate (fixed units)") * pl.col("Admission Weight (kg)"))
        .when(
            pl.col("Drug Ingredient") == "methylene blue",
            pl.col("Drug Rate Unit (fixed units)") == "mg/kg/min",
        )
        .then(pl.col("Drug Rate (fixed units)") * 20 * 60) # min -> h
        .when(
            pl.col("Drug Ingredient") == "olprinone",
            pl.col("Drug Rate Unit (fixed units)") == "mcg/kg/min",
        )
        .then(pl.col("Drug Rate (fixed units)") * 25)
                .when(
            pl.col("Drug Ingredient") == "levosimendan",
            pl.col("Drug Rate Unit (fixed units)") == "mcg/kg/min",
        )
        .then(pl.col("Drug Rate (fixed units)") * 50)
        .when(
            pl.col("Drug Ingredient").is_in(["epinephrine", "norepinephrine"]),
            pl.col("Drug Rate Unit (fixed units)") == "mcg/kg/min",
        )
        .then(pl.col("Drug Rate (fixed units)") * 100)
        .when(
            pl.col("Drug Ingredient") == "vasopressin (USP)",
            pl.col("Drug Rate Unit (fixed units)") == "U/kg/min",
        )
        .then(pl.col("Drug Rate (fixed units)") * 10_000)
        .otherwise(None)
        .alias("VIS Component"),
    ) # fmt: skip

    # Fix window borders
    medications = FIX_WINDOW_BORDERS(
        medications, TIMEWINDOW_IN_SECONDS=SECONDS_IN_1H
    ).rename(
        {
            "Window Relative to T_0": "Hour Relative to T_0",
            "Drug Duration (windows)": "Drug Duration (hours)",
        }
    )

    # Calculate VIS per hour
    vis = (
        medications.group_by(
            "Global ICU Stay ID", "Hour Relative to T_0", "Drug Ingredient"
        )
        .agg(
            pl.col("VIS Component")
            .mul(pl.col("Drug Duration (hours)"))
            .sum()
            .truediv(pl.sum("Drug Duration (hours)"))
            .alias("VIS Component"),
            pl.col("T_0 (seconds)").first(),
        )
        .group_by("Global ICU Stay ID", "Hour Relative to T_0")
        .agg(
            pl.sum("VIS Component").alias("Vasoactive-Inotropic Score (VIS)"),
            pl.col("T_0 (seconds)").first(),
        )
        .sort("Global ICU Stay ID", "Hour Relative to T_0")
        .select(
            "Global ICU Stay ID",
            "T_0 (seconds)",
            "Hour Relative to T_0",
            "Vasoactive-Inotropic Score (VIS)",
        )
    )

    return vis


# endregion

# region main
if __name__ == "__main__":
    path = "../../reprodICU_files/"

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-p",
        "--patient-information-path",
        type=str,
        help="Path to the patient information file.",
        default=Path(__file__).parent.joinpath(
            path + "patient_information.parquet"
        ),
    )
    argparser.add_argument(
        "-m",
        "--medications-path",
        type=str,
        help="Path to the medications file.",
        default=Path(__file__).parent.joinpath(path + "medications.parquet"),
    )
    argparser.add_argument(
        "-o",
        "--output-path",
        type=str,
        default=None,
        help="Path to the output file. If not specified, defaults based on aggregation type.",
    )

    args = argparser.parse_args()

    # Determine output path
    output_path = args.output_path
    if output_path is None:
        output_dir = Path(__file__).parent.joinpath(
            path + "PRECALCULATED_CONCEPTS/SCORES/"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = "VIS_long_format.parquet"
        output_path = output_dir.joinpath(output_filename)
    else:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data (using scan for lazy execution initially)
    patient_information = pl.scan_parquet(args.patient_information_path)
    medications = pl.scan_parquet(args.medications_path)

    # Print approximate runtime
    # Numbers hardcoded from empirical testing
    count = patient_information.select(pl.len()).collect().to_numpy()[0][0]
    print(f"Approximate runtime: {count / 439123 * 3:3.0f} minutes")

    # Calculate vasoactive-inotropic score based on CLI arguments
    print("Calculating vasoactive-inotropic score...") # fmt: skip
    VIS(
        patient_information=patient_information,
        medications=medications,
    ).sink_parquet(output_path)
    print("VIS calculations complete.")


# endregion
