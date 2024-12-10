# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This scripts visualizes the results of the reprodICU pipeline.

import argparse
import os

import altair as alt
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import polars as pl
import yaml
import seaborn as sns

BLENDEDICU_PLOT_VARIABLES = {
    "Heart rate": "vitals",
    "Invasive systolic arterial pressure": "vitals",
    "Invasive diastolic arterial pressure": "vitals",
    "Invasive mean arterial pressure": "vitals",
    "Non-invasive systolic arterial pressure": "vitals",
    "Non-invasive diastolic arterial pressure": "vitals",
    "Non-invasive mean arterial pressure": "vitals",
    "Peripheral oxygen saturation": "vitals",
    "Oxygen saturation": "labs",
    "Temperature": "vitals",
    "Respiratory rate": "vitals",
    "": "",  # "expiratory_tidal_volume",
    "": "",  # "Pressure.plateau Respiratory system airway --on ventilator": "respiratory",
    "Pressure.max Respiratory system airway --on ventilator": "respiratory",
    "Breath rate mechanical --on ventilator": "respiratory",
    "Tidal volume Ventilator --on ventilator": "respiratory",
    "Oxygen/Total gas setting [Volume Fraction] Ventilator": "respiratory",
    "PEEP Respiratory system --on ventilator": "respiratory",
    "Lactate [Moles/volume]": "labs",
    "Glucose [Mass/volume]": "labs",
    "Magnesium [Moles/volume]": "labs",
    "Sodium [Moles/volume]": "labs",
    "Creatinine [Mass/volume]": "labs",
    "Calcium [Moles/volume]": "labs",
    "Chloride [Moles/volume]": "labs",
    "Potassium [Moles/volume]": "labs",
    "aPTT": "labs",
    "Bilirubin.total [Moles/volume]": "labs",
    "Alanine aminotransferase [Enzymatic activity/volume]": "labs",
    "Aspartate aminotransferase [Enzymatic activity/volume]": "labs",
    "Alkaline phosphatase [Enzymatic activity/volume]": "labs",
    "Albumin [Mass/volume]": "labs",
    "Phosphate [Moles/volume]": "labs",
    "Bicarbonate [Moles/volume]": "labs",
    "Urea nitrogen [Mass/volume]": "labs",
    "pH": "labs",
    "Oxygen [Partial pressure]": "labs",
    "Carbon dioxide [Partial pressure]": "labs",
    "Hemoglobin [Mass/volume]": "labs",
    "Leukocytes [#/volume]": "labs",
    "Platelets [#/volume]": "labs",
    "Fluid output urine in and out urethral catheter": "intakeoutput",
    # "Ventilation mode Ventilator": "respiratory",
    "Glasgow Coma Score total": "vitals",
    "Glasgow Coma Score eye opening": "vitals",
    "Glasgow Coma Score motor": "vitals",
    "Glasgow Coma Score verbal": "vitals",
}


def load_mapping(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class reprodICUPaths:
    def __init__(self) -> None:
        config = load_mapping("configs/paths_local.yaml")
        for key, value in config.items():
            setattr(self, key, str(value))


def blended_plot():
    NCOLS = 5
    COLORS = {
        "eICU-CRD": "orange",
        "HiRID": "red",
        "MIMIC-III": "green",
        "MIMIC-IV": "purple",
        "SICdb": "gray",
        "AmsterdamUMCdb": "blue",
    }
    ID_TO_DB = pl.scan_parquet(
        "../reprodICU_files_/patient_information.parquet"
    ).select("Global ICU Stay ID", "Source Dataset")

    fig, axs_ = plt.subplots(
        ncols=NCOLS,
        nrows=len(BLENDEDICU_PLOT_VARIABLES.keys()) // NCOLS + 1,
        figsize=(15, 25),
    )
    axs = axs_.flatten()

    handles = [
        Patch(color=c, label=label, alpha=0.5) for label, c in COLORS.items()
    ]

    axs[0].legend(handles=handles, loc="lower left", frameon=False)
    axs[0].axis("off")

    for i, (ax, variable) in enumerate(
        zip(axs[1:], BLENDEDICU_PLOT_VARIABLES.keys())
    ):
        print(f"plotted variable {i:2.0f}: {variable}", end="\r")

        if not variable:
            ax.axis("off")
            continue

        table = BLENDEDICU_PLOT_VARIABLES[variable]
        data = (
            pl.scan_parquet(f"../reprodICU_files_/timeseries_{table}.parquet")
            .join(ID_TO_DB, on="Global ICU Stay ID", how="left")
            .select("Global ICU Stay ID", "Source Dataset", variable)
        )

        # handle labs differently
        if table == "labs":
            data = (
                data.unnest(variable)
                .rename({"value": variable})
                .filter(
                    pl.col("source").str.contains_any(
                        ["blood", "Blood", "plasma", "Plasma"]
                    )
                )
                .drop("source", "method")
            )

        # aggregate medians for vitals
        if table == "vitals":
            if variable.startswith("Glasgow Coma Score"):
                data = data.group_by("Global ICU Stay ID", "Source Dataset").agg(
                    pl.col(variable).last().alias(variable)
                )
            else:
                data = data.group_by("Global ICU Stay ID", "Source Dataset").agg(
                    pl.col(variable).median().alias(variable)
                )

        # drop outliers (1th percentile > values > 99th percentile)
        data = (
            data.drop("Global ICU Stay ID")
            .filter(
                pl.col(variable).is_not_null()
                & pl.col(variable).gt(pl.col(variable).quantile(0.01))
                & pl.col(variable).lt(pl.col(variable).quantile(0.99))
            )
            .collect(streaming=True)
        )

        ax = sns.kdeplot(
            data=data,
            x=variable,
            hue="Source Dataset",
            ax=ax,
            fill=True,
            common_norm=False,
            palette=COLORS,
        )
        ax.set_title(variable, fontsize=13, wrap=True)
        ax.set_xlabel("")
        ax.get_legend().remove()

    [ax.axis("off") for ax in axs[len(BLENDEDICU_PLOT_VARIABLES.keys()) + 1 :]]
    plt.tight_layout()
    plt.savefig("plots/blendedICU_plot.png", dpi=300)

    pass


# region main
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--datasets",
        type=str,
        nargs="+",
        default=["all"],
        help="Datasets to visuzalize.",
    )
    parser.add_argument(
        "-t",
        "--table",
        type=str,
        nargs=1,
        default=["lab"],
        help="Table to select the variable to visualize from.",
    )
    parser.add_argument(
        "-v",
        "--variable",
        type=str,
        default="Base excess",
        help="The variable to visualize.",
    )
    parser.add_argument(
        "-s",
        "--sources",
        type=str,
        nargs="*",
        help="The variable sources to select (only for table lab).",
    )
    parser.add_argument(
        "--DEMO",
        action="store_true",
        help="Use the DEMO dataset.",
    )
    parser.add_argument(
        "--BLENDEDICU",
        action="store_true",
        help="Reproduce the plot from the BlendedICU paper.",
    )
    args = parser.parse_args()

    if args.BLENDEDICU:
        blended_plot()
        exit()  # stop execution of the rest of the script

    # Initialize paths
    paths = reprodICUPaths()
    PATH = (
        paths.reprodICU_files_path
        if not args.DEMO
        else paths.reprodICU_demo_files_path
    )

    # Initialize columns
    class Columns:
        pass

    cols = Columns()
    for key, value in load_mapping("configs/COLUMN_NAMES.yaml").items():
        setattr(cols, key, value)

    # Select datasets to visualize
    if "all" in args.datasets:
        datasets = ["eICU", "HiRID", "MIMIC3", "MIMIC4", "SICdb", "UMCdb"]
        if args.DEMO:
            datasets = ["eICU", "MIMIC3", "MIMIC4"]
    else:
        datasets = args.datasets

    # Select tables to visualize
    # tables = ["admissions", "diagnoses", "lab", "medications", "patients", "procedures", "vitals"]
    tables = ["lab", "vitals", "respiratory", "intakeoutput"]
    args.table = args.table[0]
    assert args.table in tables, f"Table not found. Available tables: {tables}"

    ####################################
    # COLLECT DATA
    ####################################
    paths_ = {
        "lab": PATH + "timeseries_labs.parquet",
        "vitals": PATH + "timeseries_vitals.parquet",
        "respiratory": PATH + "timeseries_respiratory.parquet",
        "intakeoutput": PATH + "timeseries_intakeoutput.parquet",
    }

    # Load source datasets
    IDs = pl.scan_parquet(PATH + "patient_information.parquet").select(
        cols.global_icu_stay_id_col, cols.dataset_col
    )

    # Load data
    data = (
        pl.scan_parquet(paths_[args.table])
        .join(IDs, on=cols.global_icu_stay_id_col, how="left")
        .select(
            cols.global_icu_stay_id_col,
            cols.dataset_col,
            cols.timeseries_time_col,
            args.variable,
        )
        .filter(pl.col(args.variable).is_not_null())
    )

    # Filter source if specified
    if args.sources is not None:
        data = (
            data.unnest(args.variable)
            .rename({"value": args.variable})
            .filter(pl.col("source").is_in(args.sources))
            .select(
                cols.global_icu_stay_id_col,
                cols.dataset_col,
                cols.timeseries_time_col,
                args.variable,
            )
        )

    # aggregate means for vitals
    if args.table == "vitals":
        data = data.group_by(
            cols.global_icu_stay_id_col, cols.dataset_col
        ).agg(pl.col(args.variable).median().alias(args.variable))

    # aggregate data
    data = data.collect(streaming=True)

    ####################################
    # PLOT
    ####################################

    step = 20
    overlap = 1

    # Create a KDE ridgeline plot for each dataset
    chart = (
        alt.Chart(data, height=step)
        .transform_density(
            density=args.variable,
            groupby=[cols.dataset_col],
            # extent=[data[args.variable].min(), data[args.variable].max()],
            extent=[40, 140],
        )
        .mark_area(
            interpolate="monotone",
            fillOpacity=0.8,
            stroke="lightgray",
            strokeWidth=0.5,
        )
        .encode(
            alt.X("value:Q", title=args.variable),
            alt.Y("density:Q")  # stack="zero")
            .axis(None)
            .scale(range=[step, -step * overlap]),
            alt.Color(f"{cols.dataset_col}:N", legend=None),
        )
        .facet(
            row=alt.Row(f"{cols.dataset_col}:N")
            .title(None)
            .header(labelAngle=0, labelAlign="left")
        )
        .properties(
            title=f"Distribution of {args.variable} by Database", bounds="flush"
        )
        .configure_facet(spacing=0)
        .configure_view(stroke=None)
        .configure_title(anchor="end")
    )

    # Save the plot
    os.makedirs("plots", exist_ok=True)
    plot_path = f"plots/{args.variable}_distribution_by_database.png"
    chart.save(plot_path, ppi=300)
