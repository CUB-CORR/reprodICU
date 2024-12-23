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
from textwrap import wrap

BLENDEDICU_PLOT_VARIABLES = {
    "Heart rate": ["vitals", "beats per minute (/min)"],
    "Invasive systolic arterial pressure": ["vitals", "mmHg"],
    "Invasive diastolic arterial pressure": ["vitals", "mmHg"],
    "Invasive mean arterial pressure": ["vitals", "mmHg"],
    "Non-invasive systolic arterial pressure": ["vitals", "mmHg"],
    "Non-invasive diastolic arterial pressure": ["vitals", "mmHg"],
    "Non-invasive mean arterial pressure": ["vitals", "mmHg"],
    "Peripheral oxygen saturation": ["vitals", "percent (%)"],
    "Oxygen saturation": ["labs", "percent (%)"],
    "Temperature": ["vitals", "degrees Celsius (°C)"],
    "Respiratory rate": ["vitals", "breaths per minute (/min)"],
    "": "",  # "expiratory_tidal_volume",
    "Pressure.plateau Respiratory system airway --on ventilator": [
        "respiratory",
        "cmH2O",
    ],
    "Pressure.max Respiratory system airway --on ventilator": [
        "respiratory",
        "cmH2O",
    ],
    "Breath rate mechanical --on ventilator": [
        "respiratory",
        "breaths per minute (/min)",
    ],
    "Tidal volume Ventilator --on ventilator": ["respiratory", "mL"],
    "Oxygen/Total gas setting [Volume Fraction] Ventilator": [
        "respiratory",
        "percent (%)",
    ],
    "PEEP Respiratory system --on ventilator": ["respiratory", "cmH2O"],
    "Lactate [Moles/volume]": ["labs", "mmol/L"],
    "Glucose [Mass/volume]": ["labs", "mg/dL"],
    "Magnesium [Moles/volume]": ["labs", "mmol/L"],
    "Sodium [Moles/volume]": ["labs", "mmol/L"],
    "Creatinine [Mass/volume]": ["labs", "mg/dL"],
    "Calcium [Moles/volume]": ["labs", "mmol/L"],
    "Chloride [Moles/volume]": ["labs", "mmol/L"],
    "Potassium [Moles/volume]": ["labs", "mmol/L"],
    "aPTT": ["labs", "seconds"],
    "Bilirubin.total [Moles/volume]": ["labs", "µmol/L"],
    "Alanine aminotransferase [Enzymatic activity/volume]": ["labs", "U/L"],
    "Aspartate aminotransferase [Enzymatic activity/volume]": ["labs", "U/L"],
    "Alkaline phosphatase [Enzymatic activity/volume]": ["labs", "U/L"],
    "Albumin [Mass/volume]": ["labs", "g/L"],
    "Phosphate [Moles/volume]": ["labs", "mmol/L"],
    "Bicarbonate [Moles/volume]": ["labs", "mmol/L"],
    "Urea nitrogen [Mass/volume]": ["labs", "mg/dL"],
    "pH": ["labs", "pH"],
    "Oxygen [Partial pressure]": ["labs", "mmHg"],
    "Carbon dioxide [Partial pressure]": ["labs", "mmHg"],
    "Hemoglobin [Mass/volume]": ["labs", "g/dL"],
    "Leukocytes [#/volume]": ["labs", "10^3/µL"],
    "Platelets [#/volume]": ["labs", "10^3/µL"],
    "Fluid output urine in and out urethral catheter": ["intakeoutput", "mL"],
    # "Ventilation mode Ventilator": "respiratory",
    "Glasgow Coma Score total": ["vitals", "points"],
    "Glasgow Coma Score eye opening": ["vitals", "points"],
    "Glasgow Coma Score motor": ["vitals", "points"],
    "Glasgow Coma Score verbal": ["vitals", "points"],
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
        "NWICU": "black",
        "MIMIC-III": "green",
        "MIMIC-IV": "purple",
        "SICdb": "gray",
        "AmsterdamUMCdb": "blue",
    }
    ID_TO_DB = pl.scan_parquet(
        "../reprodICU_files/patient_information.parquet"
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

    for i, (ax, VARIABLE) in enumerate(
        zip(axs[1:], BLENDEDICU_PLOT_VARIABLES.keys())
    ):
        print(" " * 83, end="\r")  # clear line
        print(f"plotted variable {i:2.0f}: {VARIABLE}")  # , end="\r")

        if not VARIABLE:
            ax.axis("off")
            continue

        TABLE = BLENDEDICU_PLOT_VARIABLES[VARIABLE][0]
        UNIT = BLENDEDICU_PLOT_VARIABLES[VARIABLE][1]
        data = (
            pl.scan_parquet(f"../reprodICU_files/timeseries_{TABLE}.parquet")
            .join(ID_TO_DB, on="Global ICU Stay ID", how="left")
            .select("Global ICU Stay ID", "Source Dataset", VARIABLE)
        )

        # handle labs differently
        if TABLE == "labs":
            sources = (
                ["Blood", "Plasma"]
                if not VARIABLE
                in ["Oxygen saturation", "Lactate [Moles/volume]"]
                else ["Arterial blood"]
            )
            data = (
                data.unnest(VARIABLE)
                .rename({"value": VARIABLE})
                .filter(
                    pl.col("source").str.contains_any(
                        sources, ascii_case_insensitive=True
                    )
                )
                .drop("source", "method")
            )

        # aggregate medians for vitals
        if TABLE == "vitals":
            if VARIABLE.startswith("Glasgow Coma Score"):
                data = data.group_by(
                    "Global ICU Stay ID", "Source Dataset"
                ).agg(pl.col(VARIABLE).last().alias(VARIABLE))
            else:
                data = data.group_by(
                    "Global ICU Stay ID", "Source Dataset"
                ).agg(pl.col(VARIABLE).median().alias(VARIABLE))

        # drop outliers (1th percentile > values > 99th percentile)
        data = (
            data.drop("Global ICU Stay ID")
            .filter(
                pl.col(VARIABLE).is_not_null()
                & pl.col(VARIABLE).gt(pl.col(VARIABLE).quantile(0.01))
                & pl.col(VARIABLE).lt(pl.col(VARIABLE).quantile(0.99))
            )
            .collect(streaming=True)
        )

        ax = sns.kdeplot(
            data=data,
            x=VARIABLE,
            hue="Source Dataset",
            ax=ax,
            fill=True,
            common_norm=False,
            palette=COLORS,
            bw_adjust=2,
        )
        ax.set_title("\n".join(wrap(VARIABLE, 28)), fontsize=13)
        ax.set_xlabel(f"{UNIT}", fontsize=10)
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
        datasets = [
            "eICU",
            "HiRID",
            "MIMIC3",
            "MIMIC4",
            "NWICU",
            "SICdb",
            "UMCdb",
        ]
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
        data = data.group_by(cols.global_icu_stay_id_col, cols.dataset_col).agg(
            pl.col(args.variable).median().alias(args.variable)
        )

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
