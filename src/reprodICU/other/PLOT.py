# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This scripts visualizes the results of the reprodICU pipeline.

import argparse
import os
import sys
from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

import reprodICU

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
    "Tidal volume.expired": ["respiratory", "mL"],
    "Pressure.plateau Respiratory system airway --on ventilator": ["respiratory", "cmH2O"],
    "Pressure.max Respiratory system airway --on ventilator": ["respiratory", "cmH2O",],
    "Breath rate setting Ventilator": ["respiratory", "breaths per minute (/min)",],
    "Tidal volume setting Ventilator": ["respiratory", "mL"],
    "Oxygen/Total gas setting [Volume Fraction] Ventilator": ["respiratory", "percent (%)",],
    "Positive end expiratory pressure setting Ventilator": ["respiratory", "cmH2O",],
    "Lactate": ["labs", "mmol/L"],
    "Glucose": ["labs", "mg/dL"],
    "Magnesium": ["labs", "mmol/L"],
    "Sodium": ["labs", "mmol/L"],
    "Creatinine": ["labs", "mg/dL"],
    "Calcium": ["labs", "mmol/L"],
    "Chloride": ["labs", "mmol/L"],
    "Potassium": ["labs", "mmol/L"],
    "aPTT": ["labs", "seconds"],
    "Bilirubin": ["labs", "mg/dL"],
    "Alanine aminotransferase": ["labs", "U/L"],
    "Aspartate aminotransferase": ["labs", "U/L"],
    "Alkaline phosphatase": ["labs", "U/L"],
    "Albumin": ["labs", "g/L"],
    "Phosphate": ["labs", "mmol/L"],
    "Bicarbonate": ["labs", "mmol/L"],
    "Urea nitrogen": ["labs", "mg/dL"],
    "pH": ["labs", "pH"],
    "Oxygen": ["labs", "mmHg"],
    "Carbon dioxide": ["labs", "mmHg"],
    "Hemoglobin": ["labs", "g/dL"],
    "Leukocytes": ["labs", "10^3/µL"],
    "Platelets": ["labs", "10^3/µL"],
    "Urine output": ["intakeoutput", "mL"],
    # "Ventilation mode Ventilator": "respiratory",
    "Glasgow coma score total": ["vitals", "points"],
    "Glasgow coma score eye opening": ["vitals", "points"],
    "Glasgow coma score motor": ["vitals", "points"],
    "Glasgow coma score verbal": ["vitals", "points"],
} # fmt: skip

COLORS = {
    "AmsterdamUMCdb": "blue",
    "eICU-CRD": "orange",
    "HiRID": "red",
    "MIMIC-III": "green",
    "MIMIC-IV": "purple",
    "NWICU": "black",
    "SICdb": "gray",
}

DATASETS = list(COLORS)
LABELS = {"AmsterdamUMCdb": "AUMCdb"}  # shorter names for the plots

# Tol's muted qualitative color palett
# (https://cran.r-project.org/web/packages/khroma/vignettes/tol.html#muted)
COLORS_TOL = {
    "eICU-CRD": "#CC6677",
    "AmsterdamUMCdb": "#44AA99",
    "HiRID": "#332288",
    "MIMIC-III": "#117733",
    "MIMIC-IV": "#88CCEE",
    "NWICU": "#DDCC77",
    "SICdb": "#882255",
    # "#999933"
    # "#AA4499"
    # "#DDDDDD"
}


def _collect_data(
    table: str, variable: str, systems: str = None, data_map: dict = None
) -> pl.DataFrame:
    ####################################
    # COLLECT DATA
    ####################################

    # Load source datasets
    ID_TO_DB = reprodICU.patient_information.select(
        "Global ICU Stay ID", "Source Dataset"
    )

    # Load data from the provided data_map
    _data = data_map.get(table)
    if _data is None:
        raise ValueError(f"Table {table} not found in available data")

    # Load data, combine urine output
    URINE_COLS = [
        "Fluid output urine in and out urethral catheter",
        "Fluid output urine nephrostomy",
        "Urine output",
    ]
    value = (
        pl.sum_horizontal(URINE_COLS).alias(variable)
        if variable == "Urine output"
        else pl.col(variable)
    )
    data = (
        _data
        .join(ID_TO_DB, on="Global ICU Stay ID", how="left")
        .select("Global ICU Stay ID", "Source Dataset", value)
        .filter(pl.col(variable).is_not_null())
    )

    # filter source if specified
    if table == "labs":
        data = data.unnest(variable).rename({"value": variable})
        if systems:
            data = data.filter(
                pl.col("system").str.contains_any(
                    systems, ascii_case_insensitive=True
                )
                | pl.col("system").is_null()
            )
        data = data.drop("system", "method")

    # aggregate means for vitals
    if table == "vitals" or table == "respiratory":
        stay_value = (
            pl.col(variable).last()
            if variable.startswith("Glasgow coma score")
            else pl.col(variable).median()
        )
        data = data.group_by("Global ICU Stay ID", "Source Dataset").agg(
            stay_value
        )

    # drop outliers (1th percentile > values > 99th percentile)
    # and aggregate data
    return (
        data.drop("Global ICU Stay ID")
        .filter(
            pl.col(variable).is_not_null()
            & pl.col(variable).gt(pl.col(variable).quantile(0.01))
            & pl.col(variable).lt(pl.col(variable).quantile(0.99))
        )
        .collect()
    )


NOISY_VARIABLES = ("pressure", "urine output", "temperature")


def _kde_bandwidth(data: pl.DataFrame, variable: str) -> float:
    """One bandwidth for all datasets, so they are smoothed alike: Scott's rule
    on the pooled values, but at least two steps of the coarsest dataset's
    resolution, so that whole numbers and decimals look alike. Noisy variables
    are smoothed twice as much, the ordinal Glasgow coma scores not at all."""
    values = data[variable]
    bw = values.std() * len(values) ** (-1 / 5)
    if variable.startswith("Glasgow coma score"):
        return bw
    resolution = (
        data.group_by("Source Dataset")
        .agg(pl.col(variable).unique().sort().diff().median())[variable]
        .max()
    )
    noisy = any(name in variable.lower() for name in NOISY_VARIABLES)
    return (2 if noisy else 1) * max(bw, 2 * resolution)


RIDGE_OVERLAP = 0.25  # share of its height that a ridge may reach into the next


def _kde_curve(ax) -> tuple:
    """x and density of the filled kde on the axis, zero if it is empty."""
    if not ax.collections:
        return np.zeros(1), np.zeros(1)
    xy = ax.collections[0].get_paths()[0].vertices
    x, i = np.unique(xy[:, 0], return_inverse=True)
    y = np.zeros(len(x))
    np.maximum.at(y, i, xy[:, 1])
    return x, y


def _tail_limit(values: pl.Series) -> float | None:
    """97.5th percentile where a long right tail should be cut off, else None."""
    q = values.quantile(0.975)
    return q if values.max() > 2 * q else None


def _systems(variable: str) -> list:
    """Lab systems to select for the variable."""
    if variable in ["Oxygen saturation", "Lactate"]:
        return ["Blood arterial", "Blood"]
    return ["Blood", "Plasma"]


def _kde(x: pd.Series, bw: float, **kwargs) -> None:
    """kdeplot with an absolute bandwidth instead of one scaled per group."""
    sns.kdeplot(x=x, bw_method=bw / x.std(), **kwargs)


def _BLENDED_PLOT(data_map: dict) -> None:
    NCOLS = 8

    fig, axs_ = plt.subplots(
        ncols=NCOLS,
        nrows=len(BLENDEDICU_PLOT_VARIABLES.keys()) // NCOLS + 1,
        figsize=(25, 15),
    )
    axs = axs_.flatten()

    handles = [
        Patch(color=c, label=LABELS.get(label, label), alpha=0.5)
        for label, c in COLORS.items()
    ]

    axs[0].legend(handles=handles, loc="lower left", frameon=False)
    axs[0].axis("off")

    for i, (ax, VARIABLE) in enumerate(
        zip(axs[1:], BLENDEDICU_PLOT_VARIABLES.keys())
    ):
        sys.stdout.write("\033[K")  # Clear to the end of line
        print(f"plotted variable {i:2.0f}: {VARIABLE}")  # , end="\r")

        if not VARIABLE:
            ax.axis("off")
            continue

        TABLE = BLENDEDICU_PLOT_VARIABLES[VARIABLE][0]
        UNIT = BLENDEDICU_PLOT_VARIABLES[VARIABLE][1]

        # Load data
        data = _collect_data(
            table=TABLE,
            variable=VARIABLE,
            systems=_systems(VARIABLE),
            data_map=data_map,
        )

        # Plot
        bw = _kde_bandwidth(data, VARIABLE)
        for DATASET in data["Source Dataset"].unique().sort():
            values = data.filter(pl.col("Source Dataset") == DATASET)[VARIABLE]
            _kde(
                values.to_pandas(),
                bw=bw,
                ax=ax,
                fill=True,
                color=COLORS[DATASET],
            )
        ax.set_xlim(right=_tail_limit(data[VARIABLE]))
        ax.set_title("\n".join(wrap(VARIABLE, 28)), fontsize=13)
        ax.set_xlabel(f"{UNIT}", fontsize=10)
        if VARIABLE.startswith("Glasgow coma score"):
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    [ax.axis("off") for ax in axs[len(BLENDEDICU_PLOT_VARIABLES.keys()) + 1 :]]
    plt.tight_layout()
    plt.savefig("plots/blendedICU_plot.png", dpi=300)


def _plot_ridgeline(
    variable: str,
    table: str,
    data_map: dict,
    systems: list = None,
    all_vars: bool = False,
) -> None:
    ####################################
    # COLLECT DATA
    ####################################
    data = _collect_data(
        table=table,
        variable=variable,
        systems=systems,
        data_map=data_map,
    )

    ####################################
    # PLOT
    ####################################

    title = (
        variable
        .replace("/", "")
        .replace(" Respiratory system airway", "")
        .replace(" --on ventilator", "")
    )

    SORT = DATASETS  # rows of datasets without data stay, faded

    plot_data = (
        data.rename({variable: title})
        .to_pandas()
        .assign(**{"Source Dataset": lambda df: pd.Categorical(
            df["Source Dataset"], categories=SORT, ordered=True
        )})
        .sort_values("Source Dataset")
    )

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    palette = {ds: COLORS[ds] for ds in SORT}

    g = sns.FacetGrid(
        plot_data,
        row="Source Dataset",
        hue="Source Dataset",
        hue_order=SORT,
        row_order=SORT,
        palette=palette,
        aspect=7,
        height=1,
    )

    bw = _kde_bandwidth(data, variable)
    tail = _tail_limit(data[variable])
    g.map(
        _kde,
        title,
        bw=bw,
        clip=(None, tail),
        clip_on=False,
        fill=True,
        alpha=1,
        linewidth=1.5,
    )
    # white outline only where a ridge reaches into the one above it
    # (a bit of tolerance, so that ridges that just touch count too)
    ymax = g.axes.flat[0].get_ylim()[1]
    for ax, above in zip(g.axes.flat[1:], g.axes.flat):
        x, y = _kde_curve(ax)
        y_above = np.interp(x, *_kde_curve(above), left=0, right=0)
        shift = (1 - RIDGE_OVERLAP) * ymax
        if ((y > shift - 0.1 * ymax) & (y - shift < y_above)).any():
            ax.plot(x, y, color="w", linewidth=2, clip_on=False)
    g.refline(y=0, linewidth=1.5, linestyle="-", color=None, clip_on=False)
    # mark zero only if the data crosses it, it would stretch the axis otherwise
    if data[variable].min() < 0 < data[variable].max():
        g.refline(x=0, linewidth=1.5, linestyle="-", color="k")

    present = data["Source Dataset"].unique().to_list()
    for ax, dataset in zip(g.axes.flat, SORT):
        ax.text(
            -0.02,
            0.2,
            LABELS.get(dataset, dataset),
            fontweight="bold",
            color=COLORS[dataset],
            alpha=1 if dataset in present else 0.7,
            ha="right",
            va="center",
            transform=ax.transAxes,
        )

    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    g.figure.subplots_adjust(hspace=-RIDGE_OVERLAP)
    g.figure.suptitle(title, fontweight="bold")

    # Save the plot
    plot_path = "plots/all_vars/" if all_vars else "plots/"
    os.makedirs(plot_path, exist_ok=True)
    plt.savefig(plot_path + f"{title}.png", dpi=300, bbox_inches="tight")
    plt.close()


# region main
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--table",
        type=str,
        default="lab",
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
        "--systems",
        type=str,
        nargs="*",
        help="The variable systems to select (only for table lab).",
    )
    parser.add_argument(
        "--blendedicu",
        action="store_true",
        help="Reproduce the plot from the BlendedICU paper.",
    )
    parser.add_argument(
        "--allvars",
        action="store_true",
        help="Plot all variables from the BlendedICU paper as ridgeline plots.",
    )
    parser.add_argument(
        "--tol-colors",
        action="store_true",
        help="Use Tol's muted qualitative color palette.",
    )
    args = parser.parse_args()

    # Initialize data map from reprodICU package
    data_map = {
        "vitals": reprodICU.timeseries_vitals,
        "labs": reprodICU.timeseries_labs,
        "respiratory": reprodICU.timeseries_respiratory,
        "intakeoutput": reprodICU.timeseries_intakeoutput,
    }

    # Select color palette
    if args.tol_colors:
        COLORS = COLORS_TOL

    # Reproduce the BlendedICU plot if specified
    if args.blendedicu:
        _BLENDED_PLOT(data_map)
        exit()  # stop execution of the rest of the script
    # Visualize
    if args.allvars:
        for VARIABLE in BLENDEDICU_PLOT_VARIABLES.keys():
            if not VARIABLE:
                continue
            print(f"plotting variable: {VARIABLE}")
            _plot_ridgeline(
                variable=VARIABLE,
                table=BLENDEDICU_PLOT_VARIABLES[VARIABLE][0],
                data_map=data_map,
                systems=_systems(VARIABLE),
                all_vars=True,
            )
    else:
        _plot_ridgeline(
            variable=args.variable,
            table=args.table,
            data_map=data_map,
            systems=args.systems,
        )
