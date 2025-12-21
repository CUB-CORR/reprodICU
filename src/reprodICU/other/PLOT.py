# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This scripts visualizes the results of the reprodICU pipeline.

import argparse
import os
import sys
from textwrap import wrap

import altair as alt
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from matplotlib.patches import Patch

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

    # Load data
    data = (
        _data
        .join(ID_TO_DB, on="Global ICU Stay ID", how="left")
        .select("Global ICU Stay ID", "Source Dataset", variable)
        .filter(pl.col(variable).is_not_null())
    )

    # combine urine output
    if variable == "Urine output":
        URINE_COLS = [
            "Fluid output urine in and out urethral catheter",
            "Fluid output urine nephrostomy",
            "Urine output",
        ]
        data = (
            _data
            .join(ID_TO_DB, on="Global ICU Stay ID", how="left")
            .select(
                "Global ICU Stay ID",
                "Source Dataset",
                pl.sum_horizontal(URINE_COLS).alias("Urine output"),
            )
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
        if variable.startswith("Glasgow coma score"):
            data = data.group_by(
                "Global ICU Stay ID", "Source Dataset"
            ).agg(pl.col(variable).last().alias(variable))
        else:
            data = data.group_by(
                "Global ICU Stay ID", "Source Dataset"
            ).agg(pl.col(variable).median().alias(variable))

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


def _BLENDED_PLOT(data_map: dict) -> None:
    NCOLS = 5

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
            systems=(
                ["Blood", "Plasma"]
                if not VARIABLE in ["Oxygen saturation", "Lactate"]
                else ["Blood arterial", "Blood"]
            ),
            data_map=data_map,
        )

        # Plot
        ax = sns.kdeplot(
            data=data,
            x=VARIABLE,
            hue="Source Dataset",
            ax=ax,
            fill=True,
            common_norm=False,
            palette=COLORS,
            # bw_adjust=2,
        )
        ax.set_title("\n".join(wrap(VARIABLE, 28)), fontsize=13)
        ax.set_xlabel(f"{UNIT}", fontsize=10)
        ax.get_legend().remove()

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

    step = 20
    overlap = 1
    # Sanitize variable name for use in Altair/Vega expressions
    # First remove known long phrases
    title = (
        variable.replace(" Respiratory system airway", "")
        .replace(" --on ventilator", "")
    )
    # Then replace special characters that break Vega field references
    # Must replace dots and plus signs before replacing other punctuation
    title = (
        title.replace(".", " ")
        .replace("+", " and ")
        .replace("/", " per ")
        .replace("[", "(")
        .replace("]", ")")
    )
    SORT = [
        "AmsterdamUMCdb",
        "eICU-CRD",
        "HiRID",
        "MIMIC-III",
        "MIMIC-IV",
        "NWICU",
        "SICdb",
    ]

    data = data.rename({variable: title})

    # Create a KDE ridgeline plot for each dataset
    chart = (
        alt.Chart(data, height=step)
        .transform_density(
            density=title,
            groupby=["Source Dataset"],
            extent=[data[title].min(), data[title].max()],
        )
        .mark_area(
            interpolate="monotone",
            fillOpacity=0.8,
            stroke="lightgray",
            strokeWidth=0.5,
        )
        .encode(
            alt.X("value:Q", title=title),
            alt.Y("density:Q")
            .sort(SORT)
            .axis(None)
            .scale(range=[step, -step * overlap]),
            alt.Color("Source Dataset:N", legend=None).scale(
                domain=SORT, range=[COLORS[dataset] for dataset in SORT]
            ),
        )
        .facet(
            row=alt.Row("Source Dataset:N")
            .title(None)
            .header(labelAngle=0, labelAlign="left")
        )
        .properties(title=title, bounds="flush")
        .configure_facet(spacing=0)
        .configure_view(stroke=None)
        .configure_title(anchor="end")
    )

    # Save the plot
    plot_path = "plots/" if not all_vars else "plots/all_vars/"
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    plot_path += f"{title}.png"
    chart.save(plot_path, ppi=300)


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
                systems=(
                    ["Blood", "Plasma"]
                    if not VARIABLE in ["Oxygen saturation", "Lactate"]
                    else ["Blood arterial", "Blood"]
                ),
                all_vars=True,
            )
    else:
        _plot_ridgeline(
            variable=args.variable,
            table=args.table,
            data_map=data_map,
            systems=args.systems,
        )
