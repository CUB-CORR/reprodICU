# author: Finn Fassbender
# version: 05.10.2024

# Visualize the vital signs of a patient in a line plot

import argparse
import os

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from matplotlib.ticker import FuncFormatter, MultipleLocator
from mpl_toolkits import axisartist
from mpl_toolkits.axes_grid1 import host_subplot

import reprodICU


def plot_line_without_nan(x, y, data, label, color, marker, ax=None):
    """
    Plot a line without NaN values, i.e., remove NaN values from the data before plotting
    Vital signs this is used for are Temperature, heart rate, SpO2, and intracranial pressure

    :param x: x-axis column
    :param y: y-axis column
    :param data: DataFrame with the data
    :param label: Label for the line
    :param color: Color of the line
    :param marker: Marker of the line
    """
    x = data.filter(pl.col(y).is_not_null()).select(x).to_numpy().flatten()
    y = data.filter(pl.col(y).is_not_null()).select(y).to_numpy().flatten()
    return sns.lineplot(
        x=x,
        y=y,
        label=label,
        color=color,
        marker=marker,
        markersize=8,
        markeredgewidth=0.2,
        markeredgecolor="w",
        ax=ax,
    )


def plot_bp_without_nan(
    x,
    ys,
    data,
    label,
    color,
    markers=["^", "v", "d"],
    vlines: bool = False,
    ax=None,
):
    """
    Plot multiple lines without NaN values, i.e., remove NaN values from the data before plotting
    Vital signs this is used for is blood pressure, both invasive and non-invasive

    :param x: x-axis column
    :param ys: List of y-axis columns
    :param data: DataFrame with the data
    :param label: Label for the line
    :param color: Color of the line
    :param markers: List of markers for the lines
    :param vlines: Plot vertical lines instead
    """

    assert len(ys) == len(markers), "Number of ys and markers must be equal."
    assert (
        len(ys) == 3
    ), "Number of ys must be 3, for systolic, mean, and diastolic blood pressure."

    # filter data
    data = data.filter(pl.col(ys[0]).is_not_null())
    x = data.select(x).to_numpy().flatten()
    high = data.select(ys[0]).to_numpy().flatten()
    mean = data.select(ys[1]).to_numpy().flatten()
    low = data.select(ys[2]).to_numpy().flatten()

    # plot lines
    sns.lineplot(
        x=x,
        y=high,
        color=color,
        marker=markers[0],
        markersize=8,
        markeredgewidth=0.2,
        markeredgecolor="w",
        linestyle="--",
        lw=1,
        ax=ax,
    )
    sns.lineplot(
        x=x,
        y=mean,
        label=label,
        color=color,
        marker=markers[1],
        markersize=8,
        markeredgewidth=0.2,
        markeredgecolor="w",
        lw=1,
        ax=ax,
    )
    sns.lineplot(
        x=x,
        y=low,
        color=color,
        marker=markers[2],
        markersize=8,
        markeredgewidth=0.2,
        markeredgecolor="w",
        linestyle="--",
        lw=1,
        ax=ax,
    )

    # plot vertical lines
    if vlines:
        for i in range(len(x)):
            ax.vlines(
                x=x[i],
                ymin=low[i],
                ymax=high[i],
                color=color,
                lw=0.5,
                alpha=0.5,
            )


def twin_axis(host, label, ylim, key, loc, offset):
    """Y-axis sharing the x-axis of host, at the loc offset by offset points."""
    ax = host.twinx()
    ax.set_ylabel(label)
    if ylim:
        ax.set_ylim(ylim)
    if loc:
        ax.axis[key] = ax.new_fixed_axis(loc=loc, offset=(offset, 0))
    ax.axis[key].toggle(all=True)
    return ax


def pad_ylim(ax, low: float, high: float, share: float = 0.04) -> None:
    """Set the limits so that low and high lie a share of the height inside
    the axes, i.e. the markers are not cut off and low is at the same level
    on all axes."""
    pad = share * (high - low) / (1 - 2 * share)
    ax.set_ylim(low - pad, high + pad)


# region visualize
def visualize_vitals(
    vitals: pl.LazyFrame,
    global_icu_stay_id: str,
    save: bool = False,
    save_path: str = None,
) -> None:
    """
    Visualize the vital signs of a patient in a line plot

    :param vitals: DataFrame with vital signs
    :param global_icu_stay_id: ICU stay ID of the patient
    :param save: Save the plot to a file
    :param save_path: Path to save the plot to

    :return: None
    """
    # filter vitals for patient_id
    id_vitals = (
        vitals.filter(pl.col("Global ICU Stay ID") == global_icu_stay_id)
        .filter(pl.col("Time Relative to Admission (seconds)").is_between(0, 86400))
        .select(
            "Time Relative to Admission (seconds)",
            "Temperature",
            "Heart rate",
            "Peripheral oxygen saturation",
            "Respiratory rate",
            "Invasive systolic arterial pressure",
            "Invasive diastolic arterial pressure",
            "Invasive mean arterial pressure",
            "Non-invasive systolic arterial pressure",
            "Non-invasive diastolic arterial pressure",
            "Non-invasive mean arterial pressure",
            "Central venous pressure",
            "Intracranial pressure",
            "Glasgow coma score total",
            # "Glasgow coma score eye opening",
            # "Glasgow coma score verbal",
            # "Glasgow coma score motor",
        )
        .collect()
    )

    # create line plot
    sns.set_style("whitegrid")
    ax_hf_bp = host_subplot(111, axes_class=axisartist.Axes)
    ax_hf_bp.set_ylabel("Heart rate in bpm / Blood pressure in mmHg")

    # the y-axes of the other vitals, each offset so that they do not overlap
    ax_icp_cvp, ax_spo2, ax_temp, ax_rr, ax_gcs = (
        twin_axis(ax_hf_bp, *args)
        for args in (
            # label                                                     ylim      key      loc      offset
            ("Intracranial pressure / Central venous pressure in mmHg", None,     "left",  "left",   -50),
            ("SpO2 in %",                                               (0, 100), "right", None,       0),
            ("Temperature in °C",                                       (31, 45), "right", "right",   50),
            ("Respiratory rate in breaths/min",                         (0, 60),  "right", "left",  -100),
            ("GCS total",                                               (3, 15),  "right", "right",  100),
        )
    )  # fmt: skip

    # plot each vital sign
    # non-invasive blood pressure
    plot_bp_without_nan(
        x="Time Relative to Admission (seconds)",
        ys=[
            "Non-invasive systolic arterial pressure",
            "Non-invasive mean arterial pressure",
            "Non-invasive diastolic arterial pressure",
        ],
        data=id_vitals,
        label="Non-invasive blood pressure",
        color="purple",
        markers=["v", "d", "^"],
        vlines=True,
        ax=ax_hf_bp,
    )

    # invasive blood pressure
    plot_bp_without_nan(
        x="Time Relative to Admission (seconds)",
        ys=[
            "Invasive systolic arterial pressure",
            "Invasive mean arterial pressure",
            "Invasive diastolic arterial pressure",
        ],
        data=id_vitals,
        label="Invasive blood pressure",
        color="darkred",
        markers=["v", "d", "^"],
        vlines=True,
        ax=ax_hf_bp,
    )

    # Temperature
    plot_line_without_nan(
        x="Time Relative to Admission (seconds)",
        y="Temperature",
        data=id_vitals,
        label="Temperature",
        color="green",
        marker="o",
        ax=ax_temp,
    )

    # heart rate
    plot_line_without_nan(
        x="Time Relative to Admission (seconds)",
        y="Heart rate",
        data=id_vitals,
        label="Heart rate",
        color="red",
        marker="s",
        ax=ax_hf_bp,
    )

    # spO2
    plot_line_without_nan(
        x="Time Relative to Admission (seconds)",
        y="Peripheral oxygen saturation",
        data=id_vitals,
        label="SpO2",
        color="blue",
        marker="o",
        ax=ax_spo2,
    )

    # central venous pressure
    plot_line_without_nan(
        x="Time Relative to Admission (seconds)",
        y="Central venous pressure",
        data=id_vitals,
        label="Central venous pressure",
        color="darkblue",
        marker="D",
        ax=ax_icp_cvp,
    )

    # intracranial pressure
    plot_line_without_nan(
        x="Time Relative to Admission (seconds)",
        y="Intracranial pressure",
        data=id_vitals,
        label="Intracranial pressure",
        color="grey",
        marker="p",
        ax=ax_icp_cvp,
    )

    # Respiratory rate
    plot_line_without_nan(
        x="Time Relative to Admission (seconds)",
        y="Respiratory rate",
        data=id_vitals,
        label="Respiratory rate",
        color="orange",
        marker="^",
        ax=ax_rr,
    )

    # Glasgow Coma Scale total
    plot_line_without_nan(
        x="Time Relative to Admission (seconds)",
        y="Glasgow coma score total",
        data=id_vitals,
        label="GCS total",
        color="brown",
        marker="P",
        ax=ax_gcs,
    )

    # the same level for 0 (respiratory rate, heart rate, blood pressure, SpO2)
    # and for 3 (GCS)
    pad_ylim(ax_hf_bp, 0, ax_hf_bp.dataLim.y1)
    for ax in (ax_spo2, ax_temp, ax_rr, ax_gcs):
        pad_ylim(ax, *ax.get_ylim())
    ax_hf_bp.margins(x=0.04)  # as much as vertically

    plt.gcf().set_size_inches(20, 8)
    plt.title(f"Vital signs of patient {global_icu_stay_id}")
    plt.xlabel("Time since admission in hours")
    ax_hf_bp.xaxis.set_major_locator(MultipleLocator(4 * 3600))
    ax_hf_bp.xaxis.set_major_formatter(FuncFormatter(lambda s, _: f"{s / 3600:g}"))
    # plt.tight_layout()

    # Combine legends without duplicates
    legend_dict = {}
    for ax in [ax_hf_bp, ax_icp_cvp, ax_spo2, ax_temp, ax_rr, ax_gcs]:
        handles, labs = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labs):
            if label not in legend_dict:
                legend_dict[label] = handle
        ax.legend().remove()  # Remove individual legends

    lines = list(legend_dict.values())
    labels = list(legend_dict.keys())

    ax_hf_bp.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        fancybox=True,
        shadow=True,
        ncol=5,
    )

    # axisartist draws ticks inwards by default; colour each axis like its line
    for ax, side, color in (
        (ax_hf_bp, "left", "red"),
        (ax_spo2, "right", "blue"),
        (ax_temp, "right", "green"),
        (ax_rr, "right", "orange"),
        (ax_gcs, "right", "brown"),
    ):
        axis = ax.axis[side]
        axis.major_ticks.set_color(color)
        for artist in (axis.label, axis.major_ticklabels, axis.line):
            artist.set_color(color)
    for ax in [ax_hf_bp, ax_icp_cvp, ax_spo2, ax_temp, ax_rr, ax_gcs]:
        for axis in ax.axis.values():
            axis.major_ticks.set_tick_out(True)
            axis.minor_ticks.set_tick_out(True)

    # save plot
    if save:
        save_path = save_path or f"plots/visualize/{global_icu_stay_id}.png"
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


# endregion

# region main
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Visualize the vital signs of a patient."
    )
    parser.add_argument(
        "icu_stay_id",
        type=str,
        help="ICU stay ID of the patient.",
    )
    parser.add_argument(
        "-s",
        "--save",
        type=str,
        nargs="?",
        const=True,
        help="Save the plot to a file (optionally specify a file path).",
    )
    args = parser.parse_args()

    # Load vital signs from reprodICU package
    vitals = reprodICU.timeseries_vitals

    # Determine save parameters
    save = args.save is not None
    save_path = args.save if isinstance(args.save, str) else None

    # Visualize the vital signs
    visualize_vitals(vitals, args.icu_stay_id, save, save_path)
