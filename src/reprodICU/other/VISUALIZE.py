# author: Finn Fassbender
# version: 05.10.2024

# Visualize the vital signs of a patient in a line plot

import argparse

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
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
        x=x, y=y, label=label, color=color, marker=marker, markersize=10, ax=ax
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
        markersize=10,
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
        markersize=10,
        lw=1,
        ax=ax,
    )
    sns.lineplot(
        x=x,
        y=low,
        color=color,
        marker=markers[2],
        markersize=10,
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
            )


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

    ax_icp_cvp = ax_hf_bp.twinx()
    ax_icp_cvp.set_ylabel(
        "Intracranial pressure / Central venous pressure in mmHg"
    )
    ax_icp_cvp.axis["left"] = ax_icp_cvp.new_fixed_axis(
        loc="left", offset=(-50, 0)
    )
    ax_icp_cvp.axis["left"].toggle(all=True)

    ax_spo2 = ax_hf_bp.twinx()
    ax_spo2.set_ylabel("SpO2 in %")
    ax_spo2.set_ylim(0, 100)
    ax_spo2.axis["right"].toggle(all=True)

    ax_temp = ax_hf_bp.twinx()
    ax_temp.set_ylabel("Temperature in °C")
    ax_temp.set_ylim(31, 45)
    ax_temp.axis["right"] = ax_temp.new_fixed_axis(loc="right", offset=(50, 0))
    ax_temp.axis["right"].toggle(all=True)

    # Respiratory rate axis (right, offset further to avoid overlap)
    ax_rr = ax_hf_bp.twinx()
    ax_rr.set_ylabel("Respiratory rate in breaths/min")
    ax_rr.set_ylim(0, 60)
    ax_rr.axis["right"] = ax_rr.new_fixed_axis(loc="left", offset=(-100, 0))
    ax_rr.axis["right"].toggle(all=True)

    # GCS total axis (right, offset further to avoid overlap)
    ax_gcs = ax_hf_bp.twinx()
    ax_gcs.set_ylabel("GCS total")
    ax_gcs.set_ylim(3, 15)
    ax_gcs.axis["right"] = ax_gcs.new_fixed_axis(loc="right", offset=(100, 0))
    ax_gcs.axis["right"].toggle(all=True)

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

    plt.gcf().set_size_inches(20, 8)
    plt.title(f"Vital signs of patient {global_icu_stay_id}")
    plt.xlabel("Time in seconds relative to admission")
    plt.xticks(rotation=45)
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
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True,
        shadow=True,
        ncol=5,
    )

    ax_hf_bp.axis["left"].label.set_color("red")
    ax_spo2.axis["right"].label.set_color("blue")
    ax_temp.axis["right"].label.set_color("green")
    ax_rr.axis["right"].label.set_color("orange")
    ax_gcs.axis["right"].label.set_color("brown")

    # save plot
    if save:
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.savefig(f"{global_icu_stay_id}.png")
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
    save_path = args.save if isinstance(args.save, str) else f"{args.icu_stay_id}.png"

    # Visualize the vital signs
    visualize_vitals(vitals, args.icu_stay_id, save, save_path)
