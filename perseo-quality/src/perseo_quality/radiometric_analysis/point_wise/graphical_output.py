# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Functions to generate plots for Point-Wise Radiometric Analysis"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from perseo_quality.core.generic_dataclasses import SARProjection
from perseo_quality.radiometric_analysis.custom_dataclasses import (
    PointWiseRadiometricAnalysisOutput,
    RadiometricAnalysisAxes,
    RadiometricAnalysisDirection,
    RadiometricAnalysisValue,
)

# syncing with logger
log = logging.getLogger("quality_analysis")

COLOR_SET = ["#7E5920", "#FFA737", "#5F5449", "#5999D9"]


def radiometric_analysis_graphs(
    data: list[PointWiseRadiometricAnalysisOutput], out_dir: Path | str | None, interactive: bool = False
) -> None:
    """Generation of the Radiometric Analysis graphical output.

    Parameters
    ----------
    data : list[PointWiseRadiometricAnalysisOutput]
        list of PointWiseRadiometricAnalysisOutput results dataclass
    out_dir : str | Path | None
        output folder path, while interactive mode is on it is ignored so it can be passed as None
    interactive : bool, optional
        if set to True, this flag will show the generated plots and will not dump to disk the graphs, by default False

    Raises
    ------
    ValueError
        if output directory is None and interactive flag is False
    """

    if interactive and out_dir is not None:
        log.warning(f"Interactive mode is on, graphs won't be saved to {out_dir}")
        out_dir = None

    if not interactive and out_dir is None:
        raise ValueError("A valid output directory path must be provided when interactive mode is off")

    out_dir = Path(out_dir) if out_dir is not None else None

    log.info("Generating point-wise radiometric profiles plots...")
    # re-organizing input data
    times = list({p.time for p in data})

    for time_id, time in enumerate(times):
        log.info(f"Profile {time_id + 1}/{len(times)}")
        selected_data = [p for p in data if p.time == time]
        direction = list({p.direction for p in selected_data})[0]
        out_type = list({p.value_type for p in selected_data})[0]
        smoothed_profiles = np.ma.masked_invalid(np.concatenate([p.smoothed_profile_db for p in selected_data]))
        original_profiles = np.ma.masked_invalid(np.concatenate([p.original_profile_db for p in selected_data]))
        axes = np.concatenate([p.axis for p in selected_data])

        # initializing figure
        fig, axs = plt.subplots(figsize=(14, 10))
        fig.suptitle(f"Radiometric Profiles @ {time}", fontsize=16, fontweight="bold")
        axs.set_title(f"{direction.name.capitalize()} direction", fontsize=12)

        for num, channel in enumerate(selected_data):
            if direction == RadiometricAnalysisDirection.RANGE:
                x_label = "Slant Range time [s]"
                if channel.projection == SARProjection.GROUND_RANGE:
                    x_label = "Ground Range distance [m]"

                if channel.axis_type == RadiometricAnalysisAxes.INCIDENCE_ANGLE:
                    x_label = "Incidence Angle [deg]"
                elif channel.axis_type == RadiometricAnalysisAxes.LOOK_ANGLE:
                    x_label = "Look Angle [deg]"
            else:
                x_label = "Azimuth time (relative) [s]"

            if out_type == RadiometricAnalysisValue.AMPLITUDE:
                nought = " ".join([s.capitalize() for s in channel.radiometric_quantity.name.split("_")])
                y_label = f"{channel.value_type.name.capitalize()} {nought} [dB]"
            else:
                y_label = "Phase [rad]"

            _plotting_profiles(axis=axs, channel=channel, color=COLOR_SET[num], x_label=x_label, y_label=y_label)

        # customizing data for histogram 2D plot
        profile_mean = smoothed_profiles.mean()
        x_lim = [np.min(axes), np.max(axes)]
        y_lim = [smoothed_profiles.min(), smoothed_profiles.max()]
        y_lim[0] = y_lim[0] * 1.1 if y_lim[0] < 0 else y_lim[0] * 0.9
        y_lim[1] = y_lim[1] * 0.9 if y_lim[1] < 0 else y_lim[1] * 1.1

        # x_edges = np.linspace(x_lim[0], x_lim[1], 101)
        # y_edges = np.linspace(profile_mean-6, profile_mean+6, 51)
        y_values = original_profiles[~original_profiles.mask].data.copy()
        x_values = axes[~original_profiles.mask].copy()
        display_cond = np.logical_and(y_values > profile_mean - 6, y_values < profile_mean + 6)

        # plotting histogram
        axs.hist2d(x_values[display_cond], y_values[display_cond], (101, 51), cmin=1, cmap="bone", alpha=0.7, label="_")

        # setting axes limits
        axs.set_xlim(x_lim)
        axs.set_ylim(y_lim)

        axs.grid(visible=True, linestyle="--", alpha=0.6)

        plt.legend()

        brst_info = ("_burst" + str(selected_data[0].burst)) if selected_data[0].burst is not None else ""
        plot_name = (
            selected_data[0].swath + brst_info + "_radiometric_analysis_" + direction.name.lower() + "_" + str(time_id)
        )

        plt.tight_layout()
        if not interactive:
            plt.savefig(out_dir.joinpath(plot_name).with_suffix(".png"), dpi=200)
            plt.close(fig)

    if interactive:
        plt.show()


def _plotting_profiles(
    axis: plt.Axes, channel: PointWiseRadiometricAnalysisOutput, color: str, x_label: str, y_label: str
) -> None:
    """Function to plot radiometric analysis output profiles.

    Parameters
    ----------
    axis : plt.Axes
        axes where to plot the profiles
    channel : rdt.PointWiseRadiometricAnalysisOutput
        radiometric analysis output dataclass
    color : str
        color of the plot line
    x_label : str
        x axis label
    y_label : str
        y axis label
    """

    brst_label = (" Burst " + str(channel.burst)) if channel.burst is not None else ""
    label = channel.swath + brst_label + " " + channel.polarization.name

    axis.plot(channel.axis, channel.smoothed_profile_db, color=color, label=label)

    axis.set_xlabel(x_label, fontsize=12, fontweight="bold")
    axis.set_ylabel(y_label, fontsize=12, fontweight="bold")
