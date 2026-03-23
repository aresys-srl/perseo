# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Functions to generate plots for Block-Wise Radiometric Analysis"""

from __future__ import annotations

import warnings
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from perseo_quality.logger import quality_logger as log
from perseo_quality.radiometric_analysis.custom_dataclasses import RadiometricProfilesOutput


class PlotModes(Enum):
    """Overall profile computation mode"""

    MEAN = "mean"
    MIN = "min"


def radiometric_2D_hist_plot(
    data: RadiometricProfilesOutput,
    out_dir: str | Path | None,
    title: str | None = None,
    plot_mode: str | PlotModes = PlotModes.MEAN,
    interactive: bool = False,
) -> None:
    """Radiometric profiles 2D histogram plot.

    Parameters
    ----------
    data : RadiometricProfilesOutput
        radiometric profiles output data
    out_dir : str | Path | None
        output folder path, while interactive mode is on it is ignored so it can be passed as None
    title : str | None
        plot title
    plot_mode : str | PlotModes, optional
        overall profile extraction mode, by default PlotModes.MEAN
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
        graphs_dir = None

    if not interactive and out_dir is None:
        raise ValueError("A valid output directory path must be provided when interactive mode is off")

    if out_dir is not None:
        graphs_dir = Path(out_dir).joinpath("graphs")
        graphs_dir.mkdir(exist_ok=True)

    plot_mode = PlotModes(plot_mode)

    # figure plot
    if title is None:
        title = (
            f"Radiometric Profile Histogram {data.general_info.swath} {data.general_info.polarization} "
            + f"Channel {str(data.general_info.channel)}"
        )
    log.info(f"Generating {title}")

    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)

    ax.imshow(
        data.hist_2d,
        cmap="viridis",
        vmin=1,
        extent=[
            data.hist_x_bins_axis.min(),
            data.hist_x_bins_axis.max(),
            data.hist_y_bins_axis.max(),
            data.hist_y_bins_axis.min(),
        ],
    )
    ax.invert_yaxis()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if plot_mode == PlotModes.MEAN:
            mean_profile = np.nanmean(data.profiles, 0)
        elif plot_mode == PlotModes.MIN:
            np.ma.set_fill_value(data.profiles, np.nan)
            mean_profile = np.nanmin(data.profiles.data, 0)
        if data.look_angles is not None:
            mean_profile_axis = np.nanmean(data.look_angles, 0)
        else:
            mean_profile_axis = np.nanmean(data.block_azimuth_times, 0)
    m_indexes = np.arange(len(mean_profile))
    nan_mask = np.isfinite(mean_profile)
    mean_profile_nan_interp = np.interp(m_indexes, m_indexes[nan_mask], mean_profile[nan_mask])
    smoothed_profile = savgol_filter(
        mean_profile_nan_interp, polyorder=3, window_length=mean_profile_nan_interp.size // 10
    )

    # forcing equal aspect ratio
    aspect = 8 / 6
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)

    # ax.invert_yaxis()
    plt.plot(mean_profile_axis, smoothed_profile, color="#F0F0F0", label="smoothed-data")
    if data.noise_vectors is not None:
        plt.plot(
            mean_profile_axis, np.mean(data.noise_vectors, axis=0), color="#467A3E", linewidth=1.2, label="noise-vector"
        )

    plt.locator_params(axis="x", nbins=10)
    plt.locator_params(axis="y", nbins=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    if data.look_angles is not None:
        plt.xlabel("Elevation Angle [deg]", fontdict={"size": 12})
    else:
        plt.xlabel("Azimuth Block Times [s]", fontdict={"size": 12})
    plt.ylabel("Power [dB]", fontdict={"size": 12})
    plt.title(title, fontdict={"size": 16, "weight": "bold"})
    plt.grid(color="#7EB4B4", alpha=0.4)
    plt.legend(loc="upper right", fontsize="small")

    if not interactive:
        fig.savefig(graphs_dir.joinpath(f"radiometric_hist_{str(data.general_info.channel)}").with_suffix(".png"))
        plt.close("all")
    else:
        plt.show()
