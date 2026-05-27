# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Functions to generate plots for Elevation Notch Analysis"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch, Rectangle
from numpy.polynomial import Polynomial

from perseo_quality.elevation_notch_analysis.custom_dataclasses import ElevationNotchOutput

COLORS = {"antenna_model": "#CC3D00", "parabola": "#083B72", "scatter_data": "#7C703C"}


def plot_elevation_notch_analysis(data: list[ElevationNotchOutput], output_dir: str | Path) -> None:
    """Generating the plots for the Elevation Notch Analysis.

    Parameters
    ----------
    data : list[ElevationNotchOutput]
        list of ElevationNotchOutput results dataclasses
    output_dir : str | Path
        Path to the output directory where to save the graphs
    """
    output_dir = Path(output_dir)
    for channel in data:
        fig = plt.figure(figsize=(12, 10))
        gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.35)

        ax = fig.add_subplot(gs[0])
        ax_zoom = fig.add_subplot(gs[1])

        scatter_subsampling = int(channel.blocks_info[0].samples_block * 1 / 100)
        for blk in channel.blocks_info:
            # ---- plotting identical to before ----
            ax.scatter(
                blk.antenna_angles_deg[::scatter_subsampling],
                blk.antenna_profile_from_data_db[::scatter_subsampling],
                s=10,
                alpha=0.3,
                facecolors="none",
                edgecolors=COLORS["scatter_data"],
                label="real data (subsampled)",
            )

            ax_zoom.scatter(
                blk.antenna_angles_deg[:: scatter_subsampling // 3],
                blk.antenna_profile_from_data_db[:: scatter_subsampling // 3],
                s=10,
                alpha=0.5,
                facecolors="none",
                edgecolors=COLORS["scatter_data"],
            )

            ax_zoom.plot(
                blk.parabolic_fit_axis_deg + blk.annotated_roll_deg,
                blk.antenna_profile_parabolic_fit_db,
                color=COLORS["parabola"],
                alpha=0.25,
            )

            if blk.antenna_profile_from_model_db is not None:
                ax_zoom.plot(
                    blk.antenna_angles_deg[::scatter_subsampling],
                    blk.antenna_profile_from_model_db[::scatter_subsampling],
                    color=COLORS["antenna_model"],
                    linewidth=0.3,
                )

        # ===== Average altitude [m] =====
        average_altitude_m = round(np.mean([blk.altitude_m for blk in channel.blocks_info]), 1)

        # ===== Plot average parabolic fit =====
        avg_parabolic_axis, avg_parabola = _compute_average_parabolic_fit(channel.blocks_info)
        ax.plot(
            avg_parabolic_axis,
            avg_parabola,
            color=COLORS["parabola"],
            label="average parabolic fit",
        )

        # ===== Plot average antenna profile fit =====
        if channel.blocks_info[0].antenna_profile_from_model_db is not None:
            avg_antenna_angles, avg_antenna_profile = _compute_average_antenna_model(channel.blocks_info)
            ax.plot(
                avg_antenna_angles,
                avg_antenna_profile,
                color=COLORS["antenna_model"],
                linewidth=1.2,
                label="average antenna model fit",
            )

        # ===== Set zoom limits =====
        zoom_x = (avg_parabolic_axis[0] - 0.15, avg_parabolic_axis[-1] + 0.15)
        zoom_y = (avg_parabola.min() - 1.2, avg_parabola.max() + 0.25)
        ax_zoom.set_xlim(*zoom_x)
        ax_zoom.set_ylim(*zoom_y)

        # ===== Shaded zoom rectangle (true data-space rectangle) =====
        rect = Rectangle(
            (zoom_x[0], zoom_y[0]),
            zoom_x[1] - zoom_x[0],
            zoom_y[1] - zoom_y[0],
            linewidth=1,
            edgecolor="gray",
            facecolor="gray",
            alpha=0.15,
        )
        ax.add_patch(rect)

        # ===== Connection lines snapped to rectangle corners =====
        # From LOWER corners of rectangle (upper plot)
        # To UPPER corners of zoom plot data region

        con_left = ConnectionPatch(
            xyA=(zoom_x[0], zoom_y[0]),  # bottom-left rectangle corner
            coordsA=ax.transData,
            xyB=(zoom_x[0], zoom_y[1]),  # top-left zoom plot corner
            coordsB=ax_zoom.transData,
            color="gray",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
        )

        con_right = ConnectionPatch(
            xyA=(zoom_x[1], zoom_y[0]),  # bottom-right rectangle corner
            coordsA=ax.transData,
            xyB=(zoom_x[1], zoom_y[1]),  # top-right zoom plot corner
            coordsB=ax_zoom.transData,
            color="gray",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
        )

        fig.add_artist(con_left)
        fig.add_artist(con_right)

        # ===== Labels & grids =====
        ax.set_ylabel("Antenna Profile (Normalized) [dB]", fontweight="bold")
        ax.set_xlabel("Antenna angles [deg]", fontweight="bold")
        fig.suptitle("Elevation Notch - Antenna Model Mispointing Computation", fontsize=16, fontweight="bold", y=0.95)
        ax.set_title(
            f"Polarization: {channel.polarization.name}   Average Altitude: {average_altitude_m} m", fontsize=14
        )

        ax_zoom.set_xlabel("Antenna angles [deg]", fontweight="bold")
        ax_zoom.set_ylabel("[dB]", fontweight="bold")

        # ===== Legend =====
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles, strict=True))
        ax.legend(unique.values(), unique.keys(), loc="lower right")

        # ===== Grids =====
        ax.grid(linestyle="dashed", alpha=0.5)
        ax_zoom.grid(which="both", linestyle="dashed", alpha=0.5)

        # ===== Save and close =====
        title = "elevation_notch_analysis_" + channel.swath + "_" + channel.polarization.name
        plt.savefig(output_dir.joinpath(title + ".png"), dpi=150)
        plt.close("all")


def _compute_average_parabolic_fit(
    blocks_info: list[ElevationNotchOutput],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Computing the average parabolic fit for the given blocks info.

    Parameters
    ----------
    blocks_info : list[ElevationNotchOutput]
        block info for the current channel

    Returns
    -------
    npt.NDArray[np.floating]
        average parabolic fit axis
    npt.NDArray[np.floating]
        average parabolic fit values
    """
    min_axis = np.min([blk.parabolic_fit_axis_deg[0] + blk.annotated_roll_deg for blk in blocks_info])
    max_axis = np.max([blk.parabolic_fit_axis_deg[-1] + blk.annotated_roll_deg for blk in blocks_info])
    avg_parabolic_axis = np.linspace(min_axis, max_axis, 500)
    parabola_coeff_avg = np.stack([b.parabola_coefficients for b in blocks_info]).mean(axis=0)
    parabola = Polynomial(parabola_coeff_avg)
    return avg_parabolic_axis, 10 * np.log10(parabola(np.deg2rad(avg_parabolic_axis)))


def _compute_average_antenna_model(
    blocks_info: list[ElevationNotchOutput],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Computing average antenna model across all blocks for the current channel.

    Parameters
    ----------
    blocks_info : list[ElevationNotchOutput]
        list of BlockInfo objects

    Returns
    -------
    npt.NDArray[np.floating]
        average look angles
    npt.NDArray[np.floating]
        average antenna model
    """
    antenna_angles = np.stack([b.antenna_angles_deg for b in blocks_info])
    antenna_model = np.stack([b.antenna_profile_from_model_db for b in blocks_info])
    return antenna_angles.mean(axis=0), antenna_model.mean(axis=0)
