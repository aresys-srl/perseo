# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Functions to generate plots for Spectral Analysis"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, get_args

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from perseo_quality.logger import quality_logger as log
from perseo_quality.spectral_analysis.custom_dataclasses import (
    DistributedSpectraDataOutput,
    PointTargetSpectraDataOutput,
)

COLORS_SET = ["#282A3E", "#8783D1", "#B0413E"]
CMAP = "inferno"

GraphType = Literal["PHASE", "ABSOLUTE"]


def spectral_graphs(
    data: list[PointTargetSpectraDataOutput] | list[DistributedSpectraDataOutput],
    output_dir: str | Path,
) -> None:
    """Generating spectral analysis graphs.

    Parameters
    ----------
    data : list[PointTargetSpectraDataOutput] | list[DistributedSpectraDataOutput]
        list of spectral profiles and data portions to be plotted for each swath id and target/block
    output_dir : str | Path
        Path to the output folder where to save the files
    """

    output_dir = Path(output_dir)
    out_dir = output_dir.joinpath("Spectral Analysis")
    out_dir.mkdir(exist_ok=True)
    log.info(f"Saving graphs to {out_dir}")

    for item in data:
        spectral_graph_core(data=item, graph_mode="ABSOLUTE", out_dir=out_dir)
        if isinstance(item, PointTargetSpectraDataOutput):
            spectral_graph_core(data=item, graph_mode="PHASE", out_dir=out_dir)


def spectral_graph_core(
    data: PointTargetSpectraDataOutput | DistributedSpectraDataOutput, graph_mode: GraphType, out_dir: Path
) -> None:
    """Spectral graphs generator core functionality.

    Parameters
    ----------
    item : PointTargetSpectraDataOutput | DistributedSpectraDataOutput
        spectral profiles and data portions to be plotted for a given swath id, block or target
    graph_type : SpectralGraphType
        type of graphs to be generated, Point Target or Distributed
    graph_mode : GraphType
        graph mode, absolute or phase plots
    out_dir : Path
        Path to the output folder where to save the files
    """

    assert graph_mode in get_args(GraphType)

    is_pt_graph = isinstance(data, PointTargetSpectraDataOutput)
    items = data.targets_info if is_pt_graph else data.blocks_info

    for item in items:
        if item.spectrum_db is None:
            msg = f"Cannot create graph for swath {data.swath}, pol {data.polarization.name} "
            if is_pt_graph:
                msg += f"target {item.target_name}, burst {item.burst}"
            else:
                msg += f"block {item.block_num}"
            log.warning(msg)
            continue

        fig = plt.figure(figsize=(10, 10))
        grid = GridSpec(2, 2, figure=fig)

        spectrum = item.spectrum_db
        rng_profiles = item.range_profiles_db
        az_profiles = item.azimuth_profiles_db
        if graph_mode == "PHASE":
            spectrum = item.spectrum_deg
            rng_profiles = item.range_profiles_deg
            az_profiles = item.azimuth_profiles_deg

        # frequency spectrum & cuts
        ax = fig.add_subplot(grid[0, 0])
        graph_extent = [
            item.azimuth_frequency_axis[0],
            item.azimuth_frequency_axis[-1],
            item.range_frequency_axis[-1],
            item.range_frequency_axis[0],
        ]
        graph_ = ax.imshow(spectrum, cmap=CMAP, extent=graph_extent, aspect="auto")
        fig.colorbar(graph_, ax=ax)
        ax.hlines(
            [p / 4 + item.range_frequency_axis[0] for p in range(1, 4)],
            item.azimuth_frequency_axis[0],
            item.azimuth_frequency_axis[-1],
            linestyles="dashed",
            colors=COLORS_SET,
        )
        ax.vlines(
            [p / 4 + item.azimuth_frequency_axis[0] for p in range(1, 4)],
            item.range_frequency_axis[0],
            item.range_frequency_axis[-1],
            linestyles="dashed",
            colors=COLORS_SET,
        )
        ax.locator_params(axis="both", nbins=8)
        ax.set_title("ROI Spectrum", fontweight="bold")
        ax.set_ylabel("Range Frequency")
        ax.set_xlabel("Azimuth Frequency")

        # range profiles
        ax = fig.add_subplot(grid[0, 1])
        for prof_id, profile in enumerate(rng_profiles):
            ax.plot(item.range_frequency_axis, profile, color=COLORS_SET[prof_id])
        if graph_mode == "PHASE":
            ax.plot(
                item.range_frequency_axis,
                item.range_polynomial_fit(item.range_frequency_axis),
                color="#9DB78F",
                linestyle="dashed",
            )
        ax.grid(alpha=0.3, linestyle="--")
        ax.locator_params(axis="both", nbins=8)
        ax.set_title("Range Profiles", fontweight="bold")
        if graph_mode == "PHASE":
            ax.set_ylabel("Phase [deg]")
        else:
            ax.set_ylabel("Absolute [dB]")
        ax.set_xlabel("Range Frequency")

        # azimuth profiles
        ax = fig.add_subplot(grid[1, 1])
        for prof_id, profile in enumerate(az_profiles):
            ax.plot(item.azimuth_frequency_axis, profile, color=COLORS_SET[prof_id])
        if graph_mode == "PHASE":
            ax.plot(
                item.azimuth_frequency_axis,
                item.azimuth_polynomial_fit(item.azimuth_frequency_axis),
                color="#9DB78F",
                linestyle="dashed",
            )
        ax.grid(alpha=0.3, linestyle="--")
        ax.locator_params(axis="both", nbins=8)
        ax.set_title("Azimuth Profiles", fontweight="bold")
        if graph_mode == "PHASE":
            ax.set_ylabel("Phase [deg]")
        else:
            ax.set_ylabel("Absolute [dB]")
        ax.set_xlabel("Azimuth Frequency")

        if graph_mode == "PHASE":
            # polynomial equations
            ax = fig.add_subplot(grid[1, 0])
            ax.axis("off")
            ax.text(0, 0.8, "Range Phase Profile Polynomial:", fontdict={"weight": "bold", "fontsize": 14})
            ax.text(0, 0.7, f"phase [deg] = {item.range_polynomial_fit}")
            ax.text(0, 0.5, "Azimuth Phase Profile Polynomial:", fontdict={"weight": "bold", "fontsize": 14})
            ax.text(0, 0.4, f"phase [deg] = {item.azimuth_polynomial_fit}")
        else:
            # spectrogram
            ax = fig.add_subplot(grid[1, 0])
            graph_extent = [
                item.spectrogram_times[0],
                item.spectrogram_times[-1],
                item.spectrogram_frequencies[-1],
                item.spectrogram_frequencies[0],
            ]
            graph_ = ax.imshow(item.spectrogram_db, cmap=CMAP, extent=graph_extent, aspect="auto")
            fig.colorbar(graph_, ax=ax)

            ax.locator_params(axis="both", nbins=8)
            ax.set_title("Spectrogram", fontweight="bold")
            ax.set_ylabel("Frequency")
            ax.set_xlabel("Azimuth [lines]")

        title_str = "Point Target" if is_pt_graph else "Distributed Target"
        fig.suptitle(
            f"{title_str} Spectral Analysis - {graph_mode.capitalize()}",
            weight="bold",
            fontsize=16,
        )
        txt_str = (
            f"Product: {data.product_name}   Channel: {data.channel}  Polarization: {data.polarization.name}"
            + f"   Swath: {data.swath}   "
        )
        if is_pt_graph:
            txt_str += f"Burst: {item.burst}   Target: {item.target_name}   [Normalized Frequencies]"
        else:
            txt_str += f"Block: {item.block_num}   [Normalized Frequencies]"
        fig.text(
            0.5,
            0.93,
            txt_str,
            ha="center",
        )
        graph_name = "point_target" if is_pt_graph else "distributed_target"
        filename = "_".join(
            [
                graph_name,
                graph_mode.lower(),
                data.product_name,
                f"ch{data.channel}",
                data.swath,
                data.polarization.name,
            ]
        )
        if is_pt_graph:
            filename += f"_trgt{item.target_name}_burst{item.burst}"
        else:
            filename += f"_block{item.block_num}"
        fig.tight_layout(h_pad=2, w_pad=1.5)
        plt.subplots_adjust(top=0.88)
        fig.savefig(out_dir.joinpath(filename + ".png"), dpi=200)
        plt.close("all")
