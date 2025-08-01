# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Functions to generate plots for Spectral Analysis"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from perseo_quality.logger import quality_logger as log
from perseo_quality.spectral_analysis.custom_dataclasses import SpectraDataOutput

COLORS_SET = ["#282A3E", "#8783D1", "#B0413E"]
CMAP = "inferno"


class SpectralGraphType(Enum):
    """Type of spectral analysis plot"""

    POINT = "POINT"
    DISTRIBUTED = "DISTRIBUTED"


class SpectralGraphMode(Enum):
    """Mode of spectral analysis plot"""

    ABSOLUTE = "ABSOLUTE"
    PHASE = "PHASE"


def spectral_graphs(data: list[SpectraDataOutput], output_dir: str | Path, graph_type: str | SpectralGraphType) -> None:
    """Generating spectral analysis graphs.

    Parameters
    ----------
    data : list[SpectraDataOutput]
        list of spectral profiles and data portions to be plotted for each swath id
    output_dir : str | Path
        Path to the output folder where to save the files
    graph_type : str | SpectralGraphType
        type of graphs to be generated, Point Target or Distributed
    """

    graph_type = SpectralGraphType(graph_type)
    output_dir = Path(output_dir)
    out_dir = output_dir.joinpath("Spectral Analysis")
    out_dir.mkdir(exist_ok=True)
    log.info(f"Saving graphs to {out_dir}")

    for item in data:
        if item.spectrum_db is None:
            log.warning(
                f"Cannot create graph for target {item.target_name}, pol {item.polarization.name}, swath {item.swath}"
            )
            continue

        spectral_graph_core(item=item, graph_type=graph_type, graph_mode=SpectralGraphMode.ABSOLUTE, out_dir=out_dir)
        if graph_type == SpectralGraphType.POINT:
            spectral_graph_core(item=item, graph_type=graph_type, graph_mode=SpectralGraphMode.PHASE, out_dir=out_dir)


def spectral_graph_core(
    item: SpectraDataOutput, graph_type: SpectralGraphType, graph_mode: SpectralGraphMode, out_dir: Path
) -> None:
    """Spectral graphs generator core functionality.

    Parameters
    ----------
    item : SpectraDataOutput
        spectral profiles and data portions to be plotted for a given swath id
    graph_type : SpectralGraphType
        type of graphs to be generated, Point Target or Distributed
    graph_mode : SpectralGraphMode
        graph mode, magnitude or phase plots
    out_dir : Path
        Path to the output folder where to save the files
    """

    fig = plt.figure(figsize=(10, 10))
    grid = GridSpec(2, 2, figure=fig)

    spectrum = item.spectrum_db
    rng_profiles = item.range_profiles_db
    az_profiles = item.azimuth_profiles_db
    if graph_mode == SpectralGraphMode.PHASE:
        spectrum = item.spectrum_deg
        rng_profiles = item.range_profiles_deg
        az_profiles = item.azimuth_profiles_deg

    # frequency spectrum & cuts
    ax = fig.add_subplot(grid[0, 0])
    grph_extent = [
        item.azimuth_frequency_axis[0],
        item.azimuth_frequency_axis[-1],
        item.range_frequency_axis[-1],
        item.range_frequency_axis[0],
    ]
    grph = ax.imshow(spectrum, cmap=CMAP, extent=grph_extent, aspect="auto")
    fig.colorbar(grph, ax=ax)
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
    if graph_mode == SpectralGraphMode.PHASE:
        ax.plot(
            item.range_frequency_axis,
            item.range_polynomial_fit(item.range_frequency_axis),
            color="#9DB78F",
            linestyle="dashed",
        )
    ax.grid(alpha=0.3, linestyle="--")
    ax.locator_params(axis="both", nbins=8)
    ax.set_title("Range Profiles", fontweight="bold")
    if graph_mode == SpectralGraphMode.PHASE:
        ax.set_ylabel("Phase [deg]")
    else:
        ax.set_ylabel("Absolute [dB]")
    ax.set_xlabel("Range Frequency")

    # azimuth profiles
    ax = fig.add_subplot(grid[1, 1])
    for prof_id, profile in enumerate(az_profiles):
        ax.plot(item.azimuth_frequency_axis, profile, color=COLORS_SET[prof_id])
    if graph_mode == SpectralGraphMode.PHASE:
        ax.plot(
            item.azimuth_frequency_axis,
            item.azimuth_polynomial_fit(item.azimuth_frequency_axis),
            color="#9DB78F",
            linestyle="dashed",
        )
    ax.grid(alpha=0.3, linestyle="--")
    ax.locator_params(axis="both", nbins=8)
    ax.set_title("Azimuth Profiles", fontweight="bold")
    if graph_mode == SpectralGraphMode.PHASE:
        ax.set_ylabel("Phase [deg]")
    else:
        ax.set_ylabel("Absolute [dB]")
    ax.set_xlabel("Azimuth Frequency")

    if graph_mode == SpectralGraphMode.ABSOLUTE:
        # spectrogram
        ax = fig.add_subplot(grid[1, 0])
        grph_extent = [
            item.spectrogram_times[0],
            item.spectrogram_times[-1],
            item.spectrogram_frequencies[-1],
            item.spectrogram_frequencies[0],
        ]
        grph = ax.imshow(item.spectrogram_db, cmap=CMAP, extent=grph_extent, aspect="auto")
        fig.colorbar(grph, ax=ax)

        ax.locator_params(axis="both", nbins=8)
        ax.set_title("Spectrogram", fontweight="bold")
        ax.set_ylabel("Frequency")
        ax.set_xlabel("Azimuth [lines]")
    else:
        # polynomial equations
        ax = fig.add_subplot(grid[1, 0])
        ax.axis("off")
        ax.text(0, 0.8, "Range Phase Profile Polynomial:", fontdict={"weight": "bold", "fontsize": 14})
        ax.text(0, 0.7, f"phase [deg] = {item.range_polynomial_fit}")
        ax.text(0, 0.5, "Azimuth Phase Profile Polynomial:", fontdict={"weight": "bold", "fontsize": 14})
        ax.text(0, 0.4, f"phase [deg] = {item.azimuth_polynomial_fit}")

    title_str = "Point Target" if graph_type == SpectralGraphType.POINT else "Distributed Target"
    fig.suptitle(
        f"{title_str} Spectral Analysis - {graph_mode.name.capitalize()}",
        weight="bold",
        fontsize=16,
    )
    fig.text(
        0.5,
        0.93,
        f"Product: {item.product_name}   Channel: {item.channel}  Polarization: {item.polarization.name}"
        + f"   Swath: {item.swath}   Burst: {item.burst}   Target: {item.target_name}   [Normalized Frequencies]",
        ha="center",
    )
    graph_name = "point_target" if graph_type == SpectralGraphType.POINT else "distributed_target"
    filename = "_".join(
        [
            graph_name,
            graph_mode.name.lower(),
            "trgt" + str(item.target_name),
            item.product_name,
            "ch" + str(item.channel),
            item.swath,
            "burst" + str(item.burst),
            "apx" + str(int(item.target_azimuth_pixel)),
            "rpx" + str(int(item.target_range_pixel)),
        ]
    )
    fig.tight_layout(h_pad=2, w_pad=1.5)
    plt.subplots_adjust(top=0.88)
    fig.savefig(out_dir.joinpath(filename + ".png"), dpi=200)
    plt.close("all")
