# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Functions to generate plots for Target Ambiguity Ratio Analysis"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from perseo_quality.core.signal_processing import convert_to_db
from perseo_quality.logger import quality_logger as log
from perseo_quality.tar_analysis.custom_dataclasses import AmbiguityRatioOutput


class TargetRatioGraphType(Enum):
    """Type of ambiguity ratio plot"""

    PTAR = "PTAR"
    DTAR = "DTAR"


def ambiguities_graphs(
    data: list[AmbiguityRatioOutput], output_dir: str | Path, graph_type: str | TargetRatioGraphType
) -> None:
    """Target ambiguities and ambiguity ratio graphical representation.

    Parameters
    ----------
    data : list[AmbiguityRatioOutput]
        list of AmbiguityRatioOutput results from ambiguity analysis
    output_dir : str | Path
        output directory where to save the plots
    graph_type : str | TargetRatioGraphType
        type of graphs to be performed, PTAR or DTAR
    """

    graph_type = TargetRatioGraphType(graph_type)
    output_dir = Path(output_dir)
    ptar_out_dir = output_dir.joinpath(graph_type.name)
    ptar_out_dir.mkdir(exist_ok=True)
    cmap = "inferno"

    for item in data:
        if item.ambiguity_ratio_db is None:
            log.warning(f"Cannot create {graph_type.name} ambiguity graph for target {item.target_name}")
            continue

        fig = plt.figure(figsize=(14, 6))
        grid = GridSpec(2, 3, figure=fig, height_ratios=[1, 0.7])

        # left ambiguity
        ax = fig.add_subplot(grid[0, 0])
        grph_extent = [
            int(item.left_ambiguity_azimuth_pixel) - item.left_ambiguity_image.shape[0] // 2,
            int(item.left_ambiguity_azimuth_pixel) + item.left_ambiguity_image.shape[0] // 2,
            int(item.left_ambiguity_range_pixel) + item.left_ambiguity_image.shape[1] // 2,
            int(item.left_ambiguity_range_pixel) - item.left_ambiguity_image.shape[1] // 2,
        ]
        grph = ax.imshow(
            convert_to_db(np.abs(item.left_ambiguity_image) ** 2), cmap=cmap, extent=grph_extent, aspect="auto"
        )
        fig.colorbar(grph, ax=ax)
        ax.locator_params(axis="both", nbins=6)
        ax.set_title("Left Ambiguity")
        ax.set_ylabel("Range [Samples]")
        ax.set_xlabel("Azimuth [Lines]")

        # target
        ax = fig.add_subplot(grid[0, 1])
        grph_extent = [
            int(item.target_azimuth_pixel) - item.target_image.shape[0] // 2,
            int(item.target_azimuth_pixel) + item.target_image.shape[0] / 2,
            int(item.target_range_pixel) + item.target_image.shape[1] // 2,
            int(item.target_range_pixel) - item.target_image.shape[1] // 2,
        ]
        grph = ax.imshow(convert_to_db(np.abs(item.target_image) ** 2), cmap=cmap, extent=grph_extent, aspect="auto")
        fig.colorbar(grph, ax=ax)
        ax.locator_params(axis="both", nbins=6)
        ax.set_title("Target")
        ax.set_ylabel("Range [Samples]")
        ax.set_xlabel("Azimuth [Lines]")

        # right ambiguity
        ax = fig.add_subplot(grid[0, 2])
        grph_extent = [
            int(item.right_ambiguity_azimuth_pixel) - item.right_ambiguity_image.shape[0] // 2,
            int(item.right_ambiguity_azimuth_pixel) + item.right_ambiguity_image.shape[0] // 2,
            int(item.right_ambiguity_range_pixel) + item.right_ambiguity_image.shape[1] // 2,
            int(item.right_ambiguity_range_pixel) - item.right_ambiguity_image.shape[1] // 2,
        ]
        grph = ax.imshow(
            convert_to_db(np.abs(item.right_ambiguity_image) ** 2), cmap=cmap, extent=grph_extent, aspect="auto"
        )
        fig.colorbar(grph, ax=ax)
        ax.locator_params(axis="both", nbins=6)
        ax.set_title("Right Ambiguity")
        ax.set_ylabel("Range [Samples]")
        ax.set_xlabel("Azimuth [Lines]")

        # summary table
        ax = fig.add_subplot(grid[1, :])
        ax.axis("off")
        tbl_data = np.round(
            np.array(
                [
                    [
                        np.round(item.left_ambiguity_range_pixel).astype(int),
                        np.round(item.target_range_pixel).astype(int),
                        np.round(item.right_ambiguity_range_pixel).astype(int),
                    ],
                    [
                        np.round(item.left_ambiguity_azimuth_pixel).astype(int),
                        np.round(item.target_azimuth_pixel).astype(int),
                        np.round(item.right_ambiguity_azimuth_pixel).astype(int),
                    ],
                ]
            ),
            4,
        )
        tbl = ax.table(
            cellText=tbl_data,
            colLabels=[
                "Nominal Left Ambiguity Position [px]",
                "Nominal Target Position [px]",
                "Nominal Right Ambiguity Position [px]",
            ],
            rowLabels=["Range", "Azimuth"],
            bbox=[0.15, 0.3, 0.7, 0.5],
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.auto_set_column_width(col=[0, 1, 2])

        title_str = "Distributed" if graph_type == TargetRatioGraphType.DTAR else "Point"
        fig.suptitle(
            f"{title_str} Target Ambiguity Ratio Analysis  [{np.round(item.ambiguity_ratio_db, 3)} dB]",
            weight="bold",
            fontsize=16,
        )
        fig.text(
            0.5,
            0.9,
            f"Product: {item.product_name}   Channel: {item.channel}  Polarization: {item.polarization.name}"
            + f"   Swath: {item.swath}   Burst: {item.burst}   Target: {item.target_name}",
            ha="center",
        )
        fig.tight_layout(h_pad=2, w_pad=1.5)
        plt.subplots_adjust(top=0.82)

        filename = "_".join(
            [
                graph_type.name.lower(),
                "trgt" + str(item.target_name),
                item.product_name,
                "ch" + str(item.channel),
                item.swath,
                "burst" + str(item.burst),
                "apx" + str(int(item.target_azimuth_pixel)),
                "rpx" + str(int(item.target_range_pixel)),
            ]
        )
        fig.savefig(ptar_out_dir.joinpath(filename + ".png"), dpi=200)
        plt.close("all")
