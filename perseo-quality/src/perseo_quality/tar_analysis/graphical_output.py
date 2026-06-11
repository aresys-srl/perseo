# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Functions to generate plots for Target Ambiguity Ratio Analysis"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from perseo_quality.core.signal_processing import convert_to_db
from perseo_quality.logger import quality_logger as log
from perseo_quality.tar_analysis.custom_dataclasses import (
    DistributedTargetAmbiguityRatioDataOutput,
    PointTargetAmbiguityRatioDataOutput,
)

CMAP = "inferno"


def ambiguities_graphs(
    data: list[PointTargetAmbiguityRatioDataOutput] | list[DistributedTargetAmbiguityRatioDataOutput],
    output_dir: str | Path,
) -> None:
    """Target ambiguities and ambiguity ratio graphical representation.

    Parameters
    ----------
    data : list[PointTargetAmbiguityRatioDataOutput] | list[DistributedTargetAmbiguityRatioDataOutput]
        list of ambiguity analysis results
    output_dir : str | Path
        output directory where to save the plots
    """

    tar_type = "DTAR" if isinstance(data[0], DistributedTargetAmbiguityRatioDataOutput) else "PTAR"
    output_dir = Path(output_dir)
    tar_out_dir = output_dir.joinpath(tar_type)
    tar_out_dir.mkdir(exist_ok=True)

    for item in data:
        ambiguities_graphs_core(data=item, output_dir=tar_out_dir, tar_type=tar_type)


def ambiguities_graphs_core(
    data: PointTargetAmbiguityRatioDataOutput | DistributedTargetAmbiguityRatioDataOutput,
    output_dir: Path,
    tar_type: str,
) -> None:

    items = data.targets_info if tar_type == "PTAR" else data.roi_info
    items = [t for t in items if t is not None]

    for item in items:
        if item.ambiguity_ratio_db is None:
            log.warning(f"Cannot create {tar_type} ambiguity graph for target {item.target_name}")
            continue

        az_px = item.target_azimuth_pixel if hasattr(item, "target_azimuth_pixel") else item.roi_center_azimuth_pixel
        rng_px = item.target_range_pixel if hasattr(item, "target_range_pixel") else item.roi_center_range_pixel
        name = item.target_name if hasattr(item, "target_name") else item.roi_name

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
            convert_to_db(np.abs(item.left_ambiguity_image) ** 2), cmap=CMAP, extent=grph_extent, aspect="auto"
        )
        fig.colorbar(grph, ax=ax)
        ax.locator_params(axis="both", nbins=6)
        ax.set_title("Left Ambiguity")
        ax.set_ylabel("Range [Samples]")
        ax.set_xlabel("Azimuth [Lines]")

        # target
        ax = fig.add_subplot(grid[0, 1])
        grph_extent = [
            int(az_px) - item.target_image.shape[0] // 2,
            int(az_px) + item.target_image.shape[0] / 2,
            int(rng_px) + item.target_image.shape[1] // 2,
            int(rng_px) - item.target_image.shape[1] // 2,
        ]
        grph = ax.imshow(convert_to_db(np.abs(item.target_image) ** 2), cmap=CMAP, extent=grph_extent, aspect="auto")
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
            convert_to_db(np.abs(item.right_ambiguity_image) ** 2), cmap=CMAP, extent=grph_extent, aspect="auto"
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
                        np.round(rng_px).astype(int),
                        np.round(item.right_ambiguity_range_pixel).astype(int),
                    ],
                    [
                        np.round(item.left_ambiguity_azimuth_pixel).astype(int),
                        np.round(az_px).astype(int),
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

        title_str = "Distributed" if tar_type == "DTAR" else "Point"
        fig.suptitle(
            f"{title_str} Target Ambiguity Ratio Analysis  [{np.round(item.ambiguity_ratio_db, 3)} dB]",
            weight="bold",
            fontsize=16,
        )
        fig.text(
            0.5,
            0.9,
            f"Product: {data.general_info.product}  Channel: {data.general_info.channel}  "
            + f"Polarization: {data.general_info.polarization}  "
            + f"Swath: {data.general_info.swath}  Burst: {item.burst}  "
            + f"{'Target' if tar_type == 'PTAR' else 'ROI'} : {name}",
            ha="center",
        )
        fig.tight_layout(h_pad=2, w_pad=1.5)
        plt.subplots_adjust(top=0.82)

        filename = "_".join(
            [
                tar_type.lower(),
                f"{'trgt' if tar_type == 'PTAR' else 'roi'}{name}",
                data.general_info.product,
                f"ch{data.general_info.channel}",
                data.general_info.swath,
                f"burst{item.burst}",
                f"apx{int(az_px)}",
                f"rpx{int(rng_px)}",
            ]
        )
        fig.savefig(output_dir.joinpath(filename + ".png"), dpi=200)
        plt.close("all")
