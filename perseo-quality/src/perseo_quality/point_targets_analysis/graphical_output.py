# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Functions to generate plots for IRF and RCS analyses"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties

import perseo_quality.core.generic_dataclasses as gdt
import perseo_quality.point_targets_analysis.custom_dataclasses as ptdt
from perseo_quality.core.signal_processing import convert_to_db
from perseo_quality.logger import quality_logger as log


def point_target_graphs_generation(
    graphs_data: list[ptdt.PointTargetGraphicalData],
    results_df: pd.DataFrame,
    output_dir: str | Path | None,
    interactive: bool = False,
) -> None:
    """Full Point Target Analysis output graphs generation.

    Parameters
    ----------
    graphs_data : list[PointTargetGraphicalData]
        graphs data for plotting results
    results_df : pd.DataFrame
        point target analysis results dataframe
    output_dir : str | Path | None
        path to output directory where to save the graphs, while interactive mode is on it is ignored so it can be
        passed as None
    interactive : bool, optional
        if set to True, this flag will show the generated plots and will not dump to disk the graphs, by default False

    Raises
    ------
    ValueError
        if output directory is None and interactive flag is False
    """

    output_dir = Path(output_dir) if output_dir is not None else None

    if interactive and output_dir is not None:
        log.warning(f"Interactive mode is on, graphs won't be saved to {output_dir}")
        output_dir = None

    if not interactive and output_dir is None:
        raise ValueError("A valid output directory path must be provided when interactive mode is off")

    target_var = "target_name" if "target_name" in results_df.columns else "target"

    for item in graphs_data:
        try:
            log.debug(f"Generating graphs for {item.channel}, target {item.target}...")
            data_val = results_df.query(
                f"{target_var} == @item.target & channel == @item.channel & "
                + "burst == @item.burst & swath == @item.swath & "
                + "polarization == @item.polarization.value"
            ).to_dict("records")[0]
            label = (
                f"target_{data_val[target_var]}_{data_val['swath']}_"
                + f"polarization_{data_val['polarization'].replace('/', '')}_"
                + f"{data_val['product_type']}_b{data_val['burst']}"
            )
            irf_graphs(
                data_graph=item.irf, data_values=data_val, label=label, out_dir=output_dir, interactive=interactive
            )
            rcs_graphs(data_graph=item.rcs, label=label, out_dir=output_dir, interactive=interactive)
        except Exception:
            log.warning(f"Could not create graph for {item.channel}, target {item.target} ...")
            continue


def irf_graphs(
    data_graph: ptdt.IRFGraphDataOutput, data_values: dict, label: str, out_dir: Path | None, interactive: bool = False
) -> None:
    """Function to generate the graphical output after IRF analysis.

    Parameters
    ----------
    data_graph : ptdt.IRFGraphDataOutput
        dataclass instance containing all relevant data for plotting results
    data_values : dict
        dictionary of IRF results
    label : str
        label of point target in exam
    out_dir : Path | None
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

    if not interactive and out_dir is None:
        raise ValueError("A valid output directory path must be provided when interactive mode is off")

    # figure init
    fig = plt.figure(figsize=(9, 9))
    gs = fig.add_gridspec(3, 2)
    ax1 = fig.add_subplot(gs[:2, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])

    lobe_rng = data_graph.side_lobes_directions[0]
    lobe_az = data_graph.side_lobes_directions[1]

    rng_ax_m = data_graph.rng_axis * data_graph.rng_step_distance
    az_ax_m = data_graph.az_axis * data_graph.az_step_distance
    image_db = convert_to_db(np.abs(data_graph.image), mode=gdt.DecibelConversion.AMPLITUDE)

    # 1st plot: interpolated irf target area
    axes_ratio = az_ax_m.max() / rng_ax_m.max()
    extent = [az_ax_m[0], az_ax_m[-1], rng_ax_m[-1], rng_ax_m[0]]
    ax1.imshow(image_db, vmin=image_db.max() - 40, cmap="jet", extent=extent, aspect=axes_ratio)
    ax1.plot(data_values["azimuth_localization_error_[m]"], data_values["slant_range_localization_error_[m]"], "ro")
    ax1.plot(az_ax_m, lobe_az * rng_ax_m)
    if np.isinf(lobe_rng):
        ax1.vlines(0, rng_ax_m[0], rng_ax_m[-1])
    else:
        ax1.plot(az_ax_m, lobe_rng * rng_ax_m)

    # customization
    ax1.grid(alpha=0.3, linestyle="--")
    ax1.set_title("Interpolated Response", fontweight="bold")
    ax1.set_xlabel(
        "Azimuth [m]",
        fontweight="bold",
    )
    ax1.set_ylabel("Range [m]", fontweight="bold")

    # 2nd plot: summary table
    ax2.axis("off")
    # ax2.axis('tight')
    tbl_data = np.round(
        np.array(
            [
                [data_values["range_resolution_[m]"], data_values["azimuth_resolution_[m]"]],
                [data_values["range_pslr_[dB]"], data_values["azimuth_pslr_[dB]"]],
                [data_values["range_islr_[dB]"], data_values["azimuth_islr_[dB]"]],
            ]
        ),
        5,
    )
    tbl = ax2.table(
        cellText=tbl_data,
        colLabels=["Range", "Azimuth"],
        rowLabels=["Resolution [m]", "PSLR [dB]", "ISLR [dB]"],
        bbox=[0.45, 0.1, 0.6, 0.45],
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.auto_set_column_width(col=[0, 1])

    for (row, col), cell in tbl.get_celld().items():
        if (row == 0) or (col == -1):
            cell.set_text_props(fontproperties=FontProperties(weight="bold"))

    # 3rd plot: localization table
    ax3.axis("off")
    tbl_data1 = np.round(
        np.array(
            [
                [data_values["slant_range_localization_error_[m]"]],
                [data_values["ground_range_localization_error_[m]"]],
                [data_values["azimuth_localization_error_[m]"]],
            ]
        ),
        5,
    )
    tbl1 = ax3.table(
        cellText=tbl_data1,
        colLabels=["Localization Error"],
        rowLabels=["Slant Range [m]", "Ground Range [m]", "Azimuth [m]"],
        bbox=[0.55, 0.4, 0.55, 0.45],
        cellLoc="center",
    )
    tbl1.auto_set_font_size(False)
    tbl1.auto_set_column_width(col=[0])

    for (row, col), cell in tbl1.get_celld().items():
        if (row == 0) or (col == -1):
            cell.set_text_props(fontproperties=FontProperties(weight="bold"))

    # 4th plot: range profile
    prof = convert_to_db(np.abs(data_graph.rng_profile), mode=gdt.DecibelConversion.AMPLITUDE)

    if np.abs(lobe_rng * data_graph.image.shape[1] / data_graph.image.shape[0]) > 1:
        ax4.plot(rng_ax_m, prof)
        ax4.hlines(prof.max() - 3, rng_ax_m[0], rng_ax_m[-1], linestyle="--", color="r")
        x_lim_low = np.max([rng_ax_m[0], -5.5 * data_graph.rng_resolution * data_graph.rng_step_distance])
        x_lim_up = np.min([rng_ax_m[-1], 5.5 * data_graph.rng_resolution * data_graph.rng_step_distance])
    else:
        # with sidelobe dirs
        ax4.plot(data_graph.az_axis * data_graph.rng_step_distance, prof)
        ax4.hlines(
            prof.max() - 3,
            data_graph.az_axis[0] * data_graph.rng_step_distance,
            data_graph.az_axis[-1] * data_graph.rng_step_distance,
            linestyle="--",
            color="r",
        )
        x_lim_low = np.max(
            [
                data_graph.az_axis[0] * data_graph.rng_step_distance,
                -5.5 * data_graph.rng_resolution * data_graph.rng_step_distance,
            ]
        )
        x_lim_up = np.min(
            [
                data_graph.az_axis[-1] * data_graph.rng_step_distance,
                5.5 * data_graph.rng_resolution * data_graph.rng_step_distance,
            ]
        )

    ax4.set_xlim([x_lim_low, x_lim_up])
    ax4.set_ylim([-40, 0.5])
    ax4.grid(alpha=0.4)

    # labelling customization
    ax4.set_xlabel("Range (along cut) [m]", fontweight="bold")
    ax4.set_ylabel("Power [dB]", fontweight="bold")

    # 5th plot: azimuth profile
    prof = convert_to_db(np.abs(data_graph.az_profile), mode=gdt.DecibelConversion.AMPLITUDE)

    if np.abs(lobe_az * data_graph.image.shape[1] / data_graph.image.shape[0]) > 1:
        # with sidelobe dirs
        ax5.plot(data_graph.rng_axis * data_graph.az_step_distance, prof)
        ax5.hlines(
            prof.max() - 3,
            data_graph.rng_axis[0] * data_graph.az_step_distance,
            data_graph.rng_axis[-1] * data_graph.az_step_distance,
            linestyle="--",
            color="r",
        )
        x_lim_low = np.max(
            [
                data_graph.az_axis[0] * data_graph.az_step_distance,
                -5.5 * data_graph.az_resolution * data_graph.az_step_distance,
            ]
        )
        x_lim_up = np.min(
            [
                data_graph.az_axis[-1] * data_graph.az_step_distance,
                5.5 * data_graph.az_resolution * data_graph.az_step_distance,
            ]
        )
    else:
        ax5.plot(az_ax_m, prof)
        ax5.hlines(prof.max() - 3, az_ax_m[0], az_ax_m[-1], linestyle="--", color="r")
        x_lim_low = np.max([az_ax_m[0], -5.5 * data_graph.az_resolution * data_graph.az_step_distance])
        x_lim_up = np.min([az_ax_m[-1], 5.5 * data_graph.az_resolution * data_graph.az_step_distance])

    ax5.set_xlim([x_lim_low, x_lim_up])
    ax5.set_ylim([-40, 0.5])
    ax5.grid(alpha=0.4)

    # labelling customization
    ax5.set_xlabel("Azimuth (along cut) [m]", fontweight="bold")
    ax5.set_ylabel("Power [dB]", fontweight="bold")

    title = label + " IRF Analysis"
    fig.suptitle(title, fontsize=16, fontweight="bold")

    gs.update(wspace=0.3, hspace=0, top=0.97)

    if not interactive:
        fig.savefig(out_dir.joinpath(title).with_suffix(".png"), dpi=200)
        plt.close("all")
    else:
        plt.show()


def rcs_graphs(
    data_graph: ptdt.RCSGraphDataOutput, label: str, out_dir: Path | None, interactive: bool = False
) -> None:
    """Function to generate the graphical output after RCS analysis.

    Parameters
    ----------
    data_graph : ptdt.RCSGraphDataOutput
        dataclass instance containing all relevant data for plotting results
    label : str
        label of point target in exam
    out_dir : Path | None
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

    if not interactive and out_dir is None:
        raise ValueError("A valid output directory path must be provided when interactive mode is off")

    # figure init
    fig, ax_1 = plt.subplots(figsize=(6, 6))
    rng_axis = np.arange(-data_graph.roi_size[0] / 2, data_graph.roi_size[0] / 2) * data_graph.rng_step_distance
    az_axis = np.arange(-data_graph.roi_size[1] / 2, data_graph.roi_size[1] / 2) * data_graph.az_step_distance

    if data_graph.data_type == gdt.TargetDataType.DETECTED:
        im_db = convert_to_db(np.abs(data_graph.image))
    else:
        im_db = convert_to_db(np.abs(data_graph.image), mode=gdt.DecibelConversion.AMPLITUDE)

    axes_ratio = az_axis.max() / rng_axis.max()
    extent = [az_axis[0], az_axis[-1], rng_axis[-1], rng_axis[0]]

    # plotting image
    ax_1.imshow(im_db, vmin=im_db.max() - 40, cmap="jet", aspect=axes_ratio, extent=extent)

    # plotting peak roi rectangle
    roi_peak = np.asarray(data_graph.roi_peak)
    roi_rng = (roi_peak[:2] / data_graph.interp_factor - data_graph.roi_size[0] / 2) * data_graph.rng_step_distance
    roi_az = (roi_peak[2:] / data_graph.interp_factor - data_graph.roi_size[1] / 2) * data_graph.az_step_distance
    rect_peak = patches.Rectangle(
        (roi_az[0], roi_rng[0]),
        roi_az[1] - roi_az[0],
        roi_rng[1] - roi_rng[0],
        linewidth=3,
        edgecolor="r",
        facecolor="none",
    )
    ax_1.add_patch(rect_peak)

    # plotting background corner rectangles
    rect_corners = []
    for rect in data_graph.roi_background:
        rect = np.asarray(rect).astype(float)
        rect[:2] = (rect[:2] - data_graph.roi_size[0] / 2) * data_graph.rng_step_distance
        rect[2:] = (rect[2:] - data_graph.roi_size[1] / 2) * data_graph.az_step_distance

        rect_corners.append(
            patches.Rectangle(
                (rect[2], rect[0]), rect[3] - rect[2], rect[1] - rect[0], linewidth=2, edgecolor="m", facecolor="none"
            )
        )

    for rect in rect_corners:
        ax_1.add_patch(rect)

    # customizing labels
    ax_1.set_xlabel("Azimuth [m]", fontsize=13)
    ax_1.set_ylabel("Range [m]", fontsize=13)

    # adding title and subtitle
    title = label + " RCS Analysis"
    plt.title(r"$\sigma = $" + f"{np.round(data_graph.rcs_lin, 4)} = " + f"{np.round(data_graph.rcs_db, 4)} [dB]")
    plt.suptitle(title, fontsize=16, fontweight="bold", y=0.97)

    if not interactive:
        fig.savefig(out_dir.joinpath(title).with_suffix(".png"), dpi=200)
        plt.close("all")
    else:
        plt.show()
