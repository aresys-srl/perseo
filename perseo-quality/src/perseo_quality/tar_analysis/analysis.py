# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Target Ambiguity Ratio Analysis: Distributed (DTAR) and Point Target (PTAR) Ratios"""

from __future__ import annotations

import numpy as np
from arepytools.geometry.direct_geocoding import direct_geocoding_monostatic
from arepytools.geometry.inverse_geocoding_core import inverse_geocoding_monostatic_core

from perseo_quality.core.common import check_targets_visibility
from perseo_quality.core.generic_dataclasses import SARCoordinates
from perseo_quality.io.point_targets import PointTarget
from perseo_quality.io.quality_input_protocol import QualityInputProduct
from perseo_quality.logger import quality_logger as log
from perseo_quality.tar_analysis.config import AmbiguityRatioConfig
from perseo_quality.tar_analysis.custom_dataclasses import AmbiguityRatioOutput
from perseo_quality.tar_analysis.support import (
    ambiguity_ratio_computation_core,
    are_ambiguities_inside_scene,
    compute_ambiguities_locations,
    detect_burst_from_pixel,
    dtar_computing_function_wrapper,
    ptar_computing_function_wrapper,
)


def point_target_ambiguity_ratio_analysis(
    product: QualityInputProduct,
    point_targets: list[PointTarget],
    config: AmbiguityRatioConfig | None = None,
) -> list[AmbiguityRatioOutput]:
    """Function to compute the Point Target Ambiguity Ratio (PTAR) analysis on selected Point Target locations.

    Parameters
    ----------
    product : QualityInputProduct
        object satisfying the QualityInputProduct protocol
    point_targets : list[PointTarget]
        list of point targets locations, as PointTarget objects
    config : AmbiguityRatioConfig | None, optional
        configuration parameters, by default None

    Returns
    -------
    list[AmbiguityRatioOutput]
        ambiguity ratio results for each target location
    """

    if config is None:
        log.info("Configuration file not provided, using default")
        config = AmbiguityRatioConfig()

    log.info(f"Performing Point Target Ambiguity Ratio Analysis on {product.name}")
    log.info(f"Selected Product has {len(product.channels_list)} channels")

    # check which target are inside the scene
    log.info("Checking which targets are visible in the scene")
    visible_targets = check_targets_visibility(product, point_targets)
    not_visible_targets = visible_targets.query("burst == None")["id"]
    visible_targets = visible_targets.query("burst.notnull()", engine="python")
    if not not_visible_targets.empty:
        log.info(f"Point Targets {not_visible_targets} are not visible in the scene.")

    res = []
    for channel in visible_targets["channel"].unique():
        swath = visible_targets[visible_targets["channel"] == channel]["swath"].iloc[0]
        polar = visible_targets[visible_targets["channel"] == channel]["polarization"].iloc[0].upper()
        log.info(f"Analyzing Channel {channel}, Swath {swath}, Polarization {polar}...")

        # recovering metadata for the current channel
        channel_data = product.get_channel_data(channel_id=channel)

        # recovering only targets visible by this channel
        targets_visible_by_channel = visible_targets[visible_targets["channel"] == channel]["id"]

        for trgt_idx, trgt in enumerate(targets_visible_by_channel):
            bursts_selection = visible_targets.query("channel == @channel & id == @trgt")
            bursts_selection = bursts_selection.loc[:, "burst"].to_list()[0]
            current_point_target = [p for p in point_targets if p.name == trgt]
            assert len(current_point_target) == 1
            current_point_target = current_point_target[0]
            for burst in bursts_selection:
                log.info(
                    f"Processing Target Point {trgt} ({trgt_idx + 1}/{len(targets_visible_by_channel)}), Burst #{burst}"
                )
                output_results = AmbiguityRatioOutput(
                    target_name=trgt,
                    product_name=product.name,
                    channel=channel,
                    swath=swath,
                    polarization=channel_data.polarization,
                    burst=burst,
                    target_nominal_coordinates=current_point_target.xyz_coordinates,
                    roi_size_azimuth=config.cropping_size[1],
                    roi_size_range=config.cropping_size[0],
                )

                # extracting azimuth and range coordinates
                try:
                    trgt_az_time, trgt_rng_time = inverse_geocoding_monostatic_core(
                        trajectory=channel_data.trajectory,
                        ground_points=current_point_target.xyz_coordinates,
                        initial_guesses=channel_data.mid_azimuth_time,
                        frequencies_doppler_centroid=0,
                        wavelength=1,
                    )
                    if current_point_target.delay is not None:
                        trgt_rng_time += current_point_target.delay

                    trgt_azmth_idx, trgt_rng_idx = channel_data.times_to_pixel_conversion(
                        azimuth_time=trgt_az_time, range_time=trgt_rng_time, burst=burst
                    )

                    az_rng_coords = SARCoordinates(
                        azimuth=trgt_az_time,
                        range=trgt_rng_time,
                        azimuth_index_subpx=trgt_azmth_idx,
                        range_index_subpx=trgt_rng_idx,
                    )
                except Exception:
                    az_rng_coords = SARCoordinates()

                if None in (
                    az_rng_coords.azimuth,
                    az_rng_coords.range,
                    az_rng_coords.azimuth_index_subpx,
                    az_rng_coords.range_index_subpx,
                ):
                    res.append(output_results)
                    continue

                output_results.target_azimuth_pixel = az_rng_coords.azimuth_index_subpx
                output_results.target_range_pixel = az_rng_coords.range_index_subpx

                point_target_location_px = (
                    np.round(az_rng_coords.azimuth_index_subpx).astype("int64"),
                    np.round(az_rng_coords.range_index_subpx).astype("int64"),
                )
                doppler_rate = (
                    channel_data.doppler_rate.evaluate(
                        azimuth_time=az_rng_coords.azimuth, range_time=az_rng_coords.range
                    )
                    if channel_data.doppler_rate is not None
                    else None
                )
                if doppler_rate is None or channel_data.prf is None:
                    log.warning(
                        "Doppler Rate function not available in protocol instance, "
                        + "could not compute ambiguities location"
                    )
                    res.append(output_results)
                    continue

                try:
                    l_ambiguity_loc_px, r_ambiguity_loc_px, az_delta, rng_delta = compute_ambiguities_locations(
                        channel_data=channel_data,
                        point_target_xyz_coords=current_point_target.xyz_coordinates,
                        point_target_azimuth_time=az_rng_coords.azimuth,
                        point_target_range_time=az_rng_coords.range,
                        prf=channel_data.prf,
                        doppler_rate=doppler_rate,
                        burst=burst,
                    )
                    # check ambiguity presence in the scene
                    if not are_ambiguities_inside_scene(
                        r_amb=r_ambiguity_loc_px,
                        l_amb=l_ambiguity_loc_px,
                        lines=channel_data.azimuth_axis.size,
                        samples=channel_data.range_axis.size,
                    ):
                        log.warning("Ambiguities out of scene boundaries")
                        continue
                    output_results.azimuth_time_delta = az_delta
                    output_results.range_time_delta = rng_delta
                    output_results.left_ambiguity_azimuth_pixel = l_ambiguity_loc_px[0]
                    output_results.left_ambiguity_range_pixel = l_ambiguity_loc_px[1]
                    output_results.right_ambiguity_azimuth_pixel = r_ambiguity_loc_px[0]
                    output_results.right_ambiguity_range_pixel = r_ambiguity_loc_px[1]

                    target_roi, r_ambiguity_roi, l_ambiguity_roi, ambiguity_ratio = ambiguity_ratio_computation_core(
                        channel_data=channel_data,
                        target_location=point_target_location_px,
                        left_ambiguity_location=l_ambiguity_loc_px,
                        right_ambiguity_location=r_ambiguity_loc_px,
                        ambiguity_ratio_computing_function=ptar_computing_function_wrapper,
                        config=config,
                    )
                    log.info(f"Ambiguity Ratio [dB] = {ambiguity_ratio}")
                    output_results.ambiguity_ratio_db = ambiguity_ratio
                    output_results.target_image = target_roi
                    output_results.left_ambiguity_image = l_ambiguity_roi
                    output_results.right_ambiguity_image = r_ambiguity_roi
                except Exception as err:
                    log.warning(f"Could not evaluate PTAR for target {trgt}")
                    log.error(f"Error {err}")
                    res.append(output_results)
                    continue

            res.append(output_results)

    return res


def distributed_target_ambiguity_ratio_analysis(
    product: QualityInputProduct,
    roi_centers: list[tuple[int, int]],
    config: AmbiguityRatioConfig | None = None,
) -> list[AmbiguityRatioOutput]:
    """Function to compute the Distributed Target Ambiguity Ratio (DTAR) analysis on selected locations.

    Parameters
    ----------
    product : QualityInputProduct
        object satisfying the QualityInputProduct protocol
    roi_centers : list[tuple[int, int]]
        roi centers pixel coordinates where to compute the DTAR analysis, (range pixel index, azimuth pixel index)
    config : AmbiguityRatioConfig | None, optional
        configuration parameters, by default None

    Returns
    -------
    list[AmbiguityRatioOutput]
        ambiguity ratio results for each target location
    """

    if config is None:
        log.info("Configuration file not provided, using default")
        config = AmbiguityRatioConfig()

    log.info(f"Performing Distributed Target Ambiguity Ratio Analysis on {product.name}")
    log.info(f"Selected Product has {len(product.channels_list)} channels")

    res = []
    for channel in product.channels_list:
        # recovering metadata for the current channel
        channel_data = product.get_channel_data(channel_id=channel)
        log.info(
            f"Analyzing Channel {channel}, Swath {channel_data.swath_name},"
            + f" Polarization {channel_data.polarization.name}..."
        )
        for roi_id, roi in enumerate(roi_centers):
            log.info(f"Processing Target Point {roi_id + 1}/{len(roi_centers)}")
            burst = detect_burst_from_pixel(lines_per_burst=channel_data.lines_per_burst, azimuth_px=roi[1])
            output_results = AmbiguityRatioOutput(
                target_name=roi_id,
                product_name=product.name,
                channel=channel,
                swath=channel_data.swath_name,
                burst=burst,
                polarization=channel_data.polarization,
                roi_size_azimuth=config.cropping_size[1],
                roi_size_range=config.cropping_size[0],
                target_azimuth_pixel=roi[1],
                target_range_pixel=roi[0],
            )
            azimuth_time, range_time = channel_data.pixel_to_times_conversion(
                azimuth_index=roi[1], range_index=roi[0], burst=burst
            )
            try:
                target_ground_point = direct_geocoding_monostatic(
                    sensor_positions=channel_data.trajectory.evaluate(azimuth_time),
                    sensor_velocities=channel_data.trajectory.evaluate_first_derivatives(azimuth_time),
                    range_times=range_time,
                    geocoding_side=channel_data.looking_side.value,
                    frequencies_doppler_centroid=0,
                    wavelength=1,
                    geodetic_altitude=0,
                )
                output_results.target_nominal_coordinates = target_ground_point
            except Exception:
                log.warning("Invalid Direct Geocoding for the current Swath")
                res.append(output_results)
                continue

            doppler_rate = (
                channel_data.doppler_rate.evaluate(azimuth_time=azimuth_time, range_time=range_time)
                if channel_data.doppler_rate is not None
                else None
            )
            if doppler_rate is None or channel_data.prf is None:
                log.warning(
                    "Doppler Rate function not available in protocol instance, "
                    + "could not compute ambiguities location"
                )
                res.append(output_results)
                continue

            try:
                l_ambiguity_loc_px, r_ambiguity_loc_px, az_delta, rng_delta = compute_ambiguities_locations(
                    channel_data=channel_data,
                    point_target_xyz_coords=target_ground_point,
                    point_target_azimuth_time=azimuth_time,
                    point_target_range_time=range_time,
                    prf=channel_data.prf,
                    doppler_rate=doppler_rate,
                    burst=burst,
                )
                # check ambiguity presence in the scene
                if not are_ambiguities_inside_scene(
                    r_amb=r_ambiguity_loc_px,
                    l_amb=l_ambiguity_loc_px,
                    lines=channel_data.azimuth_axis.size,
                    samples=channel_data.range_axis.size,
                ):
                    log.warning("Ambiguities out of scene boundaries")
                    continue
                output_results.azimuth_time_delta = az_delta
                output_results.range_time_delta = rng_delta
                output_results.left_ambiguity_azimuth_pixel = l_ambiguity_loc_px[0]
                output_results.left_ambiguity_range_pixel = l_ambiguity_loc_px[1]
                output_results.right_ambiguity_azimuth_pixel = r_ambiguity_loc_px[0]
                output_results.right_ambiguity_range_pixel = r_ambiguity_loc_px[1]

                target_roi, r_ambiguity_roi, l_ambiguity_roi, ambiguity_ratio = ambiguity_ratio_computation_core(
                    channel_data=channel_data,
                    target_location=(roi[1], roi[0]),
                    left_ambiguity_location=l_ambiguity_loc_px,
                    right_ambiguity_location=r_ambiguity_loc_px,
                    ambiguity_ratio_computing_function=dtar_computing_function_wrapper,
                    config=config,
                )
                log.info(f"Ambiguity Ratio [dB] = {ambiguity_ratio}")
                output_results.ambiguity_ratio_db = ambiguity_ratio
                output_results.target_image = target_roi
                output_results.left_ambiguity_image = l_ambiguity_roi
                output_results.right_ambiguity_image = r_ambiguity_roi
            except Exception as err:
                log.warning(f"Could not evaluate DTAR for target {roi_id}")
                log.error(f"Error {err}")
                res.append(output_results)
                continue

            res.append(output_results)

    return res
