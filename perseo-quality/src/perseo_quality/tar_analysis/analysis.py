# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Target Ambiguity Ratio Analysis: Distributed (DTAR) and Point Target (PTAR) Ratios"""

from __future__ import annotations

from datetime import datetime

import numpy as np
from perseo_core.geometry.geocoding import direct_geocoding_monostatic, inverse_geocoding_monostatic

from perseo_quality.core.common import check_targets_visibility, detect_burst_from_pixel
from perseo_quality.core.generic_dataclasses import SARCoordinates
from perseo_quality.io.point_targets import PointTarget
from perseo_quality.io.quality_input_protocol import QualityInputProduct
from perseo_quality.logger import quality_logger as log
from perseo_quality.tar_analysis.config import AmbiguityRatioConfig
from perseo_quality.tar_analysis.custom_dataclasses import (
    AmbiguityRatioProductGeneralInfo,
    AmbiguityRatioROIInfo,
    AmbiguityRatioTargetInfo,
    DistributedTargetAmbiguityRatioDataOutput,
    PointTargetAmbiguityRatioDataOutput,
)
from perseo_quality.tar_analysis.support import (
    ambiguity_ratio_computation_core,
    are_ambiguities_inside_scene,
    compute_ambiguities_locations,
    dtar_computing_function_wrapper,
    ptar_computing_function_wrapper,
)


def point_target_ambiguity_ratio_analysis(
    product: QualityInputProduct,
    point_targets: list[PointTarget],
    config: AmbiguityRatioConfig | None = None,
) -> list[PointTargetAmbiguityRatioDataOutput]:
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
    list[PointTargetAmbiguityRatioDataOutput]
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

        output_results = PointTargetAmbiguityRatioDataOutput(
            general_info=AmbiguityRatioProductGeneralInfo(
                product=product.name,
                channel=str(channel),
                swath=channel_data.swath_name,
                acquisition_mode=channel_data.acquisition_mode.name,
                orbit_direction=channel_data.orbit_direction.name,
                polarization=channel_data.polarization.name,
                product_type=channel_data.image_type.name,
                sensor=channel_data.sensor_name,
                acquisition_start_time=datetime(
                    year=channel_data.azimuth_axis[0].year,
                    month=channel_data.azimuth_axis[0].month,
                    day=channel_data.azimuth_axis[0].day_of_the_month,
                    hour=channel_data.azimuth_axis[0].hour_of_day,
                    minute=channel_data.azimuth_axis[0].minute_of_hour,
                    second=channel_data.azimuth_axis[0].second_of_minute,
                ),
            )
        )

        targets_info = []
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
                # extracting azimuth and range coordinates
                try:
                    trgt_az_time, trgt_rng_time = inverse_geocoding_monostatic(
                        trajectory=channel_data.trajectory,
                        ground_points=current_point_target.xyz_coordinates,
                        doppler_frequencies=0,
                        wavelength=1,
                        az_initial_time_guesses=channel_data.mid_azimuth_time,
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
                    targets_info.append(None)
                    continue

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
                    targets_info.append(None)
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
                        targets_info.append(None)
                        continue

                    target_roi, r_ambiguity_roi, l_ambiguity_roi, ambiguity_ratio = ambiguity_ratio_computation_core(
                        channel_data=channel_data,
                        target_location=point_target_location_px,
                        left_ambiguity_location=l_ambiguity_loc_px,
                        right_ambiguity_location=r_ambiguity_loc_px,
                        ambiguity_ratio_computing_function=ptar_computing_function_wrapper,
                        config=config,
                    )
                    log.info(f"Ambiguity Ratio [dB] = {ambiguity_ratio}")
                except Exception as err:
                    log.warning(f"Could not evaluate PTAR for target {trgt}")
                    log.error(f"Error {err}")
                    targets_info.append(None)
                    continue

                targets_info.append(
                    AmbiguityRatioTargetInfo(
                        target_name=trgt,
                        burst=burst,
                        roi_size_azimuth=config.cropping_size[1],
                        roi_size_range=config.cropping_size[0],
                        target_nominal_coordinates=current_point_target.xyz_coordinates,
                        target_azimuth_pixel=az_rng_coords.azimuth_index_subpx,
                        target_range_pixel=az_rng_coords.range_index_subpx,
                        azimuth_time_delta=az_delta,
                        range_time_delta=rng_delta,
                        left_ambiguity_azimuth_pixel=l_ambiguity_loc_px[0],
                        left_ambiguity_range_pixel=l_ambiguity_loc_px[1],
                        right_ambiguity_azimuth_pixel=r_ambiguity_loc_px[0],
                        right_ambiguity_range_pixel=r_ambiguity_loc_px[1],
                        ambiguity_ratio_db=ambiguity_ratio,
                        target_image=target_roi,
                        right_ambiguity_image=r_ambiguity_roi,
                        left_ambiguity_image=l_ambiguity_roi,
                    )
                )

        output_results.targets_info = targets_info
        res.append(output_results)

    return res


def distributed_target_ambiguity_ratio_analysis(
    product: QualityInputProduct,
    roi_centers: list[tuple[int, int]],
    config: AmbiguityRatioConfig | None = None,
) -> list[DistributedTargetAmbiguityRatioDataOutput]:
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
    list[DistributedTargetAmbiguityRatioDataOutput]
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

        output_results = DistributedTargetAmbiguityRatioDataOutput(
            general_info=AmbiguityRatioProductGeneralInfo(
                product=product.name,
                channel=str(channel),
                swath=channel_data.swath_name,
                acquisition_mode=channel_data.acquisition_mode.name,
                orbit_direction=channel_data.orbit_direction.name,
                polarization=channel_data.polarization.name,
                product_type=channel_data.image_type.name,
                sensor=channel_data.sensor_name,
                acquisition_start_time=datetime(
                    year=channel_data.azimuth_axis[0].year,
                    month=channel_data.azimuth_axis[0].month,
                    day=channel_data.azimuth_axis[0].day_of_the_month,
                    hour=channel_data.azimuth_axis[0].hour_of_day,
                    minute=channel_data.azimuth_axis[0].minute_of_hour,
                    second=channel_data.azimuth_axis[0].second_of_minute,
                ),
            )
        )

        roi_info = []
        for roi_id, roi in enumerate(roi_centers):
            log.info(f"Processing Target Point {roi_id + 1}/{len(roi_centers)}")
            burst = detect_burst_from_pixel(lines_per_burst=channel_data.lines_per_burst, azimuth_px=roi[1])
            azimuth_time, range_time = channel_data.pixel_to_times_conversion(
                azimuth_index=roi[1], range_index=roi[0], burst=burst
            )
            try:
                target_ground_point = direct_geocoding_monostatic(
                    sensor_positions=channel_data.trajectory.position(azimuth_time),
                    sensor_velocities=channel_data.trajectory.velocity(azimuth_time),
                    range_times=range_time,
                    doppler_frequencies=0,
                    wavelength=1,
                    look_direction=channel_data.looking_side.value,
                    altitude=0,
                )
            except Exception:
                log.warning("Invalid Direct Geocoding for the current Swath")
                roi_info.append(None)
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
                roi_info.append(None)
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
                    roi_info.append(None)
                    continue

                target_roi, r_ambiguity_roi, l_ambiguity_roi, ambiguity_ratio = ambiguity_ratio_computation_core(
                    channel_data=channel_data,
                    target_location=(roi[1], roi[0]),
                    left_ambiguity_location=l_ambiguity_loc_px,
                    right_ambiguity_location=r_ambiguity_loc_px,
                    ambiguity_ratio_computing_function=dtar_computing_function_wrapper,
                    config=config,
                )
                log.info(f"Ambiguity Ratio [dB] = {ambiguity_ratio}")
            except Exception as err:
                log.warning(f"Could not evaluate DTAR for target {roi_id}")
                log.error(f"Error {err}")
                roi_info.append(None)
                continue

            roi_info.append(
                AmbiguityRatioROIInfo(
                    roi_name=roi_id,
                    burst=burst,
                    roi_size_azimuth=config.cropping_size[1],
                    roi_size_range=config.cropping_size[0],
                    roi_center_azimuth_pixel=roi[1],
                    roi_center_range_pixel=roi[0],
                    roi_center_ground_point_coordinates=target_ground_point,
                    azimuth_time_delta=az_delta,
                    range_time_delta=rng_delta,
                    left_ambiguity_azimuth_pixel=l_ambiguity_loc_px[0],
                    left_ambiguity_range_pixel=l_ambiguity_loc_px[1],
                    right_ambiguity_azimuth_pixel=r_ambiguity_loc_px[0],
                    right_ambiguity_range_pixel=r_ambiguity_loc_px[1],
                    ambiguity_ratio_db=ambiguity_ratio,
                    target_image=target_roi,
                    right_ambiguity_image=r_ambiguity_roi,
                    left_ambiguity_image=l_ambiguity_roi,
                )
            )

        output_results.roi_info = roi_info
        res.append(output_results)

    return res
