# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Block-Wise Radiometric Analysis profiles computation"""

from __future__ import annotations

import numpy as np
from arepytools.geometry.geometric_functions import (
    compute_incidence_angles,
    compute_look_angles,
    compute_look_angles_from_trajectory,
)

from perseo_quality.core.generic_dataclasses import SARRadiometricQuantity
from perseo_quality.core.signal_processing import radiometric_correction
from perseo_quality.io.quality_input_protocol import QualityInputProduct
from perseo_quality.logger import quality_logger as log
from perseo_quality.radiometric_analysis.block_wise.config import RadiometricProfilesConfig
from perseo_quality.radiometric_analysis.block_wise.core.kpi_estimators import RadiometricBlockKPIEstimatorType
from perseo_quality.radiometric_analysis.block_wise.core.profile_extractors import RadiometricProfileExtractorType
from perseo_quality.radiometric_analysis.block_wise.support import (
    angles_computation_setup,
    blocks_definition,
    compute_2d_histogram,
)
from perseo_quality.radiometric_analysis.custom_dataclasses import (
    RadiometricAnalysisDirection,
    RadiometricOutputProductGeneralInfo,
    RadiometricProfileAxes,
    RadiometricProfilesOutput,
)


def radiometric_profiles(
    product: QualityInputProduct,
    profile_extractor_func: RadiometricProfileExtractorType,
    kpi_estimator_func: RadiometricBlockKPIEstimatorType,
    direction: RadiometricAnalysisDirection = RadiometricAnalysisDirection.RANGE,
    output_quantity: SARRadiometricQuantity = SARRadiometricQuantity.GAMMA_NOUGHT,
    config: RadiometricProfilesConfig | None = None,
) -> list[RadiometricProfilesOutput]:
    """Block-wise Radiometric profiles computation.

    Parameters
    ----------
    product : QualityInputProduct
        object containing product information and data satisfying the QualityInputProduct protocol
    profile_extractor_func : RadiometricProfileExtractorType
        function to perform radiometric profile extraction
    kpi_estimator_func : RadiometricBlockKPIEstimatorType
        function to estimate KPI from a given block and its extracted profile
    direction : RadiometricAnalysisDirection, optional
        direction along which profiles are extracted, by default RadiometricAnalysisDirection.RANGE
    output_quantity : SARRadiometricQuantity, optional
        desired radiometric output quantity, by default SARRadiometricQuantity.GAMMA_NOUGHT
    config : RadiometricProfilesConfig | None, optional
        RadiometricProfiles configuration dataclass, by default None

    Returns
    -------
    list[RadiometricProfilesOutput]
        a RadiometricProfilesOutput dataclass for each channel
    """
    # managing inputs
    if config is None:
        config = RadiometricProfilesConfig()
    log.info("Performing radiometric analysis block-wise.")

    output_results = []
    for channel in product.channels_list:
        channel_data = product.get_channel_data(channel_id=channel)
        log.info(
            f"Analyzing channel {channel}, swath {channel_data.swath_name} and "
            + f"polarization {channel_data.polarization.value}..."
        )

        log.info("Defining blocks partitioning of the whole scene.")
        # defining scene partitioning by blocks
        az_block_size, blocks_num, blocks_centers_px = blocks_definition(
            azimuth_axis=channel_data.azimuth_axis,
            range_axis=channel_data.slant_range_axis,
            lines_per_burst=channel_data.lines_per_burst,
            default_block_size=config.azimuth_block_size,
        )

        if direction == RadiometricAnalysisDirection.RANGE:
            # creating axis for range direction
            look_angles_mid_swath = compute_look_angles_from_trajectory(
                trajectory=channel_data.trajectory,
                azimuth_time=channel_data.mid_azimuth_time,
                range_times=channel_data.slant_range_axis[config.range_pixel_margin : -config.range_pixel_margin],
                look_direction=channel_data.looking_side.value,
            )
            look_angles_mid_swath = np.rad2deg(look_angles_mid_swath)
            hist_axis = np.arange(look_angles_mid_swath[0] - 0.5, look_angles_mid_swath[-1] + 0.5, 0.01)

        elif direction == RadiometricAnalysisDirection.AZIMUTH:
            # creating axis for azimuth direction
            azimuth_rel_axis = channel_data.azimuth_axis - channel_data.azimuth_axis[0]
            hist_axis = np.arange(azimuth_rel_axis[10], azimuth_rel_axis[-10], 0.01)

        else:
            raise RuntimeError(f"{direction} invalid. It must be Range or Azimuth.")

        cropping_size = (
            channel_data.slant_range_axis.size - 2 * config.range_pixel_margin,  # range
            az_block_size,  # azimuth
        )

        profiles = []
        kpi = []
        look_angles_array = []
        incidence_angles_array = []
        az_rel_times = []
        for bc_num, center in enumerate(blocks_centers_px):
            log.info(f"Processing block {bc_num + 1} of {blocks_num}")

            # reading block, without converting the data here, applying radiometric conversion in the next steps
            target_area = channel_data.read_data(
                azimuth_index=center[0],
                range_index=center[1],
                cropping_size=cropping_size,
                output_radiometric_quantity=channel_data.radiometric_quantity,
            )
            # converting image to power
            target_area = np.abs(target_area) ** 2

            sensor_pos, ground_points, nadir = angles_computation_setup(
                trajectory=channel_data.trajectory,
                azimuth_time=channel_data.azimuth_axis[center[0]],
                range_values=channel_data.slant_range_axis[config.range_pixel_margin : -config.range_pixel_margin],
                look_direction=channel_data.looking_side.value,
            )

            look_angles_mid_block_deg = np.rad2deg(
                compute_look_angles(sensor_positions=sensor_pos, nadir_directions=nadir, points=ground_points)
            )
            if direction == RadiometricAnalysisDirection.RANGE:
                look_angles_array.append(look_angles_mid_block_deg)

            elif direction == RadiometricAnalysisDirection.AZIMUTH:
                az_axis_start_idx = center[0] - np.floor(az_block_size / 2).astype(int)
                az_block_axis = channel_data.azimuth_axis[az_axis_start_idx : az_axis_start_idx + az_block_size]
                az_rel_times.append(az_block_axis - channel_data.azimuth_axis[0])

            # NOTE: use incidence angles from product when available
            incidence_angles_mid_block_rad = compute_incidence_angles(sensor_positions=sensor_pos, points=ground_points)
            incidence_angles_array.append(np.rad2deg(incidence_angles_mid_block_rad))
            # performing radiometric correction, if needed
            if channel_data.radiometric_quantity != output_quantity:
                log.info(
                    f"Converting data from {channel_data.radiometric_quantity.name.lower().replace('_', ' ')} to "
                    + f"{output_quantity.name.lower().replace('_', ' ')}."
                )
                target_area = radiometric_correction(
                    data=target_area,
                    incidence_angle=incidence_angles_mid_block_rad,
                    input_quantity=channel_data.radiometric_quantity,
                    output_quantity=output_quantity,
                    exp_power=1,  # NOTE: power data is processed with 1 as exponent
                )

            # applying provided profile extraction function
            log.debug("Extracting profiles.")
            profile_axes = RadiometricProfileAxes(
                look_angles_deg=look_angles_mid_block_deg,
                incidence_angles_deg=np.rad2deg(incidence_angles_mid_block_rad),
                azimuth=None,  # TODO: add this when scalloping is completed
                slant_range=channel_data.slant_range_axis[config.range_pixel_margin : -config.range_pixel_margin],
            )
            profile = profile_extractor_func(target_area, config.profile_extraction_parameters)
            block_kpi = kpi_estimator_func(profile, profile_axes, target_area)
            block_kpi.block_num = bc_num
            block_kpi.first_az_line_block = int(center[0] - np.floor(cropping_size[1] / 2))
            block_kpi.lines_block = target_area.shape[1]
            profiles.append(profile)
            kpi.append(block_kpi)

        # 2D histogram
        log.info("Computing 2D radiometric histogram...")
        profiles = np.ma.stack(profiles)
        look_angles_array = np.vstack(look_angles_array) if look_angles_array else None
        incidence_angles_array = np.vstack(incidence_angles_array)
        az_rel_times = np.vstack(az_rel_times).astype(float) if az_rel_times else None
        hist, x_bins, y_bins = compute_2d_histogram(
            x_data=look_angles_array if look_angles_array is not None else az_rel_times,
            y_data=profiles,
            x_axis=hist_axis,
            config=config.histogram_parameters,
        )

        # storing results
        output_results.append(
            RadiometricProfilesOutput(
                general_info=RadiometricOutputProductGeneralInfo(
                    product_name=product.name,
                    channel=str(channel),
                    swath=channel_data.swath_name,
                    acquisition_mode=channel_data.acquisition_mode.name,
                    orbit_direction=channel_data.orbit_direction.name,
                    polarization=channel_data.polarization.name,
                    product_type=channel_data.image_type.name,
                    radiometric_quantity=output_quantity.name,
                    sensor=channel_data.sensor_name,
                ),
                direction=direction,
                kpi=kpi,
                azimuth_start_time=channel_data.azimuth_axis[0],
                azimuth_block_centers=channel_data.azimuth_axis[[t[0] for t in blocks_centers_px]],
                range_block_centers=channel_data.slant_range_axis[[t[1] for t in blocks_centers_px]],
                blocks_num=blocks_num,
                profiles=profiles,
                block_azimuth_times=az_rel_times,
                look_angles=look_angles_array,
                incidence_angles=incidence_angles_array,
                hist_2d=hist,
                hist_x_bins_axis=x_bins,
                hist_y_bins_axis=y_bins,
            )
        )

    return output_results
