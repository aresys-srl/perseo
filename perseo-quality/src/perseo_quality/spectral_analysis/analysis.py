# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Spectral Analysis: Distributed and Point Target Spectra"""

from __future__ import annotations

import numpy as np
from arepytools.geometry.inverse_geocoding_core import inverse_geocoding_monostatic_core
from scipy.fft import fft2, fftshift

from perseo_quality.core.common import blocks_partitioning, check_targets_visibility, detect_burst_from_pixel
from perseo_quality.core.custom_errors import AzimuthExceedsBoundariesError, RangeExceedsBoundariesError
from perseo_quality.core.generic_dataclasses import SARAcquisitionMode, SARCoordinates
from perseo_quality.core.signal_processing import convert_to_db
from perseo_quality.io.point_targets import PointTarget
from perseo_quality.io.quality_input_protocol import QualityInputProduct
from perseo_quality.logger import quality_logger as log
from perseo_quality.spectral_analysis.custom_dataclasses import (
    DistributedSpectraDataOutput,
    PointTargetSpectraDataOutput,
    SpectralAnalysisBlockInfo,
    SpectralAnalysisTargetInfo,
)
from perseo_quality.spectral_analysis.support import (
    compute_polynomial_fit,
    compute_spectrogram_db,
    data_deramping,
    extract_abs_profiles,
    extract_phase_profiles,
    recenter_data,
)


def point_target_spectral_analysis(
    product: QualityInputProduct,
    point_targets: list[PointTarget],
    cropping_size: tuple[int, int] = (128, 128),
) -> list[PointTargetSpectraDataOutput]:
    """Function to compute Spectral Analysis for selected Point Targets.

    Parameters
    ----------
    product : QualityInputProduct
        object satisfying the QualityInputProduct protocol
    point_targets : list[PointTarget]
        list of point targets locations, as PointTarget objects
    cropping_size : tuple[int, int], optional
        roi cropping size, (number of samples, number of lines), by default (128, 128)

    Returns
    -------
    list[PointTargetSpectraDataOutput]
        spectral analysis results for each product channel and each Point Target
    """

    log.info(f"Performing Point Target Spectral Analysis on {product.name}")
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
        log.info(f"Channel Data acquisition mode is: {channel_data.acquisition_mode.name}")

        # recovering only targets visible by this channel
        targets_visible_by_channel = visible_targets[visible_targets["channel"] == channel]["id"]

        output_results = PointTargetSpectraDataOutput(
            product_name=product.name,
            channel=channel,
            swath=channel_data.swath_name,
            polarization=channel_data.polarization,
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

                point_target_location_px = (
                    np.round(az_rng_coords.azimuth_index_subpx).astype("int64"),
                    np.round(az_rng_coords.range_index_subpx).astype("int64"),
                )
                try:
                    data = channel_data.read_data(
                        azimuth_index=point_target_location_px[0],
                        range_index=point_target_location_px[1],
                        cropping_size=cropping_size,
                        output_radiometric_quantity=channel_data.radiometric_quantity,
                        burst=burst,
                    )
                except (RangeExceedsBoundariesError, AzimuthExceedsBoundariesError):
                    log.warning("Target ROI exceeds burst or swath boundaries")
                    res.append(output_results)
                    continue

                if channel_data.acquisition_mode in (SARAcquisitionMode.TOPSAR, SARAcquisitionMode.SCANSAR):
                    # deramping data for topsar and scansar products
                    log.info("Performing data deramping...")
                    data = data_deramping(
                        data=data.copy(),
                        channel_data=channel_data,
                        burst=burst,
                        roi_center_location_px=point_target_location_px,
                    )

                data_recentered = recenter_data(data.copy())
                data_fft = fftshift(fft2(data_recentered))
                azimuth_frequency_axis = np.linspace(-0.5, 0.5, data_fft.shape[1])
                range_frequency_axis = np.linspace(-0.5, 0.5, data_fft.shape[0])
                # absolute (dB) profiles and data
                range_profiles_db, azimuth_profiles_db = extract_abs_profiles(data_fft)
                spectrogram_db, spectrogram_frequencies, spectrogram_times = compute_spectrogram_db(data)
                # phase (deg) profiles and data
                range_profiles_deg, azimuth_profiles_deg = extract_phase_profiles(data_fft)
                # only for the central profile, the one passing through the point target
                range_polynomial_fit = compute_polynomial_fit(
                    profile=range_profiles_deg[1], freq_axis=range_frequency_axis
                )
                azimuth_polynomial_fit = compute_polynomial_fit(
                    profile=azimuth_profiles_deg[1], freq_axis=azimuth_frequency_axis
                )
                targets_info.append(
                    SpectralAnalysisTargetInfo(
                        target_name=trgt,
                        burst=burst,
                        roi_size_azimuth=cropping_size[1],
                        roi_size_range=cropping_size[0],
                        azimuth_time=trgt_az_time,
                        range_time=trgt_rng_time,
                        target_azimuth_pixel=az_rng_coords.azimuth_index_subpx,
                        target_range_pixel=az_rng_coords.range_index_subpx,
                        azimuth_frequency_axis=azimuth_frequency_axis,
                        range_frequency_axis=range_frequency_axis,
                        spectrum_db=convert_to_db(np.abs(data_fft) ** 2),
                        range_profiles_db=range_profiles_db,
                        azimuth_profiles_db=azimuth_profiles_db,
                        spectrogram_db=spectrogram_db,
                        spectrogram_frequencies=spectrogram_frequencies,
                        spectrogram_times=spectrogram_times,
                        spectrum_deg=np.angle(data_fft, deg=True),
                        range_profiles_deg=range_profiles_deg,
                        azimuth_profiles_deg=azimuth_profiles_deg,
                        range_polynomial_fit=range_polynomial_fit,
                        azimuth_polynomial_fit=azimuth_polynomial_fit,
                    )
                )

        output_results.targets_info = targets_info
        res.append(output_results)

    return res


def block_wise_distributed_spectral_analysis(
    product: QualityInputProduct,
    azimuth_block_size: int = 2000,
) -> list[DistributedSpectraDataOutput]:
    """Compute Spectral Analysis for each channel of the input product raster, by partitioning the scene in blocks along
    azimuth. If the product's acquisition mode, bursts are considered as blocks and data is deramped.

    Parameters
    ----------
    product : QualityInputProduct
        object satisfying the QualityInputProduct protocol
    azimuth_block_size : int, optional
        azimuth block size, by default 2000

    Returns
    -------
    list[DistributedSpectraDataOutput]
        list of Spectral Data for every product channel and every block
    """

    log.info(f"Performing Block-Wise Distributed Target Spectral Analysis on {product.name}")
    log.info(f"Selected Product has {len(product.channels_list)} channels")

    res = []
    for channel in product.channels_list:
        # recovering metadata for the current channel
        channel_data = product.get_channel_data(channel_id=channel)
        log.info(
            f"Analyzing Channel {channel}, Swath {channel_data.swath_name},"
            + f" Polarization {channel_data.polarization.name}..."
        )
        log.info(f"Channel Data acquisition mode is: {channel_data.acquisition_mode.name}")

        log.info("Defining blocks partitioning of the whole scene.")
        # defining scene partitioning by blocks
        az_block_size, blocks_num, blocks_centers_px = blocks_partitioning(
            azimuth_axis=channel_data.azimuth_axis,
            range_axis=channel_data.slant_range_axis,
            lines_per_burst=channel_data.lines_per_burst,
            default_block_size=azimuth_block_size,
        )

        output_results = DistributedSpectraDataOutput(
            product_name=product.name,
            channel=channel,
            swath=channel_data.swath_name,
            polarization=channel_data.polarization,
        )

        blocks_info = []
        for bc_num, center in enumerate(blocks_centers_px):
            log.info(f"Processing block {bc_num + 1} of {blocks_num}")

            cropping_size = (
                channel_data.slant_range_axis.size,
                az_block_size[bc_num],
            )

            burst = detect_burst_from_pixel(lines_per_burst=channel_data.lines_per_burst, azimuth_px=center[0])
            # reading block, without converting the data here, applying radiometric conversion in the next steps
            target_area = channel_data.read_data(
                azimuth_index=center[0],
                range_index=center[1],
                cropping_size=cropping_size,
                output_radiometric_quantity=channel_data.radiometric_quantity,
            )

            if channel_data.acquisition_mode in (SARAcquisitionMode.TOPSAR, SARAcquisitionMode.SCANSAR):
                # deramping data for topsar and scansar products
                log.info("Performing data deramping...")
                target_area = data_deramping(
                    data=target_area.copy(),
                    channel_data=channel_data,
                    burst=burst,
                    roi_center_location_px=center,
                )
            data_fft = fftshift(fft2(target_area))
            range_profiles_db, azimuth_profiles_db = extract_abs_profiles(data_fft)
            spectrogram_db, spectrogram_frequencies, spectrogram_times = compute_spectrogram_db(target_area)
            blocks_info.append(
                SpectralAnalysisBlockInfo(
                    block_num=bc_num,
                    first_az_line_block=int(center[0] - np.floor(cropping_size[1] / 2)),
                    lines_block=target_area.shape[1],
                    samples_block=target_area.shape[0],
                    azimuth_frequency_axis=np.linspace(-0.5, 0.5, data_fft.shape[1]),
                    range_frequency_axis=np.linspace(-0.5, 0.5, data_fft.shape[0]),
                    spectrum_db=convert_to_db(np.abs(data_fft) ** 2),
                    range_profiles_db=range_profiles_db,
                    azimuth_profiles_db=azimuth_profiles_db,
                    spectrogram_db=spectrogram_db,
                    spectrogram_frequencies=spectrogram_frequencies,
                    spectrogram_times=spectrogram_times,
                )
            )

        output_results.blocks_info = blocks_info
        res.append(output_results)

    return res
