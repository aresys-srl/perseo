# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Spectral Analysis: Distributed and Point Target Spectra"""

from __future__ import annotations

import numpy as np
from arepytools.geometry.inverse_geocoding_core import inverse_geocoding_monostatic_core
from scipy.fft import fft2, fftshift

from perseo_quality.core.common import check_targets_visibility
from perseo_quality.core.custom_errors import AzimuthExceedsBoundariesError, RangeExceedsBoundariesError
from perseo_quality.core.generic_dataclasses import SARAcquisitionMode, SARCoordinates
from perseo_quality.core.signal_processing import convert_to_db
from perseo_quality.io.point_targets import PointTarget
from perseo_quality.io.quality_input_protocol import QualityInputProduct
from perseo_quality.logger import quality_logger as log
from perseo_quality.spectral_analysis.custom_dataclasses import SpectraDataOutput
from perseo_quality.spectral_analysis.support import (
    compute_polynomial_fit,
    compute_spectrogram_db,
    data_deramping,
    detect_burst_from_pixel,
    extract_abs_profiles,
    extract_phase_profiles,
    recenter_data,
)


def point_target_spectral_analysis(
    product: QualityInputProduct,
    point_targets: list[PointTarget],
    cropping_size: tuple[int, int] = (128, 128),
) -> list[SpectraDataOutput]:
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
    list[SpectraDataOutput]
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
                output_results = SpectraDataOutput(
                    target_name=trgt,
                    product_name=product.name,
                    channel=channel,
                    swath=channel_data.swath_name,
                    burst=burst,
                    polarization=channel_data.polarization,
                    roi_size_azimuth=cropping_size[1],
                    roi_size_range=cropping_size[0],
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
                output_results.azimuth_time = trgt_az_time
                output_results.range_time = trgt_rng_time

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
                output_results.azimuth_frequency_axis = np.linspace(-0.5, 0.5, data_fft.shape[1])
                output_results.range_frequency_axis = np.linspace(-0.5, 0.5, data_fft.shape[0])
                # absolute (dB) profiles and data
                output_results.spectrum_db = convert_to_db(np.abs(data_fft) ** 2)
                output_results.range_profiles_db, output_results.azimuth_profiles_db = extract_abs_profiles(data_fft)
                (
                    output_results.spectrogram_db,
                    output_results.spectrogram_frequencies,
                    output_results.spectrogram_times,
                ) = compute_spectrogram_db(data)
                # phase (deg) profiles and data
                output_results.spectrum_deg = np.angle(data_fft, deg=True)
                output_results.range_profiles_deg, output_results.azimuth_profiles_deg = extract_phase_profiles(
                    data_fft
                )
                # only for the central profile, the one passing through the point target
                output_results.range_polynomial_fit = compute_polynomial_fit(
                    profile=output_results.range_profiles_deg[1], freq_axis=output_results.range_frequency_axis
                )
                output_results.azimuth_polynomial_fit = compute_polynomial_fit(
                    profile=output_results.azimuth_profiles_deg[1], freq_axis=output_results.azimuth_frequency_axis
                )
                res.append(output_results)

    return res


def distributed_target_spectral_analysis(
    product: QualityInputProduct, roi_centers: list[tuple[int, int]], cropping_size: tuple[int, int] = (128, 128)
) -> list[SpectraDataOutput]:
    """Compute Spectral Analysis for each channel the input product raster at the selected location.

    Parameters
    ----------
    product : QualityInputProduct
        object satisfying the QualityInputProduct protocol
    roi_centers : list[tuple[int, int]]
        roi centers pixel coordinates where to compute the analysis, (range pixel index, azimuth pixel index)
    cropping_size : tuple[int, int], optional
        roi cropping size, (number of samples, number of lines), by default (128, 128)

    Returns
    -------
    list[SpectraDataOutput]
        list of Spectral Data for every product channel and every selected roi
    """

    log.info(f"Performing Distributed Target Spectral Analysis on {product.name}")
    log.info(f"Selected Product has {len(product.channels_list)} channels")
    log.info(f"ROI cropping size provided: azimuth {cropping_size[1]}, range {cropping_size[0]}")

    res = []
    for channel in product.channels_list:
        # recovering metadata for the current channel
        channel_data = product.get_channel_data(channel_id=channel)
        log.info(
            f"Analyzing Channel {channel}, Swath {channel_data.swath_name},"
            + f" Polarization {channel_data.polarization.name}..."
        )
        log.info(f"Channel Data acquisition mode is: {channel_data.acquisition_mode.name}")
        for roi_id, roi in enumerate(roi_centers):
            log.info(f"Processing ROI {roi_id + 1}/{len(roi_centers)}")
            log.info(f"ROI center: azimuth {roi[1]}, range {roi[0]}")
            burst = detect_burst_from_pixel(lines_per_burst=channel_data.lines_per_burst, azimuth_px=roi[1])
            azimuth_time, range_time = channel_data.pixel_to_times_conversion(
                azimuth_index=roi[1], range_index=roi[0], burst=burst
            )
            output_results = SpectraDataOutput(
                target_name=roi_id,
                product_name=product.name,
                channel=channel,
                swath=channel_data.swath_name,
                burst=burst,
                polarization=channel_data.polarization,
                roi_size_azimuth=cropping_size[1],
                roi_size_range=cropping_size[0],
                target_azimuth_pixel=roi[1],
                target_range_pixel=roi[0],
                azimuth_time=azimuth_time,
                range_time=range_time,
            )
            # processing target area
            data = channel_data.read_data(
                azimuth_index=roi[1],
                range_index=roi[0],
                cropping_size=cropping_size,
                output_radiometric_quantity=channel_data.radiometric_quantity,
            )
            if channel_data.acquisition_mode in (SARAcquisitionMode.TOPSAR, SARAcquisitionMode.SCANSAR):
                # deramping data for topsar and scansar products
                log.info("Performing data deramping...")
                data = data_deramping(
                    data=data.copy(),
                    channel_data=channel_data,
                    burst=burst,
                    roi_center_location_px=(roi[1], roi[0]),
                )
            data_fft = fftshift(fft2(data))
            output_results.spectrum_db = convert_to_db(np.abs(data_fft) ** 2)
            output_results.azimuth_frequency_axis = np.linspace(-0.5, 0.5, data_fft.shape[1])
            output_results.range_frequency_axis = np.linspace(-0.5, 0.5, data_fft.shape[0])
            output_results.range_profiles_db, output_results.azimuth_profiles_db = extract_abs_profiles(data_fft)
            output_results.spectrogram_db, output_results.spectrogram_frequencies, output_results.spectrogram_times = (
                compute_spectrogram_db(data)
            )
            res.append(output_results)

    return res
