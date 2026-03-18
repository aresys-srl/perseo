# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Point Target Analysis Pre Processing computation"""

from __future__ import annotations

import numpy as np

import perseo_quality.core.signal_processing as sp
from perseo_quality.core.custom_errors import (
    AzimuthExceedsBoundariesError,
    RangeExceedsBoundariesError,
    TargetAreaRecenteringError,
)
from perseo_quality.core.generic_dataclasses import SARCoordinates, TargetDataType
from perseo_quality.core.masking_operations import get_interpolated_lobes_cuts
from perseo_quality.io.quality_input_protocol import ChannelData
from perseo_quality.logger import quality_logger as log
from perseo_quality.point_targets_analysis.support import SideLobesDirections


def detect_data_type(target_area: np.ndarray) -> TargetDataType:
    """Detecting target data type from the raster content.

    Parameters
    ----------
    target_area : np.ndarray
        raster content

    Returns
    -------
    TargetDataType
        target data type of input raster data
    """

    if np.isrealobj(target_area):
        return TargetDataType.DETECTED

    return TargetDataType.COMPLEX


def compute_data_resolution_pixel(
    recentered_target_area_interp: np.ndarray,
    data_type: TargetDataType,
    side_lobes_directions: SideLobesDirections,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Compute data resolution in pixels along range and azimuth directions.

    Parameters
    ----------
    recentered_target_area_interp : np.ndarray
        target area centered on point target signal peak, interpolated for higher resolution, with shape (n_rng, naz)
    data_type : TargetDataType
        target data type
    side_lobes_directions : SideLobesDirections
        range and azimuth cuts angular coefficients in samples

    Returns
    -------
    np.ndarray
        range profile
    np.ndarray
        azimuth profile
    float
        range resolution in pixels
    float
        azimuth resolution in pixels
    """
    # extracting range and azimuth profiles taking into account lobes
    range_profile, azimuth_profile = _extract_profiles(
        target=recentered_target_area_interp,
        side_lobes_directions=side_lobes_directions,
    )

    if data_type == TargetDataType.DETECTED:
        # root squaring the values if data is real
        range_profile = np.sqrt(range_profile)
        azimuth_profile = np.sqrt(azimuth_profile)

    # Resolution: evaluate spatial resolution in both directions in pixels
    rng_resolution = sp.evaluate_irf_resolution(range_profile)
    az_resolution = sp.evaluate_irf_resolution(azimuth_profile)

    return range_profile, azimuth_profile, rng_resolution, az_resolution


def target_area_interpolation(
    target_area: np.ndarray, target_pos_real: tuple[float, float], oversampling_factor: int, roi: np.ndarray
) -> np.ndarray:
    """Recenter over the target position and oversample the target area.

    Target area is recentered over the target position. Then it is cropped to the size given by roi and
    finally oversampled by the given factor.

    Parameters
    ----------
    target_area : np.ndarray
        input data with shape (N, M)
    target_pos_real : tuple[float, float]
        position of the signal peak in pixels as fractional index
    oversampling_factor : int
        oversampling factor
    roi : np.ndarray
        crop size (n, m)

    Returns
    -------
    np.ndarray
        oversampled image, with shape (n * oversampling_factor, m * oversampling_factor)

    Raises
    ------
    TargetAreaRecenteringError
        if target position is invalid
    """

    if detect_data_type(target_area=target_area) == TargetDataType.DETECTED:
        target_area = target_area**2

    try:
        target_area_recentered = _recenter_data(data=target_area, center=target_pos_real)
    except ValueError as err:
        log.error("Wrong center value for target area re-centering")
        raise TargetAreaRecenteringError from err

    # cropping target area around center
    target_area_recentered = sp.crop_array_2d(data=target_area_recentered, crop_size=(roi[0], roi[1]))

    return sp.interp2_modulated_data(
        data=target_area_recentered,
        interp_factor_az=oversampling_factor,
        interp_factor_rng=oversampling_factor,
        demod_flag_az=True,
        demod_flag_rng=True,
    )


def compute_roi(data_shape: tuple[int, int], oversampling_factor: int) -> np.ndarray:
    """Computing the region of interest along range and azimuth.

    Parameters
    ----------
    data_shape : tuple[int, int]
        shape of the target area
    oversampling_factor : int
        oversampling factor

    Returns
    -------
    np.ndarray
        region of interest, range_px[0] and azimuth_px[1]
    """

    area_size_ratio = data_shape[1] / data_shape[0]
    return 2 * oversampling_factor * np.array([1, np.round(area_size_ratio).astype("int64")])


def _recenter_data(data: np.ndarray, center: tuple[float, float]) -> np.ndarray:
    """Shift the data to given center.

    Parameters
    ----------
    data : np.ndarray
        2D array to be shifted
    center : tuple[float, float]
        center position range[0], azimuth[1]

    Returns
    -------
    np.ndarray
        shifted 2D array
    """

    rng_shift = center[0] - int(data.shape[0] // 2)
    az_shift = center[1] - int(data.shape[1] // 2)

    return sp.shift_array(data=data, row_shift=rng_shift, col_shift=az_shift)


def _extract_profiles(
    target: np.ndarray,
    side_lobes_directions: SideLobesDirections,
) -> tuple[np.ndarray, np.ndarray]:
    """Extracting range and azimuth profiles taking into account side lobes directions

    Parameters
    ----------
    target : np.ndarray
        input 2D array
    side_lobes_directions : SideLobesDirections
        range and azimuth cuts angular coefficients in samples

    Returns
    -------
    np.ndarray
        range profile cut
    np.ndarray
        azimuth profile cut
    """
    roi = np.array(target.shape)
    roi_center = roi // 2

    irf_rg_axis = generate_irf_axis(stop=roi[0], offset=roi_center[0], scaling=1)
    irf_az_axis = generate_irf_axis(stop=roi[1], offset=roi_center[1], scaling=1)

    # checking side_lobes_directions value, if given
    if np.isinf(side_lobes_directions[0]):
        rng_profile = target[:, roi_center[1]].copy()
        az_profile = target[roi_center[0], :].copy()

    else:
        rng_profile, az_profile = get_interpolated_lobes_cuts(
            x_axis=irf_az_axis,
            y_axis=irf_rg_axis,
            values=target,
            side_lobes_directions=side_lobes_directions,
        )

    rng_profile = rng_profile / np.max(np.abs(rng_profile))
    az_profile = az_profile / np.max(np.abs(az_profile))

    return rng_profile, az_profile


def generate_irf_axis(stop: float, offset: float, scaling: float) -> np.ndarray:
    """Generate IRF axis from input parameters.

    Parameters
    ----------
    stop : float
        axis stop
    offset : float
        axis offset
    scaling : float
        axis scaling

    Returns
    -------
    np.ndarray
        irf axis array
    """
    return (np.arange(0, stop) - offset) * scaling


def extract_target_area(
    channel_data: ChannelData,
    azimuth_range_coordinates: SARCoordinates,
    ale_limits: tuple[float, float] | None = None,
    initial_crop: tuple[int, int] = (33, 33),
    final_crop: tuple[int, int] = (128, 128),
    ovrs_factor: int = 5,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Extract a portion of the swath around target area from input product.

    Parameters
    ----------
    channel_data : ChannelData
        product channel data instance
    azimuth_range_coordinates : dtc.SARCoordinates
        azimuth and range coordinates SARCoordinates dataclass
    ale_limits : tuple[float, float] | None, optional
        absolute localization error limits, by default None
    initial_crop : tuple[int, int], optional
        first step roi boundaries (range, azimuth), by default (33, 33)
    final_crop : tuple[int, int], optional
        final step roi boundaries (range, azimuth), by default (128, 128)
    ovrs_factor : int, optional
        oversampling factor (arbitrarily chosen)

    Returns
    -------
    np.ndarray | None
        cropped target area centered on interpolated peak coordinates
    np.ndarray | None
        peak coordinates
    np.ndarray | None
        nominal target reference coordinates in this frame [row, col]
    np.ndarray | None
        main lobe peak coordinates (rng[0], az[1]) in pixels referred to the whole swath
    """

    # managing ale limits
    if ale_limits is not None:
        initial_crop = tuple(int(ale) for ale in ale_limits)
        log.info(f"External Maximum ALE limits provided: using {initial_crop} ROI for peak searching")

    # first cropping around target nominal position
    try:
        log.debug(f"Cropping target area around nominal target position: size {initial_crop}")
        target_area = channel_data.read_data(
            azimuth_index=np.round(azimuth_range_coordinates.azimuth_index_subpx).astype("int64"),
            range_index=np.round(azimuth_range_coordinates.range_index_subpx).astype("int64"),
            cropping_size=initial_crop,
            burst=azimuth_range_coordinates.burst,
        )
    except (AzimuthExceedsBoundariesError, RangeExceedsBoundariesError) as err:
        log.warning(err)
        return None, None, None, None

    # locating real peak position in cropped image
    log.debug("Locating signal peak position")
    if np.isrealobj(target_area):
        _, peak_range_im, peak_azimuth_im = sp.locate_max_2d_interp(data=np.abs(target_area) ** 2)
    else:
        _, peak_range_im, peak_azimuth_im = sp.locate_max_2d_interp(data=target_area)

    if np.isnan(peak_range_im) or np.isnan(peak_azimuth_im):
        log.warning("Could not find peak of the target area")
        return None, None, None, None

    # evaluating distance between nominal target and peak
    delta_rng_trgt_pk = peak_range_im - target_area.shape[0] // 2
    delta_az_trgt_pk = peak_azimuth_im - target_area.shape[1] // 2

    # if peak outside of ALE distance, break
    if ale_limits is not None:
        ale_break_cond = np.logical_or(
            np.abs(delta_rng_trgt_pk) > ale_limits[0], np.abs(delta_az_trgt_pk) > ale_limits[1]
        )
        if ale_break_cond:
            log.warning("Target not within ALE limits")
            return None, None, None, None

    # second cropping, centered on peak coordinates
    peak_coords_swath = np.array(
        (
            np.round(azimuth_range_coordinates.range_index_subpx) - np.round(initial_crop[0] / 2) + peak_range_im,
            np.round(azimuth_range_coordinates.azimuth_index_subpx) - np.round(initial_crop[1] / 2) + peak_azimuth_im,
        )
    )
    peak_az_index = np.round(
        np.round(azimuth_range_coordinates.azimuth_index_subpx)
        - np.floor(initial_crop[1] / 2)
        + np.floor(peak_azimuth_im)
    )
    peak_rng_index = np.round(
        np.round(azimuth_range_coordinates.range_index_subpx) - np.floor(initial_crop[0] / 2) + np.floor(peak_range_im)
    )

    # final cropping around peak position
    try:
        log.debug(f"Cropping target area around signal peak position: size {final_crop}")
        try:
            target_area = channel_data.read_data(
                azimuth_index=int(peak_az_index),
                range_index=int(peak_rng_index),
                cropping_size=final_crop,
                burst=azimuth_range_coordinates.burst,
            )
        except (AzimuthExceedsBoundariesError, RangeExceedsBoundariesError):
            log.warning(f"Extracted ROI exceeds burst boundaries, trying a smaller roi {initial_crop}")
            target_area = channel_data.read_data(
                azimuth_index=int(peak_az_index),
                range_index=int(peak_rng_index),
                cropping_size=initial_crop,
                burst=azimuth_range_coordinates.burst,
            )
            log.debug(f"Cropping target area around signal peak position: size {initial_crop}")
    except (AzimuthExceedsBoundariesError, RangeExceedsBoundariesError) as err:
        log.warning(err)
        return None, None, None, None

    # checking for other conditions
    rng_ovrs = 1
    az_ovrs = 1

    if channel_data.sampling_constants is not None:
        try:
            rng_ovrs = np.max(
                [
                    np.round(
                        channel_data.sampling_constants.range_freq_hz
                        / channel_data.sampling_constants.range_bandwidth_freq_hz
                        / ovrs_factor
                    ),
                    1,
                ]
            )
            az_ovrs = np.max(
                [
                    np.round(
                        channel_data.sampling_constants.azimuth_freq_hz
                        / channel_data.sampling_constants.azimuth_bandwidth_freq_hz
                        / ovrs_factor
                    ),
                    1,
                ]
            )
        except ZeroDivisionError:
            rng_ovrs = 1
            az_ovrs = 1

    if rng_ovrs > 1 or az_ovrs > 1:
        try:
            target_area = channel_data.read_data(
                azimuth_index=int(peak_az_index - final_crop[1] * (az_ovrs - 1) / 2),
                range_index=int(peak_rng_index - final_crop[0] * (rng_ovrs - 1) / 2),
                cropping_size=(
                    np.round(final_crop[0] * rng_ovrs).astype("int64"),
                    np.round(final_crop[1] * az_ovrs).astype("int64"),
                ),
            )
        except (AzimuthExceedsBoundariesError, RangeExceedsBoundariesError) as err:
            log.warning(err)
            return None, None, None, None

    peak_coordinates = np.array(
        [
            target_area.shape[0] // 2 + peak_range_im - np.floor(peak_range_im),
            target_area.shape[1] // 2 + peak_azimuth_im - np.floor(peak_azimuth_im),
        ]
    )

    nom_rng = azimuth_range_coordinates.range_index_subpx - (
        np.round(peak_rng_index).astype("int64") - target_area.shape[0] // 2
    )
    nom_az = azimuth_range_coordinates.azimuth_index_subpx - (
        np.round(peak_az_index).astype("int64") - target_area.shape[1] // 2
    )
    nominal_coordinates = np.array([nom_rng, nom_az])

    return target_area, peak_coordinates, nominal_coordinates, peak_coords_swath
