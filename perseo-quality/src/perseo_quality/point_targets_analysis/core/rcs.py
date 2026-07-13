# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Point Target Radar Cross Section computation"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from scipy import ndimage
from scipy.constants import speed_of_light as LIGHT_SPEED

import perseo_quality.core.signal_processing as sp
from perseo_quality.core.generic_dataclasses import (
    RCSComputationMethod,
    SARPolarization,
    TargetDataType,
)
from perseo_quality.core.masking_operations import (
    generate_box_mask,
    generate_cross_mask,
)
from perseo_quality.io.point_targets import PointTarget
from perseo_quality.logger import quality_logger as log
from perseo_quality.point_targets_analysis.core.pre_processing import (
    detect_data_type,
)
from perseo_quality.point_targets_analysis.custom_dataclasses import RCSDataOutput
from perseo_quality.point_targets_analysis.custom_errors import (
    PointTargetComputationError,
)


def compute_point_target_rcs(
    target_area: np.ndarray,
    range_resolution_px: float,
    azimuth_resolution_px: float,
    target_pos_real: npt.NDArray[np.floating],
    rcs_interp_factor: int,
    rcs_roi: npt.NDArray[np.floating],
    k_lin: float = 1,
    s_f: float = 1,
    method: RCSComputationMethod = RCSComputationMethod.BOXES,
) -> tuple[RCSDataOutput, np.ndarray, list, list | np.ndarray]:
    """Compute the Radar Cross-Section (RCS) from target acquisition data. Input data is considered: beta-nought,
    radiometrically corrected, absolutely calibrated (if k_lin=1) and not resampled (if s_f=1).

    Parameters
    ----------
    target_area : np.ndarray
        target area where to compute the radar cross section, with shape (n_rng, n_az)
    range_resolution_px : float
        range resolution in pixels
    azimuth_resolution_px : float
        azimuth resolution in pixels
    target_pos_real : npt.NDArray[np.floating]
        position of the signal peak in pixels, range_px[0] and azimuth_px[1]
    rcs_interp_factor : int
        rcs interpolation factor
    rcs_roi : npt.NDArray[np.floating]
        region of interest for RCS computation
    k_lin : float, optional
        a value of 1 means absolutely calibrated, by default 1
    s_f : float, optional
        a value of 1 means not resampled, by default 1
    method : RCSComputationMethod, optional
        RCS computation method, by default RCSComputationMethod.BOXES

    Returns
    -------
    RCSDataOutput
        dataclass object containing all computed export variables
    np.ndarray
        roi target data
    list
        list of peak roi corner pixels
    list | np.ndarray
        background intensity corners, a list of pixels for each square corner region, if method is BOXES,
        else cross mask used for RCS computation
    """

    target = target_area.copy()
    data_type = detect_data_type(target_area=target)
    irf_resolution_px = (range_resolution_px, azimuth_resolution_px)

    if data_type == TargetDataType.DETECTED:
        target = target**2

    # selecting a roi centered on target area peak value
    max_row, max_col, roi_target = _roi_extraction(data=target, roi=rcs_roi, target_pos=target_pos_real)

    # computing intensity of target area. if data are real numbers, it has
    # already been converted into intensity before
    if data_type == TargetDataType.DETECTED:
        target_intensity = roi_target.copy()
    else:
        target_intensity = np.abs(roi_target) ** 2

    # interpolate the corrected data intensity
    target_intensity_interp = sp.interp2_modulated_data(
        data=roi_target,
        interp_factor_az=rcs_interp_factor,
        interp_factor_rng=rcs_interp_factor,
        demod_flag_az=True,
        demod_flag_rng=True,
    )

    if data_type == TargetDataType.DETECTED:
        target_intensity_interp = np.sqrt(target_intensity_interp)

    match method:
        case RCSComputationMethod.BOXES:
            results, central_box_corners, roi_background_corners = compute_rcs_with_boxes_method(
                data=target_intensity,
                data_interp=target_intensity_interp,
                target_pos_real=target_pos_real,
                max_position=(max_row, max_col),
                resolutions_px=irf_resolution_px,
                rcs_roi=rcs_roi,
                interp_factor=rcs_interp_factor,
                k_lin=k_lin,
                s_f=s_f,
            )
        case RCSComputationMethod.CROSS:
            roi_background_corners = None
            results, central_box_corners, boundary_mask = compute_rcs_with_cross_method(
                data=target_intensity,
                data_interp=target_intensity_interp,
                target_pos_real=target_pos_real,
                max_position=(max_row, max_col),
                resolutions_px=irf_resolution_px,
                interp_factor=rcs_interp_factor,
                k_lin=k_lin,
                s_f=s_f,
            )
    return (
        results,
        roi_target,
        central_box_corners,
        roi_background_corners if roi_background_corners is not None else boundary_mask,
    )


def compute_additional_rcs_values(
    rcs_input: RCSDataOutput,
    step_distances: list,
    interp_factor: int,
    polarization: SARPolarization,
    target_info: PointTarget,
    sensor_position: npt.NDArray[np.floating],
    fc_hz: float,
) -> tuple[float, float, float, float]:
    """Adjust rcs output values and calculate peak phase error.

    Parameters
    ----------
    rcs_input : ptdt.RCSDataOutput
        rcs output values from PointTargetIRF object
    step_distances : list
        step distances [range, azimuth]
    interp_factor : int
        rcs interpolation factor
    polarization : EPolarization
        polarization value [V/V, H/H, H/V, V/H]
    target_info : PointTarget
        target info as PointTarget
    sensor_position : npt.NDArray[np.floating]
        satellite position at given azimuth time
    fc_hz : float
        carrier frequency

    Returns
    -------
    tuple[float, float, float, float]
        rcs linear,
        rcs [db],
        rcs error [db],
        peak phase error [deg]
    """
    # convert rcs from intensity per unit pixel area to decibel
    rcs_pixel_area = np.prod(step_distances) / interp_factor**2
    rcs = rcs_pixel_area * rcs_input.rcs
    rcs_db = sp.convert_to_db(rcs)

    # selecting the right point target rcs reference value based on polarization
    if polarization == SARPolarization.HH:
        ptrcs = target_info.rcs_hh
    elif polarization == SARPolarization.HV:
        ptrcs = target_info.rcs_hv
    elif polarization == SARPolarization.VV:
        ptrcs = target_info.rcs_vv
    elif polarization == SARPolarization.VH:
        ptrcs = target_info.rcs_vh

    # evaluating RCS Error and Peak Phase Error
    arg = math.dist(sensor_position, target_info.xyz_coordinates) / (LIGHT_SPEED / fc_hz)
    peak_phase_error = np.angle(rcs_input.peak_value_complex * np.exp(1j * 4 * np.pi * arg), deg=True)
    ptrcs_db = sp.convert_to_db(abs(ptrcs)) if np.iscomplexobj(ptrcs) else ptrcs
    rcs_error = rcs_db - ptrcs_db

    return rcs, rcs_db, rcs_error, peak_phase_error


def _roi_extraction(
    data: np.ndarray,
    roi: npt.NDArray[np.floating],
    target_pos: npt.NDArray[np.floating] = None,
) -> tuple[int, int, np.ndarray]:
    """Extraction of a roi from the input array.

    Parameters
    ----------
    data : np.ndarray
        input array
    roi : npt.NDArray[np.floating]
        roi_size [row number, col number]
    target_pos : npt.NDArray[np.floating], optional
        position of the target peak. If None, it is calculated from input array, by default None

    Returns
    -------
    int
        row max index
    int
        column max index
    np.ndarray
        roi extracted from input array
    """

    if target_pos is None:
        max_row, max_col = sp.locate_max_2d(np.abs(data))
    else:
        max_row, max_col = np.floor(target_pos).astype("int64")

    # defining roi index boundaries
    row_lim_up = max_row - roi[0] // 2
    row_lim_dwn = max_row + roi[0] // 2
    col_lim_sx = max_col - roi[1] // 2
    col_lim_dx = max_col + roi[1] // 2

    # checking if roi exits array boundaries
    break_cond = np.logical_or.reduce(
        (
            row_lim_up < 0,
            row_lim_dwn > data.shape[0],
            col_lim_sx < 0,
            col_lim_dx > data.shape[1],
        )
    )
    if break_cond:
        log.warning("Could not evaluate RCS: extracted target area is too small")
        raise PointTargetComputationError

    roi_target = data[row_lim_up:row_lim_dwn, col_lim_sx:col_lim_dx].copy()

    return max_row, max_col, roi_target


def compute_rcs_with_boxes_method(
    data: np.ndarray,
    data_interp: np.ndarray,
    target_pos_real: npt.NDArray[np.floating],
    max_position: tuple[int, int],
    resolutions_px: tuple[float, float],
    rcs_roi: np.ndarray,
    interp_factor: int,
    k_lin: float,
    s_f: float,
    margin: int = 20,
) -> tuple[RCSDataOutput, list, list]:
    """Computing RCS using the classical corner boxes algorithm, with 4 boxes at each corner of the ROI to estimate the
    clutter.

    Parameters
    ----------
    data : np.ndarray
        target data intensity
    data_interp : np.ndarray
        interpolated target data intensity
    target_pos_real : npt.NDArray[np.floating]
        position of the target
    max_position : tuple[int, int]
        position of the max inside the target area (data)
    resolutions_px : tuple[float, float]
        irf resolutions in pixels, range[0] and azimuth[1]
    rcs_roi : np.ndarray
        roi to be used for the rcs computation
    interp_factor : int
        interpolation factor
    k_lin : float
        a value of 1 means absolutely calibrated
    s_f : float
        a value of 1 means not resampled
    margin : int, optional
        margin for the peak intensity computation, by default 20

    Returns
    -------
    RCSDataOutput
        output results for RCS computation
    list
        list of peak roi corner pixels
    list
        background intensity corners, a list of pixels for each square corner region
    """
    # initializing output structure
    results = RCSDataOutput()
    intensity_background, roi_background_corners = sp.compute_intensity_background(
        data=data, resolutions_px=resolutions_px, roi=rcs_roi
    )

    # correcting interpolated intensity data subtracting background intensity
    target_interp_intens_corr = np.abs(data_interp) ** 2 - intensity_background

    # integrate the interpolated corrected data intensity on peak region
    integrated_peak_intensity, interp_peak_position, peak_corners = sp.compute_integrated_peak_intensity(
        data=target_interp_intens_corr,
        max_position=max_position,
        target_pos_real=target_pos_real,
        resolutions_px=resolutions_px,
        interp_factor=interp_factor,
        margin=margin,
    )

    # storing results
    results.clutter = sp.convert_to_db(intensity_background)
    # Peak Value: magnitude response of the target at peak position
    results.peak_value_complex = data_interp[interp_peak_position[0], interp_peak_position[1]]
    # compute radar cross section (RCS) [per unit pixel area]
    # if results is somehow negative, 0 is returned
    results.rcs = np.max([integrated_peak_intensity / (k_lin * s_f**2), 0])
    # computing SCR
    results.scr = sp.convert_to_db(np.abs(results.peak_value_complex) ** 2) - results.clutter
    return results, peak_corners, roi_background_corners


def compute_rcs_with_cross_method(
    data: np.ndarray,
    data_interp: np.ndarray,
    target_pos_real: npt.NDArray[np.floating],
    max_position: tuple[int, int],
    resolutions_px: tuple[float, float],
    interp_factor: int,
    k_lin: float,
    s_f: float,
    margin: int = 40,
) -> tuple[RCSDataOutput, list, np.ndarray]:
    """Computing RCS using the central cross algorithm, with a cross masking the signal of the point target inside of a
    central box ROI.

    Parameters
    ----------
    data : np.ndarray
        target data intensity
    data_interp : np.ndarray
        interpolated target data intensity
    target_pos_real : npt.NDArray[np.floating]
        position of the target
    max_position : tuple[int, int]
        position of the max inside the target area (data)
    resolutions_px : tuple[float, float]
        irf resolutions in pixels, range[0] and azimuth[1]
    interp_factor : int
        interpolation factor
    k_lin : float
        a value of 1 means absolutely calibrated
    s_f : float
        a value of 1 means not resampled
    margin : int, optional
        margin for the central box computation, by default 40

    Returns
    -------
    RCSDataOutput
        output results for RCS computation
    list
        list of peak roi corner pixels
    np.ndarray
        rcs masking boundary
    """
    results = RCSDataOutput()
    # finding the rectangular roi from the resolution around peak position
    rng_start, rng_end, az_start, az_end = sp.compute_rectangular_roi_from_resolution(
        shape=data.shape,
        peak_position=max_position,
        resolutions_px=resolutions_px,
        interp_factor=1,
        margin=margin,
    )
    roi_data = data[rng_start:rng_end, az_start:az_end]
    roi_center = sp.locate_max_2d(roi_data)
    # creating a cross + internal box mask around the signal peak
    rng_cross_width = math.ceil(resolutions_px[0] / 0.8 * interp_factor) + 2
    az_cross_width = math.ceil(resolutions_px[1] / 0.8 * interp_factor) + 2
    cross_mask = generate_cross_mask(
        shape=roi_data.shape,
        center=roi_center,
        v_width=az_cross_width,
        h_width=rng_cross_width,
    ).astype(bool)
    central_box_mask = generate_box_mask(
        shape=roi_data.shape,
        center=roi_center,
        rows_length=rng_cross_width + rng_cross_width // 2,
        cols_width=az_cross_width + az_cross_width // 2,
    ).astype(bool)
    total_mask = np.logical_or(cross_mask, central_box_mask)
    # computing the intensity background
    intensity_background = np.nanmean(roi_data[~total_mask])
    target_interp_intens_corr = np.abs(data_interp) ** 2 - intensity_background
    # computing the integrated peak intensity
    integrated_peak_intensity, interp_peak_position, peak_corners = sp.compute_integrated_peak_intensity(
        data=target_interp_intens_corr,
        max_position=max_position,
        target_pos_real=target_pos_real,
        resolutions_px=resolutions_px,
        interp_factor=interp_factor,
    )
    # storing results
    results.clutter = sp.convert_to_db(intensity_background)
    # Peak Value: magnitude response of the target at peak position
    results.peak_value_complex = data_interp[interp_peak_position[0], interp_peak_position[1]]
    # if results is somehow negative, 0 is returned
    results.rcs = np.max([integrated_peak_intensity / (k_lin * s_f**2), 0])
    # computing SCR
    results.scr = sp.convert_to_db(np.abs(results.peak_value_complex) ** 2) - results.clutter
    boundary_mask = np.zeros_like(data)
    total_mask_contour = _generate_contour_mask(total_mask)
    boundary_mask[rng_start:rng_end, az_start:az_end] = total_mask_contour
    return results, peak_corners, boundary_mask


def _generate_contour_mask(data: np.ndarray) -> np.ndarray:
    """Generating total mask boundary contour.

    Parameters
    ----------
    data : np.ndarray
        input mask

    Returns
    -------
    np.ndarray
        mask boundaries
    """
    eroded = ndimage.binary_erosion(data)
    # pixels that are True but have a False neighbor
    boundary = data & ~eroded

    ys, xs = np.where(boundary)

    boundary_mask = np.zeros_like(data, dtype=bool)
    boundary_mask[ys, xs] = True
    return boundary_mask
