# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Point Target Radar Cross Section computation"""

from __future__ import annotations

import math

import numpy as np
from arepytools.io.io_support import NominalPointTarget
from scipy.constants import speed_of_light as LIGHT_SPEED

import perseo_quality.core.signal_processing as sp
from perseo_quality.core.generic_dataclasses import SARPolarization, TargetDataType
from perseo_quality.logger import quality_logger as log
from perseo_quality.point_targets_analysis.core.pre_processing import (
    detect_data_type,
)
from perseo_quality.point_targets_analysis.custom_dataclasses import RCSDataOutput
from perseo_quality.point_targets_analysis.custom_errors import PointTargetComputationError


def compute_point_target_rcs(
    target_area: np.ndarray,
    range_resolution_px: float,
    azimuth_resolution_px: float,
    target_pos_real: np.ndarray,
    rcs_interp_factor: int,
    rcs_roi: np.ndarray,
    k_lin: float = 1,
    s_f: float = 1,
) -> tuple[RCSDataOutput, np.ndarray, list, list]:
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
    target_pos_real : np.ndarray
        position of the signal peak in pixels, range_px[0] and azimuth_px[1]
    rcs_interp_factor : int
        rcs interpolation factor
    rcs_roi : np.ndarray
        region of interest for RCS computation
    k_lin : float, optional
        a value of 1 means absolutely calibrated, by default 1
    s_f : float, optional
        a value of 1 means not resampled, by default 1

    Returns
    -------
    pdt.RCSDataOutput
        dataclass object containing all computed export variables
    pdt.RCSGraphDataOutput
        dataclass object containing all data needed for plotting graphs outside
    list
        background intensity corners, a list pixels for each square corner region
    list
        list of peak roi corner pixels
    """

    target = target_area.copy()
    data_type = detect_data_type(target_area=target)
    irf_resolution_px = (range_resolution_px, azimuth_resolution_px)

    # initializing output structure
    results = RCSDataOutput()

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

    intensity_bkgnd, roi_background_corners = sp.compute_intensity_background(
        data=target_intensity, resolutions_px=irf_resolution_px, roi=rcs_roi
    )
    results.clutter = sp.convert_to_db(intensity_bkgnd)

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

    # correcting interpolated intensity data subtracting background
    # intensity
    target_interp_intens_corr = np.abs(target_intensity_interp) ** 2 - intensity_bkgnd

    # finding peak position in interpolated intensity corrected target area
    peak_row, peak_col = _peak_extraction(
        data=target_interp_intens_corr,
        target_position=target_pos_real,
        interp_factor=rcs_interp_factor,
        max_indexes=(max_row, max_col),
    )
    # Peak Value: magnitude response of the target in peak position,
    # complex number
    results.peak_value_complex = target_intensity_interp[peak_row, peak_col]

    # integrate the interpolated corrected data intensity on peak region
    integrated_peak_intensity, peak_corners = sp.compute_integrated_peak_intensity(
        data=target_interp_intens_corr,
        peak_position=(peak_row, peak_col),
        resolutions_px=irf_resolution_px,
        interp_factor=rcs_interp_factor,
    )

    # compute radar cross section (RCS) [per unit pixel area]
    # if results is somehow negative, 0 is returned
    results.rcs = np.max([integrated_peak_intensity / (k_lin * s_f**2), 0])
    # computing SCR
    results.scr = sp.convert_to_db(np.abs(results.peak_value_complex) ** 2) - results.clutter

    return results, roi_target, roi_background_corners, peak_corners


def compute_additional_rcs_values(
    rcs_input: RCSDataOutput,
    step_distances: list,
    interp_factor: int,
    polarization: SARPolarization,
    target_info: NominalPointTarget,
    sensor_position: np.ndarray,
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
    target_info : NominalPointTarget
        target info as NominalPointTarget
    sat_position : np.ndarray
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


def _roi_extraction(data: np.ndarray, roi: np.ndarray, target_pos: np.ndarray = None) -> tuple[int, int, np.ndarray]:
    """Extraction of a roi from the input array.

    Parameters
    ----------
    data : np.ndarray
        input array
    roi : np.ndarray
        roi_size [row number, col number]
    target_pos : np.ndarray, optional
        position of the target peak. If None, it is calculated from input array, by default None

    Returns
    -------
    tuple[int, int, np.ndarray]
        row max index
        column max index
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
        (row_lim_up < 0, row_lim_dwn > data.shape[0], col_lim_sx < 0, col_lim_dx > data.shape[1])
    )
    if break_cond:
        log.warning("Could not evaluate RCS: extracted target area is too small")
        raise PointTargetComputationError

    roi_target = data[row_lim_up:row_lim_dwn, col_lim_sx:col_lim_dx].copy()

    return max_row, max_col, roi_target


def _peak_extraction(
    data: np.ndarray,
    target_position: np.ndarray | None = None,
    max_indexes: tuple[float, float] | None = None,
    interp_factor: int = 8,
) -> tuple[float, float]:
    """Extraction of the peak indexes from input array.

    Parameters
    ----------
    data : np.ndarray
        2D input array
    target_position : Optional[np.ndarray], optional
        position of the target peak, by default None
    max_indexes : Optional[tuple[float, float]], optional
        row and column indexes of the max. If None, it is calculated from the input array, by default None
    interp_factor : int, optional
        interpolation factor, by default 8

    Returns
    -------
    tuple[float, float]
        peak row index
        peak column index
    """

    # checking if target position is provided
    if target_position is None:
        # computing peak position
        max_row, max_col = sp.locate_max_2d(data)
    else:
        cut_row_start = max_indexes[0] * interp_factor - 1
        cut_col_start = max_indexes[0] * interp_factor - 1

        interp_int_corrected_cut = data[
            cut_row_start : cut_row_start + 2 * interp_factor + 1,
            cut_col_start : cut_col_start + 2 * interp_factor + 1,
        ]

        max_row, max_col = sp.locate_max_2d(interp_int_corrected_cut)
        max_row += int(np.floor(target_position[0]) * interp_factor) - 1
        max_col += int(np.floor(target_position[1]) * interp_factor) - 1

    return max_row, max_col
