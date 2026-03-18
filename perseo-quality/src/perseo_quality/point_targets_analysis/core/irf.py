# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Point Target Impulse Response Function computation"""

from __future__ import annotations

import numpy as np

import perseo_quality.core.masking_operations as masking
from perseo_quality.core.generic_dataclasses import MaskingMethod
from perseo_quality.core.signal_processing import convert_to_db, locate_max_2d
from perseo_quality.logger import quality_logger as log
from perseo_quality.point_targets_analysis.custom_dataclasses import IRFComputedParameters
from perseo_quality.point_targets_analysis.support import SideLobesDirections


def compute_point_target_irf_analysis(
    recentered_target_area_interp: np.ndarray,
    range_resolution_px: float,
    azimuth_resolution_px: float,
    side_lobes_directions: SideLobesDirections | None = None,
    mask_method: MaskingMethod = MaskingMethod.PEAK,
    pslr_flag: bool = True,
    islr_flag: bool = True,
    sslr_flag: bool = True,
) -> IRFComputedParameters:
    """Computing Point Target Impulse Response Function parameters: PSLR, ISLR and SSLR, if requested.

    Parameters
    ----------
    recentered_target_area_interp : np.ndarray
        target area centered on point target signal peak, interpolated for higher resolution, with shape (n_rng, naz)
    range_resolution_px : float
        range resolution in pixels
    azimuth_resolution_px : float
        azimuth resolution in pixels
    side_lobes_directions : SideLobesDirections, optional
        range and azimuth cuts angular coefficients in samples, by default (np.inf, 0.0)
    mask_method : MaskingMethod, optional
        masking method for PSLR and ISLR computation, by default MaskingMethod.PEAK
    pslr_flag : bool, optional
        flag to enable PSLR computation, by default True
    islr_flag : bool, optional
        flag to enable ISLR computation, by default True
    sslr_flag : bool, optional
        flag to enable SSLR computation, by default True

    Returns
    -------
    IRFComputedParameters
        computed IRF parameters
    """
    side_lobes_directions = side_lobes_directions if side_lobes_directions is not None else (np.inf, 0.0)

    results = IRFComputedParameters()

    irf_resolution_px = (range_resolution_px, azimuth_resolution_px)

    if pslr_flag:
        try:
            results.range_pslr, results.azimuth_pslr, results.pslr_2d = compute_pslr_2d(
                data=recentered_target_area_interp,
                resolution=irf_resolution_px,
                mask=mask_method,
                side_lobes_directions=side_lobes_directions,
            )
        except Exception:
            log.warning("Could not evaluate PSLR properly.")
    else:
        log.warning("PSLR computation has been disabled in configuration file.")

    if islr_flag:
        try:
            results.range_islr, results.azimuth_islr, results.islr_2d = compute_islr_2d(
                data=recentered_target_area_interp,
                resolution=irf_resolution_px,
                mask=mask_method,
                side_lobes_directions=side_lobes_directions,
            )
        except Exception:
            log.warning("Could not evaluate ISLR properly.")
    else:
        log.warning("ISLR computation has been disabled in configuration file.")

    if sslr_flag:
        try:
            results.range_sslr, results.azimuth_sslr, results.sslr_2d = compute_sslr_2d(
                data=recentered_target_area_interp,
                resolution=irf_resolution_px,
                side_lobes_directions=side_lobes_directions,
            )
        except Exception:
            log.warning("Could not evaluate SSLR properly.")
    else:
        log.warning("SSLR computation has been disabled in configuration file.")

    return results


def compute_pslr_2d(
    data: np.ndarray,
    resolution: tuple[float, float],
    mask: MaskingMethod = MaskingMethod.RESOLUTION,
    side_lobes_directions: SideLobesDirections = (np.inf, 0.0),
) -> tuple[float, float, float]:
    """Compute the PSLR (Peak-to-Side-Lobe-Ratio) of the given input 2D array.

    Parameters
    ----------
    data : np.ndarray
        input 2D array to compute PSLR onto, with shape (n_rng, n_az)
    resolution : tuple[float, float]
        range [0] and azimuth [1] resolutions in pixel
    mask : MaskingMethod, optional
        masking generation method, by default MaskingMethod.RESOLUTION
    side_lobes_directions : SideLobesDirections, optional
        range and azimuth cuts angular coefficients in samples, by default (np.inf, 0.0)

    Returns
    -------
    float
        range PSLR
    float
        azimuth PSLR
    float
        2D PSLR
    """

    # Compute data power and size
    data = np.abs(data) ** 2
    # Find data peak
    max_row, max_col = locate_max_2d(data)
    main_lobe_value = data[max_row, max_col]

    # masking input data
    side_lobes_mask = masking.pslr_masking(
        data=data,
        mask_flag=mask,
        peak_pos=(max_row, max_col),
        side_lobes_directions=side_lobes_directions,
        resolution=resolution,
    )

    # masking original data
    side_lobes_data = data * side_lobes_mask
    # evaluating the max side lobes value on 2D array
    peak_row_id, peak_col_id = locate_max_2d(side_lobes_data)
    max_side_lobe_value = side_lobes_data[peak_row_id, peak_col_id]

    # extracting profile cuts
    rng_cut, az_cut = masking.pslr_profile_cutting(
        masked_data=side_lobes_data,
        peak_pos=(max_row, max_col),
        side_lobes_directions=side_lobes_directions,
    )

    if np.isinf(side_lobes_directions[0]):
        max_side_lobe_value_rng = np.abs(data[np.argmax(rng_cut), max_col])
        max_side_lobe_value_az = np.abs(data[max_row, np.argmax(az_cut)])
    else:
        max_side_lobe_value_rng = np.max(rng_cut)
        max_side_lobe_value_az = np.max(az_cut)

    # Evaluate PSLR
    # PSLR 2D
    pslr_2d = convert_to_db(max_side_lobe_value / main_lobe_value)

    # Azimuth PSLR
    pslr_azimuth = convert_to_db(max_side_lobe_value_az / main_lobe_value)

    # Range PSLR
    pslr_range = convert_to_db(max_side_lobe_value_rng / main_lobe_value)

    return pslr_range, pslr_azimuth, pslr_2d


def compute_islr_2d(
    data: np.ndarray,
    resolution: tuple[float, float],
    mask: MaskingMethod = MaskingMethod.RESOLUTION,
    side_lobes_directions: SideLobesDirections = (np.inf, 0.0),
) -> tuple[float, float, float]:
    """Compute Integral-Side-Lobe-Ratio (ISLR).

    Parameters
    ----------
    data : np.ndarray
        input 2D array to compute ISLR onto, with shape (n_rng, n_az)
    resolution : tuple[float, float]
        range [0] and azimuth [1] resolutions
    mask : MaskingMethod, optional
        masking generation method, by default MaskingMethod.RESOLUTION
    side_lobes_directions : SideLobesDirections, optional
        range and azimuth cuts angular coefficients in samples, by default np.array([np.inf, 0.0])

    Returns
    -------
    float
        range ISLR
    float
        azimuth ISLR
    float
        2D ISLR
    """

    # Compute data power and size
    data = np.abs(data) ** 2
    # Find data peak
    max_row, max_col = locate_max_2d(data)

    main_lobe_mask, islr_mask = masking.islr_masking(
        data=data,
        mask_flag=mask,
        resolution=resolution,
        peak_pos=(max_row, max_col),
        side_lobes_directions=side_lobes_directions,
    )
    main_lobe_cuts, side_lobes_cuts = masking.islr_profile_cutting(
        data=data,
        main_lobe_mask=main_lobe_mask,
        islr_mask=islr_mask,
        peak_pos=(max_row, max_col),
        side_lobes_directions=side_lobes_directions,
    )

    # evaluating integrals over main lobe and side lobes
    main_lobe_energy = np.sum(np.abs(data * main_lobe_mask))
    side_lobes_energy = np.sum(np.abs(data * islr_mask))

    main_lobe_rng_energy = np.sum(main_lobe_cuts[0])
    main_lobe_az_energy = np.sum(main_lobe_cuts[1])
    side_lobes_rng_energy = np.sum(side_lobes_cuts[0])
    side_lobes_az_energy = np.sum(side_lobes_cuts[1])

    # Compute ISLR
    # ISLR 2D
    islr_2d = convert_to_db(side_lobes_energy / main_lobe_energy)

    # Azimuth ISLR
    islr_azimuth = convert_to_db(side_lobes_az_energy / main_lobe_az_energy)

    # Range ISLR
    islr_range = convert_to_db(side_lobes_rng_energy / main_lobe_rng_energy)

    return islr_range, islr_azimuth, islr_2d


def compute_sslr_2d(
    data: np.ndarray,
    resolution: tuple[float, float],
    side_lobes_directions: SideLobesDirections,
) -> tuple[float, float, float]:
    """Compute Secondary-Side-Lobe-Ratio (SSLR).

    Parameters
    ----------
    data : np.ndarray
        input 2D array to compute SSLR onto, with shape (n_rng, n_az)
    resolution : tuple[float, float]
        range [0] and azimuth [1] resolutions
    side_lobes_directions : SideLobesDirections
        range and azimuth cuts angular coefficients in samples

    Returns
    -------
    float
        range SSLR
    float
        azimuth SSLR
    float
        2D SSLR
    """

    # Compute data power and size
    data = np.abs(data) ** 2
    # Find data peak
    max_row, max_col = locate_max_2d(data)
    main_lobe_value = data[max_row, max_col]

    # identifying only intermediate side lobes
    intermediate_side_lobes_mask = masking.sslr_masking(
        data=data,
        resolution=resolution,
        peak_pos=(max_row, max_col),
        side_lobes_directions=side_lobes_directions,
    )

    # evaluating the intermediate side lobes values from masked image
    masked_data = data * intermediate_side_lobes_mask
    peak_row_id, peak_col_id = locate_max_2d(masked_data)
    intermediate_side_lobes = masked_data[peak_row_id, peak_col_id]

    # extracting range and azimuth cuts from masked image
    intermediate_side_lobes_values_rng, intermediate_side_lobes_values_az = masking.sslr_profile_cutting(
        masked_data=masked_data,
        peak_pos=(max_row, max_col),
        side_lobes_directions=side_lobes_directions,
    )

    if np.isinf(side_lobes_directions[0]):
        intermediate_side_lobes_values_rng = np.abs(data[np.argmax(intermediate_side_lobes_values_rng), max_col])
        intermediate_side_lobes_values_az = np.abs(data[max_row, np.argmax(intermediate_side_lobes_values_az)])

    # Evaluate SSLR
    # SSLR 2D
    sslr_2d = convert_to_db(intermediate_side_lobes / main_lobe_value)

    # Azimuth SSLR
    sslr_azimuth = convert_to_db(intermediate_side_lobes_values_az / main_lobe_value)

    # Range SSLR
    sslr_range = convert_to_db(intermediate_side_lobes_values_rng / main_lobe_value)

    return sslr_range, sslr_azimuth, sslr_2d
