# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Block-Wise Radiometric Analysis common functionalities"""

from __future__ import annotations

import numpy as np
from numpy.polynomial import Polynomial
from scipy.ndimage import (
    binary_dilation,
    binary_opening,
    generate_binary_structure,
    label,
    uniform_filter,
)
from scipy.signal import convolve2d

from perseo_quality.radiometric_analysis.block_wise.config import RiverMaskingConfig


def masking_outliers_by_percentiles(
    data: np.ndarray, kernel: tuple[int, int], percentile_boundaries: tuple[int, int]
) -> np.ndarray:
    """Masking outliers outside of provided percentile boundaries setting them to NaN.

    Parameters
    ----------
    data : np.ndarray
        input 2D array
    kernel : tuple[int, int]
        kernel size, height and width in pixels
    percentile_boundaries : tuple[int, int]
        data below percentile_boundaries[0] and above percentile_boundaries[1] are set to NaN

    Returns
    -------
    np.ndarray
        input array with NaN where outliers lie
    """
    filter_kernel = np.ones(kernel)
    masking_cond = np.logical_or(
        data < np.nanpercentile(data.ravel(), percentile_boundaries[0]),
        data > np.nanpercentile(data.ravel(), percentile_boundaries[1]),
    ).astype("int64")

    # convolving data with filter kernel
    mask = np.round(convolve2d(masking_cond, filter_kernel, mode="same") / np.sum(filter_kernel))

    # masking out data
    data[np.where(mask)] = np.nan

    return data


def compute_profile_variability_index(profile: np.ndarray, look_angles_deg: np.ndarray) -> tuple[float, float]:
    """Computing radiometric variability index for the current profile, with respect to the look angles axis.

    Parameters
    ----------
    profile : np.ndarray
        current radiometric profile in [dB]
    look_angles_deg : np.ndarray
        look angles axis of the provided profile in degrees

    Returns
    -------
    float
        slope with respect to look angles axis in [dB/deg]
    float
        radiometric variability index in [dB]
    """
    # linear fit
    linear_fit_params = Polynomial.fit(look_angles_deg[~profile.mask], profile.compressed(), deg=1).convert()

    # homogeneity index
    regression_line = linear_fit_params.coef[0] + linear_fit_params.coef[1] * look_angles_deg
    radiometric_profiles_de_sloped = profile - regression_line
    variability_index = np.percentile(radiometric_profiles_de_sloped.compressed(), 90) - np.percentile(
        radiometric_profiles_de_sloped.compressed(), 10
    )
    return float(linear_fit_params.coef[1]), float(variability_index)


def compute_local_statistics(data: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute local mean and local coefficient of variation

    Parameters
    ----------
    data : np.ndarray
        Target area
    window : int
        Side length of the square sliding window for computing local mean and CV.

    Returns
    -------
    np.ndarray
        Local mean raster, same shape as data
    np.ndarray
        Local coefficient of variation, same shape as data
    """
    data = np.where(np.isnan(data), 0.0, data)
    # vectorized_filter
    local_mean = uniform_filter(data, size=window)
    local_square_mean = uniform_filter(data**2, size=window)
    local_std = np.sqrt(np.maximum(local_square_mean - local_mean**2, np.zeros(local_square_mean.shape)))
    # Avoid division by zero
    local_cv = np.where(local_mean > np.finfo(float).eps, local_std / local_mean, 0.0)
    return local_mean, local_cv


def threshold_mask(
    local_mean: np.ndarray,
    local_cv: np.ndarray,
    backscatter_thresh: float,
    cv_lower_thresh: float,
    cv_upper_thresh: float,
) -> np.ndarray:
    """Classify pixels as river where both conditions hold:

    1) local mean < backscatter threshold
    2) local coefficient of variation is outside of the interval [cv_lower_thresh, cv_upper_thresh]

    Parameters
    ----------
    local_mean : np.ndarray
        Local mean raster
    local_cv : np.ndarray
        Local coefficient of variation raster
    backscatter_thresh : float
        Backscatter threshold
    cv_lower_thresh : float
        Coefficient of variation lower threshold
    cv_upper_thresh : float
        Coefficient of variation upper threshold

    Returns
    -------
    np.ndarray
        Boolean mask: True if river pixel is identified
    """
    return (local_mean < backscatter_thresh) & ((local_cv > cv_upper_thresh) | (local_cv < cv_lower_thresh))


def _disk_structuring_element(radius: int) -> np.ndarray:
    """Create a circular (disk) binary structuring element."""
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return x**2 + y**2 <= radius**2


def morphological_cleaning(binary_mask: np.ndarray, opening_radius: int, min_area_px_percentile: int) -> np.ndarray:
    """Remove isolated speckles from a binary mask

    Parameters
    ----------
    binary_mask : np.ndarray
        Binary mask
    opening_radius : int
        radius of disk structuring element for opening
    min_area_px_percentile : int
        Connected binary mask portions whose size is below this percentile are removed.

    Returns
    -------
    np.ndarray
        Cleaned binary mask
    """
    opening_structure = _disk_structuring_element(opening_radius)
    bin_opening = binary_opening(binary_mask, opening_structure)

    labels, _ = label(bin_opening)
    component_sizes = np.bincount(labels.ravel())
    min_area_px = np.percentile(component_sizes, min_area_px_percentile)
    keep = component_sizes >= min_area_px
    keep[0] = False  # never keep background label
    cleaned = keep[labels]
    return cleaned


def region_growing(seed_mask: np.ndarray, candidate_mask: np.ndarray, n_iterations: int) -> np.ndarray:
    """Expand a binary mask if neighboring pixels pass the relaxed intensity threshold

    Parameters
    ----------
    seed_mask : np.ndarray
        Binary mask to be expanded
    candidate_mask : np.ndarray
        Pixels eligible to be added to seed mask
    n_iterations : int
        number of dilation steps

    Returns
    -------
    np.ndarray
        Expanded binary mask
    """
    # Connectivity structure: 8-connected
    struct = generate_binary_structure(2, 2)
    grown = seed_mask.copy()

    for _ in range(n_iterations):
        expanded = binary_dilation(grown, structure=struct)
        # Only absorb pixels that are eligible AND newly reached
        grown = grown | (expanded & candidate_mask)

    return grown


def full_river_masking(data: np.ndarray, config: RiverMaskingConfig = RiverMaskingConfig) -> np.ndarray:
    """Mask rivers in acquisition over rainforest, computationally expensive

    Parameters
    ----------
    data : np.ndarray
        Data raster
    config : RiverMaskingConfig, optional
        Configuration dataclass for river masking algorithm, by default RiverMaskingConfig

    Returns
    -------
    np.ndarray
        Masked data raster, same shape as data
    """
    nan_mask = np.isnan(data)
    local_mean, local_cv = compute_local_statistics(data=data, window=config.local_stats_window)
    backscatter_threshold = np.nanpercentile(local_mean, config.backscatter_threshold_percentile)
    cv_upper_threshold = np.nanpercentile(local_cv, config.cv_upper_threshold_percentile)
    cv_lower_threshold = np.nanpercentile(local_cv, config.cv_lower_threshold_percentile)
    relaxed_backscatter_threshold = np.nanpercentile(local_mean, config.relaxed_backscatter_threshold_percentile)
    thrs_mask = (
        threshold_mask(
            local_mean=local_mean,
            local_cv=local_cv,
            backscatter_thresh=backscatter_threshold,
            cv_upper_thresh=cv_upper_threshold,
            cv_lower_thresh=cv_lower_threshold,
        )
        | nan_mask
    )
    # TODO: introduce a stopping criteria instead
    grown_mask = region_growing(
        seed_mask=thrs_mask,
        candidate_mask=data <= relaxed_backscatter_threshold,
        n_iterations=config.region_grow_iterations,
    )
    cleaned_mask = morphological_cleaning(
        binary_mask=grown_mask,
        opening_radius=config.morph_opening_radius,
        min_area_px_percentile=config.min_river_area_px_percentile,
    )
    data[cleaned_mask] = np.nan
    return data


def fast_river_masking(data: np.ndarray, config: RiverMaskingConfig = RiverMaskingConfig) -> np.ndarray:
    """Apply a computationally inexpensive mask for rivers in rainforest acquisitions

    Parameters
    ----------
    data : np.ndarray
        Data raster
    config : RiverMaskingConfig, optional
        Configuration dataclass for river masking algorithm, by default RiverMaskingConfig

    Returns
    -------
    np.ndarray
        Masked data raster, same shape as data
    """
    local_mean, local_cv = compute_local_statistics(data=data, window=config.local_stats_window)
    backscatter_threshold = np.nanpercentile(local_mean, config.relaxed_backscatter_threshold_percentile)
    cv_lower_threshold = np.nanpercentile(local_cv, config.cv_lower_threshold_percentile)
    cv_upper_threshold = np.nanpercentile(local_cv, config.cv_upper_threshold_percentile)
    river_mask = threshold_mask(
        local_mean=local_mean,
        local_cv=local_cv,
        backscatter_thresh=backscatter_threshold,
        cv_lower_thresh=cv_lower_threshold,
        cv_upper_thresh=cv_upper_threshold,
    )
    data[river_mask] = np.nan
    return data
