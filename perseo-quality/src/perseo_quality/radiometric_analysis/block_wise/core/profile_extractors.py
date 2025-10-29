# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Block-Wise Radiometric Analysis core profile extractors"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from functools import wraps

import numpy as np
from numba import jit, prange
from scipy.signal import convolve2d, medfilt2d

from perseo_quality.core.signal_processing import convert_to_db
from perseo_quality.logger import quality_logger as log
from perseo_quality.radiometric_analysis.block_wise.config import ProfileExtractionParameters
from perseo_quality.radiometric_analysis.block_wise.support import masking_outliers_by_percentiles

# custom profile extractor callable type to be matched
RadiometricProfileExtractorType = Callable[[np.ndarray, ProfileExtractionParameters], np.ndarray]

PROFILE_EXTRACTORS_REGISTRY: dict[str, RadiometricProfileExtractorType] = {}


def register_profile_extractor(
    name: str,
) -> Callable[[RadiometricProfileExtractorType], RadiometricProfileExtractorType]:
    def decorator(func: RadiometricProfileExtractorType) -> RadiometricProfileExtractorType:
        @wraps(func)
        def wrapper(*args, **kwargs) -> np.ndarray:
            return func(*args, **kwargs)

        PROFILE_EXTRACTORS_REGISTRY[name] = wrapper
        return wrapper

    return decorator


@jit(nopython=True, parallel=True, cache=True)
def _compute_num_bins(data: np.ndarray) -> np.ndarray:
    # Freedman Diaconis Estimator
    rows = data.shape[0]
    result = np.empty(rows)

    for i in prange(rows):
        row_data = data[i]
        valid_data = row_data[~np.isnan(row_data)]
        if len(valid_data) == 0:
            result[i] = np.nan
            continue

        q75 = np.percentile(valid_data, 75)
        q25 = np.percentile(valid_data, 25)
        iqr = q75 - q25
        fd_bins_width = 2.0 * iqr * len(valid_data) ** (-1.0 / 3.0)
        bin_range = np.max(valid_data) - np.min(valid_data)
        result[i] = np.ceil(bin_range / fd_bins_width)

    return result


@jit(nopython=True, cache=True)
def _compute_histogram_peak(data: np.ndarray, num_bins: int) -> float:
    if num_bins <= 0 or not np.isfinite(num_bins):
        return np.nan

    valid_data = data[np.isfinite(data)]
    if len(valid_data) == 0:
        return np.nan

    hist, bin_edges = np.histogram(valid_data, bins=int(num_bins))
    max_idx = np.argmax(hist)
    return (bin_edges[max_idx] + bin_edges[max_idx + 1]) / 2


@register_profile_extractor("nesz")
def nesz_profiles_extractor(data: np.ndarray, params: ProfileExtractionParameters) -> np.ndarray:
    """Profiles extraction function for NESZ analysis.

    Parameters
    ----------
    data : np.ndarray
        2D target block to be processed
    params : ProfileExtractionParameters
        radiometric profiles configuration

    Returns
    -------
    np.ndarray
        nesz profile
    """
    # azimuth profile as a sum over range
    current_block_az_profile = np.nansum(data, axis=0)

    # if more than half of the block is populated with zeroes, discard the whole block
    if np.count_nonzero(current_block_az_profile) / current_block_az_profile.size < 0.5:
        return np.full(data.shape[0], np.nan)

    kernel = np.ones((params.filtering_kernel_size[0], params.filtering_kernel_size[1])) / (
        params.filtering_kernel_size[0] * params.filtering_kernel_size[1]
    )

    # performing multi-looking 2D convolution (moving average 2D)
    zero_rows_indexes = np.where(np.count_nonzero(data, axis=1) / data.shape[1] < 0.8)[0]
    data = convolve2d(data, kernel, mode="same")

    bin_edges = _compute_num_bins(data)
    peaks = np.array([_compute_histogram_peak(row, bins) for row, bins in zip(data, bin_edges, strict=True)])
    peaks[zero_rows_indexes] = np.nan

    averaging_window_length = np.max([5, int(0.01 * peaks.size)])
    # window length must be odd
    averaging_window_length = (
        averaging_window_length if averaging_window_length % 2 != 0 else averaging_window_length + 1
    )
    peaks = np.convolve(peaks, np.ones(averaging_window_length) / averaging_window_length, mode="same")

    masking_invalid_convoluted_data = (averaging_window_length - 1) // 2
    # removing convolution artifacts at profile borders
    peaks[:masking_invalid_convoluted_data] = np.nan
    peaks[-masking_invalid_convoluted_data:] = np.nan

    return np.ma.masked_invalid(convert_to_db(peaks))


@register_profile_extractor("average_elevation")
def average_elevation_profiles_extractor(data: np.ndarray, params: ProfileExtractionParameters) -> np.ndarray:
    """Profiles extraction function for generic average elevation radiometric profiles analysis.

    Parameters
    ----------
    data : np.ndarray
        2D target block to be processed
    params : ProfileExtractionParameters
        radiometric profiles configuration

    Returns
    -------
    np.ndarray
        average elevation profile
    """
    if params.smoothening_filter:
        log.info("Applying smoothening median filter...")
        data = medfilt2d(data, params.filtering_kernel_size)

    if params.outlier_removal:
        log.info("Masking outliers...")
        # masking data by percentiles
        data = masking_outliers_by_percentiles(
            data=data,
            kernel=params.outliers_kernel_size,
            percentile_boundaries=params.outliers_percentile_boundaries,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        profile_db = convert_to_db(np.nanmean(data, 1))
    return np.ma.masked_invalid(profile_db)


@register_profile_extractor("scalloping")
def scalloping_profiles_extractor(data: np.ndarray, params: ProfileExtractionParameters) -> np.ndarray:
    """Profiles extraction function for Scalloping analysis.

    Parameters
    ----------
    data : np.ndarray
        2D target block to be processed
    params : ProfileExtractionParameters
        radiometric profiles configuration

    Returns
    -------
    np.ndarray
        scalloping profile
    """
    if params.outlier_removal:
        log.info("Masking outliers...")
        data = masking_outliers_by_percentiles(
            data=data, kernel=params.outliers_kernel_size, percentile_boundaries=params.outliers_percentile_boundaries
        )

    azimuth_profile = np.nanmean(data, axis=0)
    azimuth_profile = azimuth_profile / np.nanmean(azimuth_profile)

    profile_db = convert_to_db(azimuth_profile)
    return np.ma.masked_invalid(profile_db)
