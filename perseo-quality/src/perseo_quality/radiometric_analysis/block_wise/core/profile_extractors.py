# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Block-Wise Radiometric Analysis core profile extractors"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.signal import convolve2d, medfilt2d

from perseo_quality.core.signal_processing import convert_to_db
from perseo_quality.logger import quality_logger as log
from perseo_quality.radiometric_analysis.block_wise.config import ProfileExtractionParameters
from perseo_quality.radiometric_analysis.block_wise.support import masking_outliers_by_percentiles

# custom profile extractor callable type to be matched
RadiometricProfileExtractorType = Callable[[np.ndarray, ProfileExtractionParameters], np.ndarray]


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
    data = convolve2d(
        data,
        kernel,
        mode="same",
    )

    # hist_bins = int(np.mean([uu.size for uu in u]))
    row_histograms = [np.histogram(row[np.isfinite(row)], bins="fd") for row in data]
    peaks = np.array([(h[1][np.argmax(h[0])] + h[1][np.argmax(h[0]) + 1]) / 2 for h in row_histograms])
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

    profile_db = convert_to_db(np.nanmean(data, 1))
    return np.ma.masked_invalid(profile_db)


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
