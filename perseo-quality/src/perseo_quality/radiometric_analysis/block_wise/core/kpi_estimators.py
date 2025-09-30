# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Block-Wise Radiometric Analysis core profile extractors"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import kurtosis, mode, skew

from perseo_quality.core.signal_processing import compute_equivalent_number_of_looks, convert_to_db
from perseo_quality.radiometric_analysis.block_wise.support import compute_profile_variability_index
from perseo_quality.radiometric_analysis.custom_dataclasses import (
    AverageElevationRadiometricKPI,
    NESZRadiometricKPI,
    RadiometricProfileAxes,
    ScallopingRadiometricKPI,
)

# custom profile extractor callable type to be matched
RadiometricBlockKPIEstimatorType = Callable[
    [np.ndarray, RadiometricProfileAxes, np.ndarray],
    AverageElevationRadiometricKPI | NESZRadiometricKPI | ScallopingRadiometricKPI,
]


def average_elevation_profile_kpi_estimator(
    profile: np.ndarray, axes: RadiometricProfileAxes, data_block: np.ndarray
) -> AverageElevationRadiometricKPI:
    """Estimating KPI for Average Elevation Profile analysis.

    Parameters
    ----------
    profile : np.ndarray
        average elevation profile for the current block, numpy masked invalid array
    axes : RadiometricProfileAxes
        axes of the current average elevation profile
    data_block : np.ndarray
        data block, numpy masked invalid array

    Returns
    -------
    AverageElevationRadiometricKPI
        KPI for the Average Elevation Profile analysis
    """
    enl_block = compute_equivalent_number_of_looks(intensity_data=data_block)
    slope, variability_index = compute_profile_variability_index(profile=profile, look_angles_deg=axes.look_angles_deg)
    return AverageElevationRadiometricKPI(
        mid_incidence_angle_deg=float(axes.incidence_angles_deg[axes.incidence_angles_deg.size // 2]),
        mid_look_angle_deg=float(axes.look_angles_deg[axes.look_angles_deg.size // 2]),
        enl_block=enl_block,
        mean_level_db=float(profile.mean()),
        std_level_db=float(profile.std()),
        slope_wrt_look_angle_db_deg=slope,
        variability_index_db=variability_index,
    )


def nesz_kpi_estimator(profile: np.ndarray, axes: RadiometricProfileAxes, data_block: np.ndarray) -> NESZRadiometricKPI:
    """Estimating KPI for NESZ analysis.

    Parameters
    ----------
    profile : np.ndarray
        nesz profile for the current block, numpy masked invalid array
    axes : RadiometricProfileAxes
        axes of the current nesz profile
    data_block : np.ndarray
        data block, numpy masked invalid array

    Returns
    -------
    NESZRadiometricKPI
        KPI for NESZ analysis
    """
    min_nesz_idx = profile.argmin()
    max_nesz_idx = profile.argmax()
    return NESZRadiometricKPI(
        min_nesz_profile_db=float(profile[min_nesz_idx]),
        min_nesz_incidence_angle_deg=float(axes.incidence_angles_deg[min_nesz_idx]),
        min_nesz_range_position=float(axes.slant_range[min_nesz_idx]),
        min_nesz_look_angle_deg=float(axes.look_angles_deg[min_nesz_idx]),
        max_nesz_profile_db=float(profile[max_nesz_idx]),
        max_nesz_incidence_angle_deg=float(axes.incidence_angles_deg[max_nesz_idx]),
        max_nesz_range_position=float(axes.slant_range[max_nesz_idx]),
        max_nesz_look_angle_deg=float(axes.look_angles_deg[max_nesz_idx]),
        mean_nesz_profile_dB=float(profile.mean()),
        std_nesz_profile_dB=float(profile.std()),
        mode_nesz_profile_dB=float(mode(profile).mode),
        skewness_profile=float(skew(profile)),
        kurtosis_profile=float(kurtosis(profile)),
        mean_block_dB=float(convert_to_db(data_block.mean())),
        std_block_dB=float(convert_to_db(data_block.std())),
        mode_block_dB=float(convert_to_db(mode(data_block, axis=None).mode)),
        skewness_block=float(skew(data_block, axis=None)),
        kurtosis_block=float(kurtosis(data_block, axis=None)),
    )


def scalloping_kpi_estimator(
    profile: np.ndarray, axes: RadiometricProfileAxes, data_block: np.ndarray
) -> ScallopingRadiometricKPI:
    """Estimating KPI for Scalloping analysis.

    Parameters
    ----------
    profile : np.ndarray
        scalloping profile for the current block, numpy masked invalid array
    axes : RadiometricProfileAxes
        axes of the current scalloping profile
    data_block : np.ndarray
        data block, numpy masked invalid array

    Returns
    -------
    ScallopingRadiometricKPI
        KPI for Scalloping analysis
    """
    return ScallopingRadiometricKPI(
        mean_level_db=float(profile.mean()),
        max_level_db=float(profile.max()),
        min_level_db=float(profile.min()),
        std_level_db=float(profile.std()),
    )
