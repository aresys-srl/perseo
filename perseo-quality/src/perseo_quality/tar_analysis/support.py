# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Distributed and Point Target Ambiguity Ratio support functions"""

from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt
from perseo_core.timing import PreciseDateTime
from scipy.constants import speed_of_light

from perseo_quality.core.signal_processing import (
    compute_distributed_target_ambiguity_ratio_db,
    compute_point_target_ambiguity_ratio_db,
)
from perseo_quality.io.quality_input_protocol import ChannelData
from perseo_quality.tar_analysis.config import AmbiguityRatioConfig

AmbiguityRatioComputingFunction = Callable[
    [npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], AmbiguityRatioConfig], float
]


def compute_ambiguities_locations(
    channel_data: ChannelData,
    point_target_xyz_coords: npt.NDArray[np.floating],
    point_target_azimuth_time: PreciseDateTime,
    point_target_range_time: float,
    doppler_rate: float,
    prf: float,
    burst: int,
) -> tuple[tuple[float, float], tuple[float, float], float, float]:
    """Compute right and left ambiguities location in pixels.

    Parameters
    ----------
    channel_data : ChannelData
        current product channel data object
    point_target_xyz_coords : npt.NDArray[np.floating]
        ECEF xyz point target nominal coordinates
    point_target_azimuth_time : PreciseDateTime
        azimuth time for point target location
    point_target_range_time : float
        range time for point target location
    doppler_rate : float
        doppler rate
    prf : float
        sensor prf
    burst : int
        current burst

    Returns
    -------
    tuple[float, float]
        left ambiguity location in pixels, (azimuth, range)
    tuple[float, float]]
        right ambiguity location in pixels, (azimuth, range)
    float
        azimuth time delta
    float
        range time delta
    """

    t_azimuth_delta = np.abs(prf / doppler_rate)
    sensor_position_at_pt = channel_data.trajectory.position(point_target_azimuth_time)
    sensor_position_at_delta = channel_data.trajectory.position(point_target_azimuth_time + t_azimuth_delta)
    pt_los = np.linalg.norm(sensor_position_at_pt - point_target_xyz_coords)
    pt_delta_los = np.linalg.norm(sensor_position_at_delta - point_target_xyz_coords)

    # TODO: check if this is good also for GRD ground range products, otherwise *(LIGHT_SPEED/2) in that case
    t_range_delta = (pt_delta_los - pt_los) / (speed_of_light / 2)
    left_ambiguity_location_px = channel_data.times_to_pixel_conversion(
        azimuth_time=point_target_azimuth_time - t_azimuth_delta,
        range_time=point_target_range_time + t_range_delta,
        burst=burst,
    )
    right_ambiguity_location_px = channel_data.times_to_pixel_conversion(
        azimuth_time=point_target_azimuth_time + t_azimuth_delta,
        range_time=point_target_range_time + t_range_delta,
        burst=burst,
    )

    return left_ambiguity_location_px, right_ambiguity_location_px, t_azimuth_delta, t_range_delta


def ptar_computing_function_wrapper(
    point_target_roi: npt.NDArray[np.floating],
    right_ambiguity_roi: npt.NDArray[np.floating],
    left_ambiguity_roi: npt.NDArray[np.floating],
    config: AmbiguityRatioConfig,
) -> float:
    """Point Target Ambiguity Ratio computing function wrapper for dependency injection.

    Parameters
    ----------
    point_target_roi : npt.NDArray[np.floating]
        raster data portion centered on the point target location
    right_ambiguity_roi : npt.NDArray[np.floating]
        raster data portion centered on the point target right ambiguity
    left_ambiguity_roi : npt.NDArray[np.floating]
        raster data portion centered on the point target left ambiguity
    config : AmbiguityRatioConfig
        ambiguity ratio computation configuration parameters

    Returns
    -------
    float
        PTAR in decibel
    """
    return compute_point_target_ambiguity_ratio_db(
        point_target_roi=point_target_roi,
        right_ambiguity_roi=right_ambiguity_roi,
        left_ambiguity_roi=left_ambiguity_roi,
        interp_factor=config.interpolation_factor,
    )


def dtar_computing_function_wrapper(
    point_target_roi: npt.NDArray[np.floating],
    right_ambiguity_roi: npt.NDArray[np.floating],
    left_ambiguity_roi: npt.NDArray[np.floating],
    config: AmbiguityRatioConfig,
) -> float:
    """Distributed Target Ambiguity Ratio computing function wrapper for dependency injection.

    Parameters
    ----------
    point_target_roi : npt.NDArray[np.floating]
        raster data portion centered on the target location
    right_ambiguity_roi : npt.NDArray[np.floating]
        raster data portion centered on the target right ambiguity
    left_ambiguity_roi : npt.NDArray[np.floating]
        raster data portion centered on the target left ambiguity
    config : AmbiguityRatioConfig
        ambiguity ratio computation configuration parameters

    Returns
    -------
    float
        DTAR in decibel
    """
    return compute_distributed_target_ambiguity_ratio_db(
        distributed_target_roi=point_target_roi,
        right_ambiguity_roi=right_ambiguity_roi,
        left_ambiguity_roi=left_ambiguity_roi,
    )


def ambiguity_ratio_computation_core(
    channel_data: ChannelData,
    target_location: tuple[int, int],
    left_ambiguity_location: tuple[int, int],
    right_ambiguity_location: tuple[int, int],
    ambiguity_ratio_computing_function: AmbiguityRatioComputingFunction,
    config: AmbiguityRatioConfig,
) -> float:
    """Computing Target Ambiguity Ratio given the coordinates of the target and its right and left ambiguities.

    Parameters
    ----------
    channel_data : ChannelData
        current product channel data object
    target_location : tuple[int, int]
        target location inside the raster in pixels, (azimuth_px, range_px)
    left_ambiguity_location : tuple[int, int]
        left ambiguity location inside the raster in pixels, (azimuth_px, range_px)
    right_ambiguity_location : tuple[int, int]
        right ambiguity location inside the raster in pixels, (azimuth_px, range_px)
    ambiguity_ratio_computing_function : AmbiguityRatioComputingFunction
        function to be used to compute the target ambiguity ratio
    config : AmbiguityRatioConfig
        ambiguity ratio computation configuration parameters

    Returns
    -------
    float
        Ambiguity Ratio in decibel
    """

    target_roi = channel_data.read_data(
        azimuth_index=np.round(target_location[0]).astype(int),
        range_index=np.round(target_location[1]).astype(int),
        cropping_size=config.cropping_size,
        output_radiometric_quantity=channel_data.radiometric_quantity,
    )
    right_ambiguity_roi = channel_data.read_data(
        azimuth_index=np.round(right_ambiguity_location[0]).astype(int),
        range_index=np.round(right_ambiguity_location[1]).astype(int),
        cropping_size=config.cropping_size,
        output_radiometric_quantity=channel_data.radiometric_quantity,
    )
    left_ambiguity_roi = channel_data.read_data(
        azimuth_index=np.round(left_ambiguity_location[0]).astype(int),
        range_index=np.round(left_ambiguity_location[1]).astype(int),
        cropping_size=config.cropping_size,
        output_radiometric_quantity=channel_data.radiometric_quantity,
    )
    ambiguity_ratio_db = ambiguity_ratio_computing_function(target_roi, right_ambiguity_roi, left_ambiguity_roi, config)
    return target_roi, right_ambiguity_roi, left_ambiguity_roi, ambiguity_ratio_db


def are_ambiguities_inside_scene(
    r_amb: tuple[float, float], l_amb: tuple[float, float], lines: int, samples: int
) -> bool:
    """Checking ambiguities are inside the scene.

    Parameters
    ----------
    r_amb : tuple[float, float]
        right ambiguity positions, (az pixel, rng pixel)
    l_amb : tuple[float, float]
        left ambiguity positions, (az pixel, rng pixel)
    lines : int
        number of lines in the scene
    samples : int
        number of samples in the scene

    Returns
    -------
    bool
        True if ambiguities inside the scene, False otherwise
    """
    return np.logical_and.reduce(
        (0 < r_amb[0] < lines, 0 < r_amb[1] < samples, 0 < l_amb[0] < lines, 0 < l_amb[1] < samples)
    )
