# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Common utilities across different analyses"""

import numpy as np
import numpy.typing as npt
import pandas as pd
from perseo_core.geometry.coordinates import llh2xyz, xyz2llh
from perseo_core.geometry.geocoding import direct_geocoding_monostatic
from perseo_core.geometry.navigation import Trajectory
from perseo_core.timing import PreciseDateTime

from perseo_quality.core.generic_dataclasses import PointTargetVisibility
from perseo_quality.io.point_targets import PointTarget
from perseo_quality.io.quality_input_protocol import QualityInputProduct


def check_targets_visibility(product: QualityInputProduct, point_targets: list[PointTarget]) -> pd.DataFrame:
    """Checking if a set of targets is seen by the sensor in the recorded swath.

    Parameters
    ----------
    product : ProductManager
        product folder ProductManager instance
    point_targets : list[PointTarget]
        list of PointTarget objects

    Returns
    -------
    pd.DataFrame
        pandas dataframe collecting all visible points
    """

    coordinates = np.vstack([c.xyz_coordinates for c in point_targets])
    target_ids = [c.name for c in point_targets]

    valid_points = []
    # iterating over channels
    for channel in product.channels_list:
        channel_data = product.get_channel_data(channel_id=channel)

        bursts_associations = channel_data.ground_points_to_burst_association(coordinates=coordinates)

        for index, item in enumerate(bursts_associations):
            valid_points.append(
                PointTargetVisibility(
                    id=target_ids[index],
                    channel=channel,
                    burst=item,
                    swath=channel_data.swath_name,
                    polarization=channel_data.polarization.name,
                )
            )

    return pd.DataFrame(valid_points)


def blocks_partitioning(
    azimuth_axis: np.ndarray,
    range_axis: npt.NDArray[np.floating],
    lines_per_burst: np.ndarray,
    default_block_size: int,
) -> tuple[np.ndarray, int, list[tuple[int, int]]]:
    """Defining the blocks partitioning of the whole scene.

    Parameters
    ----------
    azimuth_axis : np.ndarray
        azimuth axis of the whole scene
    range_axis : npt.NDArray[np.floating]
        range axis of the whole scene
    lines_per_burst : np.ndarray
        lines per burst array
    default_block_size : int
        default block size value, needed for stripmap case

    Returns
    -------
    np.ndarray
        size of each block
    int
        number of partitioning blocks
    list[tuple[int, int]]
        pixel coordinates of blocks centers (azimuth and range pixel values)
    """
    block_size = default_block_size
    blocks_num = int(np.floor(azimuth_axis.size / block_size))
    mid_range_pixel = int(range_axis.size // 2)

    if lines_per_burst.size > 1:
        # TOPSAR/SCANSAR case: blocks set using bursts
        block_size = lines_per_burst
        blocks_num = lines_per_burst.size  # number of bursts
        blocks_centers_px = [
            (int(a + b // 2), mid_range_pixel)
            for a, b in zip(np.concatenate([[0], np.cumsum(block_size)[:-1]], dtype=int), lines_per_burst, strict=True)
        ]
        return block_size, blocks_num, blocks_centers_px

    blocks_centers_px = np.arange(block_size // 2, block_size * blocks_num, block_size).tolist()
    blocks_centers_px = [(px, mid_range_pixel) for px in blocks_centers_px]

    return np.array([block_size] * blocks_num), blocks_num, blocks_centers_px


# TODO: remove this using new layout?
def detect_burst_from_pixel(lines_per_burst: np.ndarray, azimuth_px: int) -> int:
    """Detect the burst belonging to the selected azimuth pixel value.

    Parameters
    ----------
    lines_per_burst : np.ndarray
        lines per burst
    azimuth_px : int
        selected azimuth pixel

    Returns
    -------
    int
        burst
    """

    if lines_per_burst.size > 1:
        cumulative_burst = np.cumsum([0] + list(lines_per_burst))
        pixel_diff = np.ma.masked_less(azimuth_px - cumulative_burst, 0)
        return np.argmin(pixel_diff)

    return 0


def angles_computation_setup(
    trajectory: Trajectory,
    azimuth_time: PreciseDateTime,
    range_values: npt.NDArray[np.floating],
    look_direction: str,
    altitude: float = 0.0,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Setting up the stage to compute incidence and look angles by computing sensor position, ground points and nadir
    direction.

    Parameters
    ----------
    trajectory : Trajectory
        sensor trajectory
    azimuth_time : PreciseDateTime
        azimuth time at which compute the output
    range_values : npt.NDArray[np.floating]
        range values for which compute values
    look_direction : str
        sensor look direction
    altitude : float, optional
        altitude over WGS84 ellipsoid, by default 0.0

    Returns
    -------
    npt.NDArray[np.floating]
        sensor position
    npt.NDArray[np.floating]
        ground points
    npt.NDArray[np.floating]
        nadir direction
    """
    sensor_pos = trajectory.position(azimuth_time)
    sensor_vel = trajectory.velocity(azimuth_time)

    ground_points = direct_geocoding_monostatic(
        sensor_positions=sensor_pos,
        sensor_velocities=sensor_vel,
        range_times=range_values,
        doppler_frequencies=0,
        wavelength=1,
        look_direction=look_direction,
        altitude=altitude,
    )

    sensor_position_ground = xyz2llh(sensor_pos)
    sensor_position_ground[2] = 0.0
    sensor_position_ground = llh2xyz(sensor_position_ground).squeeze()

    nadir = sensor_position_ground - sensor_pos
    return sensor_pos, ground_points, nadir
