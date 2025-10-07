# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Common utilities across different analyses"""

import numpy as np
import pandas as pd

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
    range_axis: np.ndarray,
    lines_per_burst: np.ndarray,
    default_block_size: int,
) -> tuple[int, int, list[tuple[int, int]]]:
    """Defining the blocks partitioning of the whole scene.

    Parameters
    ----------
    azimuth_axis : np.ndarray
        azimuth axis of the whole scene
    range_axis : np.ndarray
        range axis of the whole scene
    lines_per_burst : np.ndarray
        lines per burst array
    default_block_size : int
        default block size value, needed for stripmap case

    Returns
    -------
    int
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
        block_size = lines_per_burst[0]
        blocks_num = lines_per_burst.size  # number of bursts

    blocks_centers_px = np.arange(block_size // 2, block_size * blocks_num, block_size).tolist()
    blocks_centers_px = [(px, mid_range_pixel) for px in blocks_centers_px]

    return block_size, blocks_num, blocks_centers_px


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
