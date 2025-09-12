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
