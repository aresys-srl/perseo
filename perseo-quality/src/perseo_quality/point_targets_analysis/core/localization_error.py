# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Point Target Localization Error analysis module"""

from __future__ import annotations

import numpy as np


def compute_localization_errors_pixels(
    target_pos_real: np.ndarray, target_pos_ref: np.ndarray
) -> tuple[float, float, float]:
    """Compute localization errors for slant range, azimuth and ground range directions in pixels.

    Parameters
    ----------
    target_pos_real : np.ndarray
        real target peak position identified in the raster data
    target_pos_ref : np.ndarray
        nominal reference target position as provided by the external calibration file

    Returns
    -------
    float
        slant range localization error in pixels
    float
        azimuth localization error in pixels
    float
        ground range localization error in pixels
    """
    slant_range_localization_error = target_pos_real[0] - target_pos_ref[0]
    azimuth_localization_error = target_pos_real[1] - target_pos_ref[1]
    return slant_range_localization_error, azimuth_localization_error, slant_range_localization_error
