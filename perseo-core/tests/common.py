# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Common testing utilities"""

import numpy as np


def compute_antenna_angles_a_posteriori(
    antenna_reference_frame: np.ndarray, vectors: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute antenna azimuth/elevation from frame and direction vectors.

    Parameters
    ----------
    antenna_reference_frame : np.ndarray
        Normalized local reference frame.
    vectors : np.ndarray
        Input look vectors.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Azimuth and elevation arrays.
    """
    vectors = vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)
    local_components = np.einsum("...jk, ...j->...k", antenna_reference_frame, vectors)
    azimuth_angles = np.arctan(local_components[..., 0] / local_components[..., 2])
    elevation_angles = np.arctan(local_components[..., 1] / local_components[..., 2])
    return azimuth_angles, elevation_angles
