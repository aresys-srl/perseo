# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Complex Rotation transformations utilities"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import transform

from perseo_core.geometry.utilities import RotationOrderLike
from perseo_core.geometry.utilities.rotations import euler_angles_to_rotation, rotation_to_euler_angles


# TODO: improve documentation
def compute_antenna_reference_frame_from_euler_angles(
    order: RotationOrderLike,
    initial_reference_frame_axis: np.ndarray,
    euler_angles_rad: ArrayLike,
) -> np.ndarray:
    """Computing the Antenna Reference Frame (ARF) from Euler Angles (YAW, PITCH and ROLL) giving a rotation order and
    an initial reference frame axis.

    Parameters
    ----------
    order : RotationOrderLike
        rotation order for the euler angles
    initial_reference_frame_axis : np.ndarray
        reference frame axis of the sensor
    euler_angles_rad : ArrayLike
        euler angles in radians, (3,) or (N, 3)

    Returns
    -------
    np.ndarray
        antenna reference frame for the sensor
    """
    rotation = euler_angles_to_rotation(order=order, euler_angles_rad=euler_angles_rad)
    return np.matmul(initial_reference_frame_axis, rotation.as_matrix())


# TODO: improve documentation
def compute_euler_angles_from_antenna_reference_frame(
    initial_reference_frame_axis: np.ndarray,
    antenna_reference_frame: np.ndarray,
    order: RotationOrderLike,
) -> np.ndarray:
    """Compute euler angles (YAW, PITCH and ROLL) from Antenna Reference Frame (ARF), the initial reference frame and
    rotation order.

    Parameters
    ----------
    initial_reference_frame_axis : np.ndarray
        initial reference frame axis of the sensor, (3, 3) or (N, 3, 3)
    antenna_reference_frame : np.ndarray
        antenna reference frame of the sensor, (3, 3) or (N, 3, 3)
    order : RotationOrderLike
        rotation order

    Returns
    -------
    np.ndarray
        euler angles array, (N, 3), columns being in the same rotation order provided as input
    """
    if initial_reference_frame_axis.shape != antenna_reference_frame.shape:
        raise RuntimeError(
            f"input shape mismatch: init ref frame {initial_reference_frame_axis.shape} != arf "
            + f"{antenna_reference_frame.shape}"
        )
    init_ref_frame = (
        np.transpose(initial_reference_frame_axis, (0, 2, 1))
        if initial_reference_frame_axis.ndim == 3
        else initial_reference_frame_axis.T
    )
    rotation = transform.Rotation.from_matrix(np.matmul(init_ref_frame, antenna_reference_frame))

    return rotation_to_euler_angles(order=order, rotation=rotation)
