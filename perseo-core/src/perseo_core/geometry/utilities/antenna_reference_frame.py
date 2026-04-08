# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Complex Rotation transformations utilities"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

from perseo_core.geometry.utilities.rotations import RotationOrder, euler_angles_to_rotation, rotation_to_euler_angles


# TODO: improve documentation
def compute_antenna_reference_frame_from_euler_angles(
    euler_angles_rad: ArrayLike,
    rotation_order: RotationOrder,
    initial_reference_frame_axis: Rotation,
) -> Rotation:
    """Computing the Antenna Reference Frame (ARF) from Euler Angles (YAW, PITCH and ROLL) giving a rotation order and
    an initial reference frame axis.

    Parameters
    ----------
    euler_angles_rad : ArrayLike
        euler angles in radians, columns being in the same order of the ``rotation_order``, with shape (3,) or (N, 3)
    rotation_order : RotationOrder
        rotation order for the euler angles
    initial_reference_frame_axis : Rotation
        reference frame axis of the sensor as a scipy Rotation object

    Returns
    -------
    Rotation
        antenna reference frame for the sensor as a Rotation object
    """
    rotation = euler_angles_to_rotation(order=rotation_order, euler_angles_rad=euler_angles_rad)
    return initial_reference_frame_axis * rotation


# TODO: improve documentation
def compute_euler_angles_from_antenna_reference_frame(
    antenna_reference_frame: Rotation,
    initial_reference_frame_axis: Rotation,
    rotation_order: RotationOrder,
) -> np.ndarray:
    """Compute euler angles (YAW, PITCH and ROLL) from Antenna Reference Frame (ARF), the initial reference frame and
    rotation order.

    Parameters
    ----------
    antenna_reference_frame : np.ndarray
        antenna reference frame of the sensor, as a Rotation Scipy object
    initial_reference_frame_axis : np.ndarray
        reference frame axis of the sensor as a scipy Rotation object
    rotation_order : RotationOrder
        rotation order for the output euler angles

    Returns
    -------
    np.ndarray
        euler angles array in radians, with shape (N, 3), columns being in the same rotation order requested as input
    """
    try:
        ref_num = len(initial_reference_frame_axis)
    except TypeError:
        ref_num = 1
    try:
        arf_num = len(antenna_reference_frame)
    except TypeError:
        arf_num = 1
    if ref_num != arf_num:
        raise RuntimeError(f"input shape mismatch: init ref frame {ref_num} != arf {arf_num}")
    rotation = initial_reference_frame_axis.inv() * antenna_reference_frame
    return rotation_to_euler_angles(order=rotation_order, rotation=rotation)
