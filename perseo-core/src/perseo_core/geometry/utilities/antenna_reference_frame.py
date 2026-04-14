# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Complex Rotation transformations utilities"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from perseo_core.geometry.utilities.rotations import RotationOrder, euler_angles_to_rotation, rotation_to_euler_angles


def compute_antenna_reference_frame_from_euler_angles(
    ypr_rad: npt.NDArray[np.floating],
    rotation_order: RotationOrder,
    initial_reference_frame_axis: Rotation,
) -> Rotation:
    """Compute the Antenna Reference Frame (ARF) from Euler angles (Yaw, Pitch and Roll).

    Parameters
    ----------
    ypr_rad : npt.NDArray[np.floating]
        euler angles in radians with shape (3,) or (N, 3) with yaw, pitch, roll order.
    rotation_order : RotationOrder
        order of applications of the rotations
    initial_reference_frame_axis : Rotation
        reference frame axis of the sensor as a scipy Rotation object

    Returns
    -------
    Rotation
        antenna reference frame for the sensor as a Rotation object
    """
    rotation = euler_angles_to_rotation(order=rotation_order, ypr_rad=ypr_rad)
    return initial_reference_frame_axis * rotation


def compute_euler_angles_from_antenna_reference_frame(
    antenna_reference_frame: Rotation,
    initial_reference_frame_axis: Rotation,
    rotation_order: RotationOrder,
) -> np.ndarray:
    """Compute euler angles (Yaw, Pitch and Roll) from Antenna Reference Frame (ARF).

    Parameters
    ----------
    antenna_reference_frame : Rotation
        antenna reference frame of the sensor, as a Rotation Scipy object
    initial_reference_frame_axis : Rotation
        reference frame axis of the sensor as a scipy Rotation object
    rotation_order : RotationOrder
        order of applications of the output euler angles

    Returns
    -------
    np.ndarray
        euler angles in radians with shape (3,) or (N, 3) with yaw, pitch, roll order.
    """
    num_of_reference_frames = len(initial_reference_frame_axis) if not initial_reference_frame_axis.single else 1
    num_of_antenna_reference_frames = len(antenna_reference_frame) if not antenna_reference_frame.single else 1

    if (
        num_of_reference_frames != num_of_antenna_reference_frames
        and num_of_reference_frames != 1
        and num_of_antenna_reference_frames != 1
    ):
        raise RuntimeError(
            f"input shape mismatch: init ref frame {num_of_reference_frames} != arf {num_of_antenna_reference_frames}"
        )
    rotation = initial_reference_frame_axis.inv() * antenna_reference_frame
    return rotation_to_euler_angles(order=rotation_order, rotation=rotation)
