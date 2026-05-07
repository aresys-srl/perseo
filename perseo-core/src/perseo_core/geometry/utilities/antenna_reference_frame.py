# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Antenna Reference Frame (ARF) computation utilities."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from perseo_core.geometry.utilities.rotations import RotationOrder, euler_angles_to_rotation, rotation_to_euler_angles


def compute_antenna_reference_frame_from_rotation(
    rotation: Rotation,
    initial_reference_frame_axis: npt.NDArray[np.floating] | None = None,
) -> npt.NDArray[np.floating]:
    """Compute the Antenna Reference Frame (ARF) from a given rotation.

    Parameters
    ----------
    rotation : Rotation
        rotation to be applied to the initial reference frame axis
    initial_reference_frame_axis : npt.NDArray[np.floating], optional
        reference frame axis of the sensor, with shape (3, 3) or (N, 3, 3). If None, the initial reference frame axis
        is assumed to be the identity matrix.

    Returns
    -------
    npt.NDArray[np.floating]
        antenna reference frame for the sensor as a numpy array with shape (3, 3) or (N, 3, 3)
    """
    if initial_reference_frame_axis is None:
        initial_reference_frame_axis = np.eye(3)
    return initial_reference_frame_axis @ rotation.as_matrix()


def compute_rotation_from_antenna_reference_frame(
    antenna_reference_frame: npt.NDArray[np.floating],
    initial_reference_frame_axis: npt.NDArray[np.floating] | None = None,
) -> Rotation:
    """Compute the rotation from a given Antenna Reference Frame (ARF).

    Parameters
    ----------
    antenna_reference_frame : npt.NDArray[np.floating]
        antenna reference frame of the sensor, with shape (3, 3) or (N, 3, 3)
    initial_reference_frame_axis : npt.NDArray[np.floating], optional
        reference frame axis of the sensor, with shape (3, 3) or (N, 3, 3). If None, the initial reference frame axis
        is assumed to be the identity matrix.

    Returns
    -------
    Rotation
        rotation from the initial reference frame axis to the antenna reference frame
    """
    if initial_reference_frame_axis is None:
        initial_reference_frame_axis = np.eye(3)

    num_of_reference_frames = initial_reference_frame_axis.size // 9
    num_of_antenna_reference_frames = antenna_reference_frame.size // 9
    if (
        num_of_reference_frames != num_of_antenna_reference_frames
        and num_of_reference_frames != 1
        and num_of_antenna_reference_frames != 1
    ):
        raise RuntimeError(
            f"input shape mismatch: init ref frame {num_of_reference_frames} != arf {num_of_antenna_reference_frames}"
        )

    # Transpose last two dimensions to handle both 2D and 3D arrays
    initial_ref_transposed = np.swapaxes(initial_reference_frame_axis, -2, -1)
    return Rotation.from_matrix(initial_ref_transposed @ antenna_reference_frame)


def compute_antenna_reference_frame_from_euler_angles(
    ypr_rad: npt.NDArray[np.floating],
    rotation_order: RotationOrder,
    initial_reference_frame_axis: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Compute the Antenna Reference Frame (ARF) from Euler angles (Yaw, Pitch and Roll).

    Parameters
    ----------
    ypr_rad : npt.NDArray[np.floating]
        euler angles in radians with shape (3,) or (N, 3) with yaw, pitch, roll order.
    rotation_order : RotationOrder
        order of applications of the rotations
    initial_reference_frame_axis : npt.NDArray[np.floating]
        reference frame axis of the sensor, with shape (3, 3) or (N, 3, 3)

    Returns
    -------
    npt.NDArray[np.floating]
        antenna reference frame for the sensor as a numpy array with shape (3, 3) or (N, 3, 3)
    """
    rotation = euler_angles_to_rotation(order=rotation_order, ypr_rad=ypr_rad)
    return compute_antenna_reference_frame_from_rotation(
        rotation=rotation, initial_reference_frame_axis=initial_reference_frame_axis
    )


def compute_euler_angles_from_antenna_reference_frame(
    antenna_reference_frame: npt.NDArray[np.floating],
    initial_reference_frame_axis: npt.NDArray[np.floating],
    rotation_order: RotationOrder,
) -> np.ndarray:
    """Compute euler angles (Yaw, Pitch and Roll) from Antenna Reference Frame (ARF).

    Parameters
    ----------
    antenna_reference_frame : npt.NDArray[np.floating]
        antenna reference frame of the sensor, with shape (3, 3) or (N, 3, 3)
    initial_reference_frame_axis : npt.NDArray[np.floating]
        reference frame axis of the sensor, with shape (3, 3) or (N, 3, 3)
    rotation_order : RotationOrder
        order of applications of the output euler angles

    Returns
    -------
    np.ndarray
        euler angles in radians with shape (3,) or (N, 3) with yaw, pitch, roll order.
    """
    rotation = compute_rotation_from_antenna_reference_frame(
        antenna_reference_frame=antenna_reference_frame,
        initial_reference_frame_axis=initial_reference_frame_axis,
    )
    return rotation_to_euler_angles(order=rotation_order, rotation=rotation)
