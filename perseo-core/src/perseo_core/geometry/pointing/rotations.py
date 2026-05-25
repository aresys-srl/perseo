# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Base Rotation transformations utilities"""

from __future__ import annotations

from typing import Literal, get_args

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

RotationOrder = Literal["YPR", "YRP", "PRY", "PYR", "RYP", "RPY"]


_ROT_TRANSLATION_TABLE = str.maketrans({"Y": "Z", "P": "Y", "R": "X"})


def euler_angles_to_rotation(
    ypr_rad: npt.NDArray[np.floating],
    order: RotationOrder,
) -> Rotation:
    """Convert input Euler angles in radians to a SciPy Rotation object.

    This is the reverse operation of
    [`rotation_to_euler_angles`][perseo_core.geometry.pointing.rotations.rotation_to_euler_angles].

    Parameters
    ----------
    ypr_rad : npt.NDArray[np.floating]
        euler angles in radians with shape (3,) or (N, 3) with yaw, pitch, roll order.
    order : "YPR", "YRP", "PRY", "PYR", "RYP", "RPY"
        order of application of the rotations corresponding to the provided euler angles.

    Returns
    -------
    Rotation
        a SciPy rotation object equivalent to the input euler angles matrix with the specified order

    Examples
    --------

    single rotation

    >>> rotation = euler_angles_to_rotation(order="YPR", ypr_rad=[[0, 0, np.deg2rad(30.0)]])
    >>> print(rotation.as_matrix())
    [[ 1.         0.         0.       ]
     [ 0.         0.8660254 -0.5      ]
     [ 0.         0.5        0.8660254]]

    multiple rotation

    >>> roll = np.deg2rad(np.arange(10, 26, 5, dtype=float))
    >>> euler_angles = np.stack([np.zeros_like(roll), np.zeros_like(roll), roll], axis=-1)
    >>> rotation = euler_angles_to_rotation(order="YPR", ypr_rad=euler_angles)
    >>> print(rotation.as_matrix().shape)
     (4, 3, 3)

    rotation order as string

    >>> euler_angles_to_rotation("YPR", euler_angles=[[0, 0, np.deg2rad(30.0)]])
    """
    euler_sequence = order.translate(_ROT_TRANSLATION_TABLE)
    euler_angles = ypr_rad[..., ["YPR".index(rotation_axis) for rotation_axis in order]]
    return Rotation.from_euler(euler_sequence, euler_angles)


def rotation_to_euler_angles(rotation: Rotation, order: RotationOrder) -> npt.NDArray[np.floating]:
    """Compute euler angles array from the Rotation object and its rotation order.

    This is the reverse operation of
    [`euler_angles_to_rotation`][perseo_core.geometry.pointing.rotations.euler_angles_to_rotation].

    Parameters
    ----------
    rotation : Rotation
        rotation matrix from which compute the euler angles
    order : "YPR", "YRP", "PRY", "PYR", "RYP", "RPY"
        rotation order corresponding to the provided rotation

    Returns
    -------
    npt.NDArray[np.floating]
        euler angles array, (N, 3) with yaw, pitch, roll order.
    """
    # upper case / lower case axis character matters
    if order not in get_args(RotationOrder):
        raise ValueError(f"Invalid rotation order {order}, must be one of '{', '.join(get_args(RotationOrder))}")

    euler_sequence = order.translate(_ROT_TRANSLATION_TABLE)
    euler_angles = rotation.as_euler(euler_sequence)

    indices = [order.index(axis) for axis in "YPR"]
    return euler_angles[..., indices]
