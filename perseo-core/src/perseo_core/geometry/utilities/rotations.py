# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Base Rotation transformations utilities"""

from __future__ import annotations

from typing import Literal, get_args

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation, Slerp

RotationOrder = Literal["YPR", "YRP", "PRY", "PYR", "RYP", "RPY"]


ROT_TRANSLATION_TABLE = str.maketrans({"Y": "Z", "P": "Y", "R": "X"})


def euler_angles_to_rotation(
    euler_angles_rad: ArrayLike,
    order: RotationOrder,
) -> Rotation:
    """Convert input Euler angles in radians, with their application rotation order to a SciPy Rotation object.

    This is the opposite of :py:func:`rotation_to_euler_angles`.

    Parameters
    ----------
    euler_angles_rad : ArrayLike
        euler angles in radians, with the same column order of the specified ``order``, with shape (3,) or (N, 3)
    order : "YPR", "YRP", "PRY", "PYR", "RYP", "RPY"
        rotation order

    Returns
    -------
    Rotation
        a SciPy rotation object equivalent to the input euler angles matrix with the specified order

    Examples
    --------

    single rotation

    >>> rotation = euler_angles_to_rotation("YPR", euler_angles=[[0, 0, np.deg2rad(30.0)]])
    >>> print(rotation.as_matrix())
    [[ 1.         0.         0.       ]
     [ 0.         0.8660254 -0.5      ]
     [ 0.         0.5        0.8660254]]

    multiple rotation

    >>> roll = np.deg2rad(np.arange(10, 26, 5, dtype=float))
    >>> euler_angles = np.stack([np.zeros_like(roll), np.zeros_like(roll), roll], axis=-1)
    >>> rotation = euler_angles_to_rotation("YPR", euler_angles=euler_angles)
    >>> print(rotation.as_matrix().shape)
     (4, 3, 3)

    rotation order as string

    >>> euler_angles_to_rotation("YPR", euler_angles=[[0, 0, np.deg2rad(30.0)]])
    """
    # euler_angles_rad = np.atleast_2d(euler_angles_rad)
    # assert euler_angles_rad.ndim == 2 and euler_angles_rad.shape[1] == 3
    # upper case / lower case axis character matters
    euler_sequence = order.translate(ROT_TRANSLATION_TABLE)
    if euler_angles_rad.ndim == 1:
        euler_angles = euler_angles_rad[["YPR".index(a) for a in order]]
    else:
        euler_angles = euler_angles_rad[:, ["YPR".index(a) for a in order]]
    return Rotation.from_euler(euler_sequence, euler_angles)


def rotation_to_euler_angles(rotation: Rotation, order: RotationOrder) -> np.ndarray:
    """Compute euler angles array from the Rotation object and its rotation order.

    This is the opposite of :py:func:`euler_angles_to_rotation`.

    Parameters
    ----------
    rotation : Rotation
        rotation matrix from which compute the euler angles
    order : "YPR", "YRP", "PRY", "PYR", "RYP", "RPY"
        rotation order corresponding to the provided rotation

    Returns
    -------
    np.ndarray
        euler angles array, (N, 3), columns being in the same rotation order provided as input
    """
    # upper case / lower case axis character matters
    if order not in get_args(RotationOrder):
        raise ValueError(f"Invalid rotation order {order}, must be one of '{', '.join(get_args(RotationOrder))}")

    euler_sequence = order.translate(ROT_TRANSLATION_TABLE)
    return rotation.as_euler(euler_sequence)


def quaternions_multiplication(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Quaternion multiplication (scalar-last).

    Parameters
    ----------
    q : np.ndarray
        First quaternions, with shape (N, 4)
    r : np.ndarray
        Second quaternions, with shape (N, 4)

    Returns
    -------
    np.ndarray
        Quaternions multiplication result
    """
    q, r = np.atleast_2d(q), np.atleast_2d(r)
    x1, y1, z1, w1 = q.T
    x2, y2, z2, w2 = r.T
    return np.column_stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


def compute_slerp_derivative(
    rotations: Rotation,
    times: np.ndarray,
    query_times: ArrayLike,
) -> Rotation:
    """Computing first derivative of provided rotations at query times. A Spherical Linear Interpolation of Rotations
    (SLERP) is used to compute the interpolator given the quaternions and the times.
    This function computes the exact derivative of the SLERP at the query times with piecewise constant angular
    velocity and discontinuous angular acceleration at keyframes.

    .. math::

        \\begin{aligned}
        \\dot{q}(t) = \\frac{1}{2}q(t)\\otimes[\\omega,0] \\
        \\omega = \\frac{\\theta}{\\Delta t} u \\
        \\end{aligned}

    where :math:`q(t)` is the SLERP interpolated quaternion at time :math:`t` between two quaternions :math:`q_0, q_1`,
    :math:`\\omega` is the angular velocity as pure quaternion :math:`[wx, wy, wz, 0]`, :math:`\\otimes` is the
    quaternion multiplication, :math:`u` is the unit rotation axis and :math:`\\theta` is the rotation angle between
    the two quaternions :math:`q_0, q_1`.

    Parameters
    ----------
    rotations : Rotation
        rotations to be interpolated
    times : np.ndarray
        relative time axis in seconds for the input rotations, monotonic increasing, with shape (N,)
    query_times : ArrayLike
        relative times in seconds at which to evaluate the derivative, float or array-like with shape (M,)

    Returns
    -------
    float | np.ndarray
        SLERP derivative at each input time
    """
    query_times = np.atleast_1d(query_times)
    slerp = Slerp(
        times=times,
        rotations=rotations,
    )
    interp_quaternions = slerp(query_times).as_quat()

    # Find segment indices (vectorized)
    idx = np.searchsorted(times, query_times) - 1
    idx = np.clip(idx, 0, len(times) - 2)

    t0, t1 = times[idx], times[idx + 1]
    dt = (t1 - t0)[:, None]  # (M,1)

    q0, q1 = rotations[idx], rotations[idx + 1]

    # Relative rotation per segment
    q_rel = q0.inv() * q1
    rotvec = q_rel.as_rotvec()  # (M,3)

    # Angular velocity (rad/sec)
    omega = np.zeros_like(rotvec)
    mask = np.linalg.norm(rotvec, axis=1) > 1e-12
    omega[mask] = rotvec[mask] / dt[mask]

    # Convert omega to quaternion form (M,4)
    omega_quat = np.zeros((len(query_times), 4))
    omega_quat[:, :3] = omega

    # Quaternion derivative: q_dot = 0.5 * q ⊗ omega_quat
    q_dot = 0.5 * quaternions_multiplication(interp_quaternions, omega_quat)

    return Rotation.from_quat(q_dot.squeeze())
