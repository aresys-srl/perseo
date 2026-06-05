# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

r"""Antenna Reference Frame (ARF) module.

This module provides functions to compute the Antenna Reference Frame (ARF) from rotations
and Euler angles, and to extract Euler angles from a given ARF.

!!! note

    The Antenna Reference Frame (ARF) is a 3x3 matrix that represents the orientation
    of the antenna reference frame.

### ARF Matrix Structure

Each column of the ARF matrix represents one axis of the antenna frame expressed in the
local frame's coordinates:

$$
ARF = [X_{antenna} | Y_{antenna} | Z_{antenna}]
$$

Where:

- X_antenna: X-axis of the antenna frame (usually along-track direction)
- Y_antenna: Y-axis of the antenna frame (cross-track direction)
- Z_antenna: Z-axis of the antenna frame (boresight direction)

### Accessing Antenna Reference Frame unit vectors

Individual axes (columns) can be accessed with:

```python
X_antenna = arf[..., 0]  # First column (X-axis)
Y_antenna = arf[..., 1]  # Second column (Y-axis)
Z_antenna = arf[..., 2]  # Third column (Z-axis)
```

### Construction via Euler Angles

The ARF is typically constructed from three sequential rotations (Euler angles):

1. Yaw (Z-axis rotation): Rotation around the local Z-axis
2. Pitch (Y-axis rotation): Rotation around the local Y-axis
3. Roll (X-axis rotation): Rotation around the local X-axis

These rotations are applied in certain order (usually YPR order (Yaw-Pitch-Roll)) to progressively
rotate the initial reference frame into the antenna reference frame.

### Vector Transformation

To rotate a vector v from the local frame to the antenna frame:

$$
v_{antenna} = ARF \, @ \, v_{local}
$$

To rotate back from the antenna frame to the local frame (using transpose/inverse):

$$
v_{local} = ARF.T \, @ \, v_{antenna}
$$

This is crucial for SAR processing where understanding antenna orientation relative to
satellite motion and Earth's surface is essential.

### Examples

Basic usage:

```python
import numpy as np

from perseo_core.geometry.pointing import (
    compute_antenna_reference_frame_from_euler_angles,
    compute_sensor_local_axis,
)

# Define rotation angles
ypr_deg = np.array([5, 1, 30])  # yaw, pitch, roll in degrees
ypr_rad = np.deg2rad(ypr_deg)

# Sensor parameters in ECEF
sensor_positions = np.array([26512.279931507, 1064819.379506800, -7083173.555337110])
sensor_velocities = np.array([7529.609430015988, -342.978175622686, -23.376907795264])

# Compute local axis (ZERODOPPLER reference frame in ECEF)
local_axis = compute_sensor_local_axis(
    sensor_positions=sensor_positions,
    sensor_velocities=sensor_velocities,
    reference_frame="ZERODOPPLER",
)

# Compute antenna reference frame (ARF) from Euler angles
arf = compute_antenna_reference_frame_from_euler_angles(
    rotation_order="YPR",
    ypr_rad=ypr_rad,
    initial_reference_frame=local_axis,
)

# Access individual axes
print("Antenna Reference Frame:")
print("  X-axis:", arf[..., 0])
print("  Y-axis:", arf[..., 1])
print("  Z-axis (boresight):", arf[..., 2])
```
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from perseo_core.geometry.pointing.rotations import RotationOrder, euler_angles_to_rotation, rotation_to_euler_angles


def compute_antenna_reference_frame_from_rotation(
    rotation: Rotation,
    initial_reference_frame: npt.NDArray[np.floating] | None = None,
) -> npt.NDArray[np.floating]:
    """Compute the Antenna Reference Frame (ARF) from a given rotation.

    The ARF is obtained by rotating the initial reference frame with the given rotation.
    The rotation is assumed to be defined along the initial reference frame axes.

    Parameters
    ----------
    rotation : Rotation
        the rotation axes implicitely defined inside the rotation are the initial_reference_frame
        axes.
    initial_reference_frame : npt.NDArray[np.floating], optional
        initial reference frame axes with shape (3, 3) or (N, 3, 3). If None, the initial reference frame
        is assumed to be the identity matrix.

    Returns
    -------
    npt.NDArray[np.floating]
        antenna reference frame axes as a numpy array with shape (3, 3) or (N, 3, 3)

    """
    # The full process would be:
    # 1. convert the initial_reference_frame axes from their own components to
    #    the components in the initial_reference_frame. This results in the identity matrix
    #    regardless of the initial_reference_frame.
    # 2. Now the 3 axes we want to rotate are described by x,y,z components that are compatible with the rotation.
    # 3. We apply the rotation to retrieve the antenna reference frame in initial_reference frame coordinates:
    #       arf_initial_frame_coords = rotation.as_matrix() @ np.eye(3) = rotation.as_matrix()
    # 4. Finally, we convert the antenna reference frame from initial_reference_frame coordinates to the original
    #    coordinates of the initial_reference_frame by applying the change of basis:
    #      arf_original_coords = initial_reference_frame mat mul arf_initial_frame_coords
    # The compact formula is used.

    arf_in_initial_frame_coords = rotation.as_matrix()
    return (
        initial_reference_frame @ arf_in_initial_frame_coords
        if initial_reference_frame is not None
        else arf_in_initial_frame_coords
    )


def compute_rotation_from_antenna_reference_frame(
    antenna_reference_frame: npt.NDArray[np.floating],
    initial_reference_frame: npt.NDArray[np.floating] | None = None,
) -> Rotation:
    """Compute the rotation from a given Antenna Reference Frame (ARF).

    Parameters
    ----------
    antenna_reference_frame : npt.NDArray[np.floating]
        antenna reference frame of the sensor, with shape (3, 3) or (N, 3, 3)
    initial_reference_frame : npt.NDArray[np.floating], optional
        reference frame of the sensor, with shape (3, 3) or (N, 3, 3). If None, the initial reference frame
        is assumed to be the identity matrix.

    Returns
    -------
    Rotation
        rotation from the initial reference frame to the antenna reference frame

    """
    if initial_reference_frame is None:
        initial_reference_frame = np.eye(3)

    # The reasoning is the opposite of the one in compute_antenna_reference_frame_from_rotation.
    # 1. We convert the antenna reference frame from the original coordinates to the initial_reference_frame coordinates
    #    by applying the change of basis:
    #      arf_initial_frame_coords = initial_reference_frame.T mat mul antenna_reference_frame
    # 2. Now the 3 axes of the antenna reference frame are described by x,y,z components that are the axis along which
    #    we want to define
    #       rotation = Rotation.from_matrix(initial_reference_frame.T mat mul antenna_reference_frame)

    initial_ref_transposed = np.swapaxes(initial_reference_frame, -2, -1)
    return Rotation.from_matrix(initial_ref_transposed @ antenna_reference_frame)


def compute_antenna_reference_frame_from_euler_angles(
    ypr_rad: npt.NDArray[np.floating],
    rotation_order: RotationOrder,
    initial_reference_frame: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Compute the Antenna Reference Frame (ARF) from Euler angles (Yaw, Pitch and Roll).

    Parameters
    ----------
    ypr_rad : npt.NDArray[np.floating]
        euler angles in radians with shape (3,) or (N, 3) with yaw, pitch, roll order.
    rotation_order : RotationOrder
        order of applications of the rotations
    initial_reference_frame : npt.NDArray[np.floating]
        reference frame of the sensor, with shape (3, 3) or (N, 3, 3)

    Returns
    -------
    npt.NDArray[np.floating]
        antenna reference frame as a numpy array with shape (3, 3) or (N, 3, 3)

    """
    rotation = euler_angles_to_rotation(order=rotation_order, ypr_rad=ypr_rad)
    return compute_antenna_reference_frame_from_rotation(
        rotation=rotation, initial_reference_frame=initial_reference_frame
    )


def compute_euler_angles_from_antenna_reference_frame(
    antenna_reference_frame: npt.NDArray[np.floating],
    initial_reference_frame: npt.NDArray[np.floating],
    rotation_order: RotationOrder,
) -> npt.NDArray[np.floating]:
    """Compute euler angles (Yaw, Pitch and Roll) from Antenna Reference Frame (ARF).

    Parameters
    ----------
    antenna_reference_frame : npt.NDArray[np.floating]
        antenna reference frame of the sensor, with shape (3, 3) or (N, 3, 3)
    initial_reference_frame : npt.NDArray[np.floating]
        reference frame of the sensor, with shape (3, 3) or (N, 3, 3)
    rotation_order : RotationOrder
        order of applications of the output euler angles

    Returns
    -------
    npt.NDArray[np.floating]
        euler angles in radians with shape (3,) or (N, 3) with yaw, pitch, roll order.

    """
    rotation = compute_rotation_from_antenna_reference_frame(
        antenna_reference_frame=antenna_reference_frame,
        initial_reference_frame=initial_reference_frame,
    )
    return rotation_to_euler_angles(order=rotation_order, rotation=rotation)


def compute_pointing_directions(
    antenna_reference_frame: npt.NDArray[np.floating],
    azimuth_antenna_angles: float | npt.NDArray[np.floating],
    elevation_antenna_angles: float | npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Compute the pointing directions corresponding to the given antenna angles.

    Parameters
    ----------
    antenna_reference_frame : npt.NDArray[np.floating]
        antenna reference frame for the sensor as a numpy array with shape (3, 3) or (N, 3, 3)
    azimuth_antenna_angles : float | npt.NDArray[np.floating]
        scalar or (N,) array like, in radians
    elevation_antenna_angles : float | npt.NDArray[np.floating]
        scalar or (N,) array like, in radians

    Returns
    -------
    npt.NDArray[np.floating]
        pointing directions, with shape (3,) or (N, 3)

    """
    if np.shape(azimuth_antenna_angles) != np.shape(elevation_antenna_angles):
        broadcast_shape = np.broadcast_shapes(np.shape(azimuth_antenna_angles), np.shape(elevation_antenna_angles))
        azimuth_antenna_angles = np.broadcast_to(azimuth_antenna_angles, broadcast_shape)
        elevation_antenna_angles = np.broadcast_to(elevation_antenna_angles, broadcast_shape)

    ux = np.tan(azimuth_antenna_angles)
    uy = np.tan(elevation_antenna_angles)
    uz = np.ones_like(ux)
    local_directions = np.stack([ux, uy, uz], axis=-1)
    local_directions = local_directions / np.linalg.norm(local_directions, axis=-1, keepdims=True)

    return np.einsum("...ij,...j->...i", antenna_reference_frame, local_directions)


__all__ = [
    "compute_antenna_reference_frame_from_euler_angles",
    "compute_euler_angles_from_antenna_reference_frame",
    "compute_pointing_directions",
]
