# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
This module provides the `Attitude` class, a wrapper around scipy's Spherical Linear
Interpolation (SLERP) for smooth attitude interpolation. The attitude can be initialized from reference frames,
quaternions, or Euler angles, and supports interpolation at arbitrary times within the defined domain.

### Key Components

- `Attitude`: Main class for attitude interpolation using SLERP.

    - Initialized with reference frames (N, 3, 3) and corresponding times
    - Supports interpolation via `evaluate(time)` method
    - Properties: `reference_frames`, `times`, `domain`

- Factory methods for creating Attitude from different representations:

    - `from_quaternions`: Create from scalar-last quaternions (N, 4)
    - `from_euler_angles`: Create from yaw-pitch-roll angles (N, 3) with specified rotation order

- Utility functions for computing attitude from sensor data:

    - `compute_antenna_attitude_from_euler_angles`: antenna attitude from Euler angles relative to sensor local axis
    - `compute_sensor_attitude_from_state_vectors`: sensor attitude directly from position/velocity vectors

All attitude reference frames are expressed as (N, 3, 3) matrices. The reference frame system is preserved through
interpolation (e.g., ECEF input yields ECEF output).
"""

from __future__ import annotations

from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation, Slerp

from perseo_core.geometry.pointing.antenna_reference_frame import compute_antenna_reference_frame_from_euler_angles
from perseo_core.geometry.pointing.reference_frames import ReferenceFrame, compute_sensor_local_axis
from perseo_core.geometry.pointing.rotations import (
    RotationOrder,
    euler_angles_to_rotation,
)

T = TypeVar("T", bound=np.generic)


class Attitude(Generic[T]):
    """Attitude based on a Spherical Linear Interpolation of Rotations (SLERP)"""

    def __init__(
        self,
        reference_frames: npt.NDArray[np.floating],
        times: npt.NDArray[T],
    ) -> None:
        """Create an Attitude SLERP interpolator from times and attitude reference frames.

        Time axis can be specified as relative or absolute (actual dates).

        Attitude reference frames must be expressed as an array with shape (N, 3, 3),
        with N being the same length as the times array.

        Parameters
        ----------
        reference_frames : npt.NDArray[np.floating]
            array of attitude reference frames expressed in a reference System (i.e. ECEF), with shape (N, 3, 3)
            the interpolated reference frames will be in the same reference system of the input reference frames.
        times : npt.NDArray[T]
            relative or absolute (actual dates) time axis, monotonic increasing, with shape (N,)
            interpolation times must be of the same type as the initialization times axis
        """
        if reference_frames.ndim not in (2, 3) or reference_frames.shape[-2:] != (3, 3):
            raise ValueError("reference_frames must have shape (3, 3) or (N, 3, 3)")

        n_frames = 1 if reference_frames.ndim == 2 else reference_frames.shape[0]
        if n_frames != len(times):
            raise ValueError("reference_frames and times must contain the same number of samples")

        self._rotations = Rotation.from_matrix(reference_frames)

        self._times = times

        self._domain = (times[0], times[-1])
        self._slerp = Slerp(
            times=np.array(times - times[0], dtype=np.float64),
            rotations=self._rotations,
        )

    @property
    def reference_frames(self) -> npt.NDArray[np.floating]:
        """Attitude reference frames used for interpolation"""
        return self._rotations.as_matrix()

    @property
    def times(self) -> npt.NDArray[T]:
        """Attitude times vector"""
        return self._times

    @property
    def domain(self) -> tuple[T, T]:
        """Attitude time domain"""
        return self._domain

    def _check_time_validity(self, time: T | npt.NDArray[T]) -> None:
        """Check input times validity with respect to attitude time domain."""
        if np.any(time < self.domain[0]) or np.any(time > self.domain[1]):  # type: ignore
            raise RuntimeError("One (or more) of the input times is outside of attitude time domain.")

    def evaluate(self, time: T | npt.NDArray[T]) -> npt.NDArray[np.floating]:
        """Retrieve attitude reference frame at given times.

        Parameters
        ----------
        time: T | npt.NDArray[T]
            time of the same type of the initialization times axis. Can be a scalar or an array of shape (M,)

        Returns
        -------
        npt.NDArray[np.floating]
            interpolated attitude reference frame at each input time as a numpy array with shape (3, 3) or (M, 3, 3)
            depending on the input time shape. The interpolated reference frames are expressed in the same
            reference system of the input reference frames
        """
        self._check_time_validity(time)
        relative_times: npt.NDArray[np.floating] = time - self.domain[0]  # type: ignore
        return self._slerp(relative_times).as_matrix()

    @classmethod
    def from_quaternions(cls, quaternions: npt.NDArray[np.floating], times: npt.NDArray[T]) -> Attitude:
        """Create an Attitude from quaternions.

        Time axis can be specified as relative or absolute
        (actual dates), while quaternions must be expressed as an array with shape (N, 4), with N being the same length
        as the times array.

        !!! warning "Reference frame consistency"

            The reference frame in which quaternions are provided is kept as the reference frame of the interpolator,
            so the interpolated reference frames will be in the same reference system of the input quaternions.

            For example, if the quaternions are defined as rotations from ECEF to the antenna reference frame,
            the interpolated reference frames will be in ECEF as well.

        Parameters
        ----------
        quaternions : npt.NDArray[np.floating]
            quaternions, scalar-last, with shape (N, 4)
        times : npt.NDArray[T]
            relative or absolute (actual dates) time axis, monotonic increasing, with shape (N,)

        Returns
        -------
        Attitude
            interpolator object
        """
        return cls(reference_frames=Rotation.from_quat(quaternions).as_matrix(), times=times)

    @classmethod
    def from_euler_angles(
        cls, euler_angles_rad: npt.NDArray[np.floating], rotation_order: RotationOrder, times: npt.NDArray[T]
    ) -> Attitude:
        """Create an Attitude from euler angles.

        Time axis can be specified as relative or absolute (actual dates).

        Euler angles must be expressed in radians as an array with shape (N, 3), with N being the
        same length as the times array. The first column of the euler angles is the yaw, the second is
        the pitch and the third is the roll, but the order of application of the rotations is defined
        by the ``rotation_order`` parameter.

        !!! warning "Reference frame consistency"

            The reference frame in which euler angles are provided is kept as the reference frame of the interpolator,
            so the interpolated reference frames will be in the same reference system of the input euler angles.

            For example, if the euler angles are defined as rotations from sensor local axis to the
            antenna reference frame, the interpolated reference frames will be defined in the sensor local frame.

            In such a case, if you need the antenna reference frames in a global reference system (i.e. ECEF),
            an additional change of coordinates from the sensor local frame to a global reference system (i.e. ECEF)
            will be needed to get the attitude reference frames in the global reference system. In this case,
            you can use the [`compute_antenna_attitude_from_euler_angles`]
            [perseo_core.geometry.pointing.attitude.compute_antenna_attitude_from_euler_angles]
            function to directly compute the overall attitude of the system in the same reference system
            of the sensor local axis.

        Parameters
        ----------
        euler_angles_rad : npt.NDArray[np.floating]
            euler angles in radians, with shape (N, 3). The first column of the euler angles is the yaw, the second is
            the pitch and the third is the roll, but the order of application of the rotations is defined by the
            ``rotation_order`` parameter. The interpolated reference frames will be in the same reference system of the
            input euler angles.
        rotation_order : RotationOrder
            order of application of the rotations defined by the euler angles.
            values are: "YPR", "YRP", "PYR", "PRY", "RYP", "RPY"
        times : npt.NDArray[T]
            relative or absolute (actual dates) time axis, monotonic increasing, with shape (N,)

        Returns
        -------
        Attitude
            interpolator object
        """
        return cls(
            reference_frames=euler_angles_to_rotation(order=rotation_order, ypr_rad=euler_angles_rad).as_matrix(),
            times=times,
        )


def compute_antenna_attitude_from_euler_angles(
    ypr_rad: npt.NDArray[np.floating],
    rotation_order: RotationOrder,
    times: npt.NDArray[T],
    sensor_local_axis: npt.NDArray[np.floating],
) -> Attitude:
    """Compute the attitude of an antenna from euler angles and sensor local axis.

    Euler angles must be expressed in radians as an array with shape (N, 3), with N being the
    same length as the times array. The first column of the euler angles is the yaw, the second is
    the pitch and the third is the roll, but the order of application of the rotations is defined
    by the ``rotation_order`` parameter.

    The sensor local axis must be expressed as an array with shape (3, 3) or (N, 3, 3) representing
    the reference frame of the sensor.

    !!! warning "Reference frame consistency"

        The reference frame in which the sensor local axis is provided is kept as the reference frame
        of the interpolator, so the interpolated reference frames will be in the same reference system
        of the input sensor local axis.

        The euler angles are assumed to be defined as rotations from the sensor local axis to the
        antenna reference frame.

        For example, if the sensor local axis is defined in ECEF, the antenna attitude will be in ECEF as well.

    Parameters
    ----------
    ypr_rad : npt.NDArray[np.floating]
        euler angles in radians, with shape (N, 3). The first column of the euler angles is the yaw, the second is
        the pitch and the third is the roll, but the order of application of the rotations is defined by the
        ``rotation_order`` parameter.
    rotation_order : RotationOrder
        order of application of the rotations defined by the euler angles.
        values are: "YPR", "YRP", "PYR", "PRY", "RYP", "RPY"
    times : npt.NDArray[T]
        relative or absolute (actual dates) time axis, monotonic increasing, with shape (N,)
    sensor_local_axis : npt.NDArray[np.floating]
        reference frame of the sensor, with shape (3, 3) or (N, 3, 3).
        The Attitude reference frames will be defined in the same reference system as the input sensor local axis.

    Returns
    -------
    Attitude
        interpolator object

    """
    antenna_reference_frame = compute_antenna_reference_frame_from_euler_angles(
        ypr_rad=ypr_rad, rotation_order=rotation_order, initial_reference_frame=sensor_local_axis
    )
    return Attitude(reference_frames=antenna_reference_frame, times=times)


def compute_sensor_attitude_from_state_vectors(
    position: npt.NDArray[np.floating],
    velocity: npt.NDArray[np.floating],
    times: npt.NDArray[T],
    reference_frame: ReferenceFrame,
) -> Attitude:
    """Compute the attitude of a sensor from its state vectors.

    The sensor local axis is computed from the state vectors and the specified reference frame, and then the attitude
    is created with the sensor local axis as reference frame.

    !!! warning "Reference frame consistency"

        The reference frame in which the position and the velocities are provided is kept as the reference frame
        of the attitude, so the interpolated reference frames will be in the same reference system
        as the input state vectors.

        For example, if the state vectors are defined in ECEF, the sensor attitude will be in ECEF as well.

    Parameters
    ----------
    position : npt.NDArray[np.floating]
        position state vectors of the sensor, with shape (N, 3)
    velocity : npt.NDArray[np.floating]
        velocity state vectors of the sensor, with shape (N, 3)
    times : npt.NDArray[T]
        relative or absolute (actual dates) time axis, monotonic increasing, with shape (N,)
    reference_frame : ReferenceFrame
        kind of reference frame to compute. The options are: "GEOCENTRIC", "GEODETIC" and "ZERODOPPLER".

    Returns
    -------
    Attitude
        interpolator object
    """
    sensor_local_axis = compute_sensor_local_axis(
        sensor_positions=position, sensor_velocities=velocity, reference_frame=reference_frame
    )
    return Attitude(reference_frames=sensor_local_axis, times=times)


__all__ = [
    "Attitude",
    "compute_antenna_attitude_from_euler_angles",
    "compute_sensor_attitude_from_state_vectors",
]
