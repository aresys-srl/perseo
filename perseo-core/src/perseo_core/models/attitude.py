# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Attitude Interpolator
---------------------
"""

from __future__ import annotations

from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation, Slerp

from perseo_core.geometry.utilities.rotations import (
    RotationOrder,
    compute_slerp_derivative,
    euler_angles_to_rotation,
)

T = TypeVar("T", bound=np.generic)


class Attitude(Generic[T]):
    """Attitude Interpolator based on a Spherical Linear Interpolation of Rotations (SLERP)"""

    def __init__(
        self,
        rotations: Rotation,
        times: npt.NDArray[T],
    ) -> None:
        """Create an Attitude SLERP interpolator from times and associated rotations. Time axis can be specified as
        relative or absolute (actual dates), while rotations must be expressed with a numerosity equals to the same
        length as the times array.

        .. note::
            The provided rotations are expressed in a reference system chosen by the user, and the interpolator will
            keep the same reference system for the interpolated rotations.

        Parameters
        ----------
        rotations : Rotation
            Scipy Rotation object
        times : np.ndarray
            relative or absolute (actual dates) time axis, monotonic increasing, with shape (N,)
        """
        self._rotations = rotations
        self._times = times

        self._domain = (times[0], times[-1])
        self._slerp = Slerp(
            times=np.array(times - times[0], dtype=np.float64),
            rotations=self._rotations,
        )

    @property
    def rotations(self) -> Rotation:
        """Accessing attitude rotations used for interpolation"""
        return self._rotations

    @property
    def times(self) -> npt.NDArray[T]:
        """Accessing attitude times vector"""
        return self._times

    @property
    def domain(self) -> tuple[T, T]:
        """Attitude time domain"""
        return self._domain

    def _check_time_validity(self, time: T | npt.NDArray[T]) -> None:
        """Check input times validity with respect to attitude time domain."""
        if np.any(time < self.domain[0]) or np.any(time > self.domain[1]):
            raise RuntimeError("One (or more) of the input times is outside of attitude time domain.")

    def evaluate(self, time: T | npt.NDArray[T]) -> Rotation:
        """Retrieve antenna reference frame rotations at given times.

        Parameters
        ----------
        time: T | npt.NDArray[T]
            time of the same type of the initialization times axis

        Returns
        -------
        Rotation
            interpolated Scipy Rotation objects at each input time
        """
        self._check_time_validity(time)
        relative_times = time - self.domain[0]
        return self._slerp(relative_times)

    def evaluate_first_derivatives(self, time: T | npt.NDArray[T]) -> Rotation:
        """Retrieve antenna reference frame rotations derivative at given times.

        This computes the exact derivative of the SLERP at the query times with piecewise constant angular velocity
         and discontinuous angular acceleration.

        Parameters
        ----------
        time: T | npt.NDArray[T]
            time of the same type of the initialization times axis

        Returns
        -------
        Rotation
            interpolated SLERP first derivative at each input time expressed as a Scipy Rotation object
        """
        self._check_time_validity(time)
        relative_times = time - self.domain[0]
        return compute_slerp_derivative(
            rotations=self._rotations, times=self.times - self.domain[0], query_times=relative_times
        )

    @classmethod
    def from_quaternions(cls, quaternions: npt.NDArray[np.floating], times: npt.NDArray[T]) -> Attitude:
        """Create an Attitude SLERP interpolator from quaternions.

        Time axis can be specified as relative or absolute
        (actual dates), while quaternions must be expressed as an array with shape (N, 4), with N being the same length
        as the times array.

        .. note::
            The provided quaternions are expressed in a reference system chosen by the user, and the interpolator will
            keep the same reference system for the interpolated rotations.

        Parameters
        ----------
        quaternions : npt.NDArray[np.floating]
            quaternions in a Global Reference System (i.e. ECEF), scalar-last, with shape (N, 4)
        times : npt.NDArray[T]
            relative or absolute (actual dates) time axis, monotonic increasing, with shape (N,)

        Returns
        -------
        Attitude
            interpolator object
        """
        return cls(rotations=Rotation.from_quat(quaternions), times=times)

    @classmethod
    def from_euler_angles(
        cls, euler_angles_rad: npt.NDArray[np.floating], rotation_order: RotationOrder, times: npt.NDArray[T]
    ) -> Attitude:
        """Create an Attitude SLERP interpolator from euler angles.

        Time axis can be specified as relative or absolute
        (actual dates), while euler angles must be expressed in radians as an array with shape (N, 3), with N being the
        same length as the times array and columns order matching the specified ``rotation_order``.

        .. note::
            The provided euler angles are expressed in a reference system chosen by the user, and the interpolator
            will keep the same reference system for the interpolated rotations.

        Parameters
        ----------
        euler_angles_rad : npt.NDArray[np.floating]
            euler angles in radians, with the same column order of the specified ``rotation_order``, with shape (N, 3)
        rotation_order : "YPR", "YRP", "PRY", "PYR", "RYP", "RPY"
            rotation order of application of Euler angles
        times : npt.NDArray[T]
            relative or absolute (actual dates) time axis, monotonic increasing, with shape (N,)

        Returns
        -------
        Attitude
            interpolator object
        """
        return cls(rotations=euler_angles_to_rotation(order=rotation_order, ypr_rad=euler_angles_rad), times=times)
