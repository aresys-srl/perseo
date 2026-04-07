# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Attitude Interpolator
---------------------
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation, Slerp

from perseo_core.geometry.utilities import RotationOrderLike
from perseo_core.geometry.utilities.rotations import compute_slerp_derivative, euler_angles_to_rotation


class Attitude:
    """Attitude Interpolator based on a Spherical Linear Interpolation of Rotations (SLERP)"""

    def __init__(
        self,
        rotations: Rotation,
        times: np.ndarray,
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
        self._time_origin = times.squeeze()[0]
        self._last_time = times.squeeze()[-1]
        self._time_relative = np.array(times.squeeze() - self._time_origin, dtype=float)
        self._domain = (self._time_origin, self._last_time)
        self._slerp = self._create_slerp()

    @property
    def rotations(self) -> Rotation:
        """Accessing attitude rotations used for interpolation"""
        return self._rotations

    @property
    def times(self) -> np.ndarray:
        """Accessing attitude times vector"""
        return self._times

    @property
    def domain(self) -> np.ndarray:
        """Attitude time domain"""
        return self._domain

    def _check_time_validity(self, times: ArrayLike) -> None:
        """Check input times validity with respect to the construction time validity boundaries.

        Parameters
        ----------
        times : ArrayLike
            input times at which interpolate the attitude

        Raises
        ------
        RuntimeError
            if one or more of the input times is not inside the time boundaries of attitude definition
        """
        if np.any(times < self._time_origin) or np.any(times > self._last_time):
            raise RuntimeError("One (or more) of the input times is outside of attitude time boundaries")

    def _create_slerp(self) -> Slerp:
        """Generating the SLERP interpolator from given inputs.

        Returns
        -------
        Slerp
            Spherical Linear Interpolation of Rotations scipy interpolator object
        """
        return Slerp(
            times=self._time_relative,
            rotations=self._rotations,
        )

    def evaluate(
        self,
        times: ArrayLike,
    ) -> Rotation:
        """Evaluate interpolated rotations at given input times using SLERP interpolator.

        Time values must be specified with a type that is the same as the construction "times" array used to build the
        interpolator.

        Parameters
        ----------
        times : ArrayLike
            time coordinates compatible with the time type used for building the interpolator

        Returns
        -------
        Rotation
            interpolated Scipy Rotation objects at each input time
        """
        self._check_time_validity(times)
        relative_times = times - self._time_origin
        return self._slerp(relative_times)

    def evaluate_first_derivatives(self, times: ArrayLike) -> Rotation:
        """Evaluate interpolated rotations first derivative at given times. This computes the exact derivative
        of the SLERP at the query times with piecewise constant angular velocity and discontinuous angular acceleration.

        Time values must be specified with a type that is the same as the construction "times" array used to build the
        interpolator.

        Parameters
        ----------
        times : ArrayLike
            time coordinates compatible with the time type used for building the interpolator

        Returns
        -------
        Rotation
            interpolated SLERP first derivative at each input time expressed as a Scipy Rotation object
        """
        self._check_time_validity(times)
        relative_times = times - self._time_origin
        return compute_slerp_derivative(
            rotations=self._rotations, times=self._time_relative, query_times=relative_times
        )

    @classmethod
    def from_quaternions(cls, quaternions: np.ndarray, times: np.ndarray) -> Attitude:
        """Create an Attitude SLERP interpolator from quaternions. Time axis can be specified as relative or absolute
        (actual dates), while quaternions must be expressed as an array with shape (N, 4), with N being the same length
        as the times array.

        .. note::
            The provided quaternions are expressed in a reference system chosen by the user, and the interpolator will
            keep the same reference system for the interpolated rotations.

        Parameters
        ----------
        quaternions : np.ndarray
            quaternions in a Global Reference System (i.e. ECEF), scalar-last, with shape (N, 4)
        times : np.ndarray
            relative or absolute (actual dates) time axis, monotonic increasing, with shape (N,)

        Returns
        -------
        Attitude
            interpolator object
        """
        return cls(rotations=Rotation.from_quat(quaternions), times=times)

    @classmethod
    def from_euler_angles(
        cls, euler_angles_rad: np.ndarray, rotation_order: RotationOrderLike, times: np.ndarray
    ) -> Attitude:
        """Create an Attitude SLERP interpolator from euler angles. Time axis can be specified as relative or absolute
        (actual dates), while euler angles must be expressed in radians as an array with shape (N, 3), with N being the
        same length as the times array and columns order matching the specified ``rotation_order``.

        .. note::
            The provided euler angles are expressed in a reference system chosen by the user, and the interpolator
            will keep the same reference system for the interpolated rotations.

        Parameters
        ----------
        euler_angles_rad : np.ndarray
            euler angles in radians, with the same column order of the specified ``rotation_order``, with shape (N, 3)
        rotation_order : RotationOrderLike
            rotation order of application of Euler angles
        times : np.ndarray
            relative or absolute (actual dates) time axis, monotonic increasing, with shape (N,)

        Returns
        -------
        Attitude
            interpolator object
        """
        return cls(
            rotations=euler_angles_to_rotation(order=rotation_order, euler_angles_rad=euler_angles_rad), times=times
        )
