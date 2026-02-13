# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Models - Custom Protocols
-------------------------
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

from perseo_core.models.types import ExtendedDatetimeType


class ExtrapolationNotAllowed(ValueError):
    """Extrapolation outside of boundaries is not allowed"""


@runtime_checkable
class RotationsDifferentiableSLERP(Protocol):
    """SLERP rotations interpolator wrapper with evaluation method for rotations and first derivatives"""

    @property
    def domain(self) -> tuple[Any, Any]:
        """Interpolator domain boundaries, min and max values"""

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


# TODO: this could be generalized to a R -> R^n curve
@runtime_checkable
class TwiceDifferentiable3DCurve(Protocol):
    """Custom 3D curve container with evaluation method for curve and first two derivatives. f: R -> R^3"""

    @property
    def domain(self) -> tuple[Any, Any]:
        """Curve domain boundaries, min and max values"""

    def evaluate(self, coordinates: ArrayLike) -> np.ndarray:
        """Evaluate curve value at given input coordinates.

        Parameters
        ----------
        coordinates : ArrayLike
            coordinates where to evaluate the curve, with equivalent shape (3,) or (N, 3)

        Returns
        -------
        np.ndarray
            values of the curve at given input values, with shape (3,) or (N, 3)
        """

    def evaluate_first_derivatives(self, coordinates: ArrayLike) -> np.ndarray:
        """Evaluate curve first derivatives values at given input coordinates.

        Parameters
        ----------
        coordinates : ArrayLike
            coordinates where to evaluate the first derivatives, with equivalent shape (3,) or (N, 3)

        Returns
        -------
        np.ndarray
            values of the curve derivatives at given input values, with shape (3,) or (N, 3)
        """

    def evaluate_second_derivatives(self, coordinates: ArrayLike) -> np.ndarray:
        """Evaluate curve second derivatives values at given input coordinates.

        Parameters
        ----------
        coordinates : ArrayLike
            coordinates where to evaluate the second derivatives, with equivalent shape (3,) or (N, 3)

        Returns
        -------
        np.ndarray
            values of the curve second derivatives at given input values, with shape (3,) or (N, 3)
        """


@runtime_checkable
class RealTwiceDifferentiableFunction(Protocol):
    """Generic protocol for a f: R -> R function twice differentiable with derivative evaluation methods implemented"""

    def evaluate(self, coordinates: ArrayLike) -> ArrayLike:
        """Evaluate function value at given coordinates.

        Parameters
        ----------
        coordinates : ArrayLike
            input coordinates where to evaluate the function

        Returns
        -------
        ArrayLike
            value of function at each input coordinate
        """

    def evaluate_first_derivative(self, coordinates: ArrayLike) -> ArrayLike:
        """Evaluate function first derivative at given coordinates.

        Parameters
        ----------
        coordinates : ArrayLike
            input coordinates where to evaluate the function derivative

        Returns
        -------
        ArrayLike
            values of function first derivative at each input coordinate
        """

    def evaluate_second_derivative(self, coordinates: ArrayLike) -> ArrayLike:
        """Evaluate function second derivative at given coordinates.

        Parameters
        ----------
        coordinates : ArrayLike
            input coordinates where to evaluate the function derivative

        Returns
        -------
        ArrayLike
            values of function second derivative at each input coordinate
        """


# TODO: check this, from quality protocols
@runtime_checkable
class SARCoordinatesFunction(Protocol):
    """Protocol to define a function taking SAR coordinates (Azimuth, Range) as inputs and returns a float
    This can be any generic f: SAR Times -> R.
    """

    def evaluate(self, azimuth_time: ExtendedDatetimeType, range_time: float) -> float:
        """Evaluate the wrapped function at given azimuth and range times.

        Parameters
        ----------
        azimuth_time : ExtendedDatetimeType
            azimuth time at which evaluate the function
        range_time : float
            range time at which evaluate the function

        Returns
        -------
        float
            output of the wrapped function
        """
