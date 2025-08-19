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


@runtime_checkable
class TwiceDifferentiable3DCurve(Protocol):
    """Custom 3D curve container with evaluation method for curve and first two derivatives.
    f: R -> R^3"""

    @property
    def domain(self) -> tuple[Any, Any]:
        """Curve domain boundaries, min and max values"""

    def evaluate(self, coordinates: ArrayLike) -> np.ndarray:
        """Evaluate curve value at given input coordinates.

        Parameters
        ----------
        coordinates : ArrayLike
            coordinates where to evaluate the curve

        Returns
        -------
        np.ndarray
            values of the curve at given input values (N, 3)
        """

    def evaluate_first_derivatives(self, coordinates: ArrayLike) -> np.ndarray:
        """Evaluate curve first derivatives values at given input coordinates.

        Parameters
        ----------
        coordinates : ArrayLike
            coordinates where to evaluate the derivatives

        Returns
        -------
        np.ndarray
            values of the curve derivatives at given input values (N, 3)
        """

    def evaluate_second_derivatives(self, coordinates: ArrayLike) -> np.ndarray:
        """Evaluate curve second derivatives values at given input coordinates.

        Parameters
        ----------
        coordinates : ArrayLike
            coordinates where to evaluate the derivatives

        Returns
        -------
        np.ndarray
            values of the polynomials second derivatives at given input times (N, 3)
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
