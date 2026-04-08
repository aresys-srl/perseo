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
