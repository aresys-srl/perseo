# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Trajectory interface
--------------------
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike


@runtime_checkable
class Trajectory(Protocol):
    """Trajectory interface"""

    @property
    def domain(self) -> tuple[Any, Any]:
        """Trajectory time domain as a tuple of [start, end]"""

    def position(self, time: ArrayLike) -> np.ndarray:
        """Retrieve position at given time.

        Parameters
        ----------
        time : ArrayLike
            evaluation time with equivalent shape (1,) or (N, 1)

        Returns
        -------
        np.ndarray
            position with shape (3,) or (N, 3)
        """

    def velocity(self, time: ArrayLike) -> np.ndarray:
        """Retrieve velocity at given time.

        Parameters
        ----------
        time : ArrayLike
            evaluation time with equivalent shape (1,) or (N, 1)

        Returns
        -------
        np.ndarray
            velocity with shape (3,) or (N, 3)
        """

    def acceleration(self, time: ArrayLike) -> np.ndarray:
        """Retrieve acceleration at given time.

        Parameters
        ----------
        time : ArrayLike
            evaluation time with equivalent shape (1,) or (N, 1)

        Returns
        -------
        np.ndarray
            velocity with shape (3,) or (N, 3)
        """
