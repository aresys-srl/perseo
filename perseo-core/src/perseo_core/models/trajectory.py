# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Trajectory interface
--------------------
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

import numpy as np
import numpy.typing as npt

T = TypeVar("T", bound=np.generic)


@runtime_checkable
class Trajectory(Protocol[T]):
    """Trajectory interface"""

    @property
    def domain(self) -> tuple[T, T]:
        """Trajectory time domain as a tuple of [start, end]"""
        ...

    def position(self, time: T | npt.NDArray[T]) -> npt.NDArray[np.floating]:
        """Retrieve position at given time.

        Parameters
        ----------
        time : T | npt.NDArray[T]
            evaluation time: scalar or array with shape (N,)

        Returns
        -------
        npt.NDArray[np.floating]
            position with shape (3,) or (N, 3)
        """
        ...

    def velocity(self, time: T | npt.NDArray[T]) -> npt.NDArray[np.floating]:
        """Retrieve velocity at given time.

        Parameters
        ----------
        time : T | npt.NDArray[T]
            evaluation time: scalar or array with shape (N,)

        Returns
        -------
        npt.NDArray[np.floating]
            velocity with shape (3,) or (N, 3)
        """
        ...

    def acceleration(self, time: T | npt.NDArray[T]) -> npt.NDArray[np.floating]:
        """Retrieve acceleration at given time.

        Parameters
        ----------
        time : T | npt.NDArray[T]
            evaluation time: scalar or array with shape (N,)

        Returns
        -------
        npt.NDArray[np.floating]
            acceleration with shape (3,) or (N, 3)
        """
        ...
