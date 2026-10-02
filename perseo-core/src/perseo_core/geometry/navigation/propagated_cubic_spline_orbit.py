# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Propagated Cubic Spline Orbit module.

This module provides the `PropagatedCubicSplineOrbit` class, which enables orbit propagation
outside of the domain defined by state vectors
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
import satkit as sk

from perseo_core.geometry.coordinates.conversions import ecef2eci, eci2ecef
from perseo_core.geometry.navigation.cubic_spline_trajectory import CubicSplineTrajectory
from perseo_core.geometry.navigation.trajectory import Trajectory
from perseo_core.timing.precise_datetime import PreciseDateTime

if TYPE_CHECKING:
    from collections.abc import Callable

_GRAVITY_DEGREE: int = 70
_GRAVITY_ORDER: int = 70
_ABS_ERROR: float = 1e-10
_REL_ERROR: float = 1e-10


def _to_satkit_time(times: npt.NDArray) -> list[sk.time]:
    """Convert PreciseDateTime to satkit.time (satkit.time has microsecond precision)."""
    return [
        sk.time(
            t.year,
            t.month,
            t.day_of_the_month,
            t.hour_of_day,
            t.minute_of_hour,
            t.second_of_minute + t.picosecond_of_second * 1e-12,
        )
        for t in times
    ]


def _to_microseconds(t: PreciseDateTime, rounder: Callable) -> PreciseDateTime:
    """Round a PreciseDateTime to a whole microsecond using the given rounding function."""
    picoseconds = rounder(t.picosecond_of_second / 1_000_000) * 1_000_000
    delta = picoseconds - t.picosecond_of_second
    return t + delta * 1e-12


class _PropagatedBranch:
    """Numerically propagated extension of an orbit, either forward (+1) or backward (-1) in time."""

    def __init__(
        self,
        anchor: PreciseDateTime,
        ecef_pos: npt.NDArray[np.floating],
        ecef_vel: npt.NDArray[np.floating],
        direction: Literal["forward", "backward"],
    ) -> None:
        self.direction_sign = 1 if direction == "forward" else -1
        self.times: npt.NDArray = np.array([anchor], dtype=object)
        self.pos = np.atleast_2d(ecef_pos)
        self.vel = np.atleast_2d(ecef_vel)
        eci_pos, eci_vel = ecef2eci(ecef_pos, ecef_vel, anchor)
        self.eci_state = np.hstack([eci_pos, eci_vel])
        self.spline: CubicSplineTrajectory | None = None

    @property
    def frontier(self) -> PreciseDateTime:
        """Farthest time reached so far in the propagation direction."""
        return self.times[-1] if self.direction_sign > 0 else self.times[0]

    def extend_by(self, duration_s: int) -> None:
        """Propagate `duration_s` seconds past the current frontier, in the branch direction."""
        t0 = self.frontier
        new_times = self.direction_sign * np.arange(0, duration_s + 1, dtype=float) + t0  # type: ignore[operator]  # pyrefly: ignore[unsupported-operation]
        sk_times = _to_satkit_time(new_times)

        result = sk.propagate(
            self.eci_state,
            sk_times[0],
            end=sk_times[-1],
            propsettings=sk.propsettings(
                gravity_model=sk.gravmodel.egm2008,
                gravity_degree=_GRAVITY_DEGREE,
                gravity_order=_GRAVITY_ORDER,
                abs_error=_ABS_ERROR,
                rel_error=_REL_ERROR,
            ),
        )
        evaluation = np.atleast_2d(result.interp(sk_times))
        new_pos, new_vel = eci2ecef(evaluation[:, 0:3], evaluation[:, 3:6], new_times)

        if self.direction_sign > 0:
            self.times = np.concatenate([self.times, new_times[1:]])
            self.pos = np.vstack([self.pos, new_pos[1:]])
            self.vel = np.vstack([self.vel, new_vel[1:]])
        else:
            self.times = np.concatenate([new_times[1:][::-1], self.times])
            self.pos = np.vstack([new_pos[1:][::-1], self.pos])
            self.vel = np.vstack([new_vel[1:][::-1], self.vel])

        self.eci_state = evaluation[-1]
        self.spline = CubicSplineTrajectory(self.times, self.pos, self.vel)


class PropagatedCubicSplineOrbit(Trajectory[PreciseDateTime]):  # type: ignore[type-var]   # pyrefly: ignore[bad-specialization]
    """Extended `CubicSplineTrajectory` that allows orbit propagation."""

    _INITIAL_EXTENSION_S: int = 120
    _EXTENSION_MARGIN_S: int = 60
    _WARNING_THRESHOLD_S: int = 3600 * 3

    def __init__(
        self,
        times: npt.NDArray,
        positions: npt.NDArray[np.floating],
        velocities: npt.NDArray[np.floating],
    ) -> None:
        """Create a PropagatedCubicSplineOrbit from state vectors: times, positions and velocities.

        Times must be of type PreciseDateTime.

        Positions and velocities must be specified as (N, 3) arrays of floats.

        PropagatedCubicSplineOrbit can extend trajectory for low orbit satellites outside the boundaries of time axis
        through high precision numerical integration.

        Parameters
        ----------
        times : npt.NDArray
            time axis as numpy array of shape (N,)
        positions : npt.NDArray[np.floating]
            positions as numpy array of shape (N, 3), with coordinates being x, y, z
        velocities : npt.NDArray[np.floating]
            velocities as numpy array of shape (N, 3), with coordinates being x, y, z

        """
        if not all(isinstance(t, PreciseDateTime) for t in times):
            msg = "Times must be an array of PreciseDateTime"
            raise TypeError(msg)

        if times.ndim != 1:
            msg = "Times must be a 1D array"
            raise ValueError(msg)

        if positions.ndim != 2 or positions.shape[1] != 3:
            msg = "Positions must be a 2D array with shape (N, 3)"
            raise ValueError(msg)

        if velocities.ndim != 2 or velocities.shape[1] != 3:
            msg = "Velocities must be a 2D array with shape (N, 3)"
            raise ValueError(msg)

        if not len(times) == positions.shape[0] == velocities.shape[0]:
            msg = "Times, positions and velocities must have the same number of samples"
            raise ValueError(msg)

        self._original = CubicSplineTrajectory(times=times, positions=positions, velocities=velocities)

        t_end = _to_microseconds(times[-1], rounder=math.floor)
        t_start = _to_microseconds(times[0], rounder=math.ceil)

        self._forward = _PropagatedBranch(
            t_end, self._original.position(t_end), self._original.velocity(t_end), direction="forward"
        )
        self._backward = _PropagatedBranch(
            t_start, self._original.position(t_start), self._original.velocity(t_start), direction="backward"
        )
        self._forward.extend_by(self._INITIAL_EXTENSION_S)
        self._backward.extend_by(self._INITIAL_EXTENSION_S)

    def _warn_if_far(self, elapsed_s: float) -> None:
        if elapsed_s > self._WARNING_THRESHOLD_S:
            warnings.warn(
                f"Requested time is {elapsed_s:0.2f} s outside the original trajectory: "
                f"propagation beyond {self._WARNING_THRESHOLD_S} s may be inaccurate",
                RuntimeWarning,
                stacklevel=5,
            )

    def _ensure_covers(self, t_min: PreciseDateTime, t_max: PreciseDateTime) -> None:
        start, end = self._original.domain
        self._warn_if_far(float(start - t_min))
        self._warn_if_far(float(t_max - end))

        if t_max > self._forward.frontier:
            missing_s = t_max - self._forward.frontier
            self._forward.extend_by(int(np.ceil(missing_s)) + self._EXTENSION_MARGIN_S)
        if t_min < self._backward.frontier:
            missing_s = self._backward.frontier - t_min
            self._backward.extend_by(int(np.ceil(missing_s)) + self._EXTENSION_MARGIN_S)

    @property
    def positions(self) -> np.ndarray:
        """Accessing trajectory positions vector."""
        return self._original.positions

    @property
    def velocities(self) -> np.ndarray:
        """Accessing trajectory velocities vector."""
        return self._original.velocities

    @property
    def times(self) -> np.ndarray:
        """Accessing trajectory times vector."""
        return self._original.times

    @property
    def domain(self) -> tuple[PreciseDateTime, PreciseDateTime]:
        """Accessing time domain (bounds grow as the trajectory is propagated)."""
        return (self._backward.times[0], self._forward.times[-1])

    def _evaluate(
        self, time: PreciseDateTime | npt.NDArray, method: Literal["position", "velocity", "acceleration"]
    ) -> np.ndarray:
        t: npt.NDArray = np.atleast_1d(time)  # type: ignore[type-var]  # pyrefly: ignore[no-matching-overload]
        self._ensure_covers(t.min(), t.max())

        start, end = self._original.domain
        before = t < start
        after = t > end
        inside = ~(before | after)

        out = np.empty((t.size, 3))
        if before.any():
            out[before] = getattr(self._backward.spline, method)(t[before])
        if inside.any():
            out[inside] = getattr(self._original, method)(t[inside])
        if after.any():
            out[after] = getattr(self._forward.spline, method)(t[after])

        return out[0] if np.ndim(time) == 0 else out  # type: ignore[arg-type]  # pyrefly: ignore[bad-argument-type]

    def position(self, time: PreciseDateTime | npt.NDArray) -> npt.NDArray[np.floating]:
        """Evaluate x, y, z position at given time.

        Parameters
        ----------
        time : PreciseDateTime | npt.NDArray
            time of type PreciseDateTime

        Returns
        -------
        np.ndarray
            position with shape (3,) or (N, 3) with coordinates being x, y, z

        """
        return self._evaluate(time, "position")

    def velocity(self, time: PreciseDateTime | npt.NDArray) -> npt.NDArray[np.floating]:
        """Evaluate vx, vy, vz velocity at given time.

        Parameters
        ----------
        time : PreciseDateTime | npt.NDArray
            time of type PreciseDateTime

        Returns
        -------
        np.ndarray
            velocity with shape (3,) or (N, 3) with coordinates being x, y, z

        """
        return self._evaluate(time, "velocity")

    def acceleration(self, time: PreciseDateTime | npt.NDArray) -> npt.NDArray[np.floating]:
        """Evaluate ax, ay, az acceleration at given time.

        Parameters
        ----------
        time : PreciseDateTime | npt.NDArray
            time of type PreciseDateTime

        Returns
        -------
        np.ndarray
            acceleration with shape (3,) or (N, 3) with coordinates being x, y, z

        """
        return self._evaluate(time, "acceleration")


__all__ = ["PropagatedCubicSplineOrbit"]
