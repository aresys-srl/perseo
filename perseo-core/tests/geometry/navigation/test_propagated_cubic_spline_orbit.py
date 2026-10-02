# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for geometry/navigation/propagated_cubic_spline_trajectory.py PropagatedCubicSplineOrbit object"""

import math
from collections.abc import Callable

import numpy as np
import pytest

from perseo_core.geometry.navigation import PropagatedCubicSplineOrbit, Trajectory
from perseo_core.geometry.navigation.propagated_cubic_spline_orbit import _to_microseconds
from perseo_core.timing.precise_datetime import PreciseDateTime


class TestToMicrosecond:
    def test_whole_second_is_unchanged(self) -> None:
        t = PreciseDateTime.from_numeric_datetime(
            year=2024, month=1, day=1, hours=0, minutes=0, seconds=0, picoseconds=0
        )
        assert _to_microseconds(t, rounder=math.floor) == t
        assert _to_microseconds(t, rounder=math.ceil) == t
        assert _to_microseconds(t, rounder=round) == t

    def test_floor(self) -> None:
        t = PreciseDateTime.from_numeric_datetime(
            year=2024, month=1, day=1, hours=1, minutes=2, seconds=3, picoseconds=1_234_567
        )
        t_expected = PreciseDateTime.from_numeric_datetime(
            year=2024, month=1, day=1, hours=1, minutes=2, seconds=3, picoseconds=1_000_000
        )
        floored = _to_microseconds(t, rounder=math.floor)
        assert floored == t_expected

    def test_floor_without_rollover(self) -> None:
        t = PreciseDateTime.from_numeric_datetime(
            year=2024, month=12, day=31, hours=23, minutes=59, seconds=59, picoseconds=999_999_999_999
        )
        t_expected = PreciseDateTime.from_numeric_datetime(
            year=2024, month=12, day=31, hours=23, minutes=59, seconds=59, picoseconds=999_999_000_000
        )
        floored = _to_microseconds(t, rounder=math.floor)
        assert floored == t_expected

    def test_ceiling(self) -> None:
        t = PreciseDateTime.from_numeric_datetime(
            year=2024, month=1, day=1, hours=1, minutes=2, seconds=3, picoseconds=1_234_567
        )
        t_expected = PreciseDateTime.from_numeric_datetime(
            year=2024, month=1, day=1, hours=1, minutes=2, seconds=3, picoseconds=2_000_000
        )
        ceiled = _to_microseconds(t, rounder=math.ceil)
        assert ceiled == t_expected

    def test_round_down(self) -> None:
        t = PreciseDateTime.from_numeric_datetime(
            year=2024, month=1, day=1, hours=1, minutes=2, seconds=3, picoseconds=1_499_999
        )
        t_expected = PreciseDateTime.from_numeric_datetime(
            year=2024, month=1, day=1, hours=1, minutes=2, seconds=3, picoseconds=1_000_000
        )
        rounded = _to_microseconds(t, rounder=round)
        assert rounded == t_expected

    def test_round_up(self) -> None:
        t = PreciseDateTime.from_numeric_datetime(
            year=2024, month=12, day=31, hours=23, minutes=59, seconds=59, picoseconds=999_999_999_999
        )
        t_expected = PreciseDateTime.from_numeric_datetime(
            year=2025, month=1, day=1, hours=0, minutes=0, seconds=0, picoseconds=0
        )
        rounded = _to_microseconds(t, rounder=round)
        assert rounded == t_expected

    def test_round_at_half(self) -> None:
        t = PreciseDateTime.from_numeric_datetime(
            year=2024, month=12, day=31, hours=23, minutes=59, seconds=59, picoseconds=1_500_000
        )
        t_expected = PreciseDateTime.from_numeric_datetime(
            year=2024, month=12, day=31, hours=23, minutes=59, seconds=59, picoseconds=2_000_000
        )
        rounded = _to_microseconds(t, rounder=round)
        assert rounded == t_expected

    def test_round_at_half_even_microsecond(self) -> None:
        # builtin round uses banker's rounding: 2.5 -> 2
        t = PreciseDateTime.from_numeric_datetime(
            year=2024, month=1, day=1, hours=0, minutes=0, seconds=0, picoseconds=2_500_000
        )
        t_expected = PreciseDateTime.from_numeric_datetime(
            year=2024, month=1, day=1, hours=0, minutes=0, seconds=0, picoseconds=2_000_000
        )
        assert _to_microseconds(t, rounder=round) == t_expected

    def test_ceiling_rollover(self) -> None:
        t = PreciseDateTime.from_numeric_datetime(
            year=2024, month=12, day=31, hours=23, minutes=59, seconds=59, picoseconds=999_999_999_999
        )
        t_expected = PreciseDateTime.from_numeric_datetime(
            year=2025, month=1, day=1, hours=0, minutes=0, seconds=0, picoseconds=0
        )
        assert _to_microseconds(t, rounder=math.ceil) == t_expected

    @pytest.mark.parametrize("rounder", [math.floor, math.ceil, round])
    @pytest.mark.parametrize(
        "ps",
        [
            0,
            1,
            499_999,
            500_000,
            1_234_567,
            999_999,
            1_000_000,
            999_999_000_000,
            999_999_500_000,
            999_999_999_999,
        ],
    )
    def test_idempotent(self, rounder: Callable, ps: int) -> None:
        t = PreciseDateTime.from_numeric_datetime(
            year=2024, month=12, day=31, hours=23, minutes=59, seconds=59, picoseconds=ps
        )
        once = _to_microseconds(t, rounder=rounder)
        twice = _to_microseconds(once, rounder=rounder)

        assert twice == once
        assert once.picosecond_of_second % 1_000_000 == 0


class TestOrbit:
    @pytest.fixture(autouse=True)
    def setup_orbit_data(self, orbit_test_data: dict) -> None:
        """Load test data from fixtures."""
        self.time_axis = orbit_test_data["time_axis"]
        self.positions = orbit_test_data["positions"]
        self.velocities = orbit_test_data["velocities"]
        self.tolerance = orbit_test_data["tolerance"]

        self.n1 = 7
        self.n2 = 23
        self.propagated_orbit = PropagatedCubicSplineOrbit(
            times=self.time_axis[self.n1 : self.n2],
            positions=self.positions[self.n1 : self.n2],
            velocities=self.velocities[self.n1 : self.n2],
        )

    def test_trajectory_subclass(self) -> None:
        """Test that PropagatedCubicSplineOrbit is subclass of Trajectory protocol."""
        assert issubclass(PropagatedCubicSplineOrbit, Trajectory)

    def test_trajectory_creation(self) -> None:
        """Test PropagatedCubicSplineOrbit constructor creates valid instance."""
        assert isinstance(self.propagated_orbit, PropagatedCubicSplineOrbit)

    def test_trajectory_properties(self) -> None:
        """Test that PropagatedCubicSplineOrbit properties return correct times, positions, velocities."""
        np.testing.assert_array_equal(self.propagated_orbit.positions, self.positions[self.n1 : self.n2])
        np.testing.assert_array_equal(self.propagated_orbit.velocities, self.velocities[self.n1 : self.n2])
        delta_times = self.propagated_orbit.times - self.time_axis[self.n1 : self.n2]
        np.testing.assert_array_equal(delta_times.astype(float), np.zeros_like(delta_times, dtype=float))

    def test_propagated_orbit_methods(self) -> None:
        """Test PropagatedCubicSplineOrbit interpolation and propagation methods
        for position, velocity, acceleration."""
        np.testing.assert_allclose(
            self.propagated_orbit.position(self.time_axis),
            self.positions,
            atol=self.tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            self.propagated_orbit.velocity(self.time_axis),
            self.velocities,
            atol=self.tolerance,
            rtol=0,
        )

    def test_incremental_extension_matches_single_extension(self) -> None:
        step_s = 100.0
        n_steps = 3
        t_start = self.time_axis[0]
        t_end = self.time_axis[-1]
        final_time = t_end + n_steps * step_s

        single = PropagatedCubicSplineOrbit(times=self.time_axis, positions=self.positions, velocities=self.velocities)
        incremental = PropagatedCubicSplineOrbit(
            times=self.time_axis, positions=self.positions, velocities=self.velocities
        )

        query_times = np.linspace(t_start, final_time, 200)
        single.position(final_time)
        single_positions = single.position(query_times)

        for t in query_times:
            incremental.position(t)
        incremental_positions = incremental.evaluate(query_times)
        single_positions = single.evaluate(query_times)

        np.testing.assert_allclose(incremental_positions, single_positions, atol=1e-5, rtol=0)

    def test_scalar_input(self) -> None:
        np.testing.assert_array_equal(
            self.propagated_orbit.position(self.time_axis[0]),
            self.propagated_orbit.position(self.time_axis[0 : self.n1])[0],
        )

    def test_warning_past_three_hour(self) -> None:
        with pytest.warns(RuntimeWarning, match="outside the original trajectory"):
            self.propagated_orbit.position(self.time_axis[self.n2] + 60 * 60 * 3 + 1)

    def test_changing_right_bound(self) -> None:
        orbit = PropagatedCubicSplineOrbit(times=self.time_axis, positions=self.positions, velocities=self.velocities)
        initial_domain = orbit.domain
        orbit.position(time=initial_domain[1] + 600)
        final_domain = orbit.domain
        np.testing.assert_equal(initial_domain[0], final_domain[0])
        assert final_domain[1] > initial_domain[1]

    def test_changing_left_bound(self) -> None:
        orbit = PropagatedCubicSplineOrbit(times=self.time_axis, positions=self.positions, velocities=self.velocities)
        initial_domain = orbit.domain
        orbit.position(time=initial_domain[1] - 600)
        final_domain = orbit.domain
        np.testing.assert_equal(initial_domain[1], final_domain[1])
        assert final_domain[0] < initial_domain[0]

    def test_time_before_start(self) -> None:
        pos = self.propagated_orbit.position(self.time_axis[0] - 1.0)
        assert isinstance(pos, np.ndarray)
        assert pos.shape == (3,)

    def test_array_with_one_time_before_start(self) -> None:
        times = np.array([self.time_axis[0] - 0.5, self.time_axis[1]])
        assert isinstance(self.propagated_orbit.velocity(times), np.ndarray)

    @pytest.mark.parametrize("bad_shape", [(5,), (5, 2), (5, 3, 1)])
    def test_positions_shape(self, bad_shape: tuple) -> None:
        with pytest.raises(ValueError, match="Positions"):
            PropagatedCubicSplineOrbit(
                self.propagated_orbit.times, np.zeros(bad_shape), self.propagated_orbit.velocities
            )

    @pytest.mark.parametrize("bad_shape", [(5,), (5, 2), (5, 3, 1)])
    def test_velocities_shape(self, bad_shape: tuple) -> None:
        with pytest.raises(ValueError, match="Velocities"):
            PropagatedCubicSplineOrbit(
                self.propagated_orbit.times, self.propagated_orbit.positions, np.zeros(bad_shape)
            )

    def test_length_mismatch(self) -> None:
        times = self.time_axis
        pos = self.positions
        vel = self.velocities
        with pytest.raises(ValueError, match="same number of samples"):
            PropagatedCubicSplineOrbit(times, pos[:-1], vel)
        with pytest.raises(ValueError, match="same number of samples"):
            PropagatedCubicSplineOrbit(times, pos, vel[:-1])
