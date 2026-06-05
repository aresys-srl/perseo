# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for geometry/navigation/cubic_spline_trajectory.py CubicSplineTrajectory object"""

import numpy as np
import pytest

from perseo_core.geometry.navigation import CubicSplineTrajectory, Trajectory


class TestTrajectory:
    """Test CubicSplineTrajectory creation, properties, interpolation, and boundary conditions."""

    @pytest.fixture(autouse=True)
    def setup_trajectory_data(self, trajectory_test_data: dict) -> None:
        """Load test data from fixtures."""
        self._state_vectors = trajectory_test_data["state_vectors"]
        self._trajectory = trajectory_test_data["trajectory"]
        self._tolerance = trajectory_test_data["tolerance"]
        self._times = trajectory_test_data["test_times"]
        self._expected_pos = trajectory_test_data["expected_positions"]
        self._expected_vel = trajectory_test_data["expected_velocities"]
        self._expected_acc = trajectory_test_data["expected_accelerations"]

    def test_trajectory_subclass(self) -> None:
        """Test that CubicSplineTrajectory is subclass of Trajectory protocol."""
        assert issubclass(CubicSplineTrajectory, Trajectory)

    def test_trajectory_creation(self) -> None:
        """Test CubicSplineTrajectory constructor creates valid instance."""
        trajectory = CubicSplineTrajectory(
            times=self._state_vectors.time_axis,
            positions=self._state_vectors.sensor_positions,
            velocities=self._state_vectors.sensor_velocities,
        )
        assert isinstance(trajectory, CubicSplineTrajectory)

    def test_trajectory_properties(self) -> None:
        """Test that CubicSplineTrajectory properties return correct times, positions, velocities."""
        np.testing.assert_array_equal(self._trajectory.positions, self._state_vectors.sensor_positions)
        np.testing.assert_array_equal(self._trajectory.velocities, self._state_vectors.sensor_velocities)
        delta_times = self._trajectory.times - self._state_vectors.time_axis
        np.testing.assert_array_equal(delta_times.astype(float), np.zeros_like(delta_times, dtype=float))

    def test_trajectory_methods(self) -> None:
        """Test CubicSplineTrajectory interpolation methods for position, velocity, acceleration."""
        np.testing.assert_allclose(
            self._trajectory.position(self._times),
            self._expected_pos,
            atol=self._tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            self._trajectory.velocity(self._times),
            self._expected_vel,
            atol=self._tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            self._trajectory.acceleration(self._times),
            self._expected_acc,
            atol=self._tolerance,
            rtol=0,
        )

    def test_trajectory_methods_extrapolation_error_1(self) -> None:
        """Test that position, velocity, acceleration raise error for times outside domain (individual values)."""
        with pytest.raises(RuntimeError):
            self._trajectory.position(self._times[0] - 2)
        with pytest.raises(RuntimeError):
            self._trajectory.velocity(self._times[-1] + 12)
        with pytest.raises(RuntimeError):
            self._trajectory.acceleration(self._times[0] - 2)

    def test_trajectory_methods_extrapolation_error_2(self) -> None:
        """Test that position, velocity, acceleration raise error for vectorized times outside domain."""
        with pytest.raises(RuntimeError):
            self._trajectory.position(self._times - 200)
        with pytest.raises(RuntimeError):
            self._trajectory.velocity(self._times + 200)
        with pytest.raises(RuntimeError):
            self._trajectory.acceleration(self._times - 200)

    def test_trajectory_methods_extrapolation_error_3(self) -> None:
        """Test error handling when one query time is outside domain (upper boundary)."""
        test_times = self._times.copy()
        test_times[3] = self._times[0] + 18
        with pytest.raises(RuntimeError):
            self._trajectory.position(test_times)
        with pytest.raises(RuntimeError):
            self._trajectory.velocity(test_times)
        with pytest.raises(RuntimeError):
            self._trajectory.acceleration(test_times)

    def test_trajectory_methods_extrapolation_error_4(self) -> None:
        """Test error handling when one query time is outside domain (lower boundary)."""
        test_times = self._times.copy()
        test_times[2] = self._times[0] - 9
        with pytest.raises(RuntimeError):
            self._trajectory.position(test_times)
        with pytest.raises(RuntimeError):
            self._trajectory.velocity(test_times)
        with pytest.raises(RuntimeError):
            self._trajectory.acceleration(test_times)
