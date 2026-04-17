# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for models/trajectory.py CubicSplineTrajectory object"""

import unittest

import numpy as np

from perseo_core.models.cubic_spline_trajectory import CubicSplineTrajectory
from perseo_core.models.trajectory import Trajectory
from tests.fixtures.models_data import (
    get_trajectory_test_data,
)


class TrajectoryTest(unittest.TestCase):
    """Test CubicSplineTrajectory creation, properties, interpolation, and boundary conditions."""

    def setUp(self) -> None:
        # Load test data from fixtures
        data = get_trajectory_test_data()
        self._state_vectors = data["state_vectors"]
        self._trajectory = data["trajectory"]
        self._tolerance = data["tolerance"]
        self._times = data["test_times"]
        self._expected_pos = data["expected_positions"]
        self._expected_vel = data["expected_velocities"]
        self._expected_acc = data["expected_accelerations"]

    def test_trajectory_subclass(self) -> None:
        """Test that CubicSplineTrajectory is subclass of Trajectory protocol."""
        self.assertTrue(issubclass(CubicSplineTrajectory, Trajectory))

    def test_trajectory_creation(self) -> None:
        """Test CubicSplineTrajectory constructor creates valid instance."""
        trajectory = CubicSplineTrajectory(
            times=self._state_vectors.time_axis,
            positions=self._state_vectors.sensor_positions,
            velocities=self._state_vectors.sensor_velocities,
        )
        self.assertIsInstance(trajectory, CubicSplineTrajectory)

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
        with self.assertRaises(RuntimeError):
            self._trajectory.position(self._times[0] - 2)
        with self.assertRaises(RuntimeError):
            self._trajectory.velocity(self._times[-1] + 12)
        with self.assertRaises(RuntimeError):
            self._trajectory.acceleration(self._times[0] - 2)

    def test_trajectory_methods_extrapolation_error_2(self) -> None:
        """Test that position, velocity, acceleration raise error for vectorized times outside domain."""
        with self.assertRaises(RuntimeError):
            self._trajectory.position(self._times - 200)
        with self.assertRaises(RuntimeError):
            self._trajectory.velocity(self._times + 200)
        with self.assertRaises(RuntimeError):
            self._trajectory.acceleration(self._times - 200)

    def test_trajectory_methods_extrapolation_error_3(self) -> None:
        """Test error handling when one query time is outside domain (upper boundary)."""
        test_times = self._times.copy()
        test_times[3] = self._times[0] + 18
        with self.assertRaises(RuntimeError):
            self._trajectory.position(test_times)
        with self.assertRaises(RuntimeError):
            self._trajectory.velocity(test_times)
        with self.assertRaises(RuntimeError):
            self._trajectory.acceleration(test_times)

    def test_trajectory_methods_extrapolation_error_4(self) -> None:
        """Test error handling when one query time is outside domain (lower boundary)."""
        test_times = self._times.copy()
        test_times[2] = self._times[0] - 9
        with self.assertRaises(RuntimeError):
            self._trajectory.position(test_times)
        with self.assertRaises(RuntimeError):
            self._trajectory.velocity(test_times)
        with self.assertRaises(RuntimeError):
            self._trajectory.acceleration(test_times)


if __name__ == "__main__":
    unittest.main()
