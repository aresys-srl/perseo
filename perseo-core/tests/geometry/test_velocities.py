# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for geometry.velocities"""

import unittest

import numpy as np

from perseo_core.geometry.velocities import compute_ground_velocity
from tests.fixtures.geometry_utilities_data import get_ground_velocity_test_data
from tests.fixtures.models_data import get_testing_trajectory


class GroundVelocityFromTrajectoryTest(unittest.TestCase):
    """Test compute_ground_velocity with scalar and vectorized look angles."""

    def setUp(self) -> None:
        # Load test data from fixtures
        self._trajectory = get_testing_trajectory()

        velocity_data = get_ground_velocity_test_data()
        self._az_time = self._trajectory.domain[0] + velocity_data["azimuth_time_offset"]
        self._look_angles = velocity_data["look_angles_rad"]
        self._tolerance = velocity_data["tolerance"]
        self._expected_velocities = velocity_data["expected_velocities"]

    def test_compute_ground_velocity_from_trajectory_0(self) -> None:
        """Test compute_ground_velocity with scalar look angles (single case)."""
        ground_velocities = compute_ground_velocity(
            trajectory=self._trajectory,
            azimuth_time=self._az_time,
            look_angles_rad=self._look_angles[0],
        )
        self.assertIsInstance(ground_velocities, float)
        np.testing.assert_allclose(
            ground_velocities,
            self._expected_velocities[0],
            atol=self._tolerance,
            rtol=0,
        )

    def test_compute_ground_velocity_from_trajectory_1(self) -> None:
        """Test compute_ground_velocity with vectorized look angles (multiple cases)."""
        ground_velocities = compute_ground_velocity(
            trajectory=self._trajectory,
            azimuth_time=self._az_time,
            look_angles_rad=self._look_angles,
        )
        np.testing.assert_allclose(ground_velocities, self._expected_velocities, atol=self._tolerance, rtol=0)


if __name__ == "__main__":
    unittest.main()
