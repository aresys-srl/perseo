# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for geometry.velocities"""

import unittest

import numpy as np

from perseo_core.geometry.velocities import compute_ground_velocity
from tests.common import get_testing_trajectory


class GroundVelocityFromTrajectoryTest(unittest.TestCase):
    """Testing ground velocity computation from CubicSplineTrajectory object"""

    def setUp(self) -> None:
        self._trajectory = get_testing_trajectory()
        self._az_time = self._trajectory.domain[0] + 2.3
        self._look_angles = np.deg2rad(np.arange(15.0, 50.0, 5.0))
        # expected results
        self._tolerance = 1e-6
        self._expected_velocities = np.array(
            [
                7073.866866931723,
                7068.0743880794025,
                7061.324649385491,
                7053.284329192286,
                7043.447653796795,
                7031.009082103516,
                7014.600593133645,
            ]
        )

    def test_compute_ground_velocity_from_trajectory_0(self) -> None:
        """Test compute_ground_velocity_from_trajectory, case 0"""
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
        """Test compute_ground_velocity_from_trajectory, case 0"""
        ground_velocities = compute_ground_velocity(
            trajectory=self._trajectory,
            azimuth_time=self._az_time,
            look_angles_rad=self._look_angles,
        )
        np.testing.assert_allclose(ground_velocities, self._expected_velocities, atol=self._tolerance, rtol=0)


if __name__ == "__main__":
    unittest.main()
