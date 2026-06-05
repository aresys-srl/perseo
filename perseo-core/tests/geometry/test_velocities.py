# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for geometry.velocities"""

import numpy as np
import pytest

from perseo_core.geometry.navigation import CubicSplineTrajectory
from perseo_core.geometry.velocities import compute_ground_velocity


class TestGroundVelocityFromTrajectory:
    """Test compute_ground_velocity with scalar and vectorized look angles."""

    @pytest.fixture(autouse=True)
    def setup_ground_velocity_data(
        self, testing_trajectory: CubicSplineTrajectory, ground_velocity_test_data: dict
    ) -> None:
        """Load test data from fixtures."""
        self._trajectory = testing_trajectory

        self._az_time = self._trajectory.domain[0] + ground_velocity_test_data["azimuth_time_offset"]
        self._look_angles = ground_velocity_test_data["look_angles_rad"]
        self._tolerance = ground_velocity_test_data["tolerance"]
        self._expected_velocities = ground_velocity_test_data["expected_velocities"]

    def test_compute_ground_velocity_from_trajectory_0(self) -> None:
        """Test compute_ground_velocity with scalar look angles (single case)."""
        ground_velocities = compute_ground_velocity(
            trajectory=self._trajectory,
            azimuth_time=self._az_time,
            look_angles_rad=self._look_angles[0],
        )
        assert isinstance(ground_velocities, float)
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
