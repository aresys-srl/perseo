# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Testing geometry/angles.py functionalities"""

import numpy as np
import pytest

from perseo_core.geometry.angles import (
    compute_incidence_angles,
    compute_look_angles,
)


class TestComputeLookIncidenceAnglesFromTrajectory:
    """Testing compute_incidence_angles_from_trajectory and compute_look_angles functions"""

    @pytest.fixture(autouse=True)
    def setup_trajectory_angles_data(self, angles_from_trajectory_test_data):
        """Load trajectory angle test fixture data."""
        self.trajectory = angles_from_trajectory_test_data["trajectory"]
        self.range_times = angles_from_trajectory_test_data["range_times"]
        self.az_time = angles_from_trajectory_test_data["az_time"]
        self.look_dir = angles_from_trajectory_test_data["look_dir"]
        self.tolerance = angles_from_trajectory_test_data["tolerance"]
        self.look_angles_expected = angles_from_trajectory_test_data["look_angles_expected"]
        self.incidence_angles_expected = angles_from_trajectory_test_data["incidence_angles_expected"]

    def test_compute_look_angles_from_trajectory_case0(self) -> None:
        """Testing compute_look_angles function, case 0"""

        # case 0: single range value
        angles = compute_look_angles(
            trajectory=self.trajectory,
            azimuth_time=self.az_time,
            range_times=self.range_times[0],
            look_direction=self.look_dir,
        )

        # checking results
        assert isinstance(angles, float)
        np.testing.assert_allclose(angles, self.look_angles_expected[0], atol=self.tolerance, rtol=0)

    def test_compute_look_angles_from_trajectory_case0_deg(self) -> None:
        """Testing compute_look_angles function, case 0, degrees"""

        # case 0: single range value
        angles = compute_look_angles(
            trajectory=self.trajectory,
            azimuth_time=self.az_time,
            range_times=self.range_times[0],
            look_direction=self.look_dir,
            radians=False,
        )

        # checking results
        assert isinstance(angles, float)
        np.testing.assert_allclose(np.deg2rad(angles), self.look_angles_expected[0], atol=self.tolerance, rtol=0)

    def test_compute_look_angles_from_trajectory_case1(self) -> None:
        """Testing compute_look_angles function, case 1"""

        # case 1: range array
        angles = compute_look_angles(
            trajectory=self.trajectory,
            azimuth_time=self.az_time,
            range_times=self.range_times,
            look_direction=self.look_dir,
        )

        # checking results
        assert isinstance(angles, np.ndarray)
        assert angles.ndim == 1
        assert angles.size == self.look_angles_expected.size
        np.testing.assert_allclose(angles, self.look_angles_expected, atol=self.tolerance, rtol=0)

    def test_compute_look_angles_from_trajectory_case1_deg(self) -> None:
        """Testing compute_look_angles function, case 1, degrees"""

        # case 1: range array
        angles = compute_look_angles(
            trajectory=self.trajectory,
            azimuth_time=self.az_time,
            range_times=self.range_times,
            look_direction=self.look_dir,
            radians=False,
        )

        # checking results
        assert isinstance(angles, np.ndarray)
        assert angles.ndim == 1
        assert angles.size == self.look_angles_expected.size
        np.testing.assert_allclose(np.deg2rad(angles), self.look_angles_expected, atol=self.tolerance, rtol=0)

    def test_compute_incidence_angles_from_trajectory_case0(self) -> None:
        """Testing compute_incidence_angles_from_trajectory function, case 0"""

        # case 0: single range value
        angles = compute_incidence_angles(
            trajectory=self.trajectory,
            azimuth_time=self.az_time,
            range_times=self.range_times[0],
            look_direction=self.look_dir,
        )

        # checking results
        assert isinstance(angles, float)
        np.testing.assert_allclose(angles, self.incidence_angles_expected[0], atol=self.tolerance, rtol=0)

    def test_compute_incidence_angles_from_trajectory_case0_deg(self) -> None:
        """Testing compute_incidence_angles_from_trajectory function, case 0, degrees"""

        # case 0: single range value
        angles = compute_incidence_angles(
            trajectory=self.trajectory,
            azimuth_time=self.az_time,
            range_times=self.range_times[0],
            look_direction=self.look_dir,
            radians=False,
        )

        # checking results
        assert isinstance(angles, float)
        np.testing.assert_allclose(np.deg2rad(angles), self.incidence_angles_expected[0], atol=self.tolerance, rtol=0)

    def test_compute_incidence_angles_from_trajectory_case1(self) -> None:
        """Testing compute_incidence_angles_from_trajectory function, case 1"""

        # case 1: range array
        angles = compute_incidence_angles(
            trajectory=self.trajectory,
            azimuth_time=self.az_time,
            range_times=self.range_times,
            look_direction=self.look_dir,
        )

        # checking results
        assert isinstance(angles, np.ndarray)
        assert angles.ndim == 1
        assert angles.size == self.incidence_angles_expected.size
        np.testing.assert_allclose(angles, self.incidence_angles_expected, atol=self.tolerance, rtol=0)

    def test_compute_incidence_angles_from_trajectory_case1_deg(self) -> None:
        """Testing compute_incidence_angles_from_trajectory function, case 1, degrees"""

        # case 1: range array
        angles = compute_incidence_angles(
            trajectory=self.trajectory,
            azimuth_time=self.az_time,
            range_times=self.range_times,
            look_direction=self.look_dir,
            radians=False,
        )

        # checking results
        assert isinstance(angles, np.ndarray)
        assert angles.ndim == 1
        assert angles.size == self.incidence_angles_expected.size
        np.testing.assert_allclose(np.deg2rad(angles), self.incidence_angles_expected, atol=self.tolerance, rtol=0)


class TestAnglesComputationFromTrajectory:
    """Testing angles computation from CubicSplineTrajectory object"""

    @pytest.fixture(autouse=True)
    def setup_trajectory_angles_data(self, testing_trajectory):
        """Load test data from fixtures."""
        self._trajectory = testing_trajectory
        self._range_times = np.array([0.00362255, 0.003623, 0.0036239, 0.003635, 0.003639, 0.003642, 0.003645])
        self._azimuth_time = self._trajectory.domain[0] + 2
        self._geocoding_side = "RIGHT"
        # expected results
        self._tolerance = 1e-9
        self._expected_look_angles = np.array(
            [
                0.20348233735250404,
                0.2040352825385317,
                0.20513633655495583,
                0.21822227400375235,
                0.22273297530481562,
                0.22605128004217567,
                0.22931680025662243,
            ]
        )
        self._expected_incidence_angles = np.array(
            [
                0.22003649511651838,
                0.2206377522894163,
                0.22183504499141393,
                0.23606867691985423,
                0.2409767142368985,
                0.24458790669056735,
                0.24814215077561505,
            ]
        )

    def test_compute_look_angles_single_range(self) -> None:
        """Testing compute_look_angles_from_trajectory with a single range value"""
        look_angles = compute_look_angles(
            trajectory=self._trajectory,
            azimuth_time=self._azimuth_time,
            range_times=self._range_times[0],
            look_direction=self._geocoding_side,
        )
        np.testing.assert_allclose(look_angles, self._expected_look_angles[0], atol=self._tolerance, rtol=0)

    def test_compute_look_angles_range_array(self) -> None:
        """Testing compute_look_angles_from_trajectory with a range array"""
        look_angles = compute_look_angles(
            trajectory=self._trajectory,
            azimuth_time=self._azimuth_time,
            range_times=self._range_times,
            look_direction=self._geocoding_side,
        )
        np.testing.assert_allclose(look_angles, self._expected_look_angles, atol=self._tolerance, rtol=0)

    def test_compute_incidence_angles_single_range(self) -> None:
        """Testing compute_incidence_angles_from_trajectory with a single range value"""
        incidence_angles = compute_incidence_angles(
            trajectory=self._trajectory,
            azimuth_time=self._azimuth_time,
            range_times=self._range_times[0],
            look_direction=self._geocoding_side,
        )
        np.testing.assert_allclose(
            incidence_angles,
            self._expected_incidence_angles[0],
            atol=self._tolerance,
            rtol=0,
        )

    def test_compute_incidence_angles_range_array(self) -> None:
        """Testing compute_incidence_angles_from_trajectory with a range array"""
        incidence_angles = compute_incidence_angles(
            trajectory=self._trajectory,
            azimuth_time=self._azimuth_time,
            range_times=self._range_times,
            look_direction=self._geocoding_side,
        )
        np.testing.assert_allclose(
            incidence_angles,
            self._expected_incidence_angles,
            atol=self._tolerance,
            rtol=0,
        )
