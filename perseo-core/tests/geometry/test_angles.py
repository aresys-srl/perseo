# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Testing geometry/angles.py functionalities"""

import itertools

import numpy as np
import pytest

from perseo_core.geometry.angles import (
    compute_incidence_angles,
    compute_incidence_angles_core,
    compute_look_angles,
    compute_look_angles_core,
    get_geometric_squint_angle,
)


class TestComputeLookAndIncidenceAngles:
    """Test compute_incidence_angles_core and compute_look_angles_core with various input dimensions."""

    @pytest.fixture(autouse=True)
    def setup_angles_data(self, angles_test_data):
        """Load test data from fixtures."""
        self.sensor_positions = angles_test_data["sensor_positions"]
        self.points = angles_test_data["points"]
        self.surface_normals = angles_test_data["surface_normals"]
        self.nadir_directions = angles_test_data["nadir_directions"]
        self.atol = angles_test_data["tolerance"]["atol"]
        self.rtol = angles_test_data["tolerance"]["rtol"]

    def test_compute_incidence_angles_core_1_pos_1_point(self):
        """Test compute_incidence_angles_core with scalar position and point (scalar and 1D array forms)."""
        reference_value = np.array([0.289504602345834])

        position_inputs = [
            self.sensor_positions[0],
            self.sensor_positions[0].reshape((1, 3)),
        ]

        point_inputs = [
            self.points[0],
            self.points[0].reshape((1, 3)),
        ]

        for position, point in itertools.product(position_inputs, point_inputs):
            incidence_angle = compute_incidence_angles_core(position, point)
            np.testing.assert_allclose(reference_value, incidence_angle, atol=self.atol, rtol=self.rtol)

    def test_compute_incidence_angles_core_1_pos_1_point_1_surface_normal(self):
        """Test compute_incidence_angles_core with scalar position, point, and surface normal.

        Validate support for different surface-normal shapes and normalization options.
        """
        reference_value = np.array([0.290207579201514])

        surface_normals = [
            (self.surface_normals[0], None),
            (self.surface_normals[0], False),
            (self.surface_normals[0].reshape((1, 3)), None),
            (self.surface_normals[0].reshape((1, 3)), False),
            (self.surface_normals[0] / np.linalg.norm(self.surface_normals[0]), True),
        ]

        for surface_normal, assume_normalized in surface_normals:
            options = {} if assume_normalized is None else {"assume_surface_normals_normalized": assume_normalized}

            incidence_angle = compute_incidence_angles_core(
                self.sensor_positions[0],
                self.points[0],
                surface_normals=surface_normal,
                **options,
            )

            np.testing.assert_allclose(reference_value, incidence_angle, atol=self.atol, rtol=self.rtol)

    def test_compute_incidence_angles_core_1_pos_n_point(self):
        """Test compute_incidence_angles_core with single sensor position and multiple points."""
        reference_value = np.array(
            [
                0.289504602345834,
                0.387105058129672,
                0.485521632720666,
                0.585097084845005,
                0.686313174870685,
                0.789892511505193,
                0.897005905237912,
            ]
        )

        position_inputs = [
            self.sensor_positions[0],
            self.sensor_positions[0].reshape((1, 3)),
        ]

        for position in position_inputs:
            incidence_angle = compute_incidence_angles_core(position, self.points)
            np.testing.assert_allclose(reference_value, incidence_angle, atol=self.atol, rtol=self.rtol)

    def test_compute_incidence_angles_core_1_pos_n_point_n_surface_normal(self):
        """Test compute_incidence_angles_core with single position, multiple points and normals."""
        reference_value = np.array(
            [
                0.290207579201514,
                0.387779384509382,
                0.486164927958404,
                0.585705866765378,
                0.686882472903164,
                0.790415131872372,
                0.897471035962306,
            ]
        )

        surface_normals = [
            (self.surface_normals, None),
            (self.surface_normals, False),
            (
                self.surface_normals / np.linalg.norm(self.surface_normals, axis=-1, keepdims=True),
                True,
            ),
        ]

        for surface_normal, assume_normalized in surface_normals:
            options = {} if assume_normalized is None else {"assume_surface_normals_normalized": assume_normalized}
            incidence_angle = compute_incidence_angles_core(
                self.sensor_positions[0],
                self.points,
                surface_normals=surface_normal,
                **options,
            )

            np.testing.assert_allclose(reference_value, incidence_angle, atol=self.atol, rtol=self.rtol)

    def test_compute_incidence_angles_core_n_pos_n_point(self):
        """Test compute_incidence_angles_core with multiple positions and points (vectorized computation)."""
        reference_value = np.array(
            [
                0.289504602345834,
                0.387283126037465,
                0.485979187965376,
                0.585817587250902,
                0.687224085709431,
                0.790895812723984,
                0.897997938422811,
            ]
        )

        incidence_angle = compute_incidence_angles_core(self.sensor_positions, self.points)
        np.testing.assert_allclose(reference_value, incidence_angle, atol=self.atol, rtol=self.rtol)

    def test_compute_incidence_angles_core_n_pos_n_point_n_surface_normal(
        self,
    ):
        """Test compute_incidence_angles_core with multiple positions, points, and surface normals."""
        reference_value = np.array(
            [
                0.290207579201514,
                0.387866437194178,
                0.486480609230522,
                0.586255132207420,
                0.687606324073378,
                0.791225249434776,
                0.898271544290844,
            ]
        )

        surface_normals = [
            (self.surface_normals, None),
            (self.surface_normals, False),
            (
                self.surface_normals / np.linalg.norm(self.surface_normals, axis=-1, keepdims=True),
                True,
            ),
        ]

        for surface_normal, assume_normalized in surface_normals:
            options = {} if assume_normalized is None else {"assume_surface_normals_normalized": assume_normalized}

            incidence_angle = compute_incidence_angles_core(
                self.sensor_positions,
                self.points,
                surface_normals=surface_normal,
                **options,
            )

            np.testing.assert_allclose(reference_value, incidence_angle, atol=self.atol, rtol=self.rtol)

    def test_compute_incidence_angles_core_invalid_inputs(self):
        """Test that compute_incidence_angles_core raises ValueError for invalid input shapes."""
        # wrong sensor positions shape
        with pytest.raises(ValueError):
            compute_incidence_angles_core(np.arange(5), np.arange(3))
        with pytest.raises(ValueError):
            compute_incidence_angles_core(np.arange(6).reshape(3, 2), np.arange(3))

        # wrong points shape
        with pytest.raises(ValueError):
            compute_incidence_angles_core(np.arange(3), np.arange(5))
        with pytest.raises(ValueError):
            compute_incidence_angles_core(np.arange(3), np.arange(6).reshape(3, 2))

        # incompatible points sensor positions shape
        with pytest.raises(ValueError):
            compute_incidence_angles_core(np.arange(12).reshape(4, 3), np.arange(6).reshape(2, 3))

        # incompatible points surface normals shape
        with pytest.raises(ValueError):
            compute_incidence_angles_core(
                np.arange(3),
                np.arange(12).reshape(4, 3),
                surface_normals=np.arange(12).reshape((6, 2)),
            )

    def test_compute_look_angles_core_1_pos_1_nadir_1_point(self):
        """Test compute_look_angles_core with scalar position, nadir direction, and point."""
        reference_value = np.array([0.261807718170898])

        position_inputs = [
            self.sensor_positions[0],
            self.sensor_positions[0].reshape((1, 3)),
        ]
        nadir_dir_inputs = [
            (self.nadir_directions[0], None),
            (self.nadir_directions[0].reshape((1, 3)), None),
            (self.nadir_directions[0], False),
            (self.nadir_directions[0].reshape((1, 3)), False),
            (self.nadir_directions[0] / np.linalg.norm(self.nadir_directions[0]), True),
        ]
        points_inputs = [
            self.points[0],
            self.points[0].reshape((1, 3)),
        ]

        for position, (nadir_dir, assume_normalized), point in itertools.product(
            position_inputs, nadir_dir_inputs, points_inputs
        ):
            options = {} if assume_normalized is None else {"assume_nadir_directions_normalized": assume_normalized}
            look_angle = compute_look_angles_core(position, point, nadir_dir, **options)

            np.testing.assert_allclose(reference_value, look_angle, atol=self.atol, rtol=self.rtol)

    def test_compute_look_angles_core_1_pos_1_nadir_n_point(self):
        """Test compute_look_angles_core with single position and nadir direction, multiple points."""
        reference_value = np.array(
            [
                0.261807718170898,
                0.349072925268265,
                0.436338618964629,
                0.523604555599314,
                0.610870630659514,
                0.698136791805967,
                0.785403009878366,
            ]
        )

        position_inputs = [
            self.sensor_positions[0],
            self.sensor_positions[0].reshape((1, 3)),
        ]
        nadir_dir_inputs = [
            (self.nadir_directions[0], None),
            (self.nadir_directions[0].reshape((1, 3)), None),
            (self.nadir_directions[0], False),
            (self.nadir_directions[0].reshape((1, 3)), False),
            (self.nadir_directions[0] / np.linalg.norm(self.nadir_directions[0]), True),
        ]

        for position, (nadir_dir, assume_normalized) in itertools.product(position_inputs, nadir_dir_inputs):
            options = {} if assume_normalized is None else {"assume_nadir_directions_normalized": assume_normalized}

            look_angle = compute_look_angles_core(position, self.points, nadir_dir, **options)

            np.testing.assert_allclose(reference_value, look_angle, atol=self.atol, rtol=self.rtol)

    def test_compute_look_angles_core_n_pos_1_nadir_n_point(self):
        """Testing compute look angles with mixed inputs"""
        reference_value = np.array(
            [
                0.261807718170898,
                0.349177447165984,
                0.436707504824461,
                0.524249216838951,
                0.611732608815392,
                0.699124457083078,
                0.786414020130282,
            ]
        )

        nadir_dir_inputs = [
            (self.nadir_directions[0], None),
            (self.nadir_directions[0].reshape((1, 3)), None),
            (self.nadir_directions[0], False),
            (self.nadir_directions[0].reshape((1, 3)), False),
            (self.nadir_directions[0] / np.linalg.norm(self.nadir_directions[0]), True),
        ]

        for nadir_dir, assume_normalized in nadir_dir_inputs:
            options = {} if assume_normalized is None else {"assume_nadir_directions_normalized": assume_normalized}

            look_angle = compute_look_angles_core(self.sensor_positions, self.points, nadir_dir, **options)

            np.testing.assert_allclose(reference_value, look_angle, atol=self.atol, rtol=self.rtol)

    def test_compute_look_angles_core_1_pos_n_nadir_n_point(self):
        """Testing compute look angles with mixed inputs"""
        reference_value = np.array(
            [
                0.261807718170898,
                0.349080180131945,
                0.436352948981580,
                0.523625656261822,
                0.610898076081333,
                0.698170031320458,
                0.785441361124307,
            ]
        )

        position_inputs = [
            self.sensor_positions[0],
            self.sensor_positions[0].reshape((1, 3)),
        ]
        nadir_dir_inputs = [
            (self.nadir_directions, None),
            (self.nadir_directions, False),
            (
                self.nadir_directions / np.linalg.norm(self.nadir_directions, axis=-1, keepdims=True),
                True,
            ),
        ]

        for position, (nadir_dir, assume_normalized) in itertools.product(position_inputs, nadir_dir_inputs):
            options = {} if assume_normalized is None else {"assume_nadir_directions_normalized": assume_normalized}

            look_angle = compute_look_angles_core(position, self.points, nadir_dir, **options)

            np.testing.assert_allclose(reference_value, look_angle, atol=self.atol, rtol=self.rtol)

    def test_compute_look_angles_core_n_pos_n_nadir_n_point(self):
        """Testing compute look angles with array inputs"""
        reference_value = np.array(
            [
                0.261807718170898,
                0.349151450027951,
                0.436618549676532,
                0.524083903169947,
                0.611489294069069,
                0.698809101292470,
                0.786038760035532,
            ]
        )

        nadir_dir_inputs = [
            (self.nadir_directions, None),
            (self.nadir_directions, False),
            (
                self.nadir_directions / np.linalg.norm(self.nadir_directions, axis=-1, keepdims=True),
                True,
            ),
        ]

        for nadir_dir, assume_normalized in nadir_dir_inputs:
            options = {} if assume_normalized is None else {"assume_nadir_directions_normalized": assume_normalized}

            look_angle = compute_look_angles_core(self.sensor_positions, self.points, nadir_dir, **options)

            np.testing.assert_allclose(reference_value, look_angle, atol=self.atol, rtol=self.rtol)

    def test_compute_look_angles_core_1_pos_1_point_no_nadir(self):
        """Testing compute look angles with mixed inputs, no nadir"""
        reference_value = 0.261807718170891

        s_positions = [self.sensor_positions[0], self.sensor_positions[0].reshape((1, 3))]
        g_positions = [self.points[0], self.points[0].reshape((1, 3))]
        for s_pos, g_pos in itertools.product(s_positions, g_positions):
            look_angle = compute_look_angles_core(sensor_positions=s_pos, ground_points=g_pos)
            np.testing.assert_allclose(reference_value, look_angle, atol=self.atol, rtol=self.rtol)

    def test_compute_look_angles_core_1_pos_n_points_no_nadir(self):
        """Testing compute look angles with mixed inputs, no nadir"""
        reference_value = np.array(
            [0.26180771817, 0.349072929252, 0.436338618964, 0.5236045556, 0.61087063065, 0.6981367918, 0.78540301]
        )

        s_positions = [self.sensor_positions[0], self.sensor_positions[0].reshape((1, 3))]

        for s_pos in s_positions:
            look_angle = compute_look_angles_core(sensor_positions=s_pos, ground_points=self.points)
            np.testing.assert_allclose(reference_value, look_angle, atol=self.atol, rtol=self.rtol)

    def test_compute_look_angles_core_n_pos_n_point_no_nadir(self):
        """Testing compute look angles with mixed inputs, no nadir"""
        reference_value = np.array(
            [
                0.2618077192020528,
                0.3491514509379109,
                0.4366185505194457,
                0.524083903974794,
                0.6114892948540702,
                0.6988091020706518,
                0.7860387608173248,
            ]
        )

        look_angle = compute_look_angles_core(sensor_positions=self.sensor_positions, ground_points=self.points)
        np.testing.assert_allclose(reference_value, look_angle, atol=self.atol, rtol=self.rtol)

    def test_compute_look_angles_core_invalid_inputs(self):
        """Testing compute look angles with invalid inputs"""
        # wrong point shape
        with pytest.raises(ValueError):
            compute_look_angles_core(
                np.arange(12).reshape(4, 3),
                np.arange(12).reshape(4, 3),
                np.arange(10).reshape(2, 5),
            )

        # wrong position shape
        with pytest.raises(ValueError):
            compute_look_angles_core(np.arange(5), np.arange(3), np.arange(3))
        with pytest.raises(ValueError):
            compute_look_angles_core(np.arange(6).reshape(3, 2), np.arange(3), np.arange(3))

        # wrong nadir direction shape
        with pytest.raises(ValueError):
            compute_look_angles_core(np.arange(3), np.arange(5), np.arange(3))
        with pytest.raises(ValueError):
            compute_look_angles_core(np.arange(3), np.arange(6).reshape(3, 2), np.arange(3))

        # incompatible shapes
        with pytest.raises(ValueError):
            compute_look_angles_core(
                np.arange(12).reshape(4, 3),
                np.arange(12).reshape(4, 3),
                np.arange(6).reshape(2, 3),
            )
        with pytest.raises(ValueError):
            compute_look_angles_core(np.arange(12).reshape(4, 3), np.arange(6), np.arange(12).reshape(4, 3))
        with pytest.raises(ValueError):
            compute_look_angles_core(
                np.arange(1, 4),
                np.arange(15).reshape((5, 3)),
                np.arange(12).reshape(4, 3),
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


class TestGeometricSquint:
    """Testing get_geometric_squint function"""

    @pytest.fixture(autouse=True)
    def setup_squint_data(self):
        """Setup squint test data."""
        self.pos_0 = np.array([-2449675.14554249, -5216814.136353868, 3907089.2898868835])
        self.pos_20 = np.array([4397940.093636902, 763963.1640477455, 4542599.8509511445])
        self.vel_0 = np.array([-3283.937062880771, -3101.7802725409233, -6163.652047976267])
        self.vel_20 = np.array([-141.05193267576627, -24.50203469983142, 139.73925487802535])
        self.point_0 = np.array([-2467483.2210648037, -4626385.185907534, 3619451.3347408967])
        self.point_20 = np.array([4397211.556651601, 766361.9969266983, 4540802.84326352])

        self.squint_ref_0 = -5.853944778390092e-11
        self.squint_ref_20 = -0.34229904620461094

        self.tolerance = 1e-16

    def test_get_geometric_squint_zero(self) -> None:
        """Testing getting an almost 0 squint angle"""
        squint = get_geometric_squint_angle(
            sensor_positions=self.pos_0,
            sensor_velocities=self.vel_0,
            ground_points=self.point_0,
        )
        assert isinstance(squint, float)
        np.testing.assert_allclose(squint, self.squint_ref_0, atol=self.tolerance, rtol=0)

    def test_get_geometric_squint_zero_deg(self) -> None:
        """Testing getting an almost 0 squint angle, degrees"""
        squint = get_geometric_squint_angle(
            sensor_positions=self.pos_0, sensor_velocities=self.vel_0, ground_points=self.point_0, radians=False
        )
        assert isinstance(squint, float)
        np.testing.assert_allclose(np.deg2rad(squint), self.squint_ref_0, atol=self.tolerance, rtol=0)

    def test_get_geometric_squint_non_zero(self) -> None:
        """Testing getting a squint angle different from 0"""
        squint = get_geometric_squint_angle(
            sensor_positions=self.pos_20,
            sensor_velocities=self.vel_20,
            ground_points=self.point_20,
        )
        assert isinstance(squint, float)
        np.testing.assert_allclose(squint, self.squint_ref_20, atol=self.tolerance, rtol=0)

    def test_get_geometric_squint_non_zero_deg(self) -> None:
        """Testing getting a squint angle different from 0, degrees"""
        squint = get_geometric_squint_angle(
            sensor_positions=self.pos_20, sensor_velocities=self.vel_20, ground_points=self.point_20, radians=False
        )
        assert isinstance(squint, float)
        np.testing.assert_allclose(np.deg2rad(squint), self.squint_ref_20, atol=self.tolerance, rtol=0)

    def test_get_multiple_squint(self) -> None:
        """Testing function vectorization"""
        squints = get_geometric_squint_angle(
            sensor_positions=np.vstack([self.pos_0, self.pos_20]),
            sensor_velocities=np.vstack([self.vel_0, self.vel_20]),
            ground_points=np.vstack([self.point_0, self.point_20]),
        )
        assert isinstance(squints, np.ndarray)
        assert squints.ndim == 1
        assert squints.size == 2
        np.testing.assert_allclose(
            squints,
            np.array([self.squint_ref_0, self.squint_ref_20]),
            atol=self.tolerance,
            rtol=0,
        )

    def test_get_multiple_squint_deg(self) -> None:
        """Testing function vectorization, degrees"""
        squints = get_geometric_squint_angle(
            sensor_positions=np.vstack([self.pos_0, self.pos_20]),
            sensor_velocities=np.vstack([self.vel_0, self.vel_20]),
            ground_points=np.vstack([self.point_0, self.point_20]),
            radians=False,
        )
        assert isinstance(squints, np.ndarray)
        assert squints.ndim == 1
        assert squints.size == 2
        np.testing.assert_allclose(
            np.deg2rad(squints),
            np.array([self.squint_ref_0, self.squint_ref_20]),
            atol=self.tolerance,
            rtol=0,
        )


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
