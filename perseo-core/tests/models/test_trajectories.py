# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for models/trajectory.py CubicSplineTrajectory object"""

import unittest

import numpy as np

from perseo_core.geometry.angles import compute_incidence_angles, compute_look_angles
from perseo_core.geometry.velocities import compute_ground_velocity
from perseo_core.models.protocols import ExtrapolationNotAllowed, TwiceDifferentiable3DCurve
from perseo_core.models.trajectories import CubicSplineTrajectory
from tests.common import get_testing_state_vectors


class TrajectoryProtocolComplianceTest(unittest.TestCase):
    """Testing CubicSplineTrajectory compliance with TwiceDifferentiable3DCurve protocol"""

    def test_trajectory_protocol_compliance(self) -> None:
        """Testing CubicSplineTrajectory protocol compliance"""
        self.assertIsInstance(CubicSplineTrajectory, TwiceDifferentiable3DCurve)


class TrajectoryTest(unittest.TestCase):
    """Testing CubicSplineTrajectory generation, properties and methods"""

    def setUp(self) -> None:
        # expected results
        self._state_vectors = get_testing_state_vectors()
        self._tolerance = 1e-6
        self._times = np.array([0.67, 2.56, 3.23, 5.8, 7.3, 8.4, 9.28]) + self._state_vectors["time_axis_origin"]
        self._expected_pos = np.array(
            [
                [-2541991.51667295, -5091823.70786736, 3905226.12193514],
                [-2541150.5219575, -5083245.3907668, 3916899.90420767],
                [-2540849.19560279, -5080199.19720824, 3921034.15400884],
                [-2539677.84205497, -5068489.32836387, 3936872.54869185],
                [-2538982.79233349, -5061636.31586544, 3946102.16084407],
                [-2538467.75565909, -5056602.13916507, 3952863.69473292],
                [-2538052.45005212, -5052569.50979786, 3958268.80885579],
            ]
        )
        self._expected_vel = np.array(
            [
                [441.4473803, 4533.05342227, 6181.09986721],
                [448.49214067, 4544.52639022, 6172.11662186],
                [450.98948273, 4548.58823598, 6168.92732246],
                [460.57005549, 4564.14349903, 6156.66343748],
                [466.16286972, 4573.20402512, 6149.48291368],
                [470.25419444, 4579.82631009, 6144.18013695],
                [473.58261429, 4585.18911946, 6140.06974106],
            ]
        )
        self._expected_acc = np.array(
            [
                [3.77763037, 6.095279, -5.02893538],
                [3.72759827, 6.06460755, -4.75951289],
                [3.72737768, 6.06038945, -4.76204088],
                [3.72838663, 6.04500478, -4.78121927],
                [3.72227767, 6.02750557, -4.80892479],
                [3.59289969, 5.85534651, -5.14276888],
                [2.60597199, 4.59067105, -7.62766475],
            ]
        )

    def test_trajectory_creation(self) -> None:
        """Test CubicSplineTrajectory creation through create_trajectory() function"""
        trajectory = CubicSplineTrajectory(
            times=self._state_vectors["time_axis"],
            positions=self._state_vectors["sensor_positions"],
            velocities=self._state_vectors["sensor_velocities"],
        )
        self.assertIsInstance(trajectory, CubicSplineTrajectory)

    def test_trajectory_properties(self) -> None:
        """Test CubicSplineTrajectory properties"""
        trajectory = CubicSplineTrajectory(
            times=self._state_vectors["time_axis"],
            positions=self._state_vectors["sensor_positions"],
            velocities=self._state_vectors["sensor_velocities"],
        )
        np.testing.assert_array_equal(trajectory.positions, self._state_vectors["sensor_positions"])
        np.testing.assert_array_equal(trajectory.velocities, self._state_vectors["sensor_velocities"])
        delta_times = trajectory.times - self._state_vectors["time_axis"]
        np.testing.assert_array_equal(delta_times.astype(float), np.zeros_like(delta_times, dtype=float))

    def test_trajectory_methods(self) -> None:
        """Test CubicSplineTrajectory evaluate methods"""
        trajectory = CubicSplineTrajectory(
            times=self._state_vectors["time_axis"],
            positions=self._state_vectors["sensor_positions"],
            velocities=self._state_vectors["sensor_velocities"],
        )
        np.testing.assert_allclose(
            trajectory.evaluate(self._times),
            self._expected_pos,
            atol=self._tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            trajectory.evaluate_first_derivatives(self._times),
            self._expected_vel,
            atol=self._tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            trajectory.evaluate_second_derivatives(self._times),
            self._expected_acc,
            atol=self._tolerance,
            rtol=0,
        )

    def test_trajectory_methods_extrapolation_error_1(self) -> None:
        """Test CubicSplineTrajectory evaluate methods raising extrapolation error"""
        trajectory = CubicSplineTrajectory(
            times=self._state_vectors["time_axis"],
            positions=self._state_vectors["sensor_positions"],
            velocities=self._state_vectors["sensor_velocities"],
        )
        with self.assertRaises(ExtrapolationNotAllowed):
            trajectory.evaluate(self._times[0] - 2)
        with self.assertRaises(ExtrapolationNotAllowed):
            trajectory.evaluate_first_derivatives(self._times[-1] + 12)
        with self.assertRaises(ExtrapolationNotAllowed):
            trajectory.evaluate_second_derivatives(self._times[0] - 2)

    def test_trajectory_methods_extrapolation_error_2(self) -> None:
        """Test CubicSplineTrajectory evaluate methods raising extrapolation error"""
        trajectory = CubicSplineTrajectory(
            times=self._state_vectors["time_axis"],
            positions=self._state_vectors["sensor_positions"],
            velocities=self._state_vectors["sensor_velocities"],
        )
        with self.assertRaises(ExtrapolationNotAllowed):
            trajectory.evaluate(self._times - 200)
        with self.assertRaises(ExtrapolationNotAllowed):
            trajectory.evaluate_first_derivatives(self._times + 200)
        with self.assertRaises(ExtrapolationNotAllowed):
            trajectory.evaluate_second_derivatives(self._times - 200)

    def test_trajectory_methods_extrapolation_error_3(self) -> None:
        """Test CubicSplineTrajectory evaluate methods raising extrapolation error"""
        trajectory = CubicSplineTrajectory(
            times=self._state_vectors["time_axis"],
            positions=self._state_vectors["sensor_positions"],
            velocities=self._state_vectors["sensor_velocities"],
        )
        test_times = self._times.copy()
        test_times[3] = self._times[0] + 18
        with self.assertRaises(ExtrapolationNotAllowed):
            trajectory.evaluate(test_times)
        with self.assertRaises(ExtrapolationNotAllowed):
            trajectory.evaluate_first_derivatives(test_times)
        with self.assertRaises(ExtrapolationNotAllowed):
            trajectory.evaluate_second_derivatives(test_times)

    def test_trajectory_methods_extrapolation_error_4(self) -> None:
        """Test CubicSplineTrajectory evaluate methods raising extrapolation error"""
        trajectory = CubicSplineTrajectory(
            times=self._state_vectors["time_axis"],
            positions=self._state_vectors["sensor_positions"],
            velocities=self._state_vectors["sensor_velocities"],
        )
        test_times = self._times.copy()
        test_times[2] = self._times[0] - 9
        with self.assertRaises(ExtrapolationNotAllowed):
            trajectory.evaluate(test_times)
        with self.assertRaises(ExtrapolationNotAllowed):
            trajectory.evaluate_first_derivatives(test_times)
        with self.assertRaises(ExtrapolationNotAllowed):
            trajectory.evaluate_second_derivatives(test_times)


class AnglesComputationFromTrajectoryTest(unittest.TestCase):
    """Testing angles computation from CubicSplineTrajectory object"""

    def setUp(self) -> None:
        self._state_vectors = get_testing_state_vectors()
        self._trajectory = CubicSplineTrajectory(
            times=self._state_vectors["time_axis"],
            positions=self._state_vectors["sensor_positions"],
            velocities=self._state_vectors["sensor_velocities"],
        )
        self._range_times = np.array([0.00362255, 0.003623, 0.0036239, 0.003635, 0.003639, 0.003642, 0.003645])
        self._azimuth_time = self._state_vectors["time_axis_origin"] + 2
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


class GroundVelocityFromTrajectoryTest(unittest.TestCase):
    """Testing ground velocity computation from CubicSplineTrajectory object"""

    def setUp(self) -> None:
        self._state_vectors = get_testing_state_vectors()
        self._trajectory = CubicSplineTrajectory(
            times=self._state_vectors["time_axis"],
            positions=self._state_vectors["sensor_positions"],
            velocities=self._state_vectors["sensor_velocities"],
        )
        self._az_time = self._state_vectors["time_axis_origin"] + 2.3
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
