# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for models/trajectory.py CubicSplineTrajectory object"""

import unittest

import numpy as np

from perseo_core.models.cubic_spline_trajectory import CubicSplineTrajectory
from perseo_core.models.protocols import TwiceDifferentiable3DCurve
from tests.common import get_testing_state_vectors, get_testing_trajectory


class TrajectoryProtocolComplianceTest(unittest.TestCase):
    """Testing CubicSplineTrajectory compliance with TwiceDifferentiable3DCurve protocol"""

    def test_trajectory_protocol_compliance(self) -> None:
        """Testing CubicSplineTrajectory protocol compliance"""
        self.assertIsInstance(CubicSplineTrajectory, TwiceDifferentiable3DCurve)


class TrajectoryTest(unittest.TestCase):
    """Testing CubicSplineTrajectory generation, properties and methods"""

    def setUp(self) -> None:
        self._state_vectors = get_testing_state_vectors()
        self._trajectory = get_testing_trajectory()
        self._tolerance = 1e-6
        self._times = np.array([0.67, 2.56, 3.23, 5.8, 7.3, 8.4, 9.28]) + self._trajectory.domain[0]
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
            times=self._state_vectors.time_axis,
            positions=self._state_vectors.sensor_positions,
            velocities=self._state_vectors.sensor_velocities,
        )
        self.assertIsInstance(trajectory, CubicSplineTrajectory)

    def test_trajectory_properties(self) -> None:
        """Test CubicSplineTrajectory properties"""
        np.testing.assert_array_equal(self._trajectory.positions, self._state_vectors.sensor_positions)
        np.testing.assert_array_equal(self._trajectory.velocities, self._state_vectors.sensor_velocities)
        delta_times = self._trajectory.times - self._state_vectors.time_axis
        np.testing.assert_array_equal(delta_times.astype(float), np.zeros_like(delta_times, dtype=float))

    def test_trajectory_methods(self) -> None:
        """Test CubicSplineTrajectory evaluate methods"""
        np.testing.assert_allclose(
            self._trajectory.evaluate(self._times),
            self._expected_pos,
            atol=self._tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            self._trajectory.evaluate_first_derivatives(self._times),
            self._expected_vel,
            atol=self._tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            self._trajectory.evaluate_second_derivatives(self._times),
            self._expected_acc,
            atol=self._tolerance,
            rtol=0,
        )

    def test_trajectory_methods_extrapolation_error_1(self) -> None:
        """Test CubicSplineTrajectory evaluate methods raising extrapolation error"""
        with self.assertRaises(RuntimeError):
            self._trajectory.evaluate(self._times[0] - 2)
        with self.assertRaises(RuntimeError):
            self._trajectory.evaluate_first_derivatives(self._times[-1] + 12)
        with self.assertRaises(RuntimeError):
            self._trajectory.evaluate_second_derivatives(self._times[0] - 2)

    def test_trajectory_methods_extrapolation_error_2(self) -> None:
        """Test CubicSplineTrajectory evaluate methods raising extrapolation error"""
        with self.assertRaises(RuntimeError):
            self._trajectory.evaluate(self._times - 200)
        with self.assertRaises(RuntimeError):
            self._trajectory.evaluate_first_derivatives(self._times + 200)
        with self.assertRaises(RuntimeError):
            self._trajectory.evaluate_second_derivatives(self._times - 200)

    def test_trajectory_methods_extrapolation_error_3(self) -> None:
        """Test CubicSplineTrajectory evaluate methods raising extrapolation error"""
        test_times = self._times.copy()
        test_times[3] = self._times[0] + 18
        with self.assertRaises(RuntimeError):
            self._trajectory.evaluate(test_times)
        with self.assertRaises(RuntimeError):
            self._trajectory.evaluate_first_derivatives(test_times)
        with self.assertRaises(RuntimeError):
            self._trajectory.evaluate_second_derivatives(test_times)

    def test_trajectory_methods_extrapolation_error_4(self) -> None:
        """Test CubicSplineTrajectory evaluate methods raising extrapolation error"""
        test_times = self._times.copy()
        test_times[2] = self._times[0] - 9
        with self.assertRaises(RuntimeError):
            self._trajectory.evaluate(test_times)
        with self.assertRaises(RuntimeError):
            self._trajectory.evaluate_first_derivatives(test_times)
        with self.assertRaises(RuntimeError):
            self._trajectory.evaluate_second_derivatives(test_times)


if __name__ == "__main__":
    unittest.main()
