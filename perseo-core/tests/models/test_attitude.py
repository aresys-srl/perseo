# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for models/attitude.py Attitude object"""

import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp

# Adjust imports to your actual package structure
from perseo_core.models.attitude import Attitude


class TestAttitude(unittest.TestCase):
    """Testing Attitude object"""

    def setUp(self):
        self.times = np.array([0.0, 4.0, 8.0])
        self._euler_angles = np.deg2rad(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 45.0], [0.0, 45.0, 90.0]]))
        self.rotations = Rotation.from_euler(seq="ZYX", angles=self._euler_angles, degrees=False)
        self.attitude = Attitude(rotations=self.rotations, times=self.times)

    def test_properties(self):
        self.assertTrue(np.array_equal(self.attitude.times, self.times))
        self.assertEqual(self.attitude.domain, (0.0, 8.0))
        self.assertIsInstance(self.attitude.rotations, Rotation)

    def test_evaluate_at_knots(self):
        result = self.attitude.evaluate(self.times)
        assert_allclose(result.as_quat(), self.rotations.as_quat(), atol=1e-12)

    def test_evaluate_midpoints(self):
        query = np.array([2.0, 6.0])

        ref_slerp = Slerp(self.times, self.rotations)
        expected = ref_slerp(query)

        result = self.attitude.evaluate(query)

        assert_allclose(result.as_quat(), expected.as_quat(), atol=1e-10)

    def test_evaluate_output_shape(self):
        query = np.array([1.0, 2.0, 3.0])
        result = self.attitude.evaluate(query)
        self.assertEqual(result.as_quat().shape[0], 3)

    def test_extrapolation_below_domain(self):
        with self.assertRaises(RuntimeError):
            self.attitude.evaluate(np.array([-1.0]))

    def test_extrapolation_above_domain(self):
        with self.assertRaises(RuntimeError):
            self.attitude.evaluate(np.array([9.0]))

    def test_first_derivative_shape(self):
        query = np.array([2.0, 4.0])
        derivative = self.attitude.evaluate_first_derivatives(query)
        self.assertIsInstance(derivative, Rotation)
        self.assertEqual(derivative.as_quat().shape[0], 2)

    def test_from_quaternions(self):
        quats = self.rotations.as_quat()
        attitude = Attitude.from_quaternions(quats, self.times)

        query = np.array([2.0])
        result = attitude.evaluate(query)

        ref_slerp = Slerp(self.times, self.rotations)
        expected = ref_slerp(query)

        assert_allclose(result.as_quat(), expected.as_quat(), atol=1e-10)

    def test_from_euler_angles(self):
        attitude = Attitude.from_euler_angles(
            euler_angles_rad=self._euler_angles,
            rotation_order="YPR",
            times=self.times,
        )

        query = np.array([6.0])
        result = attitude.evaluate(query)

        ref_slerp = Slerp(self.times, self.rotations)
        expected = ref_slerp(query)

        assert_allclose(result.as_quat(), expected.as_quat(), atol=1e-10)


if __name__ == "__main__":
    unittest.main()
