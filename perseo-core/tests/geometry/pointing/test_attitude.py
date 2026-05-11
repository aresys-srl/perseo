# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for models/attitude.py Attitude object"""

import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp

from perseo_core.geometry.pointing.attitude import Attitude
from tests.fixtures.models_data import get_attitude_test_data


class TestAttitude(unittest.TestCase):
    """Test Attitude object creation, properties, evaluation, and interpolation methods."""

    def setUp(self):
        # Load test data from fixtures
        data = get_attitude_test_data()
        self.times = data["times"]
        self._euler_angles = data["euler_angles"]
        self.antenna_reference_frames = data["antenna_reference_frames"]
        self.attitude = Attitude(antenna_reference_frames=self.antenna_reference_frames, times=self.times)

    def test_properties(self):
        """Test that Attitude properties correctly return times, domain, and rotations."""
        self.assertTrue(np.array_equal(self.attitude.times, self.times))
        self.assertEqual(self.attitude.domain, (0.0, 8.0))
        self.assertIsInstance(self.attitude.antenna_reference_frames, np.ndarray)

    def test_evaluate_at_knots(self):
        """Test that Attitude.evaluate returns exact values at knot points."""
        result = self.attitude.evaluate(self.times)
        assert_allclose(result, self.antenna_reference_frames, atol=1e-12)

    def test_evaluate_midpoints(self):
        """Test that Attitude.evaluate interpolates correctly between knot points using SLERP."""
        query = np.array([2.0, 6.0])

        ref_slerp = Slerp(self.times, Rotation.from_matrix(self.antenna_reference_frames))
        expected = ref_slerp(query)

        result = self.attitude.evaluate(query)

        assert_allclose(result, expected.as_matrix(), atol=1e-10)

    def test_evaluate_output_shape(self):
        """Test that Attitude.evaluate returns output with correct shape for multiple query points."""
        query = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        result = self.attitude.evaluate(query)
        self.assertEqual(result.shape, (7, 3, 3))

    def test_extrapolation_below_domain(self):
        """Test that Attitude.evaluate raises RuntimeError for times below domain."""
        with self.assertRaises(RuntimeError):
            self.attitude.evaluate(np.array([-1.0]))

    def test_extrapolation_above_domain(self):
        """Test that Attitude.evaluate raises RuntimeError for times above domain."""
        with self.assertRaises(RuntimeError):
            self.attitude.evaluate(np.array([9.0]))

    def test_from_quaternions(self):
        """Test that Attitude.from_quaternions creates object that evaluates correctly."""
        quats = Rotation.from_matrix(self.antenna_reference_frames).as_quat()
        attitude = Attitude.from_quaternions(quats, self.times)

        query = np.array([2.0])
        result = attitude.evaluate(query)

        ref_slerp = Slerp(self.times, Rotation.from_matrix(self.antenna_reference_frames))
        expected = ref_slerp(query)

        assert_allclose(result, expected.as_matrix(), atol=1e-10)

    def test_from_euler_angles(self):
        """Test that Attitude.from_euler_angles creates object that evaluates correctly."""
        attitude = Attitude.from_euler_angles(
            euler_angles_rad=self._euler_angles,
            rotation_order="YPR",
            times=self.times,
        )

        query = np.array([6.0])
        result = attitude.evaluate(query)

        ref_slerp = Slerp(self.times, Rotation.from_matrix(self.antenna_reference_frames))
        expected = ref_slerp(query)

        assert_allclose(result, expected.as_matrix(), atol=1e-10)


if __name__ == "__main__":
    unittest.main()
