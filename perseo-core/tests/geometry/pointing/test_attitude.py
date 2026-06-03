# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Testing for models/attitude.py Attitude object"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp

from perseo_core.geometry.pointing import (
    Attitude,
    compute_antenna_attitude_from_euler_angles,
    compute_sensor_attitude_from_state_vectors,
)


class TestAttitude:
    """Test Attitude object creation, properties, evaluation, and interpolation methods."""

    @pytest.fixture(autouse=True)
    def setup_attitude_data(self, attitude_test_data, reference_frames_test_data):
        data = attitude_test_data
        self.times = data["times"]
        self._euler_angles = data["euler_angles"]
        self.antenna_reference_frames = data["antenna_reference_frames"]
        self.attitude = Attitude(reference_frames=self.antenna_reference_frames, times=self.times)
        self._ref_data = reference_frames_test_data

    def test_properties(self):
        """Test that Attitude properties correctly return times, domain, and rotations."""
        assert np.array_equal(self.attitude.times, self.times)
        assert self.attitude.domain == (0.0, 8.0)
        assert isinstance(self.attitude.reference_frames, np.ndarray)

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
        assert result.shape == (7, 3, 3)

    def test_extrapolation_below_domain(self):
        """Test that Attitude.evaluate raises RuntimeError for times below domain."""
        with pytest.raises(RuntimeError):
            self.attitude.evaluate(np.array([-1.0]))

    def test_extrapolation_above_domain(self):
        """Test that Attitude.evaluate raises RuntimeError for times above domain."""
        with pytest.raises(RuntimeError):
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

    def test_compute_antenna_attitude_from_euler_angles(self):
        """Test compute_antenna_attitude_from_euler_angles creates correct attitude."""
        attitude = compute_antenna_attitude_from_euler_angles(
            ypr_rad=self._euler_angles,
            rotation_order="YPR",
            times=self.times,
            sensor_local_axis=np.eye(3),
        )

        assert isinstance(attitude, Attitude)

        assert np.array_equal(attitude.times, self.times)

        result = attitude.evaluate(self.times)
        assert result.shape == (3, 3, 3)

        # Verify reference frames are orthgonal matrices (R^T * R = I)
        for i in range(result.shape[0]):
            product = result[i].T @ result[i]
            assert_allclose(product, np.eye(3), atol=1e-10)

    def test_compute_sensor_attitude_from_state_vectors(self):
        """Test compute_sensor_attitude_from_state_vectors with ZERODOPPLER reference frame."""
        position = self._ref_data["sensor_position"]
        velocity = self._ref_data["sensor_velocity"]
        times = np.array([0.0, 1.0])
        position_2 = position + float(np.diff(times).squeeze()) * velocity

        positions = np.array([position, position_2])
        velocities = np.array([velocity, velocity])

        attitude = compute_sensor_attitude_from_state_vectors(
            position=positions,
            velocity=velocities,
            times=times,
            reference_frame="ZERODOPPLER",
        )

        assert isinstance(attitude, Attitude)
        assert np.array_equal(attitude.times, times)

        result = attitude.evaluate(times)
        assert result.shape == (2, 3, 3)

        # Verify reference frames are orthgonal matrices (R^T * R = I)
        for i in range(result.shape[0]):
            product = result[i].T @ result[i]
            assert_allclose(product, np.eye(3), atol=1e-10)
