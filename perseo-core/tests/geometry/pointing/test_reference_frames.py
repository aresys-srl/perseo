# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Testing geometry/utilities/reference_frames.py functionalities"""

import unittest

import numpy as np

from perseo_core.geometry.pointing.reference_frames import (
    compute_geocentric_reference_frame,
    compute_geodetic_point,
    compute_geodetic_reference_frame,
    compute_inertial_velocity,
    compute_sensor_local_axis,
    compute_zerodoppler_reference_frame,
)
from tests.fixtures.geometry_utilities_data import get_reference_frames_test_data


class SensorAxisTestCase(unittest.TestCase):
    """Test reference frame computation functions (zerodoppler, geocentric, geodetic, etc.)."""

    def setUp(self):
        # Load test data from fixtures
        data = get_reference_frames_test_data()
        self._sensor_position = data["sensor_position"]
        self._sensor_velocity = data["sensor_velocity"]
        self._zerodoppler_frame_reference = data["zerodoppler_frame_reference"]
        self._geocentric_frame_reference = data["geocentric_frame_reference"]
        self._geodetic_frame_reference = data["geodetic_frame_reference"]
        self._geodetic_point_reference = data["geodetic_point_reference"]
        self._tolerance = data["tolerance"]

    def test_compute_zerodoppler_reference_frame(self):
        """Test compute_zerodoppler_reference_frame with scalar inputs."""
        frame = compute_zerodoppler_reference_frame(self._sensor_position, self._sensor_velocity)
        np.testing.assert_allclose(frame, self._zerodoppler_frame_reference, rtol=1e-10, atol=1e-10)

    def test_compute_zerodoppler_reference_frame_vectorized(self):
        """Test compute_zerodoppler_reference_frame with vectorized inputs."""
        frame = compute_zerodoppler_reference_frame(
            self._sensor_position.reshape((1, 3)),
            self._sensor_velocity.reshape((1, 3)),
        )
        np.testing.assert_allclose(
            frame,
            self._zerodoppler_frame_reference.reshape((1, 3, 3)),
            rtol=1e-10,
            atol=1e-10,
        )

        frame = compute_zerodoppler_reference_frame(
            np.tile(self._sensor_position, (2, 1)), np.tile(self._sensor_velocity, (2, 1))
        )
        np.testing.assert_allclose(
            frame,
            np.tile(self._zerodoppler_frame_reference, (2, 1, 1)),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_compute_zerodoppler_reference_frame_invalid_input(self):
        """Test compute_zerodoppler_reference_frame raises error for invalid shape inputs."""
        with self.assertRaises(ValueError):
            compute_zerodoppler_reference_frame(np.ones((3, 10)), np.ones((3, 10)))

    def test_compute_geocentric_reference_frame(self):
        """Test compute_geocentric_reference_frame with scalar inputs."""
        frame = compute_geocentric_reference_frame(self._sensor_position, self._sensor_velocity)
        np.testing.assert_allclose(frame, self._geocentric_frame_reference, rtol=1e-10, atol=1e-10)

    def test_compute_geocentric_reference_frame_vectorized(self):
        """Test compute_geocentric_reference_frame with vectorized inputs."""
        frame = compute_geocentric_reference_frame(
            self._sensor_position.reshape((1, 3)),
            self._sensor_velocity.reshape((1, 3)),
        )
        np.testing.assert_allclose(
            frame,
            self._geocentric_frame_reference.reshape((1, 3, 3)),
            rtol=1e-10,
            atol=1e-10,
        )

        frame = compute_geocentric_reference_frame(
            np.tile(self._sensor_position, (2, 1)), np.tile(self._sensor_velocity, (2, 1))
        )
        np.testing.assert_allclose(
            frame,
            np.tile(self._geocentric_frame_reference, (2, 1, 1)),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_compute_geocentric_reference_frame_invalid_input(self):
        """Test compute_geocentric_reference_frame raises error for invalid shape combinations."""
        with self.assertRaises(ValueError):
            compute_geocentric_reference_frame(np.ones((10, 3)), np.ones((7, 3)))

        with self.assertRaises(ValueError):
            compute_geocentric_reference_frame(np.ones((3, 10)), np.ones((3, 10)))

    def test_compute_geodetic_reference_frame(self):
        """Test compute_geodetic_reference_frame with scalar inputs."""
        frame = compute_geodetic_reference_frame(self._sensor_position, self._sensor_velocity)
        np.testing.assert_allclose(frame, self._geodetic_frame_reference, rtol=1e-10, atol=1e-10)

    def test_compute_geodetic_reference_frame_vectorized(self):
        """Test compute_geodetic_reference_frame with vectorized inputs."""
        frame = compute_geodetic_reference_frame(
            self._sensor_position.reshape((1, 3)), self._sensor_velocity.reshape((1, 3))
        )
        np.testing.assert_allclose(
            frame,
            self._geodetic_frame_reference.reshape((1, 3, 3)),
            rtol=1e-10,
            atol=1e-10,
        )

        frame = compute_geodetic_reference_frame(
            np.tile(self._sensor_position, (2, 1)), np.tile(self._sensor_velocity, (2, 1))
        )
        np.testing.assert_allclose(
            frame,
            np.tile(self._geodetic_frame_reference, (2, 1, 1)),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_compute_geodetic_reference_frame_invalid_input(self):
        """Testing compute geodetic reference frame, with errors"""
        with self.assertRaises(ValueError):
            compute_geodetic_reference_frame(np.ones((10, 3)), np.ones((7, 3)))

        with self.assertRaises(ValueError):
            compute_geodetic_reference_frame(np.ones((3, 10)), np.ones((3, 10)))

    def test_compute_sensor_local_axis_zero_doppler(self):
        """Testing compute sensor local axis, zero doppler"""
        frame = compute_sensor_local_axis(self._sensor_position, self._sensor_velocity, "ZERODOPPLER")
        np.testing.assert_allclose(frame, self._zerodoppler_frame_reference, rtol=0, atol=self._tolerance)

    def test_compute_sensor_local_axis_geocentric(self):
        """Testing compute sensor local axis, geocentric"""
        frame = compute_sensor_local_axis(self._sensor_position, self._sensor_velocity, "GEOCENTRIC")
        np.testing.assert_allclose(frame, self._geocentric_frame_reference, rtol=0, atol=self._tolerance)

    def test_compute_sensor_local_axis_geodetic(self):
        """Testing compute sensor local axis, geodetic"""
        frame = compute_sensor_local_axis(self._sensor_position, self._sensor_velocity, "GEODETIC")
        np.testing.assert_allclose(frame, self._geodetic_frame_reference, rtol=0, atol=self._tolerance)

    def test_compute_sensor_local_axis_invalid_reference_frame(self):
        """Testing compute sensor local axis, with errors"""
        with self.assertRaises(ValueError):
            compute_sensor_local_axis(
                sensor_positions=self._sensor_position, sensor_velocities=self._sensor_velocity, reference_frame=None
            )

        with self.assertRaises(ValueError):
            compute_sensor_local_axis(
                sensor_positions=self._sensor_position,
                sensor_velocities=self._sensor_velocity,
                reference_frame="unknown name",
            )

        with self.assertRaises(ValueError):
            compute_sensor_local_axis(
                sensor_positions=self._sensor_position,
                sensor_velocities=self._sensor_velocity,
                reference_frame="geocentric",
            )

    def test_compute_geodetic_point(self):
        """Testing compute geodetic point"""
        geodetic_point = compute_geodetic_point(self._sensor_position)
        np.testing.assert_allclose(geodetic_point, self._geodetic_point_reference, rtol=0, atol=self._tolerance)


class InertialFramesTestCase(unittest.TestCase):
    """Testing compute inertial velocity function"""

    def setUp(self):
        self._sensor_position = np.asarray([26512.279931507, 1064819.379506800, 7083173.555337110])
        self._sensor_velocity = np.asarray([7529.609430015988, -342.978175622686, -23.376907795264])
        self._inertial_velocity_reference = np.asarray(
            [7.451961567220857e03, -3.410448694544030e02, -23.376907795264000]
        )
        self._tolerance = 1e-9

    def test_compute_inertial_velocity(self):
        inertial_velocity = compute_inertial_velocity(self._sensor_position, self._sensor_velocity)
        np.testing.assert_allclose(
            inertial_velocity,
            self._inertial_velocity_reference,
            rtol=0,
            atol=self._tolerance,
        )

    def test_compute_inertial_velocity_Vectorized(self):
        inertial_velocity = compute_inertial_velocity(
            self._sensor_position.reshape((1, 3)), self._sensor_velocity.reshape((1, 3))
        )
        np.testing.assert_allclose(
            inertial_velocity,
            self._inertial_velocity_reference.reshape((1, 3)),
            rtol=0,
            atol=self._tolerance,
        )

        inertial_velocity = compute_inertial_velocity(
            np.tile(self._sensor_position, (2, 1)), np.tile(self._sensor_velocity, (2, 1))
        )
        np.testing.assert_allclose(
            inertial_velocity,
            np.tile(self._inertial_velocity_reference, (2, 1)),
            rtol=10,
            atol=self._tolerance,
        )


if __name__ == "__main__":
    unittest.main()
