# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Testing for geometry/geocoding/direct_geocoding.py geocoding with looks functionalities"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from perseo_core.geometry.geocoding.direct_geocoding import (
    direct_geocoding_with_look_angles,
    direct_geocoding_with_looking_direction,
    direct_geocoding_with_pointing,
)
from tests.common import compute_antenna_angles_a_posteriori


class TestDirectGeocodingWithLooks:
    """Testing direct_geocoding_with_look_angles and direct_geocoding_with_looking_direction functionalities"""

    @pytest.fixture(autouse=True)
    def setup_direct_geocoding_data(self, direct_geocoding_with_looks_test_data):
        data = direct_geocoding_with_looks_test_data
        self.sensor_position = data["sensor_positions"][0, :]
        self.sensor_velocity = data["sensor_velocity"]
        self.looking_direction = data["looking_direction"]
        self.wrong_looking_direction = data["wrong_looking_direction"]
        self.reference_point = data["reference_point"]
        self.expected_ground_points_for_look_angles = data["expected_ground_points_for_look_angles"]
        self.tolerance = data["tolerance"]

    def test_direct_geocoding_with_looking_direction(self) -> None:
        """Testing direct_geocoding_with_looking_direction, single position"""
        geodetic_altitude = 1000.0

        position_inputs = [self.sensor_position, self.sensor_position.reshape((1, 3))]
        dir_inputs = [self.looking_direction, self.looking_direction.reshape((1, 3))]

        for origin, direction in itertools.product(position_inputs, dir_inputs):
            point = direct_geocoding_with_looking_direction(origin, direction, altitude=geodetic_altitude)

            assert point.shape == np.broadcast_shapes(origin.shape, direction.shape)
            np.testing.assert_allclose(
                point,
                self.reference_point.reshape(point.shape),
                atol=self.tolerance["atol"],
                rtol=self.tolerance["rtol"],
            )

    def test_direct_geocoding_with_looking_direction_vectorized(self) -> None:
        """Testing direct_geocoding_with_looking_direction, n positions"""
        directions = np.vstack(
            (
                np.tile(self.looking_direction, (5, 1)),
                self.wrong_looking_direction,
            )
        )

        geodetic_altitude = 1000.0

        points = direct_geocoding_with_looking_direction(
            self.sensor_position,
            directions,
            altitude=geodetic_altitude,
        )

        np.testing.assert_allclose(
            points[0:5], np.tile(self.reference_point, (5, 1)), atol=self.tolerance["atol"], rtol=self.tolerance["rtol"]
        )
        assert all(np.isnan(points[5]))

        points = direct_geocoding_with_looking_direction(
            np.tile(self.sensor_position, (2, 1)),
            self.looking_direction,
            altitude=geodetic_altitude,
        )

        np.testing.assert_allclose(
            points, np.tile(self.reference_point, (2, 1)), atol=self.tolerance["atol"], rtol=self.tolerance["rtol"]
        )

        points = direct_geocoding_with_looking_direction(
            np.tile(self.sensor_position, (2, 1)),
            np.tile(self.looking_direction, (2, 1)),
            altitude=geodetic_altitude,
        )

        np.testing.assert_allclose(
            points, np.tile(self.reference_point, (2, 1)), atol=self.tolerance["atol"], rtol=self.tolerance["rtol"]
        )

    def test_direct_geocoding_with_looking_direction_empty_intersection(self) -> None:
        """Testing direct_geocoding_with_looking_direction, no intersection"""
        point = direct_geocoding_with_looking_direction(self.sensor_position, self.wrong_looking_direction)

        assert all(np.isnan(point))

    def test_direct_geocoding_with_looking_direction_invalid_inputs(self) -> None:
        """Testing direct_geocoding_with_looking_direction, invalid inputs"""
        with pytest.raises(ValueError):
            direct_geocoding_with_looking_direction(np.arange(5), np.arange(3))
        with pytest.raises(ValueError):
            direct_geocoding_with_looking_direction(np.arange(6).reshape(3, 2), np.arange(3))
        with pytest.raises(ValueError):
            direct_geocoding_with_looking_direction(np.arange(3), np.arange(5))
        with pytest.raises(ValueError):
            direct_geocoding_with_looking_direction(np.arange(3), np.arange(6).reshape(3, 2))

    def test_direct_geocoding_with_look_angles_case0a(self) -> None:
        """Testing direct_geocoding_with_look_angles, single position, single velocity, n look angles"""
        look_angles = np.deg2rad(np.arange(15, 50, 5))
        altitude = 0
        points = direct_geocoding_with_look_angles(
            self.sensor_position,
            self.sensor_velocity,
            "ZERODOPPLER",
            look_angles,
            altitude=altitude,
        )

        np.testing.assert_allclose(
            points,
            self.expected_ground_points_for_look_angles,
            atol=self.tolerance["atol"],
            rtol=self.tolerance["rtol"],
        )

    def test_direct_geocoding_with_look_angles_case0b(self) -> None:
        """Testing direct_geocoding_with_look_angles, single position, single velocity, n look angles"""
        look_angles = np.deg2rad(np.arange(15, 50, 5))
        altitude = 0
        points = direct_geocoding_with_look_angles(
            self.sensor_position.reshape(1, 3),
            self.sensor_velocity.reshape(1, 3),
            "ZERODOPPLER",
            look_angles,
            altitude=altitude,
        )

        np.testing.assert_allclose(
            points,
            self.expected_ground_points_for_look_angles,
            atol=self.tolerance["atol"],
            rtol=self.tolerance["rtol"],
        )

    def test_direct_geocoding_with_look_angles_case1(self) -> None:
        """Testing direct_geocoding_with_look_angles, single position, single velocity, single look angle"""
        look_angles = np.deg2rad(np.arange(15, 50, 5))
        altitude = 0
        point = direct_geocoding_with_look_angles(
            self.sensor_position,
            self.sensor_velocity,
            "ZERODOPPLER",
            look_angles[0],
            altitude=altitude,
        )

        np.testing.assert_allclose(
            point,
            self.expected_ground_points_for_look_angles[0],
            atol=self.tolerance["atol"],
            rtol=self.tolerance["rtol"],
        )

    def test_direct_geocoding_with_look_angles_case2(self) -> None:
        """Testing direct_geocoding_with_look_angles, n positions, n velocities, n look angles"""
        look_angles = np.deg2rad(np.arange(15, 50, 5))
        altitude = 0
        points = direct_geocoding_with_look_angles(
            np.tile(self.sensor_position, (10, 1)),
            np.tile(self.sensor_velocity, (10, 1)),
            "ZERODOPPLER",
            np.tile(look_angles[0], (10,)),
            altitude=altitude,
        )

        np.testing.assert_allclose(
            points,
            np.tile(self.expected_ground_points_for_look_angles[0], (10, 1)),
            atol=self.tolerance["atol"],
            rtol=self.tolerance["rtol"],
        )

    def test_direct_geocoding_with_look_angles_invalid_inputs(self) -> None:
        """Testing direct_geocoding_with_look_angles with invalid inputs"""
        with pytest.raises(ValueError):
            direct_geocoding_with_look_angles(
                np.arange(5, dtype=float),
                np.arange(3, dtype=float),
                "ZERODOPPLER",
                np.arange(5, dtype=float),
            )
        with pytest.raises(ValueError):
            direct_geocoding_with_look_angles(
                np.arange(3, dtype=float),
                np.arange(4, dtype=float),
                "ZERODOPPLER",
                np.arange(5, dtype=float),
            )
        with pytest.raises(AssertionError):
            direct_geocoding_with_look_angles(
                np.arange(3, dtype=float),
                np.arange(3, dtype=float),
                "ZERODOPPLER",
                np.arange(6, dtype=float).reshape(2, 3),
            )


class TestDirectGeocodingWithPointing:
    """Testing direct_geocoding_with_pointing functionalities"""

    @pytest.fixture(autouse=True)
    def setup_pointing_data(self, direct_geocoding_with_looks_test_data):
        data = direct_geocoding_with_looks_test_data
        self.sensor_positions = data["sensor_positions"]
        self.arf = data["arf"]
        self.tolerance = data["tolerance"]
        self.expected_results = data["expected_ground_points_with_pointing"]

    def test_direct_geocoding_with_pointing(self):
        """Testing direct_geocoding_with_pointing, single position"""
        points = direct_geocoding_with_pointing(
            sensor_positions=self.sensor_positions[3, :],
            antenna_reference_frames=self.arf,
            azimuth_antenna_angles=-0.08726646259971647,
            elevation_antenna_angles=-0.05235987755982989,
            altitude=0,
        )
        np.testing.assert_allclose(
            points, self.expected_results[0], atol=self.tolerance["atol"], rtol=self.tolerance["rtol"]
        )

    def test_direct_geocoding_with_pointing_vectorized(self):
        """Testing direct_geocoding_with_pointing, vectorization"""
        sensor_pos_inputs = [self.sensor_positions[3, :], np.tile(self.sensor_positions[3, :], (10, 1))]
        az_angles_in = np.deg2rad(np.linspace(-5, 5, 10))
        el_angles_in = np.deg2rad(np.linspace(-3, 2, 10))

        azimuth_angles_inputs = [az_angles_in, az_angles_in[0]]
        elevation_angles_inputs = [el_angles_in, el_angles_in[4]]
        altitude_inputs = [0, 1000, -1000]

        for sensor_pos, az, el, height in itertools.product(
            sensor_pos_inputs, azimuth_angles_inputs, elevation_angles_inputs, altitude_inputs
        ):
            if sensor_pos.ndim == 1:
                arf = self.arf
            else:
                arf = np.tile(self.arf, (sensor_pos.shape[0], 1, 1))
            points = direct_geocoding_with_pointing(
                sensor_positions=sensor_pos,
                antenna_reference_frames=arf,
                azimuth_antenna_angles=az,
                elevation_antenna_angles=el,
                altitude=height,
            )
            expected_shape = (3,)
            if not (sensor_pos.size == 3 and np.ndim(az) == 0 and np.ndim(el) == 0):
                expected_shape = (
                    max(sensor_pos.shape[0], np.size(az), np.size(el)),  # type: ignore
                ) + expected_shape
            assert points.shape == expected_shape

            los = points - sensor_pos
            azimuth_out, elevation_out = compute_antenna_angles_a_posteriori(arf, los)

            assert np.max(np.abs(azimuth_out - az)) < self.tolerance["atol"]
            assert np.max(np.abs(elevation_out - el)) < self.tolerance["atol"]

    def test_direct_geocoding_with_pointing_invalid_inputs_0(self):
        """Testing direct_geocoding_with_pointing, invalid inputs, case 0"""
        with pytest.raises(ValueError):
            direct_geocoding_with_pointing(
                sensor_positions=self.sensor_positions[3, :],
                antenna_reference_frames=np.array([self.arf, self.arf]),
                azimuth_antenna_angles=-0.08726646259971647,
                elevation_antenna_angles=-0.05235987755982989,
            )

    def test_direct_geocoding_with_pointing_invalid_inputs_1(self):
        """Testing direct_geocoding_with_pointing, invalid inputs, case 1"""
        with pytest.raises(ValueError):
            direct_geocoding_with_pointing(
                sensor_positions=self.sensor_positions,
                antenna_reference_frames=self.arf,
                azimuth_antenna_angles=-0.08726646259971647,
                elevation_antenna_angles=-0.05235987755982989,
            )

    def test_direct_geocoding_with_pointing_invalid_inputs_2(self):
        """Testing direct_geocoding_with_pointing, invalid inputs, case 2"""
        with pytest.raises(ValueError):
            direct_geocoding_with_pointing(
                sensor_positions=self.sensor_positions,
                antenna_reference_frames=np.array([self.arf, self.arf]),
                azimuth_antenna_angles=-0.08726646259971647,
                elevation_antenna_angles=-0.05235987755982989,
            )

    def test_direct_geocoding_with_pointing_invalid_inputs_3(self):
        """Testing direct_geocoding_with_pointing, invalid inputs, case 3"""
        with pytest.raises(ValueError):
            direct_geocoding_with_pointing(
                sensor_positions=self.sensor_positions[3, :],
                antenna_reference_frames=self.arf,
                azimuth_antenna_angles=[-0.08726646259971647, -0.08726646259971647],
                elevation_antenna_angles=[-0.05235987755982989, -0.05235987755982989, -0.05235987755982989],
            )

    def test_direct_geocoding_with_pointing_invalid_inputs_4(self):
        """Testing direct_geocoding_with_pointing, invalid inputs, case 4"""
        with pytest.raises(ValueError):
            direct_geocoding_with_pointing(
                sensor_positions=self.sensor_positions[3:5, :],
                antenna_reference_frames=np.array([self.arf, self.arf]),
                azimuth_antenna_angles=[-0.08726646259971647, -0.08726646259971647],
                elevation_antenna_angles=[-0.05235987755982989, -0.05235987755982989, -0.05235987755982989],
            )
