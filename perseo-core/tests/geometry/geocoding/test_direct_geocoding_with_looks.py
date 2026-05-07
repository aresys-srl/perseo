# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for geometry/geocoding/direct_geocoding.py geocoding with looks functionalities"""

from __future__ import annotations

import itertools
import unittest

import numpy as np

from perseo_core.geometry.geocoding.direct_geocoding import (
    direct_geocoding_with_look_angles,
    direct_geocoding_with_looking_direction,
    direct_geocoding_with_pointing,
)
from tests.common import compute_antenna_angles_a_posteriori


class DirectGeocodingWithLooksTest(unittest.TestCase):
    """Testing direct_geocoding_with_look_angles and direct_geocoding_with_looking_direction functionalities"""

    def setUp(self) -> None:
        self.sensor_position = np.array([5317606.94350283, 610603.985945038, 4577936.89859885])
        self.sensor_velocity = np.array([-4.579225059127757, -2.318410347817750, 5.612873789290531])
        self.looking_direction = np.array([5317606.0, 610603.0, 4577936.0])
        self.wrong_looking_direction = np.array([-4.617335568769981, -2.318493268811102, 5.581386037561345])
        # expected results
        self.expected_ground_points_for_look_angles = 1.0e6 * np.array(
            [
                [4.759710115562946, 0.723739860905043, 4.169511582485821],
                [4.740767822131609, 0.785940591895703, 4.179749794811581],
                [4.719765088454115, 0.852956178108469, 4.190295797874668],
                [4.695901593234693, 0.926811645389169, 4.201333096521494],
                [4.668001681937613, 1.010339507423213, 4.213072517214957],
                [4.634228769201612, 1.107766560515324, 4.225761582103892],
                [4.591484972384566, 1.225899798385040, 4.239684548934286],
            ]
        )
        self.tolerance = 1e-6

    def test_direct_geocoding_with_looking_direction(self) -> None:
        """Testing direct_geocoding_with_looking_direction, single position"""
        geodetic_altitude = 1000.0

        reference_point = 1.0e06 * np.array([4.809369353321205, 0.552244754877962, 4.140394297540082])

        position_inputs = [self.sensor_position, self.sensor_position.reshape((1, 3))]
        dir_inputs = [self.looking_direction, self.looking_direction.reshape((1, 3))]

        for origin, direction in itertools.product(position_inputs, dir_inputs):
            point = direct_geocoding_with_looking_direction(origin, direction, altitude=geodetic_altitude)

            self.assertEqual(point.shape, np.broadcast_shapes(origin.shape, direction.shape))
            np.testing.assert_allclose(point, reference_point.reshape(point.shape), atol=self.tolerance, rtol=0)

    def test_direct_geocoding_with_looking_direction_vectorized(self) -> None:
        """Testing direct_geocoding_with_looking_direction, n positions"""
        directions = np.vstack(
            (
                np.tile(self.looking_direction, (5, 1)),
                self.wrong_looking_direction,
            )
        )

        geodetic_altitude = 1000.0

        reference_point = 1.0e06 * np.array([4.809369353321205, 0.552244754877962, 4.140394297540082])

        points = direct_geocoding_with_looking_direction(
            self.sensor_position,
            directions,
            altitude=geodetic_altitude,
        )

        np.testing.assert_allclose(points[0:5], np.tile(reference_point, (5, 1)), atol=self.tolerance, rtol=0)
        self.assertTrue(all(np.isnan(points[5])))

        points = direct_geocoding_with_looking_direction(
            np.tile(self.sensor_position, (2, 1)),
            self.looking_direction,
            altitude=geodetic_altitude,
        )

        np.testing.assert_allclose(points, np.tile(reference_point, (2, 1)), atol=self.tolerance, rtol=0)

        points = direct_geocoding_with_looking_direction(
            np.tile(self.sensor_position, (2, 1)),
            np.tile(self.looking_direction, (2, 1)),
            altitude=geodetic_altitude,
        )

        np.testing.assert_allclose(points, np.tile(reference_point, (2, 1)), atol=self.tolerance, rtol=0)

    def test_direct_geocoding_with_looking_direction_empty_intersection(self) -> None:
        """Testing direct_geocoding_with_looking_direction, no intersection"""
        point = direct_geocoding_with_looking_direction(self.sensor_position, self.wrong_looking_direction)

        self.assertTrue(all(np.isnan(point)))

    def test_direct_geocoding_with_looking_direction_invalid_inputs(self) -> None:
        """Testing direct_geocoding_with_looking_direction, invalid inputs"""
        with self.assertRaises(ValueError):
            direct_geocoding_with_looking_direction(np.arange(5), np.arange(3))
        with self.assertRaises(ValueError):
            direct_geocoding_with_looking_direction(np.arange(6).reshape(3, 2), np.arange(3))
        with self.assertRaises(ValueError):
            direct_geocoding_with_looking_direction(np.arange(3), np.arange(5))
        with self.assertRaises(ValueError):
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

        np.testing.assert_allclose(points, self.expected_ground_points_for_look_angles, atol=self.tolerance, rtol=0)

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

        np.testing.assert_allclose(points, self.expected_ground_points_for_look_angles, atol=self.tolerance, rtol=0)

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

        np.testing.assert_allclose(point, self.expected_ground_points_for_look_angles[0], atol=self.tolerance, rtol=0)

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
            points, np.tile(self.expected_ground_points_for_look_angles[0], (10, 1)), atol=self.tolerance, rtol=0
        )

    def test_direct_geocoding_with_look_angles_invalid_inputs(self) -> None:
        """Testing direct_geocoding_with_look_angles with invalid inputs"""
        with self.assertRaises(ValueError):
            direct_geocoding_with_look_angles(
                np.arange(5, dtype=float),
                np.arange(3, dtype=float),
                "ZERODOPPLER",
                np.arange(5, dtype=float),
            )
        with self.assertRaises(ValueError):
            direct_geocoding_with_look_angles(
                np.arange(3, dtype=float),
                np.arange(4, dtype=float),
                "ZERODOPPLER",
                np.arange(5, dtype=float),
            )
        with self.assertRaises(AssertionError):
            direct_geocoding_with_look_angles(
                np.arange(3, dtype=float),
                np.arange(3, dtype=float),
                "ZERODOPPLER",
                np.arange(6, dtype=float).reshape(2, 3),
            )


class DirectGeocodingWithPointingTest(unittest.TestCase):
    """Testing direct_geocoding_with_pointing functionalities"""

    def setUp(self) -> None:
        self._sensor_positions = np.array(
            [
                [5317606.94350283, 610603.985945038, 4577936.89859885],
                [5313024.53547427, 608285.563877273, 4583547.15708167],
                [5308435.7651548, 605967.120830312, 4589152.18047604],
                [5303840.63790599, 603648.660435838, 4594751.96221552],
                [5299239.15894225, 601330.18624638, 4600346.49592944],
                [5294631.33350784, 599011.701824865, 4605935.7752263],
                [5290017.16682646, 596693.210719223, 4611519.79375494],
            ]
        )
        self._arf = np.array(
            [
                [-0.6045802221754875, -0.5648527276296573, -0.5616263446844129],
                [-0.304829433093796, 0.8154747325505556, -0.49201623681674933],
                [0.7359088066288877, -0.1262630455079271, -0.6652036317287437],
            ]
        )
        self._tolerance = 1e-8

        self._expected_results = np.array(
            [
                [4928985.040167449, 206139.93335815019, 4029123.540450799],
            ]
        )

    def test_direct_geocoding_with_pointing(self):
        """Testing direct_geocoding_with_pointing, single position"""
        points = direct_geocoding_with_pointing(
            sensor_positions=self._sensor_positions[3, :],
            antenna_reference_frames=self._arf,
            azimuth_antenna_angles=-0.08726646259971647,
            elevation_antenna_angles=-0.05235987755982989,
            altitude=0,
        )
        np.testing.assert_allclose(points, self._expected_results[0], atol=self._tolerance, rtol=0)

    def test_direct_geocoding_with_pointing_vectorized(self):
        """Testing direct_geocoding_with_pointing, vectorization"""
        sensor_pos_inputs = [self._sensor_positions[3, :], np.tile(self._sensor_positions[3, :], (10, 1))]
        az_angles_in = np.deg2rad(np.linspace(-5, 5, 10))
        el_angles_in = np.deg2rad(np.linspace(-3, 2, 10))

        azimuth_angles_inputs = [az_angles_in, az_angles_in[0]]
        elevation_angles_inputs = [el_angles_in, el_angles_in[4]]
        altitude_inputs = [0, 1000, -1000]

        for sensor_pos, az, el, height in itertools.product(
            sensor_pos_inputs, azimuth_angles_inputs, elevation_angles_inputs, altitude_inputs
        ):
            if sensor_pos.ndim == 1:
                arf = self._arf
            else:
                arf = np.tile(self._arf, (sensor_pos.shape[0], 1, 1))
            points = direct_geocoding_with_pointing(
                sensor_positions=sensor_pos,
                antenna_reference_frames=arf,
                azimuth_antenna_angles=az,
                elevation_antenna_angles=el,
                altitude=height,
            )
            expected_shape = (3,)
            if not (sensor_pos.size == 3 and isinstance(az, float) and isinstance(el, float)):
                expected_shape = (
                    max(sensor_pos.shape[0], np.size(az), np.size(el)),  # type: ignore
                ) + expected_shape
            self.assertEqual(points.shape, expected_shape)

            los = points - sensor_pos
            azimuth_out, elevation_out = compute_antenna_angles_a_posteriori(arf, los)

            self.assertLess(np.max(np.abs(azimuth_out - az)), 1e-8)
            self.assertLess(np.max(np.abs(elevation_out - el)), 1e-8)

    def test_direct_geocoding_with_pointing_invalid_inputs_0(self):
        """Testing direct_geocoding_with_pointing, invalid inputs, case 0"""
        with self.assertRaises(ValueError):
            direct_geocoding_with_pointing(
                sensor_positions=self._sensor_positions[3, :],
                antenna_reference_frames=np.array([self._arf, self._arf]),
                azimuth_antenna_angles=-0.08726646259971647,
                elevation_antenna_angles=-0.05235987755982989,
            )

    def test_direct_geocoding_with_pointing_invalid_inputs_1(self):
        """Testing direct_geocoding_with_pointing, invalid inputs, case 1"""
        with self.assertRaises(ValueError):
            direct_geocoding_with_pointing(
                sensor_positions=self._sensor_positions,
                antenna_reference_frames=self._arf,
                azimuth_antenna_angles=-0.08726646259971647,
                elevation_antenna_angles=-0.05235987755982989,
            )

    def test_direct_geocoding_with_pointing_invalid_inputs_2(self):
        """Testing direct_geocoding_with_pointing, invalid inputs, case 2"""
        with self.assertRaises(ValueError):
            direct_geocoding_with_pointing(
                sensor_positions=self._sensor_positions,
                antenna_reference_frames=np.array([self._arf, self._arf]),
                azimuth_antenna_angles=-0.08726646259971647,
                elevation_antenna_angles=-0.05235987755982989,
            )

    def test_direct_geocoding_with_pointing_invalid_inputs_3(self):
        """Testing direct_geocoding_with_pointing, invalid inputs, case 3"""
        with self.assertRaises(ValueError):
            direct_geocoding_with_pointing(
                sensor_positions=self._sensor_positions[3, :],
                antenna_reference_frames=self._arf,
                azimuth_antenna_angles=[-0.08726646259971647, -0.08726646259971647],
                elevation_antenna_angles=[-0.05235987755982989, -0.05235987755982989, -0.05235987755982989],
            )

    def test_direct_geocoding_with_pointing_invalid_inputs_4(self):
        """Testing direct_geocoding_with_pointing, invalid inputs, case 4"""
        with self.assertRaises(ValueError):
            direct_geocoding_with_pointing(
                sensor_positions=self._sensor_positions[3:5, :],
                antenna_reference_frames=np.array([self._arf, self._arf]),
                azimuth_antenna_angles=[-0.08726646259971647, -0.08726646259971647],
                elevation_antenna_angles=[-0.05235987755982989, -0.05235987755982989, -0.05235987755982989],
            )


if __name__ == "__main__":
    unittest.main()
