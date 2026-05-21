# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for coordinate and field vector conversion functions."""

import unittest

import numpy as np

from perseo_perturbations.geomagnetic.core.conversions import (
    geocentric_to_geodetic,
    geodetic_to_geocentric,
    rotate_field_to_geocentric,
    rotate_field_to_geodetic,
)


class TestConversions(unittest.TestCase):
    """Tests for the coordinate conversion functions."""

    def test_geodetic_to_geocentric_equator(self):
        """Geodetic equator maps to 90° colatitude at equatorial radius."""
        theta, r = geodetic_to_geocentric(np.array([0.0]), np.array([0.0]))
        np.testing.assert_allclose(theta, 90.0, atol=1e-12)
        np.testing.assert_allclose(r, 6378.137, atol=1e-3)

    def test_geodetic_to_geocentric_north_pole(self):
        """Geodetic north pole maps to 0° colatitude at polar radius."""
        theta, r = geodetic_to_geocentric(np.array([90.0]), np.array([0.0]))
        np.testing.assert_allclose(theta, 0.0, atol=1e-12)
        np.testing.assert_allclose(r, 6356.752314245, atol=1e-3)

    def test_geodetic_to_geocentric_at_altitude(self):
        """Radius at 400 km altitude exceeds equatorial radius."""
        theta, r = geodetic_to_geocentric(np.array([45.0]), np.array([400.0]))
        self.assertGreater(r, 6378.137)

    def test_geocentric_to_geodetic_roundtrip(self):
        """Geodetic-to-geocentric round-trip preserves lat and height."""
        lat_in = np.array([45.0, 0.0, 90.0, -45.0])
        h_in = np.array([0.0, 0.0, 0.0, 100.0])
        theta, r = geodetic_to_geocentric(lat_in, h_in)
        lat_out, h_out = geocentric_to_geodetic(theta, r)
        np.testing.assert_allclose(lat_in, lat_out, atol=1e-10)
        np.testing.assert_allclose(h_in, h_out, atol=1e-6)

    def test_rotate_field_to_geocentric_identity_at_equator(self):
        """At equator (theta=90°), B_th = -Bn and B_r = Bu."""
        Bn = np.array([100.0])
        Bu = np.array([200.0])
        B_th, B_r = rotate_field_to_geocentric(Bn, Bu, np.array([0.0]), np.array([90.0]))
        np.testing.assert_allclose(B_th, -Bn, atol=1e-12)
        np.testing.assert_allclose(B_r, Bu, atol=1e-12)

    def test_rotate_field_to_geodetic_identity_at_equator(self):
        """At equator, inverse rotation recovers Bn = -B_th and Bu = B_r."""
        B_th = np.array([100.0])
        B_r = np.array([200.0])
        Bn, Bu = rotate_field_to_geodetic(B_th, B_r, np.array([0.0]), np.array([90.0]))
        np.testing.assert_allclose(Bn, -B_th, atol=1e-12)
        np.testing.assert_allclose(Bu, B_r, atol=1e-12)

    def test_field_rotation_roundtrip(self):
        """Forward and inverse field rotation round-trips at mid-latitudes."""
        Bn = np.array([100.0, -50.0])
        Bu = np.array([200.0, 75.0])
        lat = np.array([45.0, 60.0])
        theta = np.array([45.0, 30.0])
        B_th, B_r = rotate_field_to_geocentric(Bn, Bu, lat, theta)
        Bn_out, Bu_out = rotate_field_to_geodetic(B_th, B_r, lat, theta)
        np.testing.assert_allclose(Bn, Bn_out, atol=1e-10)
        np.testing.assert_allclose(Bu, Bu_out, atol=1e-10)

    def test_field_rotation_roundtrip_north_pole(self):
        """Field rotation round-trips correctly at the north pole."""
        Bn = np.array([100.0])
        Bu = np.array([200.0])
        lat = np.array([90.0])
        theta = np.array([0.0])
        B_th, B_r = rotate_field_to_geocentric(Bn, Bu, lat, theta)
        Bn_out, Bu_out = rotate_field_to_geodetic(B_th, B_r, lat, theta)
        np.testing.assert_allclose(Bn, Bn_out, atol=1e-10)
        np.testing.assert_allclose(Bu, Bu_out, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
