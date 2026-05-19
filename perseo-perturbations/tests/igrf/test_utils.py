# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for utility functions (Legendre polynomials, inclination/declination)."""

import numpy as np

from perseo_perturbations.geomagnetic.core.utils import get_legendre, get_magnetic_declination, get_magnetic_inclination


class TestGetLegendre:
    """Tests for the get_legendre function."""

    def test_shape(self):
        """Returned P and dP match input and key dimensions."""
        theta = np.array([45.0, 90.0])
        keys = [(1, 0), (1, 1)]
        P, dP = get_legendre(theta, keys)
        assert P.shape == (2, 2)
        assert dP.shape == (2, 2)

    def test_first_degree(self):
        """P(1,0)=0, P(1,1)=1 at the equator (theta=90°)."""
        theta = np.array([90.0])
        keys = [(1, 0), (1, 1)]
        P, dP = get_legendre(theta, keys)
        np.testing.assert_allclose(P[0, 0], 0.0, atol=1e-12)
        np.testing.assert_allclose(P[0, 1], 1.0, atol=1e-12)

    def test_degree_2(self):
        """Degree-2 Legendre functions have correct output shape."""
        theta = np.array([45.0])
        keys = [(2, 0), (2, 1), (2, 2)]
        P, dP = get_legendre(theta, keys)
        assert P.shape == (1, 3)

    def test_equator(self):
        """P(1,0)=0 at the equator for a single key."""
        theta = np.array([90.0])
        keys = [(1, 0)]
        P, dP = get_legendre(theta, keys)
        np.testing.assert_allclose(P[0, 0], 0.0, atol=1e-12)

    def test_multi_theta(self):
        """P(1,1) is 0 at poles, 1 at equator, 0 at opposite pole."""
        theta = np.array([0.0, 45.0, 90.0, 135.0, 180.0])
        keys = [(1, 1)]
        P, dP = get_legendre(theta, keys)
        assert P.shape == (5, 1)
        np.testing.assert_allclose(P[0, 0], 0.0, atol=1e-12)
        np.testing.assert_allclose(P[2, 0], 1.0, atol=1e-12)
        np.testing.assert_allclose(P[4, 0], 0.0, atol=1e-12)


class TestGetInclinationDeclination:
    """Tests for the get_inclination_declination function."""

    def test_horizontal(self):
        """Pure eastward field gives 0° inclination, 90° declination."""
        inc = get_magnetic_inclination(geodetic_magnetic_field=np.array([100.0, 0.0, 0.0]), degrees=True)
        dec = get_magnetic_declination(geodetic_magnetic_field=np.array([100.0, 0.0, 0.0]), degrees=True)
        np.testing.assert_allclose(inc, 0.0, atol=1e-10)
        np.testing.assert_allclose(dec, 90.0, atol=1e-10)

    def test_vertical_down(self):
        """Pure downward field gives -90° inclination, 0° declination."""
        inc = get_magnetic_inclination(geodetic_magnetic_field=np.array([0.0, 0.0, 100.0]), degrees=True)
        dec = get_magnetic_declination(geodetic_magnetic_field=np.array([0.0, 0.0, 100.0]), degrees=True)
        np.testing.assert_allclose(inc, -90.0, atol=1e-10)
        np.testing.assert_allclose(dec, 0.0, atol=1e-10)

    def test_vertical_up(self):
        """Pure upward field gives +90° inclination, 0° declination."""
        inc = get_magnetic_inclination(geodetic_magnetic_field=np.array([0.0, 0.0, -100.0]), degrees=True)
        dec = get_magnetic_declination(geodetic_magnetic_field=np.array([0.0, 0.0, -100.0]), degrees=True)
        np.testing.assert_allclose(inc, 90.0, atol=1e-10)
        np.testing.assert_allclose(dec, 0.0, atol=1e-10)

    def test_radians(self):
        """Results are in radians when degrees=False."""
        inc = get_magnetic_inclination(geodetic_magnetic_field=np.array([100.0, 0.0, 0.0]), degrees=False)
        dec = get_magnetic_declination(geodetic_magnetic_field=np.array([100.0, 0.0, 0.0]), degrees=False)
        np.testing.assert_allclose(inc, 0.0, atol=1e-10)
        np.testing.assert_allclose(dec, np.pi / 2, atol=1e-10)

    def test_north_pole(self):
        """Declination at north pole with non-zero Bn stays within bounds."""
        dec = get_magnetic_declination(geodetic_magnetic_field=np.array([0.0, 100.0, 50.0]), degrees=False)
        assert dec > -90.0
        assert dec < 90.0

    def test_vector(self):
        """Inclination and declination return correct shapes for vector input."""
        field = np.array([[100.0, 50.0, 30.0], [200.0, 0.0, 0.0]])
        inc = get_magnetic_inclination(geodetic_magnetic_field=field, degrees=False)
        dec = get_magnetic_declination(geodetic_magnetic_field=field, degrees=False)
        assert inc.shape == (2,)
        assert dec.shape == (2,)
