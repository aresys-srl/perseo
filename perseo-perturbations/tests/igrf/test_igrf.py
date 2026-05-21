# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for main IGRF module internals (coefficient preparation, degree validation)."""

import unittest
from datetime import datetime

import numpy as np

from perseo_perturbations.geomagnetic.igrf import (
    _prepare_coefficients,
    _validate_degree_range,
    get_geocentric_igrf,
    get_geocentric_igrf_potential,
    get_geodetic_igrf,
)


class TestPrepareCoefficients(unittest.TestCase):
    """Tests for coefficient preparation."""

    def test_within_range(self):
        """Interpolation for a date within the coefficient window."""
        ds = _prepare_coefficients(datetime(2021, 3, 28))
        self.assertEqual(ds["time"].size, 1)
        self.assertEqual(ds["time"].data, np.datetime64("2021-03-28T00:00:00", "s"))

    def test_at_start_error(self):
        """Date before the window is not accepted."""
        with self.assertRaises(ValueError):
            _prepare_coefficients(datetime(1266, 1, 1))

    def test_at_end_error(self):
        """Date after the window is not accepted."""
        with self.assertRaises(ValueError):
            _prepare_coefficients(datetime(2030, 12, 31))


class TestValidateDegreeRange(unittest.TestCase):
    """Tests for degree range validation."""

    def test_valid_range(self):
        """Valid degree range."""
        self.assertIsNone(_validate_degree_range(1, 13))

    def test_single_degree(self):
        """Valid degree range, reduced."""
        self.assertIsNone(_validate_degree_range(4, 10))

    def test_invalid_range_0(self):
        """Invalid range, min_degree > max_degree."""
        with self.assertRaises(ValueError):
            _validate_degree_range(10, 1)

    def test_invalid_range_1(self):
        """Invalid range, min_degree < 1."""
        with self.assertRaises(ValueError):
            _validate_degree_range(0, 1)

    def test_invalid_range_2(self):
        """Invalid range, max_degree > 13."""
        with self.assertRaises(ValueError):
            _validate_degree_range(3, 15)


class TestMainEdgeCases(unittest.TestCase):
    """Edge case tests for main IGRF functions."""

    def setUp(self):
        self.date = datetime(2021, 3, 28)
        self.geocentric_coords = np.array([[6500, 30, 4], [6580, 42, 14]])
        self.geodetic_coords = np.array([5.32415, 60.39299, 400])
        self.expected_results_geocentric = np.array(
            [
                [-45790.36286896147, -14342.922054899242, 384.08821737097634],
                [-39722.99084854141, -19075.386682615364, 1488.053766363113],
            ]
        )
        self.expected_results_geodetic = np.array([221.23386545, 12899.04319758, -41287.69413524])
        self.expected_results_potential = np.array([-154840206.36518905, -129168570.50241716])
        self.tolerance = 1e-6

    def test_geocentric_limited_degree(self):
        """Geocentric IGRF with degree range limited to 1-5."""
        result = get_geocentric_igrf(self.geocentric_coords[0, :], self.date, min_degree=1, max_degree=5)
        self.assertEqual(result.shape, (3,))
        np.testing.assert_allclose(result, self.expected_results_geocentric[0, :], atol=self.tolerance, rtol=0)

    def test_geocentric_limited_degree_vect(self):
        """Geocentric IGRF with degree range limited to 1-5."""
        result = get_geocentric_igrf(self.geocentric_coords, self.date, min_degree=1, max_degree=5)
        self.assertEqual(result.shape, (2, 3))
        np.testing.assert_allclose(result, self.expected_results_geocentric, atol=self.tolerance, rtol=0)

    def test_geocentric_invalid_degree_range(self):
        """Geocentric IGRF recovers gracefully from inverted degree range."""
        with self.assertRaises(ValueError):
            get_geocentric_igrf(self.geocentric_coords, self.date, min_degree=10, max_degree=1)

    def test_geodetic_at_altitude(self):
        """Geodetic IGRF at 400 km altitude returns correct shape."""
        result = get_geodetic_igrf(self.geodetic_coords, self.date)
        self.assertEqual(result.shape, (3,))
        np.testing.assert_allclose(result, self.expected_results_geodetic, atol=self.tolerance, rtol=0)

    def test_geocentric_potential_vector(self):
        """Potential evaluates correctly for vectorized input."""
        result = get_geocentric_igrf_potential(self.geocentric_coords, self.date)
        self.assertEqual(result.shape, (2,))
        np.testing.assert_allclose(result, self.expected_results_potential, atol=self.tolerance, rtol=0)


class TestNonRegression(unittest.TestCase):
    """Unittest for the geomagnetic model."""

    def setUp(self):
        self.date = datetime(2021, 3, 28)
        self.geocentric_coords = np.array([[6500, 30, 4], [6580, 42, 14], [6435, 18, 23]])
        self.geodetic_coords = np.array([[20, 60, 0], [120, 60, 0], [220, 60, 0]])
        self.tolerance = 1e-6
        self.expected_results_geocentric = np.array(
            [
                [-46077.31133522183, -14227.126184994451, 233.14355743924327],
                [-39792.992194547245, -19077.438046587675, 1265.9346653807133],
                [-51770.89238760771, -9114.696003972911, 2044.5653987452565],
            ]
        )
        self.expected_results_geodetic = np.array(
            [
                [2004.3254442447123, 14816.085403077299, -49713.09015929676],
                [-3310.1596147336118, 13605.770105002599, -59076.958813084435],
                [4591.4299762142, 13982.403104411673, -53726.10039648457],
            ]
        )
        self.expected_results_potential = np.array([-154840206.36518905, -129168570.50241716, -174813973.6168166])

    def test_get_geocentric_igrf(self):
        """Match reference Br, B_theta, B_phi at a single point."""
        np.testing.assert_allclose(
            get_geocentric_igrf(self.geocentric_coords[0, :], self.date),
            self.expected_results_geocentric[0, :],
            atol=self.tolerance,
            rtol=0,
        )

    def test_get_geocentric_igrf_vect(self):
        """Match reference at three vectorized points."""
        np.testing.assert_allclose(
            get_geocentric_igrf(self.geocentric_coords, self.date),
            self.expected_results_geocentric,
            atol=self.tolerance,
            rtol=0,
        )

    def test_get_geodetic_igrf(self):
        """Match reference Be, Bn, Bu at a single point above WGS84."""
        np.testing.assert_allclose(
            get_geodetic_igrf(self.geodetic_coords[0, :], self.date),
            self.expected_results_geodetic[0, :],
            atol=self.tolerance,
            rtol=0,
        )

    def test_get_geodetic_igrf_vect(self):
        """Match reference at three vectorized geodetic points."""
        np.testing.assert_allclose(
            get_geodetic_igrf(self.geodetic_coords, self.date),
            self.expected_results_geodetic,
            atol=self.tolerance,
            rtol=0,
        )

    def test_get_geocentric_igrf_potential(self):
        """Match reference magnetic potential at a single point."""
        np.testing.assert_allclose(
            get_geocentric_igrf_potential(self.geocentric_coords[0, :], self.date),
            self.expected_results_potential[0],
            atol=self.tolerance,
            rtol=0,
        )

    def test_get_geocentric_igrf_potential_vect(self):
        """Match reference magnetic potential, vect"""
        np.testing.assert_allclose(
            get_geocentric_igrf_potential(self.geocentric_coords, self.date),
            self.expected_results_potential,
            atol=self.tolerance,
            rtol=0,
        )


if __name__ == "__main__":
    unittest.main()
