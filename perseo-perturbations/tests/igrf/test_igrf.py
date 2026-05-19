# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for main IGRF module internals (coefficient preparation, degree validation)."""

from datetime import datetime

import numpy as np
import pytest

from perseo_perturbations.geomagnetic.igrf import (
    _prepare_coefficients,
    _validate_degree_range,
    get_geocentric_igrf,
    get_geocentric_igrf_potential,
    get_geodetic_igrf,
)


class TestPrepareCoefficients:
    """Tests for coefficient preparation."""

    def test_within_range(self):
        """Interpolation for a date within the coefficient window."""
        ds = _prepare_coefficients(datetime(2021, 3, 28))
        assert ds["time"].size == 1
        assert ds["time"].data == np.datetime64("2021-03-28T00:00:00", "s")

    def test_at_start_error(self):
        """Date before the window is not accepted."""
        with pytest.raises(ValueError):
            _prepare_coefficients(datetime(1266, 1, 1))

    def test_at_end_error(self):
        """Date after the window is not accepted."""
        with pytest.raises(ValueError):
            _prepare_coefficients(datetime(2030, 12, 31))


class TestValidateDegreeRange:
    """Tests for degree range validation."""

    def test_valid_range(self):
        """Valid degree range."""
        assert _validate_degree_range(1, 13) is None

    def test_single_degree(self):
        """Valid degree range, reduced."""
        assert _validate_degree_range(4, 10) is None

    def test_invalid_range_0(self):
        """Invalid range, min_degree > max_degree."""
        with pytest.raises(ValueError):
            _validate_degree_range(10, 1)

    def test_invalid_range_1(self):
        """Invalid range, min_degree < 1."""
        with pytest.raises(ValueError):
            _validate_degree_range(0, 1)

    def test_invalid_range_2(self):
        """Invalid range, max_degree > 13."""
        with pytest.raises(ValueError):
            _validate_degree_range(3, 15)


class TestMainEdgeCases:
    """Edge case tests for main IGRF functions."""

    date = datetime(2021, 3, 28)
    geocentric_coords = np.array([[6500, 30, 4], [6580, 42, 14]])
    geodetic_coords = np.array([5.32415, 60.39299, 400])
    expected_results_geocentric = np.array(
        [
            [-45790.36286896147, -14342.922054899242, 384.08821737097634],
            [-39722.99084854141, -19075.386682615364, 1488.053766363113],
        ]
    )
    expected_results_geodetic = np.array([221.23386544703567, 12674.316738986869, -41357.23271228082])
    expected_results_potential = np.array([-154840206.36518905, -129168570.50241716])
    tolerance = 1e-6

    def test_geocentric_limited_degree(self):
        """Geocentric IGRF with degree range limited to 1-5."""
        result = get_geocentric_igrf(self.geocentric_coords[0, :], self.date, min_degree=1, max_degree=5)
        assert result.shape == (3,)
        np.testing.assert_allclose(result, self.expected_results_geocentric[0, :], atol=self.tolerance, rtol=0)

    def test_geocentric_limited_degree_vect(self):
        """Geocentric IGRF with degree range limited to 1-5."""
        result = get_geocentric_igrf(self.geocentric_coords, self.date, min_degree=1, max_degree=5)
        assert result.shape == (2, 3)
        np.testing.assert_allclose(result, self.expected_results_geocentric, atol=self.tolerance, rtol=0)

    def test_geocentric_invalid_degree_range(self):
        """Geocentric IGRF recovers gracefully from inverted degree range."""
        with pytest.raises(ValueError):
            get_geocentric_igrf(self.geocentric_coords, self.date, min_degree=10, max_degree=1)

    def test_geodetic_at_altitude(self):
        """Geodetic IGRF at 400 km altitude returns correct shape."""
        result = get_geodetic_igrf(self.geodetic_coords, self.date)
        assert result.shape == (3,)
        np.testing.assert_allclose(result, self.expected_results_geodetic, atol=self.tolerance, rtol=0)

    def test_geocentric_potential_vector(self):
        """Potential evaluates correctly for vectorized input."""
        result = get_geocentric_igrf_potential(self.geocentric_coords, self.date)
        assert result.shape == (2,)
        np.testing.assert_allclose(result, self.expected_results_potential, atol=self.tolerance, rtol=0)


class TestNonRegression:
    """Tests for the geomagnetic model."""

    date = datetime(2021, 3, 28)
    geocentric_coords = np.array([[6500, 30, 4], [6580, 42, 14], [6435, 18, 23]])
    geodetic_coords = np.array([[20, 60, 0], [120, 60, 0], [220, 60, 0]])
    tolerance = 1e-6
    expected_results_geocentric = np.array(
        [
            [-46077.31133522183, -14227.126184994451, 233.14355743924327],
            [-39792.992194547245, -19077.438046587675, 1265.9346653807133],
            [-51770.89238760771, -9114.696003972911, 2044.5653987452565],
        ]
    )
    expected_results_geodetic = np.array(
        [
            [2004.3254442447114, 14526.17041033998, -49798.5752112188],
            [-3310.1596147336195, 13261.3151626232, -59155.23276017638],
            [4591.429976214207, 13669.119639121885, -53806.65970564217],
        ]
    )
    expected_results_potential = np.array([-154840206.36518905, -129168570.50241716, -174813973.6168166])

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


class TestInputShapes:
    """Verify correct output shapes for (3,) and (N, 3) inputs."""

    date = datetime(2021, 3, 28)
    point = np.array([6500.0, 30.0, 4.0])
    points = np.array([[6500.0, 30.0, 4.0], [6580.0, 42.0, 14.0]])

    def test_geocentric_single_point(self):
        """(3,) input -> (3,) output."""
        result = get_geocentric_igrf(self.point, self.date)
        assert result.shape == (3,)

    def test_geocentric_multi_point(self):
        """(N, 3) input -> (N, 3) output."""
        result = get_geocentric_igrf(self.points, self.date)
        assert result.shape == (self.points.shape[0], 3)

    def test_geodetic_single_point(self):
        """(3,) input -> (3,) output."""
        result = get_geodetic_igrf(np.array([5.32415, 60.39299, 400.0]), self.date)
        assert result.shape == (3,)

    def test_geodetic_multi_point(self):
        """(N, 3) input -> (N, 3) output."""
        geodetic_coords = np.array([[5.32415, 60.39299, 400.0], [6.0, 61.0, 500.0]])
        result = get_geodetic_igrf(geodetic_coords, self.date)
        assert result.shape == (geodetic_coords.shape[0], 3)

    def test_potential_single_point(self):
        """(3,) input -> scalar (0-d)."""
        result = get_geocentric_igrf_potential(self.point, self.date)
        assert isinstance(result, float)

    def test_potential_multi_point(self):
        """(N, 3) input -> (N,) output."""
        result = get_geocentric_igrf_potential(self.points, self.date)
        assert result.shape == (self.points.shape[0],)
