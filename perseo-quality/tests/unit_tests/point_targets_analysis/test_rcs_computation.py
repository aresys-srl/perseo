# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for point_target_analysis/rcs_geometric_computation.py core functionalities"""

from __future__ import annotations

import numpy as np
import pytest

from perseo_quality.point_targets_analysis.rcs_geometric_computation import (
    _compute_enu_axes,
    compute_elevation_azimuth_wrt_enu,
    compute_rcs_trihedral_corner_reflector,
)


class TestRCSComputation:
    """Testing rcs computation functionalities"""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.cr_arm = 0.7
        self.wavelength = 0.055
        self.elevation_rad = np.array([0, np.pi / 4, np.pi / 3])
        self.azimuth_rad = np.array([0, np.pi / 4, np.pi / 3])

        # expected results
        self.rcs_expected_array = np.array([0.0, 286.0556891916635, 66.34793846512898])
        self.rcs_expected_invalid = np.array([0.0, np.nan, np.nan])

    def test_rcs_trihedral_scalar(self):
        rcs = compute_rcs_trihedral_corner_reflector(
            cr_arm_length_m=self.cr_arm,
            wavelength_m=self.wavelength,
            elevation_rad=self.elevation_rad[1],
            azimuth_rad=self.azimuth_rad[1],
        )
        assert isinstance(rcs, float)
        np.testing.assert_allclose(rcs, self.rcs_expected_array[1], atol=1e-8, rtol=0)

    def test_rcs_trihedral_array(self):
        rcs = compute_rcs_trihedral_corner_reflector(
            cr_arm_length_m=self.cr_arm,
            wavelength_m=self.wavelength,
            elevation_rad=self.elevation_rad,
            azimuth_rad=self.azimuth_rad,
        )
        assert isinstance(rcs, np.ndarray)
        assert rcs.size == self.elevation_rad.size
        assert rcs.shape == self.elevation_rad.shape
        np.testing.assert_allclose(rcs, self.rcs_expected_array, atol=1e-8, rtol=0)

    def test_rcs_trihedral_invalid(self):
        rcs = compute_rcs_trihedral_corner_reflector(
            cr_arm_length_m=self.cr_arm,
            wavelength_m=self.wavelength,
            elevation_rad=-self.elevation_rad,
            azimuth_rad=-self.azimuth_rad,
        )
        assert isinstance(rcs, np.ndarray)
        assert rcs.size == self.elevation_rad.size
        assert rcs.shape == self.elevation_rad.shape
        np.testing.assert_allclose(rcs, self.rcs_expected_invalid, atol=1e-8, rtol=0)

    def test_rcs_elevation_above_pi_half(self):
        """Testing elevation above pi/2 returns NaN"""
        rcs = compute_rcs_trihedral_corner_reflector(
            cr_arm_length_m=self.cr_arm,
            wavelength_m=self.wavelength,
            elevation_rad=np.pi / 2 + 0.01,
            azimuth_rad=np.pi / 4,
        )
        assert np.isnan(rcs)

    def test_rcs_azimuth_above_pi_half(self):
        """Testing azimuth above pi/2 returns NaN"""
        rcs = compute_rcs_trihedral_corner_reflector(
            cr_arm_length_m=self.cr_arm,
            wavelength_m=self.wavelength,
            elevation_rad=np.pi / 4,
            azimuth_rad=np.pi / 2 + 0.01,
        )
        assert np.isnan(rcs)

    def test_rcs_elevation_at_pi_half(self):
        """Testing elevation exactly at pi/2 is valid (boundary inclusive)"""
        rcs = compute_rcs_trihedral_corner_reflector(
            cr_arm_length_m=self.cr_arm,
            wavelength_m=self.wavelength,
            elevation_rad=np.pi / 2,
            azimuth_rad=np.pi / 4,
        )
        assert not np.isnan(rcs)

    def test_rcs_azimuth_at_zero(self):
        """Testing azimuth at zero is valid"""
        rcs = compute_rcs_trihedral_corner_reflector(
            cr_arm_length_m=self.cr_arm,
            wavelength_m=self.wavelength,
            elevation_rad=np.pi / 4,
            azimuth_rad=0.0,
        )
        np.testing.assert_equal(rcs, 0.0)

    def test_rcs_first_angular_dependency(self):
        """Testing branch where geometrical_condition is True"""
        rcs = compute_rcs_trihedral_corner_reflector(
            cr_arm_length_m=self.cr_arm,
            wavelength_m=self.wavelength,
            elevation_rad=0.1,
            azimuth_rad=0.1,
        )
        assert not np.isnan(rcs)
        np.testing.assert_allclose(rcs, 1.1097863035, atol=1e-8, rtol=0)

    def test_rcs_second_angular_dependency(self):
        """Testing branch where geometrical_condition is False"""
        rcs = compute_rcs_trihedral_corner_reflector(
            cr_arm_length_m=self.cr_arm,
            wavelength_m=self.wavelength,
            elevation_rad=0.1,
            azimuth_rad=0.75,
        )
        assert not np.isnan(rcs)
        np.testing.assert_allclose(rcs, 31.6616488335, atol=1e-8, rtol=0)

    def test_rcs_mixed_valid_invalid_angles(self):
        """Testing mix of valid and invalid angles in array"""
        elevation = np.array([0.1, -0.1, 0.5])
        azimuth = np.array([0.2, 0.3, 1.6])
        rcs = compute_rcs_trihedral_corner_reflector(
            cr_arm_length_m=self.cr_arm,
            wavelength_m=self.wavelength,
            elevation_rad=elevation,
            azimuth_rad=azimuth,
        )
        assert rcs.shape == (3,)
        assert not np.isnan(rcs[0])
        assert np.isnan(rcs[1])
        assert np.isnan(rcs[2])

    def test_rcs_small_angles(self):
        """Testing with very small angles"""
        rcs = compute_rcs_trihedral_corner_reflector(
            cr_arm_length_m=self.cr_arm,
            wavelength_m=self.wavelength,
            elevation_rad=1e-3,
            azimuth_rad=1e-3,
        )
        assert rcs > 0.0


class TestComputeENUAxes:
    """Testing _compute_enu_axes function"""

    def test_enu_axes_at_equator_prime_meridian(self):
        """Testing ENU axes at equator and prime meridian"""
        e, n, u = _compute_enu_axes(latitude_rad=0, longitude_rad=0)
        np.testing.assert_allclose(e, [0.0, 1.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(n, [0.0, 0.0, 1.0], atol=1e-10)
        np.testing.assert_allclose(u, [1.0, 0.0, 0.0], atol=1e-10)

    def test_enu_axes_at_north_pole(self):
        """Testing ENU axes at north pole"""
        e, n, u = _compute_enu_axes(latitude_rad=np.pi / 2, longitude_rad=0)
        np.testing.assert_allclose(e, [0.0, 1.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(n, [-1.0, 0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(u, [0.0, 0.0, 1.0], atol=1e-10)

    def test_enu_axes_shape(self):
        """Testing ENU axes have correct shape"""
        e, n, u = _compute_enu_axes(latitude_rad=0.5, longitude_rad=1.0)
        assert e.shape == (3,)
        assert n.shape == (3,)
        assert u.shape == (3,)

    def test_enu_axes_orthogonality(self):
        """Testing ENU axes are orthonormal"""
        e, n, u = _compute_enu_axes(latitude_rad=0.5, longitude_rad=1.0)
        assert abs(np.dot(e, n)) < 1e-10
        assert abs(np.dot(e, u)) < 1e-10
        assert abs(np.dot(n, u)) < 1e-10
        assert abs(np.linalg.norm(e) - 1.0) < 1e-10
        assert abs(np.linalg.norm(n) - 1.0) < 1e-10
        assert abs(np.linalg.norm(u) - 1.0) < 1e-10


class TestComputeElevationAzimuthWrtENU:
    """Testing compute_elevation_azimuth_wrt_enu function"""

    def test_elevation_azimuth_known_geometry(self):
        """Testing with known CR and satellite positions"""
        pos_cr = np.array([5000000.0, 0.0, 0.0])
        pos_sat = np.array([5000000.0, 0.0, 700000.0])
        elevation, azimuth = compute_elevation_azimuth_wrt_enu(pos_cr=pos_cr, pos_sat=pos_sat)
        assert isinstance(elevation, float)
        assert isinstance(azimuth, float)
        assert 0 <= elevation <= np.pi / 2
        assert -np.pi <= azimuth <= np.pi
