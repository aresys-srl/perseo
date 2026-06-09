# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for point_target_analysis/core/pre_processing.py"""

from __future__ import annotations

import numpy as np

from perseo_quality.core.generic_dataclasses import TargetDataType
from perseo_quality.point_targets_analysis.core.pre_processing import (
    _extract_profiles,
    _recenter_data,
    compute_data_resolution_pixel,
    compute_roi,
    detect_data_type,
    generate_irf_axis,
    target_area_interpolation,
)


class TestDetectDataType:
    """Testing detect_data_type"""

    def test_real_data(self) -> None:
        """Testing with real-valued data"""
        result = detect_data_type(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert result == TargetDataType.DETECTED

    def test_complex_data(self) -> None:
        """Testing with complex-valued data"""
        result = detect_data_type(np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]]))
        assert result == TargetDataType.COMPLEX


class TestComputeDataResolutionPixel:
    """Testing compute_data_resolution_pixel"""

    def test_complex_data(self) -> None:
        """Testing with complex data"""
        data = np.random.default_rng(42).random((10, 10)) + 1j * np.random.default_rng(42).random((10, 10))
        data[5, 5] = 10 + 10j
        rng_prof, az_prof, rng_res, az_res = compute_data_resolution_pixel(
            recentered_target_area_interp=np.abs(data),
            data_type=TargetDataType.COMPLEX,
            side_lobes_directions=(np.inf, 0.0),
        )
        assert isinstance(rng_res, float)
        assert isinstance(az_res, float)

    def test_detected_data(self) -> None:
        """Testing with detected data (real)"""
        data = np.random.default_rng(42).random((10, 10))
        data[5, 5] = 10.0
        rng_prof, az_prof, rng_res, az_res = compute_data_resolution_pixel(
            recentered_target_area_interp=data,
            data_type=TargetDataType.DETECTED,
            side_lobes_directions=(np.inf, 0.0),
        )
        assert isinstance(rng_res, float)
        assert isinstance(az_res, float)


class TestComputeROI:
    """Testing compute_roi"""

    def test_square_aspect(self) -> None:
        """Testing with square data"""
        roi = compute_roi((10, 10), oversampling_factor=4)
        np.testing.assert_array_equal(roi, np.array([8, 8]))

    def test_rectangular_aspect(self) -> None:
        """Testing with rectangular data"""
        roi = compute_roi((10, 20), oversampling_factor=5)
        np.testing.assert_array_equal(roi, np.array([10, 20]))


class TestGenerateIRFAxis:
    """Testing generate_irf_axis"""

    def test_basic(self) -> None:
        """Testing basic axis generation"""
        axis = generate_irf_axis(stop=10, offset=5, scaling=1)
        np.testing.assert_array_equal(axis, np.arange(10) - 5)

    def test_with_scaling(self) -> None:
        """Testing axis with scaling"""
        axis = generate_irf_axis(stop=6, offset=3, scaling=2)
        np.testing.assert_array_equal(axis, (np.arange(6) - 3) * 2)


class TestRecenterData:
    """Testing _recenter_data"""

    def test_no_shift(self) -> None:
        """Testing with no shift needed"""
        data = np.eye(5)
        shifted = _recenter_data(data, center=(2, 2))
        # center is (2,2), data is (5,5), so center pos should remain at (2,2)
        assert shifted.shape == data.shape

    def test_with_shift(self) -> None:
        """Testing with actual shift"""
        data = np.zeros((5, 5))
        data[0, 0] = 1.0
        shifted = _recenter_data(data, center=(2, 2))
        assert shifted.shape == data.shape


class TestExtractProfiles:
    """Testing _extract_profiles"""

    def test_infinite_direction(self) -> None:
        """Testing with infinite side lobe direction"""
        data = np.random.default_rng(42).random((10, 10))
        rng_prof, az_prof = _extract_profiles(data, side_lobes_directions=(np.inf, 0.0))
        assert len(rng_prof) == 10
        assert len(az_prof) == 10

    def test_finite_direction(self) -> None:
        """Testing with finite side lobe direction"""
        data = np.random.default_rng(42).random((20, 20))
        rng_prof, az_prof = _extract_profiles(data, side_lobes_directions=(0.5, 0.3))
        assert len(rng_prof) == 20
        assert len(az_prof) == 20


class TestTargetAreaInterpolation:
    """Testing target_area_interpolation"""

    def test_complex_data(self) -> None:
        """Testing with complex data"""
        data = np.random.default_rng(42).random((10, 10)) + 1j * np.random.default_rng(42).random((10, 10))
        result = target_area_interpolation(
            target_area=data,
            target_pos_real=(5.0, 5.0),
            oversampling_factor=4,
            roi=np.array([4, 4]),
        )
        expected_shape = (4 * 4, 4 * 4)
        assert result.shape == expected_shape

    def test_detected_data(self) -> None:
        """Testing with detected (real) data"""
        data = np.random.default_rng(42).random((10, 10))
        result = target_area_interpolation(
            target_area=data,
            target_pos_real=(5.0, 5.0),
            oversampling_factor=4,
            roi=np.array([4, 4]),
        )
        expected_shape = (4 * 4, 4 * 4)
        assert result.shape == expected_shape
