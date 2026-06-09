# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for block-wise radiometric_analysis core common functionalities"""

from __future__ import annotations

import numpy as np
import numpy.ma as ma

from perseo_quality.radiometric_analysis.block_wise.config import (
    RiverMaskingConfig,
    RiverMaskingMode,
)
from perseo_quality.radiometric_analysis.block_wise.core.common import (
    _disk_structuring_element,
    compute_local_statistics,
    compute_profile_variability_index,
    fast_river_masking,
    full_river_masking,
    morphological_cleaning,
    region_growing,
    threshold_mask,
)


class TestRadiometricCommon:
    """Testing Radiometric Analysis common functionalities"""

    def test_disk_structuring_element_radius_0(self) -> None:
        """Testing disk structuring element with radius 0"""
        result = _disk_structuring_element(0)
        assert result.shape == (1, 1)
        assert result[0, 0]

    def test_disk_structuring_element_radius_1(self) -> None:
        """Testing disk structuring element with radius 1"""
        result = _disk_structuring_element(1)
        assert result.shape == (3, 3)
        expected = np.array([[False, True, False], [True, True, True], [False, True, False]])
        np.testing.assert_array_equal(result, expected)

    def test_disk_structuring_element_radius_3(self) -> None:
        """Testing disk structuring element with radius 3"""
        result = _disk_structuring_element(3)
        assert result.shape == (7, 7)
        assert result[3, 3]
        assert result[0, 3]
        assert result[3, 0]
        assert not result[0, 0]

    def test_compute_local_statistics(self) -> None:
        """Testing local mean and CV computation"""
        data = np.full((4, 4), 5.0)
        local_mean, local_cv = compute_local_statistics(data, window=2)
        assert local_mean.shape == data.shape
        assert local_cv.shape == data.shape
        np.testing.assert_allclose(local_mean, 5.0)
        np.testing.assert_allclose(local_cv, 0.0)

    def test_compute_local_statistics_with_nan(self) -> None:
        """Testing local statistics with NaN values"""
        data = np.array([[1.0, np.nan], [3.0, 4.0]])
        local_mean, local_cv = compute_local_statistics(data, window=2)
        assert not np.any(np.isnan(local_mean))
        assert not np.any(np.isnan(local_cv))

    def test_threshold_mask_all_conditions(self) -> None:
        """Testing threshold_mask with various combinations"""
        local_mean = np.array([[1.0, 5.0], [5.0, 10.0]])
        local_cv = np.array([[0.1, 0.5], [1.5, 2.0]])
        result = threshold_mask(
            local_mean=local_mean,
            local_cv=local_cv,
            backscatter_thresh=3.0,
            cv_lower_thresh=0.2,
            cv_upper_thresh=1.0,
        )
        expected = np.array([[True, False], [False, False]])
        np.testing.assert_array_equal(result, expected)

    def test_threshold_mask_above_upper_cv(self) -> None:
        """Testing threshold_mask when CV exceeds upper threshold"""
        local_mean = np.array([[2.0]])
        local_cv = np.array([[1.5]])
        result = threshold_mask(
            local_mean=local_mean,
            local_cv=local_cv,
            backscatter_thresh=3.0,
            cv_lower_thresh=0.2,
            cv_upper_thresh=1.0,
        )
        assert result[0, 0]

    def test_morphological_cleaning_no_removal(self) -> None:
        """Testing morphological cleaning with large enough components"""
        mask = np.array([[0, 0, 0, 0, 0], [0, 1, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 0, 0]])
        result = morphological_cleaning(mask, opening_radius=1, min_area_px_percentile=50)
        assert not np.any(result)

    def test_morphological_cleaning_keep_large(self) -> None:
        """Testing morphological cleaning keeps large components"""
        mask = np.zeros((10, 10))
        mask[2:8, 2:8] = 1
        result = morphological_cleaning(mask, opening_radius=0, min_area_px_percentile=1)
        assert result.shape == mask.shape
        assert result.dtype == bool
        assert not np.all(result)

    def test_region_growing_simple(self) -> None:
        """Testing region growing expands seed into candidate pixels"""
        seed = np.zeros((5, 5), dtype=bool)
        seed[2, 2] = True
        candidates = np.zeros((5, 5), dtype=bool)
        candidates[1:4, 1:4] = True
        result = region_growing(seed, candidates, n_iterations=1)
        assert result[2, 2]
        assert result[1:4, 1:4].sum() > 1

    def test_region_growing_no_candidates(self) -> None:
        """Testing region growing with no eligible pixels"""
        seed = np.zeros((5, 5), dtype=bool)
        seed[2, 2] = True
        candidates = np.zeros((5, 5), dtype=bool)
        result = region_growing(seed, candidates, n_iterations=3)
        assert result.sum() == 1

    def test_compute_profile_variability_index(self) -> None:
        """Testing variability index computation"""
        look_angles = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        profile = ma.MaskedArray(data=[1.0, 2.0, 3.0, 4.0, 5.0], mask=[False, False, False, False, False])
        slope, variability = compute_profile_variability_index(profile, look_angles)
        assert isinstance(slope, float)
        np.testing.assert_allclose(slope, 1.0, atol=1e-9, rtol=0)
        assert isinstance(variability, float)
        np.testing.assert_allclose(variability, 0.0, atol=1e-12, rtol=0)

    def test_compute_profile_variability_index_masked(self) -> None:
        """Testing variability index with masked values"""
        look_angles = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        profile = ma.MaskedArray(data=[1.0, 2.0, 999.0, 4.0, 5.0], mask=[False, False, True, False, False])
        slope, variability = compute_profile_variability_index(profile, look_angles)
        assert isinstance(slope, float)
        np.testing.assert_allclose(slope, 1.0, atol=1e-9, rtol=0)
        assert isinstance(variability, float)
        np.testing.assert_allclose(variability, 0.0, atol=1e-12, rtol=0)

    def test_fast_river_masking_basic(self) -> None:
        """Testing fast river masking produces valid output"""
        rng = np.random.default_rng(42)
        data = rng.random((20, 20))
        config = RiverMaskingConfig(river_masking_mode=RiverMaskingMode.FAST)
        result = fast_river_masking(data.copy(), config=config)
        assert result.shape == data.shape
        assert result.dtype == float

    def test_full_river_masking_basic(self) -> None:
        """Testing full river masking produces valid output"""
        rng = np.random.default_rng(42)
        data = rng.random((20, 20))
        config = RiverMaskingConfig(river_masking_mode=RiverMaskingMode.FULL)
        result = full_river_masking(data.copy(), config=config)
        assert result.shape == data.shape
        assert result.dtype == float
