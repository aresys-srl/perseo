# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for point_target_analysis/core/irf.py core functionalities"""

from __future__ import annotations

import numpy as np
import pytest

from perseo_quality.core.generic_dataclasses import MaskingMethod
from perseo_quality.point_targets_analysis.core.irf import (
    compute_islr_2d,
    compute_point_target_irf_analysis,
    compute_pslr_2d,
    compute_sslr_2d,
)


class TestPointTargetIRF:
    """Testing point_target_analysis/core/irf.py core functionalities"""

    @pytest.fixture(autouse=True)
    def _setup(self, test_data_128) -> None:
        # creating test data
        self.data, self.peak_pos, self.target_pos = test_data_128
        # resolution
        self.rng_res = 1.1
        self.az_res = 1.2
        self.side_lobes = (1.0, 0.5)

        # benchmarking values
        self.pslr_ref = (-25.50375928748676, -25.57507201905654, -25.50375928748676)
        self.islr_ref = (-18.58654341614361, -19.459148122742103, -15.99066660365714)
        self.sslr_ref = (-39.91270389195101, -39.912703891950976, -39.912703891950976)
        self.pslr_lobes_ref = (-51.007518574973304, -52.167483288652576, -25.503759287486545)
        self.islr_lobes_ref = (-45.481171438194544, -48.54566569622393, -23.595405962888705)
        self.sslr_lobes_ref = (-79.825407783902, -76.92797946069359, -33.56414997151292)

    def test_compute_pslr(self):
        """Testing IRF PSLR computation"""
        pslr = compute_pslr_2d(data=self.data, resolution=[self.rng_res * 8, self.az_res * 8])
        np.testing.assert_allclose(pslr, self.pslr_ref, atol=1e-6, rtol=0)

    def test_compute_islr(self):
        """Testing IRF ISLR computation"""
        islr = compute_islr_2d(data=self.data, resolution=[self.rng_res * 8, self.az_res * 8])
        np.testing.assert_allclose(islr, self.islr_ref, atol=1e-6, rtol=0)

    def test_compute_sslr(self):
        """Testing IRF SSLR computation"""
        sslr = compute_sslr_2d(
            data=self.data,
            resolution=[self.rng_res * 8, self.az_res * 8],
            side_lobes_directions=(np.inf, 0.0),
        )
        np.testing.assert_allclose(sslr, self.sslr_ref, atol=1e-6, rtol=0)

    def test_compute_pslr_finite_lobe_direction(self) -> None:
        """Testing PSLR with finite lobe direction"""
        pslr = compute_pslr_2d(
            data=self.data, resolution=[self.rng_res * 8, self.az_res * 8], side_lobes_directions=self.side_lobes
        )
        np.testing.assert_allclose(pslr, self.pslr_lobes_ref, atol=1e-8, rtol=0)

    def test_compute_islr_finite_lobe_direction(self) -> None:
        """Testing ISLR with finite lobe direction"""
        islr = compute_islr_2d(
            data=self.data, resolution=[self.rng_res * 8, self.az_res * 8], side_lobes_directions=self.side_lobes
        )
        np.testing.assert_allclose(islr, self.islr_lobes_ref, atol=1e-8, rtol=0)

    def test_compute_sslr_finite_lobe_direction(self) -> None:
        """Testing SSLR with finite lobe direction"""
        sslr = compute_sslr_2d(
            data=self.data, resolution=[self.rng_res * 8, self.az_res * 8], side_lobes_directions=self.side_lobes
        )
        np.testing.assert_allclose(sslr, self.sslr_lobes_ref, atol=1e-8, rtol=0)

    def test_compute_point_target_irf_analysis_flags_off(self) -> None:
        """Testing compute_point_target_irf_analysis with all flags off"""
        result = compute_point_target_irf_analysis(
            recentered_target_area_interp=np.abs(self.data),
            range_resolution_px=self.rng_res,
            azimuth_resolution_px=self.az_res,
            pslr_flag=False,
            islr_flag=False,
            sslr_flag=False,
        )
        assert np.isnan(result.range_pslr)
        assert np.isnan(result.azimuth_pslr)
        assert np.isnan(result.pslr_2d)
        assert np.isnan(result.range_islr)
        assert np.isnan(result.azimuth_islr)
        assert np.isnan(result.islr_2d)
        assert np.isnan(result.range_sslr)
        assert np.isnan(result.azimuth_sslr)
        assert np.isnan(result.sslr_2d)

    def test_compute_point_target_irf_analysis_default(self) -> None:
        """Testing compute_point_target_irf_analysis with default flags"""
        results = compute_point_target_irf_analysis(
            recentered_target_area_interp=self.data,
            range_resolution_px=self.rng_res * 8,
            azimuth_resolution_px=self.az_res * 8,
            mask_method=MaskingMethod.RESOLUTION,
        )
        np.testing.assert_allclose(
            (results.range_pslr, results.azimuth_pslr, results.pslr_2d), self.pslr_ref, atol=1e-8, rtol=0
        )
        np.testing.assert_allclose(
            (results.range_islr, results.azimuth_islr, results.islr_2d), self.islr_ref, atol=1e-8, rtol=0
        )
        np.testing.assert_allclose(
            (results.range_sslr, results.azimuth_sslr, results.sslr_2d), self.sslr_ref, atol=1e-8, rtol=0
        )
