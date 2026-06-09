# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for point_target_analysis/core/rcs.py core functionalities"""

from __future__ import annotations

import numpy as np
import pytest

from perseo_quality.core.generic_dataclasses import SARPolarization
from perseo_quality.point_targets_analysis.core.rcs import (
    _peak_extraction,
    _roi_extraction,
    compute_additional_rcs_values,
    compute_point_target_rcs,
)
from perseo_quality.point_targets_analysis.custom_dataclasses import RCSDataOutput


class TestPointTargetRCS:
    """Testing point_target_analysis/core/rcs.py core functionalities"""

    @pytest.fixture(autouse=True)
    def _setup(self, test_data_128, ref_points) -> None:
        # creating test data
        self.data, self.peak_pos, self.target_pos = test_data_128
        self.ref_points = ref_points
        # resolution
        self.rng_res = 1.1
        self.az_res = 1.2
        self.fc_hz = 5405000000

        # benchmarking values
        self.peak_corners = [427, 603, 419, 611]
        self.background_corners = [(10, 21, 10, 22), (106, 117, 10, 22), (10, 21, 105, 117), (106, 117, 105, 117)]
        self.clutter = -91.36377984672882
        self.rcs = 77.47622701869095
        self.scr = 91.36581122269361
        self.peak_value_complex = -0.24175901855507392 + 0.9705773683519141j

    def test_compute_rcs(self):
        """Testing RCS computation"""
        rcs, _, roi_background_corners, peak_corners = compute_point_target_rcs(
            target_area=self.data,
            rcs_roi=np.array([128, 128]),
            range_resolution_px=self.rng_res,
            azimuth_resolution_px=self.az_res,
            target_pos_real=self.peak_pos,
            rcs_interp_factor=8,
        )
        np.testing.assert_array_equal(self.peak_corners, peak_corners)
        np.testing.assert_array_equal(self.background_corners, roi_background_corners)
        np.testing.assert_allclose(rcs.peak_value_complex, self.peak_value_complex, atol=1e-9, rtol=0)
        np.testing.assert_allclose(rcs.clutter, self.clutter, atol=1e-9, rtol=0)
        np.testing.assert_allclose(rcs.scr, self.scr, atol=1e-9, rtol=0)

    def test_roi_extraction_with_target_pos(self) -> None:
        """Testing _roi_extraction with target position provided"""
        data = np.random.default_rng(42).random((20, 20))
        max_row, max_col, roi = _roi_extraction(data, roi=np.array([5, 5]), target_pos=np.array([10.0, 10.0]))
        assert roi.shape == (4, 4)
        assert max_row == 10
        assert max_col == 10

    def test_roi_extraction_without_target_pos(self) -> None:
        """Testing _roi_extraction without target position"""
        data = np.zeros((10, 10))
        data[5, 5] = 10.0
        max_row, max_col, roi = _roi_extraction(data, roi=np.array([5, 5]), target_pos=None)
        assert roi.shape == (4, 4)

    def test_roi_extraction_out_of_bounds(self) -> None:
        """Testing _roi_extraction raises error when ROI exceeds array"""
        data = np.ones((5, 5))
        from perseo_quality.point_targets_analysis.custom_errors import PointTargetComputationError

        with pytest.raises(PointTargetComputationError):
            _roi_extraction(data, roi=np.array([20, 20]), target_pos=np.array([2.0, 2.0]))

    def test_peak_extraction_with_target_position(self) -> None:
        """Testing _peak_extraction with target position"""
        data = np.random.default_rng(42).random((30, 30))
        row, col = _peak_extraction(
            data=data, target_position=np.array([15.0, 15.0]), max_indexes=(2, 2), interp_factor=8
        )
        assert isinstance(row, (int, np.integer))
        assert isinstance(col, (int, np.integer))

    def test_compute_additional_rcs_values_hh(self) -> None:
        """Testing compute_additional_rcs_values with HH polarization"""
        rcs_input = RCSDataOutput(
            peak_value_complex=0.5 + 0.5j,
            clutter=-20.0,
            rcs=100.0,
            scr=20.0,
        )
        rcs_lin, rcs_db, rcs_error, phase_error = compute_additional_rcs_values(
            rcs_input=rcs_input,
            step_distances=[10.0, 20.0],
            interp_factor=8,
            polarization=SARPolarization.HH,
            target_info=self.ref_points[0],
            sensor_position=np.array([4921229.04081908, -4051559.15884936, 216078.76707954]),
            fc_hz=self.fc_hz,
        )
        np.testing.assert_equal(rcs_lin, 312.5)
        np.testing.assert_equal(rcs_db, 10 * np.log10(rcs_lin))
        np.testing.assert_allclose(rcs_error, -25.05149978319906, atol=1e-8, rtol=0)
        np.testing.assert_equal(phase_error, 45.0)
