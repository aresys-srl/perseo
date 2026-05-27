# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for point_target_analysis/core/localization_errors.py core functionalities"""

from __future__ import annotations

import numpy as np
import pytest

from perseo_quality.point_targets_analysis.core.localization_error import (
    compute_localization_errors_pixels,
)


class TestPointTargetRCS:
    """Testing point_target_analysis/core/rcs.py core functionalities"""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        # creating test data
        self.target_pos_real = np.array([10.2, -152.6])
        self.target_pos_nominal = np.array([11.7, -149.6])

        self.slant_range_error_px = -1.5
        self.azimuth_error_px = -3.0

    def test_compute_localization_errors(self):
        """Testing Localization Errors computation"""
        slant_range_error_px, azimuth_error_px, ground_range_error_px = compute_localization_errors_pixels(
            target_pos_real=self.target_pos_real, target_pos_ref=self.target_pos_nominal
        )
        np.testing.assert_allclose(self.slant_range_error_px, slant_range_error_px, atol=1e-9, rtol=0)
        np.testing.assert_allclose(self.azimuth_error_px, azimuth_error_px, atol=1e-9, rtol=0)
        np.testing.assert_allclose(self.slant_range_error_px, ground_range_error_px, atol=1e-9, rtol=0)
