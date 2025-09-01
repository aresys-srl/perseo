# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for point_target_analysis/core/localization_errors.py core functionalities"""

from __future__ import annotations

import unittest

import numpy as np

from perseo_quality.point_targets_analysis.core.localization_error import (
    compute_localization_errors_pixels,
)


class PointTargetRCSTest(unittest.TestCase):
    """Testing point_target_analysis/core/rcs.py core functionalities"""

    def setUp(self) -> None:
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
        self.assertAlmostEqual(self.slant_range_error_px, slant_range_error_px)
        self.assertAlmostEqual(self.azimuth_error_px, azimuth_error_px)
        self.assertAlmostEqual(self.slant_range_error_px, ground_range_error_px)


if __name__ == "__main__":
    unittest.main()
