# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for point_target_analysis/core/rcs.py core functionalities"""

from __future__ import annotations

import unittest

import numpy as np

from perseo_quality.point_targets_analysis.core.rcs import compute_point_target_rcs
from tests.unit_tests import test_utils


class PointTargetRCSTest(unittest.TestCase):
    """Testing point_target_analysis/core/rcs.py core functionalities"""

    def setUp(self) -> None:
        # creating test data
        self.data, self.peak_pos, self.target_pos = test_utils.generate_data_for_test(
            lines=test_utils.default_input_data_generation["lines"],
            samples=test_utils.default_input_data_generation["samples"],
            samples_start=test_utils.default_input_data_generation["samples_start"],
            lines_step=test_utils.default_input_data_generation["lines_step"],
            samples_step=test_utils.default_input_data_generation["samples_step"],
            fc_hz=test_utils.default_input_data_generation["fc_hz"],
        )
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
        self.rcs_res = test_utils.ref_data_rcs_results

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
        self.assertAlmostEqual(rcs.peak_value_complex, self.peak_value_complex)
        self.assertAlmostEqual(rcs.clutter, self.clutter)
        self.assertAlmostEqual(rcs.scr, self.scr)


if __name__ == "__main__":
    unittest.main()
