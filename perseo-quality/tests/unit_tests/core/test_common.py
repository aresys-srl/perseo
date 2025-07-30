# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for core/common.py functionalities"""

from __future__ import annotations

import unittest

import pandas as pd

from perseo_quality.core.common import check_targets_visibility
from tests.unit_tests.test_utils import REF_POINTS, MockProduct


class CheckTargetsVisibilityTest(unittest.TestCase):
    """Testing point_target_analysis/support.py check_targets_visibility function"""

    def setUp(self) -> None:
        # creating test data
        data_dict = {
            "id": [0, 1, 2, 0, 1, 2],
            "channel": [1, 1, 1, 2, 2, 2],
            "burst": [[0], None, [1], None, [0], None],
            "swath": ["S1", "S1", "S1", "S2", "S2", "S2"],
            "polarization": ["HH"] * 6,
        }
        self.reference_df = pd.DataFrame(data=data_dict)

    def test_check_targets_visibility(self):
        """Testing check_targets_visibility"""
        targets_visibility = check_targets_visibility(product=MockProduct(), points=REF_POINTS)
        targets_visibility["id"] = targets_visibility["id"].astype("int64")
        pd.testing.assert_frame_equal(targets_visibility, self.reference_df)


if __name__ == "__main__":
    unittest.main()
