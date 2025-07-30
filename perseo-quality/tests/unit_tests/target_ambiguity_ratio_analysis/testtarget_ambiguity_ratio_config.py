# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for tar_analysis/config.py core functionalities"""

import unittest
from dataclasses import fields

from perseo_quality.tar_analysis.config import AmbiguityRatioConfig


class AmbiguityRatioConfigConfigTest(unittest.TestCase):
    """Testing ambiguity ratio config dataclasses core functionalities"""

    def setUp(self) -> None:
        # creating test data

        self.params = {
            "interpolation_factor": 10,
            "cropping_size": [15, 15],
        }

    def test_ambiguity_ratio_config_from_dict(self):
        """Testing AmbiguityRatioConfig dataclass generation from dictionary"""
        dtc = AmbiguityRatioConfig.from_dict(self.params)

        for key, item in self.params.items():
            dataclass_key = [field.name for field in fields(dtc) if key in field.name][0]
            value = getattr(dtc, dataclass_key)
            if isinstance(value, tuple):
                self.assertEqual(item, list(value))
            else:
                self.assertEqual(item, value)


if __name__ == "__main__":
    unittest.main()
