# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for elevation_notch_analysis/config.py core functionalities"""

import unittest
from dataclasses import fields

from perseo_quality.elevation_notch_analysis.config import ElevationNotchConfig


class ElevationNotchConfigConfigTest(unittest.TestCase):
    """Testing Elevation Notch config dataclasses core functionalities"""

    def setUp(self) -> None:
        # creating test data

        self.params = {
            "azimuth_block_size": 100,
            "range_pixel_margin": 10,
        }

    def test_elevation_notch_config_from_dict(self):
        """Testing ElevationNotchConfig dataclass generation from dictionary"""
        dtc = ElevationNotchConfig.from_dict(self.params)

        for key, item in self.params.items():
            dataclass_key = [field.name for field in fields(dtc) if key in field.name][0]
            value = getattr(dtc, dataclass_key)
            if isinstance(value, tuple):
                self.assertEqual(item, list(value))
            else:
                self.assertEqual(item, value)


if __name__ == "__main__":
    unittest.main()
