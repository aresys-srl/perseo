# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for elevation_notch_analysis/config.py core functionalities"""

from dataclasses import fields

import pytest

from perseo_quality.elevation_notch_analysis.config import ElevationNotchConfig


class TestElevationNotchConfig:
    """Testing Elevation Notch config dataclasses core functionalities"""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
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
                assert item == list(value)
            else:
                assert item == value
