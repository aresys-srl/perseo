# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for tar_analysis/config.py core functionalities"""

from dataclasses import fields

import pytest

from perseo_quality.tar_analysis.config import AmbiguityRatioConfig


class TestAmbiguityRatioConfig:
    """Testing ambiguity ratio config dataclasses core functionalities"""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
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
                assert item == list(value)
            else:
                assert item == value
