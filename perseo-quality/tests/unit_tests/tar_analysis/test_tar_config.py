# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for tar_analysis/config.py core functionalities"""

from __future__ import annotations

from dataclasses import fields

import pytest

from perseo_quality.tar_analysis.config import AmbiguityRatioConfig


class TestAmbiguityRatioConfig:
    """Testing ambiguity ratio config dataclasses core functionalities"""

    def test_default_values(self):
        """Testing default values"""
        config = AmbiguityRatioConfig()
        assert config.interpolation_factor == 8
        assert config.cropping_size == (128, 128)

    def test_ambiguity_ratio_config_from_dict(self):
        """Testing AmbiguityRatioConfig dataclass generation from dictionary"""
        params = {
            "interpolation_factor": 10,
            "cropping_size": [15, 15],
        }
        dtc = AmbiguityRatioConfig.from_dict(params)

        for key, item in params.items():
            dataclass_key = [field.name for field in fields(dtc) if key in field.name][0]
            value = getattr(dtc, dataclass_key)
            if isinstance(value, tuple):
                assert item == list(value)
            else:
                assert item == value

    def test_from_dict_non_dict_input_raises_error(self):
        """Testing from_dict with non-dict input raises AttributeError (before try/except)"""
        with pytest.raises(AttributeError):
            AmbiguityRatioConfig.from_dict("not a dict")

    def test_cropping_size_converted_to_tuple(self):
        """Testing that list cropping_size is converted to tuple"""
        dtc = AmbiguityRatioConfig.from_dict({"cropping_size": [64, 64]})
        assert isinstance(dtc.cropping_size, tuple)
        assert dtc.cropping_size == (64, 64)
