# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for interferometric_analysis/config.py core functionalities"""

from __future__ import annotations

from dataclasses import asdict

import pytest

from perseo_quality.interferometric_analysis.config import InterferometricConfig


class TestInterferometricConfig:
    """Testing InterferometricConfig core functionalities"""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        # creating test data
        self._config = {
            "enable_coherence_computation": True,
            "coherence_kernel": 155,
            "azimuth_blocks_number": 16,
            "range_blocks_number": 47,
            "coherence_bins_number": 800,
        }

    def test_config_from_dict(self):
        """Testing InterferometricConfig generation from dictionary"""
        dtc = InterferometricConfig.from_dict(self._config)

        int_config_dict = asdict(dtc)
        assert self._config == int_config_dict
