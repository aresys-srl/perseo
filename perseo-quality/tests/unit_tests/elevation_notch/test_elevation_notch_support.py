# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for elevation_notch_analysis/support core functionalities"""

from __future__ import annotations

import pytest
import xarray as xr

from perseo_quality.elevation_notch_analysis.support import get_valid_antenna_pattern
from tests.unit_tests.test_utils import generate_antenna_pattern


class TestElevationNotchAnalysisSupport:
    """Testing Elevation Notch Analysis support functionalities"""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        """Testing setup"""
        self._antenna_pattern_data = generate_antenna_pattern()
        self._swath = "EN"
        self._polarization = "HH"
        self._antenna_pattern = {self._swath: {self._polarization: self._antenna_pattern_data}}

    def test_get_valid_antenna_pattern(self) -> None:
        """Getting valid antenna pattern"""
        valid_antenna_pattern = get_valid_antenna_pattern(
            antenna_pattern=self._antenna_pattern, swath=self._swath, polarization=self._polarization
        )
        xr.testing.assert_equal(valid_antenna_pattern, self._antenna_pattern_data)
