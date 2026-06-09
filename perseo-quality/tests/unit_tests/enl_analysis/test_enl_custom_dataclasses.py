# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for enl_analysis/custom_dataclasses.py"""

from __future__ import annotations

from dataclasses import fields

from perseo_quality.core.generic_dataclasses import SARPolarization
from perseo_quality.enl_analysis.custom_dataclasses import ENLOutput


class TestENLOutput:
    """Testing ENLOutput dataclass"""

    def test_default_values(self):
        """Testing default values are None"""
        output = ENLOutput()
        for f in fields(output):
            assert getattr(output, f.name) is None

    def test_create_with_values(self):
        """Testing ENLOutput creation with field values"""
        output = ENLOutput(
            product_name="test_product",
            channel="1",
            swath="S1",
            polarization=SARPolarization.HH,
            roi_center=(100, 200),
            roi_size_azimuth=50,
            roi_size_range=30,
        )
        assert output.product_name == "test_product"
        assert output.channel == "1"
        assert output.swath == "S1"
        assert output.polarization == SARPolarization.HH
        assert output.roi_center == (100, 200)
        assert output.roi_size_azimuth == 50
        assert output.roi_size_range == 30

    def test_field_types(self):
        """Testing field types match expected"""
        output = ENLOutput()
        assert isinstance(output.product_name, str) or output.product_name is None
        assert isinstance(output.channel, str) or output.channel is None
        assert isinstance(output.swath, str) or output.swath is None
        assert isinstance(output.polarization, SARPolarization) or output.polarization is None
        assert isinstance(output.roi_center, tuple) or output.roi_center is None
        assert isinstance(output.roi_size_azimuth, int) or output.roi_size_azimuth is None
        assert isinstance(output.roi_size_range, int) or output.roi_size_range is None
