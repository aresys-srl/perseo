# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for tar_analysis/custom_dataclasses.py"""

from __future__ import annotations

from dataclasses import fields

import numpy as np

from perseo_quality.core.generic_dataclasses import SARPolarization
from perseo_quality.tar_analysis.custom_dataclasses import AmbiguityRatioOutput


class TestAmbiguityRatioOutput:
    """Testing AmbiguityRatioOutput dataclass"""

    def test_default_values(self):
        """Testing default values are None"""
        output = AmbiguityRatioOutput()
        for f in fields(output):
            assert getattr(output, f.name) is None

    def test_create_with_values(self):
        """Testing AmbiguityRatioOutput creation with field values"""
        output = AmbiguityRatioOutput(
            target_name="CR12",
            product_name="test_product",
            channel="1",
            swath="S1",
            polarization=SARPolarization.VV,
            burst=0,
            roi_size_azimuth=128,
            roi_size_range=128,
            target_nominal_coordinates=np.array([1.0, 2.0, 3.0]),
            target_azimuth_pixel=100.5,
            target_range_pixel=200.5,
            azimuth_time_delta=0.001,
            range_time_delta=0.002,
            left_ambiguity_azimuth_pixel=np.array([50.0]),
            left_ambiguity_range_pixel=np.array([150.0]),
            right_ambiguity_azimuth_pixel=np.array([200.0]),
            right_ambiguity_range_pixel=np.array([250.0]),
            ambiguity_ratio_db=-25.3,
            target_image=np.ones((10, 10)),
            right_ambiguity_image=np.ones((8, 8)),
            left_ambiguity_image=np.ones((8, 8)),
        )
        assert output.target_name == "CR12"
        assert output.product_name == "test_product"
        assert output.channel == "1"
        assert output.swath == "S1"
        assert output.polarization == SARPolarization.VV
        assert output.burst == 0
        assert output.roi_size_azimuth == 128
        assert output.roi_size_range == 128
        np.testing.assert_array_equal(output.target_nominal_coordinates, np.array([1.0, 2.0, 3.0]))
        assert output.target_azimuth_pixel == 100.5
        assert output.target_range_pixel == 200.5
        assert output.azimuth_time_delta == 0.001
        assert output.range_time_delta == 0.002
        np.testing.assert_array_equal(output.left_ambiguity_azimuth_pixel, np.array([50.0]))
        np.testing.assert_array_equal(output.left_ambiguity_range_pixel, np.array([150.0]))
        np.testing.assert_array_equal(output.right_ambiguity_azimuth_pixel, np.array([200.0]))
        np.testing.assert_array_equal(output.right_ambiguity_range_pixel, np.array([250.0]))
        assert output.ambiguity_ratio_db == -25.3
        np.testing.assert_array_equal(output.target_image, np.ones((10, 10)))
        np.testing.assert_array_equal(output.right_ambiguity_image, np.ones((8, 8)))
        np.testing.assert_array_equal(output.left_ambiguity_image, np.ones((8, 8)))
