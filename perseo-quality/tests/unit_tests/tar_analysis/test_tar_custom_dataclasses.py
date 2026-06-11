# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for tar_analysis/custom_dataclasses.py"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from perseo_quality.core.generic_dataclasses import SARPolarization
from perseo_quality.tar_analysis.custom_dataclasses import (
    AmbiguityRatioProductGeneralInfo,
    AmbiguityRatioROIInfo,
    AmbiguityRatioTargetInfo,
    DistributedTargetAmbiguityRatioDataOutput,
    PointTargetAmbiguityRatioDataOutput,
)


class TestPointTargetAmbiguityRatioDataOutput:
    """Testing PointTargetAmbiguityRatioDataOutput dataclass"""

    def test_default_values(self):
        """Testing default values are None"""
        output = PointTargetAmbiguityRatioDataOutput()
        assert output.general_info is None
        assert output.targets_info is None

    def test_create_with_values(self):
        """Testing PointTargetAmbiguityRatioDataOutput creation with field values"""
        general_info = AmbiguityRatioProductGeneralInfo(
            product="test_product",
            channel="1",
            swath="S1",
            polarization=SARPolarization.VV,
            sensor="test_sensor",
            product_type="test_type",
            acquisition_mode="test_mode",
            orbit_direction="Ascending",
            acquisition_start_time=datetime(2023, 1, 1),
        )
        target_info = AmbiguityRatioTargetInfo(
            target_name="CR12",
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
        output = PointTargetAmbiguityRatioDataOutput(
            general_info=general_info,
            targets_info=[target_info],
        )

        assert output.general_info.product == "test_product"
        assert output.general_info.channel == "1"
        assert output.general_info.swath == "S1"
        assert output.general_info.polarization == SARPolarization.VV
        assert output.general_info.sensor == "test_sensor"
        assert output.general_info.product_type == "test_type"
        assert output.general_info.acquisition_mode == "test_mode"
        assert output.general_info.orbit_direction == "Ascending"
        assert output.general_info.acquisition_start_time == datetime(2023, 1, 1)

        t = output.targets_info[0]
        assert t.target_name == "CR12"
        assert t.burst == 0
        assert t.roi_size_azimuth == 128
        assert t.roi_size_range == 128
        np.testing.assert_array_equal(t.target_nominal_coordinates, np.array([1.0, 2.0, 3.0]))
        assert t.target_azimuth_pixel == 100.5
        assert t.target_range_pixel == 200.5
        assert t.azimuth_time_delta == 0.001
        assert t.range_time_delta == 0.002
        np.testing.assert_array_equal(t.left_ambiguity_azimuth_pixel, np.array([50.0]))
        np.testing.assert_array_equal(t.left_ambiguity_range_pixel, np.array([150.0]))
        np.testing.assert_array_equal(t.right_ambiguity_azimuth_pixel, np.array([200.0]))
        np.testing.assert_array_equal(t.right_ambiguity_range_pixel, np.array([250.0]))
        assert t.ambiguity_ratio_db == -25.3
        np.testing.assert_array_equal(t.target_image, np.ones((10, 10)))
        np.testing.assert_array_equal(t.right_ambiguity_image, np.ones((8, 8)))
        np.testing.assert_array_equal(t.left_ambiguity_image, np.ones((8, 8)))


class TestDistributedTargetAmbiguityRatioDataOutput:
    """Testing DistributedTargetAmbiguityRatioDataOutput dataclass"""

    def test_default_values(self):
        """Testing default values are None"""
        output = DistributedTargetAmbiguityRatioDataOutput()
        assert output.general_info is None
        assert output.roi_info is None

    def test_create_with_values(self):
        """Testing DistributedTargetAmbiguityRatioDataOutput creation with field values"""
        general_info = AmbiguityRatioProductGeneralInfo(
            product="test_product",
            channel="1",
            swath="S1",
            polarization=SARPolarization.VV,
            sensor="test_sensor",
            product_type="test_type",
            acquisition_mode="test_mode",
            orbit_direction="Ascending",
            acquisition_start_time=datetime(2023, 1, 1),
        )
        roi_info = AmbiguityRatioROIInfo(
            roi_name="ROI_001",
            burst=0,
            roi_size_azimuth=128,
            roi_size_range=128,
            roi_center_ground_point_coordinates=np.array([1.0, 2.0, 3.0]),
            roi_center_azimuth_pixel=100.5,
            roi_center_range_pixel=200.5,
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
        output = DistributedTargetAmbiguityRatioDataOutput(
            general_info=general_info,
            roi_info=[roi_info],
        )

        assert output.general_info.product == "test_product"
        assert output.general_info.channel == "1"
        assert output.general_info.swath == "S1"
        assert output.general_info.polarization == SARPolarization.VV
        assert output.general_info.sensor == "test_sensor"
        assert output.general_info.product_type == "test_type"
        assert output.general_info.acquisition_mode == "test_mode"
        assert output.general_info.orbit_direction == "Ascending"
        assert output.general_info.acquisition_start_time == datetime(2023, 1, 1)

        r = output.roi_info[0]
        assert r.roi_name == "ROI_001"
        assert r.burst == 0
        assert r.roi_size_azimuth == 128
        assert r.roi_size_range == 128
        np.testing.assert_array_equal(r.roi_center_ground_point_coordinates, np.array([1.0, 2.0, 3.0]))
        assert r.roi_center_azimuth_pixel == 100.5
        assert r.roi_center_range_pixel == 200.5
        assert r.azimuth_time_delta == 0.001
        assert r.range_time_delta == 0.002
        np.testing.assert_array_equal(r.left_ambiguity_azimuth_pixel, np.array([50.0]))
        np.testing.assert_array_equal(r.left_ambiguity_range_pixel, np.array([150.0]))
        np.testing.assert_array_equal(r.right_ambiguity_azimuth_pixel, np.array([200.0]))
        np.testing.assert_array_equal(r.right_ambiguity_range_pixel, np.array([250.0]))
        assert r.ambiguity_ratio_db == -25.3
        np.testing.assert_array_equal(r.target_image, np.ones((10, 10)))
        np.testing.assert_array_equal(r.right_ambiguity_image, np.ones((8, 8)))
        np.testing.assert_array_equal(r.left_ambiguity_image, np.ones((8, 8)))
