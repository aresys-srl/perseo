# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for tar_analysis/analysis functionalities"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from perseo_core.timing import PreciseDateTime

from perseo_quality.core.generic_dataclasses import SARRadiometricQuantity
from perseo_quality.io.point_targets import PointTarget
from perseo_quality.tar_analysis.analysis import (
    distributed_target_ambiguity_ratio_analysis,
    point_target_ambiguity_ratio_analysis,
)
from perseo_quality.tar_analysis.config import AmbiguityRatioConfig
from perseo_quality.tar_analysis.custom_dataclasses import (
    DistributedTargetAmbiguityRatioDataOutput,
    PointTargetAmbiguityRatioDataOutput,
)


class TestPointTargetAmbiguityRatioAnalysis:
    """Testing PTAR analysis main function"""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        """Testing setup"""
        self._cropping_size = (128, 128)
        self._rng = np.random.default_rng(42)
        self._point_targets = [
            PointTarget(name="T1", xyz_coordinates=np.array([-4989394.044, 2746844.389, -2862070.09])),
            PointTarget(name="T2", xyz_coordinates=np.array([-4979009.54, 2766786.057, -2860862.575])),
        ]

    def _make_mock_channel_data(self, mocker, swath_name: str = "S1", polarization: str = "HH"):
        channel_data = mocker.MagicMock()
        channel_data.swath_name = swath_name
        channel_data.polarization = mocker.MagicMock()
        channel_data.polarization.name = polarization
        channel_data.prf = 1.71713e03
        channel_data.mid_azimuth_time = mocker.MagicMock()
        channel_data.trajectory = mocker.MagicMock()
        channel_data.doppler_rate = mocker.MagicMock()
        channel_data.doppler_rate.evaluate.return_value = -2.3281e03
        channel_data.lines_per_burst = np.array([200])
        channel_data.azimuth_axis = np.arange(200) + PreciseDateTime.from_numeric_datetime(2019, 1, 8, 8, 32, 40)
        channel_data.range_axis = np.arange(500) * 1e-5 + 5.365309240726020e-03
        channel_data.radiometric_quantity = SARRadiometricQuantity.BETA_NOUGHT
        channel_data.read_data.return_value = self._rng.random((128, 128))
        channel_data.times_to_pixel_conversion.return_value = (100.5, 200.5)
        channel_data.pixel_to_times_conversion.return_value = (mocker.MagicMock(), 0.001)
        channel_data.looking_side = mocker.MagicMock()
        channel_data.looking_side.value = "RIGHT"
        channel_data.acquisition_mode = mocker.MagicMock()
        channel_data.acquisition_mode.name = "TEST_MODE"
        channel_data.orbit_direction = mocker.MagicMock()
        channel_data.orbit_direction.name = "ASCENDING"
        channel_data.image_type = mocker.MagicMock()
        channel_data.image_type.name = "SLC"
        channel_data.sensor_name = "TestSensor"
        return channel_data

    def _make_mock_product(self, mocker, n_channels: int = 1):
        product = mocker.MagicMock()
        product.name = "TestProduct"
        product.channels_list = list(range(1, n_channels + 1))
        product.get_channel_data.side_effect = lambda channel_id: self._make_mock_channel_data(mocker, f"S{channel_id}")
        return product

    @staticmethod
    def _make_visible_targets_df(
        target_ids: list[str] | None = None,
        channel: int = 1,
        burst_list: list[list[int] | None] | None = None,
    ) -> pd.DataFrame:
        if target_ids is None:
            target_ids = ["T1"]
        if burst_list is None:
            burst_list = [[0]]
        return pd.DataFrame(
            {
                "id": target_ids,
                "channel": [channel] * len(target_ids),
                "burst": burst_list,
                "swath": ["S1"] * len(target_ids),
                "polarization": ["HH"] * len(target_ids),
            }
        )

    def _setup_ptar_mocks(self, mocker, visible_targets_df=None, amb_inside_scene=True, amb_ratio_db=-25.0):
        """Set up common PTAR mocks with happy path defaults"""
        if visible_targets_df is None:
            visible_targets_df = self._make_visible_targets_df()
        mocker.patch(
            "perseo_quality.tar_analysis.analysis.check_targets_visibility",
            return_value=visible_targets_df,
        )
        mocker.patch(
            "perseo_quality.tar_analysis.analysis.inverse_geocoding_monostatic",
            return_value=(mocker.MagicMock(), 0.001),
        )
        mocker.patch(
            "perseo_quality.tar_analysis.analysis.compute_ambiguities_locations",
            return_value=((50, 100), (150, 300), 0.5, 0.001),
        )
        mocker.patch(
            "perseo_quality.tar_analysis.analysis.are_ambiguities_inside_scene",
            return_value=amb_inside_scene,
        )
        mocker.patch(
            "perseo_quality.tar_analysis.analysis.ambiguity_ratio_computation_core",
            return_value=(
                np.ones((128, 128)),
                np.ones((128, 128)),
                np.ones((128, 128)),
                amb_ratio_db,
            ),
        )

    def test_ptar_default_config(self, mocker) -> None:
        """Testing with config=None uses default AmbiguityRatioConfig"""
        self._setup_ptar_mocks(mocker)
        product = self._make_mock_product(mocker)

        result = point_target_ambiguity_ratio_analysis(
            product=product, point_targets=self._point_targets[:1], config=None
        )

        assert len(result) == 1
        output = result[0]
        assert isinstance(output, PointTargetAmbiguityRatioDataOutput)
        assert output.general_info.product == "TestProduct"
        assert output.general_info.channel == "1"
        assert output.general_info.swath == "S1"
        assert output.general_info.polarization == "HH"
        t = output.targets_info[0]
        assert t.target_name == "T1"
        assert t.burst == 0
        assert t.roi_size_azimuth == 128
        assert t.roi_size_range == 128
        assert t.ambiguity_ratio_db == -25.0
        assert t.target_azimuth_pixel == 100.5
        assert t.target_range_pixel == 200.5
        assert np.array_equal(t.target_nominal_coordinates, self._point_targets[0].xyz_coordinates)

    def test_ptar_single_target_single_burst(self, mocker) -> None:
        """Testing happy path with explicit config and custom cropping size"""
        self._setup_ptar_mocks(mocker)
        config = AmbiguityRatioConfig(interpolation_factor=16, cropping_size=(64, 64))
        product = self._make_mock_product(mocker)
        channel_data = product.get_channel_data(channel_id=1)
        channel_data.read_data.return_value = self._rng.random((64, 64))

        result = point_target_ambiguity_ratio_analysis(
            product=product, point_targets=self._point_targets[:1], config=config
        )

        assert len(result) == 1
        t = result[0].targets_info[0]
        assert t.roi_size_azimuth == 64
        assert t.roi_size_range == 64
        assert t.ambiguity_ratio_db == -25.0

    def test_ptar_multiple_targets(self, mocker) -> None:
        """Testing with multiple point targets"""
        df = self._make_visible_targets_df(target_ids=["T1", "T2"], burst_list=[[0], [0]])
        self._setup_ptar_mocks(mocker, visible_targets_df=df)
        product = self._make_mock_product(mocker)

        result = point_target_ambiguity_ratio_analysis(product=product, point_targets=self._point_targets, config=None)

        assert len(result) == 1
        assert result[0].targets_info[0].target_name == "T1"
        assert result[0].targets_info[1].target_name == "T2"

    def test_ptar_target_not_visible(self, mocker) -> None:
        """Testing when target is not visible in the scene"""
        df = self._make_visible_targets_df(burst_list=[None])
        mocker.patch(
            "perseo_quality.tar_analysis.analysis.check_targets_visibility",
            return_value=df,
        )
        product = self._make_mock_product(mocker)

        result = point_target_ambiguity_ratio_analysis(
            product=product, point_targets=self._point_targets[:1], config=None
        )

        assert result == []

    def test_ptar_inverse_geocoding_fails(self, mocker) -> None:
        """Testing when inverse geocoding raises an exception"""
        df = self._make_visible_targets_df()
        mocker.patch(
            "perseo_quality.tar_analysis.analysis.check_targets_visibility",
            return_value=df,
        )
        mocker.patch(
            "perseo_quality.tar_analysis.analysis.inverse_geocoding_monostatic",
            side_effect=Exception("Geocoding failed"),
        )
        product = self._make_mock_product(mocker)

        result = point_target_ambiguity_ratio_analysis(
            product=product, point_targets=self._point_targets[:1], config=None
        )

        assert len(result) == 1
        assert result[0].targets_info[0] is None

    def test_ptar_doppler_rate_none(self, mocker) -> None:
        """Testing when channel_data.doppler_rate is None"""
        df = self._make_visible_targets_df()
        mocker.patch(
            "perseo_quality.tar_analysis.analysis.check_targets_visibility",
            return_value=df,
        )
        mocker.patch(
            "perseo_quality.tar_analysis.analysis.inverse_geocoding_monostatic",
            return_value=(mocker.MagicMock(), 0.001),
        )

        channel_data = self._make_mock_channel_data(mocker)
        channel_data.doppler_rate = None
        product = mocker.MagicMock()
        product.name = "TestProduct"
        product.channels_list = [1]
        product.get_channel_data.return_value = channel_data

        result = point_target_ambiguity_ratio_analysis(
            product=product, point_targets=self._point_targets[:1], config=None
        )

        assert len(result) == 1
        assert result[0].targets_info[0] is None

    def test_ptar_prf_none(self, mocker) -> None:
        """Testing when channel_data.prf is None"""
        df = self._make_visible_targets_df()
        mocker.patch(
            "perseo_quality.tar_analysis.analysis.check_targets_visibility",
            return_value=df,
        )
        mocker.patch(
            "perseo_quality.tar_analysis.analysis.inverse_geocoding_monostatic",
            return_value=(mocker.MagicMock(), 0.001),
        )

        channel_data = self._make_mock_channel_data(mocker)
        channel_data.prf = None
        product = mocker.MagicMock()
        product.name = "TestProduct"
        product.channels_list = [1]
        product.get_channel_data.return_value = channel_data

        result = point_target_ambiguity_ratio_analysis(
            product=product, point_targets=self._point_targets[:1], config=None
        )

        assert len(result) == 1
        assert result[0].targets_info[0] is None

    def test_ptar_ambiguities_out_of_scene(self, mocker) -> None:
        """Testing when ambiguities are outside the scene boundaries"""
        self._setup_ptar_mocks(mocker, amb_inside_scene=False)
        product = self._make_mock_product(mocker)

        result = point_target_ambiguity_ratio_analysis(
            product=product, point_targets=self._point_targets[:1], config=None
        )

        assert len(result) == 1
        assert result[0].targets_info[0] is None

    def test_ptar_computation_fails(self, mocker) -> None:
        """Testing when ambiguity_ratio_computation_core raises an exception"""
        df = self._make_visible_targets_df()
        mocker.patch(
            "perseo_quality.tar_analysis.analysis.check_targets_visibility",
            return_value=df,
        )
        mocker.patch(
            "perseo_quality.tar_analysis.analysis.inverse_geocoding_monostatic",
            return_value=(mocker.MagicMock(), 0.001),
        )
        mocker.patch(
            "perseo_quality.tar_analysis.analysis.compute_ambiguities_locations",
            return_value=((50, 100), (150, 300), 0.5, 0.001),
        )
        mocker.patch(
            "perseo_quality.tar_analysis.analysis.are_ambiguities_inside_scene",
            return_value=True,
        )
        mocker.patch(
            "perseo_quality.tar_analysis.analysis.ambiguity_ratio_computation_core",
            side_effect=Exception("Computation failed"),
        )
        product = self._make_mock_product(mocker)

        result = point_target_ambiguity_ratio_analysis(
            product=product, point_targets=self._point_targets[:1], config=None
        )

        assert len(result) == 1
        assert result[0].targets_info[0] is None

    def test_ptar_with_delay(self, mocker) -> None:
        """Testing with a point target that has a delay set"""
        self._setup_ptar_mocks(mocker)
        product = self._make_mock_product(mocker)

        pt_with_delay = PointTarget(
            name="T1", xyz_coordinates=np.array([-4989394.044, 2746844.389, -2862070.09]), delay=0.001
        )

        result = point_target_ambiguity_ratio_analysis(product=product, point_targets=[pt_with_delay], config=None)

        assert len(result) == 1
        t = result[0].targets_info[0]
        assert t.target_name == "T1"
        assert t.ambiguity_ratio_db == -25.0


class TestDistributedTargetAmbiguityRatioAnalysis:
    """Testing DTAR analysis main function"""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        """Testing setup"""
        self._cropping_size = (128, 128)
        self._roi_centers = [(20, 20), (50, 50)]
        self._rng = np.random.default_rng(42)

    def _make_mock_channel_data(self, mocker, swath_name: str = "S1", polarization: str = "HH"):
        channel_data = mocker.MagicMock()
        channel_data.swath_name = swath_name
        channel_data.polarization = mocker.MagicMock()
        channel_data.polarization.name = polarization
        channel_data.prf = 1.71713e03
        channel_data.trajectory = mocker.MagicMock()
        channel_data.doppler_rate = mocker.MagicMock()
        channel_data.doppler_rate.evaluate.return_value = -2.3281e03
        channel_data.lines_per_burst = np.array([200])
        channel_data.azimuth_axis = np.arange(200) + PreciseDateTime.from_numeric_datetime(2019, 1, 8, 8, 32, 40)
        channel_data.range_axis = np.arange(500) * 1e-5 + 5.365309240726020e-03
        channel_data.radiometric_quantity = SARRadiometricQuantity.BETA_NOUGHT
        channel_data.read_data.return_value = self._rng.random((128, 128))
        channel_data.pixel_to_times_conversion.return_value = (mocker.MagicMock(), 0.001)
        channel_data.looking_side = mocker.MagicMock()
        channel_data.looking_side.value = "RIGHT"
        channel_data.acquisition_mode = mocker.MagicMock()
        channel_data.acquisition_mode.name = "TEST_MODE"
        channel_data.orbit_direction = mocker.MagicMock()
        channel_data.orbit_direction.name = "ASCENDING"
        channel_data.image_type = mocker.MagicMock()
        channel_data.image_type.name = "SLC"
        channel_data.sensor_name = "TestSensor"
        return channel_data

    def _make_mock_product(self, mocker, n_channels: int = 1):
        product = mocker.MagicMock()
        product.name = "TestProduct"
        product.channels_list = list(range(1, n_channels + 1))
        product.get_channel_data.side_effect = lambda channel_id: self._make_mock_channel_data(
            mocker, f"S{channel_id}", "HH" if channel_id == 1 else "VV"
        )
        return product

    def _setup_dtar_mocks(self, mocker, amb_inside_scene=True, amb_ratio_db=-25.0):
        """Set up common DTAR mocks with happy path defaults"""
        mocker.patch(
            "perseo_quality.tar_analysis.analysis.direct_geocoding_monostatic",
            return_value=np.array([-4989397.154, 2746837.255, -2862071.786]),
        )
        mocker.patch(
            "perseo_quality.tar_analysis.analysis.detect_burst_from_pixel",
            return_value=0,
        )
        mocker.patch(
            "perseo_quality.tar_analysis.analysis.compute_ambiguities_locations",
            return_value=((50, 100), (150, 300), 0.5, 0.001),
        )
        mocker.patch(
            "perseo_quality.tar_analysis.analysis.are_ambiguities_inside_scene",
            return_value=amb_inside_scene,
        )
        mocker.patch(
            "perseo_quality.tar_analysis.analysis.ambiguity_ratio_computation_core",
            return_value=(
                np.ones((128, 128)),
                np.ones((128, 128)),
                np.ones((128, 128)),
                amb_ratio_db,
            ),
        )

    def test_dtar_default_config(self, mocker) -> None:
        """Testing with config=None uses default AmbiguityRatioConfig"""
        self._setup_dtar_mocks(mocker)
        product = self._make_mock_product(mocker)

        result = distributed_target_ambiguity_ratio_analysis(
            product=product, roi_centers=self._roi_centers[:1], config=None
        )

        assert len(result) == 1
        output = result[0]
        assert isinstance(output, DistributedTargetAmbiguityRatioDataOutput)
        assert output.general_info.product == "TestProduct"
        assert output.general_info.channel == "1"
        assert output.general_info.swath == "S1"
        assert output.general_info.polarization == "HH"
        r = output.roi_info[0]
        assert r.roi_name == 0
        assert r.burst == 0
        assert r.roi_size_azimuth == 128
        assert r.roi_size_range == 128
        assert r.ambiguity_ratio_db == -25.0
        assert r.roi_center_azimuth_pixel == 20
        assert r.roi_center_range_pixel == 20
        assert np.array_equal(
            r.roi_center_ground_point_coordinates, np.array([-4989397.154, 2746837.255, -2862071.786])
        )

    def test_dtar_single_channel_single_roi(self, mocker) -> None:
        """Testing happy path with explicit config"""
        self._setup_dtar_mocks(mocker)
        config = AmbiguityRatioConfig(interpolation_factor=16, cropping_size=(64, 64))
        product = self._make_mock_product(mocker)
        channel_data = product.get_channel_data(channel_id=1)
        channel_data.read_data.return_value = self._rng.random((64, 64))

        result = distributed_target_ambiguity_ratio_analysis(
            product=product, roi_centers=self._roi_centers[:1], config=config
        )

        assert len(result) == 1
        r = result[0].roi_info[0]
        assert r.roi_size_azimuth == 64
        assert r.roi_size_range == 64
        assert r.ambiguity_ratio_db == -25.0

    def test_dtar_multiple_channels(self, mocker) -> None:
        """Testing with multiple channels"""
        self._setup_dtar_mocks(mocker)
        product = self._make_mock_product(mocker, n_channels=2)

        result = distributed_target_ambiguity_ratio_analysis(
            product=product, roi_centers=self._roi_centers[:1], config=None
        )

        assert len(result) == 2
        assert result[0].general_info.channel == "1"
        assert result[0].general_info.swath == "S1"
        assert result[1].general_info.channel == "2"
        assert result[1].general_info.swath == "S2"

    def test_dtar_multiple_roi(self, mocker) -> None:
        """Testing with multiple ROIs"""
        self._setup_dtar_mocks(mocker)
        product = self._make_mock_product(mocker)

        result = distributed_target_ambiguity_ratio_analysis(
            product=product, roi_centers=self._roi_centers, config=None
        )

        assert len(result) == 1
        for roi_center, r in zip(self._roi_centers, result[0].roi_info, strict=True):
            assert r.roi_center_azimuth_pixel == roi_center[1]
            assert r.roi_center_range_pixel == roi_center[0]

    def test_dtar_direct_geocoding_fails(self, mocker) -> None:
        """Testing when direct geocoding raises an exception"""
        mocker.patch(
            "perseo_quality.tar_analysis.analysis.detect_burst_from_pixel",
            return_value=0,
        )
        mocker.patch(
            "perseo_quality.tar_analysis.analysis.direct_geocoding_monostatic",
            side_effect=Exception("Geocoding failed"),
        )
        product = self._make_mock_product(mocker)

        result = distributed_target_ambiguity_ratio_analysis(
            product=product, roi_centers=self._roi_centers[:1], config=None
        )

        assert len(result) == 1
        assert result[0].roi_info[0] is None

    def test_dtar_doppler_rate_none(self, mocker) -> None:
        """Testing when channel_data.doppler_rate is None"""
        mocker.patch(
            "perseo_quality.tar_analysis.analysis.detect_burst_from_pixel",
            return_value=0,
        )
        mocker.patch(
            "perseo_quality.tar_analysis.analysis.direct_geocoding_monostatic",
            return_value=np.array([-4989397.154, 2746837.255, -2862071.786]),
        )

        channel_data = self._make_mock_channel_data(mocker)
        channel_data.doppler_rate = None
        product = mocker.MagicMock()
        product.name = "TestProduct"
        product.channels_list = [1]
        product.get_channel_data.return_value = channel_data

        result = distributed_target_ambiguity_ratio_analysis(
            product=product, roi_centers=self._roi_centers[:1], config=None
        )

        assert len(result) == 1
        assert result[0].roi_info[0] is None

    def test_dtar_prf_none(self, mocker) -> None:
        """Testing when channel_data.prf is None"""
        mocker.patch(
            "perseo_quality.tar_analysis.analysis.detect_burst_from_pixel",
            return_value=0,
        )
        mocker.patch(
            "perseo_quality.tar_analysis.analysis.direct_geocoding_monostatic",
            return_value=np.array([-4989397.154, 2746837.255, -2862071.786]),
        )

        channel_data = self._make_mock_channel_data(mocker)
        channel_data.prf = None
        product = mocker.MagicMock()
        product.name = "TestProduct"
        product.channels_list = [1]
        product.get_channel_data.return_value = channel_data

        result = distributed_target_ambiguity_ratio_analysis(
            product=product, roi_centers=self._roi_centers[:1], config=None
        )

        assert len(result) == 1
        assert result[0].roi_info[0] is None

    def test_dtar_ambiguities_out_of_scene(self, mocker) -> None:
        """Testing when ambiguities are outside the scene boundaries"""
        self._setup_dtar_mocks(mocker, amb_inside_scene=False)
        product = self._make_mock_product(mocker)

        result = distributed_target_ambiguity_ratio_analysis(
            product=product, roi_centers=self._roi_centers[:1], config=None
        )

        assert len(result) == 1
        assert result[0].roi_info[0] is None

    def test_dtar_computation_fails(self, mocker) -> None:
        """Testing when ambiguity_ratio_computation_core raises an exception"""
        mocker.patch(
            "perseo_quality.tar_analysis.analysis.detect_burst_from_pixel",
            return_value=0,
        )
        mocker.patch(
            "perseo_quality.tar_analysis.analysis.direct_geocoding_monostatic",
            return_value=np.array([-4989397.154, 2746837.255, -2862071.786]),
        )
        mocker.patch(
            "perseo_quality.tar_analysis.analysis.compute_ambiguities_locations",
            return_value=((50, 100), (150, 300), 0.5, 0.001),
        )
        mocker.patch(
            "perseo_quality.tar_analysis.analysis.are_ambiguities_inside_scene",
            return_value=True,
        )
        mocker.patch(
            "perseo_quality.tar_analysis.analysis.ambiguity_ratio_computation_core",
            side_effect=Exception("Computation failed"),
        )
        product = self._make_mock_product(mocker)

        result = distributed_target_ambiguity_ratio_analysis(
            product=product, roi_centers=self._roi_centers[:1], config=None
        )

        assert len(result) == 1
        assert result[0].roi_info[0] is None
