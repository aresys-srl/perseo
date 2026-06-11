# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for spectral_analysis/analysis functionalities"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from perseo_core.timing import PreciseDateTime

from perseo_quality.core.generic_dataclasses import (
    SARAcquisitionMode,
    SARImageType,
    SAROrbitDirection,
    SARPolarization,
    SARRadiometricQuantity,
)
from perseo_quality.io.point_targets import PointTarget
from perseo_quality.spectral_analysis.analysis import (
    block_wise_distributed_spectral_analysis,
    point_target_spectral_analysis,
)
from perseo_quality.spectral_analysis.custom_dataclasses import (
    DistributedSpectraDataOutput,
    PointTargetSpectraDataOutput,
    SpectralAnalysisTargetInfo,
)


class TestPointTargetSpectralAnalysis:
    """Testing Point Target Spectral Analysis main function"""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        """Testing setup"""
        self._cropping_size = (128, 128)
        self._rng = np.random.default_rng(42)
        self._point_targets = [
            PointTarget(name="T1", xyz_coordinates=np.array([-4989394.044, 2746844.389, -2862070.09])),
            PointTarget(name="T2", xyz_coordinates=np.array([-4979009.54, 2766786.057, -2860862.575])),
        ]

    def _make_mock_channel_data(self, mocker, swath_name: str = "S1", acq_mode=SARAcquisitionMode.STRIPMAP):
        channel_data = mocker.MagicMock()
        channel_data.swath_name = swath_name
        channel_data.polarization = SARPolarization.HH
        channel_data.acquisition_mode = acq_mode
        channel_data.orbit_direction = SAROrbitDirection.ASCENDING
        channel_data.image_type = SARImageType.SLC
        channel_data.sensor_name = "TestSensor"
        channel_data.radiometric_quantity = SARRadiometricQuantity.BETA_NOUGHT
        channel_data.prf = 1.71713e03
        channel_data.mid_azimuth_time = mocker.MagicMock()
        channel_data.trajectory = mocker.MagicMock()
        channel_data.lines_per_burst = np.array([200])
        channel_data.azimuth_axis = np.arange(200) + PreciseDateTime.from_numeric_datetime(2019, 1, 8, 8, 32, 40)
        channel_data.slant_range_axis = np.arange(500) * 1e-5 + 5.365309240726020e-03
        channel_data.doppler_centroid = mocker.MagicMock()
        channel_data.doppler_centroid.evaluate.return_value = 0.0
        channel_data.read_data.return_value = self._rng.random((128, 128)) + 1j * self._rng.random((128, 128))
        channel_data.times_to_pixel_conversion.return_value = (100.5, 200.5)
        return channel_data

    def _make_mock_product(self, mocker, n_channels: int = 1, acq_mode=SARAcquisitionMode.STRIPMAP):
        product = mocker.MagicMock()
        product.name = "TestProduct"
        product.channels_list = list(range(1, n_channels + 1))
        product.get_channel_data.side_effect = lambda channel_id: self._make_mock_channel_data(
            mocker, f"S{channel_id}", acq_mode
        )
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

    def _setup_pt_spectra_mocks(self, mocker, data_shape=(128, 128)):
        """Set up common PT spectral analysis mocks with happy path defaults"""
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.check_targets_visibility",
            return_value=self._make_visible_targets_df(),
        )
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.inverse_geocoding_monostatic",
            return_value=(mocker.MagicMock(), 0.001),
        )
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.recenter_data",
            side_effect=lambda x: x,
        )
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.fft2",
            return_value=np.zeros(data_shape, dtype=complex),
        )
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.extract_abs_profiles",
            return_value=(
                [np.zeros(data_shape[1]), np.zeros(data_shape[1]), np.zeros(data_shape[1])],
                [np.zeros(data_shape[0]), np.zeros(data_shape[0]), np.zeros(data_shape[0])],
            ),
        )
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.compute_spectrum_boundaries",
            return_value=(40, 88),
        )
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.compute_spectrogram_db",
            return_value=(
                np.zeros((data_shape[1], data_shape[1])),
                np.linspace(-0.5, 0.5, data_shape[1]),
                np.arange(data_shape[1]),
            ),
        )
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.extract_phase_profiles",
            return_value=(
                [np.zeros(data_shape[1]), np.zeros(data_shape[1]), np.zeros(data_shape[1])],
                [np.zeros(data_shape[0]), np.zeros(data_shape[0]), np.zeros(data_shape[0])],
            ),
        )
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.compute_polynomial_fit",
            return_value=mocker.MagicMock(),
        )
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.convert_to_db",
            side_effect=lambda x: np.abs(x),
        )

    def test_pt_spectra_default_cropping_size(self, mocker) -> None:
        """Testing with default cropping size"""
        self._setup_pt_spectra_mocks(mocker)
        product = self._make_mock_product(mocker)

        result = point_target_spectral_analysis(product=product, point_targets=self._point_targets[:1])

        assert len(result) == 1
        output = result[0]
        assert isinstance(output, PointTargetSpectraDataOutput)
        assert output.general_info.product == "TestProduct"
        assert output.general_info.channel == "1"
        assert output.general_info.swath == "S1"
        assert output.general_info.polarization == "HH"
        assert output.general_info.sensor == "TestSensor"
        assert output.general_info.product_type == "SLC"
        assert output.general_info.acquisition_mode == "STRIPMAP"
        assert output.general_info.orbit_direction == "ASCENDING"
        assert isinstance(output.general_info.acquisition_start_time, type(output.general_info.acquisition_start_time))
        assert len(output.targets_info) == 1
        target_info = output.targets_info[0]
        assert isinstance(target_info, SpectralAnalysisTargetInfo)
        assert target_info.target_name == "T1"
        assert target_info.burst == 0
        assert target_info.roi_size_azimuth == 128
        assert target_info.roi_size_range == 128
        assert target_info.target_azimuth_pixel == 100.5
        assert target_info.target_range_pixel == 200.5

    def test_pt_spectra_custom_cropping_size(self, mocker) -> None:
        """Testing with custom cropping size"""
        self._setup_pt_spectra_mocks(mocker, data_shape=(64, 64))
        channel_data = self._make_mock_channel_data(mocker)
        channel_data.read_data.return_value = self._rng.random((64, 64)) + 1j * self._rng.random((64, 64))
        product = mocker.MagicMock()
        product.name = "TestProduct"
        product.channels_list = [1]
        product.get_channel_data.return_value = channel_data

        result = point_target_spectral_analysis(
            product=product, point_targets=self._point_targets[:1], cropping_size=(64, 64)
        )

        assert len(result) == 1
        target_info = result[0].targets_info[0]
        assert target_info.roi_size_azimuth == 64
        assert target_info.roi_size_range == 64

    def test_pt_spectra_multiple_targets(self, mocker) -> None:
        """Testing with multiple point targets"""
        df = self._make_visible_targets_df(target_ids=["T1", "T2"], burst_list=[[0], [0]])
        self._setup_pt_spectra_mocks(mocker)
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.check_targets_visibility",
            return_value=df,
        )
        product = self._make_mock_product(mocker)

        result = point_target_spectral_analysis(product=product, point_targets=self._point_targets)

        assert len(result) == 1
        assert len(result[0].targets_info) == 2
        assert result[0].targets_info[0].target_name == "T1"
        assert result[0].targets_info[1].target_name == "T2"

    def test_pt_spectra_target_not_visible(self, mocker) -> None:
        """Testing when target is not visible in the scene"""
        df = self._make_visible_targets_df(burst_list=[None])
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.check_targets_visibility",
            return_value=df,
        )
        product = self._make_mock_product(mocker)

        result = point_target_spectral_analysis(product=product, point_targets=self._point_targets[:1])

        assert result == []

    def test_pt_spectra_inverse_geocoding_fails(self, mocker) -> None:
        """Testing when inverse geocoding raises an exception"""
        df = self._make_visible_targets_df()
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.check_targets_visibility",
            return_value=df,
        )
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.inverse_geocoding_monostatic",
            side_effect=Exception("Geocoding failed"),
        )
        product = self._make_mock_product(mocker)

        result = point_target_spectral_analysis(product=product, point_targets=self._point_targets[:1])

        assert len(result) == 1
        assert result[0].targets_info[0] is None

    def test_pt_spectra_read_data_fails(self, mocker) -> None:
        """Testing when read_data raises a boundary error"""
        from perseo_quality.core.custom_errors import RangeExceedsBoundariesError

        df = self._make_visible_targets_df()
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.check_targets_visibility",
            return_value=df,
        )
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.inverse_geocoding_monostatic",
            return_value=(mocker.MagicMock(), 0.001),
        )
        channel_data = self._make_mock_channel_data(mocker)
        channel_data.read_data.side_effect = RangeExceedsBoundariesError("Out of bounds")
        product = mocker.MagicMock()
        product.name = "TestProduct"
        product.channels_list = [1]
        product.get_channel_data.return_value = channel_data

        result = point_target_spectral_analysis(product=product, point_targets=self._point_targets[:1])

        assert len(result) == 1
        assert result[0].targets_info[0] is None

    def test_pt_spectra_topsar_deramping(self, mocker) -> None:
        """Testing TOPSAR acquisition mode triggers deramping"""
        df = self._make_visible_targets_df()
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.check_targets_visibility",
            return_value=df,
        )
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.inverse_geocoding_monostatic",
            return_value=(mocker.MagicMock(), 0.001),
        )
        mock_deramping = mocker.patch(
            "perseo_quality.spectral_analysis.analysis.data_deramping",
            side_effect=lambda data, **kwargs: data,
        )
        self._setup_pt_spectra_mocks(mocker)

        channel_data = self._make_mock_channel_data(mocker, acq_mode=SARAcquisitionMode.TOPSAR)
        channel_data.lines_per_burst = np.array([100, 100])
        product = mocker.MagicMock()
        product.name = "TestProduct"
        product.channels_list = [1]
        product.get_channel_data.return_value = channel_data

        result = point_target_spectral_analysis(product=product, point_targets=self._point_targets[:1])

        assert len(result) == 1
        assert result[0].targets_info[0] is not None
        mock_deramping.assert_called_once()

    def test_pt_spectra_with_delay(self, mocker) -> None:
        """Testing with a point target that has a delay set"""
        self._setup_pt_spectra_mocks(mocker)
        product = self._make_mock_product(mocker)

        pt_with_delay = PointTarget(name="T1", xyz_coordinates=np.array([1.0, 2.0, 3.0]), delay=0.001)

        result = point_target_spectral_analysis(product=product, point_targets=[pt_with_delay])

        assert len(result) == 1
        assert result[0].targets_info[0] is not None
        assert result[0].targets_info[0].target_name == "T1"

    def test_pt_spectra_multiple_channels(self, mocker) -> None:
        """Testing with multiple channels"""
        df = self._make_visible_targets_df(target_ids=["T1"], channel=1, burst_list=[[0]])
        df2 = self._make_visible_targets_df(target_ids=["T1"], channel=2, burst_list=[[0]])
        multi_df = pd.concat([df, df2], ignore_index=True)
        self._setup_pt_spectra_mocks(mocker)
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.check_targets_visibility",
            return_value=multi_df,
        )
        product = self._make_mock_product(mocker, n_channels=2)

        result = point_target_spectral_analysis(product=product, point_targets=self._point_targets[:1])

        assert len(result) == 2
        assert result[0].general_info.channel == "1"
        assert result[0].general_info.swath == "S1"
        assert result[1].general_info.channel == "2"
        assert result[1].general_info.swath == "S2"


class TestBlockWiseDistributedSpectralAnalysis:
    """Testing Block Wise Distributed Spectral Analysis main function"""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        """Testing setup"""
        self._rng = np.random.default_rng(42)

    def _make_mock_channel_data(self, mocker, swath_name: str = "S1", acq_mode=SARAcquisitionMode.STRIPMAP):
        channel_data = mocker.MagicMock()
        channel_data.swath_name = swath_name
        channel_data.polarization = SARPolarization.HH
        channel_data.acquisition_mode = acq_mode
        channel_data.orbit_direction = SAROrbitDirection.ASCENDING
        channel_data.image_type = SARImageType.SLC
        channel_data.sensor_name = "TestSensor"
        channel_data.radiometric_quantity = SARRadiometricQuantity.BETA_NOUGHT
        channel_data.lines_per_burst = np.array([200])
        channel_data.azimuth_axis = np.arange(500) + PreciseDateTime.from_numeric_datetime(2019, 1, 8, 8, 32, 40)
        channel_data.slant_range_axis = np.arange(500) * 1e-5 + 5.365309240726020e-03
        channel_data.doppler_centroid = mocker.MagicMock()
        channel_data.doppler_centroid.evaluate.return_value = 0.0
        channel_data.read_data.return_value = self._rng.random((128, 128)) + 1j * self._rng.random((128, 128))
        return channel_data

    def _make_mock_product(self, mocker, n_channels: int = 1, acq_mode=SARAcquisitionMode.STRIPMAP):
        product = mocker.MagicMock()
        product.name = "TestProduct"
        product.channels_list = list(range(1, n_channels + 1))
        product.get_channel_data.side_effect = lambda channel_id: self._make_mock_channel_data(
            mocker, f"S{channel_id}", acq_mode
        )
        return product

    def _setup_distributed_spectra_mocks(self, mocker, data_shape=(200, 500), n_blocks=2):
        """Set up common BW distributed spectral analysis mocks"""
        blocks_partitioning_return = (
            np.array([data_shape[0] // n_blocks] * n_blocks),
            n_blocks,
            [(100, 250), (300, 250)],
        )
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.blocks_partitioning",
            return_value=blocks_partitioning_return,
        )
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.detect_burst_from_pixel",
            return_value=0,
        )
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.recenter_data",
            side_effect=lambda x: x,
        )
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.fft2",
            return_value=np.zeros((500, 100), dtype=complex),
        )
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.extract_abs_profiles",
            return_value=(
                [np.zeros(100), np.zeros(100), np.zeros(100)],
                [np.zeros(500), np.zeros(500), np.zeros(500)],
            ),
        )
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.compute_spectrogram_db",
            return_value=(
                np.zeros((100, 100)),
                np.linspace(-0.5, 0.5, 100),
                np.arange(100),
            ),
        )
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.convert_to_db",
            side_effect=lambda x: np.abs(x),
        )
        return blocks_partitioning_return

    def test_distributed_spectra_single_channel(self, mocker) -> None:
        """Testing basic happy path with stripmap mode"""
        self._setup_distributed_spectra_mocks(mocker)
        product = self._make_mock_product(mocker)

        result = block_wise_distributed_spectral_analysis(product=product)

        assert len(result) == 1
        output = result[0]
        assert isinstance(output, DistributedSpectraDataOutput)
        assert output.general_info.product == "TestProduct"
        assert output.general_info.channel == "1"
        assert output.general_info.swath == "S1"
        assert len(output.blocks_info) == 2
        assert output.blocks_info[0].block_num == 0
        assert output.blocks_info[1].block_num == 1
        assert output.blocks_info[0].doppler_centroid_mid_block == 0.0

    def test_distributed_spectra_multiple_channels(self, mocker) -> None:
        """Testing with multiple channels"""
        self._setup_distributed_spectra_mocks(mocker)
        product = self._make_mock_product(mocker, n_channels=2)

        result = block_wise_distributed_spectral_analysis(product=product)

        assert len(result) == 2
        assert result[0].general_info.channel == "1"
        assert result[1].general_info.channel == "2"
        assert len(result[0].blocks_info) == 2
        assert len(result[1].blocks_info) == 2

    def test_distributed_spectra_topsar_deramping(self, mocker) -> None:
        """Testing TOPSAR acquisition mode triggers deramping"""
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.blocks_partitioning",
            return_value=(np.array([100, 100]), 2, [(100, 250), (300, 250)]),
        )
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.detect_burst_from_pixel",
            return_value=0,
        )
        mock_deramping = mocker.patch(
            "perseo_quality.spectral_analysis.analysis.data_deramping",
            side_effect=lambda data, **kwargs: data,
        )
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.recenter_data",
            side_effect=lambda x: x,
        )
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.fft2",
            return_value=np.zeros((500, 100), dtype=complex),
        )
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.extract_abs_profiles",
            return_value=(
                [np.zeros(100), np.zeros(100), np.zeros(100)],
                [np.zeros(500), np.zeros(500), np.zeros(500)],
            ),
        )
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.compute_spectrogram_db",
            return_value=(
                np.zeros((100, 100)),
                np.linspace(-0.5, 0.5, 100),
                np.arange(100),
            ),
        )
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.convert_to_db",
            side_effect=lambda x: np.abs(x),
        )

        channel_data = self._make_mock_channel_data(mocker, acq_mode=SARAcquisitionMode.TOPSAR)
        channel_data.lines_per_burst = np.array([100, 100])
        product = mocker.MagicMock()
        product.name = "TestProduct"
        product.channels_list = [1]
        product.get_channel_data.return_value = channel_data

        result = block_wise_distributed_spectral_analysis(product=product)

        assert len(result) == 1
        assert len(result[0].blocks_info) == 2
        assert mock_deramping.call_count == 2

    def test_distributed_spectra_blocks_partitioning_called(self, mocker) -> None:
        """Testing that blocks_partitioning is called with correct arguments"""
        mock_bp = mocker.patch(
            "perseo_quality.spectral_analysis.analysis.blocks_partitioning",
            return_value=(np.array([200]), 1, [(100, 250)]),
        )
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.detect_burst_from_pixel",
            return_value=0,
        )
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.recenter_data",
            side_effect=lambda x: x,
        )
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.fft2",
            return_value=np.zeros((500, 200), dtype=complex),
        )
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.extract_abs_profiles",
            return_value=(
                [np.zeros(200), np.zeros(200), np.zeros(200)],
                [np.zeros(500), np.zeros(500), np.zeros(500)],
            ),
        )
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.compute_spectrogram_db",
            return_value=(
                np.zeros((200, 200)),
                np.linspace(-0.5, 0.5, 200),
                np.arange(200),
            ),
        )
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.convert_to_db",
            side_effect=lambda x: np.abs(x),
        )

        product = self._make_mock_product(mocker)

        result = block_wise_distributed_spectral_analysis(product=product, azimuth_block_size=2000)

        assert len(result) == 1
        mock_bp.assert_called_once()
        call_args = mock_bp.call_args.kwargs
        assert "azimuth_axis" in call_args
        assert "range_axis" in call_args
        assert call_args["default_block_size"] == 2000

    def test_distributed_spectra_doppler_centroid_computed(self, mocker) -> None:
        """Testing that doppler_centroid is evaluated for each block"""
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.blocks_partitioning",
            return_value=(np.array([200]), 1, [(100, 250)]),
        )
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.detect_burst_from_pixel",
            return_value=0,
        )
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.recenter_data",
            side_effect=lambda x: x,
        )
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.fft2",
            return_value=np.zeros((500, 200), dtype=complex),
        )
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.extract_abs_profiles",
            return_value=(
                [np.zeros(200), np.zeros(200), np.zeros(200)],
                [np.zeros(500), np.zeros(500), np.zeros(500)],
            ),
        )
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.compute_spectrogram_db",
            return_value=(
                np.zeros((200, 200)),
                np.linspace(-0.5, 0.5, 200),
                np.arange(200),
            ),
        )
        mocker.patch(
            "perseo_quality.spectral_analysis.analysis.convert_to_db",
            side_effect=lambda x: np.abs(x),
        )

        product = self._make_mock_product(mocker)

        result = block_wise_distributed_spectral_analysis(product=product)

        assert len(result) == 1
        assert len(result[0].blocks_info) == 1
        assert result[0].blocks_info[0].doppler_centroid_mid_block == 0.0
