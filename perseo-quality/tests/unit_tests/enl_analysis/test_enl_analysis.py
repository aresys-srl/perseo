# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for enl_analysis/analysis functionalities"""

from __future__ import annotations

import numpy as np
import pytest

from perseo_quality.core.generic_dataclasses import SARPolarization, SARRadiometricQuantity
from perseo_quality.enl_analysis.analysis import equivalent_number_of_looks_analysis
from perseo_quality.enl_analysis.custom_dataclasses import ENLOutput


class TestENLAnalysis:
    """Testing ENL Analysis main function"""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        """Testing setup"""
        self._cropping_size = (8, 8)
        self._roi_centers = [(20, 20), (50, 50)]
        self._rng = np.random.default_rng(42)

    def _make_mock_channel_data(self, mocker, name: str = "S1"):
        """Create a mock channel data object"""
        channel_data = mocker.MagicMock()
        channel_data.swath_name = name
        channel_data.polarization = SARPolarization.HH
        channel_data.radiometric_quantity = SARRadiometricQuantity.BETA_NOUGHT
        channel_data.read_data.return_value = self._rng.random((8, 8)) + 1j * self._rng.random((8, 8))
        return channel_data

    def _make_mock_product(self, mocker, n_channels: int = 1):
        """Create a mock product object"""
        product = mocker.MagicMock()
        product.name = "TestProduct"
        product.channels_list = list(range(1, n_channels + 1))
        product.get_channel_data.side_effect = lambda channel_id: self._make_mock_channel_data(mocker, f"S{channel_id}")
        return product

    def test_single_channel_single_roi(self, mocker) -> None:
        """Testing with one channel and one ROI"""
        product = self._make_mock_product(mocker, n_channels=1)
        result = equivalent_number_of_looks_analysis(
            product=product, roi_centers=[self._roi_centers[0]], cropping_size=self._cropping_size
        )
        assert len(result) == 1
        output = result[0]
        assert isinstance(output, ENLOutput)
        assert output.product_name == "TestProduct"
        assert output.channel == 1
        assert output.swath == "S1"
        assert output.polarization == SARPolarization.HH
        assert output.roi_center == (20, 20)
        assert output.roi_size_azimuth == 8
        assert output.roi_size_range == 8

    def test_multiple_rois(self, mocker) -> None:
        """Testing with one channel and multiple ROIs"""
        product = self._make_mock_product(mocker, n_channels=1)
        result = equivalent_number_of_looks_analysis(
            product=product, roi_centers=self._roi_centers, cropping_size=self._cropping_size
        )
        assert len(result) == 2
        for roi_center, output in zip(self._roi_centers, result, strict=True):
            assert output.roi_center == roi_center

    def test_multiple_channels(self, mocker) -> None:
        """Testing with multiple channels and ROIs"""
        product = self._make_mock_product(mocker, n_channels=2)
        result = equivalent_number_of_looks_analysis(
            product=product, roi_centers=self._roi_centers, cropping_size=self._cropping_size
        )
        assert len(result) == 4
        # first channel outputs
        assert result[0].channel == 1
        assert result[0].swath == "S1"
        assert result[0].roi_center == (20, 20)
        assert result[1].channel == 1
        assert result[1].swath == "S1"
        assert result[1].roi_center == (50, 50)
        # second channel outputs
        assert result[2].channel == 2
        assert result[2].swath == "S2"
        assert result[2].roi_center == (20, 20)
        assert result[3].channel == 2
        assert result[3].swath == "S2"
        assert result[3].roi_center == (50, 50)

    def test_read_data_called_with_correct_params(self, mocker) -> None:
        """Testing that read_data is called with the right parameters"""
        channel_data = self._make_mock_channel_data(mocker)
        product = mocker.MagicMock()
        product.name = "TestProduct"
        product.channels_list = [1]
        product.get_channel_data.return_value = channel_data
        equivalent_number_of_looks_analysis(
            product=product, roi_centers=[self._roi_centers[0]], cropping_size=self._cropping_size
        )
        channel_data.read_data.assert_called_once_with(
            azimuth_index=self._roi_centers[0][1],
            range_index=self._roi_centers[0][0],
            cropping_size=self._cropping_size,
            output_radiometric_quantity=channel_data.radiometric_quantity,
        )

    def test_empty_roi_list(self, mocker) -> None:
        """Testing with empty ROI list"""
        product = self._make_mock_product(mocker, n_channels=1)
        result = equivalent_number_of_looks_analysis(product=product, roi_centers=[], cropping_size=self._cropping_size)
        assert result == []

    def test_output_type(self, mocker) -> None:
        """Testing output type is list of ENLOutput"""
        product = self._make_mock_product(mocker, n_channels=1)
        result = equivalent_number_of_looks_analysis(
            product=product, roi_centers=[self._roi_centers[0]], cropping_size=self._cropping_size
        )
        assert all(isinstance(out, ENLOutput) for out in result)
