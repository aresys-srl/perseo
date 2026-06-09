# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for interferometric_analysis main functions"""

from __future__ import annotations

import numpy as np
import pytest

from perseo_quality.core.generic_dataclasses import SARPolarization
from perseo_quality.interferometric_analysis.analysis import interferometric_analysis


class TestInterferometricAnalysis:
    """Testing interferometric_analysis main function"""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        """Testing setup"""
        self._burst_lines = 20
        self._n_range_pixels = 10
        self._slant_range_axis = np.arange(self._n_range_pixels)

    def _make_mock_channel_data(self, mocker):
        """Create a mock channel data object"""
        channel_data = mocker.MagicMock()
        channel_data.swath_name = "S1"
        channel_data.polarization = SARPolarization.HH
        channel_data.lines_per_burst = np.array([self._burst_lines])
        channel_data.slant_range_axis = self._slant_range_axis
        channel_data.read_data.return_value = np.random.default_rng(42).random(
            (self._n_range_pixels * 2, self._burst_lines)
        )
        return channel_data

    def _make_mock_product(self, mocker):
        """Create a mock product object"""
        product = mocker.MagicMock()
        product.name = "TestProduct"
        product.channels_list = [1]
        product.get_channel_data.return_value = self._make_mock_channel_data(mocker)
        return product

    def test_coherence_map_path(self, mocker) -> None:
        """Testing with coherence map as input (no coherence computation needed)"""
        mock_interferogram_core = mocker.patch(
            "perseo_quality.interferometric_analysis.analysis.coherence_computation_interferogram_core"
        )
        mock_histogram_core = mocker.patch(
            "perseo_quality.interferometric_analysis.analysis.coherence_2d_histogram_computation_core"
        )
        product = self._make_mock_product(mocker)
        mock_histogram_core.return_value = mocker.MagicMock()

        result = interferometric_analysis(product=product, second_product=None, config=None)

        # For coherence map: enable_coherence_computation=False
        # Should call histogram, NOT interferogram core
        mock_interferogram_core.assert_not_called()
        mock_histogram_core.assert_called_once()

        assert len(result) == 1
        output = result[0]
        assert output.channel_name == 1
        assert output.swath == "S1"
        assert output.polarization == SARPolarization.HH
        assert output.burst == 0

    def test_interferogram_path(self, mocker) -> None:
        """Testing with interferogram as input (coherence computation enabled)"""
        mock_interferogram_core = mocker.patch(
            "perseo_quality.interferometric_analysis.analysis.coherence_computation_interferogram_core"
        )
        mock_histogram_core = mocker.patch(
            "perseo_quality.interferometric_analysis.analysis.coherence_2d_histogram_computation_core"
        )
        product = self._make_mock_product(mocker)
        mock_histogram_core.return_value = mocker.MagicMock()
        mock_interferogram_core.return_value = np.random.default_rng(42).random(
            (self._burst_lines, self._n_range_pixels * 2)
        )

        config = mocker.MagicMock()
        config.enable_coherence_computation = True
        config.coherence_kernel = 15
        config.azimuth_blocks_number = None
        config.range_blocks_number = None
        config.coherence_bins_number = 80

        result = interferometric_analysis(product=product, second_product=None, config=config)

        # For interferogram: enable_coherence_computation=True
        # Should call both interferogram and histogram cores
        mock_interferogram_core.assert_called_once()
        mock_histogram_core.assert_called_once()

        assert len(result) == 1

    def test_co_registered_path(self, mocker) -> None:
        """Testing with two co-registered products"""
        mock_coreg_core = mocker.patch(
            "perseo_quality.interferometric_analysis.analysis.coherence_computation_co_registered_core"
        )
        mock_histogram_core = mocker.patch(
            "perseo_quality.interferometric_analysis.analysis.coherence_2d_histogram_computation_core"
        )
        product = self._make_mock_product(mocker)
        second_product = self._make_mock_product(mocker)
        mock_histogram_core.return_value = mocker.MagicMock()
        mock_coreg_core.return_value = np.random.default_rng(42).random((self._burst_lines, self._n_range_pixels * 2))

        result = interferometric_analysis(product=product, second_product=second_product, config=None)

        # For co-registered: should call co_registered core
        mock_coreg_core.assert_called_once()
        mock_histogram_core.assert_called_once()

        assert len(result) == 1

    def test_nan_replacement_on_coherence_map(self, mocker) -> None:
        """Testing that zeros are replaced by NaN in the coherence map path"""
        mock_histogram_core = mocker.patch(
            "perseo_quality.interferometric_analysis.analysis.coherence_2d_histogram_computation_core"
        )
        product = self._make_mock_product(mocker)
        mock_histogram_core.return_value = mocker.MagicMock()

        # Force read_data to return data that becomes coherence map with zeros
        channel_data = product.get_channel_data.return_value
        channel_data.read_data.return_value = np.zeros((self._n_range_pixels * 2, self._burst_lines))

        interferometric_analysis(product=product, second_product=None, config=None)

        # The data passed to histogram core should have NaN where zeros were
        call_args = mock_histogram_core.call_args
        coherence_arg = call_args.kwargs["coherence"]
        assert np.all(np.isnan(coherence_arg))

    def test_config_none_uses_default(self, mocker) -> None:
        """Testing that config=None uses default InterferometricConfig"""
        product = self._make_mock_product(mocker)
        # With default config, enable_coherence_computation is False
        mock_hist = mocker.patch(
            "perseo_quality.interferometric_analysis.analysis.coherence_2d_histogram_computation_core"
        )
        mock_hist.return_value = mocker.MagicMock()
        result = interferometric_analysis(product=product)
        mock_hist.assert_called_once()
        assert len(result) == 1
