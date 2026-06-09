# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for spectral_analysis/graphical_output.py"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
import numpy as np
import pytest
from numpy.polynomial import Polynomial

matplotlib.use("Agg")

from perseo_quality.spectral_analysis.custom_dataclasses import (
    DistributedSpectraDataOutput,
    PointTargetSpectraDataOutput,
    SpectralAnalysisBlockInfo,
    SpectralAnalysisProductGeneralInfo,
    SpectralAnalysisTargetInfo,
)
from perseo_quality.spectral_analysis.graphical_output import (
    spectral_graph_core,
    spectral_graphs,
)


class TestSpectralAnalysisPlot:
    """Testing Spectral Analysis graphical output"""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.general_info = SpectralAnalysisProductGeneralInfo(
            product="test_product",
            channel="1",
            swath="S1",
            polarization="HH",
            sensor="",
            product_type="SLC",
            acquisition_mode="STRIPMAP",
            orbit_direction="DESCENDING",
            acquisition_start_time=datetime(2020, 1, 1),
        )
        self.az_freq = np.linspace(-0.5, 0.5, 20)
        self.rng_freq = np.linspace(-0.5, 0.5, 30)
        self.spectrum = np.random.default_rng(42).random((30, 20))
        self.spectrogram = np.random.default_rng(43).random((10, 15))

    def test_spectral_graphs_point_target_absolute(self):
        """Testing spectral_graphs with PointTargetSpectraDataOutput (ABSOLUTE mode)"""
        target_info = SpectralAnalysisTargetInfo(
            target_name="CR12",
            burst=0,
            roi_size_azimuth=20,
            roi_size_range=30,
            target_azimuth_pixel=10.0,
            target_range_pixel=15.0,
            azimuth_frequency_axis=self.az_freq,
            range_frequency_axis=self.rng_freq,
            spectrum_db=self.spectrum,
            spectrum_deg=np.ones_like(self.spectrum),
            spectrogram_db=self.spectrogram,
            spectrogram_frequencies=np.linspace(-0.5, 0.5, 10),
            spectrogram_times=np.linspace(0, 1, 15),
            range_profiles_db=[np.ones(30) * (-10), np.ones(30) * (-12), np.ones(30) * (-8)],
            azimuth_profiles_db=[np.ones(20) * (-10), np.ones(20) * (-12), np.ones(20) * (-8)],
            range_profiles_norm_deg=[np.zeros(30), np.zeros(30), np.zeros(30)],
            azimuth_profiles_norm_deg=[np.zeros(20), np.zeros(20), np.zeros(20)],
            target_phase_value_deg=45.0,
            target_doppler_centroid_Hz=100.0,
            range_polynomial_fit=Polynomial([0.0, 0.0, 0.0]),
            azimuth_polynomial_fit=Polynomial([0.0, 0.0, 0.0]),
            rng_spectrum_boundaries=(5, 25),
            az_spectrum_boundaries=(3, 17),
        )
        data = PointTargetSpectraDataOutput(general_info=self.general_info, targets_info=[target_info])

        with TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir).joinpath("Spectral Analysis")
            out_dir.mkdir()
            spectral_graphs(data=[data], output_dir=temp_dir)
            png_files = list(out_dir.glob("*.png"))
            assert len(png_files) > 0

    def test_spectral_graph_core_point_target_phase(self):
        """Testing spectral_graph_core with PointTarget phase mode"""
        target_info = SpectralAnalysisTargetInfo(
            target_name="CR12",
            burst=0,
            roi_size_azimuth=20,
            roi_size_range=30,
            target_azimuth_pixel=10.0,
            target_range_pixel=15.0,
            azimuth_frequency_axis=self.az_freq,
            range_frequency_axis=self.rng_freq,
            spectrum_db=self.spectrum,
            spectrum_deg=np.ones_like(self.spectrum),
            spectrogram_db=self.spectrogram,
            spectrogram_frequencies=np.linspace(-0.5, 0.5, 10),
            spectrogram_times=np.linspace(0, 1, 15),
            range_profiles_db=[np.ones(30) * (-10), np.ones(30) * (-12), np.ones(30) * (-8)],
            azimuth_profiles_db=[np.ones(20) * (-10), np.ones(20) * (-12), np.ones(20) * (-8)],
            range_profiles_norm_deg=[np.zeros(30), np.zeros(30), np.zeros(30)],
            azimuth_profiles_norm_deg=[np.zeros(20), np.zeros(20), np.zeros(20)],
            target_phase_value_deg=45.0,
            target_doppler_centroid_Hz=100.0,
            range_polynomial_fit=Polynomial([0.0, 0.0, 0.0]),
            azimuth_polynomial_fit=Polynomial([0.0, 0.0, 0.0]),
            rng_spectrum_boundaries=(5, 25),
            az_spectrum_boundaries=(3, 17),
        )
        data = PointTargetSpectraDataOutput(general_info=self.general_info, targets_info=[target_info])

        with TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir).joinpath("Spectral Analysis")
            out_dir.mkdir()
            spectral_graph_core(data=data, graph_mode="PHASE", out_dir=out_dir)
            png_files = list(out_dir.glob("*.png"))
            assert len(png_files) > 0

    def test_spectral_graphs_distributed_target(self):
        """Testing spectral_graphs with DistributedSpectraDataOutput skips None spectrum"""
        block_info = SpectralAnalysisBlockInfo(
            block_num=0,
            spectrum_db=None,
        )
        data = DistributedSpectraDataOutput(general_info=self.general_info, blocks_info=[block_info])

        with TemporaryDirectory() as temp_dir:
            spectral_graphs(data=[data], output_dir=temp_dir)

    def test_spectral_graph_core_skips_none_spectrum(self):
        """Testing spectral_graph_core skips items with None spectrum_db"""
        target_info = SpectralAnalysisTargetInfo(
            target_name="CR12",
            burst=0,
            spectrum_db=None,
            target_phase_value_deg=np.nan,
        )
        data = PointTargetSpectraDataOutput(general_info=self.general_info, targets_info=[target_info])

        with TemporaryDirectory() as temp_dir:
            spectral_graph_core(data=data, graph_mode="ABSOLUTE", out_dir=temp_dir)
