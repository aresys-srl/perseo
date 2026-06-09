# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for radiometric_analysis/point_wise/graphical_output.py"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from perseo_quality.core.generic_dataclasses import SARPolarization, SARProjection, SARRadiometricQuantity
from perseo_quality.radiometric_analysis.custom_dataclasses import (
    PointWiseRadiometricAnalysisOutput,
    RadiometricAnalysisAxes,
    RadiometricAnalysisDirection,
    RadiometricAnalysisValue,
)
from perseo_quality.radiometric_analysis.point_wise.graphical_output import (
    radiometric_analysis_graphs,
)


class TestPointWiseRadiometricPlot:
    """Testing Point-Wise Radiometric Analysis graphical output"""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.axis = np.linspace(0, 10, 100)
        self.profile = np.random.default_rng(42).normal(loc=-15, scale=2, size=100)

    def test_radiometric_analysis_graphs_saves_file(self):
        """Testing radiometric_analysis_graphs saves png file"""
        data = [
            PointWiseRadiometricAnalysisOutput(
                swath="S1",
                burst=0,
                channel="1",
                polarization=SARPolarization.HH,
                projection=SARProjection.SLANT_RANGE,
                original_profile_db=self.profile,
                smoothed_profile_db=self.profile,
                axis=self.axis,
                time=0.0,
                direction=RadiometricAnalysisDirection.RANGE,
                value_type=RadiometricAnalysisValue.AMPLITUDE,
                axis_type=RadiometricAnalysisAxes.NATURAL,
                radiometric_quantity=SARRadiometricQuantity.BETA_NOUGHT,
            )
        ]
        with TemporaryDirectory() as temp_dir:
            radiometric_analysis_graphs(data=data, out_dir=temp_dir, interactive=False)
            out_path = Path(temp_dir)
            png_files = list(out_path.glob("*.png"))
            assert len(png_files) > 0

    def test_radiometric_analysis_graphs_raises_value_error(self):
        """Testing radiometric_analysis_graphs raises ValueError when out_dir=None and interactive=False"""
        data = [
            PointWiseRadiometricAnalysisOutput(
                swath="S1",
                channel="1",
                polarization=SARPolarization.HH,
                projection=SARProjection.SLANT_RANGE,
                original_profile_db=self.profile,
                smoothed_profile_db=self.profile,
                axis=self.axis,
                time=0.0,
                direction=RadiometricAnalysisDirection.RANGE,
                value_type=RadiometricAnalysisValue.AMPLITUDE,
                axis_type=RadiometricAnalysisAxes.NATURAL,
                radiometric_quantity=SARRadiometricQuantity.BETA_NOUGHT,
            )
        ]
        with pytest.raises(ValueError, match="valid output directory path"):
            radiometric_analysis_graphs(data=data, out_dir=None, interactive=False)

    def test_radiometric_analysis_graphs_azimuth_direction(self):
        """Testing radiometric_analysis_graphs with azimuth direction"""
        data = [
            PointWiseRadiometricAnalysisOutput(
                swath="S1",
                channel="1",
                polarization=SARPolarization.VV,
                projection=SARProjection.SLANT_RANGE,
                original_profile_db=self.profile,
                smoothed_profile_db=self.profile,
                axis=self.axis,
                time=0.0,
                direction=RadiometricAnalysisDirection.AZIMUTH,
                value_type=RadiometricAnalysisValue.PHASE,
                axis_type=RadiometricAnalysisAxes.NATURAL,
                radiometric_quantity=SARRadiometricQuantity.BETA_NOUGHT,
            )
        ]
        with TemporaryDirectory() as temp_dir:
            radiometric_analysis_graphs(data=data, out_dir=temp_dir, interactive=False)
            out_path = Path(temp_dir)
            png_files = list(out_path.glob("*.png"))
            assert len(png_files) > 0

    def test_radiometric_analysis_graphs_incidence_angle_axis(self):
        """Testing radiometric_analysis_graphs with INCIDENCE_ANGLE axis type"""
        data = [
            PointWiseRadiometricAnalysisOutput(
                swath="S1",
                channel="1",
                polarization=SARPolarization.HH,
                projection=SARProjection.GROUND_RANGE,
                original_profile_db=self.profile,
                smoothed_profile_db=self.profile,
                axis=self.axis,
                time=0.0,
                direction=RadiometricAnalysisDirection.RANGE,
                value_type=RadiometricAnalysisValue.AMPLITUDE,
                axis_type=RadiometricAnalysisAxes.INCIDENCE_ANGLE,
                radiometric_quantity=SARRadiometricQuantity.BETA_NOUGHT,
            )
        ]
        with TemporaryDirectory() as temp_dir:
            radiometric_analysis_graphs(data=data, out_dir=temp_dir, interactive=False)
            out_path = Path(temp_dir)
            png_files = list(out_path.glob("*.png"))
            assert len(png_files) > 0
