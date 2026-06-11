# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for tar_analysis/graphical_output.py"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from perseo_quality.core.generic_dataclasses import SARPolarization
from perseo_quality.tar_analysis.custom_dataclasses import (
    AmbiguityRatioProductGeneralInfo,
    AmbiguityRatioROIInfo,
    AmbiguityRatioTargetInfo,
    DistributedTargetAmbiguityRatioDataOutput,
    PointTargetAmbiguityRatioDataOutput,
)
from perseo_quality.tar_analysis.graphical_output import (
    ambiguities_graphs,
)


class TestAmbiguitiesGraphs:
    """Testing ambiguities_graphs function"""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.image_8 = np.ones((8, 8))
        self.image_16 = np.ones((16, 16))

        general_info = AmbiguityRatioProductGeneralInfo(
            product="test_product",
            channel="1",
            swath="S1",
            polarization=SARPolarization.HH,
            sensor="test_sensor",
            product_type="test_type",
            acquisition_mode="test_mode",
            orbit_direction="Ascending",
            acquisition_start_time=datetime(2023, 1, 1),
        )

        target_info = AmbiguityRatioTargetInfo(
            target_name="CR12",
            burst=0,
            target_azimuth_pixel=100.0,
            target_range_pixel=200.0,
            left_ambiguity_azimuth_pixel=np.float64(50.0),
            left_ambiguity_range_pixel=np.float64(150.0),
            right_ambiguity_azimuth_pixel=np.float64(200.0),
            right_ambiguity_range_pixel=np.float64(250.0),
            ambiguity_ratio_db=-25.3,
            target_image=self.image_16,
            right_ambiguity_image=self.image_8,
            left_ambiguity_image=self.image_8,
        )
        self.ptar_item = PointTargetAmbiguityRatioDataOutput(
            general_info=general_info,
            targets_info=[target_info],
        )

        roi_info = AmbiguityRatioROIInfo(
            roi_name="ROI_001",
            burst=0,
            roi_center_azimuth_pixel=100.0,
            roi_center_range_pixel=200.0,
            left_ambiguity_azimuth_pixel=np.float64(50.0),
            left_ambiguity_range_pixel=np.float64(150.0),
            right_ambiguity_azimuth_pixel=np.float64(200.0),
            right_ambiguity_range_pixel=np.float64(250.0),
            ambiguity_ratio_db=-25.3,
            target_image=self.image_16,
            right_ambiguity_image=self.image_8,
            left_ambiguity_image=self.image_8,
        )
        self.dtar_item = DistributedTargetAmbiguityRatioDataOutput(
            general_info=general_info,
            roi_info=[roi_info],
        )

    def test_ambiguities_graphs_ptar(self):
        """Testing ambiguities_graphs with PTAR data"""
        with TemporaryDirectory() as temp_dir:
            ambiguities_graphs(data=[self.ptar_item], output_dir=temp_dir)
            ptar_dir = Path(temp_dir).joinpath("PTAR")
            assert ptar_dir.exists()
            png_files = list(ptar_dir.glob("*.png"))
            assert len(png_files) > 0

    def test_ambiguities_graphs_dtar(self):
        """Testing ambiguities_graphs with DTAR data"""
        with TemporaryDirectory() as temp_dir:
            ambiguities_graphs(data=[self.dtar_item], output_dir=temp_dir)
            dtar_dir = Path(temp_dir).joinpath("DTAR")
            assert dtar_dir.exists()
            png_files = list(dtar_dir.glob("*.png"))
            assert len(png_files) > 0

    def test_ambiguities_graphs_skip_none_ambiguity_ratio(self):
        """Testing ambiguities_graphs skips items with None ambiguity_ratio_db"""
        general_info = AmbiguityRatioProductGeneralInfo(
            product="test_product",
            channel="1",
            swath="S1",
            polarization=SARPolarization.HH,
            sensor="test_sensor",
            product_type="test_type",
            acquisition_mode="test_mode",
            orbit_direction="Ascending",
            acquisition_start_time=datetime(2023, 1, 1),
        )
        target_info = AmbiguityRatioTargetInfo(
            target_name="CR12",
            ambiguity_ratio_db=None,
        )
        item_no_ratio = PointTargetAmbiguityRatioDataOutput(
            general_info=general_info,
            targets_info=[target_info],
        )
        with TemporaryDirectory() as temp_dir:
            ambiguities_graphs(data=[item_no_ratio], output_dir=temp_dir)
