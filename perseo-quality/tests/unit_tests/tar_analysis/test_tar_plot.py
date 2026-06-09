# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for tar_analysis/graphical_output.py"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from perseo_quality.core.generic_dataclasses import SARPolarization
from perseo_quality.tar_analysis.custom_dataclasses import AmbiguityRatioOutput
from perseo_quality.tar_analysis.graphical_output import (
    TargetRatioGraphType,
    ambiguities_graphs,
)


class TestTargetRatioGraphType:
    """Testing TargetRatioGraphType enum"""

    def test_enum_values(self):
        """Testing TargetRatioGraphType enum values"""
        assert TargetRatioGraphType.PTAR.value == "PTAR"
        assert TargetRatioGraphType.DTAR.value == "DTAR"

    def test_enum_members(self):
        """Testing TargetRatioGraphType enum members"""
        assert TargetRatioGraphType("PTAR") == TargetRatioGraphType.PTAR
        assert TargetRatioGraphType("DTAR") == TargetRatioGraphType.DTAR


class TestAmbiguitiesGraphs:
    """Testing ambiguities_graphs function"""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.image_8 = np.ones((8, 8))
        self.image_16 = np.ones((16, 16))
        self.item = AmbiguityRatioOutput(
            target_name="CR12",
            product_name="test_product",
            channel="1",
            swath="S1",
            polarization=SARPolarization.HH,
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

    def test_ambiguities_graphs_ptar(self):
        """Testing ambiguities_graphs with PTAR type"""
        with TemporaryDirectory() as temp_dir:
            ambiguities_graphs(data=[self.item], output_dir=temp_dir, graph_type=TargetRatioGraphType.PTAR)
            ptar_dir = Path(temp_dir).joinpath("PTAR")
            assert ptar_dir.exists()
            png_files = list(ptar_dir.glob("*.png"))
            assert len(png_files) > 0

    def test_ambiguities_graphs_dtar(self):
        """Testing ambiguities_graphs with DTAR type"""
        with TemporaryDirectory() as temp_dir:
            ambiguities_graphs(data=[self.item], output_dir=temp_dir, graph_type=TargetRatioGraphType.DTAR)
            dtar_dir = Path(temp_dir).joinpath("DTAR")
            assert dtar_dir.exists()
            png_files = list(dtar_dir.glob("*.png"))
            assert len(png_files) > 0

    def test_ambiguities_graphs_string_graph_type(self):
        """Testing ambiguities_graphs accepts string graph_type"""
        with TemporaryDirectory() as temp_dir:
            ambiguities_graphs(data=[self.item], output_dir=temp_dir, graph_type="PTAR")
            ptar_dir = Path(temp_dir).joinpath("PTAR")
            assert ptar_dir.exists()

    def test_ambiguities_graphs_skip_none_ambiguity_ratio(self):
        """Testing ambiguities_graphs skips items with None ambiguity_ratio_db"""
        item_no_ratio = AmbiguityRatioOutput(target_name="CR12", ambiguity_ratio_db=None)
        with TemporaryDirectory() as temp_dir:
            ambiguities_graphs(data=[item_no_ratio], output_dir=temp_dir, graph_type=TargetRatioGraphType.PTAR)
