# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for elevation_notch_analysis/graphical_output.py"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
import numpy as np

matplotlib.use("Agg")

from perseo_quality.core.generic_dataclasses import SARPolarization
from perseo_quality.elevation_notch_analysis.custom_dataclasses import (
    ElevationNotchBlockInfo,
    ElevationNotchOutput,
)
from perseo_quality.elevation_notch_analysis.graphical_output import (
    plot_elevation_notch_analysis,
)


class TestElevationNotchPlot:
    """Testing Elevation Notch graphical output"""

    def test_plot_elevation_notch_analysis(self):
        """Testing plot_elevation_notch_analysis generates png file"""
        samples_block = 500
        blocks_info = [
            ElevationNotchBlockInfo(
                block_num=0,
                first_az_line_block=0,
                lines_block=200,
                samples_block=samples_block,
                altitude_m=500.0,
                annotated_roll_deg=-0.5,
                estimated_roll_deg=-0.45,
                antenna_profile_from_data_db=np.random.default_rng(42).normal(loc=-10, scale=1, size=samples_block),
                antenna_profile_from_model_db=np.ones(samples_block) * -9,
                antenna_profile_parabolic_fit_db=np.ones(samples_block) * -8,
                parabolic_fit_axis_deg=np.linspace(-1, 1, samples_block),
                parabola_minimum_deg=0.0,
                parabola_coefficients=np.array([10.0, 0.0, -0.5]),
                antenna_angles_deg=np.linspace(-2, 2, samples_block),
            )
        ]
        data = [
            ElevationNotchOutput(
                product_name="test_product",
                channel="1",
                swath="S1",
                polarization=SARPolarization.HH,
                blocks_info=blocks_info,
            )
        ]

        with TemporaryDirectory() as temp_dir:
            plot_elevation_notch_analysis(data=data, output_dir=temp_dir)
            expected_file = Path(temp_dir).joinpath("elevation_notch_analysis_S1_HH.png")
            assert expected_file.exists()
            assert expected_file.is_file()
