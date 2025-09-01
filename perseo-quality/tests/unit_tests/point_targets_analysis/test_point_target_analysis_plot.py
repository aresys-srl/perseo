# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for point_target_analysis/graphical_output.py core functionalities"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from perseo_quality.point_targets_analysis.custom_dataclasses import (
    IRFGraphDataOutput,
    PTAGraphsInfo,
    RCSGraphDataOutput,
)
from perseo_quality.point_targets_analysis.graphical_output import (
    irf_graphs,
    rcs_graphs,
)


class PointTargetsGraphicalOutputTest(unittest.TestCase):
    """Testing Point Target Analysis graphical output functionalities"""

    def test_irf_graphs(self) -> None:
        """Testing irf_graphs"""
        data_graph = IRFGraphDataOutput(
            image=np.ones((20, 20)),
            rng_axis=np.linspace(1, 10, 20),
            rng_profile=np.ones(20),
            rng_resolution=1.07,
            rng_step_distance=1.95,
            az_axis=np.linspace(3, 25, 20),
            az_profile=np.ones(20),
            az_resolution=1.1,
            az_step_distance=4.1,
            side_lobes_directions=(np.inf, 0),
        )
        data_val = {
            "azimuth_localization_error_[m]": 19.13,
            "slant_range_localization_error_[m]": 0.118,
            "range_resolution_[m]": 2,
            "azimuth_resolution_[m]": 4.5,
            "range_pslr_[dB]": -15,
            "azimuth_pslr_[dB]": -14,
            "range_islr_[dB]": -12,
            "azimuth_islr_[dB]": -11,
            "ground_range_localization_error_[m]": 4,
        }
        graphs_info = PTAGraphsInfo(channel="s1_vv", burst=str(0), polarization="VV", swath="S1", target="CR12")
        with TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            irf_graphs(data_graph=data_graph, data_values=data_val, graphs_info=graphs_info, out_dir=temp_dir)
            self.assertTrue(
                temp_dir.joinpath(
                    f"irf_{graphs_info.channel}_trgt_{graphs_info.target}_b_{graphs_info.burst}.png"
                ).is_file()
            )

    def test_rcs_graphs(self) -> None:
        """Testing rcs_graphs"""
        data_graph = RCSGraphDataOutput(
            image=np.ones((20, 20)),
            rng_step_distance=1.95,
            az_step_distance=4.08,
            interp_factor=2,
            rcs_db=-70,
            rcs_lin=1,
            roi_background=[(2, 3, 2, 4), (16, 17, 10, 11), (10, 11, 16, 17), (8, 9, 11, 12)],
            roi_peak=[7, 8, 12, 15],
            roi_size=np.array([20, 20]),
        )
        graphs_info = PTAGraphsInfo(channel="s1_vv", burst=str(0), polarization="VV", swath="S1", target="CR12")
        with TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            rcs_graphs(data_graph=data_graph, graphs_info=graphs_info, out_dir=temp_dir)
            self.assertTrue(
                temp_dir.joinpath(
                    f"rcs_{graphs_info.channel}_trgt_{graphs_info.target}_b_{graphs_info.burst}.png"
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
