# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for radiometric_analysis/block_wise/support.py core functionalities"""

from __future__ import annotations

import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import numpy.typing as npt
from arepytools.timing.precisedatetime import PreciseDateTime
from netCDF4 import Dataset

import perseo_quality.radiometric_analysis.block_wise.support as support
from perseo_quality.radiometric_analysis.block_wise.config import (
    Radiometric2DHistogramParameters,
)
from perseo_quality.radiometric_analysis.custom_dataclasses import (
    RadiometricAnalysisDirection,
    RadiometricOutputProductGeneralInfo,
    RadiometricProfilesOutput,
)


class MockTrajectory:
    """Mocking trajectory class"""

    def evaluate(self, time) -> npt.NDArray[np.floating]:
        """Mocking position interpolation"""
        return [5634298.570491991, -4277813.834855013, 183850.74790036504]

    def evaluate_first_derivatives(self, time) -> npt.NDArray[np.floating]:
        """Mocking velocity interpolation"""
        return [-797.011102366091, -1383.8309567802658, -7427.764230040876]


class Compute2DHistogramTest(unittest.TestCase):
    """Testing radiometric_analysis/support.py compute_2d_histogram function"""

    def setUp(self) -> None:
        # reference results
        self.tolerance = 1e-9
        self._ref_hist = np.stack(
            [
                [0, 0, 0, 0],
                [0, 0, 189, 17],
                [0, 0, 0, 0],
            ]
        )
        self._ref_x_bins = np.array([0.0, 0.975609756097561, 1.951219512195122, 2.926829268292683, 3.902439024390244])
        self._ref_y_bins = np.array([1.9841646002547808, 8.650831266921447, 15.317497933588115, 21.98416460025478])

    def test_compute_2d_histogram(self):
        """Testing compute_2d_histogram function"""
        random_rng1 = np.random.default_rng(12345)
        hist, x_bins, y_bins = support.compute_2d_histogram(
            x_data=random_rng1.random((206)) + random_rng1.integers(2, 6),
            y_data=random_rng1.random((206)) * 4 + 10,
            x_axis=np.linspace(0, 4, 206),
            config=Radiometric2DHistogramParameters(y_bins_center_margin=10, y_bins_num=4, x_bins_step=50),
        )
        np.testing.assert_allclose(hist, self._ref_hist, atol=self.tolerance, rtol=0)
        np.testing.assert_allclose(x_bins, self._ref_x_bins, atol=self.tolerance, rtol=0)
        np.testing.assert_allclose(y_bins, self._ref_y_bins, atol=self.tolerance, rtol=0)


class MaskingOutliersByPercentileTest(unittest.TestCase):
    """Testing radiometric_analysis/support.py masking_outliers_by_percentiles function"""

    def setUp(self) -> None:
        # reference results
        self._ref_masked_num = 17

    def test_masking_outliers_by_percentiles(self):
        """Testing masking_outliers_by_percentiles"""
        random_rng1 = np.random.default_rng(123)
        raster = random_rng1.random((15, 15))
        masked = support.masking_outliers_by_percentiles(data=raster, kernel=(5, 5), percentile_boundaries=[20, 80])
        self.assertEqual(np.sum(np.isnan(masked)), self._ref_masked_num)


class RadiometricProfilesToNetCDF(unittest.TestCase):
    """Testing radiometric_analysis/support.py radiometric_profiles_to_netcdf function"""

    def test_radiometric_profiles_to_netcdf(self):
        """Testing radiometric_profiles_to_netcdf function"""
        with TemporaryDirectory() as temp_dir:
            out_fldr = Path(temp_dir).joinpath("out")
            out_fldr.mkdir()
            tag = "test"
            data = RadiometricProfilesOutput(
                general_info=RadiometricOutputProductGeneralInfo(
                    product="test",
                    channel="1",
                    swath="S1",
                    acquisition_mode="SCANSAR",
                    orbit_direction="DESCENDING",
                    polarization="HH",
                    product_type="SLC",
                    radiometric_quantity="BETA_NOUGHT",
                    sensor="",
                    acquisition_start_time=datetime(2020, 1, 1),
                ),
                blocks_num=3,
                direction=RadiometricAnalysisDirection.RANGE,
                azimuth_block_centers=np.array(
                    [
                        PreciseDateTime.from_numeric_datetime(2020, 1, 1),
                        PreciseDateTime.from_numeric_datetime(2020, 1, 2),
                        PreciseDateTime.from_numeric_datetime(2020, 1, 3),
                    ]
                ),
                range_block_centers=np.array([250, 250, 250]),
                azimuth_start_time=PreciseDateTime.from_numeric_datetime(2020, 1, 1),
                hist_2d=np.array([[0, 0, 0], [0, 10, 0], [1, 2, 0]]).reshape(3, 3),
                block_azimuth_times=np.tile(np.linspace(9, 13, 10), 3).reshape((3, 10)),
                hist_x_bins_axis=np.ones(10),
                hist_y_bins_axis=np.ones(10),
                look_angles=np.ones((3, 10)),
                incidence_angles=np.ones((3, 10)),
                profiles=np.ones((3, 10)),
            )
            out_file = out_fldr.joinpath(tag + "_profiles_" + data.general_info.product + ".nc")
            support.radiometric_profiles_to_netcdf(data=[data], out_path=out_fldr, tag=tag)

            # checking results
            self.assertTrue(out_file.exists())
            self.assertTrue(out_file.is_file())
            root = Dataset(out_file, "r", format="NETCDF4")
            self.assertEqual(root.product, data.general_info.product)
            self.assertEqual(root.sensor, data.general_info.sensor)
            self.assertEqual(root.product_type, data.general_info.product_type)
            self.assertEqual(root.acquisition_mode, data.general_info.acquisition_mode)
            self.assertEqual(root.orbit_direction, data.general_info.orbit_direction)
            self.assertEqual(root.acquisition_start_time, str(data.general_info.acquisition_start_time))
            self.assertEqual(root.direction, data.direction.name.lower())
            self.assertEqual(root.output_radiometric_quantity, data.general_info.radiometric_quantity)
            self.assertIn(data.general_info.swath, root.groups)
            self.assertIn(data.general_info.polarization, root[data.general_info.swath].groups)
            pol_grp = root[data.general_info.swath][data.general_info.polarization]
            self.assertEqual(pol_grp.swath, data.general_info.swath)
            self.assertEqual(pol_grp.channel, data.general_info.channel)
            self.assertEqual(pol_grp.polarization, data.general_info.polarization)
            self.assertEqual(pol_grp.azimuth_blocks_num, data.blocks_num)
            self.assertListEqual(pol_grp.azimuth_block_centers, [str(d) for d in data.azimuth_block_centers])
            np.testing.assert_array_equal(pol_grp.range_block_centers, data.range_block_centers)
            np.testing.assert_array_equal(pol_grp.variables["look_angles"][:].data, data.look_angles)
            np.testing.assert_array_equal(pol_grp.variables["incidence_angles"][:].data, data.incidence_angles)
            np.testing.assert_array_equal(pol_grp.variables["radiometric_profiles"][:].data, data.profiles)
            np.testing.assert_array_equal(pol_grp.variables["azimuth_times"][:].data, data.block_azimuth_times)
            root.close()


if __name__ == "__main__":
    unittest.main()
