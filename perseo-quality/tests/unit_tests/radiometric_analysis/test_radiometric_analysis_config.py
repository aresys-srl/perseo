# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for radiometric_analysis/block_wise/config.py core functionalities"""

from __future__ import annotations

import unittest
from dataclasses import fields

from perseo_quality.radiometric_analysis.block_wise.config import (
    ProfileExtractionParameters,
    Radiometric2DHistogramParameters,
    RadiometricProfilesConfig,
    RiverMaskingConfig,
    RiverMaskingMode,
)


class RadiometricProfilesConfigTest(unittest.TestCase):
    """Testing radiometric profiles config dataclasses core functionalities"""

    def setUp(self) -> None:
        # creating test data

        self.rp_flags = {
            "azimuth_block_size": 1500,
            "range_pixel_margin": 50,
        }
        self.rp_hist_flags = {
            "x_bins_step": 5,
            "y_bins_num": 120,
            "y_bins_center_margin": 1.5,
        }
        self.rp_prof_flags = {
            "outlier_removal": False,
            "smoothening_filter": True,
            "filtering_kernel_size": [2, 2],
            "outliers_percentile_boundaries": [10, 80],
            "outliers_kernel_size": [15, 15],
        }
        self.rp_river_flags = {
            "river_masking_mode": "FULL",
            "local_stats_window": 30,
            "backscatter_threshold_percentile": 35,
            "cv_lower_threshold_percentile": 30,
            "cv_upper_threshold_percentile": 40,
        }

    def test_river_masking_parameters_from_dict(self):
        """Testing RiverMaskingConfig dataclass generation from dictionary"""
        dtc = RiverMaskingConfig.from_dict(self.rp_river_flags)

        for key, item in self.rp_river_flags.items():
            dataclass_key = [field.name for field in fields(dtc) if key in field.name][0]
            value = getattr(dtc, dataclass_key)
            if isinstance(value, RiverMaskingMode):
                self.assertEqual(RiverMaskingMode[item], value)
            else:
                self.assertEqual(item, value)

    def test_radiometric_profiles_histogram_parameters_from_dict(self):
        """Testing Radiometric2DHistogramParameters dataclass generation from dictionary"""
        dtc = Radiometric2DHistogramParameters.from_dict(self.rp_hist_flags)

        for key, item in self.rp_hist_flags.items():
            dataclass_key = [field.name for field in fields(dtc) if key in field.name][0]
            value = getattr(dtc, dataclass_key)
            self.assertEqual(item, value)

    def test_radiometric_profiles_prof_parameters_from_dict(self):
        """Testing RadiometricAnalysisParameters dataclass generation from dictionary"""
        dtc = ProfileExtractionParameters.from_dict(self.rp_prof_flags)

        for key, item in self.rp_prof_flags.items():
            dataclass_key = [field.name for field in fields(dtc) if key in field.name][0]
            value = getattr(dtc, dataclass_key)
            if isinstance(value, tuple):
                self.assertEqual(tuple(item), value)
            else:
                self.assertEqual(item, value)

    def test_radiometric_profiles_config_from_dict(self):
        """Testing RadiometricProfilesConfig dataclass generation from dictionary"""
        total_dict = self.rp_flags.copy()
        total_dict["histogram_parameters"] = self.rp_hist_flags
        prof_config = self.rp_prof_flags
        prof_config["river_masking"] = self.rp_river_flags
        total_dict["profile_extraction_parameters"] = prof_config

        dtc = RadiometricProfilesConfig.from_dict(total_dict)

        for key, item in total_dict.items():
            if key == "profile_extraction_parameters":
                for key, item in self.rp_prof_flags.items():
                    dataclass_key = [field.name for field in fields(ProfileExtractionParameters) if key in field.name][
                        0
                    ]
                    value = getattr(dtc.profile_extraction_parameters, dataclass_key)
                    if isinstance(value, tuple):
                        self.assertEqual(tuple(item), value)
                    elif key == "river_masking":
                        self.assertEqual(RiverMaskingConfig.from_dict(item), value)
                    else:
                        self.assertEqual(item, value)
            elif key == "histogram_parameters":
                for key, item in self.rp_hist_flags.items():
                    dataclass_key = [
                        field.name for field in fields(Radiometric2DHistogramParameters) if key in field.name
                    ][0]
                    value = getattr(dtc.histogram_parameters, dataclass_key)
                    if isinstance(value, tuple):
                        self.assertEqual(tuple(item), value)
                    else:
                        self.assertEqual(item, value)
            else:
                dataclass_key = [field.name for field in fields(dtc) if key in field.name][0]
                value = getattr(dtc, dataclass_key)
                self.assertEqual(item, value)


if __name__ == "__main__":
    unittest.main()
