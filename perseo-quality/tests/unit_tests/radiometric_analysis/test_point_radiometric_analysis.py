# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for radiometric_analysis/point_wise/analysis.py core functionalities"""

from dataclasses import fields

import numpy as np
import numpy.typing as npt
import pytest

from perseo_quality.core.generic_dataclasses import SARRadiometricQuantity
from perseo_quality.radiometric_analysis.custom_dataclasses import (
    RadiometricAnalysisAxes,
    RadiometricAnalysisValue,
)
from perseo_quality.radiometric_analysis.point_wise.analysis import (
    _extract_radiometric_profiles,
)
from perseo_quality.radiometric_analysis.point_wise.config import (
    PointWiseRadiometricAnalysisConfig,
    RadiometricAnalysisDirection,
    RadiometricAnalysisParameters,
)


class TestRadiometricAnalysisConfig:
    """Testing radiometric analysis config dataclasses core functionalities"""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        # creating test data

        self.ra_flags = {
            "output_quantity": SARRadiometricQuantity.SIGMA_NOUGHT.name.lower(),
            "value": RadiometricAnalysisValue.AMPLITUDE.name.lower(),
            "direction": RadiometricAnalysisDirection.RANGE.name.lower(),
            "axis": RadiometricAnalysisAxes.NATURAL.name.lower(),
            "outlier_removal": True,
            "smoothening_filter": True,
            "az_average_lines": 1000,
            "rng_average_samples": 1500,
        }
        self.ra_params_flags = {
            "outliers_kernel_size": [5, 5],
            "outliers_filter_kernel_size": [10, 10],
            "outliers_percentile_boundaries": [10, 80],
            "smoothening_order": 5,
            "smoothening_window_length": 70,
        }

    def test_radiometric_analysis_config_from_dict(self):
        """Testing PointWiseRadiometricAnalysisConfig dataclass generation from dictionary"""
        test_dict = self.ra_flags.copy()
        test_dict["parameters"] = self.ra_params_flags
        dtc = PointWiseRadiometricAnalysisConfig.from_dict(test_dict)

        for key, item in test_dict.items():
            if key != "parameters":
                dataclass_key = [field.name for field in fields(dtc) if key in field.name][0]
                value = getattr(dtc, dataclass_key)
                if key in ["output_quantity", "value", "direction", "axis"]:
                    assert item == value.name.lower()
                else:
                    assert item == value
            else:
                for key, item in self.ra_params_flags.items():
                    dataclass_key = [
                        field.name for field in fields(RadiometricAnalysisParameters) if key in field.name
                    ][0]
                    value = getattr(dtc.parameters, dataclass_key)
                    if isinstance(value, tuple):
                        assert tuple(item) == value
                    else:
                        assert item == value

    def test_radiometric_analysis_parameters_from_dict(self):
        """Testing RadiometricAnalysisParameters dataclass generation from dictionary"""
        test_dict = self.ra_params_flags.copy()
        dtc = RadiometricAnalysisParameters.from_dict(test_dict)

        for key, item in test_dict.items():
            dataclass_key = [field.name for field in fields(dtc) if key in field.name][0]
            value = getattr(dtc, dataclass_key)
            if "outlier" in key:
                assert tuple(item) == value
            else:
                assert item == value


class TestRadiometricAnalysis:
    """Testing radiometric_analysis.py core functionalities"""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.size = 128
        self.data = self._generate_gaussian_kernel(size_x=self.size, sigma_x=10)

    @staticmethod
    def _generate_gaussian_kernel(
        size_x: int, size_y: int = None, sigma_x: float = 5.0, sigma_y: float = None
    ) -> npt.NDArray[np.floating]:
        """Generating a 2D gaussian kernel.

        Parameters
        ----------
        size_x : int
            size of the kernel along x
        size_y : int, optional
            size of the kernel along y, by default None
        sigma_x : float, optional
            sigma in x direction, by default 5.0
        sigma_y : float, optional
            sigma in y direction, by default None

        Returns
        -------
        npt.NDArray[np.floating]
            2D centered gaussian kernel
        """

        if size_y is None:
            size_y = size_x
        if sigma_y is None:
            sigma_y = sigma_x

        x_axis, y_axis = np.meshgrid(
            np.linspace(-size_x // 2, size_x // 2, size_x), np.linspace(-size_y // 2, size_y // 2, size_y)
        )

        exp_part = x_axis**2 / (2 * sigma_x**2) + y_axis**2 / (2 * sigma_y**2)

        return 1 / (2 * np.pi * sigma_x * sigma_y) * np.exp(-exp_part)

    def test_extract_radiometric_profiles(self):
        """Testing _extract_radiometric_profiles function"""
        directions = [RadiometricAnalysisDirection.RANGE, RadiometricAnalysisDirection.AZIMUTH]
        smth = []
        orig = []
        for direction in directions:
            smoothed_prof, original_prof = _extract_radiometric_profiles(
                data=self.data,
                direction=direction,
                outlier=False,
                smoothening=True,
                config_params=RadiometricAnalysisParameters(),
            )
            smth.append(smoothed_prof)
            orig.append(original_prof)

        np.testing.assert_array_almost_equal(smth[0], smth[1], decimal=15)
        np.testing.assert_array_almost_equal(orig[0], orig[1], decimal=15)
        assert smth[0].size == self.size
        assert orig[0].size == self.size
        assert smth[0].size == smth[1].size
        assert orig[0].size == orig[1].size
