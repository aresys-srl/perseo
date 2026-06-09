# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for elevation_notch_analysis/support core functionalities"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import xarray as xr
from netCDF4 import Dataset

from perseo_quality.core.generic_dataclasses import SARPolarization
from perseo_quality.elevation_notch_analysis.custom_dataclasses import (
    ElevationNotchBlockInfo,
    ElevationNotchOutput,
)
from perseo_quality.elevation_notch_analysis.support import (
    InvalidAntennaPatternError,
    elevation_notch_profiles_to_netcdf,
    get_valid_antenna_pattern,
    validate_antenna_pattern,
)


class TestElevationNotchAnalysisSupport:
    """Testing Elevation Notch Analysis support functionalities"""

    @pytest.fixture(autouse=True)
    def _setup(self, antenna_pattern) -> None:
        """Testing setup"""
        self._antenna_pattern_data = antenna_pattern
        self._swath = "EN"
        self._polarization = "HH"
        self._antenna_pattern = {self._swath: {self._polarization: self._antenna_pattern_data}}

    def test_get_valid_antenna_pattern(self) -> None:
        """Getting valid antenna pattern"""
        valid_antenna_pattern = get_valid_antenna_pattern(
            antenna_pattern=self._antenna_pattern, swath=self._swath, polarization=self._polarization
        )
        xr.testing.assert_equal(valid_antenna_pattern, self._antenna_pattern_data)

    def test_get_valid_antenna_pattern_missing_swath(self) -> None:
        """Getting antenna pattern with missing swath raises error"""
        with pytest.raises(InvalidAntennaPatternError, match="does not contain swath"):
            get_valid_antenna_pattern(
                antenna_pattern=self._antenna_pattern, swath="NONEXISTENT", polarization=self._polarization
            )

    def test_get_valid_antenna_pattern_missing_polarization(self) -> None:
        """Getting antenna pattern with missing polarization raises error"""
        with pytest.raises(InvalidAntennaPatternError, match="does not contain polarization"):
            get_valid_antenna_pattern(antenna_pattern=self._antenna_pattern, swath=self._swath, polarization="VV")


class TestInvalidAntennaPatternError:
    """Testing InvalidAntennaPatternError exception"""

    def test_is_exception(self):
        """Testing InvalidAntennaPatternError is an Exception subclass"""
        assert issubclass(InvalidAntennaPatternError, Exception)

    def test_can_be_raised(self):
        """Testing InvalidAntennaPatternError can be raised with message"""
        with pytest.raises(InvalidAntennaPatternError, match="test error"):
            raise InvalidAntennaPatternError("test error")


class TestValidateAntennaPattern:
    """Testing validate_antenna_pattern function"""

    @pytest.fixture(autouse=True)
    def _setup(self, antenna_pattern) -> None:
        self._valid_ap = antenna_pattern

    def test_valid_antenna_pattern(self):
        """Testing valid antenna pattern passes validation"""
        validate_antenna_pattern(self._valid_ap)

    def test_missing_azimuth_angles(self):
        """Testing missing azimuth_angles raises error"""
        ap = xr.Dataset({"gain": (("azimuth_angles", "elevation_angles"), np.ones((10, 20)))})
        ap.coords["elevation_angles"] = np.linspace(-5, 5, 20)
        with pytest.raises(InvalidAntennaPatternError, match="azimuth_angles"):
            validate_antenna_pattern(ap)

    def test_missing_elevation_angles(self):
        """Testing missing elevation_angles raises error"""
        ap = xr.Dataset({"gain": (("azimuth_angles", "elevation_angles"), np.ones((10, 20)))})
        ap.coords["azimuth_angles"] = np.linspace(-5, 5, 10)
        with pytest.raises(InvalidAntennaPatternError, match="elevation_angles"):
            validate_antenna_pattern(ap)

    def test_missing_gain_variable(self):
        """Testing missing gain variable raises error"""
        ap = xr.Dataset({"other": (("azimuth_angles", "elevation_angles"), np.ones((10, 20)))})
        ap.coords["azimuth_angles"] = np.linspace(-5, 5, 10)
        ap.coords["elevation_angles"] = np.linspace(-5, 5, 20)
        with pytest.raises(InvalidAntennaPatternError, match="gain"):
            validate_antenna_pattern(ap)


class TestElevationNotchProfilesToNetCDF:
    """Testing elevation_notch_profiles_to_netcdf function"""

    def test_elevation_notch_profiles_to_netcdf(self):
        """Testing elevation_notch_profiles_to_netcdf"""
        n_blocks = 3
        samples_block = 100
        blocks_info = [
            ElevationNotchBlockInfo(
                block_num=i,
                first_az_line_block=i * 100,
                lines_block=100,
                samples_block=samples_block,
                altitude_m=500.0 + i * 10,
                annotated_roll_deg=-0.5 + i * 0.1,
                estimated_roll_deg=-0.45 + i * 0.1,
                antenna_profile_from_data_db=np.ones(samples_block) * (-10 + i),
                antenna_profile_from_model_db=np.ones(samples_block) * (-9 + i),
                antenna_profile_parabolic_fit_db=np.ones(samples_block) * (-8 + i),
                parabolic_fit_axis_deg=np.linspace(-1, 1, samples_block),
                parabola_minimum_deg=0.0 + i * 0.1,
                parabola_coefficients=np.array([-0.5, 0.0, 0.1]),
                antenna_angles_deg=np.linspace(-2, 2, samples_block),
                mispointing_error_deg=0.01 + i * 0.005,
                calibration_constant=1.0 + i * 0.1,
                noise_floor=0.001 + i * 0.0001,
                notch_minimum_position_deg=0.0 + i * 0.1,
            )
            for i in range(n_blocks)
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
            out_file = elevation_notch_profiles_to_netcdf(data=data, output_dir=temp_dir)
            out_path = Path(out_file)
            assert out_path.exists()
            assert out_path.is_file()
            assert out_path.name == "elevation_notch_results.nc"

            root = Dataset(out_path, "r", format="NETCDF4")
            assert root.title == "Elevation Notch Results"
            assert root.product_name == "test_product"
            assert "S1" in root.groups
            assert "HH" in root.groups["S1"].groups
            pol_grp = root.groups["S1"].groups["HH"]
            assert pol_grp.azimuth_blocks_num == n_blocks
            assert pol_grp.lines_per_block == 100
            assert pol_grp.samples_per_block == samples_block
            np.testing.assert_array_equal(pol_grp.variables["first_az_line_block"][:], [0, 100, 200])
            np.testing.assert_array_equal(pol_grp.variables["annotated_roll"][:], [-0.5, -0.4, -0.3])
            np.testing.assert_array_equal(pol_grp.variables["estimated_roll"][:], [-0.45, -0.35, -0.25])
            np.testing.assert_array_equal(pol_grp.variables["altitude"][:], [500.0, 510.0, 520.0])
            expected_angles = np.stack([b.antenna_angles_deg for b in blocks_info])
            np.testing.assert_array_equal(pol_grp.variables["antenna_angles"][:], expected_angles)
            np.testing.assert_array_equal(pol_grp.variables["parabola_minimum"][:], [0.0, 0.1, 0.2])
            assert "notch_minimum_position" in pol_grp.variables
            assert "calibration_constant" in pol_grp.variables
            assert "mispointing_error" in pol_grp.variables
            assert "noise_floor" in pol_grp.variables
            assert "antenna_profile" in pol_grp.variables
            assert "antenna_profile_model" in pol_grp.variables
            root.close()

    def test_elevation_notch_profiles_to_netcdf_without_optional(self):
        """Testing elevation_notch_profiles_to_netcdf without optional fields"""
        samples_block = 50
        blocks_info = [
            ElevationNotchBlockInfo(
                block_num=0,
                first_az_line_block=0,
                lines_block=200,
                samples_block=samples_block,
                altitude_m=500.0,
                annotated_roll_deg=-0.5,
                estimated_roll_deg=-0.45,
                antenna_profile_from_data_db=np.ones(samples_block) * -10,
                parabolic_fit_axis_deg=np.linspace(-1, 1, samples_block),
                antenna_profile_parabolic_fit_db=np.ones(samples_block) * -8,
                parabola_minimum_deg=0.0,
                parabola_coefficients=np.array([-0.5, 0.0, 0.1]),
                antenna_angles_deg=np.linspace(-2, 2, samples_block),
            )
        ]
        data = [
            ElevationNotchOutput(
                product_name="test_product",
                channel="1",
                swath="S1",
                polarization=SARPolarization.VV,
                blocks_info=blocks_info,
            )
        ]

        with TemporaryDirectory() as temp_dir:
            out_file = elevation_notch_profiles_to_netcdf(data=data, output_dir=temp_dir)
            out_path = Path(out_file)
            assert out_path.exists()

            root = Dataset(out_path, "r", format="NETCDF4")
            pol_grp = root.groups["S1"].groups["VV"]
            assert "notch_minimum_position" not in pol_grp.variables
            assert "calibration_constant" not in pol_grp.variables
            assert "mispointing_error" not in pol_grp.variables
            assert "noise_floor" not in pol_grp.variables
            assert "antenna_profile_model" not in pol_grp.variables
            root.close()
