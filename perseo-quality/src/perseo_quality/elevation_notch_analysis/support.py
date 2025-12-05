# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Elevation Notch Analysis support functionalities"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from netCDF4 import Dataset

from perseo_quality.elevation_notch_analysis.custom_dataclasses import ElevationNotchOutput


class InvalidAntennaPatternError(Exception):
    """Exception raised when antenna pattern is not valid."""


def _get_case_insensitive(d, key):
    """Getting a value from a dictionary with a case insensitive key."""
    key_lower = key.lower()
    for k, v in d.items():
        if k.lower() == key_lower:
            return v
    return None


def validate_antenna_pattern(antenna_pattern: xr.Dataset) -> None:
    """Checking antenna pattern validity.

    Parameters
    ----------
    antenna_pattern : xr.Dataset
        antenna pattern dataset

    Raises
    ------
    InvalidAntennaPatternError
        if antenna pattern is not valid
    """
    if "azimuth_angles" not in antenna_pattern.coords:
        raise InvalidAntennaPatternError("Antenna pattern does not contain 'azimuth_angles' coordinate")
    if "elevation_angles" not in antenna_pattern.coords:
        raise InvalidAntennaPatternError("Antenna pattern does not contain 'elevation_angles' coordinate")
    if "gain" not in antenna_pattern.variables:
        raise InvalidAntennaPatternError("Antenna pattern does not contain 'gain' variable")


def get_valid_antenna_pattern(
    antenna_pattern: dict[str, dict[str, xr.Dataset]], swath: str, polarization: str
) -> xr.Dataset:
    """Getting the valid antenna pattern for the given swath and polarization from the antenna pattern dataset provided.

    Parameters
    ----------
    antenna_pattern : dict[str, dict[str, xr.Dataset]]
        antenna pattern dataset, as a dictionary with: swath/pol hierarchy
    swath : str
        swath for the current channel
    polarization : str
        polarization for the current channel

    Returns
    -------
    xr.Dataset
        antenna pattern data set for the given swath and polarization
    """
    swath_dict = _get_case_insensitive(antenna_pattern, swath)
    if swath_dict is None:
        raise InvalidAntennaPatternError(f"Antenna pattern does not contain swath {swath}")
    pol_dict = _get_case_insensitive(swath_dict, polarization)
    if pol_dict is None:
        raise InvalidAntennaPatternError(
            f"Antenna pattern for swath {swath} does not contain polarization {polarization}"
        )
    validate_antenna_pattern(pol_dict)
    return pol_dict


def elevation_notch_profiles_to_netcdf(data: list[ElevationNotchOutput], output_dir: str | Path) -> Path:
    """Saving Elevation Notch analysis results to NetCDF file.

    Hierarchy::

        root/
        └── swath
            └── polarization
                ├── first_az_line_block
                ├── annotated_roll
                ├── estimated_roll
                ├── notch_minimum_position  # optional
                ├── altitude
                ├── calibration_constant  # optional
                ├── mispointing_error  # optional
                ├── noise_floor  # optional
                ├── parabola_minimum
                ├── parabola_coefficients
                ├── antenna_angles
                ├── antenna_profile_model  # optional
                └── antenna_profile

    Some fields are optional and will be saved only if the Elevation Notch analysis was performed using external
    Antenna Pattern data.

    Parameters
    ----------
    data : list[ElevationNotchOutput]
        list of ElevationNotchOutput results dataclasses
    output_dir : str | Path
        Path to the output directory where to save the NetCDF file

    Returns
    -------
    Path
        path to the NetCDF file
    """
    output_dir = Path(output_dir)
    output_file = output_dir.joinpath("elevation_notch_results.nc")
    root = Dataset(output_file, "w", format="NETCDF4")

    root.title = "Elevation Notch Results"
    root.product_name = data[0].product_name

    swath_grp = root.createGroup(data[0].swath)
    for element in data:
        assert element.swath == swath_grp.name
        assert element.polarization not in swath_grp.groups.keys()
        pol_grp = swath_grp.createGroup(element.polarization.name)
        pol_grp.azimuth_blocks_num = len(element.blocks_info)
        pol_grp.lines_per_block = element.blocks_info[0].lines_block
        pol_grp.samples_per_block = element.blocks_info[0].samples_block

        pol_grp.createDimension("az_blocks", pol_grp.azimuth_blocks_num)
        pol_grp.createDimension("elevation", pol_grp.samples_per_block)
        pol_grp.createDimension("coefficients", 3)

        first_az_line_blk = np.array([e.first_az_line_block for e in element.blocks_info])
        first_az_line_block = pol_grp.createVariable("first_az_line_block", first_az_line_blk.dtype, ("az_blocks",))
        first_az_line_block[:] = first_az_line_blk

        roll = np.array([e.annotated_roll_deg for e in element.blocks_info])
        annotated_roll = pol_grp.createVariable("annotated_roll", roll.dtype, ("az_blocks",))
        annotated_roll[:] = roll
        annotated_roll.units = "deg"

        est_roll = np.array([e.estimated_roll_deg for e in element.blocks_info])
        estimated_roll = pol_grp.createVariable("estimated_roll", est_roll.dtype, ("az_blocks",))
        estimated_roll[:] = est_roll
        estimated_roll.units = "deg"

        altitude = np.array([e.altitude_m for e in element.blocks_info])
        altitude_m = pol_grp.createVariable("altitude", altitude.dtype, ("az_blocks",))
        altitude_m[:] = altitude
        altitude_m.units = "m"

        ant_angles = np.array([e.antenna_angles_deg for e in element.blocks_info])
        antenna_angles = pol_grp.createVariable("antenna_angles", ant_angles.dtype, ("az_blocks", "elevation"))
        antenna_angles[:] = ant_angles
        antenna_angles.units = "deg"

        parabola_min = np.array([e.parabola_minimum_deg for e in element.blocks_info])
        parabola_minimum = pol_grp.createVariable("parabola_minimum", parabola_min.dtype, ("az_blocks",))
        parabola_minimum[:] = parabola_min
        parabola_minimum.units = "deg"

        parabola_coeff = np.array([e.parabola_coefficients for e in element.blocks_info])
        parabola_coefficients = pol_grp.createVariable(
            "parabola_coefficients", parabola_coeff.dtype, ("az_blocks", "coefficients")
        )
        parabola_coefficients[:] = parabola_coeff

        if element.blocks_info[0].notch_minimum_position_deg is not None:
            notch_min_pos = np.array([e.notch_minimum_position_deg for e in element.blocks_info])
            notch_minimum_position = pol_grp.createVariable(
                "notch_minimum_position", notch_min_pos.dtype, ("az_blocks",)
            )
            notch_minimum_position[:] = notch_min_pos
            notch_minimum_position.units = "deg"

        if element.blocks_info[0].calibration_constant is not None:
            cal_const = np.array([e.calibration_constant for e in element.blocks_info])
            calibration_constant = pol_grp.createVariable("calibration_constant", cal_const.dtype, ("az_blocks",))
            calibration_constant[:] = cal_const

        if element.blocks_info[0].mispointing_error_deg is not None:
            pointing_error = np.array([e.mispointing_error_deg for e in element.blocks_info])
            mispointing_error = pol_grp.createVariable("mispointing_error", pointing_error.dtype, ("az_blocks",))
            mispointing_error[:] = pointing_error
            mispointing_error.units = "deg"

        if element.blocks_info[0].noise_floor is not None:
            noise = np.array([e.noise_floor for e in element.blocks_info])
            noise_floor = pol_grp.createVariable("noise_floor", noise.dtype, ("az_blocks",))
            noise_floor[:] = noise

        if element.blocks_info[0].antenna_profile_from_data_db is not None:
            ant_profile = np.array([e.antenna_profile_from_data_db for e in element.blocks_info])
            antenna_profile = pol_grp.createVariable("antenna_profile", ant_profile.dtype, ("az_blocks", "elevation"))
            antenna_profile[:] = ant_profile
            antenna_profile.units = "dB"

        if element.blocks_info[0].antenna_profile_from_model_db is not None:
            ant_profile_model = np.array([e.antenna_profile_from_model_db for e in element.blocks_info])
            antenna_profile_model = pol_grp.createVariable(
                "antenna_profile_model", ant_profile_model.dtype, ("az_blocks", "elevation")
            )
            antenna_profile_model[:] = ant_profile_model
            antenna_profile_model.units = "dB"

    root.close()
    return output_file
