# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Block-Wise Radiometric Analysis support functionalities"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from numpy.polynomial import Polynomial
from scipy.signal import convolve2d

from perseo_quality.radiometric_analysis.block_wise.config import Radiometric2DHistogramParameters
from perseo_quality.radiometric_analysis.custom_dataclasses import RadiometricProfilesOutput


def radiometric_profiles_to_netcdf(
    data: list[RadiometricProfilesOutput], out_path: str | Path, tag: str | None = None
) -> Path:
    """Saving Radiometric Profiles output data to NetCDF4 file.

    Hierarchy::

        root/
        ├── product_attributes...
        └── swath
            └── polarization
                ├── channel_attributes...
                ├── incidence_angles
                ├── look_angles  [optional]
                ├── azimuth_times  [optional]
                └── radiometric_profiles

    Parameters
    ----------
    data : list[RadiometricProfilesOutput]
        list of RadiometricProfilesOutput dataclass, corresponding to the full output of the radiometric analysis
    out_path : str | Path
        path where to save the NetCDF file
    tag : str | None, optional
        tag string to be added to the output filename, by default None

    Returns
    -------
    Path
        path to the output netCDF file
    """
    tag = "radiometric" if tag is None else tag
    out_name = tag + "_profiles_" + data[0].general_info.product
    out_path = Path(out_path)
    output_file = out_path.joinpath(out_name).with_suffix(".nc")

    root = Dataset(output_file, "w", format="NETCDF4")
    root.product = data[0].general_info.product
    root.sensor = data[0].general_info.sensor
    root.product_type = data[0].general_info.product_type
    root.acquisition_mode = data[0].general_info.acquisition_mode
    root.orbit_direction = data[0].general_info.orbit_direction
    root.acquisition_start_time = str(data[0].general_info.acquisition_start_time)
    root.direction = data[0].direction.name.lower()
    root.output_radiometric_quantity = data[0].general_info.radiometric_quantity

    for item in data:
        if item.general_info.swath not in root.groups:
            swath_grp = root.createGroup(item.general_info.swath)
        else:
            swath_grp = root.groups[item.general_info.swath]
        if item.general_info.polarization not in swath_grp.groups:
            pol_grp = swath_grp.createGroup(item.general_info.polarization)
        else:
            pol_grp = swath_grp.groups[item.general_info.polarization]
        pol_grp.swath = item.general_info.swath
        pol_grp.channel = item.general_info.channel
        pol_grp.polarization = item.general_info.polarization
        pol_grp.azimuth_blocks_num = item.blocks_num
        pol_grp.azimuth_block_centers = [str(d) for d in item.azimuth_block_centers]
        pol_grp.range_block_centers = item.range_block_centers

        # creating common dimensions
        pol_grp.createDimension("samples", item.profiles.shape[1])
        pol_grp.createDimension("azimuth_blocks", item.blocks_num)

        # creating elevation angles variable
        angles_axis = pol_grp.createVariable(
            "incidence_angles", item.incidence_angles.dtype, ("azimuth_blocks", "samples")
        )
        angles_axis.unit = "deg"
        angles_axis[:] = item.incidence_angles

        # creating elevation angles variable
        if item.look_angles is not None:
            data_axis = pol_grp.createVariable("look_angles", item.look_angles.dtype, ("azimuth_blocks", "samples"))
            data_axis.unit = "deg"
            data_axis[:] = item.look_angles

        if item.block_azimuth_times is not None:
            data_axis = pol_grp.createVariable(
                "azimuth_times", item.block_azimuth_times.dtype, ("azimuth_blocks", "samples")
            )
            data_axis.unit = "s"
            data_axis[:] = item.block_azimuth_times

        # creating nesz profile variable
        profs = pol_grp.createVariable("radiometric_profiles", item.profiles.dtype, ("azimuth_blocks", "samples"))
        profs.unit = "dB"
        profs[:] = item.profiles

    root.close()

    return output_file


def compute_2d_histogram(
    x_data: np.ndarray, y_data: np.ndarray, x_axis: np.ndarray, config: Radiometric2DHistogramParameters
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 2D histogram from input data.

    Parameters
    ----------
    x_data : np.ndarray
        data along the selected x axis
    y_data : np.ndarray
        data along the selected y axis
    x_axis : np.ndarray
        histogram x axis
    config : Radiometric2DHistogramParameters
        configuration parameters for the 2D histogram

    Returns
    -------
    np.ndarray
        2D histogram
    np.ndarray
        x bins axis
    np.ndarray
        y bins axis
    """

    assert config.y_bins_center_margin is not None
    assert config.y_bins_num is not None
    assert config.x_bins_step is not None

    # bins axis generation
    y_bins_center = np.nanmean(y_data)

    y_bins = np.linspace(
        start=y_bins_center - config.y_bins_center_margin,
        stop=y_bins_center + config.y_bins_center_margin,
        num=config.y_bins_num,
    )
    x_bins = x_axis[:: config.x_bins_step]

    # 2D histogram generation
    hist, _, _ = np.histogram2d(
        x=x_data.ravel(),
        y=y_data.ravel(),
        bins=[x_bins, y_bins],
    )
    hist = hist.T

    return hist, x_bins, y_bins


def masking_outliers_by_percentiles(
    data: np.ndarray, kernel: tuple[int, int], percentile_boundaries: tuple[int, int]
) -> np.ndarray:
    """Masking outliers outside of provided percentile boundaries setting them to NaN.

    Parameters
    ----------
    data : np.ndarray
        input 2D array
    kernel : tuple[int, int]
        kernel size, height and width in pixels
    percentile_boundaries : tuple[int, int]
        data below percentile_boundaries[0] and above percentile_boundaries[1] are set to NaN

    Returns
    -------
    np.ndarray
        input array with NaN where outliers lie
    """
    filter_kernel = np.ones(kernel)
    masking_cond = np.logical_or(
        data < np.nanpercentile(data.ravel(), percentile_boundaries[0]),
        data > np.nanpercentile(data.ravel(), percentile_boundaries[1]),
    ).astype("int64")

    # convolving data with filter kernel
    mask = np.round(convolve2d(masking_cond, filter_kernel, mode="same") / np.sum(filter_kernel))

    # masking out data
    data[np.where(mask)] = np.nan

    return data


def compute_profile_variability_index(profile: np.ndarray, look_angles_deg: np.ndarray) -> tuple[float, float]:
    """Computing radiometric variability index for the current profile, with respect to the look angles axis.

    Parameters
    ----------
    profile : np.ndarray
        current radiometric profile in [dB]
    look_angles_deg : np.ndarray
        look angles axis of the provided profile in degrees

    Returns
    -------
    float
        slope with respect to look angles axis in [dB/deg]
    float
        radiometric variability index in [dB]
    """
    # linear fit
    linear_fit_params = Polynomial.fit(look_angles_deg[~profile.mask], profile.compressed(), deg=1).convert()

    # homogeneity index
    regression_line = linear_fit_params.coef[0] + linear_fit_params.coef[1] * look_angles_deg
    radiometric_profiles_de_sloped = profile - regression_line
    variability_index = np.percentile(radiometric_profiles_de_sloped.compressed(), 90) - np.percentile(
        radiometric_profiles_de_sloped.compressed(), 10
    )
    return float(linear_fit_params.coef[1]), float(variability_index)


def radiometric_statistical_analysis_to_df(data: list[RadiometricProfilesOutput]) -> pd.DataFrame:
    """Converting statistical radiometric output to pandas DataFrame.

    Parameters
    ----------
    data : list[RadiometricProfilesOutput]
        radiometric statistics

    Returns
    -------
    pd.DataFrame
        dataframe with radiometric statistics
    """
    item_df = []
    for item in data:
        kpi_info = [asdict(blk) for blk in item.kpi]
        general_info = [asdict(item.general_info)] * len(kpi_info)
        item_df.append(pd.concat([pd.DataFrame(general_info), pd.DataFrame(kpi_info)], axis=1))
    return pd.concat(item_df).reset_index(drop=True)
