# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Block-Wise Radiometric Analysis support functionalities"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from arepytools.geometry.conversions import llh2xyz, xyz2llh
from arepytools.geometry.curve_protocols import TwiceDifferentiable3DCurve
from arepytools.geometry.direct_geocoding import GeocodingSide, direct_geocoding_monostatic
from arepytools.timing.precisedatetime import PreciseDateTime
from netCDF4 import Dataset
from numpy.polynomial import Polynomial
from scipy.signal import convolve2d

from perseo_quality.logger import quality_logger as log
from perseo_quality.radiometric_analysis.block_wise.config import Radiometric2DHistogramParameters
from perseo_quality.radiometric_analysis.custom_dataclasses import RadiometricProfilesOutput


def radiometric_profiles_to_netcdf(
    data: RadiometricProfilesOutput, out_path: str | Path, tag: str | None = None
) -> Path:
    """Saving Radiometric Profiles output data to NetCDF4 file.

    Parameters
    ----------
    data : RadiometricProfilesOutput
        RadiometricProfilesOutput dataclass
    out_path : str | Path
        path where to save the NetCDF file
    tag : str | None, optional
        tag string to be added to the output filename, by default None

    Returns
    -------
    Path
        path to the output netCDF file
    """
    out_path = Path(out_path)
    tag = "radiometric" if tag is None else tag

    out_name = tag + "_profiles_" + data.general_info.swath + "_" + data.general_info.polarization
    log.info(f"Saving {out_name} data to NetCDF file.")

    root = Dataset(out_path.joinpath(out_name).with_suffix(".nc"), "w", format="NETCDF4")
    root.swath = data.general_info.swath
    root.channel = data.general_info.channel
    root.polarization = data.general_info.polarization
    root.direction = data.direction.name.lower()
    root.output_radiometric_quantity = data.general_info.radiometric_quantity
    root.azimuth_blocks_num = data.blocks_num
    root.azimuth_block_centers = [str(d) for d in data.azimuth_block_centers]
    root.range_block_centers = data.range_block_centers

    # creating common dimensions
    root.createDimension("samples", data.profiles.shape[1])
    root.createDimension("azimuth_blocks", data.blocks_num)

    # creating elevation angles variable
    angles_axis = root.createVariable("incidence_angles", data.incidence_angles.dtype, ("azimuth_blocks", "samples"))
    angles_axis.unit = "deg"
    angles_axis[:] = data.incidence_angles

    # creating elevation angles variable
    if data.look_angles is not None:
        data_axis = root.createVariable("look_angles", data.look_angles.dtype, ("azimuth_blocks", "samples"))
        data_axis.unit = "deg"
        data_axis[:] = data.look_angles

    if data.block_azimuth_times is not None:
        data_axis = root.createVariable("azimuth_times", data.block_azimuth_times.dtype, ("azimuth_blocks", "samples"))
        data_axis.unit = "s"
        data_axis[:] = data.block_azimuth_times

    # creating nesz profile variable
    profs = root.createVariable("radiometric_profiles", data.profiles.dtype, ("azimuth_blocks", "samples"))
    profs.unit = "dB"
    profs[:] = data.profiles

    root.close()

    return out_path.joinpath(out_name).with_suffix(".nc")


def angles_computation_setup(
    trajectory: TwiceDifferentiable3DCurve,
    azimuth_time: PreciseDateTime,
    range_values: np.ndarray,
    look_direction: GeocodingSide | str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Setting up the stage to compute incidence and look angles by computing sensor position, ground points and nadir
    direction.

    Parameters
    ----------
    trajectory : TwiceDifferentiable3DCurve
        sensor trajectory
    azimuth_time : PreciseDateTime
        azimuth time at which compute the output
    range_values : np.ndarray
        range values for which compute values
    look_direction : GeocodingSide | str
        sensor look direction

    Returns
    -------
    np.ndarray
        sensor position
    np.ndarray
        ground points
    np.ndarray
        nadir direction
    """
    look_direction = GeocodingSide(look_direction)
    sensor_pos = trajectory.evaluate(azimuth_time)
    sensor_vel = trajectory.evaluate_first_derivatives(azimuth_time)

    ground_points = direct_geocoding_monostatic(
        sensor_positions=sensor_pos,
        sensor_velocities=sensor_vel,
        range_times=range_values,
        geocoding_side=look_direction.value,
        frequencies_doppler_centroid=0,
        wavelength=1,
        geodetic_altitude=0,
    )

    sensor_position_ground = xyz2llh(sensor_pos)
    sensor_position_ground[2] = 0.0
    sensor_position_ground = llh2xyz(sensor_position_ground).squeeze()

    nadir = sensor_position_ground - sensor_pos
    return sensor_pos, ground_points, nadir


def blocks_definition(
    azimuth_axis: np.ndarray,
    range_axis: np.ndarray,
    lines_per_burst: np.ndarray,
    default_block_size: int,
) -> tuple[int, int, list[tuple[int, int]]]:
    """Defining the blocks partitioning of the whole scene.

    Parameters
    ----------
    azimuth_axis : np.ndarray
        azimuth axis of the whole scene
    range_axis : np.ndarray
        range axis of the whole scene
    lines_per_burst : np.ndarray
        lines per burst array
    default_block_size : int
        default block size value, needed for stripmap case

    Returns
    -------
    int
        size of each block
    int
        number of partitioning blocks
    list[tuple[int, int]]
        pixel coordinates of blocks centers (azimuth and range pixel values)
    """
    block_size = default_block_size
    blocks_num = int(np.floor(azimuth_axis.size / block_size))
    mid_range_pixel = int(range_axis.size // 2)

    if lines_per_burst.size > 1:
        # TOPSAR/SCANSAR case: blocks set using bursts
        block_size = lines_per_burst[0]
        blocks_num = lines_per_burst.size  # number of bursts

    blocks_centers_px = np.arange(block_size // 2, block_size * blocks_num, block_size).tolist()
    blocks_centers_px = [(px, mid_range_pixel) for px in blocks_centers_px]

    return block_size, blocks_num, blocks_centers_px


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
    linear_fit_params = Polynomial.fit(look_angles_deg, profile, deg=1).convert()

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
