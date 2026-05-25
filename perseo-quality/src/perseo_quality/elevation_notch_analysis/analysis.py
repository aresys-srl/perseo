# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Elevation Notch Analysis"""

from __future__ import annotations

import numpy as np
import xarray as xr
from arepytools.geometry.geometric_functions import (
    compute_incidence_angles,
    compute_look_angles,
)
from numpy.polynomial import Polynomial
from scipy.constants import speed_of_light
from scipy.optimize import least_squares

from perseo_quality.core.common import angles_computation_setup, blocks_partitioning
from perseo_quality.core.generic_dataclasses import SARRadiometricQuantity
from perseo_quality.core.signal_processing import parabolic_interp_by_3_closest_samples, radiometric_correction
from perseo_quality.elevation_notch_analysis.config import ElevationNotchConfig
from perseo_quality.elevation_notch_analysis.custom_dataclasses import (
    ElevationNotchBlockInfo,
    ElevationNotchOutput,
)
from perseo_quality.elevation_notch_analysis.support import get_valid_antenna_pattern
from perseo_quality.io.quality_input_protocol import QualityInputProduct
from perseo_quality.logger import quality_logger as log

OUTPUT_RADIOMETRIC_QUANTITY = SARRadiometricQuantity.GAMMA_NOUGHT
FILTERING_KERNEL_SIZE = 21


def elevation_notch_analysis(
    product: QualityInputProduct,
    antenna_pattern: dict[str, dict[str, xr.Dataset]] | None = None,
    config: ElevationNotchConfig | None = None,
) -> list[ElevationNotchOutput]:
    """Performing Block-Wise Elevation Notch Analysis on a given product.

    Parameters
    ----------
    product : QualityInputProduct
        object satisfying the QualityInputProduct protocol
    antenna_pattern : dict[str, dict[str, xr.Dataset]] | None, optional
        antenna pattern for the current product, by default None

        Data must be provided in the following format:

        ```python
        {
            "swath": {
                "polarization": xr.Dataset(
                    {
                        "gain": (
                            ["azimuth_angles", "elevation_angles"],
                            gain_data,  # in dB
                        ),
                        ...
                    },
                    coords={
                        "elevation_angles": elevation_angles_axis,  # in deg
                        "azimuth_angles": azimuth_angles_axis,  # in deg
                        ...
                    },
                )
            }
        }
        ```

    config : ElevationNotchConfig | None, optional
        ElevationNotchConfig configuration dataclass, by default None

    Returns
    -------
    list[ElevationNotchOutput]
        an InterferometricCoherenceOutput dataclass for each channel and each block

    Raises
    ------
    RuntimeError
        if Roll angle is not available in the input product
    """
    log.info(f"Performing Block-Wise Elevation Notch Analysis on {product.name}")
    log.info(f"Selected Product has {len(product.channels_list)} channels")
    if config is None:
        log.info("Configuration not provided, using default")
        config = ElevationNotchConfig()

    res = []
    for channel in product.channels_list:
        # recovering metadata for the current channel
        channel_data = product.get_channel_data(channel_id=channel)
        slant_range_axis = channel_data.slant_range_axis[config.range_pixel_margin : -config.range_pixel_margin]
        log.info(
            f"Analyzing Channel {channel}, Swath {channel_data.swath_name},"
            + f" Polarization {channel_data.polarization.name}..."
        )
        log.info(f"Channel Data acquisition mode is: {channel_data.acquisition_mode.name}")

        log.info("Defining blocks partitioning of the whole scene.")
        # defining scene partitioning by blocks
        az_block_size, blocks_num, blocks_centers_px = blocks_partitioning(
            azimuth_axis=channel_data.azimuth_axis,
            range_axis=channel_data.slant_range_axis,
            lines_per_burst=channel_data.lines_per_burst,
            default_block_size=config.azimuth_block_size,
        )

        output_results = ElevationNotchOutput(
            product_name=product.name,
            channel=channel,
            swath=channel_data.swath_name,
            polarization=channel_data.polarization,
        )

        blocks_info = []
        for bc_num, center in enumerate(blocks_centers_px):
            log.info(f"Processing block {bc_num + 1} of {blocks_num}")

            # TODO: these two protocol methods are specific for this analysis, they should be generalized better
            altitude_m = channel_data.get_altitude_m(channel_data.azimuth_axis[center[0]])
            roll_rad = np.deg2rad(channel_data.get_roll_angle_deg(channel_data.azimuth_axis[center[0]]))[0]
            if roll_rad is None:
                log.critical("Roll angle is not available.")
                raise RuntimeError("Roll angle is not available.")

            log.info("Computing Incidence and Look angles...")
            sensor_pos, ground_points, nadir = angles_computation_setup(
                trajectory=channel_data.trajectory,
                azimuth_time=channel_data.azimuth_axis[center[0]],
                range_values=slant_range_axis,
                look_direction=channel_data.looking_side.value,
                altitude=altitude_m,
            )
            look_angles_mid_block_rad = compute_look_angles(
                sensor_positions=sensor_pos, nadir_directions=nadir, points=ground_points
            )
            antenna_angles_rad = look_angles_mid_block_rad - roll_rad
            incidence_angles_mid_block_rad = compute_incidence_angles(sensor_positions=sensor_pos, points=ground_points)

            cropping_size = (
                channel_data.slant_range_axis.size,
                az_block_size[bc_num],
            )
            # reading block, without converting the data here, applying radiometric conversion in the next steps
            log.info("Reading azimuth block data and discarding lines with no data...")
            target_area = channel_data.read_data(
                azimuth_index=center[0],
                range_index=center[1],
                cropping_size=cropping_size,
                output_radiometric_quantity=channel_data.radiometric_quantity,
            )
            # removing range margin at near and far range
            target_area = abs(target_area[config.range_pixel_margin : -config.range_pixel_margin, :] ** 2)
            # taking only lines with more than 50% of non-zero values
            target_area = target_area[:, np.count_nonzero(target_area, axis=0) > target_area.shape[0] / 2]
            if channel_data.radiometric_quantity != OUTPUT_RADIOMETRIC_QUANTITY:
                log.info(
                    f"Converting data from {channel_data.radiometric_quantity.name.lower().replace('_', ' ')} to "
                    + f"{OUTPUT_RADIOMETRIC_QUANTITY.name.lower().replace('_', ' ')}."
                )
                target_area = radiometric_correction(
                    data=target_area,
                    incidence_angle=incidence_angles_mid_block_rad,
                    input_quantity=channel_data.radiometric_quantity,
                    output_quantity=OUTPUT_RADIOMETRIC_QUANTITY,
                    exp_power=1,  # NOTE: power data is processed with 1 as exponent
                )

            log.info("Computing average profiles...")
            antenna_profile_from_data = np.nanmean(target_area, axis=1)

            log.info("Applying low pass filter...")
            cutoff_boundaries = [FILTERING_KERNEL_SIZE // 2, -(FILTERING_KERNEL_SIZE - FILTERING_KERNEL_SIZE // 2) + 1]
            antenna_profile_from_data = np.convolve(
                antenna_profile_from_data, np.ones(FILTERING_KERNEL_SIZE) / FILTERING_KERNEL_SIZE, mode="valid"
            )
            antenna_angles_rad = antenna_angles_rad[cutoff_boundaries[0] : cutoff_boundaries[1]]

            mispointing_error_rad, calibration_constant, noise_floor = None, None, None
            antenna_pattern_model_profile = None
            notch_min_pos_deg = None
            if antenna_pattern is not None:
                log.info(
                    f"Getting valid antenna pattern for {channel_data.swath_name}, {channel_data.polarization.name}..."
                )
                valid_antenna_pattern = get_valid_antenna_pattern(
                    antenna_pattern=antenna_pattern,
                    swath=channel_data.swath_name,
                    polarization=channel_data.polarization.name,
                )
                min_antenna_pos = int(abs(valid_antenna_pattern["elevation_angles"]).data.argmin())
                _, delta_min_pos = parabolic_interp_by_3_closest_samples(
                    valid_antenna_pattern["gain"].data.squeeze()[min_antenna_pos - 1 : min_antenna_pos + 2]
                )
                notch_min_pos_deg = np.interp(
                    min_antenna_pos + delta_min_pos,
                    np.arange(0, valid_antenna_pattern["elevation_angles"].size),
                    valid_antenna_pattern["elevation_angles"].data.squeeze(),
                )
                log.info("Normalizing antenna pattern elevation profile...")
                normalized_antenna_pattern, elevation_angle_rad_at_max_pos = antenna_pattern_normalization(
                    antenna_pattern=valid_antenna_pattern, antenna_angles_rad=antenna_angles_rad
                )
                log.info("Normalizing profile around antenna pattern maximum...")
                normalized_profile = profile_normalization(
                    profile=antenna_profile_from_data,
                    antenna_angles_rad=antenna_angles_rad,
                    elevation_angle_rad_at_max_pos=elevation_angle_rad_at_max_pos,
                )
                noise_floor_guess = min(normalized_profile)
                log.info("Computing Antenna Pattern pointing mismatch...")

                spread_loss_rng_axis_m = slant_range_axis[cutoff_boundaries[0] : cutoff_boundaries[1]] * (
                    speed_of_light / 2
                )
                spread_loss = (spread_loss_rng_axis_m / spread_loss_rng_axis_m[0]) ** 3
                noise_profile = radiometric_correction(
                    data=spread_loss.reshape(-1, 1),
                    incidence_angle=incidence_angles_mid_block_rad[cutoff_boundaries[0] : cutoff_boundaries[1]],
                    input_quantity=channel_data.radiometric_quantity,
                    output_quantity=OUTPUT_RADIOMETRIC_QUANTITY,
                    exp_power=1,
                ).squeeze()
                initial_guess = [0, 1, noise_floor_guess]
                log.info("Least-Square optimization Antenna Model vs Data profile to find mispointing error...")
                mispointing_error_rad, calibration_constant, noise_floor = antenna_pattern_pointing_mismatch(
                    antenna_pattern=normalized_antenna_pattern,
                    data_profile=normalized_profile,
                    antenna_angles_from_data_rad=antenna_angles_rad,
                    noise_profile=noise_profile,
                    initial_guess=initial_guess,
                )
                antenna_pattern_model_profile = (
                    calibration_constant
                    * np.interp(
                        np.rad2deg(antenna_angles_rad - mispointing_error_rad),
                        normalized_antenna_pattern["elevation_angles"].data,
                        normalized_antenna_pattern["linear_gain"].data.squeeze(),
                    )
                    ** 2
                    + noise_floor * noise_profile
                )
            log.info("Computing profile parabolic fit...")
            parabolic_fit_axis, parabolic_fit, parabola_minimum_rad, parabola_coefficients = (
                compute_parabolic_profile_fit(
                    profile=normalized_profile if antenna_pattern is not None else antenna_profile_from_data,
                    off_boresight_angles_rad=antenna_angles_rad,
                )
            )
            info = ElevationNotchBlockInfo(
                block_num=bc_num,
                first_az_line_block=int(center[0] - np.floor(cropping_size[1] / 2)),
                lines_block=target_area.shape[1],
                samples_block=antenna_angles_rad.size,
                altitude_m=altitude_m,
                annotated_roll_deg=-float(np.rad2deg(roll_rad)),
                estimated_roll_deg=float(np.rad2deg(parabola_minimum_rad - roll_rad)),
                antenna_profile_parabolic_fit_db=10 * np.log10(parabolic_fit),
                parabolic_fit_axis_deg=np.rad2deg(parabolic_fit_axis + roll_rad),
                parabola_minimum_deg=np.rad2deg(parabola_minimum_rad),
                parabola_coefficients=parabola_coefficients,
                antenna_angles_deg=np.rad2deg(antenna_angles_rad),
                mispointing_error_deg=float(np.rad2deg(mispointing_error_rad))
                if mispointing_error_rad is not None
                else None,
                calibration_constant=float(calibration_constant) if calibration_constant is not None else None,
                noise_floor=float(noise_floor) if noise_floor is not None else None,
                notch_minimum_position_deg=float(notch_min_pos_deg) if notch_min_pos_deg is not None else None,
            )
            info.antenna_profile_from_data_db = 10 * np.log10(
                normalized_profile if antenna_pattern is not None else antenna_profile_from_data
            )
            if antenna_pattern is not None:
                info.antenna_profile_from_model_db = 10 * np.log10(antenna_pattern_model_profile)
            blocks_info.append(info)

        output_results.blocks_info = blocks_info
        res.append(output_results)
    return res


def compute_parabolic_profile_fit(
    profile: np.ndarray,
    off_boresight_angles_rad: np.ndarray,
    fitting_interval_rad: float | None = None,
    smoothing_kernel_size: int = 201,
) -> tuple[np.ndarray, np.ndarray, float, list[float]]:
    """Computing a parabolic fit for the given profile and off-boresight angles around the minimum.

    Parameters
    ----------
    profile : np.ndarray
        azimuth block average profile
    off_boresight_angles_rad : np.ndarray
        antenna off boresight angles in radians
    fitting_interval_rad : float | None, optional
        fitting interval in radians, if not provided 0.3 deg in radians is used, by default None
    smoothing_kernel_size : int, optional
        smoothing kernel size, by default 201

    Returns
    -------
    np.ndarray
        antenna off boresight angles axis for the parabolic fit, in radians
    np.ndarray
        parabolic fit values
    float
        parabola minimum in radians
    list[float]
        parabola coefficients
    """
    if fitting_interval_rad is None:
        fitting_interval_rad = np.deg2rad(0.3)

    kernel = np.ones(smoothing_kernel_size) / smoothing_kernel_size

    data_minimum_idx = np.convolve(profile, kernel, mode="same").argmin()

    # boolean mask for interval around minimum
    indexes_to_fit = (off_boresight_angles_rad >= off_boresight_angles_rad[data_minimum_idx] - fitting_interval_rad) & (
        off_boresight_angles_rad <= off_boresight_angles_rad[data_minimum_idx] + fitting_interval_rad
    )
    parabolic_fit_axis = off_boresight_angles_rad[indexes_to_fit]

    # fitting a parabola to the profile
    parabolic_fit_poly = Polynomial.fit(parabolic_fit_axis, profile[indexes_to_fit], deg=2).convert()
    # computing parabola minimum as -b / (2a)
    parabola_minimum = -parabolic_fit_poly.coef[1] / (2 * parabolic_fit_poly.coef[2])

    return parabolic_fit_axis, parabolic_fit_poly(parabolic_fit_axis), parabola_minimum, parabolic_fit_poly.coef


def antenna_pattern_normalization(
    antenna_pattern: xr.Dataset, antenna_angles_rad: np.ndarray
) -> tuple[xr.Dataset, float]:
    """Antenna pattern linearization and normalization in the data region.

    Parameters
    ----------
    antenna_pattern : xr.Dataset
        antenna pattern for the current swath and polarization
    antenna_angles_rad : np.ndarray
        data antenna angles in radians

    Returns
    -------
    xr.Dataset
        antenna pattern with linearized and normalized gain
    float
        elevation angle in radians at antenna pattern maximum in data region
    """
    # considering only antenna pattern inside data angles region
    mask = (antenna_pattern["elevation_angles"] > np.rad2deg(antenna_angles_rad.min())) & (
        antenna_pattern["elevation_angles"] < np.rad2deg(antenna_angles_rad.max())
    )
    masked_gain = antenna_pattern["gain"].where(mask, drop=True)
    elevation_angle_at_max_pos = antenna_pattern["elevation_angles"].where(mask, drop=True)[
        int(masked_gain.data.argmax())
    ]
    antenna_pattern_data_region_max_lin = 10 ** (masked_gain.max() / 20)

    # normalizing full pattern to data region maximum
    linear_normalized_gain = 10 ** (antenna_pattern["gain"] / 20) / antenna_pattern_data_region_max_lin
    antenna_pattern_lin = antenna_pattern.drop_vars("gain").assign(linear_gain=linear_normalized_gain)
    return antenna_pattern_lin, np.deg2rad(elevation_angle_at_max_pos.item())


def profile_normalization(
    profile: np.ndarray,
    antenna_angles_rad: np.ndarray,
    elevation_angle_rad_at_max_pos: float,
    mask_margin_rad: float = 0.001745,
) -> np.ndarray:
    """Normalize the profile to the antenna pattern maximum.

    Parameters
    ----------
    profile : np.ndarray
        profile to be normalized
    antenna_angles_rad : np.ndarray
        data antenna angles in radians
    elevation_angle_rad_at_max_pos : float
        elevation angle in radians at antenna pattern maximum in data region
    mask_margin_rad : float, optional
        margin in radians to be applied to the mask, by default 0.001745

    Returns
    -------
    np.ndarray
        normalized profile at antenna pattern maximum
    """
    # normalize data around pattern maximum
    mask = (antenna_angles_rad >= (elevation_angle_rad_at_max_pos - mask_margin_rad)) & (
        antenna_angles_rad <= (elevation_angle_rad_at_max_pos + mask_margin_rad)
    )

    return profile / np.nanmean(profile[mask])


def residuals(
    params: list[float, float, float],
    data_profile: np.ndarray,
    antenna_pattern: xr.Dataset,
    noise_profile: np.ndarray,
    antenna_angles_from_data_rad: np.ndarray,
) -> np.ndarray:
    """Residuals function for the Least Squares optimization. It represents the difference between the data extracted
    profile and the antenna model profile, to be minimized. Parameters of optimization are
    $\\theta_{\\text{mis}})$ (mispointing angle), $k$ (gain) and $f(\\theta_{\\text{off}})$ (noise
    floor).

    $$
    data_profile - k \\dot p(\\theta_{\\text{off}} - \\theta_{\\text{mis}}) + noise \\dot f(\theta_{\text{off}})
    $$

    Parameters
    ----------
    params : list[float, float, float]
        model parameters residuals
    data_profile : np.ndarray
        profile pattern extracted from data
    antenna_pattern : xr.Dataset
        antenna pattern elevation profile
    noise_profile : np.ndarray
        noise profile as part of the model to be optimized
    antenna_angles_from_data_rad : np.ndarray
        antenna angles in radians from the data

    Returns
    -------
    np.ndarray
        residuals of data profile and antenna model
    """
    abs_pattern_interp = antenna_pattern["linear_gain"].interp(
        elevation_angles=np.rad2deg(antenna_angles_from_data_rad - params[0]),
        method="cubic",
        kwargs={"fill_value": "extrapolate"},
    )
    return data_profile - (params[1] * abs_pattern_interp.data.squeeze() ** 2 + params[2] * noise_profile)


def antenna_pattern_pointing_mismatch(
    antenna_pattern: xr.Dataset,
    data_profile: np.ndarray,
    noise_profile: np.ndarray,
    antenna_angles_from_data_rad: np.ndarray,
    initial_guess: list[float, float, float],
) -> tuple[float, float, float]:
    """Computing the pointing mismatch between a given data set and the pattern obtained from the
    antenna model. Optimizing the discrepancy between the data and the model to determine the best fit parameters.

    Function to be optimized using Least Squares method in the [`residuals`][] function.

    Parameters
    ----------
    antenna_pattern : xr.Dataset
        antenna model elevation pattern for the current swath and polarization
    data_profile : np.ndarray
        profile pattern extracted from data
    noise_profile : np.ndarray
        noise profile extracted from data
    antenna_angles_from_data_rad : np.ndarray
        antenna angles in radians from the data
    initial_guess : list[float, float, float]
        initial guess for the least squares optimization

    Returns
    -------
    float
        mispointing angle error in radians
    float
        calibration constant
    float
        noise floor

    Raises
    ------
    RuntimeError
        if optimization fails
    """

    init_guess = np.array(initial_guess)
    res = least_squares(
        residuals,
        init_guess,
        args=(data_profile, antenna_pattern, noise_profile, antenna_angles_from_data_rad),
        bounds=[(-np.inf, 0, 0), (np.inf, np.inf, np.inf)],
    )
    log.debug(f"Result of the optimization: {res.x}")
    log.debug(f"Success of the least square solver: {res.success}, {res.message}")
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")
    log.info(f"Least square solver: cost {res.cost}")

    return res.x[0], res.x[1], res.x[2]
