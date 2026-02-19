# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Spectral Analysis support functionalities"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from arepytools.timing.precisedatetime import PreciseDateTime
from netCDF4 import Dataset
from numpy.polynomial import Polynomial
from scipy.constants import speed_of_light
from scipy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, spectrogram
from scipy.signal.windows import hamming

from perseo_quality import __version__
from perseo_quality.core.generic_dataclasses import SARAcquisitionMode
from perseo_quality.core.signal_processing import (
    convert_to_db,
    estimate_modulation_frequency2d,
    locate_max_2d_interp,
)
from perseo_quality.io.quality_input_protocol import ChannelData, SARCoordinatesFunction
from perseo_quality.spectral_analysis.custom_dataclasses import (
    DistributedSpectraDataOutput,
    PointTargetSpectraDataOutput,
)

# TODO: check what can be moved into Perseo Core


def data_deramping(
    data: np.ndarray,
    channel_data: ChannelData,
    burst: int,
    roi_center_location_px: tuple[int, int],
) -> np.ndarray:
    """Deramping input raster data portion.

    Parameters
    ----------
    data : np.ndarray
        raster data portion, with shape (samples, lines)
    channel_data : ChannelData
        channel data manager instance
    burst : int
        current burst number
    roi_center_location_px : tuple[int, int]
        roi center location in pixels, (azimuth pixel, range pixel)

    Returns
    -------
    np.ndarray
        deramped raster data portion
    """

    burst_start_time_px = np.sum(channel_data.lines_per_burst[:burst])
    mid_burst_az_time, _ = channel_data.get_mid_burst_times(burst=burst)
    lines_step = channel_data.azimuth_axis[1] - channel_data.azimuth_axis[0]

    # default are for scansar mode
    azimuth_steering_rate_axis = None
    sensor_velocity_norm_mid_burst = None
    if channel_data.acquisition_mode == SARAcquisitionMode.TOPSAR:
        azimuth_steering_rate_axis = np.array(
            [
                channel_data.get_steering_rate(azimuth_time=mid_burst_az_time, burst=burst)
                for _ in channel_data.slant_range_axis
            ]
        )
        sensor_velocity_norm_mid_burst = np.linalg.norm(
            channel_data.trajectory.evaluate_first_derivatives(mid_burst_az_time)
        )

    burst_deramping_function = compute_burst_deramping_function(
        mid_burst_az_time=mid_burst_az_time,
        lines_per_burst=channel_data.lines_per_burst[burst],
        lines_step=lines_step,
        slant_range_axis=channel_data.slant_range_axis,
        doppler_centroid_poly=channel_data.doppler_centroid,
        doppler_rate_poly=channel_data.doppler_rate,
        azimuth_steering_rate_axis=azimuth_steering_rate_axis,
        sensor_velocity_norm_mid_burst=sensor_velocity_norm_mid_burst,
        wavelength=speed_of_light / channel_data.carrier_frequency,
    )
    burst_az_relative_target_location_px = roi_center_location_px[0] - burst_start_time_px
    burst_target_start_az_roi_crop = burst_az_relative_target_location_px - np.floor(data.shape[1] / 2).astype(int)
    burst_target_start_rng_roi_crop = roi_center_location_px[1] - np.floor(data.shape[0] / 2).astype(int)
    roi_deramping_function = burst_deramping_function[
        burst_target_start_az_roi_crop : burst_target_start_az_roi_crop + data.shape[1],
        burst_target_start_rng_roi_crop : burst_target_start_rng_roi_crop + data.shape[0],
    ]
    return data * roi_deramping_function.T


def compute_deramping_phase_exponential(
    lines_per_burst: int,
    lines_step: float,
    doppler_centroid_axis: np.ndarray,
    doppler_rate_axis: np.ndarray,
    steering_rate_factor: np.ndarray | None = None,
) -> np.ndarray:
    """Computing deramping exponential phase function. It can be used both for Scansar and Topsar acquisition modes,
    providing the azimuth steering rate for Topsar mode.

    .. math::

        \\begin{align}exp \\left( \\pi i \\cdot coeff \\cdot \\left(t_{burst}^{rel} - \\left(t_{mid\\_burst}^{rel} -
        \\frac{(f_{DC} - f_{DC_{mid}})}{k_D} \\right) \\right)^2 \\right)\\
        coeff_{topsar} = -\\frac{-k_D \\cdot a_r}{a_r - k_D}\\
        coeff_{scansar} = k_D\\
        \\end{align}

    Parameters
    ----------
    lines_per_burst : int
        lines per burst
    lines_step : float
        lines step in seconds
    sensor_velocity_norm_mid_burst : float
        sensor velocity norm computed at burst mid azimuth time
    carrier_frequency : float
        signal carrier frequency
    doppler_centroid_axis : np.ndarray
        doppler centroid frequency axis, computed for each range sample, with shape (samples,)
    doppler_rate_axis : np.ndarray
        doppler rate axis, computed for each range sample, with shape (samples,)
    azimuth_steering_rate_axis : np.ndarray
        azimuth steering rate axis, computed for each range sample, with shape (samples,)

    Returns
    -------
    np.ndarray
        deramping exponential phase function, with shape (lines per burst, samples)
    """
    burst_deramping_azimuth_time_axis = compute_deramping_azimuth_axis(
        lines_per_burst=lines_per_burst,
        lines_step=lines_step,
        doppler_centroid_axis=doppler_centroid_axis,
        doppler_rate_axis=doppler_rate_axis,
    )
    coeff = doppler_rate_axis
    if steering_rate_factor is not None:
        # TOPSAR deramping steering rate K_TU factor
        coeff = -(-doppler_rate_axis * steering_rate_factor) / (steering_rate_factor - doppler_rate_axis)
    deramping_phase = np.pi * coeff * (burst_deramping_azimuth_time_axis) ** 2

    return np.exp(1j * deramping_phase)


def compute_deramping_azimuth_axis(
    lines_per_burst: int, lines_step: float, doppler_centroid_axis: np.ndarray, doppler_rate_axis: np.ndarray
) -> np.ndarray:
    """Computing deramping phase burst azimuth relative axis.

    .. math::

        t_{burst}^{rel} - \\left(t_{mid\\_burst}^{rel} - \\frac{(f_{DC} - f_{DC_{mid}})}{k_D}\\right)

    Parameters
    ----------
    lines_per_burst : int
        lines per burst
    lines_step : float
        lines step in seconds
    doppler_centroid_axis : np.ndarray
        doppler centroid frequency axis, computed for each range sample, with shape (samples,)
    doppler_rate_axis : np.ndarray
        doppler rate axis, computed for each range sample, with shape (samples,)

    Returns
    -------
    np.ndarray
        deramping phase burst azimuth relative axis, with shape (lines per burst, samples)
    """

    burst_azimuth_relative_axis = np.arange(0, lines_per_burst, 1) * lines_step
    beam_center_times = (
        -(doppler_centroid_axis - doppler_centroid_axis[doppler_centroid_axis.size // 2]) / doppler_rate_axis
    )
    reference_times = lines_per_burst / 2 * lines_step + beam_center_times
    return burst_azimuth_relative_axis.reshape(-1, 1) - reference_times


def compute_demodulation_phase_exponential(
    lines_per_burst: int,
    lines_step: float,
    doppler_centroid_axis: np.ndarray,
    doppler_rate_axis: np.ndarray,
) -> np.ndarray:
    """Computing demodulation exponential phase function.

    .. math::

        exp \\left( -2 \\pi i \\cdot f_{DC} t_{burst}^{rel} \\right)

    Parameters
    ----------
    lines_per_burst : int
        lines per burst
    lines_step : float
        lines step in seconds
    doppler_centroid_axis : np.ndarray
        doppler centroid frequency axis, with shape (samples,)
    doppler_rate_axis : np.ndarray
        doppler rate axis, computed for each range sample, with shape (samples,)

    Returns
    -------
    np.ndarray
        demodulation exponential phase function, with shape (lines per burst, samples)
    """
    burst_deramping_azimuth_time_axis = compute_deramping_azimuth_axis(
        lines_per_burst=lines_per_burst,
        lines_step=lines_step,
        doppler_centroid_axis=doppler_centroid_axis,
        doppler_rate_axis=doppler_rate_axis,
    )
    demodulation_phase = -2 * np.pi * doppler_centroid_axis * burst_deramping_azimuth_time_axis
    return np.exp(1j * demodulation_phase)


def compute_burst_deramping_function(
    mid_burst_az_time: PreciseDateTime,
    lines_per_burst: int,
    lines_step: float,
    slant_range_axis: np.ndarray,
    doppler_centroid_poly: SARCoordinatesFunction,
    doppler_rate_poly: SARCoordinatesFunction,
    azimuth_steering_rate_axis: np.ndarray | None = None,
    sensor_velocity_norm_mid_burst: float | None = None,
    wavelength: float | None = None,
) -> np.ndarray:
    """Computing the deramping and demodulation phase exponential function for Topsar and Scansar acquisition for a
    given burst, defined by the mid burst azimuth time.

    Azimuth steering rate, Normalized sensor velocity and sensor wavelength are needed only for Topsar deramping phase
    computation.

    Parameters
    ----------
    mid_burst_az_time : PreciseDateTime
        azimuth time at mid burst
    lines_per_burst : int
        number of lines of the selected burst
    lines_step : float
        azimuth lines step in seconds
    slant_range_axis : np.ndarray
        slant range values
    doppler_centroid_poly : SARCoordinatesFunction
        doppler centroid polynomial
    doppler_rate_poly : SARCoordinatesFunction
        doppler rate polynomial
    azimuth_steering_rate_axis : np.ndarray | None, optional
        steering rate axis, needed for Topsar deramping, by default None
    sensor_velocity_norm_mid_burst : float | None, optional
        normalized sensor velocity at mid burst, needed for Topsar deramping, by default None
    wavelength : float | None, optional
        sensor wavelength, needed for Topsar deramping, by default None

    Returns
    -------
    np.ndarray
        deramping and demodulation phase exponential function to be multiplied to the selected burst data
    """

    doppler_centroid_axis = np.array(
        [doppler_centroid_poly.evaluate(azimuth_time=mid_burst_az_time, range_time=r) for r in slant_range_axis]
    )
    doppler_rate_axis = np.array(
        [doppler_rate_poly.evaluate(azimuth_time=mid_burst_az_time, range_time=r) for r in slant_range_axis]
    )
    steering_rate_factor = None
    if azimuth_steering_rate_axis is not None:
        # topsar
        if sensor_velocity_norm_mid_burst is None or wavelength is None:
            raise RuntimeError(
                "For Topsar deramping 'steering rate', 'normalized velocity' and 'wavelength' are needed"
            )
        steering_rate_factor = compute_steering_rate_factor(
            wavelength=wavelength,
            sensor_velocity_norm_mid_burst=sensor_velocity_norm_mid_burst,
            azimuth_steering_rate_axis=azimuth_steering_rate_axis,
        )

    deramping_phase_exp = compute_deramping_phase_exponential(
        lines_per_burst=lines_per_burst,
        lines_step=lines_step,
        doppler_centroid_axis=doppler_centroid_axis,
        doppler_rate_axis=doppler_rate_axis,
        steering_rate_factor=steering_rate_factor,
    )

    demodulation_phase_exp = compute_demodulation_phase_exponential(
        lines_per_burst=lines_per_burst,
        lines_step=lines_step,
        doppler_centroid_axis=doppler_centroid_axis,
        doppler_rate_axis=doppler_rate_axis,
    )

    return deramping_phase_exp * demodulation_phase_exp


def compute_steering_rate_factor(
    sensor_velocity_norm_mid_burst: float, wavelength: float, azimuth_steering_rate_axis: np.ndarray
) -> np.ndarray:
    """Computing steering rate factor for Topsar deramping.

    Parameters
    ----------
    sensor_velocity_norm_mid_burst : float
        normalized sensor velocity at mid burst
    wavelength : float
        sensor wavelength
    azimuth_steering_rate_axis : np.ndarray
        azimuth steering rate axis

    Returns
    -------
    np.ndarray
        steering rate factor for deramping
    """
    return 2 * sensor_velocity_norm_mid_burst / wavelength * azimuth_steering_rate_axis


def recenter_data(data: np.ndarray) -> np.ndarray:
    """Centering target area on point target signal peak.

    Parameters
    ----------
    data : np.ndarray
        target area 2D array

    Returns
    -------
    np.ndarray
        recentered 2D array
    """
    _, row_peak_pos, col_peak_pos = locate_max_2d_interp(data=data)
    y_shift = row_peak_pos - data.shape[0] // 2
    x_shift = col_peak_pos - data.shape[1] // 2
    _, rng_frq_vect, _, az_freq_vect = estimate_modulation_frequency2d(data=data)
    frequency_y_axis = _frequency_axis_generation(freq_vect=rng_frq_vect.ravel(), samples=data.shape[0])
    frequency_x_axis = _frequency_axis_generation(freq_vect=az_freq_vect.ravel(), samples=data.shape[1])
    phi = np.exp(2j * np.pi * (frequency_x_axis * x_shift + frequency_y_axis.T * y_shift))
    return ifftshift(ifft2(fft2(data) * phi))


def compute_spectrogram_db(
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computing the spectrogram and its time and frequency axes.

    Parameters
    ----------
    data : np.ndarray
        target area 2D

    Returns
    -------
    np.ndarray
        spectrogram in dB
    np.ndarray
        spectrogram frequency axis
    np.ndarray
        spectrogram times axis
    """
    _, _, near_spectrogram = spectrogram(
        data[0, :],
        fs=1,
        window=hamming(32),
        nfft=64,
        noverlap=31,
        return_onesided=False,
    )
    spectrogram_freq, spectrogram_times, mid_spectrogram = spectrogram(
        data[data.shape[0] // 2, :],
        fs=1,
        window=hamming(32),
        nfft=64,
        noverlap=31,
        return_onesided=False,
    )
    _, _, far_spectrogram = spectrogram(
        data[-1, :],
        fs=1,
        window=hamming(32),
        nfft=64,
        noverlap=31,
        return_onesided=False,
    )
    spectrogram_db = convert_to_db(
        np.abs((np.abs(near_spectrogram) + np.abs(mid_spectrogram) + np.abs(far_spectrogram)) / 3)
    )
    return fftshift(spectrogram_db, axes=0), fftshift(spectrogram_freq), spectrogram_times


def compute_polynomial_fit(profile: np.ndarray, freq_axis: np.ndarray, boundaries: tuple[float, float]) -> Polynomial:
    """FItting polynomial on profile portion.

    Parameters
    ----------
    profile : np.ndarray
        profile to be fit
    freq_axis : np.ndarray
        frequency axis
    boundaries : tuple[float, float]
        fitting region boundaries

    Returns
    -------
    Polynomial
        fitting Polynomial
    """
    return Polynomial.fit(freq_axis[boundaries[0] : boundaries[1]], profile[boundaries[0] : boundaries[1]], deg=2)


def extract_abs_profiles(
    data_fft: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Extract range and azimuth absolute profiles in dB.

    Parameters
    ----------
    data_fft : np.ndarray
        fft of the target area

    Returns
    -------
    list[np.ndarray]
        3 range absolute profiles in dB
    list[np.ndarray]
        3 azimuth absolute profiles in dB
    """
    rng_profiles_starts, rng_profiles_stops = _compute_profiles_extraction_boundaries(data_fft.shape[1])
    range_profiles_db = [
        convert_to_db(np.nanmean(np.abs(data_fft[:, start:stop]) ** 2, axis=1))
        for start, stop in zip(rng_profiles_starts, rng_profiles_stops, strict=True)
    ]
    az_profiles_starts, az_profiles_stops = _compute_profiles_extraction_boundaries(data_fft.shape[0])
    azimuth_profiles_db = [
        convert_to_db(np.nanmean(np.abs(data_fft[start:stop, :]) ** 2, axis=0))
        for start, stop in zip(az_profiles_starts, az_profiles_stops, strict=True)
    ]
    return range_profiles_db, azimuth_profiles_db


def extract_phase_profiles(
    data_fft: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Extract range and azimuth phase profiles in deg.

    Parameters
    ----------
    data_fft : np.ndarray
        fft of the target area

    Returns
    -------
    list[np.ndarray]
        3 range phase profiles in deg
    list[np.ndarray]
        3 azimuth phase profiles in deg
    """
    rng_profiles_starts, rng_profiles_stops = _compute_profiles_extraction_boundaries(data_fft.shape[1])
    range_profiles_deg = [
        np.angle(np.nanmean(data_fft[:, start:stop], axis=1), deg=True)
        for start, stop in zip(rng_profiles_starts, rng_profiles_stops, strict=True)
    ]
    az_profiles_starts, az_profiles_stops = _compute_profiles_extraction_boundaries(data_fft.shape[0])
    azimuth_profiles_deg = [
        np.angle(np.nanmean(data_fft[start:stop, :], axis=0), deg=True)
        for start, stop in zip(az_profiles_starts, az_profiles_stops, strict=True)
    ]
    return range_profiles_deg, azimuth_profiles_deg


def compute_spectrum_boundaries(profile: np.ndarray) -> tuple[int, int]:
    """Computing relevant spectrum boundaries from profile.

    Parameters
    ----------
    profile : np.ndarray
        profile to be analyzed

    Returns
    -------
    int
        start index of spectrum
    int
        stop index of spectrum
    """
    profile_smooth = gaussian_filter1d(profile, sigma=3)
    abs_derivative = np.abs(np.diff(profile_smooth))
    peak_indexes, _ = find_peaks(abs_derivative / abs_derivative.max(), height=0.8, distance=50)
    if peak_indexes.size < 2:
        return 0, profile.size
    start_idx = max([0, int(peak_indexes[0]) + 2])
    stop_idx = min([profile.size, int(peak_indexes[-1]) - 2])
    return start_idx, stop_idx


def spectral_analysis_profiles_to_netcdf(
    data: list[PointTargetSpectraDataOutput] | list[DistributedSpectraDataOutput],
    out_path: str | Path,
) -> Path:
    """Saving Spectral Analysis Profiles output data to NetCDF4 file.

    Hierarchy::

    Point Target Spectral Analysis Hierarchy::

        root/
        ├── product_attributes...
        └── swath
            └── polarization
                ├── channel_attributes...
                ├── doppler_centroid [at target position]
                ├── phase_value [at target position]
                ├── azimuth_frequency_axis
                ├── range_frequency_axis
                ├── azimuth_profiles_abs
                ├── range_profiles_abs
                ├── azimuth_profiles_phase
                ├── range_profiles_phase
                ├── az_phase_polynomial_coefficients
                └── rng_phase_polynomial_coefficients

    Distributed Spectral Analysis Hierarchy::

        root/
        ├── product_attributes...
        └── swath
            └── polarization
                ├── channel_attributes...
                └── block
                    ├── block_attributes...
                    ├── azimuth_frequency_axis
                    ├── range_frequency_axis
                    ├── azimuth_profiles_abs
                    └── range_profiles_abs

    Parameters
    ----------
    data : list[PointTargetSpectraDataOutput] | list[DistributedSpectraDataOutput]
        list of PointTargetSpectraDataOutput | DistributedSpectraDataOutput dataclasses, corresponding to the full
        output of the radiometric analysis
    out_path : str | Path
        path where to save the NetCDF file


    Returns
    -------
    Path
        path to the output NetCDF file
    """
    is_pt_analysis = isinstance(data[0], PointTargetSpectraDataOutput)
    tag = "pt" if is_pt_analysis else "distr"
    out_name = tag + "_spectral_profiles_" + data[0].general_info.product
    out_path = Path(out_path)
    output_file = out_path.joinpath(out_name).with_suffix(".nc")

    root = Dataset(output_file, "w", format="NETCDF4")
    root.title = f"{'Point Target' if is_pt_analysis else 'Distributed'} Spectral Analysis"
    root.history = f"Created by PERSEO Quality v{__version__}"
    root.product = data[0].general_info.product
    root.sensor = data[0].general_info.sensor
    root.product_type = data[0].general_info.product_type
    root.acquisition_mode = data[0].general_info.acquisition_mode
    root.orbit_direction = data[0].general_info.orbit_direction

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
        pol_grp.acquisition_start_time = str(item.general_info.acquisition_start_time)

        # creating common dimensions
        if is_pt_analysis:
            pol_grp.createDimension("targets", len(item.targets_info))
            pol_grp.createDimension("azimuth", item.targets_info[0].azimuth_frequency_axis.size)
            pol_grp.createDimension("range", item.targets_info[0].range_frequency_axis.size)
            pol_grp.createDimension("slices", 3)
            pol_grp.createDimension("coeffs", 3)

            # name of analyzed targets
            pol_grp.bursts = [n.burst for n in item.targets_info]
            pol_grp.targets = [n.target_name for n in item.targets_info]

            # doppler centroid and phase value at target position
            dc_target = pol_grp.createVariable(
                "doppler_centroid", item.targets_info[0].target_doppler_centroid_Hz.dtype, ("targets",)
            )
            dc_target.unit = "Hz"
            dc_target.description = "Doppler centroid at the target position"
            dc_target[:] = np.array([p.target_doppler_centroid_Hz for p in item.targets_info])
            phase_target = pol_grp.createVariable(
                "phase_value", item.targets_info[0].target_phase_value_deg.dtype, ("targets",)
            )
            phase_target.unit = "deg"
            phase_target.description = "Phase value at the target position"
            phase_target[:] = np.array([p.target_phase_value_deg for p in item.targets_info])

            # frequency axes
            az_freq_axis = pol_grp.createVariable(
                "azimuth_frequency_axis", item.targets_info[0].azimuth_frequency_axis.dtype, ("azimuth",)
            )
            az_freq_axis.unit = "Hz"
            az_freq_axis[:] = item.targets_info[0].azimuth_frequency_axis
            rng_freq_axis = pol_grp.createVariable(
                "range_frequency_axis", item.targets_info[0].range_frequency_axis.dtype, ("range",)
            )
            rng_freq_axis.unit = "Hz"
            rng_freq_axis[:] = item.targets_info[0].range_frequency_axis

            # absolute profiles
            az_abs_profiles = pol_grp.createVariable(
                "azimuth_profiles_abs",
                item.targets_info[0].azimuth_profiles_db[0].dtype,
                ("targets", "slices", "azimuth"),
            )
            az_abs_profiles.unit = "dB"
            az_abs_profiles[:] = np.stack([p.azimuth_profiles_db for p in item.targets_info])
            rng_abs_profiles = pol_grp.createVariable(
                "range_profiles_abs", item.targets_info[0].range_profiles_db[0].dtype, ("targets", "slices", "range")
            )
            rng_abs_profiles.unit = "dB"
            rng_abs_profiles[:] = np.stack([p.range_profiles_db for p in item.targets_info])

            # phase profiles
            az_phase_profiles = pol_grp.createVariable(
                "azimuth_profiles_phase",
                item.targets_info[0].azimuth_profiles_deg[0].dtype,
                ("targets", "slices", "azimuth"),
            )
            az_phase_profiles.unit = "deg"
            az_phase_profiles[:] = np.stack([p.azimuth_profiles_deg for p in item.targets_info])
            rng_phase_profiles = pol_grp.createVariable(
                "range_profiles_phase",
                item.targets_info[0].range_profiles_deg[0].dtype,
                ("targets", "slices", "range"),
            )
            rng_phase_profiles.unit = "deg"
            rng_phase_profiles[:] = np.stack([p.range_profiles_deg for p in item.targets_info])

            # polynomials
            az_poly_coeffs = pol_grp.createVariable(
                "az_phase_polynomial_coefficients",
                item.targets_info[0].azimuth_polynomial_fit.convert().coef.dtype,
                ("targets", "coeffs"),
            )
            az_poly_coeffs[:] = np.stack([p.azimuth_polynomial_fit.convert().coef for p in item.targets_info])
            rng_poly_coeffs = pol_grp.createVariable(
                "rng_phase_polynomial_coefficients",
                item.targets_info[0].range_polynomial_fit.convert().coef.dtype,
                ("targets", "coeffs"),
            )
            rng_poly_coeffs[:] = np.stack([p.range_polynomial_fit.convert().coef for p in item.targets_info])

        else:
            for block in item.blocks_info:
                blk_grp = pol_grp.createGroup(f"block_{block.block_num}")
                blk_grp.first_az_line_block = block.first_az_line_block
                blk_grp.lines_block = block.lines_block
                blk_grp.samples_block = block.samples_block
                blk_grp.doppler_centroid_mid_block_Hz = block.doppler_centroid_mid_block

                blk_grp.createDimension("azimuth", block.azimuth_frequency_axis.size)
                blk_grp.createDimension("range", block.range_frequency_axis.size)
                blk_grp.createDimension("slices", 3)

                # frequency axes
                az_freq_axis = blk_grp.createVariable(
                    "azimuth_frequency_axis", block.azimuth_frequency_axis.dtype, ("azimuth",)
                )
                az_freq_axis.unit = "Hz"
                az_freq_axis[:] = block.azimuth_frequency_axis
                rng_freq_axis = blk_grp.createVariable(
                    "range_frequency_axis", block.range_frequency_axis.dtype, ("range",)
                )
                rng_freq_axis.unit = "Hz"
                rng_freq_axis[:] = block.range_frequency_axis

                # absolute profiles
                az_abs_profiles = blk_grp.createVariable(
                    "azimuth_profiles_abs",
                    block.azimuth_profiles_db[0].dtype,
                    ("slices", "azimuth"),
                )
                az_abs_profiles.unit = "dB"
                az_abs_profiles[:] = np.stack(block.azimuth_profiles_db)
                rng_abs_profiles = blk_grp.createVariable(
                    "range_profiles_abs", block.range_profiles_db[0].dtype, ("slices", "range")
                )
                rng_abs_profiles.unit = "dB"
                rng_abs_profiles[:] = np.stack(block.range_profiles_db)

        root.close()
        return output_file


def _frequency_axis_generation(freq_vect: np.ndarray, samples: int, prf: int = 1) -> np.ndarray:
    """Compute frequency axis.

    Parameters
    ----------
    freq_vect : np.ndarray
        frequency vector
    samples : int
        samples
    prf : int, optional
        sensor prf, by default 1

    Returns
    -------
    np.ndarray
        frequency axis
    """
    freq_shift = freq_vect % prf
    delta_freq = prf / samples
    freq_0 = np.arange(samples) * delta_freq
    freq_axis = np.zeros((freq_vect.size, samples))
    for f_id, freq in enumerate(freq_vect):
        f = (freq_0 - freq_shift[f_id] + prf / 2) % prf - prf / 2
        freq_axis[f_id, :] = f + freq

    return freq_axis


def _compute_profiles_extraction_boundaries(
    axis_length: int,
) -> tuple[list[int], list[int]]:
    """Define start and stop indexes for profiles to be extracted. 3 profiles are extracted properly spaced along the
    axis length.

    Parameters
    ----------
    axis_length : int
        full length of the axis

    Returns
    -------
    list[int]
        profiles start indexes
    list[int]
        profiles stop indexes
    """
    profiles_starts = [round(axis_length * p / 4 - axis_length * 0.1) for p in range(1, 4)]
    profiles_stops = [round(axis_length * p / 4 + axis_length * 0.1) for p in range(1, 4)]
    return profiles_starts, profiles_stops
