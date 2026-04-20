# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Processing - ROI Features
-------------------------
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numpydantic import NDArray, Shape
from scipy.constants import speed_of_light
from scipy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.signal import spectrogram
from scipy.signal.windows import hamming

from perseo_core.models.enums import SARAcquisitionMode, SARRadiometricQuantity
from perseo_core.models.protocols import SARCoordinatesFunction
from perseo_core.processing.signal_features import estimate_modulation_frequency2d, locate_max_2d_interp
from perseo_core.timing.precise_datetime import PreciseDateTime

ROIArrayType = NDArray[Shape["* samples, * lines"], float] | NDArray[Shape["2"], float]  # type: ignore


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


def radiometric_correction(
    data: ROIArrayType,
    incidence_angle: npt.NDArray[np.floating] | ROIArrayType,
    input_quantity: SARRadiometricQuantity,
    output_quantity: SARRadiometricQuantity,
    exp_power: float = 0.5,
) -> ROIArrayType:
    """Data radiometric correction based on data acquisition type and desired output type, choosing between Beta, Sigma
    and Gamma Nought.

    Parameters
    ----------
    data : ROIArrayType
        input array whose data need to be converted from a type to another, array should be in the form (samples, lines)
    incidence_angle : npt.NDArray[np.floating]
        incidence angle array, same size as range axis of input data or same data size: (samples,) or (samples, lines)
    input_quantity : SARRadiometricQuantity
        input radiometric quantity
    output_quantity : SARRadiometricQuantity
        output radiometric quantity
    exp_power : float, optional
        exponential power correction in computing different radiometric corrections, by default 0.5

    Returns
    -------
    np.ndarray
        corrected data array, in the form (samples, lines)

    Raises
    ------
    ValueError
        input_quantity and output_quantity not of the proper enum type
    ValueError
        data and incidence_angle do not have the same shape

    """
    if (not isinstance(input_quantity, SARRadiometricQuantity)) or (
        not isinstance(output_quantity, SARRadiometricQuantity)
    ):
        raise ValueError("Input and output type must be of type SARRadiometricQuantity")

    if incidence_angle.ndim == 1:
        incidence_angle = np.atleast_2d(incidence_angle).T
    elif data.shape != incidence_angle.shape:
        raise ValueError(f"Incidence angle shape must be (n_rng,) or input data shape {data.shape}")

    if input_quantity == SARRadiometricQuantity.BETA_NOUGHT:
        if output_quantity == SARRadiometricQuantity.SIGMA_NOUGHT:
            out_data = data * (np.sin(incidence_angle) ** exp_power)
        elif output_quantity == SARRadiometricQuantity.GAMMA_NOUGHT:
            out_data = data * (np.sin(incidence_angle) ** exp_power) / (np.cos(incidence_angle) ** exp_power)

    elif input_quantity == SARRadiometricQuantity.SIGMA_NOUGHT:
        if output_quantity == SARRadiometricQuantity.BETA_NOUGHT:
            out_data = data / (np.sin(incidence_angle) ** exp_power)
        elif output_quantity == SARRadiometricQuantity.GAMMA_NOUGHT:
            out_data = data / (np.cos(incidence_angle) ** exp_power)

    elif input_quantity == SARRadiometricQuantity.GAMMA_NOUGHT:
        if output_quantity == SARRadiometricQuantity.BETA_NOUGHT:
            out_data = data / (np.sin(incidence_angle) ** exp_power) * (np.cos(incidence_angle) ** exp_power)
        elif output_quantity == SARRadiometricQuantity.SIGMA_NOUGHT:
            out_data = data * (np.cos(incidence_angle) ** exp_power)

    return out_data


def compute_equivalent_number_of_looks(power_data: ROIArrayType) -> float:
    """Computing Equivalent Number of Looks (ENL) from input roi data.

    .. math::

        ENL = \\frac{I_{mean}^2}{I_{std}^2}

    Parameters
    ----------
    power_data : ROIArrayType
        power of 2D roi data image (abs ** 2), with shape (samples, lines)

    Returns
    -------
    float
        Equivalent Number of Looks (ENL) for the given data region
    """
    pow_mean = np.nanmean(power_data)
    pow_std = np.nanstd(power_data)

    return float((pow_mean**2) / (pow_std**2))


def compute_point_target_ambiguity_ratio_db(
    point_target_roi: ROIArrayType,
    right_ambiguity_roi: ROIArrayType,
    left_ambiguity_roi: ROIArrayType,
    interp_factor: int,
) -> float:
    """Computing the Point Target Ambiguity Ratio (PTAR).

    This parameter is computed using the following formula:

    .. math::

        PTAR = 20\\log_{10}\\left(\\frac{{|I_{amb_{left}}| + |I_{amb_{right}}|}}{2|I_{pt}|}\\right)

    Parameters
    ----------
    point_target_roi : ROIArrayType
        raster data portion centered on the point target location, with shape (samples, lines)
    right_ambiguity_roi : ROIArrayType
        raster data portion centered on the point target right ambiguity, with shape (samples, lines)
    left_ambiguity_roi : ROIArrayType
        raster data portion centered on the point target left ambiguity, with shape (samples, lines)
    interp_factor : int
        interpolation factor

    Returns
    -------
    float
        Point Target Ambiguity Ratio

    """
    pt_peak_value, _, _ = locate_max_2d_interp(data=point_target_roi, interp_factor=interp_factor)
    r_ambiguity_peak_value, _, _ = locate_max_2d_interp(data=right_ambiguity_roi, interp_factor=interp_factor)
    l_ambiguity_peak_value, _, _ = locate_max_2d_interp(data=left_ambiguity_roi, interp_factor=interp_factor)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 20 * np.log10(
            np.nanmean([np.abs(r_ambiguity_peak_value), np.abs(l_ambiguity_peak_value)]) / np.abs(pt_peak_value),
        )


def compute_distributed_target_ambiguity_ratio_db(
    distributed_target_roi: ROIArrayType,
    right_ambiguity_roi: ROIArrayType,
    left_ambiguity_roi: ROIArrayType,
) -> float:
    """Computing the Distributed Target Ambiguity Ratio (DTAR).

    This parameter is computed using the following formula:

    .. math::

        DTAR = \\frac{E(\\Sigma |amb_{left}|^2) + E(\\Sigma |amb_{right}|^2)}{2*E(\\Sigma |target|^2)}

    Parameters
    ----------
    distributed_target_roi : ROIArrayType
        raster data portion centered on the point target location, with shape (samples, lines)
    right_ambiguity_roi : ROIArrayType
        raster data portion centered on the point target right ambiguity, with shape (samples, lines)
    left_ambiguity_roi : ROIArrayType
        raster data portion centered on the point target left ambiguity, with shape (samples, lines)

    Returns
    -------
    float
        Distributed Target Ambiguity Ratio

    """
    distributed_peak = np.sum(np.abs(distributed_target_roi) ** 2) / distributed_target_roi.size
    r_ambiguity_distributed_peak = np.sum(np.abs(right_ambiguity_roi) ** 2) / right_ambiguity_roi.size
    l_ambiguity_distributed_peak = np.sum(np.abs(left_ambiguity_roi) ** 2) / left_ambiguity_roi.size
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10 * np.log10(
            np.nanmean([np.abs(r_ambiguity_distributed_peak), np.abs(l_ambiguity_distributed_peak)])
            / np.abs(distributed_peak),
        )


def recenter_data(data: ROIArrayType) -> ROIArrayType:
    """Centering target area on point target signal peak.

    Parameters
    ----------
    data : ROIArrayType
        target area 2D array

    Returns
    -------
    ROIArrayType
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
    data: ROIArrayType,
    nfft: int = 64,
    noverlap: int = 31,
    window: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computing the spectrogram and its time and frequency axes.

    Parameters
    ----------
    data : np.ndarray
        target area 2D
    nfft : int, optional
        number of points in the Fourier transform, by default 64
    noverlap : int, optional
        number of points to overlap between segments, by default 31
    window : np.ndarray | None, optional
        window function, by default Hamming(32) if None is provided

    Returns
    -------
    np.ndarray
        spectrogram in dB
    np.ndarray
        spectrogram frequency axis
    np.ndarray
        spectrogram times axis
    """
    if window is None:
        window = hamming(32)
    _, _, near_spectrogram = spectrogram(
        data[0, :],
        fs=1,
        window=window,
        nfft=nfft,
        noverlap=noverlap,
        return_onesided=False,
    )
    spectrogram_freq, spectrogram_times, mid_spectrogram = spectrogram(
        data[data.shape[0] // 2, :],
        fs=1,
        window=window,
        nfft=nfft,
        noverlap=noverlap,
        return_onesided=False,
    )
    _, _, far_spectrogram = spectrogram(
        data[-1, :],
        fs=1,
        window=window,
        nfft=nfft,
        noverlap=noverlap,
        return_onesided=False,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        spectrogram_db = 20 * np.log10(
            np.abs((np.abs(near_spectrogram) + np.abs(mid_spectrogram) + np.abs(far_spectrogram)) / 3)
        )
    return spectrogram_db, fftshift(spectrogram_freq), spectrogram_times


# TODO: the following part must be completely refactored and abstracted


def data_deramping(
    data: np.ndarray,
    channel_data: ChannelData,  # noqa: F821  # TODO: change entry point here
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
    beam_center_times = -doppler_centroid_axis / doppler_rate_axis
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
