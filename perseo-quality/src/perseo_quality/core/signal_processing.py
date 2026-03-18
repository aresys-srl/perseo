# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Module containing all the signal processing related functions"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy import signal
from scipy.constants import speed_of_light

from perseo_quality.core.generic_dataclasses import DecibelConversion, GetFrequencyMethod, SARRadiometricQuantity

if TYPE_CHECKING:
    from arepytools.geometry.curve_protocols import TwiceDifferentiable3DCurve
    from arepytools.timing.precisedatetime import PreciseDateTime
    from numpy.typing import ArrayLike


def convert_to_db(data: ArrayLike, mode: DecibelConversion = DecibelConversion.POWER) -> ArrayLike:
    """Converting input data to decibel.

    Parameters
    ----------
    data : ArrayLike
        input array or float
    mode : DecibelConversion, optional
        if input data is already in power (abs(data)^2) use DecibelConversion.POWER (factor 10), else use
        DecibelConversion.AMPLITUDE if input data is just (abs(data)), by default DecibelConversion.POWER

    Returns
    -------
    ArrayLike
        array of floats or float in decibel

    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered in log10")
        # setting the multiplying factor value
        factor = 10  # intensity is the default value
        if mode == DecibelConversion.AMPLITUDE:
            factor = 20

        # correcting decibel of 0 values to nan
        if isinstance(data, (int, float)):
            if data == 0:
                return np.nan

            return factor * np.log10(data)

        return factor * np.log10(data)


def shift_spectrum(data: np.ndarray, shift: ArrayLike, axis: int = 0) -> np.ndarray:
    """Shift the spectrum of time domain signal along the selected axis by applying a time domain modulation.

    The shift can be a single value or an array of values with the size of the other axis.
    This allows to apply a shift that varies along the other axis.
    For example, in case of an (azimuth, range) signal a range variant azimuth spectral shift can be specified.

    shift is normalized: to apply a shift 'df' in Hertz you need to set 'shift' to 'df * dt'.

    Parameters
    ----------
    data : np.ndarray
        signal of shape (L, S)
    shift : ArrayLike
        either a constant or an array of shape (L,) or (S,) depending on the specified axis.
    axis : int, optional
        axis along which the spectral shift is applied, by default 0

    Returns
    -------
    np.ndarray
        the modulated signal of shape (L, S)

    """
    normalized_time = 2 * np.pi * np.arange(data.shape[axis])

    if axis == 0:
        phase = normalized_time.reshape(-1, 1) * np.asarray(shift)
    else:
        phase = normalized_time * np.asarray(shift).reshape(-1, 1)

    if np.ndim(data) == 1:
        phase = phase.squeeze()

    return data * np.exp(1j * phase)


def modulate_data(data: np.ndarray, mod_freq: ArrayLike) -> np.ndarray:
    """Time domain data modulation to shift a signal in the frequency domain along the selected axis.
    Signal spectrum is shifted by mod_freq value.

    The modulation frequency can be a single value or an array of values with the size of the other axis.
    This allows to apply frequency shift that varies along the other axis.
    For example, if the signal is of shape (azimuth, range) you can specify a range variant azimuth shift.

    mod_freq is normalized, i.e. to apply a shift df in Hertz you need to set mod_freq = df * dt.

    Parameters
    ----------
    data : np.ndarray
        signal of shape (L, S)
    mod_freq : ArrayLike
        either a constant or an array of shape (L,) or (S,) depending on the axis.

    Returns
    -------
    np.ndarray
        the modulated signal of shape (L, S)

    """
    return shift_spectrum(data, mod_freq, axis=0)


def linear_best_fit_by_fft(
    input_array: np.ndarray,
    weights: np.ndarray,
    substitute_value: float,
    nyquist_position: float,
    interp_factor: int = 16,
) -> np.ndarray:
    """Computing the line best fit of the input array using fft.
    This function can be used to extract a circularity-safe line of best fit from an input noisy array.
    It's a way to smooth the input noisy array and extract only the linear trend by means of an fft.
    Ensuring circularity means that the last and first value of the line of best fit are equal, to take into account
    the periodicity of the spectrum (first and last column/row must be processed the same way).

    Parameters
    ----------
    input_array : np.ndarray
        input array from which to extract the line of best fit
    weights : np.ndarray
        weights to be applied to the input array before performing the fft
    substitute_value : float
        if the input array is constant, this is the substitute value used to provide a constant line output
    nyquist_position : float
        half of the sampling frequency
    interp_factor : int, optional
        interpolation factor to be applied in the frequency domain, by default 16

    Returns
    -------
    np.ndarray
        linear fit of the input array

    """
    # linear fitting by using fft
    # weighting applied to estimates
    array_fit = weights * np.exp(1j * 2 * np.pi * input_array.copy())
    # applying fft and interpolating to increase resolution in frequency
    array_fit_fft = np.fft.fftn(array_fit, s=(input_array.size * interp_factor,), axes=(0,))
    # detecting the position of the maximum of the fft
    # it corresponds to the angular coefficient of the line best fit
    pos = np.argmax(np.abs(array_fit_fft))

    # checking position of the maximum: re-shifting it to ensure it is placed inside +-sampling_freq/2 domain
    if pos > input_array.size * interp_factor / 2 - 1:
        pos = pos - input_array.size * interp_factor

    if pos == 0:
        # if the angular coefficient of the best fit line is 0, it means that the input array was constant,
        # i.e. contained only a single value
        # ????
        # why i can't return the input array instead? or a input_array.mean() * np.ones_like(input_array)
        line_trend = substitute_value * np.ones((1, input_array.size))
    else:
        # composing the best fit linear trend
        line_trend = np.arange(input_array.size) * (pos / interp_factor / input_array.size)

        # ensuring circularity (first sample = last sample)
        # aligning first sample
        line_trend = line_trend + input_array[0]
        # creating discontinuity so that array start value is equal to end one
        line_trend[nyquist_position + 1 :] = line_trend[nyquist_position + 1 :] - line_trend[-1] + line_trend[0]

    return line_trend


def estimate_modulation_frequency(
    data: np.ndarray,
    method: GetFrequencyMethod = GetFrequencyMethod.AUTOCORRELATION,
    axis: int = 1,
) -> tuple[float, np.ndarray]:
    """Estimate modulation frequency of the data spectrum using the selected method.

    This function estimates the modulation frequency of the frequency spectrum of the input time domain data.
    This is necessary to down-convert the input signal to baseband, that is to recenter the spectrum in the frequency
    domain.

    This function returns both the constant modulation frequency, determined as a sort of mean from the whole spectrum,
    and an array with a frequency value for each column of the input array (not averaged).
    The first output is enough to demodulate simple systems (low resolution sensor) with rectangular spectrum (not so
    warped) while the second one can be used in complex scenarios where the spectrum is remarkably warped to recenter
    the spectrum column-wise.

    Parameters
    ----------
    data : np.ndarray
        input array in the time domain
    method : GetFrequencyMethod, optional
        method of computation of the local frequency, by default GetFrequencyMethod.AUTOCORRELATION

    Returns
    -------
    float
        local frequency
    np.ndarray
        local frequency array

    Raises
    ------
    NotImplementedError
        FFT method not supported yet
    NotImplementedError
        POWER_BALANCE method not supported yet

    """
    data = data.squeeze()

    if method == GetFrequencyMethod.AUTOCORRELATION:
        # 1D array
        if data.ndim == 1:
            if axis == 1:
                temp_corr_ = np.correlate(data, data, mode="full").astype(np.complex128)[data.size - 2 : data.size + 1]
                local_frequency_ = np.angle(temp_corr_[2]) / (2 * np.pi)

                return local_frequency_, np.atleast_1d(local_frequency_)

            if axis == 0:
                temp_corr_ = np.concatenate(
                    [np.correlate(np.atleast_1d(d), np.atleast_1d(d), mode="full").astype(np.complex128) for d in data],
                )
                tot_corr_ = np.sum(temp_corr_)
                # local frequency is the angle of the after-max value, normalized
                local_frequency_ = np.angle(tot_corr_) / (2 * np.pi)
                # local frequency array that can be used for a finer modulation
                local_frequency_array_ = np.angle(temp_corr_) / (2 * np.pi)

                return local_frequency_, local_frequency_array_

        # 2D array
        n_rows, n_col = data.shape
        temp_corr = np.zeros((3, n_col), dtype=np.complex128)
        for col in range(n_col):
            # compute autocorrelation of the input signal storing resulting values only across the peak of the
            # correlation (max autocorrelation is at n_rows)
            # this will result in 3 values: before-max, max, after-max
            temp_corr[:, col] = np.correlate(data[:, col], data[:, col], mode="full")[n_rows - 2 : n_rows + 1]
        # summing the three values for each dimension column-wise
        # that leaves cumulative before-max, max, after-max values
        tot_corr = np.sum(temp_corr, axis=1)

        # local frequency is the angle of the after-max value, normalized
        local_frequency = np.angle(tot_corr[2]) / (2 * np.pi)
        # local frequency array that can be used for a finer modulation
        local_frequency_array = np.angle(temp_corr[2, :]) / (2 * np.pi)

    elif method in (GetFrequencyMethod.FFT, GetFrequencyMethod.POWER_BALANCE):
        raise NotImplementedError  # TBD

    return local_frequency, local_frequency_array


def estimate_modulation_frequency2d(
    data: np.ndarray,
    method: GetFrequencyMethod = GetFrequencyMethod.AUTOCORRELATION,
) -> tuple[float, np.ndarray, float, np.ndarray]:
    """Computing the estimate of the modulation frequency along both axes of the input 2D array.
    This functions applied the estimate_modulation_frequency algorithm along both range and azimuth and
    returns the modulation frequencies and their arrays (linear best fit).

    Main algorithm to be applied along each direction:
        - computing fft along an axis of the 2D array
        - this allows to estimate the demodulation frequency in the other direction
        - linear fit of the demodulation frequency array

    Parameters
    ----------
    data : np.ndarray
        input 2D array
    method : GetFrequencyMethod, optional
        method of get_local_frequency application, by default GetFrequencyMethod.AUTOCORRELATION

    Returns
    -------
    float
        local frequency range
    np.ndarray
        local frequency range array (linear fit)
    float
        local frequency azimuth
    np.ndarray
        local frequency azimuth array (linear fit)

    """
    # Retrieve range dimension of input data
    n_rg, _ = data.shape

    # Computing demodulation frequency along AZIMUTH
    # transpose input data to estimate azimuth frequency first
    data = data.copy().transpose()

    # transform computed along range to get demodulation frequency along azimuth
    data_portion_fft = np.fft.fftn(data, axes=(1,))
    # to estimate demodulation frequency for azimuth direction
    loc_freq_az, loc_freq_az_vect = estimate_modulation_frequency(data=data_portion_fft, method=method)

    # finding location of Niquist frequency (half of signal sampling frequency) along azimuth
    nyquist_position = np.argmin(np.sum(np.abs(data_portion_fft), axis=0))

    # finding the linear best fit of the demodulation frequency array (smoothing it out to remove noise)
    # ensuring circularity (first point = last point) and taking into account periodicity and wrapping issues
    loc_freq_az_vect = linear_best_fit_by_fft(
        input_array=loc_freq_az_vect,
        weights=np.abs(np.sum(data_portion_fft, axis=0)),
        substitute_value=loc_freq_az,
        nyquist_position=nyquist_position,
    )

    # applying modulation along azimuth before computing modulation frequency along range
    data_portion_fft = modulate_data(data=data_portion_fft, mod_freq=-loc_freq_az_vect)

    # going back to time domain but transposing the array in order to compute the demodulation frequency in
    # the other direction
    # transform computed along azimuth to get demodulation frequency along range
    mat_temp = np.fft.ifftn(data_portion_fft, axes=(1,)).transpose()

    # Computing demodulation frequency along RANGE
    # going back to frequency domain, this time computing transform along azimuth
    mat_temp_fft = np.fft.fftn(mat_temp, axes=(1,))

    loc_freq_rg, loc_freq_rg_vect = estimate_modulation_frequency(data=mat_temp_fft, method=method)

    # finding location of Niquist frequency (half of signal sampling frequency) along range
    nyquist_position = n_rg // 2 - 1

    # finding the linear best fit of the demodulation frequency array (smoothing it out to remove noise)
    # ensuring circularity (first point = last point) and taking into account periodicity and wrapping issues
    loc_freq_rg_vect = linear_best_fit_by_fft(
        input_array=loc_freq_rg_vect,
        weights=np.abs(np.sum(mat_temp_fft, axis=0)),
        substitute_value=loc_freq_rg,
        nyquist_position=nyquist_position,
    )

    return loc_freq_rg, loc_freq_rg_vect, loc_freq_az, loc_freq_az_vect


def get_frequency_axis(central_freqs: ArrayLike, sampling_freq: float, n_samples: int) -> np.ndarray:
    """Generating a frequency axis from input parameters.

    Creating a monotonically increasing frequency axis from 0 to sampling_freq/2 + central_freq, and from
    -sampling_freq/2 + central_freq to sampling frequency with an abrupt discontinuity.
    This takes into account the wrapping of frequencies, marking the position of the peak (central_frequency) with the
    discontinuity point.

    Parameters
    ----------
    central_freqs : ArrayLike
        central frequencies, a.k.a. modulation frequency
    sampling_freq : float
        sampling frequency
    n_samples : int
        number of samples

    Returns
    -------
    np.ndarray
        frequency axis

    """
    if not isinstance(central_freqs, float):
        central_freqs = central_freqs.squeeze()

    # taking only the modulo of the central frequency with respect to sampling frequency
    # to remove periodicity of frequencies
    freq_shift = central_freqs % sampling_freq

    sampling_step = sampling_freq / n_samples

    starting_freq = np.arange(n_samples) * sampling_step

    freq_axis = np.array(
        [((starting_freq - freq + sampling_freq / 2) % sampling_freq) - sampling_freq / 2 for freq in freq_shift],
    )
    return freq_axis + central_freqs.reshape(-1, 1)


def parabolic_interp_by_3_closest_samples(array: np.ndarray) -> tuple[float, float]:
    """Parabolic peak interpolation using the three samples closest to the peak.

    Fitting a parabola to the 3-points input array, containing the closest point before the peak, the peak itself and
    the closest point after the peak.

    Considering a parabola written with explicit dependency from the position of its interpolated peak location in bins

    $$
    y(x)\\overset{\\Delta}{=}a(x-p)^2+b
    $$

    at the three samples nearest the peak, considering their bins as -1 (before), 0 (peak), 1 (after) we have:

    $$
    \\begin{aligned}
    & y(-1) = ap^2+2ap+a+b = \\alpha \\\\[10pt]
    & y(0) = ap^2+b = \\beta \\\\[10pt]
    & y(1) = ap^2-2ap+a+b = \\gamma
    \\end{aligned}
    $$

    meaning that:

    $$
    \\begin{aligned}
    & \\alpha - \\gamma = 4ap \\\\[10pt]
    & p = \\frac{\\alpha - \\gamma}{4a} \\\\[10pt]
    & p = \\frac{\\alpha - \\gamma}{2(\\alpha -2\\beta +\\gamma)}
    \\end{aligned}
    $$

    Parameters
    ----------
    array : np.ndarray
        input array with 3 points, (before peak, peak and after peak)

    See Also
    --------
    https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html

    Returns
    -------
    float
        interpolated peak value
    float
        delta position between the old peak position (second value of input array) and new estimated position

    """
    alpha = array[0]  # before max
    beta = array[1]  # max
    gamma = array[2]  # after max
    peak_relative_position = (np.abs(alpha) - np.abs(gamma)) / (np.abs(alpha) - 2 * np.abs(beta) + np.abs(gamma)) / 2
    peak_value = beta - (alpha - gamma) * peak_relative_position / 4

    return peak_value, peak_relative_position


def interp1_modulated_data(
    data: np.ndarray,
    interp_factor: int,
    demodulation_flag: int,
    demodulation_frequency: np.ndarray,
) -> np.ndarray:
    """Interpolating input data along rows direction.

    Input 2D array is interpolated (i.e. oversampled) by a factor interp_factor. If input data is already demodulated,
    demodulation_flag can be provided <1, otherwise data can be demodulated before interpolation (and than re-modulated
    back before returning the results) if demodulation_flag is set >=1 and a demodulation frequency array is provided.

    Parameters
    ----------
    data : np.ndarray
        2D array
    interp_factor : int
        interpolation factor
    demodulation_flag : int
        modulation flag, if >1 modulation is applied before interpolation (and removed after), otherwise data is left
        as it is
    demodulation_frequency : np.ndarray
        modulation frequency values to be used for modulation, if needed

    Returns
    -------
    np.ndarray
        interpolated 2D array

    """
    # retrieve dimensions of input data to be interpolated (rows direction)
    data = data.copy()
    n_rg, _ = data.shape

    if demodulation_flag >= 1:
        # if modulation is required, data are FFT transformed in other domain (along columns)
        # as in estimate_modulation_frequency2d
        data_portion_fft = np.fft.fftn(data, axes=(1,))

        # apply demodulation
        data_portion_fft = modulate_data(data=data_portion_fft, mod_freq=-demodulation_frequency)
    else:
        data_portion_fft = data

    # apply zero padding in time domain (x2 factor)
    data_portion_fft = np.concatenate((data_portion_fft, np.zeros(data_portion_fft.shape)), axis=0)

    # signal interpolation
    data_portion_fft_int = signal.resample(data_portion_fft, interp_factor * 2 * n_rg, axis=0)

    # remove padded data
    data_portion_fft_int = data_portion_fft_int[0 : interp_factor * n_rg, :]

    # to keep consistency with input 2D array, if modulation has been applied inside this function,
    # data must be re-modulated before being returned

    if demodulation_flag >= 1:
        # apply interpolation factor to local frequencies
        frequency_vect = demodulation_frequency / interp_factor

        # apply re-modulation
        data_portion_fft_int = modulate_data(data=data_portion_fft_int, mod_freq=frequency_vect)

        # returning to time domain
        data = np.fft.ifftn(data_portion_fft_int, axes=(1,))
    else:
        data = data_portion_fft_int

    return data


def interp2_modulated_data(
    data: np.ndarray,
    interp_factor_az: int,
    interp_factor_rng: int,
    demod_flag_az: bool = False,
    demod_flag_rng: bool = False,
) -> np.ndarray:
    """This functions applies the interp1_modulated_data on both axis of the input 2D array. It is used to interpolate
    both along azimuth and range. It performs also modulation before interpolating data, if needed.

    Parameters
    ----------
    data : np.ndarray
        2D array
    interp_factor_az : int
        interpolation factor along azimuth direction
    interp_factor_rng : int
        interpolation factor along range direction
    demod_flag_az : bool, optional
        if True demodulation frequency for azimuth is estimated and data are demodulate before interpolating them.
        At the end of the operation, if data have been demodulate, they are re-modulated, by default False
    demod_flag_rng : bool, optional
        if True demodulation frequency for range is estimated and data are demodulate before interpolating them.
        At the end of the operation, if data have been demodulate, they are re-modulated, by default False

    Returns
    -------
    np.ndarray
        interpolated 2D array along both axes

    """
    data = data.copy()

    # estimating demodulation frequencies if demodulation is required
    if demod_flag_az:
        f_az, f_az_vect = estimate_modulation_frequency(data.T, axis=1)
        f_az_vect = f_az * np.ones((f_az_vect.shape[0],))
    if demod_flag_rng:
        f_rg, f_rg_vect = estimate_modulation_frequency(data, axis=0)
        f_rg_vect = f_rg * np.ones((f_rg_vect.shape[0] * interp_factor_az,))

    # Perform interpolation along azimuth direction
    if interp_factor_az > 1:
        data_interpolated = interp1_modulated_data(data.T, interp_factor_az, demod_flag_az, f_az_vect)
        data_interpolated = data_interpolated.T

    # Perform interpolation along range direction
    if interp_factor_rng > 1:
        data_interpolated = interp1_modulated_data(data_interpolated, interp_factor_rng, demod_flag_rng, f_rg_vect)

    return data_interpolated


def evaluate_irf_resolution(profile: np.ndarray) -> float:
    """Finding the resolution of the main lobe in pixel.

    Parameters
    ----------
    profile : np.ndarray
        profile to be analyzed

    Returns
    -------
    float
        resolution of the main lobe in pixel

    """
    max_prof = np.abs(profile).max()
    profile = profile.copy() / max_prof
    profile_m3db = convert_to_db(np.abs(profile**2)) - convert_to_db(0.5)
    profile_m3db = profile_m3db.squeeze()
    # indexes of main lobe above 3db
    indexes = np.where(profile_m3db > 0)[0]

    break_cond = np.logical_or.reduce((indexes.size == 0, indexes[0] == 0, indexes[-1] == profile.size - 1))

    if break_cond:
        return np.nan

    # peak width is where values are > maximum - 3dB
    dsx = 1 + profile_m3db[(indexes[0] - 1)] / (profile_m3db[indexes[0]] - profile_m3db[(indexes[0] - 1)])
    ddx = -profile_m3db[indexes[-1]] / (profile_m3db[indexes[-1] + 1] - profile_m3db[indexes[-1]])

    return (indexes.size - 1) + ddx + dsx


def compute_intensity_background(
    data: np.ndarray,
    resolutions_px: tuple[float, float],
    roi: tuple[float, float],
    margin: int = 10,
) -> tuple[float, list]:
    """Compute image intensity background.

    Parameters
    ----------
    data : np.ndarray
        input 2D array
    resolutions_px : tuple[float, float]
        range[0] and azimuth[1] pixel resolutions
    roi : tuple[float, float]
        roi size, range [0] and azimuth[1]
    margin : int, optional
        margin from array borders, by default 10

    Returns
    -------
    float
        intensity background
    list
        list of range and azimuth positions

    """
    # unpacking inputs
    roi_rng, roi_az = roi

    margin_rg = np.ceil(margin * resolutions_px[0])
    margin_az = np.ceil(margin * resolutions_px[1])

    # creating a mask on the image to select indented square portions of the
    # background, one for each corner, properly indented from the border
    pos_rng_1 = np.array(
        [
            margin,
            np.max([roi_rng - margin - margin_rg, roi_rng // 2 + margin]) - 1,
            margin,
            np.max([roi_rng - margin - margin_rg, roi_rng // 2 + margin]) - 1,
        ],
    ).astype("int64")
    pos_rng_2 = np.array(
        [
            np.min([pos_rng_1[0] + margin_rg, roi_rng // 2 - margin]),
            np.min([pos_rng_1[1] + margin_rg, roi_rng - margin]),
            np.min([pos_rng_1[2] + margin_rg, roi_rng // 2 - margin]),
            np.min([pos_rng_1[3] + margin_rg, roi_rng - margin]),
        ],
    ).astype("int64")
    pos_az_1 = np.array(
        [
            margin,
            margin,
            np.max([roi_az - margin - margin_az, roi_az // 2 + margin]) - 1,
            np.max([roi_az - margin - margin_az, roi_az // 2 + margin]) - 1,
        ],
    ).astype("int64")
    pos_az_2 = np.array(
        [
            np.min([pos_az_1[0] + margin_az, roi_az // 2 - margin]),
            np.min([pos_az_1[1] + margin_az, roi_az // 2 - margin]),
            np.min([pos_az_1[2] + margin_az, roi_az - margin]),
            np.min([pos_az_1[3] + margin_az, roi_az - margin]),
        ],
    ).astype("int64")

    rect_roi = [(pos_rng_1[p], pos_rng_2[p], pos_az_1[p], pos_az_2[p]) for p in range(pos_rng_1.size)]
    intensity_background = sum([np.sum(data[p[0] : p[1], p[2] : p[3]]) for p in rect_roi]) / (4 * margin_rg * margin_az)

    return intensity_background, rect_roi


def compute_integrated_peak_intensity(
    data: np.ndarray,
    peak_position: np.ndarray,
    resolutions_px: tuple[float, float],
    interp_factor: int = 8,
    margin: int = 20,
) -> tuple[float, list]:
    """Computing the integrated peak intensity of the input array as sum of
    values inside a calculated roi.

    Parameters
    ----------
    data : np.ndarray
        input array
    peak_position : np.ndarray
        peak position inside array [row, col]
    resolutions_px : tuple[float, float]
        resolution [range, azimuth] in pixels
    interp_factor : int, optional
        interpolation factor, by default 8
    margin : int, optional
        roi margin from the array borders, by default 20

    Returns
    -------
    float
        integrated peak intensity over roi
    list
        list of peak roi corners

    """
    # integrate the interpolated corrected data intensity
    margin_rng_int = np.ceil(margin * resolutions_px[0] * interp_factor)
    margin_az_int = np.ceil(margin * resolutions_px[1] * interp_factor)

    # evaluating roi range start index
    index = int(peak_position[0] - np.ceil(margin_rng_int / 2)) - 1
    rng_start = max(1, index)

    # evaluating roi range end index
    index = int(peak_position[0] + (margin_rng_int - np.ceil(margin_rng_int / 2) - 1))
    rng_end = min(data.shape[0], index)

    # evaluating roi azimuth start index
    index = int(peak_position[1] - np.ceil(margin_az_int / 2)) - 1
    az_start = max(1, index)

    # evaluating roi azimuth end index
    index = int(peak_position[1] + (margin_az_int - np.ceil(margin_az_int / 2) - 1))
    az_end = min(data.shape[1], index)

    # integrating over values inside of the roi
    integrated_peak_intensity = np.sum(data[rng_start:rng_end, az_start:az_end])

    return integrated_peak_intensity, [rng_start, rng_end, az_start, az_end]


def padded_hamming_windowing(
    num_points: int,
    alpha: float,
    zero_pad_side_len: int = 0,
    top_plateau_len: int = 0,
) -> np.ndarray:
    """Padded hamming windowing. Hamming window with the possibility to add zeroes on sides and extend the peak with
    a plateau of ones.

    Parameters
    ----------
    num_points : int
        number of points of un-padded hamming window
    alpha : float
        alpha hamming parameter
    zero_pad_side_len : int, optional
        zeroes padding length on each side, by default 0
    top_plateau_len : int, optional
        ones padding length for peak extension, by default 0

    Returns
    -------
    np.ndarray
        padded hamming window

    """
    # computing the hamming window
    hamming_window = signal.windows.general_hamming(num_points, alpha)

    # additional window configuration
    first_half_window = hamming_window[: num_points // 2]
    second_half_window = hamming_window[num_points // 2 :]
    side_zero_padding = np.zeros(zero_pad_side_len)
    top_plateau_padding = np.ones(top_plateau_len)

    # assembling the padded window
    return np.hstack(
        [side_zero_padding, first_half_window, top_plateau_padding, second_half_window, side_zero_padding],
    )


def locate_max_2d(data: np.ndarray) -> tuple[int, int]:
    """Function used to determine the indexes of the maximum value in a 2D array.

    Parameters
    ----------
    data : np.ndarray
        input array where to find the maximum

    Returns
    -------
    int
        row max index
    int
        column max index

    """
    indexes = np.unravel_index(data.argmax(), data.shape)

    return indexes[0], indexes[1]


def locate_max_2d_interp(
    data: np.ndarray,
    interp_factor: int = 8,
    demod_flag_az: bool = True,
    demod_flag_rng: bool = True,
) -> tuple[float, float, float]:
    """This function shifts the input data by modulating it, compute an FFT, oversamples the data to find
    the peak coordinates with sub-pixel accuracy.

    Parameters
    ----------
    data : np.ndarray
        input 2D array
    interp_factor : int, optional
        interpolating factor, by default 8
    demod_flag_az : bool, optional
        if True demodulation frequency for azimuth is estimated and data are demodulate before interpolating them.
        At the end of the operation, if data have been demodulate, they are re-modulated, by default True
    demod_flag_rng : bool, optional
        if True demodulation frequency for range is estimated and data are demodulate before interpolating them.
        At the end of the operation, if data have been demodulate, they are re-modulated, by default True

    Returns
    -------
    float
        peak value
    float
        row peak coordinate (subpixel)
    float
        column peak coordinate (subpixel)

    """
    # Coarse peak estimation
    y_max_pos_coarse, x_max_pos_coarse = locate_max_2d(np.abs(data))
    # range/azimuth cuts extraction
    slice_x = data[[y_max_pos_coarse], :]
    slice_y = data[:, [x_max_pos_coarse]]

    # Oversampling each slice by a factor interpolation_factor, with demodulation if needed
    slice_x = interp2_modulated_data(
        data=slice_x,
        interp_factor_az=interp_factor,
        interp_factor_rng=1,
        demod_flag_az=demod_flag_az,
        demod_flag_rng=demod_flag_rng,
    )
    slice_y = interp2_modulated_data(
        data=slice_y.transpose().conjugate(),
        interp_factor_az=interp_factor,
        interp_factor_rng=1,
        demod_flag_az=demod_flag_az,
        demod_flag_rng=demod_flag_rng,
    )

    # Coarse peak estimation for each interpolated slice
    x_max_pos_coarse = np.argmax(np.abs(slice_x[0, :]))
    x_max_pos_coarse = np.min([np.max([2, x_max_pos_coarse]), slice_x.size - 1])
    y_max_pos_coarse = np.argmax(np.abs(slice_y[0, :]))
    y_max_pos_coarse = np.min([np.max([2, y_max_pos_coarse]), slice_y.size - 1])

    # Interpolation around maximum coordinates with parabolic fitting around 3 points near maximum
    # to better estimate the the peak position (subpixel precision), one direction at a time
    _, x_delta_position = parabolic_interp_by_3_closest_samples(
        np.abs(slice_x[0, x_max_pos_coarse - 1 : x_max_pos_coarse + 2]),
    )
    _, y_delta_position = parabolic_interp_by_3_closest_samples(
        np.abs(slice_y[0, y_max_pos_coarse - 1 : y_max_pos_coarse + 2]),
    )

    # Final peak position in [x y] coordinate and index correction.
    x_max_pos = x_delta_position + x_max_pos_coarse
    y_max_pos = y_delta_position + y_max_pos_coarse

    x_max_pos = x_max_pos / interp_factor
    y_max_pos = y_max_pos / interp_factor

    y_axis = np.arange(data.shape[0]) - y_max_pos
    x_axis = np.arange(data.shape[1]) - x_max_pos

    filter_rg = np.sinc(y_axis)
    filter_az = np.sinc(x_axis)
    peak_value = np.matmul(filter_rg, np.matmul(data, filter_az))

    return peak_value, y_max_pos, x_max_pos


def shift_array(
    data: np.ndarray, row_shift: float, col_shift: float, integer_shift_tolerance: float = 1.0e-3
) -> np.ndarray:
    """Shift data in the time domain.

    A rigid shift and a zero padding or a phase ramp in frequency, depending if the shift is approximately an integer.


    Parameters
    ----------
    data : np.ndarray
        input 2D array to be shifted
    row_shift : float
        shift expressed in samples
    col_shift : float
        shift expressed in lines

    Returns
    -------
    np.ndarray
        shifted 2D array
    """
    integer_shift = (
        np.abs(row_shift - np.round(row_shift)) < integer_shift_tolerance
        and np.abs(col_shift - np.round(col_shift)) < integer_shift_tolerance
    )

    if integer_shift:
        row_shift = int(np.round(row_shift))
        col_shift = int(np.round(col_shift))

        rows, cols = data.shape

        padding = np.zeros((np.abs(row_shift), cols))
        if row_shift < 0:
            data = np.vstack((padding, data[: -abs(row_shift), :]))
        elif row_shift > 0:
            data = np.vstack((data[abs(row_shift) :, :], padding))

        padding = np.zeros((rows, abs(col_shift)))
        if col_shift < 0:
            data = np.hstack((padding, data[:, : -abs(col_shift)]))
        elif col_shift > 0:
            data = np.hstack((data[:, abs(col_shift) :], padding))

    else:
        rows, cols = data.shape

        # estimate range and azimuth modulation frequency
        rng_freq, rng_frq_vect, az_freq, az_freq_vect = estimate_modulation_frequency2d(data)
        rng_frq_vect = rng_freq * np.ones(rng_frq_vect.shape)
        az_freq_vect = az_freq * np.ones(az_freq_vect.shape)

        fft_data = np.fft.fft2(data)

        rng_freq_axis = get_frequency_axis(rng_frq_vect, 1, rows)
        az_freq_axis = get_frequency_axis(az_freq_vect, 1, cols)

        phi = np.exp(1j * 2 * np.pi * (az_freq_axis * col_shift + rng_freq_axis.T * row_shift))
        fft_data *= phi

        data = np.fft.ifft2(fft_data)

    return data


def crop_array_2d(
    data: np.ndarray,
    crop_size: tuple,
    indexes: tuple[int | None, int | None] = (None, None),
) -> np.ndarray:
    """Cropping the input 2D array around specific indexes (if provided, otherwise the center of the array)
    with cropping length and width equal to the specified values crop_size.

    Parameters
    ----------
    data : np.ndarray
        input 2D array to be cropped
    crop_size : tuple
        cropping size, one value for each axis
    indexes : tuple, optional
        if provided, cropping around these indexes, otherwise cropping around center of array, by default (None, None)

    Returns
    -------
    np.ndarray
        cropped array

    Raises
    ------
    ValueError
        Cropping size out of array boundaries
    """

    provided_indexes = indexes[0] is not None and indexes[1] is not None
    indexes = indexes if provided_indexes else (data.shape[0] // 2, data.shape[1] // 2)
    assert indexes[0] is not None and indexes[1] is not None

    line_crop_size = int(crop_size[0])
    line_start = indexes[0] - line_crop_size // 2
    line_end = indexes[0] + line_crop_size // 2
    sample_crop_size = int(crop_size[1])
    sample_start = indexes[1] - sample_crop_size // 2
    sample_end = indexes[1] + sample_crop_size // 2

    if line_start < 0 or sample_start < 0 or line_end > data.shape[0] or sample_end > data.shape[1]:
        raise ValueError("Cropping size out of array boundaries")

    return data[line_start:line_end, sample_start:sample_end].copy()


def radiometric_correction(
    data: np.ndarray,
    incidence_angle: np.ndarray,
    input_quantity: SARRadiometricQuantity,
    output_quantity: SARRadiometricQuantity,
    exp_power: float = 0.5,
) -> np.ndarray:
    """Data radiometric correction based on data acquisition type and desired output type, choosing between Beta, Sigma
    and Gamma Nought.

    Parameters
    ----------
    data : np.ndarray
        input array whose data need to be converted from a type to another, array should be in the form (samples, lines)
    incidence_angle : np.ndarray
        incidence angle array, same size as range axis of input data or same data size: (n_rng,) or (n_rng, n_az)
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
        msg = "Input and output type must be of type SARRadiometricQuantity"
        raise ValueError(msg)

    if incidence_angle.ndim == 1:
        incidence_angle = np.atleast_2d(incidence_angle).T
    elif data.shape != incidence_angle.shape:
        msg = f"Incidence angle shape must be (n_rng,) or input data shape {data.shape}"
        raise ValueError(msg)

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


def compute_doppler_rate_theoretical(
    trajectory: TwiceDifferentiable3DCurve,
    azimuth_time: PreciseDateTime,
    coords: np.ndarray,
    fc_hz: float,
) -> np.ndarray:
    """Compute theoretical doppler rate.

    Parameters
    ----------
    trajectory : TwiceDifferentiable3DCurve
        sensor trajectory
    azimuth_time : PreciseDateTime
        azimuth time when to evaluate the doppler rate
    coords : np.ndarray
        ground point coordinates
    fc_hz : float
        signal carrier frequency

    Returns
    -------
    np.ndarray
        theoretical doppler rate

    """
    sat_pos = trajectory.evaluate(azimuth_time)
    sat_vel = trajectory.evaluate_first_derivatives(azimuth_time)
    sat_acc = trajectory.evaluate_second_derivatives(azimuth_time)

    los = (sat_pos - coords).transpose()
    los_norm = np.linalg.norm(los)

    return (
        -2
        / (speed_of_light / fc_hz)
        / los_norm
        * (np.linalg.norm(sat_vel) ** 2 + float(np.dot(los, sat_acc)) - (float(np.dot(los, sat_vel)) / los_norm) ** 2)
    )


def compute_steering_doppler_frequency(
    trajectory: TwiceDifferentiable3DCurve,
    azimuth_time: PreciseDateTime,
    az_mid_burst_time: PreciseDateTime,
    doppler_rate: float,
    az_steering_rate_rad_s: float,
    fc_hz: float,
) -> float:
    """Compute doppler frequency related to the antenna electrical steering.

    Parameters
    ----------
    trajectory : TwiceDifferentiable3DCurve
        sensor trajectory
    azimuth_time : PreciseDateTime
        azimuth time at which compute the steering frequency
    az_mid_burst_time : PreciseDateTime
        azimuth mid burst time
    doppler_rate : float
        sensor doppler rate
    az_steering_rate_rad_s : float
        azimuth steering rate in rad/s
    fc_hz : float
        signal carrier frequency

    Returns
    -------
    float
        steering doppler frequency

    """
    sat_vel_norm = np.linalg.norm(trajectory.evaluate_first_derivatives(azimuth_time))
    # azimuth steering rate conversion from rad/s to Hz/s
    az_steering_rate_hz_s = 2 * sat_vel_norm / (speed_of_light / fc_hz) * az_steering_rate_rad_s
    # antenna modulation rate
    antenna_modulation_rate = -doppler_rate * az_steering_rate_hz_s / (az_steering_rate_hz_s - doppler_rate)

    return antenna_modulation_rate * (azimuth_time - az_mid_burst_time)


def compute_point_target_ambiguity_ratio_db(
    point_target_roi: np.ndarray,
    right_ambiguity_roi: np.ndarray,
    left_ambiguity_roi: np.ndarray,
    interp_factor: int,
) -> float:
    """Computing the Point Target Ambiguity Ratio (PTAR).

    This parameter is computed using the following formula:

    $$
    PTAR = 20\\log_{10}\\left(\\frac{{|I_{amb_{left}}| + |I_{amb_{right}}|}}{2|I_{pt}|}\\right)
    $$

    Parameters
    ----------
    point_target_roi : np.ndarray
        raster data portion centered on the point target location, with shape (samples, lines)
    right_ambiguity_roi : np.ndarray
        raster data portion centered on the point target right ambiguity, with shape (samples, lines)
    left_ambiguity_roi : np.ndarray
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

    return convert_to_db(
        np.nanmean([np.abs(r_ambiguity_peak_value), np.abs(l_ambiguity_peak_value)]) / np.abs(pt_peak_value),
        mode=DecibelConversion.AMPLITUDE,
    )


def compute_distributed_target_ambiguity_ratio_db(
    distributed_target_roi: np.ndarray,
    right_ambiguity_roi: np.ndarray,
    left_ambiguity_roi: np.ndarray,
) -> float:
    """Computing the Distributed Target Ambiguity Ratio (DTAR).

    This parameter is computed using the following formula:

    $$
    DTAR = \\frac{E(\\Sigma |amb_{left}|^2) + E(\\Sigma |amb_{right}|^2)}{2*E(\\Sigma |target|^2)}
    $$

    Parameters
    ----------
    distributed_target_roi : np.ndarray
        raster data portion centered on the point target location, with shape (samples, lines)
    right_ambiguity_roi : np.ndarray
        raster data portion centered on the point target right ambiguity, with shape (samples, lines)
    left_ambiguity_roi : np.ndarray
        raster data portion centered on the point target left ambiguity, with shape (samples, lines)

    Returns
    -------
    float
        Distributed Target Ambiguity Ratio

    """
    distributed_peak = np.sum(np.abs(distributed_target_roi) ** 2) / distributed_target_roi.size
    r_ambiguity_distributed_peak = np.sum(np.abs(right_ambiguity_roi) ** 2) / right_ambiguity_roi.size
    l_ambiguity_distributed_peak = np.sum(np.abs(left_ambiguity_roi) ** 2) / left_ambiguity_roi.size
    return convert_to_db(
        np.nanmean([np.abs(r_ambiguity_distributed_peak), np.abs(l_ambiguity_distributed_peak)])
        / np.abs(distributed_peak),
    )


def compute_equivalent_number_of_looks(intensity_data: np.ndarray) -> float:
    """Computing Equivalent Number of Looks (ENL) from input roi data.

    $$
    ENL = \\frac{I_{mean}^2}{I_{std}^2}
    $$

    Parameters
    ----------
    intensity_data : np.ndarray
        intensity of 2D roi data image (abs ** 2), with shape (samples, lines)

    Returns
    -------
    float
        Equivalent Number of Looks (ENL) for the given data region

    """

    intensity_mean = np.nanmean(intensity_data)
    intensity_std = np.nanstd(intensity_data)

    return float((intensity_mean**2) / (intensity_std**2))
