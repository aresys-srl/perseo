# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Processing - Signal Features
----------------------------
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numpydantic import NDArray, Shape
from scipy import signal

from perseo_core.models.enums import GetFrequencyMethod

InverseROIArrayType = NDArray[Shape["* lines, * samples"], float] | NDArray[Shape["2"], float]  # type: ignore

# TODO: this whole module should be typed using the Types defined in perseo-core, but to do this, it must be
# TODO: understood the right ROI axes in 2D data processing for each function and maybe a full function refactoring must
# TODO: be done


def shift_spectrum(
    data: InverseROIArrayType, shift: float | npt.NDArray[np.floating], axis: int = 0
) -> InverseROIArrayType:
    """Shift the spectrum of time domain signal along the selected axis by applying a time domain modulation.

    The shift can be a single value or an array of values with the size of the other axis.
    This allows to apply a shift that varies along the other axis.
    For example, in case of an (azimuth, range) signal a range variant azimuth spectral shift can be specified.

    shift is normalized: to apply a shift 'df' in Hertz you need to set 'shift' to 'df * dt'.

    Parameters
    ----------
    data : InverseROIArrayType
        signal of shape (lines, samples)
    shift : float | npt.NDArray[np.floating]
        either a constant or an array of shape (lines,) or (samples,) depending on the specified axis.
    axis : int, optional
        axis along which the spectral shift is applied, by default 0

    Returns
    -------
    InverseROIArrayType
        the modulated signal of shape (lines, samples)
    """
    normalized_time = 2 * np.pi * np.arange(data.shape[axis])

    if axis == 0:
        phase = normalized_time.reshape(-1, 1) * np.asarray(shift)
    else:
        phase = normalized_time * np.asarray(shift).reshape(-1, 1)

    if np.ndim(data) == 1:
        phase = phase.squeeze()

    return data * np.exp(1j * phase)


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
        # ????  # TODO: ?
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


def modulate_data(data: InverseROIArrayType, mod_freq: float | npt.NDArray[np.floating]) -> InverseROIArrayType:
    """Time domain data modulation to shift a signal in the frequency domain along the selected axis.
    Signal spectrum is shifted by mod_freq value.

    The modulation frequency can be a single value or an array of values with the size of the other axis.
    This allows to apply frequency shift that varies along the other axis.
    For example, if the signal is of shape (azimuth, range) you can specify a range variant azimuth shift.

    mod_freq is normalized, i.e. to apply a shift df in Hertz you need to set mod_freq = df * dt.

    Parameters
    ----------
    data : InverseROIArrayType
        signal of shape (lines, samples)
    mod_freq : float | npt.NDArray[np.floating]
        either a constant or an array of shape (lines,) or (samples,) depending on the axis.

    Returns
    -------
    InverseROIArrayType
        the modulated signal of shape (L, S)
    """
    return shift_spectrum(data, mod_freq, axis=0)


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


def parabolic_interp_by_3_closest_samples(array: np.ndarray) -> tuple[float, float]:
    """Parabolic peak interpolation using the three samples closest to the peak.

    Fitting a parabola to the 3-points input array, containing the closest point before the peak, the peak itself and
    the closest point after the peak.

    Considering a parabola written with explicit dependency from the position of its interpolated peak location in bins

    .. math::

        y(x)\\overset{\\Delta}{=}a(x-p)^2+b

    at the three samples nearest the peak, considering their bins as -1 (before), 0 (peak), 1 (after) we have:

    .. math::

        y(-1) = ap^2+2ap+a+b = \\alpha
        y(0) = ap^2+b = \\beta
        y(1) = ap^2-2ap+a+b = \\gamma

    meaning that:

    .. math::

        \\alpha - \\gamma = 4ap
        p = \\frac{\\alpha - \\gamma}{4a}
        p = \\frac{\\alpha - \\gamma}{2(\\alpha -2\\beta +\\gamma)}

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
    interp_factor_rg : int
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
