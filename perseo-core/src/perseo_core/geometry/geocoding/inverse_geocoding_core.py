# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Inverse geocoding core utilities"""

from __future__ import annotations

import numpy as np
from scipy.constants import speed_of_light
from scipy.interpolate import interp1d

from perseo_core.geometry.doppler import doppler_equation_monostatic_residuals
from perseo_core.models.trajectory import TwiceDifferentiable3DCurve
from perseo_core.models.types import (
    CoordinatesArrayType,
    ExtendedDatetimeArrayType,
    ExtendedDatetimeType,
    FloatArrayType,
)


def inverse_geocoding_monostatic_core(
    trajectory: TwiceDifferentiable3DCurve,
    ground_points: CoordinatesArrayType,
    initial_guesses: ExtendedDatetimeType | ExtendedDatetimeArrayType,
    frequencies_doppler_centroid: float | FloatArrayType,
    wavelength: float,
    scene_velocity: CoordinatesArrayType | None = None,
    abs_time_tolerance: float = 1e-8,
    max_iter: int = 8,
) -> tuple[ExtendedDatetimeType | ExtendedDatetimeArrayType, float | FloatArrayType]:
    """Core algorithm for precise inverse geocoding monostatic.

    This function finds one zero of the doppler equation as a function of the azimuth time. The search of the zero is
    restricted to the interval associated to the polynomial indexed by polynomial_index. Newton iterations are performed
    to find the exact azimuth time, once that the sensor position is known the computation of the range time is trivial.

    Vectorized function with the following supported cases:

        - 1 guess and 1 earth point --> PDT, float
        - N guesses and N earth points --> array[PDT], array[float]
        - 1 guess and N earth points --> array[PDT], array[float]
        - N guesses and 1 earth point --> array[PDT], array[float]
        - N earth points and N frequencies and 1 guess --> array[PDT], array[float]
        - N earth points and N frequencies and N guess --> array[PDT], array[float]
        - 1 earth point and N frequencies and N guess --> array[PDT], array[float]
        - 1 earth point and N frequencies and 1 guess --> array[PDT], array[float]

    Parameters
    ----------
    trajectory : TwiceDifferentiable3DCurve
        sensor's trajectory, compliant to the TwiceDifferentiable3DCurve protocol
    ground_points : CoordinatesArrayType
        ground points to inverse geocode in XYZ coordinates, in the form (3,) or (N, 3)
    initial_guesses : ExtendedDatetimeType | ExtendedDatetimeArrayType
        azimuth times initial guesses to limit and guide the search of solutions, in the form (N,) or as a single time
    frequencies_doppler_centroid : float | FloatArrayType
        doppler frequencies centroid values to perform the inverse geocoding, in the form float or (N,).
        the number of frequencies must be 1 or equal to the number of points provided (if more than 1).
        If just 1 ground point is provided, several frequencies can be given to compute inverse geocoding at
        each different input value
    wavelength : float
        carrier signal wavelength
    scene_velocity : CoordinatesArrayType | None, optional
        ground scene velocity. If None, it's set to the default value of 0, by default None
    abs_time_tolerance : float, optional
        absolute time tolerance for newton convergence criteria, by default 1e-8
    max_iter : int, optional
        maximum number of iterations for newton method, by default 8

    Returns
    -------
    ExtendedDatetimeType | ExtendedDatetimeArrayType
        azimuth times array
    float | FloatArrayType
        range times array

    Raises
    ------
    RuntimeError
        ambiguous association between N ground points and M init guesses
    RuntimeError
        ambiguous association between N ground points and M frequencies
    RuntimeError
        Newton method could not converge
    """

    # input conversion and management
    ground_points = np.asarray(ground_points)
    azimuth_times = np.asarray(initial_guesses).copy()
    frequencies_doppler_centroid = (
        np.asarray(frequencies_doppler_centroid)
        if not np.isscalar(frequencies_doppler_centroid)
        else frequencies_doppler_centroid
    )
    if scene_velocity is None:
        scene_velocity = np.zeros(3)
    else:
        scene_velocity = np.asarray(scene_velocity)

    if np.size(initial_guesses) != ground_points.size // 3 and not (
        ground_points.size // 3 == 1 or np.size(initial_guesses) == 1
    ):
        raise RuntimeError(
            "Ambiguous matching between initial guess times "
            + f"{initial_guesses.shape} and "
            + f"ground points {ground_points.shape}"
        )

    if np.size(frequencies_doppler_centroid) != ground_points.size // 3 and not (
        ground_points.size // 3 == 1 or np.size(frequencies_doppler_centroid) == 1
    ):
        raise RuntimeError(
            "Ambiguous matching between doppler frequencies "
            + f"{frequencies_doppler_centroid.shape} and "
            + f"ground points {ground_points.shape}"
        )

    # starting the newton method to solve the equation
    for _ in range(max_iter):
        # computing orbit position, first and second derivatives
        sensor_position = trajectory.evaluate(azimuth_times)
        sensor_velocity = trajectory.evaluate_first_derivatives(azimuth_times)
        sensor_acceleration = trajectory.evaluate_second_derivatives(azimuth_times)

        # slant range correspondent to the actual position of satellite
        line_of_sight = ground_points - sensor_position
        slant_range = np.linalg.norm(line_of_sight, axis=-1)

        # function to be computed and solved with Newton method
        # equation:
        # f = [(earth_point - sensor_position) * (scene_velocity - sat_velocity)] + doppler_term
        doppler_term = wavelength * frequencies_doppler_centroid / 2.0 * slant_range
        func = np.sum((line_of_sight * (scene_velocity - sensor_velocity)), axis=-1) + doppler_term

        # derivative equation:
        # df = [-sensor_velocity*(sensor_velocity - scene_velocity) - (earth_point - sensor_pos) * sensor_acceleration -
        # (earth_point - sensor_pos) * sensor_velocity * lambda * doppler_freq / 2 / slant_range]
        func_der = (
            -np.sum(sensor_velocity * (scene_velocity - sensor_velocity), axis=-1)
            - np.sum(sensor_acceleration * line_of_sight, axis=-1)
            - np.sum(sensor_velocity * line_of_sight, axis=-1)
            * wavelength
            * frequencies_doppler_centroid
            / 2
            / slant_range
        )

        # updating the current value
        delta_err = func / func_der
        azimuth_times = azimuth_times - delta_err

        if np.max(np.abs(delta_err)) <= abs_time_tolerance:
            break

    else:
        raise RuntimeError(
            "Newton did not converge: maximum number of iterations" + f"{max_iter} reached. Residual error {delta_err}"
        )

    # re-evaluating slant range
    sensor_pos_curr = trajectory.evaluate(azimuth_times)
    line_of_sight = ground_points - sensor_pos_curr
    slant_range = np.linalg.norm(line_of_sight, axis=-1) * 2 / speed_of_light

    return azimuth_times, slant_range


def inverse_geocoding_bistatic_core(
    trajectory_rx: TwiceDifferentiable3DCurve,
    trajectory_tx: TwiceDifferentiable3DCurve,
    ground_points: CoordinatesArrayType,
    initial_guesses: ExtendedDatetimeType | ExtendedDatetimeArrayType,
    frequencies_doppler_centroid: float | FloatArrayType,
    wavelength: float,
    abs_time_tolerance: float = 1e-8,
    max_iter: int = 8,
) -> tuple[ExtendedDatetimeType | ExtendedDatetimeArrayType, float | FloatArrayType]:
    """Core algorithm for precise inverse geocoding bistatic.

    Vectorized function with the following supported cases:

        - 1 guess and 1 earth point --> PDT, float
        - N guesses and N earth points --> array[PDT], array[float]
        - 1 guess and N earth points --> array[PDT], array[float]
        - N guesses and 1 earth point --> array[PDT], array[float]
        - N guesses and N frequencies --> array[PDT], array[float]
        - N earth points and N frequencies and 1 guess --> array[PDT], array[float]
        - N earth points and N frequencies and N guess --> array[PDT], array[float]
        - 1 earth point and N frequencies and N guess --> array[PDT], array[float]
        - 1 earth point and N frequencies and 1 guess --> array[PDT], array[float]

    Parameters
    ----------
    trajectory_rx : TwiceDifferentiable3DCurve
        receiving sensor's trajectory, compliant to the TwiceDifferentiable3DCurve protocol
    trajectory_tx : TwiceDifferentiable3DCurve
        transmitting sensor's trajectory, compliant to the TwiceDifferentiable3DCurve protocol
    ground_points : CoordinatesArrayType
        ground points to inverse geocode in XYZ coordinates, in the form (3,) or (N, 3)
    initial_guesses : ExtendedDatetimeType | ExtendedDatetimeArrayType
        azimuth times initial guesses to limit and guide the search of solutions, in the form (N,) or as a single time
    frequencies_doppler_centroid : float | FloatArrayType
        doppler frequencies centroid values to perform the inverse geocoding, in the form float or (N,).
        the number of frequencies must be 1 or equal to the number of points provided (if more than 1).
        If just 1 ground point is provided, several frequencies can be given to compute inverse geocoding at
        each different input value
    wavelength : float
        carrier signal wavelength
    abs_time_tolerance : float, optional
        absolute time tolerance for newton convergence criteria, by default 1e-8
    max_iter : int, optional
        maximum number of iterations for newton method, by default 8

    Returns
    -------
    ExtendedDatetimeType | ExtendedDatetimeArrayType
        azimuth times array
    float | FloatArrayType
        range times array

    Raises
    ------
    RuntimeError
        ambiguous association between N ground points and M init guesses
    RuntimeError
        ambiguous association between N ground points and M frequencies
    RuntimeError
        Newton method could not converge
    """

    # input conversion and management
    ground_points = np.asarray(ground_points)
    num_points = ground_points.size // 3
    one_size_array = 0
    if ground_points.ndim != 1 and ground_points.size // 3 == 1:
        one_size_array = 1
    azimuth_times_rx = np.asarray(initial_guesses).copy()
    azimuth_times_tx = azimuth_times_rx.copy()
    frequencies_doppler_centroid = (
        np.asarray(frequencies_doppler_centroid)
        if not np.isscalar(frequencies_doppler_centroid)
        else frequencies_doppler_centroid
    )

    if azimuth_times_rx.size > ground_points.size // 3 == 1:
        ground_points = np.full((azimuth_times_rx.size, 3), ground_points)
        num_points = ground_points.size // 3

    if np.size(frequencies_doppler_centroid) > ground_points.size // 3 == 1:
        ground_points = np.full((np.size(frequencies_doppler_centroid), 3), ground_points)
        num_points = ground_points.size // 3

    if np.size(initial_guesses) != ground_points.size // 3 and not (
        ground_points.size // 3 == 1 or np.size(initial_guesses) == 1
    ):
        raise RuntimeError(
            "Ambiguous matching between initial guess times "
            + f"{initial_guesses.shape} and "
            + f"ground points {ground_points.shape}"
        )

    if np.size(frequencies_doppler_centroid) != ground_points.size // 3 and not (
        ground_points.size // 3 == 1 or np.size(frequencies_doppler_centroid) == 1
    ):
        raise RuntimeError(
            "Ambiguous matching between doppler frequencies "
            + f"{frequencies_doppler_centroid.shape} and "
            + f"ground points {ground_points.shape}"
        )

    # computing tx guess from rx by adding a delay
    position_rx = trajectory_rx.evaluate(azimuth_times_rx)
    position_tx = trajectory_tx.evaluate(azimuth_times_rx)
    delay_estimate = (
        np.linalg.norm(position_tx - ground_points, axis=-1, keepdims=True)
        + np.linalg.norm(position_rx - ground_points, axis=-1, keepdims=True)
    ) / speed_of_light
    azimuth_times_tx = np.array(azimuth_times_rx - delay_estimate.mean())

    # Newton method to solve the equation
    for _ in range(max_iter):
        # computing orbit position, first and second derivatives for sensor rx
        position_rx = trajectory_rx.evaluate(
            azimuth_times_rx,
        )
        line_of_sight_rx = ground_points - position_rx
        velocity_rx = trajectory_rx.evaluate_first_derivatives(
            azimuth_times_rx,
        )
        norm_vel_rx_square = np.sum(velocity_rx * velocity_rx, axis=-1)
        acceleration_rx = trajectory_rx.evaluate_second_derivatives(
            azimuth_times_rx,
        )

        # computing orbit position, first and second derivatives for sensor tx
        position_tx = trajectory_tx.evaluate(
            azimuth_times_tx,
        )
        line_of_sight_tx = ground_points - position_tx
        velocity_tx = trajectory_tx.evaluate_first_derivatives(
            azimuth_times_tx,
        )
        norm_vel_tx_square = np.sum(velocity_tx * velocity_tx, axis=-1)
        acceleration_tx = trajectory_tx.evaluate_second_derivatives(
            azimuth_times_tx,
        )

        # computing slant range
        slant_range_rx = np.linalg.norm(position_rx - ground_points, axis=-1)
        slant_range_tx = np.linalg.norm(position_tx - ground_points, axis=-1)
        slant_range = (azimuth_times_rx - azimuth_times_tx) * speed_of_light

        rng_vel_product_rx = np.sum(line_of_sight_rx * velocity_rx, axis=-1)
        rng_vel_product_tx = np.sum(line_of_sight_tx * velocity_tx, axis=-1)

        # equation residuals
        distance_equation_residual = slant_range_rx + slant_range_tx - slant_range
        doppler_equation_freq_term = wavelength * frequencies_doppler_centroid * slant_range_rx * slant_range_tx
        doppler_equation_residual = (
            -rng_vel_product_tx * slant_range_rx - rng_vel_product_rx * slant_range_tx
        ) + doppler_equation_freq_term

        # composing equations array
        equations = np.vstack([distance_equation_residual, doppler_equation_residual])

        # first derivative with respect to rx
        df1_dt_tx = rng_vel_product_tx / slant_range_tx + speed_of_light
        # first derivative with respect to tx
        df1_dt_rx = rng_vel_product_rx / slant_range_rx - speed_of_light

        # second derivative with respect to rx
        df2_dt_rx_doppler_freq_term = (
            rng_vel_product_rx
            / slant_range_rx
            * (rng_vel_product_tx + wavelength * frequencies_doppler_centroid * slant_range_tx)
        )
        df2_dt_rx = (
            slant_range_tx * (norm_vel_rx_square - np.sum(line_of_sight_rx * acceleration_rx, axis=-1))
            + df2_dt_rx_doppler_freq_term
        )

        # second derivative with respect to tx
        df2_dt_tx_doppler_freq_term = (
            rng_vel_product_tx
            / slant_range_tx
            * (rng_vel_product_rx + wavelength * frequencies_doppler_centroid * slant_range_rx)
        )
        df2_dt_tx = (
            slant_range_rx * (norm_vel_tx_square - np.sum(line_of_sight_tx * acceleration_tx, axis=-1))
            + df2_dt_tx_doppler_freq_term
        )

        # composing jacobian
        jac_det = df1_dt_tx * df2_dt_rx - df1_dt_rx * df2_dt_tx

        # assembling jacobian
        jac = np.zeros((num_points, 2, 2))
        jac[..., 0, 0] = df2_dt_rx
        jac[..., 0, 1] = -df1_dt_rx
        jac[..., 1, 0] = -df2_dt_tx
        jac[..., 1, 1] = df1_dt_tx

        # inverse of jacobian
        if jac.shape[0] > 1:
            inv_jac = np.array([m / jac_det[m_id] for m_id, m in enumerate(jac)])
            delta = np.sum(inv_jac * equations.T[:, None], axis=-1).T
        else:
            inv_jac = jac / jac_det
            delta = np.sum(inv_jac * equations.T, axis=-1).squeeze()

        # updating azimuth and range values
        azimuth_times_tx = azimuth_times_tx - delta[0]
        azimuth_times_rx = azimuth_times_rx - delta[1]
        slant_range = azimuth_times_rx - azimuth_times_tx

        delta_err = np.dot(delta, delta.T)

        if np.max(np.abs(delta_err)) <= abs_time_tolerance:
            break

    else:
        raise RuntimeError(
            "Newton did not converge: maximum number of iterations" + f"{max_iter} reached. Residual error {delta_err}"
        )

    if one_size_array:
        return np.atleast_1d(azimuth_times_rx), np.atleast_1d(slant_range).astype(float)

    if isinstance(slant_range, np.ndarray):
        slant_range = slant_range.astype(float)

    return azimuth_times_rx, slant_range


def inverse_geocoding_monostatic_init_core(
    trajectory: TwiceDifferentiable3DCurve,
    time_axis: ExtendedDatetimeArrayType,
    ground_points: CoordinatesArrayType,
    frequencies_doppler_centroid: float | FloatArrayType,
    wavelength: float,
) -> list[ExtendedDatetimeArrayType]:
    """Function to compute initialization of inverse geocoding.

    The azimuth time we are looking for is such that the doppler equation is equal to zero.
    In this function we look for the time interval where such equation crosses zero. We compute the corresponding
    polynomial index that will be used subsequently for the precise computation of the azimuth time.
    This function evaluates the doppler equation on the azimuth time axis of the gso. It checks where it changes sign
    (from plus to minus). For each of those time positions the index of polynomial to use for interpolating in that
    neighborhood is computed and returned. If no crossing points are found, either the first or the last polynomials
    ids are returned depending on the time instants that has smaller values of the doppler equation.

    In principle each input ground point could be seen several times by the sensor orbit if it contains
    multiple periods. In this case, all possible solutions are kept and returned in a list with an array
    for each input point and the size of the array matches the number of solutions found for that point.

    Parameters
    ----------
    trajectory : TwiceDifferentiable3DCurve
        sensor's trajectory, compliant to the TwiceDifferentiable3DCurve protocol
    time_axis : np.ndarray
        trajectory's time axis array
    ground_points : np.ndarray
        ground points to inverse geocode in XYZ coordinates, in the form (3,) or (N, 3)
    frequencies_doppler_centroid : float | FloatArrayType
        doppler frequencies centroid values to perform the inverse geocoding, in the form float or (N,).
        the number of frequencies must be 1 or equal to the number of points provided (if more than 1).
        If just 1 ground point is provided, several frequencies can be given to compute inverse geocoding at
        each different input value
    wavelength : float
        carrier signal wavelength

    Returns
    -------
    list[ExtendedDatetimeArrayType]
        list of azimuth times initial guesses arrays, one for each input ground point
    """

    frequencies_doppler_centroid = np.atleast_1d(frequencies_doppler_centroid)
    points = ground_points.copy()
    if ground_points.size // 3 == 1 and ground_points.ndim == 1:
        points = points.reshape(1, points.size)

    # if there are more frequencies than points, broadcast points up to frequencies
    if points.size // 3 == 1 and frequencies_doppler_centroid.size > 1:
        points = np.full((frequencies_doppler_centroid.size, 3), points)
    # otherwise do the opposite
    if frequencies_doppler_centroid.size == 1 and points.size // 3 > 1:
        frequencies_doppler_centroid = np.repeat(frequencies_doppler_centroid, points.size // 3)

    zero_crossing_pts_idx = []
    for id_point, point in enumerate(points):
        doppler_centroid_equation = doppler_equation_monostatic_residuals(
            sensor_positions=trajectory.evaluate(time_axis),
            sensor_velocities=trajectory.evaluate_first_derivatives(time_axis),
            ground_point=point,
            frequency_doppler_centroid=frequencies_doppler_centroid[id_point],
            wavelength=wavelength,
        )

        zero_crossing_indexes = _compute_zero_downcrossings(doppler_centroid_equation)

        interval_index = []
        if zero_crossing_indexes:
            azimuth_time = _interpolate_axis(axis=time_axis, values=np.asarray(zero_crossing_indexes) - 0.5)
            interval_index = _get_interval_id_from_axis(axis=time_axis, values=azimuth_time).tolist()

        else:
            if abs(doppler_centroid_equation[0]) < abs(doppler_centroid_equation[-1]):
                interval_index.append(0)
            else:
                interval_index.append(np.size(doppler_centroid_equation) - 1)

        zero_crossing_pts_idx.append(np.array(interval_index))

    # converting indexes to times
    az_initial_time_guesses = _interpolate_axis(axis=time_axis, values=zero_crossing_pts_idx)

    return az_initial_time_guesses


def inverse_geocoding_bistatic_init_core(
    trajectory_rx: TwiceDifferentiable3DCurve,
    trajectory_tx: TwiceDifferentiable3DCurve,
    time_axis_rx: ExtendedDatetimeArrayType,
    time_axis_tx: ExtendedDatetimeArrayType,
    ground_points: CoordinatesArrayType,
    frequencies_doppler_centroid: float | FloatArrayType,
    wavelength: float,
) -> ExtendedDatetimeType | ExtendedDatetimeArrayType:
    """Function to compute azimuth initial guess for Newton method for bistatic inverse geocoding.

    Parameters
    ----------
    trajectory_rx : TwiceDifferentiable3DCurve
        receiving sensor's trajectory, compliant to the TwiceDifferentiable3DCurve protocol
    trajectory_tx : TwiceDifferentiable3DCurve
        transmitting sensor's trajectory, compliant to the TwiceDifferentiable3DCurve protocol
    time_axis_rx : ExtendedDatetimeArrayType
        receiving sensor's trajectory time axis array
    time_axis_tx : ExtendedDatetimeArrayType
        transmitting sensor's trajectory time axis array
    ground_points : CoordinatesArrayType
        ground points to inverse geocode in XYZ coordinates, in the form (3,) or (N, 3)
    frequencies_doppler_centroid : float | FloatArrayType
        doppler frequencies centroid values to perform the inverse geocoding, in the form float or (N,).
        the number of frequencies must be 1 or equal to the number of points provided (if more than 1).
        If just 1 ground point is provided, several frequencies can be given to compute inverse geocoding at
        each different input value
    wavelength : float
        carrier signal wavelength

    Returns
    -------
    ExtendedDatetimeType | ExtendedDatetimeArrayType
        azimuth times initial guesses

    Raises
    ------
    RuntimeError
        ambiguous association between N ground points and M frequencies
    RuntimeError
        if orbit rx and orbit tx are not overlapped
    """

    points = ground_points.copy()
    if ground_points.size // 3 == 1 and ground_points.ndim == 1:
        points = points.reshape(1, points.size)

    array_one_dim = 0
    frequencies_doppler_centroid = np.asarray(frequencies_doppler_centroid)
    if frequencies_doppler_centroid.ndim == 1:
        array_one_dim = 1
    frequencies_doppler_centroid = np.atleast_1d(frequencies_doppler_centroid)

    if points.size // 3 > np.size(frequencies_doppler_centroid) == 1:
        frequencies_doppler_centroid = np.repeat(frequencies_doppler_centroid[0], points.size // 3)

    if np.size(frequencies_doppler_centroid) != points.size // 3 and not (
        points.size // 3 == 1 or np.size(frequencies_doppler_centroid) == 1
    ):
        raise RuntimeError(
            "Ambiguous matching between doppler frequencies "
            + f"{frequencies_doppler_centroid.shape} and "
            + f"ground points {points.shape}"
        )

    # if there are more frequencies than points, broadcast points up to frequencies
    if points.size // 3 == 1 and frequencies_doppler_centroid.size > 1:
        points = np.full((frequencies_doppler_centroid.size, 3), points)

    # creating a common time axis valid for both orbits
    d_t = min(np.mean(np.diff(time_axis_rx)), np.mean(np.diff(time_axis_tx)))

    axis_start_time = np.max([time_axis_tx[0], time_axis_rx[0]])
    axis_end_time = np.min([time_axis_tx[-1], time_axis_rx[-1]])
    axis_length = axis_end_time - axis_start_time

    if axis_length < 0:
        raise RuntimeError("Orbits are not overlapped")

    common_time_axis = axis_start_time + np.arange(axis_length / d_t) * d_t
    relative_time_axis = (common_time_axis - axis_start_time).astype(float)

    zero_crossing_pts_idx = []
    for id_point, point in enumerate(points):
        # computing doppler equations at zero doppler for both orbits
        doppler_centroid_equation_rx = doppler_equation_monostatic_residuals(
            sensor_positions=trajectory_rx.evaluate(time_axis_rx),
            sensor_velocities=trajectory_rx.evaluate_first_derivatives(time_axis_rx),
            ground_point=point,
            frequency_doppler_centroid=0,
            wavelength=1,
        )
        doppler_centroid_equation_tx = doppler_equation_monostatic_residuals(
            sensor_positions=trajectory_tx.evaluate(time_axis_tx),
            sensor_velocities=trajectory_tx.evaluate_first_derivatives(time_axis_tx),
            ground_point=point,
            frequency_doppler_centroid=0,
            wavelength=1,
        )

        # interpolating doppler equation values on the new relative time axis for both orbits
        doppler_centroid_equation_rx = np.interp(
            relative_time_axis,
            (time_axis_rx - axis_start_time).astype(float),
            doppler_centroid_equation_rx,
        )
        doppler_centroid_equation_tx = np.interp(
            relative_time_axis,
            (time_axis_tx - axis_start_time).astype(float),
            doppler_centroid_equation_tx,
        )

        residual = (
            doppler_centroid_equation_tx
            + doppler_centroid_equation_rx
            - frequencies_doppler_centroid[id_point] * wavelength
        )

        zero_crossing_indexes = _compute_zero_downcrossings(residual)

        if len(zero_crossing_indexes) == 0:
            if abs(residual[0]) < abs(residual[-1]):
                zero_crossing_indexes.append(0)
            else:
                zero_crossing_indexes.append(len(residual) - 1)

        # saving only the first zero crossing occurrence, that is the smallest time
        zero_crossing_pts_idx.append(zero_crossing_indexes[0])

    azimuth_init_guesses = common_time_axis[zero_crossing_pts_idx]
    if azimuth_init_guesses.size == 1 and ground_points.ndim == 1 and array_one_dim == 0:
        azimuth_init_guesses = azimuth_init_guesses[0]

    return azimuth_init_guesses


def inverse_geocoding_attitude_core(
    trajectory: TwiceDifferentiable3DCurve,
    boresight_normal: TwiceDifferentiable3DCurve,
    ground_points: CoordinatesArrayType,
    initial_guesses: ExtendedDatetimeType | ExtendedDatetimeArrayType,
    abs_time_tolerance: float = 1e-8,
    max_iter: int = 8,
) -> tuple[ExtendedDatetimeType | ExtendedDatetimeArrayType, float | FloatArrayType]:
    """Core algorithm for precise inverse geocoding monostatic using attitude and 3D Differentiable curve objects.

    Vectorized function with the following supported cases:

        - 1 guess and 1 earth point --> PDT, float
        - N guesses and N earth points --> array[PDT], array[float]
        - 1 guess and N earth points --> array[PDT], array[float]
        - N guesses and 1 earth point --> array[PDT], array[float]

    Parameters
    ----------
    trajectory : TwiceDifferentiable3DCurve
        general sar orbit polynomial 3D curve wrapper
    boresight_normal : TwiceDifferentiable3DCurve
        boresight normal vector from attitude polynomial 3D curve
    ground_points : CoordinatesArrayType
        earth points to inverse geocode in XYZ coordinates, in the form (3,) or (N, 3)
    initial_guesses : ExtendedDatetimeType | ExtendedDatetimeArrayType
        initial guesses for newton resolution method, initial guess can be a
        single PDT value or an array of PDT (N,)
    abs_time_tolerance : float, optional
        absolute time tolerance for newton convergence criteria, by default 1e-8
    max_iter : int, optional
        maximum number of iterations for newton method, by default 8

    Returns
    -------
    ExtendedDatetimeType | ExtendedDatetimeArrayType
        azimuth times array
    float | FloatArrayType
        range times array

    Raises
    ------
    RuntimeError
        ambiguous association between N input guesses and M earth points
    RuntimeError
        Newton method could not converge
    """

    ground_points = np.asarray(ground_points)

    if np.size(initial_guesses) != ground_points.size // 3 and not (
        ground_points.size // 3 == 1 or np.size(initial_guesses) == 1
    ):
        raise RuntimeError(
            "Ambiguous matching between initial guess times "
            + f"{np.shape(initial_guesses)} and "
            + f"ground points {ground_points.shape}"
        )

    azimuth_times = np.asarray(initial_guesses).copy()

    # starting the newton method to solve the equation
    for _ in range(max_iter):
        # computing polynomials and polynomials derivatives
        sensor_pos_curr = trajectory.evaluate(azimuth_times)
        sensor_vel_curr = trajectory.evaluate_first_derivatives(azimuth_times)

        arf1_curr = boresight_normal.evaluate(azimuth_times)
        arf1_derivative_curr = boresight_normal.evaluate_first_derivatives(azimuth_times)

        # slant range correspondent to the actual position of satellite
        line_of_sight = ground_points - sensor_pos_curr
        slant_range = np.linalg.norm(line_of_sight, axis=-1)

        # function to be computed and solved with Newton method
        # equation: f = [(earth_point - sensor_position) * arf1] / slant_range
        func = np.sum((line_of_sight * arf1_curr), axis=-1) / slant_range

        # derivative equation: df = [-sensor_velocity*arf1 + (earth_point - sensor_pos) * arf1_derivative] / slant_range
        func_der = (
            np.sum(
                -sensor_vel_curr * arf1_curr + (ground_points - sensor_pos_curr) * arf1_derivative_curr,
                axis=-1,
            )
            / slant_range
        )

        # updating the current value
        delta_err = func / func_der
        azimuth_times = azimuth_times - delta_err

        if np.max(np.abs(delta_err)) <= abs_time_tolerance:
            break
    else:
        raise RuntimeError(
            "Newton did not converge: maximum number of iterations" + f"{max_iter} reached. Residual error {delta_err}"
        )

    # re-evaluating slant range
    sensor_pos_curr = trajectory.evaluate(azimuth_times)
    line_of_sight = ground_points - sensor_pos_curr
    slant_range = np.linalg.norm(line_of_sight, axis=-1) * 2 / speed_of_light

    return azimuth_times, slant_range


def _compute_zero_downcrossings(values: np.ndarray) -> list[int]:
    """Compute the indexes of the descending zero crossing values.

    Parameters
    ----------
    values : np.ndarray
        values to analyse

    Returns
    -------
    list[int]
        list of indexes after a descending zero crossing
    """
    return [k for k in range(1, len(values)) if (values[k] * values[k - 1] <= 0 and values[k] < values[k - 1])]


def _get_interval_id_from_axis(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Get the interval indexes of the input axis corresponding to the input values.

    Parameters
    ----------
    axis : np.ndarray
        axis to get interval from
    values : np.ndarray
        values from which to get the closest axis index

    Returns
    -------
    np.ndarray
        indexes of the axis corresponding to the input values

    Raises
    ------
    RuntimeError
        if the axis is not monotone
    """

    axis_origin = axis[0]
    relative_axis = np.array(axis - axis_origin, dtype=float)
    relative_values = np.array(np.atleast_1d(values) - axis_origin, dtype=float)
    steps = np.diff(relative_axis)
    mean_step = np.mean(steps)

    # assess axis monotony
    if (steps > 0).all():
        is_increasing_axis = True
    elif (steps < 0).all():
        is_increasing_axis = False
    else:
        raise RuntimeError("Expecting monotone axis")

    # assess axis regularity
    is_regular_axis = False
    if np.all(np.isclose(steps, steps[0])):
        is_regular_axis = True

    if is_regular_axis:

        def to_int_and_clip(value):
            return np.clip(int(np.floor(value)), 0, relative_axis.size - 1)

        val = relative_values / mean_step

        if isinstance(val, np.ndarray):
            return np.array([to_int_and_clip(v) for v in val])

        return np.array([to_int_and_clip(val)])

    else:
        out = []
        for val in relative_values:
            closest_pos = np.argmin(np.abs((relative_axis - val)))
            if is_increasing_axis:
                if relative_axis[closest_pos] > val:
                    out.append(closest_pos - 1)
                else:
                    out.append(closest_pos)
            else:
                if relative_axis[closest_pos] < val:
                    out.append(closest_pos - 1)
                else:
                    out.append(closest_pos)
        return np.clip(np.array(out, dtype=int), 0, relative_values.size - 1)


def _interpolate_axis(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Interpolate input axis at the selected values.

    Parameters
    ----------
    axis : np.ndarray
        axis to be interpolate
    values : np.ndarray
        values at which interpolate the axis

    Returns
    -------
    np.ndarray
        interpolated values
    """
    return interp1d(range(axis.size), axis - axis[0])(values) + axis[0]
