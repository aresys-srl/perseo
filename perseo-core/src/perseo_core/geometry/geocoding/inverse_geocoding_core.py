# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Inverse geocoding core utilities"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.constants import speed_of_light

from perseo_core.geometry.doppler import doppler_equation_monostatic_residuals
from perseo_core.geometry.navigation.trajectory import Trajectory
from perseo_core.geometry.pointing.attitude import Attitude
from perseo_core.timing.precise_datetime import PreciseDateTime


def inverse_geocoding_monostatic_core(
    trajectory: Trajectory,
    ground_points: npt.NDArray[np.floating],
    initial_guesses: PreciseDateTime | np.datetime64 | npt.NDArray,
    doppler_frequencies: float | npt.NDArray[np.floating],
    wavelength: float,
    scene_velocity: npt.NDArray[np.floating] | None = None,
    abs_time_tolerance: float = 1e-8,
    max_iter: int = 8,
) -> tuple[PreciseDateTime | np.datetime64 | npt.NDArray, float | npt.NDArray[np.floating]]:
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
    trajectory : Trajectory
        sensor trajectory
    ground_points : npt.NDArray[np.floating]
        ground points to inverse geocode in XYZ coordinates, in the form (3,) or (N, 3)
    initial_guesses : PreciseDateTime | np.datetime64 | npt.NDArray
        azimuth times initial guesses to limit and guide the search of solutions, in the form (N,) or as a single time
    doppler_frequencies : float | npt.NDArray[np.floating]
        doppler frequencies centroid values to perform the inverse geocoding, in the form float or (N,).
        the number of frequencies must be 1 or equal to the number of points provided (if more than 1).
        If just 1 ground point is provided, several frequencies can be given to compute inverse geocoding at
        each different input value
    wavelength : float
        carrier signal wavelength
    scene_velocity : npt.NDArray[np.floating] | None, optional
        ground scene velocity. If None, it's set to the default value of 0, by default None
    abs_time_tolerance : float, optional
        absolute time tolerance for newton convergence criteria, by default 1e-8
    max_iter : int, optional
        maximum number of iterations for newton method, by default 8

    Returns
    -------
    PreciseDateTime | np.datetime64 | npt.NDArray
        azimuth times array
    float | npt.NDArray[np.floating]
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
    azimuth_times = np.asarray(initial_guesses).copy()

    scene_velocity = np.zeros(3) if scene_velocity is None else scene_velocity

    if azimuth_times.size != ground_points.size // 3 and not (ground_points.size // 3 == 1 or azimuth_times.size == 1):
        raise RuntimeError(
            "Ambiguous matching between initial guess times "
            + f"{azimuth_times.shape} and "
            + f"ground points {np.shape(ground_points)}"
        )

    if np.size(doppler_frequencies) != ground_points.size // 3 and not (
        ground_points.size // 3 == 1 or np.size(doppler_frequencies) == 1
    ):
        raise RuntimeError(
            "Ambiguous matching between doppler frequencies "
            + f"{np.shape(doppler_frequencies)} and "
            + f"ground points {np.shape(ground_points)}"
        )

    # starting the newton method to solve the equation
    for _ in range(max_iter):
        # computing orbit position, first and second derivatives
        sensor_position = trajectory.position(azimuth_times)
        sensor_velocity = trajectory.velocity(azimuth_times)
        sensor_acceleration = trajectory.acceleration(azimuth_times)

        # slant range correspondent to the actual position of satellite
        line_of_sight = ground_points - sensor_position
        slant_range = np.linalg.norm(line_of_sight, axis=-1)

        # function to be computed and solved with Newton method
        # equation:
        # f = [(earth_point - sensor_position) * (scene_velocity - sat_velocity)] + doppler_term
        doppler_term = wavelength * doppler_frequencies / 2.0 * slant_range
        func = np.sum((line_of_sight * (scene_velocity - sensor_velocity)), axis=-1) + doppler_term

        # derivative equation:
        # df = [-sensor_velocity*(sensor_velocity - scene_velocity) - (earth_point - sensor_pos) * sensor_acceleration -
        # (earth_point - sensor_pos) * sensor_velocity * lambda * doppler_freq / 2 / slant_range]
        func_der = (
            -np.sum(sensor_velocity * (scene_velocity - sensor_velocity), axis=-1)
            - np.sum(sensor_acceleration * line_of_sight, axis=-1)
            - np.sum(sensor_velocity * line_of_sight, axis=-1) * wavelength * doppler_frequencies / 2 / slant_range
        )

        # updating the current value
        delta_err = func / func_der
        azimuth_times = azimuth_times - delta_err

        if np.max(np.abs(delta_err)) <= abs_time_tolerance:
            break

    else:
        raise RuntimeError(
            # pyrefly: ignore [unbound-name]
            "Newton did not converge: maximum number of iterations " + f"{max_iter} reached. Residual error {delta_err}"
        )

    # re-evaluating slant range
    sensor_pos_curr = trajectory.position(azimuth_times)
    line_of_sight = ground_points - sensor_pos_curr
    slant_range = np.linalg.norm(line_of_sight, axis=-1) * 2 / speed_of_light

    if np.ndim(slant_range) == 0:
        slant_range = float(slant_range)

    return azimuth_times, slant_range


def inverse_geocoding_bistatic_core(
    trajectory_rx: Trajectory,
    trajectory_tx: Trajectory,
    ground_points: npt.NDArray[np.floating],
    initial_guesses: PreciseDateTime | np.datetime64 | npt.NDArray,
    doppler_frequencies: float | npt.NDArray[np.floating],
    wavelength: float,
    abs_time_tolerance: float = 1e-8,
    max_iter: int = 8,
) -> tuple[PreciseDateTime | np.datetime64 | npt.NDArray, float | npt.NDArray[np.floating]]:
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
    trajectory_rx : Trajectory
        receiving sensors trajectory
    trajectory_tx : Trajectory
        transmitting sensor trajectory
    ground_points : npt.NDArray[np.floating]
        ground points to inverse geocode in XYZ coordinates, in the form (3,) or (N, 3)
    initial_guesses : PreciseDateTime | np.datetime64 | npt.NDArray
        azimuth times initial guesses to limit and guide the search of solutions, in the form (N,) or as a single time
    doppler_frequencies : float | npt.NDArray[np.floating]
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
    PreciseDateTime | np.datetime64 | npt.NDArray
        azimuth times array
    float | npt.NDArray[np.floating]
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
    num_points = ground_points.size // 3
    one_size_array = 0
    if ground_points.ndim != 1 and ground_points.size // 3 == 1:
        one_size_array = 1
    azimuth_times_rx = np.array(initial_guesses).copy()
    azimuth_times_tx = azimuth_times_rx.copy()

    if azimuth_times_rx.size > ground_points.size // 3 == 1:
        ground_points = np.full((azimuth_times_rx.size, 3), ground_points)
        num_points = ground_points.size // 3

    if np.size(doppler_frequencies) > ground_points.size // 3 == 1:
        ground_points = np.full((np.size(doppler_frequencies), 3), ground_points)
        num_points = ground_points.size // 3

    if azimuth_times_rx.size != ground_points.size // 3 and not (
        ground_points.size // 3 == 1 or azimuth_times_rx.size == 1
    ):
        raise RuntimeError(
            "Ambiguous matching between initial guess times "
            + f"{azimuth_times_rx.shape} and "
            + f"ground points {ground_points.shape}"
        )

    if np.size(doppler_frequencies) != ground_points.size // 3 and not (
        ground_points.size // 3 == 1 or np.size(doppler_frequencies) == 1
    ):
        raise RuntimeError(
            "Ambiguous matching between doppler frequencies "
            + f"{np.shape(doppler_frequencies)} and "
            + f"ground points {ground_points.shape}"
        )

    # computing tx guess from rx by adding a delay
    position_rx = trajectory_rx.position(azimuth_times_rx)
    position_tx = trajectory_tx.position(azimuth_times_rx)
    delay_estimate = (
        np.linalg.norm(position_tx - ground_points, axis=-1, keepdims=True)
        + np.linalg.norm(position_rx - ground_points, axis=-1, keepdims=True)
    ) / speed_of_light
    azimuth_times_tx = np.array(azimuth_times_rx - delay_estimate.mean())

    # Newton method to solve the equation
    for _ in range(max_iter):
        # computing orbit position, first and second derivatives for sensor rx
        position_rx = trajectory_rx.position(
            azimuth_times_rx,
        )
        line_of_sight_rx = ground_points - position_rx
        velocity_rx = trajectory_rx.velocity(
            azimuth_times_rx,
        )
        norm_vel_rx_square = np.sum(velocity_rx * velocity_rx, axis=-1)
        acceleration_rx = trajectory_rx.acceleration(
            azimuth_times_rx,
        )

        # computing orbit position, first and second derivatives for sensor tx
        position_tx = trajectory_tx.position(
            azimuth_times_tx,
        )
        line_of_sight_tx = ground_points - position_tx
        velocity_tx = trajectory_tx.velocity(
            azimuth_times_tx,
        )
        norm_vel_tx_square = np.sum(velocity_tx * velocity_tx, axis=-1)
        acceleration_tx = trajectory_tx.acceleration(
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
        doppler_equation_freq_term = wavelength * doppler_frequencies * slant_range_rx * slant_range_tx
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
            * (rng_vel_product_tx + wavelength * doppler_frequencies * slant_range_tx)
        )
        df2_dt_rx = (
            slant_range_tx * (norm_vel_rx_square - np.sum(line_of_sight_rx * acceleration_rx, axis=-1))
            + df2_dt_rx_doppler_freq_term
        )

        # second derivative with respect to tx
        df2_dt_tx_doppler_freq_term = (
            rng_vel_product_tx
            / slant_range_tx
            * (rng_vel_product_rx + wavelength * doppler_frequencies * slant_range_rx)
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
            # pyrefly: ignore [unbound-name]
            "Newton did not converge: maximum number of iterations " + f"{max_iter} reached. Residual error {delta_err}"
        )

    if one_size_array:
        return np.atleast_1d(azimuth_times_rx), np.atleast_1d(slant_range).astype(float)

    if isinstance(slant_range, np.ndarray):
        slant_range = slant_range.astype(float)

    if np.ndim(slant_range) == 0:
        slant_range = float(slant_range)

    return azimuth_times_rx, slant_range


def inverse_geocoding_monostatic_init_core(
    trajectory: Trajectory,
    time_axis: npt.NDArray,
    ground_points: npt.NDArray[np.floating],
    doppler_frequencies: float | npt.NDArray[np.floating],
    wavelength: float,
) -> list[npt.NDArray]:
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
    trajectory : Trajectory
        sensor trajectory
    time_axis : np.ndarray
        trajectory's time axis array
    ground_points : npt.NDArray[np.floating]
        ground points to inverse geocode in XYZ coordinates, in the form (3,) or (N, 3)
    doppler_frequencies : float | npt.NDArray[np.floating]
        doppler frequencies centroid values to perform the inverse geocoding, in the form float or (N,).
        the number of frequencies must be 1 or equal to the number of points provided (if more than 1).
        If just 1 ground point is provided, several frequencies can be given to compute inverse geocoding at
        each different input value
    wavelength : float
        carrier signal wavelength

    Returns
    -------
    list[npt.NDArray]
        list of azimuth times initial guesses arrays, one for each input ground point
    """

    doppler_frequencies = np.atleast_1d(doppler_frequencies)
    points = ground_points.copy()
    if ground_points.size // 3 == 1 and ground_points.ndim == 1:
        points = points.reshape(1, points.size)

    # if there are more frequencies than points, broadcast points up to frequencies
    if points.size // 3 == 1 and doppler_frequencies.size > 1:
        points = np.full((doppler_frequencies.size, 3), points)
    # otherwise do the opposite
    if doppler_frequencies.size == 1 and points.size // 3 > 1:
        doppler_frequencies = np.repeat(doppler_frequencies, points.size // 3)

    zero_crossing_pts_idx = []
    for id_point, point in enumerate(points):
        doppler_centroid_equation = doppler_equation_monostatic_residuals(
            sensor_positions=trajectory.position(time_axis),
            sensor_velocities=trajectory.velocity(time_axis),
            ground_point=point,
            doppler_frequency=doppler_frequencies[id_point],
            wavelength=wavelength,
        )
        assert isinstance(doppler_centroid_equation, np.ndarray)
        zero_crossing_indexes = _compute_zero_downcrossings(doppler_centroid_equation)

        if zero_crossing_indexes.size > 0:
            zero_crossing_pts_idx.append(zero_crossing_indexes)  # left interval edge
        elif abs(doppler_centroid_equation[0]) < abs(doppler_centroid_equation[-1]):
            zero_crossing_pts_idx.append(np.array([0]))
        else:
            zero_crossing_pts_idx.append(np.array([doppler_centroid_equation.size - 1]))

    return [time_axis[idx] for idx in zero_crossing_pts_idx]


def inverse_geocoding_bistatic_init_core(
    trajectory_rx: Trajectory,
    trajectory_tx: Trajectory,
    time_axis_rx: npt.NDArray,
    time_axis_tx: npt.NDArray,
    ground_points: npt.NDArray[np.floating],
    doppler_frequencies: float | npt.NDArray[np.floating],
    wavelength: float,
) -> PreciseDateTime | np.datetime64 | npt.NDArray:
    """Function to compute azimuth initial guess for Newton method for bistatic inverse geocoding.

    Parameters
    ----------
    trajectory_rx : Trajectory
        receiving sensor trajectory
    trajectory_tx : Trajectory
        transmitting sensor trajectory
    time_axis_rx : npt.NDArray
        receiving sensor trajectory time axis array
    time_axis_tx : npt.NDArray
        transmitting sensor trajectory time axis array
    ground_points : npt.NDArray[np.floating]
        ground points to inverse geocode in XYZ coordinates, in the form (3,) or (N, 3)
    doppler_frequencies : float | npt.NDArray[np.floating]
        doppler frequencies centroid values to perform the inverse geocoding, in the form float or (N,).
        the number of frequencies must be 1 or equal to the number of points provided (if more than 1).
        If just 1 ground point is provided, several frequencies can be given to compute inverse geocoding at
        each different input value
    wavelength : float
        carrier signal wavelength

    Returns
    -------
    PreciseDateTime | np.datetime64 | npt.NDArray
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
    doppler_frequencies = np.array(doppler_frequencies)
    if doppler_frequencies.ndim == 1:
        array_one_dim = 1
    doppler_frequencies = np.atleast_1d(doppler_frequencies)

    if points.size // 3 > np.size(doppler_frequencies) == 1:
        doppler_frequencies = np.repeat(doppler_frequencies[0], points.size // 3)

    if np.size(doppler_frequencies) != points.size // 3 and not (
        points.size // 3 == 1 or np.size(doppler_frequencies) == 1
    ):
        raise RuntimeError(
            "Ambiguous matching between doppler frequencies "
            + f"{doppler_frequencies.shape} and "
            + f"ground points {points.shape}"
        )

    # if there are more frequencies than points, broadcast points up to frequencies
    if points.size // 3 == 1 and doppler_frequencies.size > 1:
        points = np.full((doppler_frequencies.size, 3), points)

    # creating a common time axis valid for both orbits
    d_t = min(np.mean(np.diff(time_axis_rx)), np.mean(np.diff(time_axis_tx)))

    axis_start_time = np.max([time_axis_tx[0], time_axis_rx[0]])
    axis_end_time = np.min([time_axis_tx[-1], time_axis_rx[-1]])
    axis_length = axis_end_time - axis_start_time

    if axis_length < 0:
        raise RuntimeError("Orbits are not overlapped")

    common_time_axis = axis_start_time + np.arange(axis_length / d_t) * d_t
    relative_time_axis = (common_time_axis - axis_start_time).astype(float)

    zero_crossing_pts_idx: list[int] = []
    for id_point, point in enumerate(points):
        # computing doppler equations at zero doppler for both orbits
        doppler_centroid_equation_rx = doppler_equation_monostatic_residuals(
            sensor_positions=trajectory_rx.position(time_axis_rx),
            sensor_velocities=trajectory_rx.velocity(time_axis_rx),
            ground_point=point,
            doppler_frequency=0,
            wavelength=1,
        )
        doppler_centroid_equation_tx = doppler_equation_monostatic_residuals(
            sensor_positions=trajectory_tx.position(time_axis_tx),
            sensor_velocities=trajectory_tx.velocity(time_axis_tx),
            ground_point=point,
            doppler_frequency=0,
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
            doppler_centroid_equation_tx + doppler_centroid_equation_rx - doppler_frequencies[id_point] * wavelength
        )

        zero_crossing_indexes = _compute_zero_downcrossings(residual)

        if zero_crossing_indexes.size > 0:
            # saving only the first zero crossing occurrence, that is the smallest time
            zero_crossing_pts_idx.append(zero_crossing_indexes[0] + 1)  # right interval edge
        elif abs(residual[0]) < abs(residual[-1]):
            zero_crossing_pts_idx.append(0)
        else:
            zero_crossing_pts_idx.append(residual.size - 1)

    azimuth_init_guesses = common_time_axis[zero_crossing_pts_idx]
    if azimuth_init_guesses.size == 1 and ground_points.ndim == 1 and array_one_dim == 0:
        return azimuth_init_guesses[0]

    return azimuth_init_guesses


def inverse_geocoding_monostatic_attitude_core(
    trajectory: Trajectory,
    attitude: Attitude,
    ground_points: npt.NDArray[np.floating],
    initial_guesses: PreciseDateTime | np.datetime64 | npt.NDArray,
    dt: float = 0.1,
    abs_time_tolerance: float = 1e-8,
    max_iter: int = 8,
) -> tuple[PreciseDateTime | np.datetime64 | npt.NDArray, float | npt.NDArray[np.floating]]:
    """Core algorithm for precise inverse geocoding monostatic using attitude and 3D Differentiable curve objects.

    Vectorized function with the following supported cases:

        - 1 guess and 1 earth point --> PDT, float
        - N guesses and N earth points --> array[PDT], array[float]
        - 1 guess and N earth points --> array[PDT], array[float]
        - N guesses and 1 earth point --> array[PDT], array[float]

    Parameters
    ----------
    trajectory : Trajectory
        sensor trajectory
    attitude : Attitude
        sensor attitude
    ground_points : npt.NDArray[np.floating]
        earth points to inverse geocode in XYZ coordinates, in the form (3,) or (N, 3)
    initial_guesses : PreciseDateTime | np.datetime64 | npt.NDArray
        initial guesses for newton resolution method, initial guess can be a
        single PDT value or an array of PDT (N,)
    dt : float, optional
        time step for computing the approximate derivative of the boresight normal unit vector, by default 0.1
    abs_time_tolerance : float, optional
        absolute time tolerance for newton convergence criteria, by default 1e-8
    max_iter : int, optional
        maximum number of iterations for newton method, by default 8

    Returns
    -------
    PreciseDateTime | np.datetime64 | npt.NDArray
        azimuth times array
    float | npt.NDArray[np.floating]
        range times array

    Raises
    ------
    RuntimeError
        ambiguous association between N input guesses and M earth points
    RuntimeError
        Newton method could not converge
    """
    intial_guess_size = 1 if isinstance(initial_guesses, (PreciseDateTime, np.datetime64)) else np.size(initial_guesses)

    if intial_guess_size != ground_points.size // 3 and not (ground_points.size // 3 == 1 or intial_guess_size == 1):
        initial_guess_shape = (
            () if isinstance(initial_guesses, (PreciseDateTime, np.datetime64)) else np.shape(initial_guesses)
        )
        raise RuntimeError(
            "Ambiguous matching between initial guess times "
            + f"{initial_guess_shape} and "
            + f"ground points {ground_points.shape}"
        )

    azimuth_times = np.array(initial_guesses).copy()

    # starting the newton method to solve the equation
    for _ in range(max_iter):
        # computing polynomials and polynomials derivatives
        sensor_pos_curr = trajectory.position(azimuth_times)
        sensor_vel_curr = trajectory.velocity(azimuth_times)

        arf1_curr = attitude.evaluate(azimuth_times)[..., 0, :]
        arf1_curr_dt = attitude.evaluate(azimuth_times + dt)[..., 0, :]
        arf1_derivative_curr = (arf1_curr_dt - arf1_curr) / dt

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
            # pyrefly: ignore [unbound-name]
            "Newton did not converge: maximum number of iterations " + f"{max_iter} reached. Residual error {delta_err}"
        )

    # re-evaluating slant range
    sensor_pos_curr = trajectory.position(azimuth_times)
    line_of_sight = ground_points - sensor_pos_curr
    slant_range = np.linalg.norm(line_of_sight, axis=-1) * 2 / speed_of_light

    if np.ndim(slant_range) == 0:
        slant_range = float(slant_range)

    return azimuth_times, slant_range


def _compute_zero_downcrossings(values: np.ndarray) -> npt.NDArray[np.integer]:
    """Compute the indexes of the descending zero crossing values."""
    mask = (values[1:] * values[:-1] <= 0) & (values[1:] < values[:-1])
    return np.where(mask)[0]
