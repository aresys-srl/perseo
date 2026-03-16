# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Direct geocoding core utilities"""

from __future__ import annotations

import numpy as np
from scipy.constants import speed_of_light

from perseo_core.geometry.utilities.ellipsoid import WGS84
from perseo_core.models.types import CoordinatesArrayType, FloatArrayType


def direct_geocoding_monostatic_core(
    sensor_positions: CoordinatesArrayType,
    sensor_velocities: CoordinatesArrayType,
    range_times: float | FloatArrayType,
    frequencies_doppler_centroid: float | FloatArrayType,
    wavelength: float,
    geodetic_altitude: float,
    initial_guesses: CoordinatesArrayType,
) -> CoordinatesArrayType:
    """Core computation of direct geocoding for monostatic systems.

    Parameters
    ----------
    sensor_positions : CoordinatesArrayType
        sensor position array, with shape (3,) or (N, 3)
    sensor_velocities : CoordinatesArrayType
        sensor velocity array, with shape (3,) or (N, 3)
    range_times : float | FloatArrayType
        range times at which evaluate the geocoding equation, with shape float or (M,)
    frequencies_doppler_centroid : float | FloatArrayType
        frequency_doppler_centroid value, single value or array (M,), if a single value is passed and there is more
        than 1 range times, it is broadcasted to all of them
    wavelength : float
        carrier signal wavelength
    geodetic_altitude : float
        geodetic altitude with respect to WGS84 ellipsoid
    initial_guesses : np.ndarray
        initial guess for the Newton method, with shape (3,) or (N, 3) or (M, 3) if 1 position and
        M range times

    Returns
    -------
    CoordinatesArrayType
        ground points, solution to the Newton method for direct geocoding

    Raises
    ------
    RuntimeError
        if inputs shapes are ambiguous to match, this error is raised
    """

    range_times = np.asarray(range_times) if not isinstance(range_times, float) else np.asarray([range_times])

    try:
        frequencies_doppler_centroid = np.broadcast_to(frequencies_doppler_centroid, range_times.shape)
    except ValueError as exc:
        raise RuntimeError(
            f"frequencies {frequencies_doppler_centroid.shape} != range times {range_times.shape}"
        ) from exc

    try:
        sensor_positions = np.broadcast_to(sensor_positions, sensor_velocities.shape)
    except ValueError as exc:
        raise RuntimeError(
            f"sensor position {sensor_positions.shape} != sensor velocities {sensor_velocities.shape}"
        ) from exc

    try:
        initial_guesses = np.broadcast_to(initial_guesses, sensor_positions.shape)
    except ValueError as exc:
        raise RuntimeError(
            f"sensor position {sensor_positions.shape} != initial guesses {initial_guesses.shape}"
        ) from exc

    if not sensor_positions.shape == sensor_velocities.shape == initial_guesses.shape:
        raise RuntimeError(
            f"Mismatch between input shapes: pos {sensor_positions.shape},"
            + f"vel {sensor_velocities.shape}, guess {initial_guesses.shape}"
        )

    one_size_array_flag = 0
    if sensor_positions.ndim == 2 and sensor_positions.size / 3 == 1:
        one_size_array_flag = 1

    ground_points = np.zeros((sensor_positions.size // 3, range_times.size, 3))
    for id_rng, rng_freq in enumerate(zip(range_times, frequencies_doppler_centroid, strict=True)):
        ground_points[..., id_rng, :] = _newton_for_direct_geocoding_monostatic(
            sensor_positions=sensor_positions,
            sensor_velocities=sensor_velocities,
            initial_guesses=initial_guesses,
            range_time=rng_freq[0],
            frequency_doppler_centroid=rng_freq[1],
            geodetic_altitude=geodetic_altitude,
            wavelength=wavelength,
        )

    return ground_points.squeeze() if not one_size_array_flag else ground_points.squeeze(axis=0)


def direct_geocoding_bistatic_core(
    sensor_positions_rx: CoordinatesArrayType,
    sensor_velocities_rx: CoordinatesArrayType,
    sensor_positions_tx: CoordinatesArrayType,
    sensor_velocities_tx: CoordinatesArrayType,
    range_times: float | FloatArrayType,
    frequencies_doppler_centroid: float | FloatArrayType,
    wavelength: float,
    geodetic_altitude: float,
    initial_guesses: CoordinatesArrayType,
) -> CoordinatesArrayType:
    """Core computation of direct geocoding for bistatic systems.

    Parameters
    ----------
    sensor_positions_rx : CoordinatesArrayType
        position of the sensor rx, with shape (3,) or (N, 3)
    sensor_velocities_rx : CoordinatesArrayType
        velocity of the sensor rx, with shape (3,) or (N, 3)
    sensor_positions_tx : CoordinatesArrayType
        position of the sensor tx, with shape (3,) or (M, 3), where M is the number of range times
    sensor_velocities_tx : CoordinatesArrayType
        velocity of the sensor tx, with shape (3,) or (M, 3), where M is the number of range times
    range_times : float | FloatArrayType
        range times where to evaluate the direct geocoding, with shape float or (M,)
    frequencies_doppler_centroid : float | FloatArrayType
        frequency_doppler_centroid value, single value or array (M,), if a single value is passed and there is more
        than 1 range times, it is broadcasted to all of them
    wavelength : float
        carrier signal wavelength
    geodetic_altitude : float
        altitude with respect to the WGS84 ellipsoid
    initial_guesses : CoordinatesArrayType
        initial guess for Newton method, with shape (3,), (N, 3) or (M, 3) if just 1 position and M range times

    Returns
    -------
    CoordinatesArrayType
        ground points for each input time and position rx value

    Raises
    ------
    RuntimeError
        if inputs shapes are ambiguous to match, this error is raised
    """

    range_times = np.asarray(range_times) if not isinstance(range_times, float) else np.asarray([range_times])

    try:
        frequencies_doppler_centroid = np.broadcast_to(frequencies_doppler_centroid, range_times.shape)
    except ValueError as exc:
        raise RuntimeError(
            f"frequencies {frequencies_doppler_centroid.shape} != range times {range_times.shape}"
        ) from exc

    one_size_array_flag = 0
    if (sensor_positions_rx.ndim == 2 and sensor_positions_rx.size / 3 == 1) or (
        sensor_positions_tx.ndim == 2 and sensor_positions_tx.size / 3 == 1 and sensor_positions_rx.size / 3 == 1
    ):
        one_size_array_flag = 1

    if sensor_positions_tx.ndim == sensor_velocities_tx.ndim == 1 and sensor_positions_tx.size // 3 == 1:
        sensor_positions_tx = sensor_positions_tx.reshape(1, sensor_positions_tx.size)
        sensor_velocities_tx = sensor_velocities_tx.reshape(1, sensor_velocities_tx.size)

    ground_points = np.zeros((sensor_positions_rx.size // 3, range_times.size, 3))
    looping_items = zip(
        range_times, frequencies_doppler_centroid, sensor_positions_tx, sensor_velocities_tx, strict=True
    )
    for id_rng, items in enumerate(looping_items):
        ground_points[..., id_rng, :] = _newton_for_direct_geocoding_bistatic(
            sensor_positions_rx=sensor_positions_rx,
            sensor_velocities_rx=sensor_velocities_rx,
            initial_guesses=initial_guesses,
            sensor_position_tx=items[2],
            sensor_velocity_tx=items[3],
            range_time=items[0],
            frequency_doppler_centroid=items[1],
            geodetic_altitude=geodetic_altitude,
            wavelength=wavelength,
        )

    return ground_points.squeeze() if not one_size_array_flag else ground_points.squeeze(axis=0)


def _newton_for_direct_geocoding_monostatic(
    sensor_positions: CoordinatesArrayType,
    sensor_velocities: CoordinatesArrayType,
    initial_guesses: CoordinatesArrayType,
    range_time: float,
    frequency_doppler_centroid: float,
    wavelength: float,
    geodetic_altitude: float,
    max_iter: int = 8,
    tolerance: float = 1e-5,
) -> CoordinatesArrayType:
    """Newton solving method for direct geocoding monostatic.

    Parameters
    ----------
    sensor_positions : CoordinatesArrayType
        sensor position array, with shape (3,) or (N,3)
    sensor_velocities : CoordinatesArrayType
        sensor velocities array, with shape (3,) or (N,3)
    initial_guesses : CoordinatesArrayType
        initial guesses array, with shape (3,) or (N,3)
    range_time : float
        range time at which compute the geocoding equation
    frequency_doppler_centroid : float
        frequency doppler centroid at which compute the geocoding equation
    wavelength : float
        carrier signal wavelength
    geodetic_altitude : float
        geodetic altitude with respect to WGS84 ellipse
    max_iter : int, optional
        maximum iterations for Newton method, by default 8
    tolerance : float, optional
        tolerance below which assert Newton convergence in meters, by default 1E-5

    Returns
    -------
    CoordinatesArrayType
        ground points at a given range value

    Raises
    ------
    RuntimeError
        raised if Newton method did not converge after max_iterations
    """

    tolerance_squared = tolerance * tolerance

    # variables and constants computation
    range_distance_square = (speed_of_light * range_time / 2.0) ** 2
    geoid_r_min = WGS84.b + geodetic_altitude
    geoid_r_max = WGS84.a + geodetic_altitude
    r_ep2 = geoid_r_min**2
    r_ee2 = geoid_r_max**2

    # input arguments array conversion
    ground_points_guess = initial_guesses.copy()

    array_size_one_flag = 0
    if ground_points_guess.ndim == sensor_positions.ndim == sensor_velocities.ndim == 1:
        array_size_one_flag = 1
        ground_points_guess = ground_points_guess.reshape(1, ground_points_guess.size)
        sensor_positions = sensor_positions.reshape(1, sensor_positions.size)
        sensor_velocities = sensor_velocities.reshape(1, sensor_velocities.size)

    # Newton method for direct geocoding
    for _ in range(max_iter):
        line_of_sight = sensor_positions - ground_points_guess
        distance_square = np.sum(line_of_sight * line_of_sight, axis=-1)
        distance = np.sqrt(distance_square)
        los_vel_product = np.sum(sensor_velocities * line_of_sight, axis=-1)

        range_equation = distance_square - range_distance_square
        grad_range_equation = -2 * line_of_sight

        doppler_equation, grad_doppler_equation = _doppler_equation(
            pv_scalar=los_vel_product,
            sat2point=line_of_sight,
            sat_velocity=sensor_velocities,
            distance=distance,
            wavelength=wavelength,
            frequency_doppler_centroid=frequency_doppler_centroid,
        )

        # assembling system of equations to be solved using Newton method
        functions_to_be_solved = [
            range_equation,
            _ellipse_equation(ground_points_guess, r_ee2, r_ep2),
            doppler_equation,
        ]
        functions_jacobians = [
            [
                grad_range_equation[..., k],
                _der_ellipse_equation_xi(ground_points_guess, k, r_ee2, r_ep2),
                grad_doppler_equation[..., k],
            ]
            for k in range(3)
        ]

        delta_err = -_inv_3x3_transpose(functions_jacobians, functions_to_be_solved).squeeze().T
        ground_points_guess = ground_points_guess + delta_err

        err_for_convergence = np.sum(delta_err * delta_err, axis=-1)
        if np.max(np.abs(err_for_convergence)) <= tolerance_squared:
            break
    else:
        raise RuntimeError(
            f"Newton did not converge: maximum number of iterations {max_iter} reached. Residual error {delta_err}"
        )

    return ground_points_guess if not array_size_one_flag else ground_points_guess.squeeze()


def _newton_for_direct_geocoding_bistatic(
    sensor_positions_rx: CoordinatesArrayType,
    sensor_velocities_rx: CoordinatesArrayType,
    initial_guesses: CoordinatesArrayType,
    sensor_position_tx: CoordinatesArrayType,
    sensor_velocity_tx: CoordinatesArrayType,
    range_time: float,
    frequency_doppler_centroid: float,
    wavelength: float,
    geodetic_altitude: float,
    max_iter: int = 8,
    tolerance: float = 1e-5,
) -> CoordinatesArrayType:
    """Newton solving method for direct geocoding bistatic.

    Parameters
    ----------
    sensor_positions_rx : CoordinatesArrayType
        sensor rx position array, with shape (3,) or (N,3)
    sensor_velocities_rx : CoordinatesArrayType
        sensor rx velocities array, with shape (3,) or (N,3)
    initial_guesses : CoordinatesArrayType
        initial guesses array, with shape (3,) or (N,3)
    sensor_position_tx : CoordinatesArrayType
        sensor tx position, with shape (3,)
    sensor_velocity_tx : CoordinatesArrayType
        sensor tx velocity, with shape (3,)
    range_time : float
        range time at which compute the geocoding equation
    frequency_doppler_centroid : float
        frequency doppler centroid at which compute the geocoding equation
    wavelength : float
        carrier signal wavelength
    geodetic_altitude : float
        geodetic altitude with respect to WGS84 ellipse
    max_iter : int, optional
        maximum iterations for Newton method, by default 8
    tolerance : float, optional
        tolerance below which assert Newton convergence in meters, by default 1E-5

    Returns
    -------
    CoordinatesArrayType
        earth points at a given range value

    Raises
    ------
    RuntimeError
        raised if Newton method did not converge after max_iterations
    """

    tolerance_squared = tolerance * tolerance

    # variables and constants computation
    range_distance_square = (speed_of_light * range_time) ** 2  # two-way distance
    geoid_r_min = WGS84.b + geodetic_altitude
    geoid_r_max = WGS84.a + geodetic_altitude
    r_ep2 = geoid_r_min**2
    r_ee2 = geoid_r_max**2

    # input arguments array conversion
    sensor_positions_rx = np.atleast_2d(sensor_positions_rx)
    sensor_velocities_rx = np.atleast_2d(sensor_velocities_rx)
    sensor_position_tx = np.atleast_2d(sensor_position_tx)
    sensor_velocity_tx = np.atleast_2d(sensor_velocity_tx)
    ground_points_guess = initial_guesses.copy()

    # Newton method for direct geocoding
    for _ in range(max_iter):
        # first sensor data
        line_of_sight_rx = sensor_positions_rx - ground_points_guess
        distance_square_rx = np.sum(line_of_sight_rx * line_of_sight_rx, axis=-1)
        distance_rx = np.sqrt(distance_square_rx)
        los_vel_product_rx = np.sum(sensor_velocities_rx * line_of_sight_rx, axis=-1)

        # second sensor data
        line_of_sight_tx = sensor_position_tx - ground_points_guess
        distance_square_tx = np.sum(line_of_sight_tx * line_of_sight_tx, axis=-1)
        distance_tx = np.sqrt(distance_square_tx)
        los_vel_product_tx = np.sum(sensor_velocity_tx * line_of_sight_tx, axis=-1)

        # range equation
        distance = distance_rx + distance_tx
        range_equation = distance**2 - range_distance_square
        grad_range_equation = (
            -2
            * distance[:, np.newaxis]
            * (line_of_sight_rx / distance_rx[:, np.newaxis] + line_of_sight_tx / distance_tx[:, np.newaxis])
        )

        # doppler equations
        doppler_equation_rx, grad_doppler_equation_rx = _doppler_equation(
            wavelength=wavelength,
            pv_scalar=los_vel_product_rx,
            distance=distance_rx,
            frequency_doppler_centroid=frequency_doppler_centroid,
            sat_velocity=sensor_velocities_rx,
            sat2point=line_of_sight_rx,
        )
        doppler_equation_tx, grad_doppler_equation_tx = _doppler_equation(
            wavelength=wavelength,
            pv_scalar=los_vel_product_tx,
            distance=distance_tx,
            frequency_doppler_centroid=frequency_doppler_centroid,
            sat_velocity=sensor_velocity_tx,
            sat2point=line_of_sight_tx,
        )

        # assembling doppler equations and their gradients
        doppler_equation = (doppler_equation_rx + doppler_equation_tx) / 2
        grad_doppler_equation = (grad_doppler_equation_rx + grad_doppler_equation_tx) / 2

        # assembling system of equations to be solved using Newton method
        functions_to_be_solved = [
            range_equation,
            _ellipse_equation(ground_points_guess, r_ee2, r_ep2),
            doppler_equation,
        ]
        functions_jacobians = [
            [
                grad_range_equation[..., k],
                _der_ellipse_equation_xi(ground_points_guess, k, r_ee2, r_ep2),
                grad_doppler_equation[..., k],
            ]
            for k in range(3)
        ]

        delta_err = -_inv_3x3_transpose(functions_jacobians, functions_to_be_solved).squeeze()
        ground_points_guess = ground_points_guess + delta_err.T

        err_for_convergence = np.dot(delta_err, delta_err.T)
        if np.max(np.abs(err_for_convergence)) <= tolerance_squared:
            break
    else:
        raise RuntimeError(
            f"Newton did not converge: maximum number of iterations {max_iter} reached. Residual error {delta_err}"
        )

    return ground_points_guess


def _inv_3x3_transpose(jac: np.ndarray, func: np.ndarray) -> np.ndarray:
    """Performing inverse of 3x3 matrix using explicit form.

    Parameters
    ----------
    jac : np.ndarray
        jacobians array
    func : np.ndarray
        functions array

    Returns
    -------
    np.ndarray
        inverse of input func matrix
    """
    det = (
        +jac[0][0] * (jac[2][2] * jac[1][1] - jac[2][1] * jac[1][2])
        - jac[1][0] * (jac[2][2] * jac[0][1] - jac[2][1] * jac[0][2])
        + jac[2][0] * (jac[1][2] * jac[0][1] - jac[1][1] * jac[0][2])
    )

    x_val = (
        func[0] * (jac[1][1] * jac[2][2] - jac[2][1] * jac[1][2])
        + func[1] * (jac[2][0] * jac[1][2] - jac[1][0] * jac[2][2])
        + func[2] * (jac[1][0] * jac[2][1] - jac[2][0] * jac[1][1])
    )

    y_val = (
        func[0] * (jac[2][1] * jac[0][2] - jac[0][1] * jac[2][2])
        + func[1] * (jac[0][0] * jac[2][2] - jac[2][0] * jac[0][2])
        + func[2] * (jac[2][0] * jac[0][1] - jac[0][0] * jac[2][1])
    )

    z_val = (
        func[0] * (jac[0][1] * jac[1][2] - jac[1][1] * jac[0][2])
        + func[1] * (jac[1][0] * jac[0][2] - jac[0][0] * jac[1][2])
        + func[2] * (jac[0][0] * jac[1][1] - jac[1][0] * jac[0][1])
    )

    return np.asarray([x_val, y_val, z_val]) / det


def _ellipse_equation(coords: np.ndarray, r_ee2: float, r_ep2: float) -> float:
    """3D Ellipse generic equation.

    Parameters
    ----------
    x : np.ndarray
        x, y, z coordinates array where to evaluate the ellipse
    r_ee2 : float
        radius square along x and y directions
    r_ep2 : float
        radius square along z direction

    Returns
    -------
    float
        value of the ellipse at the input coordinate
    """
    return (
        (coords[..., 0] * coords[..., 0] + coords[..., 1] * coords[..., 1]) / r_ee2
        + coords[..., 2] * coords[..., 2] / r_ep2
        - 1.0
    )


def _der_ellipse_equation_xi(coords: np.ndarray, i_coord: int, r_ee2: float, r_ep2: float) -> float:
    """Derivative of ellipse equation.

    Parameters
    ----------
    x : np.ndarray
        x, y, z array coordinate where to evaluate the derivative
    i_coord : int
        direction index where to evaluate the derivative
    r_ee2 : float
        radius square along x and y directions
    r_ep2 : float
        radius square along z direction

    Returns
    -------
    float
        derivative value along the selected direction at the selected coordinate
    """

    radius_square = r_ee2 if i_coord < 2 else r_ep2

    return 2 * coords[..., i_coord] / radius_square


def _doppler_equation(
    wavelength: float,
    pv_scalar: float,
    distance: float,
    frequency_doppler_centroid: float,
    sat_velocity: np.ndarray,
    sat2point: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Doppler equation solver.

    Parameters
    ----------
    wavelength : float
        carrier signal wavelength
    pv_scalar : float
        scalar product between sensor velocity and line of sight
    distance : float
        ground point - sensor distance
    frequency_doppler_centroid : float
        frequency doppler centroid
    sat_velocity : np.ndarray
        sensor velocity
    sat2point : np.ndarray
        line of sight

    Returns
    -------
    float
        doppler equation solution
    np.ndarray
        doppler equation gradient
    """

    c_factor = 2.0 / wavelength / distance
    doppler_equation = c_factor * pv_scalar + frequency_doppler_centroid
    norm_pv = pv_scalar / distance**2
    grad_doppler_equation = (c_factor * (-sat_velocity + (norm_pv * sat2point.T).T).T).T
    return doppler_equation, grad_doppler_equation
