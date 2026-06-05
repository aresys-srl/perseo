# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Direct geocoding core utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy.constants import speed_of_light

from perseo_core.geometry.doppler import doppler_equation
from perseo_core.geometry.ellipsoid import WGS84

if TYPE_CHECKING:
    from collections.abc import Sequence


def direct_geocoding_monostatic_core(
    sensor_positions: npt.NDArray[np.floating],
    sensor_velocities: npt.NDArray[np.floating],
    range_times: float | npt.NDArray[np.floating],
    doppler_frequencies: float | npt.NDArray[np.floating],
    wavelength: float,
    altitude: float,
    initial_guesses: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Compute ground points via monostatic direct geocoding.

    It supports N different sensor positions and M different range time values.

    Parameters
    ----------
    sensor_positions : npt.NDArray[np.floating]
        sensor positions with shape (3,) or (N, 3)
    sensor_velocities : npt.NDArray[np.floating]
        sensor velocities with shape (3,) or (N, 3)
    range_times : float | npt.NDArray[np.floating]
        range times with shape float or (M,)
    doppler_frequencies : float | npt.NDArray[np.floating]
        frequency_doppler_centroid value, single value or array (M,)
    wavelength : float
        carrier signal wavelength
    altitude : float
        altitude with respect to WGS84 ellipsoid
    initial_guesses : npt.NDArray[np.floating]
        initial guess for the Newton method, with shape (3,) or (N, 3) or (M, 3)

    Returns
    -------
    npt.NDArray[np.floating]
        ground points with shape (N, M, 3)

    """
    try:
        doppler_frequencies = np.broadcast_to(doppler_frequencies, np.shape(range_times))
    except ValueError as exc:
        msg = f"doppler frequencies {np.shape(doppler_frequencies)} != range times {np.shape(range_times)}"
        raise RuntimeError(msg) from exc

    try:
        sensor_positions = np.broadcast_to(sensor_positions, sensor_velocities.shape)
    except ValueError as exc:
        msg = f"sensor positions {sensor_positions.shape} != sensor velocities {sensor_velocities.shape}"
        raise RuntimeError(msg) from exc

    try:
        initial_guesses = np.broadcast_to(initial_guesses, sensor_positions.shape)
    except ValueError as exc:
        msg = f"sensor position {sensor_positions.shape} != initial guesses {initial_guesses.shape}"
        raise RuntimeError(msg) from exc

    if not sensor_positions.shape == sensor_velocities.shape == initial_guesses.shape:
        msg = (
            f"Mismatch between input shapes: pos {sensor_positions.shape},"
            f"vel {sensor_velocities.shape}, guess {initial_guesses.shape}"
        )
        raise RuntimeError(msg)

    ground_points = np.zeros((sensor_positions.size // 3, np.size(range_times), 3))
    for range_index, (range_time, doppler_frequency) in enumerate(
        zip(np.atleast_1d(range_times), np.atleast_1d(doppler_frequencies), strict=True)
    ):
        ground_points[..., range_index, :] = _direct_geocoding_monostatic_newton(
            sensor_positions=sensor_positions,
            sensor_velocities=sensor_velocities,
            initial_guesses=initial_guesses,
            range_times=range_time,
            doppler_frequencies=doppler_frequency,
            altitude=altitude,
            wavelength=wavelength,
        )

    one_size_array_flag = sensor_positions.ndim == 2 and sensor_positions.size / 3 == 1
    return ground_points.squeeze() if not one_size_array_flag else ground_points.squeeze(axis=0)


def direct_geocoding_monostatic_core_range_vectorized(
    sensor_positions: npt.NDArray[np.floating],
    sensor_velocities: npt.NDArray[np.floating],
    range_times: float | npt.NDArray[np.floating],
    doppler_frequencies: float | npt.NDArray[np.floating],
    wavelength: float,
    altitude: float,
    initial_guesses: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Perform direct geocoding for monostatic systems, vectorized computation along range times.

    Parameters
    ----------
    sensor_positions : npt.NDArray[np.floating]
        sensor positions with shape (3,) or (N, 3)
    sensor_velocities : npt.NDArray[np.floating]
        sensor velocities with shape (3,) or (N, 3)
    range_times : float | npt.NDArray[np.floating]
        range times with shape float or (M,)
    doppler_frequencies : float | npt.NDArray[np.floating]
        frequency_doppler_centroid value, single value or array (M,)
    wavelength : float
        carrier signal wavelength
    altitude : float
        altitude with respect to WGS84 ellipsoid
    initial_guesses : npt.NDArray[np.floating]
        initial guess for the Newton method, with shape (3,) or (N, 3) or (M, 3)

    Returns
    -------
    npt.NDArray[np.floating]
        ground points with shape (N, M, 3)

    """
    range_times = np.atleast_1d(range_times)

    one_size_array_flag = sensor_positions.ndim == 2 and sensor_positions.size / 3 == 1

    try:
        doppler_frequencies = np.broadcast_to(doppler_frequencies, range_times.shape)
    except ValueError as exc:
        msg = f"doppler frequencies {np.shape(doppler_frequencies)} != range times {range_times.shape}"
        raise RuntimeError(msg) from exc

    sensor_positions = np.atleast_2d(sensor_positions)
    sensor_velocities = np.atleast_2d(sensor_velocities)
    initial_guesses = np.atleast_2d(initial_guesses)

    num_sensor_pos, coord_dim = sensor_positions.shape
    num_ranges = range_times.size

    n = initial_guesses.shape[0]

    if n == num_sensor_pos:
        initial_guesses = initial_guesses[:, None, :]  # (num_sensor_pos, 1, coord_dim)
    elif n == num_ranges:
        initial_guesses = initial_guesses[None, :, :]  # (1, num_ranges, coord_dim)
    elif n == 1:
        initial_guesses = initial_guesses[None, None, :]  # (1, 1, coord_dim)
    else:
        msg = f"initial guesses ({n}) must match sensors ({num_sensor_pos}), times ({num_ranges}), or be 1"
        raise RuntimeError(msg)

    initial_guesses = np.broadcast_to(initial_guesses, (num_sensor_pos, num_ranges, coord_dim))

    try:
        sensor_positions = np.broadcast_to(sensor_positions, sensor_velocities.shape)
    except ValueError as exc:
        msg = f"sensor positions {sensor_positions.shape} != sensor velocities {sensor_velocities.shape}"
        raise RuntimeError(msg) from exc

    ground_points = np.zeros((sensor_positions.size // 3, range_times.size, 3))
    for id_az, sensor_params in enumerate(zip(sensor_positions, sensor_velocities, strict=True)):
        ground_points[id_az, ...] = _direct_geocoding_monostatic_newton(
            sensor_positions=sensor_params[0],
            sensor_velocities=sensor_params[1],
            initial_guesses=initial_guesses[id_az, ...],
            range_times=range_times,
            doppler_frequencies=doppler_frequencies,
            altitude=altitude,
            wavelength=wavelength,
        )

    return ground_points.squeeze() if not one_size_array_flag else ground_points.squeeze(axis=0)


def direct_geocoding_bistatic_core(
    sensor_positions_rx: npt.NDArray[np.floating],
    sensor_velocities_rx: npt.NDArray[np.floating],
    sensor_positions_tx: npt.NDArray[np.floating],
    sensor_velocities_tx: npt.NDArray[np.floating],
    range_times: float | npt.NDArray[np.floating],
    doppler_frequencies: float | npt.NDArray[np.floating],
    wavelength: float,
    altitude: float,
    initial_guesses: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Compute ground points via bistatic direct geocoding.

    Parameters
    ----------
    sensor_positions_rx : npt.NDArray[np.floating]
        position of the receiver, with shape (3,) or (N, 3)
    sensor_velocities_rx : npt.NDArray[np.floating]
        velocities of the receiver, with shape (3,) or (N, 3)
    sensor_positions_tx : npt.NDArray[np.floating]
        position of the transmitter, with shape (3,) or (M, 3), where M is the number of range times
    sensor_velocities_tx : npt.NDArray[np.floating]
        velocities of the transmitter, with shape (3,) or (M, 3), where M is the number of range times
    range_times : float | npt.NDArray[np.floating]
        range times with shape float or (M,)
    doppler_frequencies : float | npt.NDArray[np.floating]
        frequency_doppler_centroid value, single value or array (M,)
    wavelength : float
        carrier signal wavelength
    altitude : float
        altitude with respect to WGS84 ellipsoid
    initial_guesses : npt.NDArray[np.floating]
        initial guess for Newton method, with shape (3,), (N, 3) or (M, 3)

    Returns
    -------
    npt.NDArray[np.floating]
        ground points with shape (N, M, 3)

    """
    try:
        doppler_frequencies = np.broadcast_to(doppler_frequencies, np.shape(range_times))
    except ValueError as exc:
        msg = f"doppler frequencies {np.shape(doppler_frequencies)} != range times {np.shape(range_times)}"
        raise RuntimeError(msg) from exc

    ground_points = np.zeros((sensor_positions_rx.size // 3, np.size(range_times), 3))
    for id_rng, items in enumerate(
        zip(
            np.atleast_1d(range_times),
            np.atleast_1d(doppler_frequencies),
            np.atleast_2d(sensor_positions_tx),
            np.atleast_2d(sensor_velocities_tx),
            strict=True,
        )
    ):
        ground_points[..., id_rng, :] = _direct_geocoding_bistatic_newton(
            sensor_positions_rx=sensor_positions_rx,
            sensor_velocities_rx=sensor_velocities_rx,
            initial_guesses=initial_guesses,
            sensor_position_tx=items[2],
            sensor_velocity_tx=items[3],
            range_times=items[0],
            doppler_frequencies=items[1],
            altitude=altitude,
            wavelength=wavelength,
        )

    one_size_array_flag = (sensor_positions_rx.ndim == 2 and sensor_positions_rx.size / 3 == 1) or (
        sensor_positions_tx.ndim == 2 and sensor_positions_tx.size / 3 == 1 and sensor_positions_rx.size / 3 == 1
    )
    return ground_points.squeeze() if not one_size_array_flag else ground_points.squeeze(axis=0)


def _direct_geocoding_monostatic_newton(
    sensor_positions: npt.NDArray[np.floating],
    sensor_velocities: npt.NDArray[np.floating],
    initial_guesses: npt.NDArray[np.floating],
    range_times: float | npt.NDArray[np.floating],
    doppler_frequencies: float | npt.NDArray[np.floating],
    wavelength: float,
    altitude: float,
    max_iterations: int = 8,
    increment_tolerance: float = 1e-5,
) -> npt.NDArray[np.floating]:
    """Solve direct geocoding monostatic equations with Newton iterations.

    Parameters
    ----------
    sensor_positions : npt.NDArray[np.floating]
        sensor positions with shape (3,) or (N, 3)
    sensor_velocities : npt.NDArray[np.floating]
        sensor velocities with shape (3,) or (N, 3)
    initial_guesses : npt.NDArray[np.floating]
        initial guesses with shape (3,) or (N, 3)
    range_times : float | npt.NDArray[np.floating]
        range times with shape float or (N,)
    doppler_frequencies : float | npt.NDArray[np.floating]
        doppler frequencies with shape float or (N,)
    wavelength : float
        carrier signal wavelength
    altitude : float
        altitude with respect to WGS84 ellipsoid
    max_iterations : int, optional
        maximum number of iterations for Newton method, by default 8
    increment_tolerance : float, optional
        tolerance for Newton convergence in meters, by default 1e-5

    Returns
    -------
    npt.NDArray[np.floating]
        ground points with shape (3,) or (N, 3)

    """
    tolerance_squared = increment_tolerance * increment_tolerance

    range_distance_square = (speed_of_light * range_times / 2.0) ** 2
    geoid_r_min = WGS84.b + altitude
    geoid_r_max = WGS84.a + altitude
    r_ep2 = geoid_r_min**2
    r_ee2 = geoid_r_max**2

    ground_points_guess = initial_guesses.copy()

    for _ in range(max_iterations):
        line_of_sight = sensor_positions - ground_points_guess
        distance_square = np.sum(line_of_sight * line_of_sight, axis=-1)
        distance = np.sqrt(distance_square)
        los_vel_product = np.sum(sensor_velocities * line_of_sight, axis=-1)

        range_equation = distance_square - range_distance_square
        grad_range_equation = -2 * line_of_sight

        doppler_eq, grad_doppler_eq = doppler_equation(
            pv_scalar=los_vel_product,
            los=line_of_sight,
            sensor_velocity=sensor_velocities,
            distance=distance,
            wavelength=wavelength,
            doppler_frequency=doppler_frequencies,
        )

        residuals = [
            range_equation,
            _ellipse_equation(ground_points_guess, r_ee2, r_ep2),
            doppler_eq,
        ]
        jacobians = [
            [
                grad_range_equation[..., k],
                _der_ellipse_equation_xi(ground_points_guess, k, r_ee2, r_ep2),
                grad_doppler_eq[..., k],
            ]
            for k in range(3)
        ]

        delta_err = -_inv_3x3_transpose(jacobians, residuals)
        ground_points_guess = ground_points_guess + delta_err

        increment_squared = np.sum(delta_err * delta_err, axis=-1)
        if np.max(np.abs(increment_squared)) <= tolerance_squared:
            break
    else:
        msg = (
            f"Newton did not converge: maximum number of iterations {max_iterations} reached."
            # pyrefly: ignore [unbound-name]
            f"Residual error {delta_err}"
        )
        raise RuntimeError(msg)

    array_size_one_flag = ground_points_guess.ndim == sensor_positions.ndim == sensor_velocities.ndim == 1
    return ground_points_guess if not array_size_one_flag else ground_points_guess.squeeze()


def _direct_geocoding_bistatic_newton(  # noqa: PLR0913
    sensor_positions_rx: npt.NDArray[np.floating],
    sensor_velocities_rx: npt.NDArray[np.floating],
    initial_guesses: npt.NDArray[np.floating],
    sensor_position_tx: npt.NDArray[np.floating],
    sensor_velocity_tx: npt.NDArray[np.floating],
    range_times: float | npt.NDArray[np.floating],
    doppler_frequencies: float | npt.NDArray[np.floating],
    wavelength: float,
    altitude: float,
    max_iterations: int = 8,
    tolerance: float = 1e-5,
) -> npt.NDArray[np.floating]:
    """Newton solving method for direct geocoding bistatic.

    Parameters
    ----------
    sensor_positions_rx : npt.NDArray[np.floating]
        sensor rx position array, with shape (3,) or (N,3)
    sensor_velocities_rx : npt.NDArray[np.floating]
        sensor rx velocities array, with shape (3,) or (N,3)
    initial_guesses : npt.NDArray[np.floating]
        initial guesses array, with shape (3,) or (N,3)
    sensor_position_tx : npt.NDArray[np.floating]
        sensor tx position, with shape (3,)
    sensor_velocity_tx : npt.NDArray[np.floating]
        sensor tx velocity, with shape (3,)
    range_times : float | npt.NDArray[np.floating]
        range times with shape float or (N,)
    doppler_frequencies : float | npt.NDArray[np.floating]
        doppler frequencies with shape float or (N,)
    wavelength : float
        carrier signal wavelength
    altitude : float
        altitude with respect to WGS84 ellipsoid
    max_iterations : int, optional
        maximum number of iterations for Newton method, by default 8
    tolerance : float, optional
        tolerance for Newton convergence in meters, by default 1e-5

    Returns
    -------
    npt.NDArray[np.floating]
        ground points with shape (3,) or (N, 3)

    """
    tolerance_squared = tolerance * tolerance

    range_distance_square = (speed_of_light * range_times) ** 2
    geoid_r_min = WGS84.b + altitude
    geoid_r_max = WGS84.a + altitude
    r_ep2 = geoid_r_min**2
    r_ee2 = geoid_r_max**2

    ground_points_guess = initial_guesses.copy()

    for _ in range(max_iterations):
        line_of_sight_rx = sensor_positions_rx - ground_points_guess
        distance_square_rx = np.sum(line_of_sight_rx * line_of_sight_rx, axis=-1)
        distance_rx = np.sqrt(distance_square_rx)
        los_vel_product_rx = np.sum(sensor_velocities_rx * line_of_sight_rx, axis=-1)

        line_of_sight_tx = sensor_position_tx - ground_points_guess
        distance_square_tx = np.sum(line_of_sight_tx * line_of_sight_tx, axis=-1)
        distance_tx = np.sqrt(distance_square_tx)
        los_vel_product_tx = np.sum(sensor_velocity_tx * line_of_sight_tx, axis=-1)

        distance = distance_rx + distance_tx
        range_equation = distance**2 - range_distance_square
        grad_range_equation = (
            -2
            * distance[..., np.newaxis]
            * (line_of_sight_rx / distance_rx[..., np.newaxis] + line_of_sight_tx / distance_tx[..., np.newaxis])
        )

        doppler_eq_rx, grad_doppler_eq_rx = doppler_equation(
            wavelength=wavelength,
            pv_scalar=los_vel_product_rx,
            distance=distance_rx,
            doppler_frequency=doppler_frequencies,
            sensor_velocity=sensor_velocities_rx,
            los=line_of_sight_rx,
        )
        doppler_eq_tx, grad_doppler_eq_tx = doppler_equation(
            wavelength=wavelength,
            pv_scalar=los_vel_product_tx,
            distance=distance_tx,
            doppler_frequency=doppler_frequencies,
            sensor_velocity=sensor_velocity_tx,
            los=line_of_sight_tx,
        )

        doppler_eq = (doppler_eq_rx + doppler_eq_tx) / 2
        grad_doppler_eq = (grad_doppler_eq_rx + grad_doppler_eq_tx) / 2

        residuals = [
            range_equation,
            _ellipse_equation(ground_points_guess, r_ee2, r_ep2),
            doppler_eq,
        ]
        jacobians = [
            [
                grad_range_equation[..., k],
                _der_ellipse_equation_xi(ground_points_guess, k, r_ee2, r_ep2),
                grad_doppler_eq[..., k],
            ]
            for k in range(3)
        ]

        delta_err = -_inv_3x3_transpose(jacobians, residuals)
        ground_points_guess = ground_points_guess + delta_err

        increment_squared = np.sum(delta_err * delta_err, axis=-1)
        if np.max(np.abs(increment_squared)) <= tolerance_squared:
            break
    else:
        msg = (
            f"Newton did not converge: maximum number of iterations {max_iterations} reached."
            # pyrefly: ignore [unbound-name]
            f"Residual error {delta_err}"
        )
        raise RuntimeError(msg)

    return ground_points_guess


def _inv_3x3_transpose(
    jac: Sequence[Sequence[float | npt.NDArray[np.floating]]], func: Sequence[float | npt.NDArray[np.floating]]
) -> npt.NDArray[np.floating]:
    """Perform inverse of 3x3 matrix using explicit form."""
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

    return (np.stack([x_val, y_val, z_val], axis=-1) / np.asarray(det)[..., np.newaxis]).squeeze()


def _ellipse_equation(coords: npt.NDArray[np.floating], r_ee2: float, r_ep2: float) -> npt.NDArray[np.floating]:
    """Evaluate ellipsoid equation residual.

    Parameters
    ----------
    coords : npt.NDArray[np.floating]
        coords shape (3,) or (N, 3)
    r_ee2 : float
        radius square along x and y directions
    r_ep2 : float
        radius square along z direction

    Returns
    -------
    npt.NDArray[np.floating]
        residuals scalar or with shape (N, 3)

    """
    return (
        (coords[..., 0] * coords[..., 0] + coords[..., 1] * coords[..., 1]) / r_ee2
        + coords[..., 2] * coords[..., 2] / r_ep2
        - 1.0
    )


def _der_ellipse_equation_xi(
    coords: npt.NDArray[np.floating], i_coord: int, r_ee2: float, r_ep2: float
) -> npt.NDArray[np.floating]:
    """Evaluate ellipsoid equation partial derivative w.r.t. to i_coord.

    Parameters
    ----------
    coords : npt.NDArray[np.floating]
        coords shape (3,) or (N, 3)
    i_coord : int
        direction index of the partial derivative dx_i
    r_ee2 : float
        radius square along x and y directions
    r_ep2 : float
        radius square along z direction

    Returns
    -------
    npt.NDArray[np.floating]
        derivative along the selected direction scalar or with shape (N, 3)

    """
    radius_square = r_ee2 if i_coord < 2 else r_ep2

    return 2 * coords[..., i_coord] / radius_square
