# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Inverse Geocoding module.

This module provides high-level inverse geocoding functions for computing sensor times (azimuth time, range time)
from known ground point coordinates. Inverse geocoding is the reverse operation of direct geocoding, solving for the
acquisition geometry that observes a specific Earth surface location.

The module supports both monostatic (single sensor) and bistatic (separate transmitter/receiver)
configurations, using Newton iterations to solve the system of equations.

All functions support vectorized operations on (N, 3) ground point arrays and handle both scalar
and array inputs for Doppler frequencies. The Trajectory interface allows flexible sensor
path representations while PreciseDateTime ensures precise time handling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

import perseo_core.geometry.geocoding.inverse_geocoding_core as inverse_core

if TYPE_CHECKING:
    from perseo_core.geometry.navigation.trajectory import Trajectory
    from perseo_core.geometry.pointing.attitude import Attitude
    from perseo_core.timing.precise_datetime import PreciseDateTime


def inverse_geocoding_monostatic(
    trajectory: Trajectory,
    ground_points: npt.NDArray[np.floating],
    doppler_frequencies: float | npt.NDArray[np.floating],
    wavelength: float,
    az_initial_time_guesses: PreciseDateTime | np.datetime64 | npt.NDArray | None = None,
    init_guess_search_time_step: float | None = None,
) -> tuple[PreciseDateTime | np.datetime64 | npt.NDArray, float | npt.NDArray[np.floating]]:
    """Monostatic inverse geocoding computation.

    !!! note "Initial guesses"

        One between `az_initial_time_guesses` and `init_guess_search_time_step` inputs must be provided.

    Parameters
    ----------
    trajectory : Trajectory
        sensor trajectory
    ground_points : npt.NDArray[np.floating]
        ground points to inverse geocode in XYZ coordinates, in the form (3,) or (N, 3)
    doppler_frequencies : float | npt.NDArray[np.floating]
        doppler frequencies centroid values to perform the inverse geocoding, in the form float or (N,).
        the number of frequencies must be 1 or equal to the number of points provided (if more than 1).
        If just 1 ground point is provided, several frequencies can be given to compute inverse geocoding at
        each different input value
    wavelength : float
        carrier signal wavelength
    az_initial_time_guesses : PreciseDateTime | np.datetime64 | npt.NDArray | None, optional
        azimuth times initial guesses to limit and guide the search of solutions, in the form (N,) or as a single time.
        If None, it is automatically computed, by default None
    init_guess_search_time_step : float | None, optional
        if an azimuth initial guess for the Newton method is not provided, this parameter will be used to generate a
        time axis with this time step and the same time domain as the input trajectory to automatically compute the
        guess, the same step is used for both trajectories, by default None

    Returns
    -------
    PreciseDateTime | np.datetime64 | npt.NDArray
        azimuth times array
    float | npt.NDArray[np.floating]
        range times array

    """
    ground_points = np.asarray(ground_points)

    if az_initial_time_guesses is not None:
        az_initial_time_guesses = np.asarray(az_initial_time_guesses)
    else:
        if init_guess_search_time_step is None:
            msg = "Invalid inputs: specify one between az_initial_time_guesses and init_guess_search_time_step"
            raise RuntimeError(msg)
        # computing an initial guess for the Newton method
        time_axis = (
            np.arange(0, trajectory.domain[1] - trajectory.domain[0], init_guess_search_time_step)
            + trajectory.domain[0]
        )
        # computing an initial guess for the Newton method
        az_initial_time_guesses = inverse_geocoding_monostatic_init(
            trajectory=trajectory,
            ground_points=ground_points,
            time_axis=time_axis,
            doppler_frequencies=doppler_frequencies,
            wavelength=wavelength,
        )

    # performing actual inverse geocoding monostatic
    azimuth_times, range_times = inverse_core.inverse_geocoding_monostatic_core(
        trajectory=trajectory,
        ground_points=ground_points,
        doppler_frequencies=doppler_frequencies,
        wavelength=wavelength,
        initial_guesses=az_initial_time_guesses,
    )

    return azimuth_times, range_times


def inverse_geocoding_monostatic_with_attitude(
    trajectory: Trajectory,
    attitude: Attitude,
    ground_points: npt.NDArray[np.floating],
    doppler_frequencies: float | npt.NDArray[np.floating],
    wavelength: float,
    dt: float = 0.1,
    az_initial_time_guesses: PreciseDateTime | np.datetime64 | npt.NDArray | None = None,
    init_guess_search_time_step: float | None = None,
) -> tuple[PreciseDateTime | np.datetime64 | npt.NDArray, float | npt.NDArray[np.floating]]:
    """Monostatic inverse geocoding with attitude computation.

    !!! note "Initial guesses"

        One between `az_initial_time_guesses` and `init_guess_search_time_step` inputs must be provided.

    Parameters
    ----------
    trajectory : Trajectory
        sensor trajectory
    attitude : Attitude
        sensor attitude
    ground_points : npt.NDArray[np.floating]
        ground points to inverse geocode in XYZ coordinates, in the form (3,) or (N, 3)
    doppler_frequencies : float | npt.NDArray[np.floating]
        doppler frequencies centroid values to perform the inverse geocoding, in the form float or (N,).
        the number of frequencies must be 1 or equal to the number of points provided (if more than 1).
        If just 1 ground point is provided, several frequencies can be given to compute inverse geocoding at
        each different input value
    wavelength : float
        carrier signal wavelength
    dt : float, optional
        time step for computing the approximate derivative of the boresight normal unit vector, by default 0.1
    az_initial_time_guesses : PreciseDateTime | np.datetime64 | npt.NDArray | None, optional
        azimuth times initial guesses to limit and guide the search of solutions, in the form (N,) or as a single time.
        If None, it is automatically computed, by default None
    init_guess_search_time_step : float | None, optional
        if an azimuth initial guess for the Newton method is not provided, this parameter will be used to generate a
        time axis with this time step and the same time domain as the input trajectory to automatically compute the
        guess, the same step is used for both trajectories, by default None

    Returns
    -------
    PreciseDateTime | np.datetime64 | npt.NDArray
        azimuth times array
    float | npt.NDArray[np.floating]
        range times array

    """
    ground_points = np.asarray(ground_points)

    if az_initial_time_guesses is not None:
        az_initial_time_guesses = np.asarray(az_initial_time_guesses)
    else:
        if init_guess_search_time_step is None:
            msg = "Invalid inputs: specify one between az_initial_time_guesses and init_guess_search_time_step"
            raise RuntimeError(msg)
        # computing an initial guess for the Newton method
        time_axis = (
            np.arange(0, trajectory.domain[1] - trajectory.domain[0], init_guess_search_time_step)
            + trajectory.domain[0]
        )
        # computing an initial guess for the Newton method
        az_initial_time_guesses = inverse_geocoding_monostatic_init(
            trajectory=trajectory,
            ground_points=ground_points,
            time_axis=time_axis,
            doppler_frequencies=doppler_frequencies,
            wavelength=wavelength,
        )

    return inverse_core.inverse_geocoding_monostatic_attitude_core(
        trajectory=trajectory,
        attitude=attitude,
        ground_points=ground_points,
        initial_guesses=az_initial_time_guesses,
        dt=dt,
    )


def inverse_geocoding_monostatic_init(
    trajectory: Trajectory,
    ground_points: npt.NDArray[np.floating],
    time_axis: npt.NDArray,
    doppler_frequencies: float | npt.NDArray[np.floating],
    wavelength: float,
) -> PreciseDateTime | np.datetime64 | npt.NDArray:
    """Compute azimuth initial guess for Newton method for monostatic inverse geocoding.

    In principle each input ground point could be seen several times by the orbit if it contains multiple periods.
    In this case, only the first occurrence is taken, i.e. the solution corresponding to the first period (the smallest
    in terms of time).

    If all the occurrences are needed, please refer to inverse_geocoding_monostatic_init_core in
    geocoding/inverse_geocoding_core instead, and perform the inverse geocoding operation providing the initial guesses
    as inputs.

    Parameters
    ----------
    trajectory : Trajectory
        sensor trajectory
    ground_points : npt.NDArray[np.floating]
        ground points to inverse geocode in XYZ coordinates, in the form (3,) or (N, 3)
    time_axis : npt.NDArray
        sensor trajectory time axis array
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
        azimuth times initial guesses, one for each input point

    """
    # detecting multiple azimuth solutions
    az_initial_time_guesses = inverse_core.inverse_geocoding_monostatic_init_core(
        trajectory=trajectory,
        time_axis=time_axis,
        ground_points=ground_points,
        doppler_frequencies=doppler_frequencies,
        wavelength=wavelength,
    )

    # keeping only first solution for each point
    az_initial_time_guess = np.array([g[0] for g in az_initial_time_guesses])

    if az_initial_time_guess.size == 1 and ground_points.ndim == 1:
        return az_initial_time_guess[0]

    return az_initial_time_guess


def inverse_geocoding_bistatic(
    trajectory_rx: Trajectory,
    trajectory_tx: Trajectory,
    ground_points: npt.NDArray[np.floating],
    doppler_frequencies: float | npt.NDArray[np.floating],
    wavelength: float,
    az_initial_time_guesses: PreciseDateTime | np.datetime64 | npt.NDArray | None = None,
    init_guess_search_time_step: float | None = None,
) -> tuple[PreciseDateTime | np.datetime64 | npt.NDArray, float | npt.NDArray[np.floating]]:
    """Bistatic inverse geocoding computation.

    !!! note "Initial guesses"

        One between `az_initial_time_guesses` and `init_guess_search_time_step` inputs must be provided.

    Parameters
    ----------
    trajectory_rx : Trajectory
        receiving sensor trajectory
    trajectory_tx : Trajectory
        transmitting sensor trajectory
    ground_points : npt.NDArray[np.floating]
        ground points to inverse geocode in XYZ coordinates, in the form (3,) or (N, 3)
    doppler_frequencies : float | npt.NDArray[np.floating]
        doppler frequencies centroid values to perform the inverse geocoding, in the form float or (N,).
        the number of frequencies must be 1 or equal to the number of points provided (if more than 1).
        If just 1 ground point is provided, several frequencies can be given to compute inverse geocoding at
        each different input value
    wavelength : float
        carrier signal wavelength
    az_initial_time_guesses : PreciseDateTime | np.datetime64 | npt.NDArray | None, optional
        azimuth times initial guesses to limit and guide the search of solutions, in the form (N,) or as a single time.
        If None, it is automatically computed, by default None
    init_guess_search_time_step : float | None, optional
        if an azimuth initial guess for the Newton method is not provided, this parameter will be used to generate a
        time axis with this time step and the same time domain as the input trajectory to automatically compute the
        guess, the same step is used for both trajectories, by default None

    Returns
    -------
    PreciseDateTime | np.datetime64 | npt.NDArray
        azimuth times array
    float | npt.NDArray[np.floating]
        range times array

    Raises
    ------
    RuntimeError
        if the input sensors' trajectories are overlapping

    """
    if az_initial_time_guesses is not None:
        az_initial_time_guesses = np.asarray(az_initial_time_guesses)
    else:
        if init_guess_search_time_step is None:
            msg = "Invalid inputs: specify one between az_initial_time_guesses and init_guess_search_time_step"
            raise RuntimeError(msg)
        # computing an initial guess for the Newton method
        time_axis_rx = (
            np.arange(0, trajectory_rx.domain[1] - trajectory_rx.domain[0], init_guess_search_time_step)
            + trajectory_rx.domain[0]
        )
        time_axis_tx = (
            np.arange(0, trajectory_tx.domain[1] - trajectory_tx.domain[0], init_guess_search_time_step)
            + trajectory_tx.domain[0]
        )
        az_initial_time_guesses = inverse_core.inverse_geocoding_bistatic_init_core(
            trajectory_rx=trajectory_rx,
            trajectory_tx=trajectory_tx,
            time_axis_rx=time_axis_rx,
            time_axis_tx=time_axis_tx,
            ground_points=ground_points,
            doppler_frequencies=doppler_frequencies,
            wavelength=wavelength,
        )

    # checking if orbits are overlapped
    axis_start_time = np.max([trajectory_tx.domain[0], trajectory_rx.domain[0]])
    axis_end_time = np.min([trajectory_tx.domain[-1], trajectory_rx.domain[-1]])
    axis_length = axis_end_time - axis_start_time

    if axis_length <= 0:
        msg = "Invalid input trajectories: the two orbits are not overlapped"
        raise RuntimeError(msg)

    # performing actual inverse geocoding bistatic
    azimuth_times, range_times = inverse_core.inverse_geocoding_bistatic_core(
        trajectory_rx=trajectory_rx,
        trajectory_tx=trajectory_tx,
        ground_points=ground_points,
        doppler_frequencies=doppler_frequencies,
        initial_guesses=az_initial_time_guesses,
        wavelength=wavelength,
    )

    return azimuth_times, range_times


__all__ = [
    "inverse_geocoding_bistatic",
    "inverse_geocoding_monostatic",
    "inverse_geocoding_monostatic_init",
    "inverse_geocoding_monostatic_with_attitude",
]
