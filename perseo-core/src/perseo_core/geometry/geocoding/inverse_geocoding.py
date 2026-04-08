# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Inverse geocoding main functionalities"""

from __future__ import annotations

import numpy as np

import perseo_core.geometry.geocoding.inverse_geocoding_core as inverse_core
from perseo_core.models.trajectory import Trajectory
from perseo_core.models.types import (
    CoordinatesArrayType,
    ExtendedDatetimeArrayType,
    ExtendedDatetimeType,
    FloatArrayType,
)


def inverse_geocoding_monostatic(
    trajectory: Trajectory,
    ground_points: CoordinatesArrayType,
    frequencies_doppler_centroid: float | FloatArrayType,
    wavelength: float,
    az_initial_time_guesses: ExtendedDatetimeType | ExtendedDatetimeArrayType | None = None,
    init_guess_search_time_step: float | None = None,
) -> tuple[ExtendedDatetimeType | ExtendedDatetimeArrayType, float | FloatArrayType]:
    """Monostatic inverse geocoding computation.

    Parameters
    ----------
    trajectory : TwiceDifferentiable3DCurve
        sensor's trajectory, compliant to the TwiceDifferentiable3DCurve protocol
    ground_points : CoordinatesArrayType
        ground points to inverse geocode in XYZ coordinates, in the form (3,) or (N, 3)
    frequencies_doppler_centroid : float | FloatArrayType
        doppler frequencies centroid values to perform the inverse geocoding, in the form float or (N,).
        the number of frequencies must be 1 or equal to the number of points provided (if more than 1).
        If just 1 ground point is provided, several frequencies can be given to compute inverse geocoding at
        each different input value
    wavelength : float
        carrier signal wavelength
    az_initial_time_guesses : ExtendedDatetimeType | ExtendedDatetimeArrayType | None, optional
        azimuth times initial guesses to limit and guide the search of solutions, in the form (N,) or as a single time.
        If None, it is automatically computed, by default None
    init_guess_search_time_step : float | None, optional
        if an azimuth initial guess for the Newton method is not provided, this parameter will be used to generate a
        time axis with this time step and the same time domain as the input trajectory to automatically compute the
        guess, the same step is used for both trajectories, by default None

    Returns
    -------
    ExtendedDatetimeType | ExtendedDatetimeArrayType
        azimuth times array
    float | FloatArrayType
        range times array
    """

    ground_points = np.asarray(ground_points)

    if az_initial_time_guesses is not None:
        az_initial_time_guesses = np.asarray(az_initial_time_guesses)
    else:
        if init_guess_search_time_step is None:
            raise RuntimeError(
                "Invalid inputs: specify one between az_initial_time_guesses and init_guess_search_time_step"
            )
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
            frequencies_doppler_centroid=frequencies_doppler_centroid,
            wavelength=wavelength,
        )

    # performing actual inverse geocoding monostatic
    azimuth_times, range_times = inverse_core.inverse_geocoding_monostatic_core(
        trajectory=trajectory,
        ground_points=ground_points,
        frequencies_doppler_centroid=frequencies_doppler_centroid,
        wavelength=wavelength,
        initial_guesses=az_initial_time_guesses,
    )

    return azimuth_times, range_times


def inverse_geocoding_monostatic_init(
    trajectory: Trajectory,
    ground_points: CoordinatesArrayType,
    time_axis: ExtendedDatetimeArrayType,
    frequencies_doppler_centroid: float | FloatArrayType,
    wavelength: float,
) -> ExtendedDatetimeType | ExtendedDatetimeArrayType:
    """Function to compute azimuth initial guess for Newton method for monostatic inverse geocoding.

    In principle each input ground point could be seen several times by the orbit if it contains multiple periods.
    In this case, only the first occurrence is taken, i.e. the solution corresponding to the first period (the smallest
    in terms of time).

    If all the occurrences are needed, please refer to inverse_geocoding_monostatic_init_core in
    geocoding/inverse_geocoding_core instead, and perform the inverse geocoding operation providing the initial guesses
    as inputs.

    Parameters
    ----------
    trajectory : TwiceDifferentiable3DCurve
        sensor's trajectory, compliant to the TwiceDifferentiable3DCurve protocol
    ground_points : CoordinatesArrayType
        ground points to inverse geocode in XYZ coordinates, in the form (3,) or (N, 3)
    time_axis : ExtendedDatetimeArrayType
        sensor's trajectory time axis array
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
        azimuth times initial guesses, one for each input point
    """

    # detecting multiple azimuth solutions
    az_initial_time_guesses = inverse_core.inverse_geocoding_monostatic_init_core(
        trajectory=trajectory,
        time_axis=time_axis,
        ground_points=ground_points,
        frequencies_doppler_centroid=frequencies_doppler_centroid,
        wavelength=wavelength,
    )

    # keeping only first solution for each point
    az_initial_time_guesses = np.array([g[0] for g in az_initial_time_guesses])

    if az_initial_time_guesses.size == 1 and ground_points.ndim == 1:
        az_initial_time_guesses = az_initial_time_guesses[0]

    return az_initial_time_guesses


def inverse_geocoding_bistatic(
    trajectory_rx: Trajectory,
    trajectory_tx: Trajectory,
    ground_points: CoordinatesArrayType,
    frequencies_doppler_centroid: float | FloatArrayType,
    wavelength: float,
    az_initial_time_guesses: ExtendedDatetimeType | ExtendedDatetimeArrayType | None = None,
    init_guess_search_time_step: float | None = None,
) -> tuple[ExtendedDatetimeType | ExtendedDatetimeArrayType, float | FloatArrayType]:
    """Bistatic inverse geocoding computation.

    Parameters
    ----------
    trajectory_rx : TwiceDifferentiable3DCurve
        receiving sensor's trajectory, compliant to the TwiceDifferentiable3DCurve protocol
    trajectory_tx : TwiceDifferentiable3DCurve
        transmitting sensor's trajectory, compliant to the TwiceDifferentiable3DCurve protocol
    ground_points : CoordinatesArrayType
        ground points to inverse geocode in XYZ coordinates, in the form (3,) or (N, 3)
    frequencies_doppler_centroid : float | FloatArrayType
        doppler frequencies centroid values to perform the inverse geocoding, in the form float or (N,).
        the number of frequencies must be 1 or equal to the number of points provided (if more than 1).
        If just 1 ground point is provided, several frequencies can be given to compute inverse geocoding at
        each different input value
    wavelength : float
        carrier signal wavelength
    az_initial_time_guesses : ExtendedDatetimeType | ExtendedDatetimeArrayType | None, optional
        azimuth times initial guesses to limit and guide the search of solutions, in the form (N,) or as a single time.
        If None, it is automatically computed, by default None
    init_guess_search_time_step : float | None, optional
        if an azimuth initial guess for the Newton method is not provided, this parameter will be used to generate a
        time axis with this time step and the same time domain as the input trajectory to automatically compute the
        guess, the same step is used for both trajectories, by default None

    Returns
    -------
    ExtendedDatetimeType | ExtendedDatetimeArrayType
        azimuth times array
    float | FloatArrayType
        range times array

    Raises
    ------
    inverse_core.OrbitsNotOverlappedError
        if the input sensors' trajectories are overlapping
    """

    frequencies_doppler_centroid = (
        np.asarray(frequencies_doppler_centroid)
        if not np.isscalar(frequencies_doppler_centroid)
        else frequencies_doppler_centroid
    )

    if az_initial_time_guesses is not None:
        az_initial_time_guesses = np.asarray(az_initial_time_guesses)
    else:
        if init_guess_search_time_step is None:
            raise RuntimeError(
                "Invalid inputs: specify one between az_initial_time_guesses and init_guess_search_time_step"
            )
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
            frequencies_doppler_centroid=frequencies_doppler_centroid,
            wavelength=wavelength,
        )

    # checking if orbits are overlapped
    axis_start_time = np.max([trajectory_tx.domain[0], trajectory_rx.domain[0]])
    axis_end_time = np.min([trajectory_tx.domain[-1], trajectory_rx.domain[-1]])
    axis_length = axis_end_time - axis_start_time

    if axis_length < 0:
        raise inverse_core.OrbitsNotOverlappedError

    # performing actual inverse geocoding bistatic
    azimuth_times, range_times = inverse_core.inverse_geocoding_bistatic_core(
        trajectory_rx=trajectory_rx,
        trajectory_tx=trajectory_tx,
        ground_points=ground_points,
        frequencies_doppler_centroid=frequencies_doppler_centroid,
        initial_guesses=az_initial_time_guesses,
        wavelength=wavelength,
    )

    return azimuth_times, range_times
