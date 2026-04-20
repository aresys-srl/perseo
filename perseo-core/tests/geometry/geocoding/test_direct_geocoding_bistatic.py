# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for geometry/geocoding/direct_geocoding.py and direct_geocoding_core.py bistatic functionalities"""

from __future__ import annotations

import unittest

import numpy as np
from numpy import typing as npt
from scipy.constants import speed_of_light

from perseo_core.geometry.doppler import doppler_equation_bistatic_residuals
from perseo_core.geometry.geocoding.direct_geocoding import direct_geocoding_bistatic
from perseo_core.geometry.geocoding.direct_geocoding_core import (
    _direct_geocoding_bistatic_newton,
    _ellipse_equation,
    direct_geocoding_bistatic_core,
)
from perseo_core.geometry.utilities.ellipsoid import WGS84
from perseo_core.models.enums import SensorLookDirection


def _range_equation_residual_bistatic(
    sensor_pos_rx: np.ndarray,
    sensor_pos_tx: np.ndarray,
    ground_points: np.ndarray,
    range_time: float | np.ndarray,
) -> np.ndarray:
    """Evaluating range equation residual for bistatic sensors.

    Parameters
    ----------
    sensor_pos_rx : np.ndarray
        sensor rx position, (3,) or (N, 3)
    sensor_pos_tx : np.ndarray
        sensor tx position, (3,) or (N, 3)
    ground_points : np.ndarray
        ground points from direct geocoding solution, (3,) or (N, 3)
    range_time : float
        range time

    Returns
    -------
    np.ndarray
        range equation residual
    """

    rng_dst = speed_of_light * range_time / 2.0
    los_rx = sensor_pos_rx - ground_points
    los_tx = sensor_pos_tx - ground_points
    distance_rx = np.sqrt(np.sum(los_rx * los_rx, axis=-1))
    distance_tx = np.sqrt(np.sum(los_tx * los_tx, axis=-1))

    rng_residual_rx = (distance_rx - rng_dst) / speed_of_light
    rng_residual_tx = (distance_tx - rng_dst) / speed_of_light

    return np.array(rng_residual_rx + rng_residual_tx)


def _ellipse_equation_residual(ground_points: np.ndarray) -> np.ndarray:
    """Ellipse equation residual.

    Parameters
    ----------
    ground_points : np.ndarray
        ground points from direct geocoding solution, (3,) or (N, 3)

    Returns
    -------
    np.ndarray
        ellipse equation residual
    """
    r_ep2 = WGS84.b**2
    r_ee2 = WGS84.a**2

    ellipse_residual = _ellipse_equation(ground_points, r_ee2, r_ep2)

    return ellipse_residual


class _DirectGeocodingBistaticBase(unittest.TestCase):
    def setUp(self) -> None:
        self.position = np.array([4387348.749948771, 762123.3489877012, 4553067.931912004])
        self.velocity = np.array([-856.1384108174528, -329.7629775067583, 398.55830806407346])
        self.initial_guess = np.array([4385932.628762595, 764443.4718341012, 4551945.624046889])

        self.range_times = np.array([2.05624579e-05])
        self.doppler_frequency = 0.0
        self.altitude = 0.0
        self.look_direction = SensorLookDirection.RIGHT_LOOKING
        self.wavelength = 1.0

        self.N = 4
        self.M = 5

        self.tolerance = 1e-6
        self.residual_tolerance = 1e-8

        self.results = np.array([4385882.195057692, 764600.9869913795, 4551967.6143934])

    def _expected_output(self, shape: tuple[int, ...]) -> npt.NDArray[np.floating]:
        return np.full(shape, self.results)


class DirectGeocodingBistaticTest(_DirectGeocodingBistaticBase):
    """Testing direct_geocoding_bistatic with consolidated subtests."""

    def _check_residuals(
        self,
        sensor_positions_rx: npt.NDArray,
        sensor_velocities_rx: npt.NDArray,
        sensor_positions_tx: npt.NDArray,
        sensor_velocities_tx: npt.NDArray,
        range_times: float | npt.NDArray,
        out: npt.NDArray[np.floating],
    ) -> None:
        doppler_residual = doppler_equation_bistatic_residuals(
            sensor_pos_rx=sensor_positions_rx,
            sensor_pos_tx=sensor_positions_tx,
            sensor_vel_rx=sensor_velocities_rx,
            sensor_vel_tx=sensor_velocities_tx,
            ground_points=out,
            doppler_freq=self.doppler_frequency,
            wavelength=self.wavelength,
        )
        range_residual = _range_equation_residual_bistatic(
            sensor_pos_rx=sensor_positions_rx,
            sensor_pos_tx=sensor_positions_tx,
            ground_points=out,
            range_time=range_times,
        )
        ellipse_residual = _ellipse_equation_residual(ground_points=out)

        np.testing.assert_allclose(doppler_residual, 0, atol=self.residual_tolerance, rtol=0)
        np.testing.assert_allclose(range_residual, 0, atol=self.residual_tolerance, rtol=0)
        np.testing.assert_allclose(ellipse_residual, 0, atol=self.residual_tolerance, rtol=0)

    def _check_output(self, expected_shape: tuple[int, ...], out: npt.NDArray[np.floating]) -> None:
        self.assertEqual(out.shape, expected_shape)
        np.testing.assert_allclose(
            out,
            self._expected_output(expected_shape),
            atol=self.tolerance,
            rtol=0,
        )

    def test_direct_geocoding_bistatic_1_point(self) -> None:
        """Testing direct_geocoding_bistatic with 1-point cases."""
        test_cases = [
            {
                "name": "1-point rx_pos=1d rx_vel=1d tx_pos=1d tx_vel=1d init=none",
                "sensor_positions_rx": self.position,
                "sensor_velocities_rx": self.velocity,
                "sensor_positions_tx": self.position,
                "sensor_velocities_tx": self.velocity,
                "range_times": self.range_times[0],
                "doppler_frequencies": self.doppler_frequency,
                "initial_guesses": None,
                "expected_shape": (3,),
            },
            {
                "name": "1-point rx_pos=1d rx_vel=1d tx_pos=1d tx_vel=1d init=1d",
                "sensor_positions_rx": self.position,
                "sensor_velocities_rx": self.velocity,
                "sensor_positions_tx": self.position,
                "sensor_velocities_tx": self.velocity,
                "range_times": self.range_times[0],
                "doppler_frequencies": self.doppler_frequency,
                "initial_guesses": self.initial_guess,
                "expected_shape": (3,),
            },
            {
                "name": "1-point rx_pos=1d rx_vel=1d tx_pos=1d tx_vel=1d init=2d",
                "sensor_positions_rx": self.position,
                "sensor_velocities_rx": self.velocity,
                "sensor_positions_tx": self.position,
                "sensor_velocities_tx": self.velocity,
                "range_times": self.range_times[0],
                "doppler_frequencies": self.doppler_frequency,
                "initial_guesses": self.initial_guess.reshape(1, 3),
                "expected_shape": (3,),
            },
            {
                "name": "1-point rx_pos=1d rx_vel=1d tx_pos=2d tx_vel=2d init=none",
                "sensor_positions_rx": self.position,
                "sensor_velocities_rx": self.velocity,
                "sensor_positions_tx": self.position.reshape(1, 3),
                "sensor_velocities_tx": self.velocity.reshape(1, 3),
                "range_times": self.range_times[0],
                "doppler_frequencies": self.doppler_frequency,
                "initial_guesses": None,
                "expected_shape": (1, 3),
            },
            {
                "name": "1-point rx_pos=1d rx_vel=1d tx_pos=2d tx_vel=2d init=1d",
                "sensor_positions_rx": self.position,
                "sensor_velocities_rx": self.velocity,
                "sensor_positions_tx": self.position.reshape(1, 3),
                "sensor_velocities_tx": self.velocity.reshape(1, 3),
                "range_times": self.range_times[0],
                "doppler_frequencies": self.doppler_frequency,
                "initial_guesses": self.initial_guess,
                "expected_shape": (1, 3),
            },
            {
                "name": "1-point rx_pos=1d rx_vel=1d tx_pos=2d tx_vel=2d init=2d",
                "sensor_positions_rx": self.position,
                "sensor_velocities_rx": self.velocity,
                "sensor_positions_tx": self.position.reshape(1, 3),
                "sensor_velocities_tx": self.velocity.reshape(1, 3),
                "range_times": self.range_times[0],
                "doppler_frequencies": self.doppler_frequency,
                "initial_guesses": self.initial_guess.reshape(1, 3),
                "expected_shape": (1, 3),
            },
            {
                "name": "1-point rx_pos=2d rx_vel=2d tx_pos=1d tx_vel=1d init=none",
                "sensor_positions_rx": self.position.reshape(1, 3),
                "sensor_velocities_rx": self.velocity.reshape(1, 3),
                "sensor_positions_tx": self.position,
                "sensor_velocities_tx": self.velocity,
                "range_times": self.range_times[0],
                "doppler_frequencies": self.doppler_frequency,
                "initial_guesses": None,
                "expected_shape": (1, 3),
            },
            {
                "name": "1-point rx_pos=2d rx_vel=2d tx_pos=1d tx_vel=1d init=1d",
                "sensor_positions_rx": self.position.reshape(1, 3),
                "sensor_velocities_rx": self.velocity.reshape(1, 3),
                "sensor_positions_tx": self.position,
                "sensor_velocities_tx": self.velocity,
                "range_times": self.range_times[0],
                "doppler_frequencies": self.doppler_frequency,
                "initial_guesses": self.initial_guess,
                "expected_shape": (1, 3),
            },
            {
                "name": "1-point rx_pos=2d rx_vel=2d tx_pos=2d tx_vel=2d init=none",
                "sensor_positions_rx": self.position.reshape(1, 3),
                "sensor_velocities_rx": self.velocity.reshape(1, 3),
                "sensor_positions_tx": self.position.reshape(1, 3),
                "sensor_velocities_tx": self.velocity.reshape(1, 3),
                "range_times": self.range_times[0],
                "doppler_frequencies": self.doppler_frequency,
                "initial_guesses": None,
                "expected_shape": (1, 3),
            },
            {
                "name": "1-point rx_pos=2d rx_vel=2d tx_pos=2d tx_vel=2d init=2d",
                "sensor_positions_rx": self.position.reshape(1, 3),
                "sensor_velocities_rx": self.velocity.reshape(1, 3),
                "sensor_positions_tx": self.position.reshape(1, 3),
                "sensor_velocities_tx": self.velocity.reshape(1, 3),
                "range_times": self.range_times[0],
                "doppler_frequencies": self.doppler_frequency,
                "initial_guesses": self.initial_guess.reshape(1, 3),
                "expected_shape": (1, 3),
            },
        ]

        for case in test_cases:
            with self.subTest(case=case["name"]):
                out = direct_geocoding_bistatic(
                    sensor_positions_rx=case["sensor_positions_rx"],
                    sensor_velocities_rx=case["sensor_velocities_rx"],
                    sensor_positions_tx=case["sensor_positions_tx"],
                    sensor_velocities_tx=case["sensor_velocities_tx"],
                    range_times=case["range_times"],
                    look_direction=self.look_direction,
                    altitude=self.altitude,
                    doppler_frequencies=case["doppler_frequencies"],
                    wavelength=self.wavelength,
                    initial_guesses=case["initial_guesses"],
                )

                self._check_residuals(
                    case["sensor_positions_rx"],
                    case["sensor_velocities_rx"],
                    case["sensor_positions_tx"],
                    case["sensor_velocities_tx"],
                    case["range_times"],
                    out,
                )
                self._check_output(case["expected_shape"], out)

    def test_direct_geocoding_bistatic_N_points(self) -> None:
        """Testing direct_geocoding_bistatic with N-point cases."""
        test_cases = [
            {
                "name": "N-points tx_pos=1d tx_vel=1d init=1d",
                "sensor_positions_rx": np.full((self.N, 3), self.position),
                "sensor_velocities_rx": np.full((self.N, 3), self.velocity),
                "sensor_positions_tx": self.position,
                "sensor_velocities_tx": self.velocity,
                "range_times": self.range_times,
                "doppler_frequencies": self.doppler_frequency,
                "initial_guesses": self.initial_guess,
                "expected_shape": (self.N, 3),
            },
            {
                "name": "N-points tx_pos=1d tx_vel=1d init=2d",
                "sensor_positions_rx": np.full((self.N, 3), self.position),
                "sensor_velocities_rx": np.full((self.N, 3), self.velocity),
                "sensor_positions_tx": self.position,
                "sensor_velocities_tx": self.velocity,
                "range_times": self.range_times,
                "doppler_frequencies": self.doppler_frequency,
                "initial_guesses": np.full((self.N, 3), self.initial_guess),
                "expected_shape": (self.N, 3),
            },
            {
                "name": "N-points tx_pos=2d tx_vel=2d init=1d",
                "sensor_positions_rx": np.full((self.N, 3), self.position),
                "sensor_velocities_rx": np.full((self.N, 3), self.velocity),
                "sensor_positions_tx": self.position.reshape(1, 3),
                "sensor_velocities_tx": self.velocity.reshape(1, 3),
                "range_times": self.range_times,
                "doppler_frequencies": self.doppler_frequency,
                "initial_guesses": self.initial_guess,
                "expected_shape": (self.N, 3),
            },
            {
                "name": "N-points tx_pos=2d tx_vel=2d init=2d",
                "sensor_positions_rx": np.full((self.N, 3), self.position),
                "sensor_velocities_rx": np.full((self.N, 3), self.velocity),
                "sensor_positions_tx": self.position.reshape(1, 3),
                "sensor_velocities_tx": self.velocity.reshape(1, 3),
                "range_times": self.range_times,
                "doppler_frequencies": self.doppler_frequency,
                "initial_guesses": self.initial_guess.reshape(1, 3),
                "expected_shape": (self.N, 3),
            },
        ]

        for case in test_cases:
            with self.subTest(case=case["name"]):
                out = direct_geocoding_bistatic(
                    sensor_positions_rx=case["sensor_positions_rx"],
                    sensor_velocities_rx=case["sensor_velocities_rx"],
                    sensor_positions_tx=case["sensor_positions_tx"],
                    sensor_velocities_tx=case["sensor_velocities_tx"],
                    range_times=case["range_times"],
                    look_direction=self.look_direction,
                    altitude=self.altitude,
                    doppler_frequencies=case["doppler_frequencies"],
                    wavelength=self.wavelength,
                    initial_guesses=case["initial_guesses"],
                )

                self._check_residuals(
                    case["sensor_positions_rx"],
                    case["sensor_velocities_rx"],
                    case["sensor_positions_tx"],
                    case["sensor_velocities_tx"],
                    case["range_times"],
                    out,
                )
                self._check_output(case["expected_shape"], out)

    def test_direct_geocoding_bistatic_M_points(self) -> None:
        """Testing direct_geocoding_bistatic with M-point cases."""
        test_cases = [
            {
                "name": "M-points rx_pos=1d rx_vel=1d init=1d doppler=scalar",
                "sensor_positions_rx": self.position,
                "sensor_velocities_rx": self.velocity,
                "sensor_positions_tx": np.full((self.M, 3), self.position),
                "sensor_velocities_tx": np.full((self.M, 3), self.velocity),
                "range_times": np.repeat(self.range_times, self.M),
                "doppler_frequencies": self.doppler_frequency,
                "initial_guesses": self.initial_guess,
                "expected_shape": (self.M, 3),
            },
            {
                "name": "M-points rx_pos=1d rx_vel=1d init=1d doppler=array",
                "sensor_positions_rx": self.position,
                "sensor_velocities_rx": self.velocity,
                "sensor_positions_tx": np.full((self.M, 3), self.position),
                "sensor_velocities_tx": np.full((self.M, 3), self.velocity),
                "range_times": np.repeat(self.range_times, self.M),
                "doppler_frequencies": np.repeat(self.doppler_frequency, self.M),
                "initial_guesses": self.initial_guess,
                "expected_shape": (self.M, 3),
            },
            {
                "name": "M-points rx_pos=1d rx_vel=1d init=2d doppler=scalar",
                "sensor_positions_rx": self.position,
                "sensor_velocities_rx": self.velocity,
                "sensor_positions_tx": np.full((self.M, 3), self.position),
                "sensor_velocities_tx": np.full((self.M, 3), self.velocity),
                "range_times": np.repeat(self.range_times, self.M),
                "doppler_frequencies": self.doppler_frequency,
                "initial_guesses": self.initial_guess.reshape(1, 3),
                "expected_shape": (self.M, 3),
            },
            {
                "name": "M-points rx_pos=2d rx_vel=2d init=1d doppler=scalar",
                "sensor_positions_rx": self.position.reshape(1, 3),
                "sensor_velocities_rx": self.velocity.reshape(1, 3),
                "sensor_positions_tx": np.full((self.M, 3), self.position),
                "sensor_velocities_tx": np.full((self.M, 3), self.velocity),
                "range_times": np.repeat(self.range_times, self.M),
                "doppler_frequencies": self.doppler_frequency,
                "initial_guesses": self.initial_guess,
                "expected_shape": (self.M, 3),
            },
            {
                "name": "M-points rx_pos=1d rx_vel=1d init=2d doppler=scalar",
                "sensor_positions_rx": self.position,
                "sensor_velocities_rx": self.velocity,
                "sensor_positions_tx": np.full((self.M, 3), self.position),
                "sensor_velocities_tx": np.full((self.M, 3), self.velocity),
                "range_times": np.repeat(self.range_times, self.M),
                "doppler_frequencies": self.doppler_frequency,
                "initial_guesses": self.initial_guess.reshape(1, 3),
                "expected_shape": (self.M, 3),
            },
            {
                "name": "M-points rx_pos=1d rx_vel=1d init=2d doppler=array",
                "sensor_positions_rx": self.position,
                "sensor_velocities_rx": self.velocity,
                "sensor_positions_tx": np.full((self.M, 3), self.position),
                "sensor_velocities_tx": np.full((self.M, 3), self.velocity),
                "range_times": np.repeat(self.range_times, self.M),
                "doppler_frequencies": np.repeat(self.doppler_frequency, self.M),
                "initial_guesses": self.initial_guess.reshape(1, 3),
                "expected_shape": (self.M, 3),
            },
            {
                "name": "M-points rx_pos=2d rx_vel=2d init=1d doppler=scalar",
                "sensor_positions_rx": self.position.reshape(1, 3),
                "sensor_velocities_rx": self.velocity.reshape(1, 3),
                "sensor_positions_tx": np.full((self.M, 3), self.position),
                "sensor_velocities_tx": np.full((self.M, 3), self.velocity),
                "range_times": np.repeat(self.range_times, self.M),
                "doppler_frequencies": self.doppler_frequency,
                "initial_guesses": self.initial_guess,
                "expected_shape": (self.M, 3),
            },
            {
                "name": "M-points rx_pos=2d rx_vel=2d init=1d doppler=array",
                "sensor_positions_rx": self.position.reshape(1, 3),
                "sensor_velocities_rx": self.velocity.reshape(1, 3),
                "sensor_positions_tx": np.full((self.M, 3), self.position),
                "sensor_velocities_tx": np.full((self.M, 3), self.velocity),
                "range_times": np.repeat(self.range_times, self.M),
                "doppler_frequencies": np.repeat(self.doppler_frequency, self.M),
                "initial_guesses": self.initial_guess,
                "expected_shape": (self.M, 3),
            },
            {
                "name": "M-points rx_pos=2d rx_vel=2d init=2d doppler=array",
                "sensor_positions_rx": self.position.reshape(1, 3),
                "sensor_velocities_rx": self.velocity.reshape(1, 3),
                "sensor_positions_tx": np.full((self.M, 3), self.position),
                "sensor_velocities_tx": np.full((self.M, 3), self.velocity),
                "range_times": np.repeat(self.range_times, self.M),
                "doppler_frequencies": np.repeat(self.doppler_frequency, self.M),
                "initial_guesses": self.initial_guess.reshape(1, 3),
                "expected_shape": (self.M, 3),
            },
        ]

        for case in test_cases:
            with self.subTest(case=case["name"]):
                out = direct_geocoding_bistatic(
                    sensor_positions_rx=case["sensor_positions_rx"],
                    sensor_velocities_rx=case["sensor_velocities_rx"],
                    sensor_positions_tx=case["sensor_positions_tx"],
                    sensor_velocities_tx=case["sensor_velocities_tx"],
                    range_times=case["range_times"],
                    look_direction=self.look_direction,
                    altitude=self.altitude,
                    doppler_frequencies=case["doppler_frequencies"],
                    wavelength=self.wavelength,
                    initial_guesses=case["initial_guesses"],
                )

                if out.ndim == 2:
                    self._check_residuals(
                        case["sensor_positions_rx"],
                        case["sensor_velocities_rx"],
                        case["sensor_positions_tx"],
                        case["sensor_velocities_tx"],
                        case["range_times"],
                        out,
                    )
                self._check_output(case["expected_shape"], out)

    def test_direct_geocoding_bistatic_3_points(self) -> None:
        """Testing direct_geocoding_bistatic with 3-point cases (N x M)."""
        test_cases = [
            {
                "name": "3-points init=1d doppler=scalar",
                "sensor_positions_rx": np.full((self.N, 3), self.position),
                "sensor_velocities_rx": np.full((self.N, 3), self.velocity),
                "sensor_positions_tx": np.full((self.M, 3), self.position),
                "sensor_velocities_tx": np.full((self.M, 3), self.velocity),
                "range_times": np.repeat(self.range_times[0], self.M),
                "doppler_frequencies": self.doppler_frequency,
                "initial_guesses": self.initial_guess,
                "expected_shape": (self.N, self.M, 3),
            },
            {
                "name": "3-points init=2d doppler=scalar",
                "sensor_positions_rx": np.full((self.N, 3), self.position),
                "sensor_velocities_rx": np.full((self.N, 3), self.velocity),
                "sensor_positions_tx": np.full((self.M, 3), self.position),
                "sensor_velocities_tx": np.full((self.M, 3), self.velocity),
                "range_times": np.repeat(self.range_times[0], self.M),
                "doppler_frequencies": self.doppler_frequency,
                "initial_guesses": self.initial_guess.reshape(1, 3),
                "expected_shape": (self.N, self.M, 3),
            },
            {
                "name": "3-points init=2d doppler=array",
                "sensor_positions_rx": np.full((self.N, 3), self.position),
                "sensor_velocities_rx": np.full((self.N, 3), self.velocity),
                "sensor_positions_tx": np.full((self.M, 3), self.position),
                "sensor_velocities_tx": np.full((self.M, 3), self.velocity),
                "range_times": np.repeat(self.range_times[0], self.M),
                "doppler_frequencies": np.repeat(self.doppler_frequency, self.M),
                "initial_guesses": self.initial_guess.reshape(1, 3),
                "expected_shape": (self.N, self.M, 3),
            },
            {
                "name": "3-points init=3d doppler=scalar",
                "sensor_positions_rx": np.full((self.N, 3), self.position),
                "sensor_velocities_rx": np.full((self.N, 3), self.velocity),
                "sensor_positions_tx": np.full((self.M, 3), self.position),
                "sensor_velocities_tx": np.full((self.M, 3), self.velocity),
                "range_times": np.repeat(self.range_times[0], self.M),
                "doppler_frequencies": self.doppler_frequency,
                "initial_guesses": np.full((self.N, 3), self.initial_guess),
                "expected_shape": (self.N, self.M, 3),
            },
            {
                "name": "3-points init=3d doppler=array",
                "sensor_positions_rx": np.full((self.N, 3), self.position),
                "sensor_velocities_rx": np.full((self.N, 3), self.velocity),
                "sensor_positions_tx": np.full((self.M, 3), self.position),
                "sensor_velocities_tx": np.full((self.M, 3), self.velocity),
                "range_times": np.repeat(self.range_times[0], self.M),
                "doppler_frequencies": np.repeat(self.doppler_frequency, self.M),
                "initial_guesses": np.full((self.N, 3), self.initial_guess),
                "expected_shape": (self.N, self.M, 3),
            },
        ]

        for case in test_cases:
            with self.subTest(case=case["name"]):
                out = direct_geocoding_bistatic(
                    sensor_positions_rx=case["sensor_positions_rx"],
                    sensor_velocities_rx=case["sensor_velocities_rx"],
                    sensor_positions_tx=case["sensor_positions_tx"],
                    sensor_velocities_tx=case["sensor_velocities_tx"],
                    range_times=case["range_times"],
                    look_direction=self.look_direction,
                    altitude=self.altitude,
                    doppler_frequencies=case["doppler_frequencies"],
                    wavelength=self.wavelength,
                    initial_guesses=case["initial_guesses"],
                )

                if out.ndim == 3:
                    for range_index in range(self.M):
                        self._check_residuals(
                            case["sensor_positions_rx"],
                            case["sensor_velocities_rx"],
                            case["sensor_positions_tx"][range_index, :],
                            case["sensor_velocities_tx"][range_index, :],
                            case["range_times"][range_index]
                            if np.size(case["range_times"]) > 1
                            else case["range_times"],
                            out[:, range_index, :],
                        )
                else:
                    self._check_residuals(
                        case["sensor_positions_rx"],
                        case["sensor_velocities_rx"],
                        case["sensor_positions_tx"],
                        case["sensor_velocities_tx"],
                        case["range_times"],
                        out,
                    )

                self._check_output(case["expected_shape"], out)


class DirectGeocodingBistaticCoreTest(_DirectGeocodingBistaticBase):
    """Testing direct geocoding bistatic core with consolidated subtests."""

    def setUp(self) -> None:
        super().setUp()
        self.doppler_freqs = self.doppler_frequency
        self.geodetic_altitude = self.altitude

    def test_direct_geocoding_bistatic_core_cases(self) -> None:
        """Testing direct_geocoding_bistatic_core success cases."""
        test_cases = [
            {
                "name": "case0a: 1 pos (3,), 1 vel (3,), 1 rng time, 1 initial guess (3,)",
                "sensor_positions_rx": self.position,
                "sensor_velocities_rx": self.velocity,
                "sensor_positions_tx": self.position,
                "sensor_velocities_tx": self.velocity,
                "range_times": self.range_times[0],
                "doppler_frequencies": self.doppler_freqs,
                "initial_guesses": self.initial_guess,
                "expected_shape": (3,),
            },
            {
                "name": "case0b: 1 pos (1, 3), 1 vel (1, 3), 1 rng time, 1 initial guess (1, 3)",
                "sensor_positions_rx": self.position.reshape(1, 3),
                "sensor_velocities_rx": self.velocity.reshape(1, 3),
                "sensor_positions_tx": self.position.reshape(1, 3),
                "sensor_velocities_tx": self.velocity.reshape(1, 3),
                "range_times": self.range_times[0],
                "doppler_frequencies": self.doppler_freqs,
                "initial_guesses": self.initial_guess.reshape(1, 3),
                "expected_shape": (1, 3),
            },
            {
                "name": "case0c: 1 pos (3,) + (1, 3), 1 vel (3,) + (1, 3), 1 rng time, 1 initial guess (3,)",
                "sensor_positions_rx": self.position,
                "sensor_velocities_rx": self.velocity,
                "sensor_positions_tx": self.position.reshape(1, 3),
                "sensor_velocities_tx": self.velocity.reshape(1, 3),
                "range_times": self.range_times[0],
                "doppler_frequencies": self.doppler_freqs,
                "initial_guesses": self.initial_guess,
                "expected_shape": (1, 3),
            },
            {
                "name": "case1a: N pos (N, 3), N vel (N, 3), 1 rng time, 1 initial guess (3,)",
                "sensor_positions_rx": np.full((self.N, 3), self.position),
                "sensor_velocities_rx": np.full((self.N, 3), self.velocity),
                "sensor_positions_tx": self.position,
                "sensor_velocities_tx": self.velocity,
                "range_times": self.range_times[0],
                "doppler_frequencies": self.doppler_freqs,
                "initial_guesses": self.initial_guess,
                "expected_shape": (self.N, 3),
            },
            {
                "name": "case1b: N pos (N, 3), N vel (N, 3), 1 rng time, N initial guess (N, 3)",
                "sensor_positions_rx": np.full((self.N, 3), self.position),
                "sensor_velocities_rx": np.full((self.N, 3), self.velocity),
                "sensor_positions_tx": self.position,
                "sensor_velocities_tx": self.velocity,
                "range_times": self.range_times[0],
                "doppler_frequencies": self.doppler_freqs,
                "initial_guesses": np.full((self.N, 3), self.initial_guess),
                "expected_shape": (self.N, 3),
            },
            {
                "name": "case1c: N pos (N, 3), N vel (N, 3), 1 rng time, 1 initial guess (1, 3)",
                "sensor_positions_rx": np.full((self.N, 3), self.position),
                "sensor_velocities_rx": np.full((self.N, 3), self.velocity),
                "sensor_positions_tx": self.position.reshape(1, 3),
                "sensor_velocities_tx": self.velocity.reshape(1, 3),
                "range_times": self.range_times[0],
                "doppler_frequencies": self.doppler_freqs,
                "initial_guesses": self.initial_guess.reshape(1, 3),
                "expected_shape": (self.N, 3),
            },
            {
                "name": "case2a: 1 pos (3,), 1 vel (3,), M rng times, 1 initial guess (3,)",
                "sensor_positions_rx": self.position,
                "sensor_velocities_rx": self.velocity,
                "sensor_positions_tx": np.full((self.M, 3), self.position),
                "sensor_velocities_tx": np.full((self.M, 3), self.velocity),
                "range_times": np.repeat(self.range_times[0], self.M),
                "doppler_frequencies": self.doppler_freqs,
                "initial_guesses": self.initial_guess,
                "expected_shape": (self.M, 3),
            },
            {
                "name": "case2b: 1 pos (1, 3), 1 vel (1, 3), M rng times, 1 initial guess (1, 3)",
                "sensor_positions_rx": self.position.reshape(1, 3),
                "sensor_velocities_rx": self.velocity.reshape(1, 3),
                "sensor_positions_tx": np.full((self.M, 3), self.position),
                "sensor_velocities_tx": np.full((self.M, 3), self.velocity),
                "range_times": np.repeat(self.range_times[0], self.M),
                "doppler_frequencies": self.doppler_freqs,
                "initial_guesses": self.initial_guess.reshape(1, 3),
                "expected_shape": (self.M, 3),
            },
            {
                "name": "case2c: 1 pos (3,), 1 vel (3,), M rng times, M doppler freq",
                "sensor_positions_rx": self.position,
                "sensor_velocities_rx": self.velocity,
                "sensor_positions_tx": np.full((self.M, 3), self.position),
                "sensor_velocities_tx": np.full((self.M, 3), self.velocity),
                "range_times": np.repeat(self.range_times[0], self.M),
                "doppler_frequencies": np.repeat(self.doppler_freqs, self.M),
                "initial_guesses": self.initial_guess,
                "expected_shape": (self.M, 3),
            },
            {
                "name": "case3a: N pos (N, 3), N vel (N, 3), M rng times, 1 initial guess (3,)",
                "sensor_positions_rx": np.full((self.N, 3), self.position),
                "sensor_velocities_rx": np.full((self.N, 3), self.velocity),
                "sensor_positions_tx": np.full((self.M, 3), self.position),
                "sensor_velocities_tx": np.full((self.M, 3), self.velocity),
                "range_times": np.repeat(self.range_times[0], self.M),
                "doppler_frequencies": self.doppler_freqs,
                "initial_guesses": self.initial_guess,
                "expected_shape": (self.N, self.M, 3),
            },
            {
                "name": "case3b: N pos (N, 3), N vel (N, 3), M rng times, M doppler freq, 1 init (1, 3)",
                "sensor_positions_rx": np.full((self.N, 3), self.position),
                "sensor_velocities_rx": np.full((self.N, 3), self.velocity),
                "sensor_positions_tx": np.full((self.M, 3), self.position),
                "sensor_velocities_tx": np.full((self.M, 3), self.velocity),
                "range_times": np.repeat(self.range_times[0], self.M),
                "doppler_frequencies": np.repeat(self.doppler_freqs, self.M),
                "initial_guesses": self.initial_guess.reshape(1, 3),
                "expected_shape": (self.N, self.M, 3),
            },
            {
                "name": "case3c: N pos (N, 3), N vel (N, 3), M rng times, 1 doppler, N init (N, 3)",
                "sensor_positions_rx": np.full((self.N, 3), self.position),
                "sensor_velocities_rx": np.full((self.N, 3), self.velocity),
                "sensor_positions_tx": np.full((self.M, 3), self.position),
                "sensor_velocities_tx": np.full((self.M, 3), self.velocity),
                "range_times": np.repeat(self.range_times[0], self.M),
                "doppler_frequencies": self.doppler_freqs,
                "initial_guesses": np.full((self.N, 3), self.initial_guess),
                "expected_shape": (self.N, self.M, 3),
            },
            {
                "name": "case4: N pos (N, 3), N vel (N, 3), M rng times, M doppler freq, N init (N, 3)",
                "sensor_positions_rx": np.full((self.N, 3), self.position),
                "sensor_velocities_rx": np.full((self.N, 3), self.velocity),
                "sensor_positions_tx": np.full((self.M, 3), self.position),
                "sensor_velocities_tx": np.full((self.M, 3), self.velocity),
                "range_times": np.repeat(self.range_times[0], self.M),
                "doppler_frequencies": np.repeat(self.doppler_freqs, self.M),
                "initial_guesses": np.full((self.N, 3), self.initial_guess),
                "expected_shape": (self.N, self.M, 3),
            },
        ]

        for case in test_cases:
            with self.subTest(case=case["name"]):
                out = direct_geocoding_bistatic_core(
                    sensor_positions_rx=case["sensor_positions_rx"],
                    sensor_velocities_rx=case["sensor_velocities_rx"],
                    initial_guesses=case["initial_guesses"],
                    sensor_positions_tx=case["sensor_positions_tx"],
                    sensor_velocities_tx=case["sensor_velocities_tx"],
                    range_times=case["range_times"],
                    altitude=self.geodetic_altitude,
                    doppler_frequencies=case["doppler_frequencies"],
                    wavelength=self.wavelength,
                )

                self.assertEqual(out.shape, case["expected_shape"])
                np.testing.assert_allclose(
                    out, self._expected_output(case["expected_shape"]), atol=self.tolerance, rtol=0
                )

    def test_direct_geocoding_bistatic_core_error_cases(self) -> None:
        """Testing direct_geocoding_bistatic_core error cases."""
        error_cases = [
            {
                "name": "case5: N doppler (N,) with M doppler (M,) - size mismatch",
                "sensor_positions_rx": np.full((self.N, 3), self.position),
                "sensor_velocities_rx": np.full((self.N, 3), self.velocity),
                "initial_guesses": np.full((self.N, 3), self.initial_guess),
                "sensor_positions_tx": np.full((self.M, 3), self.position),
                "sensor_velocities_tx": np.full((self.M, 3), self.velocity),
                "range_times": np.repeat(self.range_times[0], self.M),
                "doppler_frequencies": np.repeat(self.doppler_freqs, self.N),
            },
        ]

        for case in error_cases:
            with self.subTest(case=case["name"]):
                with self.assertRaises(RuntimeError):
                    direct_geocoding_bistatic_core(
                        sensor_positions_rx=case["sensor_positions_rx"],
                        sensor_velocities_rx=case["sensor_velocities_rx"],
                        initial_guesses=case["initial_guesses"],
                        sensor_positions_tx=case["sensor_positions_tx"],
                        sensor_velocities_tx=case["sensor_velocities_tx"],
                        range_times=case["range_times"],
                        altitude=self.geodetic_altitude,
                        doppler_frequencies=case["doppler_frequencies"],
                        wavelength=self.wavelength,
                    )


class NewtonForDirectGeocodingBistaticTest(_DirectGeocodingBistaticBase):
    """Testing Newton for direct geocoding bistatic core with consolidated subtests."""

    def setUp(self) -> None:
        super().setUp()
        self.range_time = self.range_times[0]
        self.doppler_freqs = self.doppler_frequency
        self.geodetic_altitude = self.altitude

    def test_newton_for_direct_geocoding_bistatic_cases(self) -> None:
        """Testing _direct_geocoding_bistatic_newton cases."""
        test_cases = [
            {
                "name": "case0: 1 pos (3,), 1 vel (3,), 1 init guess (3,)",
                "sensor_positions_rx": self.position,
                "sensor_velocities_rx": self.velocity,
                "initial_guesses": self.initial_guess,
                "expected_shape": (3,),
            },
            {
                "name": "case1: 1 pos (1, 3), 1 vel (1, 3), 1 init guess (1, 3)",
                "sensor_positions_rx": self.position.reshape(1, 3),
                "sensor_velocities_rx": self.velocity.reshape(1, 3),
                "initial_guesses": self.initial_guess.reshape(1, 3),
                "expected_shape": (1, 3),
            },
            {
                "name": "case2a: N pos (N, 3), N vel (N, 3), 1 init guess (3,)",
                "sensor_positions_rx": np.full((self.N, 3), self.position),
                "sensor_velocities_rx": np.full((self.N, 3), self.velocity),
                "initial_guesses": self.initial_guess,
                "expected_shape": (self.N, 3),
            },
            {
                "name": "case2b: N pos (N, 3), N vel (N, 3), 1 init guess (1, 3)",
                "sensor_positions_rx": np.full((self.N, 3), self.position),
                "sensor_velocities_rx": np.full((self.N, 3), self.velocity),
                "initial_guesses": self.initial_guess.reshape(1, 3),
                "expected_shape": (self.N, 3),
            },
            {
                "name": "case3: N pos (N, 3), N vel (N, 3), 1 init guess (1, 3)",
                "sensor_positions_rx": np.full((self.N, 3), self.position),
                "sensor_velocities_rx": np.full((self.N, 3), self.velocity),
                "initial_guesses": self.initial_guess.reshape(1, 3),
                "expected_shape": (self.N, 3),
            },
        ]

        for case in test_cases:
            with self.subTest(case=case["name"]):
                out = _direct_geocoding_bistatic_newton(
                    sensor_positions_rx=case["sensor_positions_rx"],
                    sensor_velocities_rx=case["sensor_velocities_rx"],
                    initial_guesses=case["initial_guesses"],
                    sensor_position_tx=self.position,
                    sensor_velocity_tx=self.velocity,
                    range_time=self.range_time,
                    doppler_frequency=self.doppler_freqs,
                    wavelength=self.wavelength,
                    altitude=self.geodetic_altitude,
                )

                self.assertEqual(out.shape, case["expected_shape"])
                np.testing.assert_allclose(
                    out, self._expected_output(case["expected_shape"]), atol=self.tolerance, rtol=0
                )


if __name__ == "__main__":
    unittest.main()
