# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for geometry/geocoding/direct_geocoding.py and direct_geocoding_core.py monostatic functionalities"""

from __future__ import annotations

import unittest

import numpy as np
from scipy.constants import speed_of_light

from perseo_core.geometry.geocoding.direct_geocoding import direct_geocoding_init, direct_geocoding_monostatic
from perseo_core.geometry.geocoding.direct_geocoding_core import (
    _direct_geocoding_monostatic_newton,
    _doppler_equation,
    _ellipse_equation,
    direct_geocoding_monostatic_core,
    direct_geocoding_monostatic_core_range_vectorized,
)
from perseo_core.geometry.utilities.ellipsoid import WGS84


def _doppler_equation_residual(
    sensor_pos: np.ndarray,
    sensor_vel: np.ndarray,
    ground_points: np.ndarray,
    wavelength: float,
    doppler_freq: float,
) -> np.ndarray:
    """Evaluating doppler equation residual.

    Parameters
    ----------
    sensor_pos : np.ndarray
        sensor position, (3,) or (N, 3)
    sensor_vel : np.ndarray
        sensor velocity, (3,) or (N, 3)
    ground_points : np.ndarray
        ground points from direct geocoding solution, (3,) or (N, 3)
    wavelength : float
        carrier signal wavelength
    doppler_freq : float
        doppler frequency

    Returns
    -------
    np.ndarray
        doppler equation residual
    """

    los = sensor_pos - ground_points
    los_vel_prod = np.sum(sensor_vel * los, axis=-1)
    distance = np.sqrt(np.sum(los * los, axis=-1))

    doppler_residual, _ = _doppler_equation(
        pv_scalar=los_vel_prod.ravel(),
        los=los.reshape(-1, 3),
        sensor_velocity=sensor_vel,
        distance=distance.reshape(-1),
        wavelength=wavelength,
        doppler_frequency=doppler_freq,
    )
    return np.array(doppler_residual)


def _range_equation_residual(sensor_pos: np.ndarray, ground_points: np.ndarray, range_time: float) -> np.ndarray:
    """Evaluating range equation residual.

    Parameters
    ----------
    sensor_pos : np.ndarray
        sensor position, (3,) or (N, 3)
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
    los = sensor_pos - ground_points
    distance = np.sqrt(np.sum(los * los, axis=-1))

    rng_residual = (distance - rng_dst) / speed_of_light

    return np.array(rng_residual)


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


class DirectGeocodingMonostaticTest(unittest.TestCase):
    """Test direct_geocoding_monostatic with various input dimension combinations using subtests."""

    def setUp(self):
        self.positions = np.array(
            [4387348.749948771, 762123.3489877012, 4553067.931912004],
        )
        self.scaled_arf_velocities = np.array(
            [-856.1384108174528, -329.7629775067583, 398.55830806407346],
        )
        self.initial_guesses = np.array([4385932.628762595, 764443.4718341012, 4551945.624046889])
        self.range_times = np.array([2.05624579e-05])
        self.doppler_freqs = 0
        self.geodetic_altitude = 0
        self.look_direction = "RIGHT"
        self.wavelength = 1
        self.N = 4
        self.M = 5
        self.Q = 120

        self.tolerance = 1e-6
        self.residual_tolerance = 1e-8

        self.results = np.array([4385882.195057692, 764600.9869913795, 4551967.6143934])

    def test_direct_geocoding_monostatic_cases(self) -> None:
        """Test direct_geocoding_monostatic with all input dimension combinations using subtests."""

        test_cases = [
            {
                "name": "case0a: 1 pos (3,), 1 vel (3,), 1 rng time, 1 initial guess (3,)",
                "positions": self.positions,
                "velocities": self.scaled_arf_velocities,
                "initial_guesses": self.initial_guesses,
                "range_times": self.range_times[0],
                "doppler_frequencies": self.doppler_freqs,
                "expected_ndim": 1,
                "expected_shape": (3,),
                "check_residuals": True,
            },
            {
                "name": "case0b: 1 pos (1,3), 1 vel (1,3), 1 rng time, 1 initial guess (1,3)",
                "positions": self.positions.reshape(1, 3),
                "velocities": self.scaled_arf_velocities.reshape(1, 3),
                "initial_guesses": self.initial_guesses.reshape(1, 3),
                "range_times": self.range_times[0],
                "doppler_frequencies": self.doppler_freqs,
                "expected_ndim": 2,
                "expected_shape": (1, 3),
                "check_residuals": True,
            },
            {
                "name": "case0c: 1 pos (3,), 1 vel (3,), 1 rng time, no initial guess",
                "positions": self.positions,
                "velocities": self.scaled_arf_velocities,
                "initial_guesses": None,
                "range_times": self.range_times[0],
                "doppler_frequencies": self.doppler_freqs,
                "expected_ndim": 1,
                "expected_shape": (3,),
                "check_residuals": True,
            },
            {
                "name": "case0d: 1 pos (1,3), 1 vel (1,3), 1 rng time, no initial guess",
                "positions": self.positions.reshape(1, 3),
                "velocities": self.scaled_arf_velocities.reshape(1, 3),
                "initial_guesses": None,
                "range_times": self.range_times[0],
                "doppler_frequencies": self.doppler_freqs,
                "expected_ndim": 2,
                "expected_shape": (1, 3),
                "check_residuals": True,
            },
            {
                "name": "case1a: N pos (N,3), N vel (N,3), 1 rng time, 1 initial guess (3,)",
                "positions": np.full((self.N, 3), self.positions),
                "velocities": np.full((self.N, 3), self.scaled_arf_velocities),
                "initial_guesses": self.initial_guesses,
                "range_times": self.range_times[0],
                "doppler_frequencies": self.doppler_freqs,
                "expected_ndim": 2,
                "expected_shape": (self.N, 3),
                "check_residuals": True,
            },
            {
                "name": "case1b: N pos (N,3), N vel (N,3), 1 rng time, 1 initial guess (1,3)",
                "positions": np.full((self.N, 3), self.positions),
                "velocities": np.full((self.N, 3), self.scaled_arf_velocities),
                "initial_guesses": self.initial_guesses.reshape(1, 3),
                "range_times": self.range_times[0],
                "doppler_frequencies": self.doppler_freqs,
                "expected_ndim": 2,
                "expected_shape": (self.N, 3),
                "check_residuals": True,
            },
            {
                "name": "case1c: N pos (N,3), N vel (N,3), 1 rng time, N initial guesses (N,3)",
                "positions": np.full((self.N, 3), self.positions),
                "velocities": np.full((self.N, 3), self.scaled_arf_velocities),
                "initial_guesses": np.full((self.N, 3), self.initial_guesses),
                "range_times": self.range_times[0],
                "doppler_frequencies": self.doppler_freqs,
                "expected_ndim": 2,
                "expected_shape": (self.N, 3),
                "check_residuals": True,
            },
            {
                "name": "case1d: N pos (N,3), N vel (N,3), 1 rng time, no initial guess",
                "positions": np.full((self.N, 3), self.positions),
                "velocities": np.full((self.N, 3), self.scaled_arf_velocities),
                "initial_guesses": None,
                "range_times": self.range_times[0],
                "doppler_frequencies": self.doppler_freqs,
                "expected_ndim": 2,
                "expected_shape": (self.N, 3),
                "check_residuals": True,
            },
            {
                "name": "case2a: 1 pos (3,), 1 vel (3,), M rng times (M,), 1 initial guess (3,)",
                "positions": self.positions,
                "velocities": self.scaled_arf_velocities,
                "initial_guesses": self.initial_guesses,
                "range_times": np.repeat(self.range_times[0], self.M),
                "doppler_frequencies": self.doppler_freqs,
                "expected_ndim": 2,
                "expected_shape": (self.M, 3),
                "check_residuals": True,
            },
            {
                "name": "case2b: 1 pos (1,3), 1 vel (1,3), M rng times (M,), 1 initial guess (1,3)",
                "positions": self.positions.reshape(1, 3),
                "velocities": self.scaled_arf_velocities.reshape(1, 3),
                "initial_guesses": self.initial_guesses.reshape(1, 3),
                "range_times": np.repeat(self.range_times[0], self.M),
                "doppler_frequencies": self.doppler_freqs,
                "expected_ndim": 2,
                "expected_shape": (self.M, 3),
                "check_residuals": True,
            },
            {
                "name": "case2c: 1 pos (3,), 1 vel (3,), M rng times (M,), no initial guess",
                "positions": self.positions,
                "velocities": self.scaled_arf_velocities,
                "initial_guesses": None,
                "range_times": np.repeat(self.range_times[0], self.M),
                "doppler_frequencies": self.doppler_freqs,
                "expected_ndim": 2,
                "expected_shape": (self.M, 3),
                "check_residuals": True,
            },
            {
                "name": "case2d: 1 pos (3,), 1 vel (3,), M rng times (M,), 1 initial guess (3,), M doppler freqs",
                "positions": self.positions,
                "velocities": self.scaled_arf_velocities,
                "initial_guesses": self.initial_guesses,
                "range_times": np.repeat(self.range_times[0], self.M),
                "doppler_frequencies": np.repeat(self.doppler_freqs, self.M),
                "expected_ndim": 2,
                "expected_shape": (self.M, 3),
                "check_residuals": True,
            },
            {
                "name": "case2e: 1 pos (1,3), 1 vel (1,3), M rng times (M,), 1 initial guess (1,3), M doppler freqs",
                "positions": self.positions.reshape(1, 3),
                "velocities": self.scaled_arf_velocities.reshape(1, 3),
                "initial_guesses": self.initial_guesses.reshape(1, 3),
                "range_times": np.repeat(self.range_times[0], self.M),
                "doppler_frequencies": np.repeat(self.doppler_freqs, self.M),
                "expected_ndim": 2,
                "expected_shape": (self.M, 3),
                "check_residuals": True,
            },
            {
                "name": "case2f: 1 pos (3,), 1 vel (3,), Q rng times (M,), 1 initial guess (3,)",
                "positions": self.positions,
                "velocities": self.scaled_arf_velocities,
                "initial_guesses": self.initial_guesses,
                "range_times": np.repeat(self.range_times[0], self.Q),
                "doppler_frequencies": self.doppler_freqs,
                "expected_ndim": 2,
                "expected_shape": (self.Q, 3),
                "check_residuals": True,
            },
            {
                "name": "case2g: 1 pos (1,3), 1 vel (1,3), Q rng times (Q,), 1 initial guess (1,3)",
                "positions": self.positions.reshape(1, 3),
                "velocities": self.scaled_arf_velocities.reshape(1, 3),
                "initial_guesses": self.initial_guesses.reshape(1, 3),
                "range_times": np.repeat(self.range_times[0], self.Q),
                "doppler_frequencies": self.doppler_freqs,
                "expected_ndim": 2,
                "expected_shape": (self.Q, 3),
                "check_residuals": True,
            },
            {
                "name": "case2h: 1 pos (3,), 1 vel (3,), Q rng times (Q,), no initial guess",
                "positions": self.positions,
                "velocities": self.scaled_arf_velocities,
                "initial_guesses": None,
                "range_times": np.repeat(self.range_times[0], self.Q),
                "doppler_frequencies": self.doppler_freqs,
                "expected_ndim": 2,
                "expected_shape": (self.Q, 3),
                "check_residuals": True,
            },
            {
                "name": "case2i: 1 pos (3,), 1 vel (3,), Q rng times (Q,), 1 initial guess (3,), Q doppler freqs",
                "positions": self.positions,
                "velocities": self.scaled_arf_velocities,
                "initial_guesses": self.initial_guesses,
                "range_times": np.repeat(self.range_times[0], self.Q),
                "doppler_frequencies": np.repeat(self.doppler_freqs, self.Q),
                "expected_ndim": 2,
                "expected_shape": (self.Q, 3),
                "check_residuals": True,
            },
            {
                "name": "case2j: 1 pos (1,3), 1 vel (1,3), Q rng times (Q,), 1 initial guess (1,3), Q doppler freqs",
                "positions": self.positions.reshape(1, 3),
                "velocities": self.scaled_arf_velocities.reshape(1, 3),
                "initial_guesses": self.initial_guesses.reshape(1, 3),
                "range_times": np.repeat(self.range_times[0], self.Q),
                "doppler_frequencies": np.repeat(self.doppler_freqs, self.Q),
                "expected_ndim": 2,
                "expected_shape": (self.Q, 3),
                "check_residuals": True,
            },
            {
                "name": "case2k: 1 pos (1,3), 1 vel (1,3), Q rng times (Q,), Q initial guesses (Q,3), Q doppler freqs",
                "positions": self.positions.reshape(1, 3),
                "velocities": self.scaled_arf_velocities.reshape(1, 3),
                "initial_guesses": np.full((self.Q, 3), self.initial_guesses),
                "range_times": np.repeat(self.range_times[0], self.Q),
                "doppler_frequencies": np.repeat(self.doppler_freqs, self.Q),
                "expected_ndim": 2,
                "expected_shape": (self.Q, 3),
                "check_residuals": True,
            },
            {
                "name": "case3a: N pos (N,3), N vel (N,3), M rng times (M,), 1 initial guess (3,)",
                "positions": np.full((self.N, 3), self.positions),
                "velocities": np.full((self.N, 3), self.scaled_arf_velocities),
                "initial_guesses": self.initial_guesses,
                "range_times": np.repeat(self.range_times[0], self.M),
                "doppler_frequencies": self.doppler_freqs,
                "expected_ndim": 3,
                "expected_shape": (self.N, self.M, 3),
                "check_residuals": True,
            },
            {
                "name": "case3b: N pos (N,3), N vel (N,3), M rng times (M,), no initial guess",
                "positions": np.full((self.N, 3), self.positions),
                "velocities": np.full((self.N, 3), self.scaled_arf_velocities),
                "initial_guesses": None,
                "range_times": np.repeat(self.range_times[0], self.M),
                "doppler_frequencies": self.doppler_freqs,
                "expected_ndim": 3,
                "expected_shape": (self.N, self.M, 3),
                "check_residuals": True,
            },
            {
                "name": "case3c: N pos (N,3), N vel (N,3), M rng times (M,), 1 initial guess (3,), M doppler freqs",
                "positions": np.full((self.N, 3), self.positions),
                "velocities": np.full((self.N, 3), self.scaled_arf_velocities),
                "initial_guesses": self.initial_guesses,
                "range_times": np.repeat(self.range_times[0], self.M),
                "doppler_frequencies": np.repeat(self.doppler_freqs, self.M),
                "expected_ndim": 3,
                "expected_shape": (self.N, self.M, 3),
                "check_residuals": True,
            },
            {
                "name": "case3d: N pos (N,3), N vel (N,3), Q rng times (Q,), 1 initial guess (3,)",
                "positions": np.full((self.N, 3), self.positions),
                "velocities": np.full((self.N, 3), self.scaled_arf_velocities),
                "initial_guesses": self.initial_guesses,
                "range_times": np.repeat(self.range_times[0], self.Q),
                "doppler_frequencies": self.doppler_freqs,
                "expected_ndim": 3,
                "expected_shape": (self.N, self.Q, 3),
                "check_residuals": True,
            },
            {
                "name": "case3e: N pos (N,3), N vel (N,3), Q rng times (Q,), no initial guess",
                "positions": np.full((self.N, 3), self.positions),
                "velocities": np.full((self.N, 3), self.scaled_arf_velocities),
                "initial_guesses": None,
                "range_times": np.repeat(self.range_times[0], self.Q),
                "doppler_frequencies": self.doppler_freqs,
                "expected_ndim": 3,
                "expected_shape": (self.N, self.Q, 3),
                "check_residuals": True,
            },
            {
                "name": "case3f: N pos (N,3), N vel (N,3), Q rng times (Q,), 1 initial guess (3,), Q doppler freqs",
                "positions": np.full((self.N, 3), self.positions),
                "velocities": np.full((self.N, 3), self.scaled_arf_velocities),
                "initial_guesses": self.initial_guesses,
                "range_times": np.repeat(self.range_times[0], self.Q),
                "doppler_frequencies": np.repeat(self.doppler_freqs, self.Q),
                "expected_ndim": 3,
                "expected_shape": (self.N, self.Q, 3),
                "check_residuals": True,
            },
            {
                "name": "case3g: N pos (N,3), N vel (N,3), Q rng times (Q,), Q initial guesses (Q,3)",
                "positions": np.full((self.N, 3), self.positions),
                "velocities": np.full((self.N, 3), self.scaled_arf_velocities),
                "initial_guesses": np.full((self.Q, 3), self.initial_guesses),
                "range_times": np.repeat(self.range_times[0], self.Q),
                "doppler_frequencies": np.repeat(self.doppler_freqs, self.Q),
                "expected_ndim": 3,
                "expected_shape": (self.N, self.Q, 3),
                "check_residuals": True,
            },
        ]

        for case in test_cases:
            with self.subTest(case=case["name"]):
                out = direct_geocoding_monostatic(
                    sensor_positions=case["positions"],
                    sensor_velocities=case["velocities"],
                    initial_guesses=case["initial_guesses"],
                    range_times=case["range_times"],
                    doppler_frequencies=case["doppler_frequencies"],
                    altitude=self.geodetic_altitude,
                    look_direction=self.look_direction,
                    wavelength=self.wavelength,
                )

                self.assertEqual(out.ndim, case["expected_ndim"])
                self.assertEqual(out.shape, case["expected_shape"])

                if case["check_residuals"]:
                    doppler_residual = _doppler_equation_residual(
                        sensor_pos=self.positions,
                        sensor_vel=self.scaled_arf_velocities,
                        ground_points=out,
                        doppler_freq=self.doppler_freqs,
                        wavelength=self.wavelength,
                    )
                    range_residual = _range_equation_residual(
                        sensor_pos=self.positions, ground_points=out, range_time=self.range_times[0]
                    )
                    ellipse_residual = _ellipse_equation_residual(ground_points=out)

                    np.testing.assert_allclose(
                        doppler_residual,
                        np.zeros_like(doppler_residual),
                        atol=self.residual_tolerance,
                        rtol=0,
                    )
                    np.testing.assert_allclose(
                        range_residual,
                        np.zeros_like(range_residual),
                        atol=self.residual_tolerance,
                        rtol=0,
                    )
                    np.testing.assert_allclose(
                        ellipse_residual,
                        np.zeros_like(ellipse_residual),
                        atol=self.residual_tolerance,
                        rtol=0,
                    )

    def test_direct_geocoding_monostatic_error_cases(self) -> None:
        """Test direct_geocoding_monostatic error handling for mismatched dimensions."""

        error_cases = [
            {
                "name": "case4: N pos (N,3), M vel (M,3), mismatch position/velocity",
                "positions": np.full((self.N, 3), self.positions),
                "velocities": np.full((self.M, 3), self.scaled_arf_velocities),
                "initial_guesses": np.full((self.N, 3), self.initial_guesses),
                "range_times": np.repeat(self.range_times[0], self.M),
                "doppler_frequencies": self.doppler_freqs,
            },
            {
                "name": "case5: N pos (N,3), M init guesses (M,3), mismatch position/init guesses",
                "positions": np.full((self.N, 3), self.positions),
                "velocities": np.full((self.N, 3), self.scaled_arf_velocities),
                "initial_guesses": np.full((self.N // 2, 3), self.initial_guesses),
                "range_times": np.repeat(self.range_times[0], self.M),
                "doppler_frequencies": self.doppler_freqs,
            },
            {
                "name": "case6: N range (N,), M freqs (M,), mismatch frequency/ranges",
                "positions": np.full((self.N, 3), self.positions),
                "velocities": np.full((self.N, 3), self.scaled_arf_velocities),
                "initial_guesses": np.full((self.N, 3), self.initial_guesses),
                "range_times": np.repeat(self.range_times[0], self.N),
                "doppler_frequencies": np.repeat(self.doppler_freqs, self.M),
            },
        ]

        for case in error_cases:
            with self.subTest(case=case["name"]):
                with self.assertRaises(RuntimeError):
                    direct_geocoding_monostatic(
                        sensor_positions=case["positions"],
                        sensor_velocities=case["velocities"],
                        initial_guesses=case["initial_guesses"],
                        range_times=case["range_times"],
                        doppler_frequencies=case["doppler_frequencies"],
                        altitude=self.geodetic_altitude,
                        look_direction=self.look_direction,
                        wavelength=self.wavelength,
                    )


class DirectGeocodingMonostaticCoreTest(unittest.TestCase):
    """Testing direct geocoding monostatic core with various input combinations using subtests."""

    def setUp(self):
        """Setting up variables for testing"""
        self.position = np.array(
            [4387348.749948771, 762123.3489877012, 4553067.931912004],
        )
        self.velocity = np.array(
            [-856.1384108174528, -329.7629775067583, 398.55830806407346],
        )
        self.initial_guess = np.array([4385882.165360568, 764600.914414172, 4551967.490551733])
        self.range_time = np.array([2.05624579e-05])
        self.doppler_freq = 0
        self.geodetic_altitude = 0
        self.wavelength = 1

        self.N = 4
        self.M = 5
        self.tolerance = 1e-6

        self.results = np.array([4385882.195057692, 764600.9869913795, 4551967.6143934])

    def test_monostatic_core_cases(self) -> None:
        """Test direct_geocoding_monostatic_core with various input combinations using subtests."""

        test_cases = [
            {
                "name": "case0a: 1 pos (3,), 1 vel (3,), 1 guess (3,), 1 rng time",
                "positions": self.position,
                "velocities": self.velocity,
                "initial_guesses": self.initial_guess,
                "range_times": self.range_time[0],
                "doppler_frequencies": self.doppler_freq,
                "expected_ndim": 1,
                "expected_shape": (3,),
            },
            {
                "name": "case0b: 1 pos (1,3), 1 vel (1,3), 1 guess (1,3), 1 rng time",
                "positions": self.position.reshape(1, 3),
                "velocities": self.velocity.reshape(1, 3),
                "initial_guesses": self.initial_guess.reshape(1, 3),
                "range_times": self.range_time[0],
                "doppler_frequencies": self.doppler_freq,
                "expected_ndim": 2,
                "expected_shape": (1, 3),
            },
            {
                "name": "case0c: 1 pos (3,), 1 vel (3,), 1 guess (3,), M range times",
                "positions": self.position,
                "velocities": self.velocity,
                "initial_guesses": self.initial_guess,
                "range_times": np.repeat(self.range_time[0], self.M),
                "doppler_frequencies": self.doppler_freq,
                "expected_ndim": 2,
                "expected_shape": (self.M, 3),
            },
            {
                "name": "case0d: 1 pos (1,3), 1 vel (1,3), 1 guess (1,3), M range times",
                "positions": self.position.reshape(1, 3),
                "velocities": self.velocity.reshape(1, 3),
                "initial_guesses": self.initial_guess.reshape(1, 3),
                "range_times": np.repeat(self.range_time[0], self.M),
                "doppler_frequencies": self.doppler_freq,
                "expected_ndim": 2,
                "expected_shape": (self.M, 3),
            },
            {
                "name": "case0e: 1 pos (3,), 1 vel (3,), 1 guess (3,), M range times, M doppler freqs",
                "positions": self.position,
                "velocities": self.velocity,
                "initial_guesses": self.initial_guess,
                "range_times": np.repeat(self.range_time[0], self.M),
                "doppler_frequencies": np.repeat(self.doppler_freq, self.M),
                "expected_ndim": 2,
                "expected_shape": (self.M, 3),
            },
            {
                "name": "case0f: 1 pos (1,3), 1 vel (1,3), 1 guess (1,3), M range times, M doppler freqs",
                "positions": self.position.reshape(1, 3),
                "velocities": self.velocity.reshape(1, 3),
                "initial_guesses": self.initial_guess.reshape(1, 3),
                "range_times": np.repeat(self.range_time[0], self.M),
                "doppler_frequencies": np.repeat(self.doppler_freq, self.M),
                "expected_ndim": 2,
                "expected_shape": (self.M, 3),
            },
            {
                "name": "case1a: N pos (N,3), N vel (N,3), N guess (N,3), 1 rng time",
                "positions": np.full((self.N, 3), self.position),
                "velocities": np.full((self.N, 3), self.velocity),
                "initial_guesses": np.full((self.N, 3), self.initial_guess),
                "range_times": self.range_time[0],
                "doppler_frequencies": self.doppler_freq,
                "expected_ndim": 2,
                "expected_shape": (self.N, 3),
            },
            {
                "name": "case1b: N pos (N,3), N vel (N,3), N guess (N,3), M rng times",
                "positions": np.full((self.N, 3), self.position),
                "velocities": np.full((self.N, 3), self.velocity),
                "initial_guesses": np.full((self.N, 3), self.initial_guess),
                "range_times": np.repeat(self.range_time[0], self.M),
                "doppler_frequencies": self.doppler_freq,
                "expected_ndim": 3,
                "expected_shape": (self.N, self.M, 3),
            },
            {
                "name": "case1c: N pos (N,3), N vel (N,3), N guess (N,3), M rng times, M doppler freqs",
                "positions": np.full((self.N, 3), self.position),
                "velocities": np.full((self.N, 3), self.velocity),
                "initial_guesses": np.full((self.N, 3), self.initial_guess),
                "range_times": np.repeat(self.range_time[0], self.M),
                "doppler_frequencies": np.repeat(self.doppler_freq, self.M),
                "expected_ndim": 3,
                "expected_shape": (self.N, self.M, 3),
            },
        ]

        for case in test_cases:
            with self.subTest(case=case["name"]):
                out = direct_geocoding_monostatic_core(
                    sensor_positions=case["positions"],
                    sensor_velocities=case["velocities"],
                    initial_guesses=case["initial_guesses"],
                    range_times=case["range_times"],
                    doppler_frequencies=case["doppler_frequencies"],
                    wavelength=self.wavelength,
                    altitude=self.geodetic_altitude,
                )

                self.assertEqual(out.ndim, case["expected_ndim"])
                self.assertEqual(out.shape, case["expected_shape"])

                # Determine expected result shape
                if case["expected_shape"] == (3,):
                    expected_result = self.results
                elif case["expected_shape"] == (1, 3):
                    expected_result = self.results.reshape(1, 3)
                elif case["expected_shape"] == (self.M, 3):
                    expected_result = np.full((self.M, 3), self.results)
                elif case["expected_shape"] == (self.N, 3):
                    expected_result = np.full((self.N, 3), self.results)
                elif case["expected_shape"] == (self.N, self.M, 3):
                    expected_result = np.full((self.N, self.M, 3), self.results)
                else:
                    expected_result = self.results

                np.testing.assert_allclose(out, expected_result, atol=self.tolerance, rtol=0)

    def test_monostatic_core_error_case(self) -> None:
        """Test direct_geocoding_monostatic_core error handling for mismatched dimensions."""

        # case: N range (M,), N doppler freqs (N,), mismatched frequencies/ranges
        with self.assertRaises(RuntimeError):
            direct_geocoding_monostatic_core(
                sensor_positions=np.full((self.N, 3), self.position),
                sensor_velocities=np.full((self.N, 3), self.velocity),
                initial_guesses=np.full((self.N, 3), self.initial_guess),
                range_times=np.repeat(self.range_time[0], self.M),
                doppler_frequencies=np.repeat(self.doppler_freq, self.N),
                wavelength=self.wavelength,
                altitude=self.geodetic_altitude,
            )


class DirectGeocodingRangeVectorizedMonostaticCoreTest(unittest.TestCase):
    """Testing direct geocoding monostatic range vectorized core with various input combinations using subtests."""

    def setUp(self):
        """Setting up variables for testing"""
        self.position = np.array(
            [4387348.749948771, 762123.3489877012, 4553067.931912004],
        )
        self.velocity = np.array(
            [-856.1384108174528, -329.7629775067583, 398.55830806407346],
        )
        self.initial_guess = np.array([4385882.165360568, 764600.914414172, 4551967.490551733])
        self.range_time = np.array([2.05624579e-05])
        self.doppler_freq = 0
        self.geodetic_altitude = 0
        self.wavelength = 1

        self.N = 4
        self.M = 100
        self.tolerance = 1e-6

        self.results = np.array([4385882.195057692, 764600.9869913795, 4551967.6143934])

    def test_monostatic_core_cases(self) -> None:
        """Test direct_geocoding_monostatic_core_range_vectorized with various input combinations using subtests."""

        test_cases = [
            {
                "name": "case0a: 1 pos (3,), 1 vel (3,), 1 guess (3,), 1 rng time",
                "positions": self.position,
                "velocities": self.velocity,
                "initial_guesses": self.initial_guess,
                "range_times": self.range_time[0],
                "doppler_frequencies": self.doppler_freq,
                "expected_ndim": 1,
                "expected_shape": (3,),
            },
            {
                "name": "case0b: 1 pos (1,3), 1 vel (1,3), 1 guess (1,3), 1 rng time",
                "positions": self.position.reshape(1, 3),
                "velocities": self.velocity.reshape(1, 3),
                "initial_guesses": self.initial_guess.reshape(1, 3),
                "range_times": self.range_time[0],
                "doppler_frequencies": self.doppler_freq,
                "expected_ndim": 2,
                "expected_shape": (1, 3),
            },
            {
                "name": "case0c: 1 pos (3,), 1 vel (3,), 1 guess (3,), M range times",
                "positions": self.position,
                "velocities": self.velocity,
                "initial_guesses": self.initial_guess,
                "range_times": np.repeat(self.range_time[0], self.M),
                "doppler_frequencies": self.doppler_freq,
                "expected_ndim": 2,
                "expected_shape": (self.M, 3),
            },
            {
                "name": "case0d: 1 pos (1,3), 1 vel (1,3), 1 guess (1,3), M range times",
                "positions": self.position.reshape(1, 3),
                "velocities": self.velocity.reshape(1, 3),
                "initial_guesses": self.initial_guess.reshape(1, 3),
                "range_times": np.repeat(self.range_time[0], self.M),
                "doppler_frequencies": self.doppler_freq,
                "expected_ndim": 2,
                "expected_shape": (self.M, 3),
            },
            {
                "name": "case0e: 1 pos (1,3), 1 vel (1,3), M guess (M,3), M range times",
                "positions": self.position.reshape(1, 3),
                "velocities": self.velocity.reshape(1, 3),
                "initial_guesses": np.full((self.M, 3), self.initial_guess),
                "range_times": np.repeat(self.range_time[0], self.M),
                "doppler_frequencies": self.doppler_freq,
                "expected_ndim": 2,
                "expected_shape": (self.M, 3),
            },
            {
                "name": "case0f: 1 pos (3,), 1 vel (3,), 1 guess (3,), M range times, M doppler freqs",
                "positions": self.position,
                "velocities": self.velocity,
                "initial_guesses": self.initial_guess,
                "range_times": np.repeat(self.range_time[0], self.M),
                "doppler_frequencies": np.repeat(self.doppler_freq, self.M),
                "expected_ndim": 2,
                "expected_shape": (self.M, 3),
            },
            {
                "name": "case0g: 1 pos (1,3), 1 vel (1,3), 1 guess (1,3), M range times, M doppler freqs",
                "positions": self.position.reshape(1, 3),
                "velocities": self.velocity.reshape(1, 3),
                "initial_guesses": self.initial_guess.reshape(1, 3),
                "range_times": np.repeat(self.range_time[0], self.M),
                "doppler_frequencies": np.repeat(self.doppler_freq, self.M),
                "expected_ndim": 2,
                "expected_shape": (self.M, 3),
            },
            {
                "name": "case0h: 1 pos (1,3), 1 vel (1,3), M guess (M,3), M range times, M doppler freqs",
                "positions": self.position.reshape(1, 3),
                "velocities": self.velocity.reshape(1, 3),
                "initial_guesses": np.full((self.M, 3), self.initial_guess),
                "range_times": np.repeat(self.range_time[0], self.M),
                "doppler_frequencies": np.repeat(self.doppler_freq, self.M),
                "expected_ndim": 2,
                "expected_shape": (self.M, 3),
            },
            {
                "name": "case1a: N pos (N,3), N vel (N,3), N guess (N,3), 1 rng time",
                "positions": np.full((self.N, 3), self.position),
                "velocities": np.full((self.N, 3), self.velocity),
                "initial_guesses": np.full((self.N, 3), self.initial_guess),
                "range_times": self.range_time[0],
                "doppler_frequencies": self.doppler_freq,
                "expected_ndim": 2,
                "expected_shape": (self.N, 3),
            },
            {
                "name": "case1b: N pos (N,3), N vel (N,3), N guess (N,3), M rng times",
                "positions": np.full((self.N, 3), self.position),
                "velocities": np.full((self.N, 3), self.velocity),
                "initial_guesses": np.full((self.N, 3), self.initial_guess),
                "range_times": np.repeat(self.range_time[0], self.M),
                "doppler_frequencies": self.doppler_freq,
                "expected_ndim": 3,
                "expected_shape": (self.N, self.M, 3),
            },
            {
                "name": "case1c: N pos (N,3), N vel (N,3), M guess (M,3), M rng times",
                "positions": np.full((self.N, 3), self.position),
                "velocities": np.full((self.N, 3), self.velocity),
                "initial_guesses": np.full((self.M, 3), self.initial_guess),
                "range_times": np.repeat(self.range_time[0], self.M),
                "doppler_frequencies": self.doppler_freq,
                "expected_ndim": 3,
                "expected_shape": (self.N, self.M, 3),
            },
            {
                "name": "case1d: N pos (N,3), N vel (N,3), N guess (N,3), M rng times, M doppler freqs",
                "positions": np.full((self.N, 3), self.position),
                "velocities": np.full((self.N, 3), self.velocity),
                "initial_guesses": np.full((self.N, 3), self.initial_guess),
                "range_times": np.repeat(self.range_time[0], self.M),
                "doppler_frequencies": np.repeat(self.doppler_freq, self.M),
                "expected_ndim": 3,
                "expected_shape": (self.N, self.M, 3),
            },
            {
                "name": "case1e: N pos (N,3), N vel (N,3), M guess (M,3), M rng times, M doppler freqs",
                "positions": np.full((self.N, 3), self.position),
                "velocities": np.full((self.N, 3), self.velocity),
                "initial_guesses": np.full((self.M, 3), self.initial_guess),
                "range_times": np.repeat(self.range_time[0], self.M),
                "doppler_frequencies": np.repeat(self.doppler_freq, self.M),
                "expected_ndim": 3,
                "expected_shape": (self.N, self.M, 3),
            },
        ]

        for case in test_cases:
            with self.subTest(case=case["name"]):
                out = direct_geocoding_monostatic_core_range_vectorized(
                    sensor_positions=case["positions"],
                    sensor_velocities=case["velocities"],
                    initial_guesses=case["initial_guesses"],
                    range_times=case["range_times"],
                    doppler_frequencies=case["doppler_frequencies"],
                    wavelength=self.wavelength,
                    altitude=self.geodetic_altitude,
                )

                self.assertEqual(out.ndim, case["expected_ndim"])
                self.assertEqual(out.shape, case["expected_shape"])

                # Determine expected result shape
                if case["expected_shape"] == (3,):
                    expected_result = self.results
                elif case["expected_shape"] == (1, 3):
                    expected_result = self.results.reshape(1, 3)
                elif case["expected_shape"] == (self.M, 3):
                    expected_result = np.full((self.M, 3), self.results)
                elif case["expected_shape"] == (self.N, 3):
                    expected_result = np.full((self.N, 3), self.results)
                elif case["expected_shape"] == (self.N, self.M, 3):
                    expected_result = np.full((self.N, self.M, 3), self.results)
                else:
                    expected_result = self.results

                np.testing.assert_allclose(out, expected_result, atol=self.tolerance, rtol=0)

    def test_monostatic_core_error_case(self) -> None:
        """Test direct_geocoding_monostatic_core error handling for mismatched dimensions."""

        # case: M range (M,), N doppler freqs (N,), mismatched frequencies/ranges
        with self.assertRaises(RuntimeError):
            direct_geocoding_monostatic_core(
                sensor_positions=np.full((self.N, 3), self.position),
                sensor_velocities=np.full((self.N, 3), self.velocity),
                initial_guesses=np.full((self.N, 3), self.initial_guess),
                range_times=np.repeat(self.range_time[0], self.M),
                doppler_frequencies=np.repeat(self.doppler_freq, self.N),
                wavelength=self.wavelength,
                altitude=self.geodetic_altitude,
            )

        # case: M range (M,), Q initial guesses (Q,), mismatched initial guesses/ranges
        with self.assertRaises(RuntimeError):
            direct_geocoding_monostatic_core(
                sensor_positions=np.full((self.N, 3), self.position),
                sensor_velocities=np.full((self.N, 3), self.velocity),
                initial_guesses=np.full((17, 3), self.initial_guess),
                range_times=np.repeat(self.range_time[0], self.M),
                doppler_frequencies=np.repeat(self.doppler_freq, self.N),
                wavelength=self.wavelength,
                altitude=self.geodetic_altitude,
            )


class NewtonForDirectGeocodingMonostaticTest(unittest.TestCase):
    """Testing Newton method for direct geocoding monostatic using subtests."""

    def setUp(self):
        self.position = np.array([4387348.749948771, 762123.3489877012, 4553067.931912004])
        self.velocity = np.array([-856.1384108174528, -329.7629775067583, 398.55830806407346])
        self.init_guess = np.array([4385932.628762595, 764443.4718341012, 4551945.624046889])
        self.geodetic_altitude = 0
        self.wavelength = 1
        self.doppler_frequency = 0
        self.range_time = 2.05624579e-05

        self.results = np.array([4385882.195057692, 764600.9869913795, 4551967.6143934])
        self.tolerance = 1e-6

    def test_newton_for_geocoding_array_cases(self) -> None:
        """Testing Newton for geocoding with array inputs using subtests."""

        test_cases = [
            {
                "name": "case0a: 1 pos (3,), 1 vel (3,), 1 init guess (3,)",
                "positions": self.position,
                "velocities": self.velocity,
                "initial_guesses": self.init_guess,
                "expected_ndim": 1,
                "expected_shape": (3,),
            },
            {
                "name": "case0b: 1 pos (1,3), 1 vel (1,3), 1 init guess (1,3)",
                "positions": self.position.reshape(1, 3),
                "velocities": self.velocity.reshape(1, 3),
                "initial_guesses": self.init_guess.reshape(1, 3),
                "expected_ndim": 2,
                "expected_shape": (1, 3),
            },
            {
                "name": "case1: N pos (N,3), N vel (N,3), N init guess (N,3)",
                "positions": np.full((4, 3), self.position),
                "velocities": np.full((4, 3), self.velocity),
                "initial_guesses": np.full((4, 3), self.init_guess),
                "expected_ndim": 2,
                "expected_shape": (4, 3),
            },
        ]

        for case in test_cases:
            with self.subTest(case=case["name"]):
                out = _direct_geocoding_monostatic_newton(
                    sensor_positions=case["positions"],
                    sensor_velocities=case["velocities"],
                    initial_guesses=case["initial_guesses"],
                    range_times=self.range_time,
                    altitude=self.geodetic_altitude,
                    wavelength=self.wavelength,
                    doppler_frequencies=self.doppler_frequency,
                )

                self.assertEqual(out.ndim, case["expected_ndim"])
                self.assertEqual(out.shape, case["expected_shape"])

                if case["expected_shape"] == (3,):
                    expected_result = self.results
                elif case["expected_shape"] == (1, 3):
                    expected_result = self.results.reshape(1, 3)
                elif case["expected_shape"] == (4, 3):
                    expected_result = np.full((4, 3), self.results)
                else:
                    expected_result = self.results

                np.testing.assert_allclose(out, expected_result, atol=self.tolerance, rtol=0)


class DirectGeocodingMonostaticInitTest(unittest.TestCase):
    """Testing direct_geocoding_monostatic_init with various input combinations using subtests."""

    def setUp(self):
        self.position = np.array([4387348.749948771, 762123.3489877012, 4553067.931912004])
        self.velocity = np.array([-856.1384108174528, -329.7629775067583, 398.55830806407346])
        self.init_guess = np.array([4385932.628762595, 764443.4718341012, 4551945.624046889])
        self.geodetic_altitude = 0
        self.wavelength = 1
        self.doppler_frequency = 0
        range_time = 2.05624579e-05
        self.look_direction = "RIGHT"
        self.range_distance = np.median(range_time) * speed_of_light / 2

        self.N = 4
        self.results = np.array([4385882.165361054, 764600.91441278, 4551967.49055163])
        self.tolerance = 1e-6

    def test_direct_geocoding_monostatic_init_cases(self) -> None:
        """Testing direct_geocoding_monostatic_init with various dimension combinations using subtests."""

        test_cases = [
            {
                "name": "case0a: 1 sensor pos (3,), 1 sensor vel (3,)",
                "positions": self.position,
                "velocities": self.velocity,
                "expected_ndim": 1,
                "expected_shape": (3,),
                "expected_result": self.results,
            },
            {
                "name": "case0b: 1 sensor pos (1,3), 1 sensor vel (3,)",
                "positions": self.position.reshape(1, 3),
                "velocities": self.velocity,
                "expected_ndim": 2,
                "expected_shape": (1, 3),
                "expected_result": self.results.reshape(1, 3),
            },
            {
                "name": "case0c: 1 sensor pos (3,), 1 sensor vel (1,3)",
                "positions": self.position,
                "velocities": self.velocity.reshape(1, 3),
                "expected_ndim": 2,
                "expected_shape": (1, 3),
                "expected_result": self.results.reshape(1, 3),
            },
            {
                "name": "case0d: 1 sensor pos (1,3), 1 sensor vel (1,3)",
                "positions": self.position.reshape(1, 3),
                "velocities": self.velocity.reshape(1, 3),
                "expected_ndim": 2,
                "expected_shape": (1, 3),
                "expected_result": self.results.reshape(1, 3),
            },
            {
                "name": "case1a: N sensor pos (N,3), 1 sensor vel (3,)",
                "positions": np.full((self.N, 3), self.position),
                "velocities": self.velocity,
                "expected_ndim": 2,
                "expected_shape": (self.N, 3),
                "expected_result": np.full((self.N, 3), self.results),
            },
            {
                "name": "case1b: N sensor pos (N,3), 1 sensor vel (1,3)",
                "positions": np.full((self.N, 3), self.position),
                "velocities": self.velocity.reshape(1, 3),
                "expected_ndim": 2,
                "expected_shape": (self.N, 3),
                "expected_result": np.full((self.N, 3), self.results),
            },
            {
                "name": "case1c: 1 sensor pos (3,), N sensor vel (N,3)",
                "positions": self.position,
                "velocities": np.full((self.N, 3), self.velocity),
                "expected_ndim": 2,
                "expected_shape": (self.N, 3),
                "expected_result": np.full((self.N, 3), self.results),
            },
            {
                "name": "case1d: 1 sensor pos (1,3), N sensor vel (N,3)",
                "positions": self.position.reshape(1, 3),
                "velocities": np.full((self.N, 3), self.velocity),
                "expected_ndim": 2,
                "expected_shape": (self.N, 3),
                "expected_result": np.full((self.N, 3), self.results),
            },
            {
                "name": "case1e: N sensor pos (N,3), N sensor vel (N,3)",
                "positions": np.full((self.N, 3), self.position),
                "velocities": np.full((self.N, 3), self.velocity),
                "expected_ndim": 2,
                "expected_shape": (self.N, 3),
                "expected_result": np.full((self.N, 3), self.results),
            },
        ]

        for case in test_cases:
            with self.subTest(case=case["name"]):
                out = direct_geocoding_init(
                    sensor_positions=case["positions"],
                    sensor_velocities=case["velocities"],
                    range_distance=self.range_distance,
                    look_direction=self.look_direction,
                )
                self.assertEqual(out.ndim, case["expected_ndim"])
                self.assertEqual(out.shape, case["expected_shape"])
                np.testing.assert_allclose(out, case["expected_result"], atol=self.tolerance, rtol=0)


if __name__ == "__main__":
    unittest.main()
