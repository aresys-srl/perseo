# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for geometry/geocoding/inverse_geocoding.py and inverse_geocoding_core.py monostatic functionalities"""

from __future__ import annotations

import unittest

import numpy as np
import numpy.typing as npt

from perseo_core.geometry.geocoding.inverse_geocoding import (
    inverse_geocoding_monostatic,
    inverse_geocoding_monostatic_init,
)
from perseo_core.geometry.geocoding.inverse_geocoding_core import inverse_geocoding_monostatic_core
from perseo_core.models.trajectory import Trajectory
from perseo_core.timing.precise_datetime import PreciseDateTime
from tests.fixtures.models_data import get_testing_trajectory


def _doppler_equation_residual(
    trajectory: Trajectory,
    az_times: npt.ArrayLike,
    ground_points: np.ndarray,
    wavelength: float,
    frequency_doppler: float,
    scene_velocity: float = 0.0,
) -> np.ndarray:
    """Evaluating doppler equation residual for inverse geocoding monostatic.

    Parameters
    ----------
    trajectory : TwiceDifferentiable3DCurve
        3D curve trajectory from orbit
    az_times : npt.ArrayLike
        azimuth times at which evaluate the equation
    ground_points : np.ndarray
        ground points, in the form (3,) or (N, 3)
    wavelength : float
        carrier signal wavelength
    frequency_doppler : float
        doppler frequency
    scene_velocity : float, optional
        scene velocity, by default 0.0

    Returns
    -------
    np.ndarray
        doppler equation residual
    """

    sensor_position = trajectory.position(az_times)
    sensor_velocity = trajectory.velocity(az_times)
    scene_velocity = np.zeros_like(sensor_velocity)

    line_of_sight = ground_points - sensor_position
    slant_range = np.linalg.norm(line_of_sight, axis=-1)
    doppler_term = wavelength * frequency_doppler / 2.0 * slant_range
    doppler_residual = np.sum((line_of_sight * (scene_velocity - sensor_velocity)), axis=-1) + doppler_term
    return np.array(doppler_residual / slant_range / wavelength)


def _assert_azimuth_output(
    test_case: unittest.TestCase,
    az_times: PreciseDateTime | np.ndarray,
    expected_size: int | None,
    expected_value: PreciseDateTime,
    tolerance: float,
) -> None:
    if expected_size is None:
        test_case.assertIsInstance(az_times, PreciseDateTime)
        test_case.assertLess(abs(az_times - expected_value), tolerance)
        return

    test_case.assertIsInstance(az_times, np.ndarray)
    test_case.assertEqual(az_times.ndim, 1)
    test_case.assertEqual(az_times.size, expected_size)
    test_case.assertTrue(all(isinstance(value, PreciseDateTime) for value in az_times))

    delta_az = np.array(az_times - expected_value, dtype=float)
    np.testing.assert_allclose(delta_az, np.zeros_like(delta_az), atol=tolerance, rtol=0)


def _assert_range_output(
    test_case: unittest.TestCase,
    rng_times: float | np.ndarray,
    expected_size: int | None,
    expected_value: float,
    tolerance: float,
) -> None:
    if expected_size is None:
        test_case.assertIsInstance(rng_times, (float, np.floating))
        test_case.assertLess(abs(float(rng_times) - expected_value), tolerance)
        return

    test_case.assertIsInstance(rng_times, np.ndarray)
    test_case.assertEqual(rng_times.ndim, 1)
    test_case.assertEqual(rng_times.size, expected_size)
    test_case.assertTrue(all(isinstance(value, (float, np.floating)) for value in rng_times))
    np.testing.assert_allclose(rng_times, np.repeat(expected_value, expected_size), atol=tolerance, rtol=0)


def _assert_inverse_geocoding_output(
    test_case: unittest.TestCase,
    az_times: PreciseDateTime | np.ndarray,
    rng_times: float | np.ndarray,
    expected_size: int | None,
    expected_azimuth: PreciseDateTime,
    expected_range: float,
    az_tolerance: float,
    rng_tolerance: float,
) -> None:
    _assert_azimuth_output(test_case, az_times, expected_size, expected_azimuth, az_tolerance)
    _assert_range_output(test_case, rng_times, expected_size, expected_range, rng_tolerance)


def _assert_init_output(
    test_case: unittest.TestCase,
    az_times: PreciseDateTime | np.ndarray,
    expected_size: int | None,
    expected_value: PreciseDateTime,
    tolerance: float,
) -> None:
    _assert_azimuth_output(test_case, az_times, expected_size, expected_value, tolerance)


class _InverseGeocodingMonostaticBase(unittest.TestCase):
    def setUp(self) -> None:
        self.trajectory = get_testing_trajectory()
        self.wavelength = 1
        self.doppler_freq = 0
        self.time_step = self.trajectory.times[1] - self.trajectory.times[0]
        self.init_guess = PreciseDateTime.from_utc_string("13-FEB-2023 09:34:01.500000000000")
        self.ground_point = np.array(
            [-2243618.48435212, -4728341.28615007, 3633267.229522297],
        )
        self.az_abs_tolerance = 1e-10
        self.N = 5
        self.M = 7
        self.azimuth_res = PreciseDateTime.from_utc_string("13-FEB-2023 09:33:58.480826322795")
        self.range_res = 0.0036229998773038815

    def _ground_points(self, size: int) -> np.ndarray:
        return np.full((size, 3), self.ground_point)


class InverseGeocodingMonostaticCoreTest(_InverseGeocodingMonostaticBase):
    """Testing inverse geocoding monostatic core"""

    def setUp(self) -> None:
        super().setUp()
        self.rng_abs_tolerance = 1e-10
        self.residual_tolerance = 1e-10

    def test_inverse_geocoding_monostatic_core_cases(self) -> None:
        """Testing inverse_geocoding_monostatic_core with subtests."""

        test_cases = [
            {
                "name": "case0a: 1 ground point (3,), 1 doppler freq, 1 init guess PDT",
                "ground_points": self.ground_point,
                "frequencies": self.doppler_freq,
                "initial_guesses": self.init_guess,
                "expected_size": None,
                "residual_ground_points": self.ground_point,
            },
            {
                "name": "case0b: 1 ground point (1,3), 1 doppler freq, 1 init guess PDT",
                "ground_points": self.ground_point.reshape(1, 3),
                "frequencies": self.doppler_freq,
                "initial_guesses": self.init_guess,
                "expected_size": 1,
                "residual_ground_points": self.ground_point.reshape(1, 3),
            },
            {
                "name": "case0c: 1 ground point (1,3), 1 doppler freq, 1 init guess PDT",
                "ground_points": self.ground_point.reshape(1, 3),
                "frequencies": self.doppler_freq,
                "initial_guesses": self.init_guess,
                "expected_size": 1,
                "residual_ground_points": self.ground_point.reshape(1, 3),
            },
            {
                "name": "case1a: N ground points (N, 3), 1 doppler freq, 1 init guess PDT",
                "ground_points": self._ground_points(self.N),
                "frequencies": self.doppler_freq,
                "initial_guesses": self.init_guess,
                "expected_size": self.N,
                "residual_ground_points": self._ground_points(self.N),
            },
            {
                "name": "case1b: N ground points (N, 3), 1 doppler freq, N init guesses (N,)",
                "ground_points": self._ground_points(self.N),
                "frequencies": self.doppler_freq,
                "initial_guesses": np.repeat(self.init_guess, self.N),
                "expected_size": self.N,
                "residual_ground_points": self._ground_points(self.N),
            },
            {
                "name": "case1c: N ground points (N, 3), N doppler freqs (N,), N init guesses (N,)",
                "ground_points": self._ground_points(self.N),
                "frequencies": np.repeat(self.doppler_freq, self.N),
                "initial_guesses": np.repeat(self.init_guess, self.N),
                "expected_size": self.N,
                "residual_ground_points": self._ground_points(self.N),
            },
            {
                "name": "case2a: 1 ground point (3,), 1 doppler freq, N init guesses (N,)",
                "ground_points": self.ground_point,
                "frequencies": self.doppler_freq,
                "initial_guesses": np.repeat(self.init_guess, self.N),
                "expected_size": self.N,
                "residual_ground_points": self._ground_points(self.N),
            },
            {
                "name": "case2b: 1 ground point (1,3), 1 doppler freq, N init guesses (N,)",
                "ground_points": self.ground_point.reshape(1, 3),
                "frequencies": self.doppler_freq,
                "initial_guesses": np.repeat(self.init_guess, self.N),
                "expected_size": self.N,
                "residual_ground_points": self._ground_points(self.N),
            },
            {
                "name": "case3: 1 ground point (3,), N doppler freqs (N,), 1 init guess PDT",
                "ground_points": self.ground_point,
                "frequencies": np.repeat(self.doppler_freq, self.N),
                "initial_guesses": self.init_guess,
                "expected_size": self.N,
                "residual_ground_points": self._ground_points(self.N),
            },
        ]

        for case in test_cases:
            with self.subTest(case=case["name"]):
                az_times, rng_times = inverse_geocoding_monostatic_core(
                    trajectory=self.trajectory,
                    ground_points=case["ground_points"],
                    frequencies_doppler_centroid=case["frequencies"],
                    initial_guesses=case["initial_guesses"],
                    wavelength=self.wavelength,
                )
                doppler_residual = _doppler_equation_residual(
                    trajectory=self.trajectory,
                    ground_points=case["residual_ground_points"],
                    az_times=az_times,
                    frequency_doppler=self.doppler_freq,
                    wavelength=self.wavelength,
                )

                _assert_inverse_geocoding_output(
                    self,
                    az_times,
                    rng_times,
                    case["expected_size"],
                    self.azimuth_res,
                    self.range_res,
                    self.az_abs_tolerance,
                    self.rng_abs_tolerance,
                )
                np.testing.assert_allclose(
                    doppler_residual,
                    np.zeros_like(doppler_residual),
                    atol=self.residual_tolerance,
                    rtol=0,
                )

    def test_inverse_geocoding_monostatic_core_error_cases(self) -> None:
        """Testing inverse_geocoding_monostatic_core error cases with subtests."""

        error_cases = [
            {
                "name": "case4a: N ground points (N, 3), 1 doppler freq, M init guesses (M,)",
                "ground_points": self._ground_points(self.N),
                "frequencies": self.doppler_freq,
                "initial_guesses": np.repeat(self.init_guess, self.M),
            },
            {
                "name": "case4b: N ground points (N, 3), M doppler freqs (M,), 1 init guess",
                "ground_points": self._ground_points(self.N),
                "frequencies": np.repeat(self.doppler_freq, self.M),
                "initial_guesses": self.init_guess,
            },
        ]

        for case in error_cases:
            with self.subTest(case=case["name"]):
                with self.assertRaises(RuntimeError):
                    inverse_geocoding_monostatic_core(
                        trajectory=self.trajectory,
                        ground_points=case["ground_points"],
                        frequencies_doppler_centroid=case["frequencies"],
                        initial_guesses=case["initial_guesses"],
                        wavelength=self.wavelength,
                    )


class InverseGeocodingMonostaticTest(_InverseGeocodingMonostaticBase):
    """Testing inverse geocoding monostatic"""

    def setUp(self) -> None:
        super().setUp()
        self.rng_abs_tolerance = 1e-12

    def test_inverse_geocoding_monostatic_cases(self) -> None:
        """Testing inverse_geocoding_monostatic with subtests."""

        test_cases = [
            {
                "name": "case0a: 1 ground point (3,), 1 doppler freq, no init guess",
                "ground_points": self.ground_point,
                "frequencies": self.doppler_freq,
                "search_time_step": 1,
                "initial_guesses": None,
                "expected_size": None,
            },
            {
                "name": "case0b: 1 ground point (1,3), 1 doppler freq, no init guess",
                "ground_points": self.ground_point.reshape(1, 3),
                "frequencies": self.doppler_freq,
                "search_time_step": self.time_step,
                "initial_guesses": None,
                "expected_size": 1,
            },
            {
                "name": "case0c: 1 ground point (3,), 1 doppler freq, 1 init guess",
                "ground_points": self.ground_point,
                "frequencies": self.doppler_freq,
                "search_time_step": None,
                "initial_guesses": self.init_guess,
                "expected_size": None,
            },
            {
                "name": "case0d: 1 ground point (3,), 1 doppler freq, 1 init guess (1,)",
                "ground_points": self.ground_point,
                "frequencies": self.doppler_freq,
                "search_time_step": None,
                "initial_guesses": np.array([self.init_guess]),
                "expected_size": 1,
            },
            {
                "name": "case1a: 1 ground point (3,), M doppler freqs",
                "ground_points": self.ground_point,
                "frequencies": np.repeat(self.doppler_freq, self.M),
                "search_time_step": 1,
                "initial_guesses": None,
                "expected_size": self.M,
            },
            {
                "name": "case1b: 1 ground point (3,), M doppler freqs, 1 init guess PDT",
                "ground_points": self.ground_point,
                "frequencies": np.repeat(self.doppler_freq, self.M),
                "search_time_step": None,
                "initial_guesses": self.init_guess,
                "expected_size": self.M,
            },
            {
                "name": "case1c: 1 ground point (1,3), M doppler freqs",
                "ground_points": self.ground_point.reshape(1, 3),
                "frequencies": np.repeat(self.doppler_freq, self.M),
                "search_time_step": self.time_step,
                "initial_guesses": None,
                "expected_size": self.M,
            },
            {
                "name": "case2a: N ground points (N, 3), 1 doppler freq",
                "ground_points": self._ground_points(self.N),
                "frequencies": self.doppler_freq,
                "search_time_step": 1,
                "initial_guesses": None,
                "expected_size": self.N,
            },
            {
                "name": "case2b: N ground points (N, 3), 1 doppler freq, 1 init guess",
                "ground_points": self._ground_points(self.N),
                "frequencies": self.doppler_freq,
                "search_time_step": None,
                "initial_guesses": self.init_guess,
                "expected_size": self.N,
            },
            {
                "name": "case2c: N ground points (N, 3), 1 doppler freq, N init guesses",
                "ground_points": self._ground_points(self.N),
                "frequencies": self.doppler_freq,
                "search_time_step": None,
                "initial_guesses": np.repeat(self.init_guess, self.N),
                "expected_size": self.N,
            },
            {
                "name": "case3a: N ground points (N, 3), N doppler freqs",
                "ground_points": self._ground_points(self.N),
                "frequencies": np.repeat(self.doppler_freq, self.N),
                "search_time_step": self.time_step,
                "initial_guesses": None,
                "expected_size": self.N,
            },
            {
                "name": "case3b: N ground points (N, 3), N doppler freqs, 1 init guess",
                "ground_points": self._ground_points(self.N),
                "frequencies": np.repeat(self.doppler_freq, self.N),
                "search_time_step": None,
                "initial_guesses": self.init_guess,
                "expected_size": self.N,
            },
            {
                "name": "case3c: N ground points (N, 3), N doppler freqs, N init guesses",
                "ground_points": self._ground_points(self.N),
                "frequencies": np.repeat(self.doppler_freq, self.N),
                "search_time_step": None,
                "initial_guesses": np.repeat(self.init_guess, self.N),
                "expected_size": self.N,
            },
        ]

        for case in test_cases:
            with self.subTest(case=case["name"]):
                kwargs = {
                    "trajectory": self.trajectory,
                    "ground_points": case["ground_points"],
                    "frequencies_doppler_centroid": case["frequencies"],
                    "wavelength": self.wavelength,
                }
                if case["search_time_step"] is not None:
                    kwargs["init_guess_search_time_step"] = case["search_time_step"]
                if case["initial_guesses"] is not None:
                    kwargs["az_initial_time_guesses"] = case["initial_guesses"]

                az_times, rng_times = inverse_geocoding_monostatic(**kwargs)
                _assert_inverse_geocoding_output(
                    self,
                    az_times,
                    rng_times,
                    case["expected_size"],
                    self.azimuth_res,
                    self.range_res,
                    self.az_abs_tolerance,
                    self.rng_abs_tolerance,
                )

    def test_inverse_geocoding_monostatic_error_cases(self) -> None:
        """Testing inverse_geocoding_monostatic error cases with subtests."""

        error_cases = [
            {
                "name": "missing guesses and time step",
                "ground_points": self._ground_points(self.N),
                "frequencies": np.repeat(self.doppler_freq, self.N),
            },
        ]

        for case in error_cases:
            with self.subTest(case=case["name"]):
                with self.assertRaises(RuntimeError):
                    inverse_geocoding_monostatic(
                        trajectory=self.trajectory,
                        ground_points=case["ground_points"],
                        frequencies_doppler_centroid=case["frequencies"],
                        wavelength=self.wavelength,
                    )


class InverseGeocodingMonostaticInitTest(_InverseGeocodingMonostaticBase):
    """Testing inverse geocoding monostatic init"""

    def setUp(self) -> None:
        super().setUp()
        self.result = PreciseDateTime.from_utc_string("13-FEB-2023 09:33:58.000000000000")

    def test_inverse_geocoding_monostatic_init_cases(self) -> None:
        """Testing inverse_geocoding_monostatic_init with subtests."""

        test_cases = [
            {
                "name": "case0a: 1 ground point (3,), 1 freq",
                "ground_points": self.ground_point,
                "frequencies": self.doppler_freq,
                "expected_size": None,
            },
            {
                "name": "case0b: 1 ground point (1, 3), 1 freq",
                "ground_points": self.ground_point.reshape(1, 3),
                "frequencies": self.doppler_freq,
                "expected_size": 1,
            },
            {
                "name": "case0c: 1 ground point (1, 3), 1 freq (array)",
                "ground_points": self.ground_point.reshape(1, 3),
                "frequencies": np.array([self.doppler_freq]),
                "expected_size": 1,
            },
            {
                "name": "case1: N ground points (N, 3), 1 freq",
                "ground_points": self._ground_points(self.N),
                "frequencies": self.doppler_freq,
                "expected_size": self.N,
            },
            {
                "name": "case2: N ground points (N, 3), 1 freq (array)",
                "ground_points": self._ground_points(self.N),
                "frequencies": np.array([self.doppler_freq]),
                "expected_size": self.N,
            },
            {
                "name": "case3: 1 ground point (3,), M freq",
                "ground_points": self.ground_point,
                "frequencies": np.repeat(self.doppler_freq, self.M),
                "expected_size": self.M,
            },
            {
                "name": "case4: N ground points (N, 3), N freq",
                "ground_points": self._ground_points(self.N),
                "frequencies": np.repeat(self.doppler_freq, self.N),
                "expected_size": self.N,
            },
        ]

        for case in test_cases:
            with self.subTest(case=case["name"]):
                az_times = inverse_geocoding_monostatic_init(
                    trajectory=self.trajectory,
                    ground_points=case["ground_points"],
                    time_axis=self.trajectory.times,
                    frequencies_doppler_centroid=case["frequencies"],
                    wavelength=self.wavelength,
                )
                _assert_init_output(
                    self,
                    az_times,
                    case["expected_size"],
                    self.result,
                    self.az_abs_tolerance,
                )


if __name__ == "__main__":
    unittest.main()
