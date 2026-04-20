# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for geometry/geocoding/inverse_geocoding.py and inverse_geocoding_core.py bistatic functionalities"""

from __future__ import annotations

import unittest

import numpy as np

from perseo_core.geometry.geocoding.inverse_geocoding import inverse_geocoding_bistatic
from perseo_core.geometry.geocoding.inverse_geocoding_core import (
    inverse_geocoding_bistatic_init_core,
)
from perseo_core.timing.precise_datetime import PreciseDateTime
from tests.fixtures.models_data import get_testing_trajectory


def _doppler_equation_residual(
    position_rx: np.ndarray,
    position_tx: np.ndarray,
    velocity_rx: np.ndarray,
    velocity_tx: np.ndarray,
    ground_points: np.ndarray,
    wavelength: float,
    freq_doppler: float,
) -> np.ndarray:
    line_of_sight_rx = ground_points - position_rx
    line_of_sight_tx = ground_points - position_tx

    slant_range_rx = np.linalg.norm(position_rx - ground_points, axis=-1)
    slant_range_tx = np.linalg.norm(position_tx - ground_points, axis=-1)

    rng_vel_product_rx = np.sum(line_of_sight_rx * velocity_rx, axis=-1)
    rng_vel_product_tx = np.sum(line_of_sight_tx * velocity_tx, axis=-1)

    doppler_equation_freq_term = wavelength * freq_doppler * slant_range_rx * slant_range_tx
    residual = (-rng_vel_product_tx * slant_range_rx - rng_vel_product_rx * slant_range_tx) + doppler_equation_freq_term

    return np.array(residual / wavelength / slant_range_rx / slant_range_tx)


class InverseGeocodingBistaticTest(unittest.TestCase):
    """Testing inverse geocoding bistatic functionality"""

    def setUp(self) -> None:
        # creating orbit and orbit curve wrapper
        self.trajectory = get_testing_trajectory()
        self.wavelength = 1
        self.doppler_freq = 0

        # inputs
        self.init_guess = PreciseDateTime.from_utc_string("13-FEB-2023 09:34:01.500000000000")
        self.ground_point = np.array(
            [-2243618.48435212, -4728341.28615007, 3633267.229522297],
        )
        self.az_abs_tolerance = 1e-10
        self.rng_abs_tolerance = 1e-10
        self.residual_tolerance = 1e-10  # Hz
        self.N = 5
        self.M = 7

        # expected results
        self.azimuth_res = PreciseDateTime.from_utc_string("13-FEB-2023 09:33:58.482637823016")
        self.range_res = 0.0036229998783991087

    def test_inverse_geocoding_bistatic_cases(self) -> None:
        """Testing inverse_geocoding_bistatic success cases."""

        test_cases = [
            {
                "name": "case0a: scalar",
                "ground_points": self.ground_point,
                "doppler_freqs": self.doppler_freq,
                "init_guess": self.init_guess,
                "init_guess_time_step": None,
                "expected_az_size": None,
                "expected_rng_size": None,
            },
            {
                "name": "case0b: 1x3 array",
                "ground_points": self.ground_point.reshape(1, 3),
                "doppler_freqs": self.doppler_freq,
                "init_guess": self.init_guess,
                "init_guess_time_step": None,
                "expected_az_size": 1,
                "expected_rng_size": 1,
            },
            {
                "name": "case0c: 1x3 array init array",
                "ground_points": self.ground_point.reshape(1, 3),
                "doppler_freqs": self.doppler_freq,
                "init_guess": np.array([self.init_guess]),
                "init_guess_time_step": None,
                "expected_az_size": 1,
                "expected_rng_size": 1,
            },
            {
                "name": "case0d: scalar init array",
                "ground_points": self.ground_point,
                "doppler_freqs": self.doppler_freq,
                "init_guess": np.array([self.init_guess]),
                "init_guess_time_step": None,
                "expected_az_size": 1,
                "expected_rng_size": 1,
            },
            {
                "name": "case1a: Nx3 array",
                "ground_points": np.full((self.N, 3), self.ground_point),
                "doppler_freqs": self.doppler_freq,
                "init_guess": self.init_guess,
                "init_guess_time_step": None,
                "expected_az_size": self.N,
                "expected_rng_size": self.N,
            },
            {
                "name": "case1b: Nx3 init array",
                "ground_points": np.full((self.N, 3), self.ground_point),
                "doppler_freqs": self.doppler_freq,
                "init_guess": np.array([self.init_guess]),
                "init_guess_time_step": None,
                "expected_az_size": self.N,
                "expected_rng_size": self.N,
            },
            {
                "name": "case1c: Nx3 N init",
                "ground_points": np.full((self.N, 3), self.ground_point),
                "doppler_freqs": self.doppler_freq,
                "init_guess": np.repeat(self.init_guess, self.N),
                "init_guess_time_step": None,
                "expected_az_size": self.N,
                "expected_rng_size": self.N,
            },
            {
                "name": "case2a: scalar N init",
                "ground_points": self.ground_point,
                "doppler_freqs": self.doppler_freq,
                "init_guess": np.repeat(self.init_guess, self.N),
                "init_guess_time_step": None,
                "expected_az_size": self.N,
                "expected_rng_size": self.N,
            },
            {
                "name": "case2b: 1x3 N init",
                "ground_points": self.ground_point.reshape(1, 3),
                "doppler_freqs": self.doppler_freq,
                "init_guess": np.repeat(self.init_guess, self.N),
                "init_guess_time_step": None,
                "expected_az_size": self.N,
                "expected_rng_size": self.N,
            },
            {
                "name": "case3a: scalar M doppler",
                "ground_points": self.ground_point,
                "doppler_freqs": np.repeat(self.doppler_freq, self.M),
                "init_guess": self.init_guess,
                "init_guess_time_step": None,
                "expected_az_size": self.M,
                "expected_rng_size": self.M,
            },
            {
                "name": "case3b: Nx3 N doppler",
                "ground_points": np.full((self.N, 3), self.ground_point),
                "doppler_freqs": np.repeat(self.doppler_freq, self.N),
                "init_guess": self.init_guess,
                "init_guess_time_step": None,
                "expected_az_size": self.N,
                "expected_rng_size": self.N,
            },
            {
                "name": "case3c: Nx3 N doppler N init",
                "ground_points": np.full((self.N, 3), self.ground_point),
                "doppler_freqs": np.repeat(self.doppler_freq, self.N),
                "init_guess": np.repeat(self.init_guess, self.N),
                "init_guess_time_step": None,
                "expected_az_size": self.N,
                "expected_rng_size": self.N,
            },
            {
                "name": "case3d: scalar M doppler M init",
                "ground_points": self.ground_point,
                "doppler_freqs": np.repeat(self.doppler_freq, self.M),
                "init_guess": np.repeat(self.init_guess, self.M),
                "init_guess_time_step": None,
                "expected_az_size": self.M,
                "expected_rng_size": self.M,
            },
            {
                "name": "case3e: 1x3 M doppler M init",
                "ground_points": self.ground_point.reshape(1, 3),
                "doppler_freqs": np.repeat(self.doppler_freq, self.M),
                "init_guess": np.repeat(self.init_guess, self.M),
                "init_guess_time_step": None,
                "expected_az_size": self.M,
                "expected_rng_size": self.M,
            },
            {
                "name": "case4c: Nx3 time step",
                "ground_points": np.full((self.N, 3), self.ground_point),
                "doppler_freqs": self.doppler_freq,
                "init_guess": None,
                "init_guess_time_step": 1,
                "expected_az_size": self.N,
                "expected_rng_size": self.N,
            },
            {
                "name": "case4d: Nx3 N doppler time step",
                "ground_points": np.full((self.N, 3), self.ground_point),
                "doppler_freqs": np.repeat(self.doppler_freq, self.N),
                "init_guess": None,
                "init_guess_time_step": self.trajectory.times[1] - self.trajectory.times[0],
                "expected_az_size": self.N,
                "expected_rng_size": self.N,
            },
            {
                "name": "case4e: scalar M doppler time step",
                "ground_points": self.ground_point,
                "doppler_freqs": np.repeat(self.doppler_freq, self.M),
                "init_guess": None,
                "init_guess_time_step": 1,
                "expected_az_size": self.M,
                "expected_rng_size": self.M,
            },
            {
                "name": "case4f: 1x3 M doppler time step",
                "ground_points": self.ground_point.reshape(1, 3),
                "doppler_freqs": np.repeat(self.doppler_freq, self.M),
                "init_guess": None,
                "init_guess_time_step": self.trajectory.times[1] - self.trajectory.times[0],
                "expected_az_size": self.M,
                "expected_rng_size": self.M,
            },
        ]

        for case in test_cases:
            with self.subTest(case=case["name"]):
                if case["init_guess"] is not None:
                    az_times, rng_times = inverse_geocoding_bistatic(
                        trajectory_rx=self.trajectory,
                        trajectory_tx=self.trajectory,
                        ground_points=case["ground_points"],
                        frequencies_doppler_centroid=case["doppler_freqs"],
                        az_initial_time_guesses=case["init_guess"],
                        wavelength=self.wavelength,
                    )
                else:
                    az_times, rng_times = inverse_geocoding_bistatic(
                        trajectory_rx=self.trajectory,
                        trajectory_tx=self.trajectory,
                        ground_points=case["ground_points"],
                        frequencies_doppler_centroid=case["doppler_freqs"],
                        wavelength=self.wavelength,
                        init_guess_search_time_step=case["init_guess_time_step"],
                    )

                az_times_tx = az_times - rng_times
                position_rx = self.trajectory.position(az_times)
                velocity_rx = self.trajectory.velocity(az_times)
                position_tx = self.trajectory.position(az_times_tx)
                velocity_tx = self.trajectory.velocity(az_times_tx)
                doppler_residual_new = _doppler_equation_residual(
                    position_rx=position_rx,
                    position_tx=position_tx,
                    velocity_rx=velocity_rx,
                    velocity_tx=velocity_tx,
                    ground_points=case["ground_points"],
                    wavelength=self.wavelength,
                    freq_doppler=self.doppler_freq,
                )

                np.testing.assert_allclose(
                    doppler_residual_new, np.zeros_like(doppler_residual_new), atol=self.residual_tolerance, rtol=0
                )

                if case["expected_az_size"] is None:
                    self.assertIsInstance(az_times, PreciseDateTime)
                    self.assertIsInstance(rng_times, (float, np.floating))
                    self.assertLess(abs(az_times - self.azimuth_res), self.az_abs_tolerance)
                    self.assertLess(abs(float(rng_times) - self.range_res), self.rng_abs_tolerance)
                else:
                    self.assertIsInstance(az_times, np.ndarray)
                    self.assertIsInstance(rng_times, np.ndarray)
                    self.assertEqual(az_times.ndim, 1)
                    self.assertEqual(rng_times.ndim, 1)
                    self.assertEqual(az_times.size, case["expected_az_size"])
                    self.assertEqual(rng_times.size, case["expected_rng_size"])
                    self.assertTrue(all(isinstance(value, PreciseDateTime) for value in az_times))
                    self.assertTrue(all(isinstance(value, (float, np.floating)) for value in rng_times))

                    delta_az = np.array(az_times - self.azimuth_res, dtype=float)
                    np.testing.assert_allclose(delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0)
                    np.testing.assert_allclose(
                        rng_times,
                        np.repeat(self.range_res, case["expected_rng_size"]),
                        atol=self.rng_abs_tolerance,
                        rtol=0,
                    )

    def test_inverse_geocoding_bistatic_error_cases(self) -> None:
        """Testing inverse_geocoding_bistatic error cases."""

        error_cases = [
            {
                "name": "case4a: error Nx3 M init",
                "ground_points": np.full((self.N, 3), self.ground_point),
                "doppler_freqs": self.doppler_freq,
                "init_guess": np.repeat(self.init_guess, self.M),
                "init_guess_time_step": None,
            },
            {
                "name": "case4b: error Nx3 M doppler",
                "ground_points": np.full((self.N, 3), self.ground_point),
                "doppler_freqs": np.repeat(self.doppler_freq, self.M),
                "init_guess": self.init_guess,
                "init_guess_time_step": None,
            },
            {
                "name": "error: no init guess no time step",
                "ground_points": self.ground_point.reshape(1, 3),
                "doppler_freqs": np.repeat(self.doppler_freq, self.M),
                "init_guess": None,
                "init_guess_time_step": None,
            },
        ]

        for case in error_cases:
            with self.subTest(case=case["name"]):
                with self.assertRaises(RuntimeError):
                    if case["init_guess"] is not None:
                        inverse_geocoding_bistatic(
                            trajectory_rx=self.trajectory,
                            trajectory_tx=self.trajectory,
                            ground_points=case["ground_points"],
                            frequencies_doppler_centroid=case["doppler_freqs"],
                            az_initial_time_guesses=case["init_guess"],
                            wavelength=self.wavelength,
                        )
                    else:
                        inverse_geocoding_bistatic(
                            trajectory_rx=self.trajectory,
                            trajectory_tx=self.trajectory,
                            ground_points=case["ground_points"],
                            frequencies_doppler_centroid=case["doppler_freqs"],
                            wavelength=self.wavelength,
                            init_guess_search_time_step=case["init_guess_time_step"],
                        )


class InverseGeocodingBistaticInitTest(unittest.TestCase):
    """Testing inverse geocoding bistatic initialization"""

    def setUp(self) -> None:
        self.trajectory = get_testing_trajectory()
        self.wavelength = 1
        self.doppler_freq = 0
        self.init_guess = PreciseDateTime.from_utc_string("13-FEB-2023 09:34:01.500000000000")
        self.ground_point = np.array([-2243618.48435212, -4728341.28615007, 3633267.229522297])
        self.az_abs_tolerance = 1e-10
        self.N = 5
        self.M = 7
        self.result = PreciseDateTime.from_utc_string("13-FEB-2023 09:33:58.500000000000")

    def test_inverse_geocoding_bistatic_init_core_cases(self) -> None:
        """Parameterized test with 8 subtests (7 valid cases + 1 error case)."""

        test_cases = [
            {
                "name": "case0a: scalar",
                "ground_points": self.ground_point,
                "doppler_freqs": self.doppler_freq,
                "expected_size": None,
                "should_raise": False,
            },
            {
                "name": "case0b: 1x3",
                "ground_points": self.ground_point.reshape(1, 3),
                "doppler_freqs": self.doppler_freq,
                "expected_size": 1,
                "should_raise": False,
            },
            {
                "name": "case0c: scalar doppler array",
                "ground_points": self.ground_point,
                "doppler_freqs": np.array([self.doppler_freq]),
                "expected_size": 1,
                "should_raise": False,
            },
            {
                "name": "case1a: Nx3",
                "ground_points": np.full((self.N, 3), self.ground_point),
                "doppler_freqs": self.doppler_freq,
                "expected_size": self.N,
                "should_raise": False,
            },
            {
                "name": "case1b: Nx3 N doppler",
                "ground_points": np.full((self.N, 3), self.ground_point),
                "doppler_freqs": np.repeat(self.doppler_freq, self.N),
                "expected_size": self.N,
                "should_raise": False,
            },
            {
                "name": "case2a: scalar M doppler",
                "ground_points": self.ground_point,
                "doppler_freqs": np.repeat(self.doppler_freq, self.M),
                "expected_size": self.M,
                "should_raise": False,
            },
            {
                "name": "case2b: 1x3 M doppler",
                "ground_points": self.ground_point.reshape(1, 3),
                "doppler_freqs": np.repeat(self.doppler_freq, self.M),
                "expected_size": self.M,
                "should_raise": False,
            },
            {
                "name": "case3: error Nx3 M doppler",
                "ground_points": np.full((self.N, 3), self.ground_point),
                "doppler_freqs": np.repeat(self.doppler_freq, self.M),
                "expected_size": None,
                "should_raise": True,
            },
        ]

        for case in test_cases:
            with self.subTest(case=case["name"]):
                if case["should_raise"]:
                    with self.assertRaises(RuntimeError):
                        inverse_geocoding_bistatic_init_core(
                            trajectory_rx=self.trajectory,
                            trajectory_tx=self.trajectory,
                            time_axis_rx=self.trajectory.times,
                            time_axis_tx=self.trajectory.times,
                            ground_points=case["ground_points"],
                            frequencies_doppler_centroid=case["doppler_freqs"],
                            wavelength=self.wavelength,
                        )
                else:
                    az_times = inverse_geocoding_bistatic_init_core(
                        trajectory_rx=self.trajectory,
                        trajectory_tx=self.trajectory,
                        time_axis_rx=self.trajectory.times,
                        time_axis_tx=self.trajectory.times,
                        ground_points=case["ground_points"],
                        frequencies_doppler_centroid=case["doppler_freqs"],
                        wavelength=self.wavelength,
                    )

                    if case["expected_size"] is None:
                        self.assertTrue(isinstance(az_times, PreciseDateTime))
                        self.assertTrue(np.abs(az_times - self.result) < self.az_abs_tolerance)
                    else:
                        self.assertTrue(isinstance(az_times, np.ndarray))
                        self.assertTrue(az_times.size == case["expected_size"])
                        delta_az = np.array(az_times - self.result, dtype=float)
                        np.testing.assert_allclose(
                            delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0
                        )


if __name__ == "__main__":
    unittest.main()
