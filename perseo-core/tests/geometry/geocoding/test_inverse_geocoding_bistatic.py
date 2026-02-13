# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for geometry/geocoding/inverse_geocoding.py and inverse_geocoding_core.py bistatic functionalities"""

from __future__ import annotations

import unittest

import numpy as np

from perseo_core.geometry.geocoding.inverse_geocoding import inverse_geocoding_bistatic
from perseo_core.geometry.geocoding.inverse_geocoding_core import (
    AmbiguousInputCorrelation,
    inverse_geocoding_bistatic_init_core,
)
from perseo_core.timing.precise_datetime import PreciseDateTime
from tests.common import get_testing_trajectory


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
    """Testing inverse geocoding bistatic_core"""

    def setUp(self):
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

    def test_inverse_geocoding_bistatic_case0a(self) -> None:
        """Testing inverse_geocoding_bistatic, case 0a"""

        # case0a: 1 ground point (3,), 1 doppler freq, 1 init guess PDT
        az_times, rng_times = inverse_geocoding_bistatic(
            trajectory_rx=self.trajectory,
            trajectory_tx=self.trajectory,
            ground_points=self.ground_point,
            frequencies_doppler_centroid=self.doppler_freq,
            az_initial_time_guesses=self.init_guess,
            wavelength=self.wavelength,
        )
        az_times_tx = az_times - rng_times
        position_rx = self.trajectory.evaluate(az_times)
        velocity_rx = self.trajectory.evaluate_first_derivatives(az_times)
        position_tx = self.trajectory.evaluate(az_times_tx)
        velocity_tx = self.trajectory.evaluate_first_derivatives(az_times_tx)
        doppler_residual_new = _doppler_equation_residual(
            position_rx=position_rx,
            position_tx=position_tx,
            velocity_rx=velocity_rx,
            velocity_tx=velocity_tx,
            ground_points=self.ground_point,
            wavelength=self.wavelength,
            freq_doppler=self.doppler_freq,
        )

        # checking results
        self.assertTrue(isinstance(az_times, PreciseDateTime))
        self.assertTrue(isinstance(rng_times, float))

        np.testing.assert_allclose(
            doppler_residual_new,
            np.zeros_like(doppler_residual_new),
            atol=self.residual_tolerance,
            rtol=0,
        )
        self.assertTrue(np.abs(az_times - self.azimuth_res) < self.az_abs_tolerance)
        self.assertTrue(np.abs(rng_times - self.range_res) < self.rng_abs_tolerance)

    def test_inverse_geocoding_bistatic_case0b(self) -> None:
        """Testing inverse_geocoding_bistatic, case 0b"""

        # case0b: 1 ground point (1, 3), 1 doppler freq, 1 init guess PDT
        az_times, rng_times = inverse_geocoding_bistatic(
            trajectory_rx=self.trajectory,
            trajectory_tx=self.trajectory,
            ground_points=self.ground_point.reshape(1, 3),
            frequencies_doppler_centroid=self.doppler_freq,
            az_initial_time_guesses=self.init_guess,
            wavelength=self.wavelength,
        )
        az_times_tx = az_times - rng_times
        position_rx = self.trajectory.evaluate(az_times)
        velocity_rx = self.trajectory.evaluate_first_derivatives(az_times)
        position_tx = self.trajectory.evaluate(az_times_tx)
        velocity_tx = self.trajectory.evaluate_first_derivatives(az_times_tx)
        doppler_residual_new = _doppler_equation_residual(
            position_rx=position_rx,
            position_tx=position_tx,
            velocity_rx=velocity_rx,
            velocity_tx=velocity_tx,
            ground_points=self.ground_point,
            wavelength=self.wavelength,
            freq_doppler=self.doppler_freq,
        )

        # checking results
        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(isinstance(az_times[0], PreciseDateTime))
        self.assertTrue(isinstance(rng_times, np.ndarray))
        self.assertTrue(isinstance(rng_times[0], float))
        self.assertTrue(az_times.ndim == 1)
        self.assertTrue(rng_times.ndim == 1)
        self.assertTrue(az_times.size == 1)
        self.assertTrue(rng_times.size == 1)

        np.testing.assert_allclose(
            doppler_residual_new,
            np.zeros_like(doppler_residual_new),
            atol=self.residual_tolerance,
            rtol=0,
        )
        delta_az = np.array(az_times - self.azimuth_res, dtype=float)
        np.testing.assert_allclose(delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0)
        np.testing.assert_allclose(rng_times, np.array(self.range_res), atol=self.rng_abs_tolerance, rtol=0)

    def test_inverse_geocoding_bistatic_case0c(self) -> None:
        """Testing inverse_geocoding_bistatic, case 0c"""

        # case0c: 1 ground point (1, 3), 1 doppler freq, 1 init guess (array)
        az_times, rng_times = inverse_geocoding_bistatic(
            trajectory_rx=self.trajectory,
            trajectory_tx=self.trajectory,
            ground_points=self.ground_point.reshape(1, 3),
            frequencies_doppler_centroid=self.doppler_freq,
            az_initial_time_guesses=np.array([self.init_guess]),
            wavelength=self.wavelength,
        )
        az_times_tx = az_times - rng_times
        position_rx = self.trajectory.evaluate(az_times)
        velocity_rx = self.trajectory.evaluate_first_derivatives(az_times)
        position_tx = self.trajectory.evaluate(az_times_tx)
        velocity_tx = self.trajectory.evaluate_first_derivatives(az_times_tx)
        doppler_residual_new = _doppler_equation_residual(
            position_rx=position_rx,
            position_tx=position_tx,
            velocity_rx=velocity_rx,
            velocity_tx=velocity_tx,
            ground_points=self.ground_point,
            wavelength=self.wavelength,
            freq_doppler=self.doppler_freq,
        )

        # checking results
        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(isinstance(az_times[0], PreciseDateTime))
        self.assertTrue(isinstance(rng_times, np.ndarray))
        self.assertTrue(isinstance(rng_times[0], float))
        self.assertTrue(az_times.ndim == 1)
        self.assertTrue(rng_times.ndim == 1)
        self.assertTrue(az_times.size == 1)
        self.assertTrue(rng_times.size == 1)

        np.testing.assert_allclose(
            doppler_residual_new,
            np.zeros_like(doppler_residual_new),
            atol=self.residual_tolerance,
            rtol=0,
        )
        delta_az = np.array(az_times - self.azimuth_res, dtype=float)
        np.testing.assert_allclose(delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0)
        np.testing.assert_allclose(rng_times, np.array(self.range_res), atol=self.rng_abs_tolerance, rtol=0)

    def test_inverse_geocoding_bistatic_case0d(self) -> None:
        """Testing inverse_geocoding_bistatic, case 0d"""

        # case0d: 1 ground point (3,), 1 doppler freq, 1 init guess (array)
        az_times, rng_times = inverse_geocoding_bistatic(
            trajectory_rx=self.trajectory,
            trajectory_tx=self.trajectory,
            ground_points=self.ground_point,
            frequencies_doppler_centroid=self.doppler_freq,
            az_initial_time_guesses=np.array([self.init_guess]),
            wavelength=self.wavelength,
        )
        az_times_tx = az_times - rng_times
        position_rx = self.trajectory.evaluate(az_times)
        velocity_rx = self.trajectory.evaluate_first_derivatives(az_times)
        position_tx = self.trajectory.evaluate(az_times_tx)
        velocity_tx = self.trajectory.evaluate_first_derivatives(az_times_tx)
        doppler_residual_new = _doppler_equation_residual(
            position_rx=position_rx,
            position_tx=position_tx,
            velocity_rx=velocity_rx,
            velocity_tx=velocity_tx,
            ground_points=self.ground_point,
            wavelength=self.wavelength,
            freq_doppler=self.doppler_freq,
        )

        np.testing.assert_allclose(
            doppler_residual_new,
            np.zeros_like(doppler_residual_new),
            atol=self.residual_tolerance,
            rtol=0,
        )
        # checking results
        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(isinstance(az_times[0], PreciseDateTime))
        self.assertTrue(isinstance(rng_times, np.ndarray))
        self.assertTrue(isinstance(rng_times[0], float))
        self.assertTrue(az_times.ndim == 1)
        self.assertTrue(rng_times.ndim == 1)
        self.assertTrue(az_times.size == 1)
        self.assertTrue(rng_times.size == 1)

        delta_az = np.array(az_times - self.azimuth_res, dtype=float)
        np.testing.assert_allclose(delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0)
        np.testing.assert_allclose(rng_times, np.array(self.range_res), atol=self.rng_abs_tolerance, rtol=0)

    def test_inverse_geocoding_bistatic_case1a(self) -> None:
        """Testing inverse_geocoding_bistatic, case 1a"""

        # case1a: N ground point (N,3), 1 doppler freq, 1 init guess PDT
        az_times, rng_times = inverse_geocoding_bistatic(
            trajectory_rx=self.trajectory,
            trajectory_tx=self.trajectory,
            ground_points=np.full((self.N, 3), self.ground_point),
            frequencies_doppler_centroid=self.doppler_freq,
            az_initial_time_guesses=self.init_guess,
            wavelength=self.wavelength,
        )
        az_times_tx = az_times - rng_times
        position_rx = self.trajectory.evaluate(az_times)
        velocity_rx = self.trajectory.evaluate_first_derivatives(az_times)
        position_tx = self.trajectory.evaluate(az_times_tx)
        velocity_tx = self.trajectory.evaluate_first_derivatives(az_times_tx)
        doppler_residual_new = _doppler_equation_residual(
            position_rx=position_rx,
            position_tx=position_tx,
            velocity_rx=velocity_rx,
            velocity_tx=velocity_tx,
            ground_points=self.ground_point,
            wavelength=self.wavelength,
            freq_doppler=self.doppler_freq,
        )

        # checking results
        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(isinstance(az_times[0], PreciseDateTime))
        self.assertTrue(isinstance(rng_times, np.ndarray))
        self.assertTrue(isinstance(rng_times[0], float))
        self.assertTrue(az_times.ndim == 1)
        self.assertTrue(rng_times.ndim == 1)
        self.assertTrue(az_times.size == self.N)
        self.assertTrue(rng_times.size == self.N)

        np.testing.assert_allclose(
            doppler_residual_new,
            np.zeros_like(doppler_residual_new),
            atol=self.residual_tolerance,
            rtol=0,
        )
        delta_az = np.array(az_times - self.azimuth_res, dtype=float)
        np.testing.assert_allclose(delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0)
        np.testing.assert_allclose(
            rng_times,
            np.repeat(self.range_res, self.N),
            atol=self.rng_abs_tolerance,
            rtol=0,
        )

    def test_inverse_geocoding_bistatic_case1b(self) -> None:
        """Testing inverse_geocoding_bistatic, case 1b"""

        # case1b: N ground point (N,3), 1 doppler freq, 1 init guess array
        az_times, rng_times = inverse_geocoding_bistatic(
            trajectory_rx=self.trajectory,
            trajectory_tx=self.trajectory,
            ground_points=np.full((self.N, 3), self.ground_point),
            frequencies_doppler_centroid=self.doppler_freq,
            az_initial_time_guesses=np.array([self.init_guess]),
            wavelength=self.wavelength,
        )
        az_times_tx = az_times - rng_times
        position_rx = self.trajectory.evaluate(az_times)
        velocity_rx = self.trajectory.evaluate_first_derivatives(az_times)
        position_tx = self.trajectory.evaluate(az_times_tx)
        velocity_tx = self.trajectory.evaluate_first_derivatives(az_times_tx)
        doppler_residual_new = _doppler_equation_residual(
            position_rx=position_rx,
            position_tx=position_tx,
            velocity_rx=velocity_rx,
            velocity_tx=velocity_tx,
            ground_points=self.ground_point,
            wavelength=self.wavelength,
            freq_doppler=self.doppler_freq,
        )

        # checking results
        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(isinstance(az_times[0], PreciseDateTime))
        self.assertTrue(isinstance(rng_times, np.ndarray))
        self.assertTrue(isinstance(rng_times[0], float))
        self.assertTrue(az_times.ndim == 1)
        self.assertTrue(rng_times.ndim == 1)
        self.assertTrue(az_times.size == self.N)
        self.assertTrue(rng_times.size == self.N)

        np.testing.assert_allclose(
            doppler_residual_new,
            np.zeros_like(doppler_residual_new),
            atol=self.residual_tolerance,
            rtol=0,
        )
        delta_az = np.array(az_times - self.azimuth_res, dtype=float)
        np.testing.assert_allclose(delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0)
        np.testing.assert_allclose(
            rng_times,
            np.repeat(self.range_res, self.N),
            atol=self.rng_abs_tolerance,
            rtol=0,
        )

    def test_inverse_geocoding_bistatic_case1c(self) -> None:
        """Testing inverse_geocoding_bistatic, case 1c"""

        # case1c: N ground point (N,3), 1 doppler freq, N init guesses
        az_times, rng_times = inverse_geocoding_bistatic(
            trajectory_rx=self.trajectory,
            trajectory_tx=self.trajectory,
            ground_points=np.full((self.N, 3), self.ground_point),
            frequencies_doppler_centroid=self.doppler_freq,
            az_initial_time_guesses=np.repeat(self.init_guess, self.N),
            wavelength=self.wavelength,
        )
        az_times_tx = az_times - rng_times
        position_rx = self.trajectory.evaluate(az_times)
        velocity_rx = self.trajectory.evaluate_first_derivatives(az_times)
        position_tx = self.trajectory.evaluate(az_times_tx)
        velocity_tx = self.trajectory.evaluate_first_derivatives(az_times_tx)
        doppler_residual_new = _doppler_equation_residual(
            position_rx=position_rx,
            position_tx=position_tx,
            velocity_rx=velocity_rx,
            velocity_tx=velocity_tx,
            ground_points=self.ground_point,
            wavelength=self.wavelength,
            freq_doppler=self.doppler_freq,
        )

        # checking results
        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(isinstance(az_times[0], PreciseDateTime))
        self.assertTrue(isinstance(rng_times, np.ndarray))
        self.assertTrue(isinstance(rng_times[0], float))
        self.assertTrue(az_times.ndim == 1)
        self.assertTrue(rng_times.ndim == 1)
        self.assertTrue(az_times.size == self.N)
        self.assertTrue(rng_times.size == self.N)

        np.testing.assert_allclose(
            doppler_residual_new,
            np.zeros_like(doppler_residual_new),
            atol=self.residual_tolerance,
            rtol=0,
        )
        delta_az = np.array(az_times - self.azimuth_res, dtype=float)
        np.testing.assert_allclose(delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0)
        np.testing.assert_allclose(
            rng_times,
            np.repeat(self.range_res, self.N),
            atol=self.rng_abs_tolerance,
            rtol=0,
        )

    def test_inverse_geocoding_bistatic_case2a(self) -> None:
        """Testing inverse_geocoding_bistatic, case 2a"""

        # case2a: 1 ground point (3,), 1 doppler freq, N init guesses (N,)
        az_times, rng_times = inverse_geocoding_bistatic(
            trajectory_rx=self.trajectory,
            trajectory_tx=self.trajectory,
            ground_points=self.ground_point,
            frequencies_doppler_centroid=self.doppler_freq,
            az_initial_time_guesses=np.repeat(self.init_guess, self.N),
            wavelength=self.wavelength,
        )
        az_times_tx = az_times - rng_times
        position_rx = self.trajectory.evaluate(az_times)
        velocity_rx = self.trajectory.evaluate_first_derivatives(az_times)
        position_tx = self.trajectory.evaluate(az_times_tx)
        velocity_tx = self.trajectory.evaluate_first_derivatives(az_times_tx)
        doppler_residual_new = _doppler_equation_residual(
            position_rx=position_rx,
            position_tx=position_tx,
            velocity_rx=velocity_rx,
            velocity_tx=velocity_tx,
            ground_points=self.ground_point,
            wavelength=self.wavelength,
            freq_doppler=self.doppler_freq,
        )

        # checking results
        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(isinstance(az_times[0], PreciseDateTime))
        self.assertTrue(isinstance(rng_times, np.ndarray))
        self.assertTrue(isinstance(rng_times[0], float))
        self.assertTrue(az_times.ndim == 1)
        self.assertTrue(rng_times.ndim == 1)
        self.assertTrue(az_times.size == self.N)
        self.assertTrue(rng_times.size == self.N)

        np.testing.assert_allclose(
            doppler_residual_new,
            np.zeros_like(doppler_residual_new),
            atol=self.residual_tolerance,
            rtol=0,
        )
        delta_az = np.array(az_times - self.azimuth_res, dtype=float)
        np.testing.assert_allclose(delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0)
        np.testing.assert_allclose(
            rng_times,
            np.repeat(self.range_res, self.N),
            atol=self.rng_abs_tolerance,
            rtol=0,
        )

    def test_inverse_geocoding_bistatic_case2b(self) -> None:
        """Testing inverse_geocoding_bistatic, case 2b"""

        # case2b: 1 ground point (1, 3), 1 doppler freq, N init guesses (N,)
        az_times, rng_times = inverse_geocoding_bistatic(
            trajectory_rx=self.trajectory,
            trajectory_tx=self.trajectory,
            ground_points=self.ground_point.reshape(1, 3),
            frequencies_doppler_centroid=self.doppler_freq,
            az_initial_time_guesses=np.repeat(self.init_guess, self.N),
            wavelength=self.wavelength,
        )
        az_times_tx = az_times - rng_times
        position_rx = self.trajectory.evaluate(az_times)
        velocity_rx = self.trajectory.evaluate_first_derivatives(az_times)
        position_tx = self.trajectory.evaluate(az_times_tx)
        velocity_tx = self.trajectory.evaluate_first_derivatives(az_times_tx)
        doppler_residual_new = _doppler_equation_residual(
            position_rx=position_rx,
            position_tx=position_tx,
            velocity_rx=velocity_rx,
            velocity_tx=velocity_tx,
            ground_points=self.ground_point,
            wavelength=self.wavelength,
            freq_doppler=self.doppler_freq,
        )

        # checking results
        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(isinstance(az_times[0], PreciseDateTime))
        self.assertTrue(isinstance(rng_times, np.ndarray))
        self.assertTrue(isinstance(rng_times[0], float))
        self.assertTrue(az_times.ndim == 1)
        self.assertTrue(rng_times.ndim == 1)
        self.assertTrue(az_times.size == self.N)
        self.assertTrue(rng_times.size == self.N)

        np.testing.assert_allclose(
            doppler_residual_new,
            np.zeros_like(doppler_residual_new),
            atol=self.residual_tolerance,
            rtol=0,
        )
        delta_az = np.array(az_times - self.azimuth_res, dtype=float)
        np.testing.assert_allclose(delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0)
        np.testing.assert_allclose(
            rng_times,
            np.repeat(self.range_res, self.N),
            atol=self.rng_abs_tolerance,
            rtol=0,
        )

    def test_inverse_geocoding_bistatic_case3a(self) -> None:
        """Testing inverse_geocoding_bistatic, case 3a"""

        # case3a: 1 ground point (3,), M doppler freqs (M,), 1 init guess
        az_times, rng_times = inverse_geocoding_bistatic(
            trajectory_rx=self.trajectory,
            trajectory_tx=self.trajectory,
            ground_points=self.ground_point,
            frequencies_doppler_centroid=np.repeat(self.doppler_freq, self.M),
            az_initial_time_guesses=self.init_guess,
            wavelength=self.wavelength,
        )
        az_times_tx = az_times - rng_times
        position_rx = self.trajectory.evaluate(az_times)
        velocity_rx = self.trajectory.evaluate_first_derivatives(az_times)
        position_tx = self.trajectory.evaluate(az_times_tx)
        velocity_tx = self.trajectory.evaluate_first_derivatives(az_times_tx)
        doppler_residual_new = _doppler_equation_residual(
            position_rx=position_rx,
            position_tx=position_tx,
            velocity_rx=velocity_rx,
            velocity_tx=velocity_tx,
            ground_points=self.ground_point,
            wavelength=self.wavelength,
            freq_doppler=self.doppler_freq,
        )

        # checking results
        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(isinstance(az_times[0], PreciseDateTime))
        self.assertTrue(isinstance(rng_times, np.ndarray))
        self.assertTrue(isinstance(rng_times[0], float))
        self.assertTrue(az_times.ndim == 1)
        self.assertTrue(rng_times.ndim == 1)
        self.assertTrue(az_times.size == self.M)
        self.assertTrue(rng_times.size == self.M)

        np.testing.assert_allclose(
            doppler_residual_new,
            np.zeros_like(doppler_residual_new),
            atol=self.residual_tolerance,
            rtol=0,
        )
        delta_az = np.array(az_times - self.azimuth_res, dtype=float)
        np.testing.assert_allclose(delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0)
        np.testing.assert_allclose(
            rng_times,
            np.repeat(self.range_res, self.M),
            atol=self.rng_abs_tolerance,
            rtol=0,
        )

    def test_inverse_geocoding_bistatic_case3b(self) -> None:
        """Testing inverse_geocoding_bistatic, case 3b"""

        # case3b: N ground point (N,3), N doppler freqs (N,), 1 init guess
        az_times, rng_times = inverse_geocoding_bistatic(
            trajectory_rx=self.trajectory,
            trajectory_tx=self.trajectory,
            ground_points=np.full((self.N, 3), self.ground_point),
            frequencies_doppler_centroid=np.repeat(self.doppler_freq, self.N),
            az_initial_time_guesses=self.init_guess,
            wavelength=self.wavelength,
        )
        az_times_tx = az_times - rng_times
        position_rx = self.trajectory.evaluate(az_times)
        velocity_rx = self.trajectory.evaluate_first_derivatives(az_times)
        position_tx = self.trajectory.evaluate(az_times_tx)
        velocity_tx = self.trajectory.evaluate_first_derivatives(az_times_tx)
        doppler_residual_new = _doppler_equation_residual(
            position_rx=position_rx,
            position_tx=position_tx,
            velocity_rx=velocity_rx,
            velocity_tx=velocity_tx,
            ground_points=self.ground_point,
            wavelength=self.wavelength,
            freq_doppler=self.doppler_freq,
        )

        # checking results
        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(isinstance(az_times[0], PreciseDateTime))
        self.assertTrue(isinstance(rng_times, np.ndarray))
        self.assertTrue(isinstance(rng_times[0], float))
        self.assertTrue(az_times.ndim == 1)
        self.assertTrue(rng_times.ndim == 1)
        self.assertTrue(az_times.size == self.N)
        self.assertTrue(rng_times.size == self.N)

        np.testing.assert_allclose(
            doppler_residual_new,
            np.zeros_like(doppler_residual_new),
            atol=self.residual_tolerance,
            rtol=0,
        )
        delta_az = np.array(az_times - self.azimuth_res, dtype=float)
        np.testing.assert_allclose(delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0)
        np.testing.assert_allclose(
            rng_times,
            np.repeat(self.range_res, self.N),
            atol=self.rng_abs_tolerance,
            rtol=0,
        )

    def test_inverse_geocoding_bistatic_case3c(self) -> None:
        """Testing inverse_geocoding_bistatic, case 3c"""

        # case3c: N ground point (N,3), N doppler freqs (N,), N init guesses (N,)
        az_times, rng_times = inverse_geocoding_bistatic(
            trajectory_rx=self.trajectory,
            trajectory_tx=self.trajectory,
            ground_points=np.full((self.N, 3), self.ground_point),
            frequencies_doppler_centroid=np.repeat(self.doppler_freq, self.N),
            az_initial_time_guesses=np.repeat(self.init_guess, self.N),
            wavelength=self.wavelength,
        )
        az_times_tx = az_times - rng_times
        position_rx = self.trajectory.evaluate(az_times)
        velocity_rx = self.trajectory.evaluate_first_derivatives(az_times)
        position_tx = self.trajectory.evaluate(az_times_tx)
        velocity_tx = self.trajectory.evaluate_first_derivatives(az_times_tx)
        doppler_residual_new = _doppler_equation_residual(
            position_rx=position_rx,
            position_tx=position_tx,
            velocity_rx=velocity_rx,
            velocity_tx=velocity_tx,
            ground_points=self.ground_point,
            wavelength=self.wavelength,
            freq_doppler=self.doppler_freq,
        )

        # checking results
        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(isinstance(az_times[0], PreciseDateTime))
        self.assertTrue(isinstance(rng_times, np.ndarray))
        self.assertTrue(isinstance(rng_times[0], float))
        self.assertTrue(az_times.ndim == 1)
        self.assertTrue(rng_times.ndim == 1)
        self.assertTrue(az_times.size == self.N)
        self.assertTrue(rng_times.size == self.N)

        np.testing.assert_allclose(
            doppler_residual_new,
            np.zeros_like(doppler_residual_new),
            atol=self.residual_tolerance,
            rtol=0,
        )
        delta_az = np.array(az_times - self.azimuth_res, dtype=float)
        np.testing.assert_allclose(delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0)
        np.testing.assert_allclose(
            rng_times,
            np.repeat(self.range_res, self.N),
            atol=self.rng_abs_tolerance,
            rtol=0,
        )

    def test_inverse_geocoding_bistatic_case3d(self) -> None:
        """Testing inverse_geocoding_bistatic, case 3d"""

        # case3d: 1 ground point (3,), M doppler freqs (M,), M init guesses (M,)
        az_times, rng_times = inverse_geocoding_bistatic(
            trajectory_rx=self.trajectory,
            trajectory_tx=self.trajectory,
            ground_points=self.ground_point,
            frequencies_doppler_centroid=np.repeat(self.doppler_freq, self.M),
            az_initial_time_guesses=np.repeat(self.init_guess, self.M),
            wavelength=self.wavelength,
        )
        az_times_tx = az_times - rng_times
        position_rx = self.trajectory.evaluate(az_times)
        velocity_rx = self.trajectory.evaluate_first_derivatives(az_times)
        position_tx = self.trajectory.evaluate(az_times_tx)
        velocity_tx = self.trajectory.evaluate_first_derivatives(az_times_tx)
        doppler_residual_new = _doppler_equation_residual(
            position_rx=position_rx,
            position_tx=position_tx,
            velocity_rx=velocity_rx,
            velocity_tx=velocity_tx,
            ground_points=self.ground_point,
            wavelength=self.wavelength,
            freq_doppler=self.doppler_freq,
        )

        # checking results
        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(isinstance(az_times[0], PreciseDateTime))
        self.assertTrue(isinstance(rng_times, np.ndarray))
        self.assertTrue(isinstance(rng_times[0], float))
        self.assertTrue(az_times.ndim == 1)
        self.assertTrue(rng_times.ndim == 1)
        self.assertTrue(az_times.size == self.M)
        self.assertTrue(rng_times.size == self.M)

        np.testing.assert_allclose(
            doppler_residual_new,
            np.zeros_like(doppler_residual_new),
            atol=self.residual_tolerance,
            rtol=0,
        )
        delta_az = np.array(az_times - self.azimuth_res, dtype=float)
        np.testing.assert_allclose(delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0)
        np.testing.assert_allclose(
            rng_times,
            np.repeat(self.range_res, self.M),
            atol=self.rng_abs_tolerance,
            rtol=0,
        )

    def test_inverse_geocoding_bistatic_case3e(self) -> None:
        """Testing inverse_geocoding_bistatic, case 3e"""

        # case3e: 1 ground point (1, 3), M doppler freqs (M,), M init guesses (M,)
        az_times, rng_times = inverse_geocoding_bistatic(
            trajectory_rx=self.trajectory,
            trajectory_tx=self.trajectory,
            ground_points=self.ground_point.reshape(1, 3),
            frequencies_doppler_centroid=np.repeat(self.doppler_freq, self.M),
            az_initial_time_guesses=np.repeat(self.init_guess, self.M),
            wavelength=self.wavelength,
        )
        az_times_tx = az_times - rng_times
        position_rx = self.trajectory.evaluate(az_times)
        velocity_rx = self.trajectory.evaluate_first_derivatives(az_times)
        position_tx = self.trajectory.evaluate(az_times_tx)
        velocity_tx = self.trajectory.evaluate_first_derivatives(az_times_tx)
        doppler_residual_new = _doppler_equation_residual(
            position_rx=position_rx,
            position_tx=position_tx,
            velocity_rx=velocity_rx,
            velocity_tx=velocity_tx,
            ground_points=self.ground_point,
            wavelength=self.wavelength,
            freq_doppler=self.doppler_freq,
        )

        # checking results
        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(isinstance(az_times[0], PreciseDateTime))
        self.assertTrue(isinstance(rng_times, np.ndarray))
        self.assertTrue(isinstance(rng_times[0], float))
        self.assertTrue(az_times.ndim == 1)
        self.assertTrue(rng_times.ndim == 1)
        self.assertTrue(az_times.size == self.M)
        self.assertTrue(rng_times.size == self.M)

        np.testing.assert_allclose(
            doppler_residual_new,
            np.zeros_like(doppler_residual_new),
            atol=self.residual_tolerance,
            rtol=0,
        )
        delta_az = np.array(az_times - self.azimuth_res, dtype=float)
        np.testing.assert_allclose(delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0)
        np.testing.assert_allclose(
            rng_times,
            np.repeat(self.range_res, self.M),
            atol=self.rng_abs_tolerance,
            rtol=0,
        )

    def test_inverse_geocoding_bistatic_case4a(self) -> None:
        """Testing inverse_geocoding_bistatic, case 4a"""

        # case4a: N ground point (N,3), 1 doppler freq, M init guesses (M,)
        with self.assertRaises(AmbiguousInputCorrelation):
            _, _ = inverse_geocoding_bistatic(
                trajectory_rx=self.trajectory,
                trajectory_tx=self.trajectory,
                ground_points=np.full((self.N, 3), self.ground_point),
                frequencies_doppler_centroid=self.doppler_freq,
                az_initial_time_guesses=np.repeat(self.init_guess, self.M),
                wavelength=self.wavelength,
            )

    def test_inverse_geocoding_bistatic_case4b(self) -> None:
        """Testing inverse_geocoding_bistatic, case 4b"""

        # case4b: N ground point (N,3), M doppler freqs (M,), 1 init guess
        with self.assertRaises(AmbiguousInputCorrelation):
            _, _ = inverse_geocoding_bistatic(
                trajectory_rx=self.trajectory,
                trajectory_tx=self.trajectory,
                ground_points=np.full((self.N, 3), self.ground_point),
                frequencies_doppler_centroid=np.repeat(self.doppler_freq, self.M),
                az_initial_time_guesses=self.init_guess,
                wavelength=self.wavelength,
            )

    def test_inverse_geocoding_bistatic_case4c(self) -> None:
        """Testing inverse_geocoding_bistatic, case 4c"""

        # case4c: N ground point (N,3), 1 doppler freq, init guess time step
        az_times, rng_times = inverse_geocoding_bistatic(
            trajectory_rx=self.trajectory,
            trajectory_tx=self.trajectory,
            ground_points=np.full((self.N, 3), self.ground_point),
            frequencies_doppler_centroid=self.doppler_freq,
            wavelength=self.wavelength,
            init_guess_search_time_step=1,
        )

        # checking results
        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(isinstance(az_times[0], PreciseDateTime))
        self.assertTrue(isinstance(rng_times, np.ndarray))
        self.assertTrue(isinstance(rng_times[0], float))
        self.assertTrue(az_times.ndim == 1)
        self.assertTrue(rng_times.ndim == 1)
        self.assertTrue(az_times.size == self.N)
        self.assertTrue(rng_times.size == self.N)

        delta_az = np.array(az_times - self.azimuth_res, dtype=float)
        np.testing.assert_allclose(delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0)
        np.testing.assert_allclose(
            rng_times,
            np.repeat(self.range_res, self.N),
            atol=self.rng_abs_tolerance,
            rtol=0,
        )

    def test_inverse_geocoding_bistatic_case4d(self) -> None:
        """Testing inverse_geocoding_bistatic, case 4d"""

        # case4d: N ground point (N,3), N doppler freqs (N,), init guess time step
        az_times, rng_times = inverse_geocoding_bistatic(
            trajectory_rx=self.trajectory,
            trajectory_tx=self.trajectory,
            ground_points=np.full((self.N, 3), self.ground_point),
            frequencies_doppler_centroid=np.repeat(self.doppler_freq, self.N),
            wavelength=self.wavelength,
            init_guess_search_time_step=self.trajectory.times[1] - self.trajectory.times[0],
        )

        # checking results
        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(isinstance(az_times[0], PreciseDateTime))
        self.assertTrue(isinstance(rng_times, np.ndarray))
        self.assertTrue(isinstance(rng_times[0], float))
        self.assertTrue(az_times.ndim == 1)
        self.assertTrue(rng_times.ndim == 1)
        self.assertTrue(az_times.size == self.N)
        self.assertTrue(rng_times.size == self.N)

        delta_az = np.array(az_times - self.azimuth_res, dtype=float)
        np.testing.assert_allclose(delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0)
        np.testing.assert_allclose(
            rng_times,
            np.repeat(self.range_res, self.N),
            atol=self.rng_abs_tolerance,
            rtol=0,
        )

    def test_inverse_geocoding_bistatic_case4e(self) -> None:
        """Testing inverse_geocoding_bistatic, case 4e"""

        # case4e: 1 ground point (3,), M doppler freqs (M,), init guess time step
        az_times, rng_times = inverse_geocoding_bistatic(
            trajectory_rx=self.trajectory,
            trajectory_tx=self.trajectory,
            ground_points=self.ground_point,
            frequencies_doppler_centroid=np.repeat(self.doppler_freq, self.M),
            wavelength=self.wavelength,
            init_guess_search_time_step=1,
        )

        # checking results
        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(isinstance(az_times[0], PreciseDateTime))
        self.assertTrue(isinstance(rng_times, np.ndarray))
        self.assertTrue(isinstance(rng_times[0], float))
        self.assertTrue(az_times.ndim == 1)
        self.assertTrue(rng_times.ndim == 1)
        self.assertTrue(az_times.size == self.M)
        self.assertTrue(rng_times.size == self.M)

        delta_az = np.array(az_times - self.azimuth_res, dtype=float)
        np.testing.assert_allclose(delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0)
        np.testing.assert_allclose(
            rng_times,
            np.repeat(self.range_res, self.M),
            atol=self.rng_abs_tolerance,
            rtol=0,
        )

    def test_inverse_geocoding_bistatic_case4f(self) -> None:
        """Testing inverse_geocoding_bistatic, case 4f"""

        # case4f: 1 ground point (1,3), M doppler freqs (M,), init guess time step
        az_times, rng_times = inverse_geocoding_bistatic(
            trajectory_rx=self.trajectory,
            trajectory_tx=self.trajectory,
            ground_points=self.ground_point.reshape(1, 3),
            frequencies_doppler_centroid=np.repeat(self.doppler_freq, self.M),
            wavelength=self.wavelength,
            init_guess_search_time_step=self.trajectory.times[1] - self.trajectory.times[0],
        )

        # checking results
        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(isinstance(az_times[0], PreciseDateTime))
        self.assertTrue(isinstance(rng_times, np.ndarray))
        self.assertTrue(isinstance(rng_times[0], float))
        self.assertTrue(az_times.ndim == 1)
        self.assertTrue(rng_times.ndim == 1)
        self.assertTrue(az_times.size == self.M)
        self.assertTrue(rng_times.size == self.M)

        delta_az = np.array(az_times - self.azimuth_res, dtype=float)
        np.testing.assert_allclose(delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0)
        np.testing.assert_allclose(
            rng_times,
            np.repeat(self.range_res, self.M),
            atol=self.rng_abs_tolerance,
            rtol=0,
        )

    def test_inverse_geocoding_bistatic_error(self) -> None:
        """Testing inverse_geocoding_bistatic, error for no init guess and no guess search time step"""

        with self.assertRaises(RuntimeError):
            inverse_geocoding_bistatic(
                trajectory_rx=self.trajectory,
                trajectory_tx=self.trajectory,
                ground_points=self.ground_point.reshape(1, 3),
                frequencies_doppler_centroid=np.repeat(self.doppler_freq, self.M),
                wavelength=self.wavelength,
            )


class InverseGeocodingBistaticInitTest(unittest.TestCase):
    """Testing inverse geocoding bistatic init"""

    def setUp(self):
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
        self.N = 5
        self.M = 7

        # expected results
        self.result = PreciseDateTime.from_utc_string("13-FEB-2023 09:33:58.500000000000")

    def test_inverse_geocoding_bistatic_init_core_case0a(self) -> None:
        """Testing inverse geocoding bistatic init, case 0a"""

        # case 0a: 1 ground point (3,), 1 doppler freq
        az_times = inverse_geocoding_bistatic_init_core(
            trajectory_rx=self.trajectory,
            trajectory_tx=self.trajectory,
            time_axis_rx=self.trajectory.times,
            time_axis_tx=self.trajectory.times,
            ground_points=self.ground_point,
            frequencies_doppler_centroid=self.doppler_freq,
            wavelength=self.wavelength,
        )

        # checking results
        self.assertTrue(isinstance(az_times, PreciseDateTime))
        self.assertTrue(np.abs(az_times - self.result) < self.az_abs_tolerance)

    def test_inverse_geocoding_bistatic_init_core_case0b(self) -> None:
        """Testing inverse geocoding bistatic init, case 0a"""

        # case 0a: 1 ground point (1, 3), 1 doppler freq
        az_times = inverse_geocoding_bistatic_init_core(
            trajectory_rx=self.trajectory,
            trajectory_tx=self.trajectory,
            time_axis_rx=self.trajectory.times,
            time_axis_tx=self.trajectory.times,
            ground_points=self.ground_point.reshape(1, 3),
            frequencies_doppler_centroid=self.doppler_freq,
            wavelength=self.wavelength,
        )

        # checking results
        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(az_times.size == 1)
        delta_az = np.array(az_times - self.result, dtype=float)
        np.testing.assert_allclose(delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0)

    def test_inverse_geocoding_bistatic_init_core_case0c(self) -> None:
        """Testing inverse geocoding bistatic init, case 0c"""

        # case 0c: 1 ground point (3,), 1 doppler freq array
        az_times = inverse_geocoding_bistatic_init_core(
            trajectory_rx=self.trajectory,
            trajectory_tx=self.trajectory,
            time_axis_rx=self.trajectory.times,
            time_axis_tx=self.trajectory.times,
            ground_points=self.ground_point,
            frequencies_doppler_centroid=np.array([self.doppler_freq]),
            wavelength=self.wavelength,
        )

        # checking results
        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(az_times.size == 1)
        delta_az = np.array(az_times - self.result, dtype=float)
        np.testing.assert_allclose(delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0)

    def test_inverse_geocoding_bistatic_init_core_case1a(self) -> None:
        """Testing inverse geocoding bistatic init, case 1a"""

        # case 1a: N ground point (N, 3), 1 doppler freq
        az_times = inverse_geocoding_bistatic_init_core(
            trajectory_rx=self.trajectory,
            trajectory_tx=self.trajectory,
            time_axis_rx=self.trajectory.times,
            time_axis_tx=self.trajectory.times,
            ground_points=np.full((self.N, 3), self.ground_point),
            frequencies_doppler_centroid=self.doppler_freq,
            wavelength=self.wavelength,
        )

        # checking results
        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(az_times.size == self.N)
        delta_az = np.array(az_times - self.result, dtype=float)
        np.testing.assert_allclose(delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0)

    def test_inverse_geocoding_bistatic_init_core_case1b(self) -> None:
        """Testing inverse geocoding bistatic init, case 1b"""

        # case 1b: N ground point (N, 3), N doppler freq
        az_times = inverse_geocoding_bistatic_init_core(
            trajectory_rx=self.trajectory,
            trajectory_tx=self.trajectory,
            time_axis_rx=self.trajectory.times,
            time_axis_tx=self.trajectory.times,
            ground_points=np.full((self.N, 3), self.ground_point),
            frequencies_doppler_centroid=np.repeat(self.doppler_freq, self.N),
            wavelength=self.wavelength,
        )

        # checking results
        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(az_times.size == self.N)
        delta_az = np.array(az_times - self.result, dtype=float)
        np.testing.assert_allclose(delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0)

    def test_inverse_geocoding_bistatic_init_core_case2a(self) -> None:
        """Testing inverse geocoding bistatic init, case 2a"""

        # case 2a: 1 ground point (3,), M doppler freq
        az_times = inverse_geocoding_bistatic_init_core(
            trajectory_rx=self.trajectory,
            trajectory_tx=self.trajectory,
            time_axis_rx=self.trajectory.times,
            time_axis_tx=self.trajectory.times,
            ground_points=self.ground_point,
            frequencies_doppler_centroid=np.repeat(self.doppler_freq, self.M),
            wavelength=self.wavelength,
        )

        # checking results
        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(az_times.size == self.M)
        delta_az = np.array(az_times - self.result, dtype=float)
        np.testing.assert_allclose(delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0)

    def test_inverse_geocoding_bistatic_init_core_case2b(self) -> None:
        """Testing inverse geocoding bistatic init, case 2b"""

        # case 2b: 1 ground point (1, 3), M doppler freq
        az_times = inverse_geocoding_bistatic_init_core(
            trajectory_rx=self.trajectory,
            trajectory_tx=self.trajectory,
            time_axis_rx=self.trajectory.times,
            time_axis_tx=self.trajectory.times,
            ground_points=self.ground_point.reshape(1, 3),
            frequencies_doppler_centroid=np.repeat(self.doppler_freq, self.M),
            wavelength=self.wavelength,
        )

        # checking results
        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(az_times.size == self.M)
        delta_az = np.array(az_times - self.result, dtype=float)
        np.testing.assert_allclose(delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0)

    def test_inverse_geocoding_bistatic_init_core_case3(self) -> None:
        """Testing inverse geocoding bistatic init, case 3"""

        # case 3: N ground point (N, 3), M doppler freq
        with self.assertRaises(AmbiguousInputCorrelation):
            inverse_geocoding_bistatic_init_core(
                trajectory_rx=self.trajectory,
                trajectory_tx=self.trajectory,
                time_axis_rx=self.trajectory.times,
                time_axis_tx=self.trajectory.times,
                ground_points=np.full((self.N, 3), self.ground_point),
                frequencies_doppler_centroid=np.repeat(self.doppler_freq, self.M),
                wavelength=self.wavelength,
            )


if __name__ == "__main__":
    unittest.main()
