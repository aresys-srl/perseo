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
from perseo_core.geometry.geocoding.inverse_geocoding_core import (
    inverse_geocoding_monostatic_core,
)
from perseo_core.models.trajectory import Trajectory
from perseo_core.timing.precise_datetime import PreciseDateTime
from tests.common import get_testing_trajectory


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

    sensor_position = trajectory.evaluate(az_times)
    sensor_velocity = trajectory.evaluate_first_derivatives(az_times)
    scene_velocity = np.zeros_like(sensor_velocity)

    line_of_sight = ground_points - sensor_position
    slant_range = np.linalg.norm(line_of_sight, axis=-1)
    doppler_term = wavelength * frequency_doppler / 2.0 * slant_range
    doppler_residual = np.sum((line_of_sight * (scene_velocity - sensor_velocity)), axis=-1) + doppler_term
    return np.array(doppler_residual / slant_range / wavelength)


class InverseGeocodingMonostaticCoreTest(unittest.TestCase):
    """Testing inverse geocoding monostatic core"""

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
        self.azimuth_res = PreciseDateTime.from_utc_string("13-FEB-2023 09:33:58.480826322795")
        self.range_res = 0.0036229998773038815

    def test_inverse_geocoding_monostatic_core_case0a(self) -> None:
        """Testing inverse_geocoding_monostatic_core, case 0a"""

        # case0a: 1 ground point (3,), 1 doppler freq, 1 init guess PDT
        az_times, rng_times = inverse_geocoding_monostatic_core(
            trajectory=self.trajectory,
            ground_points=self.ground_point,
            frequencies_doppler_centroid=self.doppler_freq,
            initial_guesses=self.init_guess,
            wavelength=self.wavelength,
        )
        doppler_residual = _doppler_equation_residual(
            trajectory=self.trajectory,
            ground_points=self.ground_point,
            az_times=az_times,
            frequency_doppler=self.doppler_freq,
            wavelength=self.wavelength,
        )

        # checking results
        self.assertTrue(isinstance(az_times, PreciseDateTime))
        self.assertTrue(isinstance(rng_times, float))

        np.testing.assert_allclose(
            doppler_residual,
            np.zeros_like(doppler_residual),
            atol=self.residual_tolerance,
            rtol=0,
        )
        self.assertTrue(np.abs(az_times - self.azimuth_res) < self.az_abs_tolerance)
        self.assertTrue(np.abs(rng_times - self.range_res) < self.rng_abs_tolerance)

    def test_inverse_geocoding_monostatic_core_case0b(self) -> None:
        """Testing inverse_geocoding_monostatic_core, case 0b"""

        # case0b: 1 ground point (1,3), 1 doppler freq, 1 init guess PDT
        az_times, rng_times = inverse_geocoding_monostatic_core(
            trajectory=self.trajectory,
            ground_points=self.ground_point.reshape(1, 3),
            frequencies_doppler_centroid=self.doppler_freq,
            initial_guesses=self.init_guess,
            wavelength=self.wavelength,
        )
        doppler_residual = _doppler_equation_residual(
            trajectory=self.trajectory,
            ground_points=self.ground_point,
            az_times=az_times,
            frequency_doppler=self.doppler_freq,
            wavelength=self.wavelength,
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
            doppler_residual,
            np.zeros_like(doppler_residual),
            atol=self.residual_tolerance,
            rtol=0,
        )
        delta_az = np.array(az_times - self.azimuth_res, dtype=float)
        np.testing.assert_allclose(delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0)
        np.testing.assert_allclose(rng_times, np.array(self.range_res), atol=self.rng_abs_tolerance, rtol=0)

    def test_inverse_geocoding_monostatic_core_case0c(self) -> None:
        """Testing inverse_geocoding_monostatic_core, case 0c"""

        # case0c: 1 ground point (1,3), 1 doppler freq, 1 init guess PDT
        az_times, rng_times = inverse_geocoding_monostatic_core(
            trajectory=self.trajectory,
            ground_points=self.ground_point.reshape(1, 3),
            frequencies_doppler_centroid=self.doppler_freq,
            initial_guesses=self.init_guess,
            wavelength=self.wavelength,
        )
        doppler_residual = _doppler_equation_residual(
            trajectory=self.trajectory,
            ground_points=self.ground_point,
            az_times=az_times,
            frequency_doppler=self.doppler_freq,
            wavelength=self.wavelength,
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
            doppler_residual,
            np.zeros_like(doppler_residual),
            atol=self.residual_tolerance,
            rtol=0,
        )
        delta_az = np.array(az_times - self.azimuth_res, dtype=float)
        np.testing.assert_allclose(delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0)
        np.testing.assert_allclose(rng_times, np.array(self.range_res), atol=self.rng_abs_tolerance, rtol=0)

    def test_inverse_geocoding_monostatic_core_case1a(self) -> None:
        """Testing inverse_geocoding_monostatic_core, case 1a"""

        # case1a: N ground points (N, 3), 1 doppler freq, 1 init guess PDT
        az_times, rng_times = inverse_geocoding_monostatic_core(
            trajectory=self.trajectory,
            ground_points=np.full((self.N, 3), self.ground_point),
            frequencies_doppler_centroid=self.doppler_freq,
            initial_guesses=self.init_guess,
            wavelength=self.wavelength,
        )
        doppler_residual = _doppler_equation_residual(
            trajectory=self.trajectory,
            ground_points=np.full((self.N, 3), self.ground_point),
            az_times=az_times,
            frequency_doppler=self.doppler_freq,
            wavelength=self.wavelength,
        )

        # checking results
        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(bool([isinstance(p, PreciseDateTime) for p in az_times]))
        self.assertTrue(isinstance(rng_times, np.ndarray))
        self.assertTrue(bool([isinstance(r, float) for r in rng_times]))
        self.assertTrue(az_times.ndim == 1)
        self.assertTrue(rng_times.ndim == 1)
        self.assertTrue(az_times.size == self.N)
        self.assertTrue(rng_times.size == self.N)

        np.testing.assert_allclose(
            doppler_residual,
            np.zeros_like(doppler_residual),
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

    def test_inverse_geocoding_monostatic_core_case1b(self) -> None:
        """Testing inverse_geocoding_monostatic_core, case 1b"""

        # case1b: N ground points (N, 3), 1 doppler freq, N init guesses (N,)
        az_times, rng_times = inverse_geocoding_monostatic_core(
            trajectory=self.trajectory,
            ground_points=np.full((self.N, 3), self.ground_point),
            frequencies_doppler_centroid=self.doppler_freq,
            initial_guesses=np.repeat(self.init_guess, self.N),
            wavelength=self.wavelength,
        )
        doppler_residual = _doppler_equation_residual(
            trajectory=self.trajectory,
            ground_points=np.full((self.N, 3), self.ground_point),
            az_times=az_times,
            frequency_doppler=self.doppler_freq,
            wavelength=self.wavelength,
        )

        # checking results
        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(bool([isinstance(p, PreciseDateTime) for p in az_times]))
        self.assertTrue(isinstance(rng_times, np.ndarray))
        self.assertTrue(bool([isinstance(r, float) for r in rng_times]))
        self.assertTrue(az_times.ndim == 1)
        self.assertTrue(rng_times.ndim == 1)
        self.assertTrue(az_times.size == self.N)
        self.assertTrue(rng_times.size == self.N)

        np.testing.assert_allclose(
            doppler_residual,
            np.zeros_like(doppler_residual),
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

    def test_inverse_geocoding_monostatic_core_case1c(self) -> None:
        """Testing inverse_geocoding_monostatic_core, case 1c"""

        # case1c: N ground points (N, 3), N doppler freqs (N,), N init guesses (N,)
        az_times, rng_times = inverse_geocoding_monostatic_core(
            trajectory=self.trajectory,
            ground_points=np.full((self.N, 3), self.ground_point),
            frequencies_doppler_centroid=np.repeat(self.doppler_freq, self.N),
            initial_guesses=np.repeat(self.init_guess, self.N),
            wavelength=self.wavelength,
        )
        doppler_residual = _doppler_equation_residual(
            trajectory=self.trajectory,
            ground_points=np.full((self.N, 3), self.ground_point),
            az_times=az_times,
            frequency_doppler=self.doppler_freq,
            wavelength=self.wavelength,
        )

        # checking results
        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(bool([isinstance(p, PreciseDateTime) for p in az_times]))
        self.assertTrue(isinstance(rng_times, np.ndarray))
        self.assertTrue(bool([isinstance(r, float) for r in rng_times]))
        self.assertTrue(az_times.ndim == 1)
        self.assertTrue(rng_times.ndim == 1)
        self.assertTrue(az_times.size == self.N)
        self.assertTrue(rng_times.size == self.N)

        np.testing.assert_allclose(
            doppler_residual,
            np.zeros_like(doppler_residual),
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

    def test_inverse_geocoding_monostatic_core_case2a(self) -> None:
        """Testing inverse_geocoding_monostatic_core, case 2a"""

        # case2a: 1 ground point (3,), 1 doppler freq, N init guesses (N,)
        az_times, rng_times = inverse_geocoding_monostatic_core(
            trajectory=self.trajectory,
            ground_points=self.ground_point,
            frequencies_doppler_centroid=self.doppler_freq,
            initial_guesses=np.repeat(self.init_guess, self.N),
            wavelength=self.wavelength,
        )
        doppler_residual = _doppler_equation_residual(
            trajectory=self.trajectory,
            ground_points=np.full((self.N, 3), self.ground_point),
            az_times=az_times,
            frequency_doppler=self.doppler_freq,
            wavelength=self.wavelength,
        )

        # checking results
        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(bool([isinstance(p, PreciseDateTime) for p in az_times]))
        self.assertTrue(isinstance(rng_times, np.ndarray))
        self.assertTrue(bool([isinstance(r, float) for r in rng_times]))
        self.assertTrue(az_times.ndim == 1)
        self.assertTrue(rng_times.ndim == 1)
        self.assertTrue(az_times.size == self.N)
        self.assertTrue(rng_times.size == self.N)

        np.testing.assert_allclose(
            doppler_residual,
            np.zeros_like(doppler_residual),
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

    def test_inverse_geocoding_monostatic_core_case2b(self) -> None:
        """Testing inverse_geocoding_monostatic_core, case 2b"""

        # case2b: 1 ground point (1,3), 1 doppler freq, N init guesses (N,)
        az_times, rng_times = inverse_geocoding_monostatic_core(
            trajectory=self.trajectory,
            ground_points=self.ground_point.reshape(1, 3),
            frequencies_doppler_centroid=self.doppler_freq,
            initial_guesses=np.repeat(self.init_guess, self.N),
            wavelength=self.wavelength,
        )
        doppler_residual = _doppler_equation_residual(
            trajectory=self.trajectory,
            ground_points=np.full((self.N, 3), self.ground_point),
            az_times=az_times,
            frequency_doppler=self.doppler_freq,
            wavelength=self.wavelength,
        )

        # checking results
        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(bool([isinstance(p, PreciseDateTime) for p in az_times]))
        self.assertTrue(isinstance(rng_times, np.ndarray))
        self.assertTrue(bool([isinstance(r, float) for r in rng_times]))
        self.assertTrue(az_times.ndim == 1)
        self.assertTrue(rng_times.ndim == 1)
        self.assertTrue(az_times.size == self.N)
        self.assertTrue(rng_times.size == self.N)

        np.testing.assert_allclose(
            doppler_residual,
            np.zeros_like(doppler_residual),
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

    def test_inverse_geocoding_monostatic_core_case3(self) -> None:
        """Testing inverse_geocoding_monostatic_core, case 3"""

        # case3: 1 ground point (3,), N doppler freqs (N,), 1 init guess PDT
        az_times, rng_times = inverse_geocoding_monostatic_core(
            trajectory=self.trajectory,
            ground_points=self.ground_point,
            frequencies_doppler_centroid=np.repeat(self.doppler_freq, self.N),
            initial_guesses=self.init_guess,
            wavelength=self.wavelength,
        )
        doppler_residual = _doppler_equation_residual(
            trajectory=self.trajectory,
            ground_points=np.full((self.N, 3), self.ground_point),
            az_times=az_times,
            frequency_doppler=self.doppler_freq,
            wavelength=self.wavelength,
        )

        # checking results
        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(bool([isinstance(p, PreciseDateTime) for p in az_times]))
        self.assertTrue(isinstance(rng_times, np.ndarray))
        self.assertTrue(bool([isinstance(r, float) for r in rng_times]))
        self.assertTrue(az_times.ndim == 1)
        self.assertTrue(rng_times.ndim == 1)
        self.assertTrue(az_times.size == self.N)
        self.assertTrue(rng_times.size == self.N)

        np.testing.assert_allclose(
            doppler_residual,
            np.zeros_like(doppler_residual),
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

    def test_inverse_geocoding_monostatic_core_case4a(self) -> None:
        """Testing inverse_geocoding_monostatic_core, case 4a"""

        # case4a: N ground points (N, 3), 1 doppler freq, M init guess (M,)
        with self.assertRaises(RuntimeError):
            _, _ = inverse_geocoding_monostatic_core(
                trajectory=self.trajectory,
                ground_points=np.full((self.N, 3), self.ground_point),
                frequencies_doppler_centroid=self.doppler_freq,
                initial_guesses=np.repeat(self.init_guess, self.M),
                wavelength=self.wavelength,
            )

    def test_inverse_geocoding_monostatic_core_case4b(self) -> None:
        """Testing inverse_geocoding_monostatic_core, case 4b"""

        # case4b: N ground points (N, 3), M doppler freqs (M,), 1 init guess
        with self.assertRaises(RuntimeError):
            _, _ = inverse_geocoding_monostatic_core(
                trajectory=self.trajectory,
                ground_points=np.full((self.N, 3), self.ground_point),
                frequencies_doppler_centroid=np.repeat(self.doppler_freq, self.M),
                initial_guesses=self.init_guess,
                wavelength=self.wavelength,
            )


class InverseGeocodingMonostaticTest(unittest.TestCase):
    """Testing inverse geocoding monostatic"""

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
        self.rng_abs_tolerance = 1e-12
        self.N = 5
        self.M = 7

        # expected results
        self.azimuth_res = PreciseDateTime.from_utc_string("13-FEB-2023 09:33:58.480826322795")
        self.range_res = 0.0036229998773038815

    def test_inverse_geocoding_monostatic_case0a(self) -> None:
        """Testing inverse_geocoding_monostatic, case 0a"""

        # case0a: 1 ground point (3,), 1 doppler freq, no init guess
        az_times, rng_times = inverse_geocoding_monostatic(
            trajectory=self.trajectory,
            ground_points=self.ground_point,
            frequencies_doppler_centroid=self.doppler_freq,
            wavelength=self.wavelength,
            init_guess_search_time_step=1,
        )

        # checking results
        self.assertTrue(isinstance(az_times, PreciseDateTime))
        self.assertTrue(isinstance(rng_times, float))

        self.assertTrue(np.abs(az_times - self.azimuth_res) < self.az_abs_tolerance)
        self.assertTrue(np.abs(rng_times - self.range_res) < self.rng_abs_tolerance)

    def test_inverse_geocoding_monostatic_case0b(self) -> None:
        """Testing inverse_geocoding_monostatic, case 0b"""

        # case0b: 1 ground point (1,3), 1 doppler freq, no init guess
        az_times, rng_times = inverse_geocoding_monostatic(
            trajectory=self.trajectory,
            ground_points=self.ground_point.reshape(1, 3),
            frequencies_doppler_centroid=self.doppler_freq,
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
        self.assertTrue(az_times.size == 1)
        self.assertTrue(rng_times.size == 1)

        delta_az = np.array(az_times - self.azimuth_res, dtype=float)
        np.testing.assert_allclose(delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0)
        np.testing.assert_allclose(rng_times, np.array(self.range_res), self.rng_abs_tolerance)

    def test_inverse_geocoding_monostatic_case0c(self) -> None:
        """Testing inverse_geocoding_monostatic, case 0c"""

        # case0c: 1 ground point (3,), 1 doppler freq, 1 init guess (3,)
        az_times, rng_times = inverse_geocoding_monostatic(
            trajectory=self.trajectory,
            ground_points=self.ground_point,
            frequencies_doppler_centroid=self.doppler_freq,
            wavelength=self.wavelength,
            az_initial_time_guesses=self.init_guess,
        )

        # checking results
        self.assertTrue(isinstance(az_times, PreciseDateTime))
        self.assertTrue(isinstance(rng_times, float))

        self.assertTrue(np.abs(az_times - self.azimuth_res) < self.az_abs_tolerance)
        np.testing.assert_almost_equal(rng_times, self.range_res, self.rng_abs_tolerance)

    def test_inverse_geocoding_monostatic_case0d(self) -> None:
        """Testing inverse_geocoding_monostatic, case 0d"""

        # case0d: 1 ground point (3,), 1 doppler freq, 1 init guess (1,)
        az_times, rng_times = inverse_geocoding_monostatic(
            trajectory=self.trajectory,
            ground_points=self.ground_point,
            frequencies_doppler_centroid=self.doppler_freq,
            wavelength=self.wavelength,
            az_initial_time_guesses=np.array([self.init_guess]),
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
        np.testing.assert_allclose(rng_times, np.array(self.range_res), self.rng_abs_tolerance)

    def test_inverse_geocoding_monostatic_case1a(self) -> None:
        """Testing inverse_geocoding_monostatic, case 1a"""

        # case1a: 1 ground point (3, ), M doppler freq
        az_times, rng_times = inverse_geocoding_monostatic(
            trajectory=self.trajectory,
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
        np.testing.assert_allclose(rng_times, np.repeat(self.range_res, self.M), self.rng_abs_tolerance)

    def test_inverse_geocoding_monostatic_case1b(self) -> None:
        """Testing inverse_geocoding_monostatic, case 1b"""

        # case1b: 1 ground point (3, ), M doppler freq, 1 init guess PDT
        az_times, rng_times = inverse_geocoding_monostatic(
            trajectory=self.trajectory,
            ground_points=self.ground_point,
            frequencies_doppler_centroid=np.repeat(self.doppler_freq, self.M),
            wavelength=self.wavelength,
            az_initial_time_guesses=self.init_guess,
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
        np.testing.assert_allclose(rng_times, np.repeat(self.range_res, self.M), self.rng_abs_tolerance)

    def test_inverse_geocoding_monostatic_case1c(self) -> None:
        """Testing inverse_geocoding_monostatic, case 1c"""

        # case1c: 1 ground point (1,3), M doppler freq
        az_times, rng_times = inverse_geocoding_monostatic(
            trajectory=self.trajectory,
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
        np.testing.assert_allclose(rng_times, np.repeat(self.range_res, self.M), self.rng_abs_tolerance)

    def test_inverse_geocoding_monostatic_case2a(self) -> None:
        """Testing inverse_geocoding_monostatic, case 2a"""

        # case2a: N ground points (N, 3), 1 doppler freq
        az_times, rng_times = inverse_geocoding_monostatic(
            trajectory=self.trajectory,
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
        np.testing.assert_allclose(rng_times, np.repeat(self.range_res, self.N), self.rng_abs_tolerance)

    def test_inverse_geocoding_monostatic_case2b(self) -> None:
        """Testing inverse_geocoding_monostatic, case 2b"""

        # case2b: N ground points (N, 3), 1 doppler freq, 1 init guess
        az_times, rng_times = inverse_geocoding_monostatic(
            trajectory=self.trajectory,
            ground_points=np.full((self.N, 3), self.ground_point),
            frequencies_doppler_centroid=self.doppler_freq,
            wavelength=self.wavelength,
            az_initial_time_guesses=self.init_guess,
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
        np.testing.assert_allclose(rng_times, np.repeat(self.range_res, self.N), self.rng_abs_tolerance)

    def test_inverse_geocoding_monostatic_case2c(self) -> None:
        """Testing inverse_geocoding_monostatic, case 2c"""

        # case2c: N ground points (N, 3), 1 doppler freq, N init guess
        az_times, rng_times = inverse_geocoding_monostatic(
            trajectory=self.trajectory,
            ground_points=np.full((self.N, 3), self.ground_point),
            frequencies_doppler_centroid=self.doppler_freq,
            wavelength=self.wavelength,
            az_initial_time_guesses=np.repeat(self.init_guess, self.N),
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
        np.testing.assert_allclose(rng_times, np.repeat(self.range_res, self.N), self.rng_abs_tolerance)

    def test_inverse_geocoding_monostatic_case3a(self) -> None:
        """Testing inverse_geocoding_monostatic, case 3a"""

        # case3a: N ground points (N, 3), N doppler freqs
        az_times, rng_times = inverse_geocoding_monostatic(
            trajectory=self.trajectory,
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
        np.testing.assert_allclose(rng_times, np.repeat(self.range_res, self.N), self.rng_abs_tolerance)

    def test_inverse_geocoding_monostatic_case3b(self) -> None:
        """Testing inverse_geocoding_monostatic, case 3b"""

        # case3b: N ground points (N, 3), N doppler freqs, 1 init guess
        az_times, rng_times = inverse_geocoding_monostatic(
            trajectory=self.trajectory,
            ground_points=np.full((self.N, 3), self.ground_point),
            frequencies_doppler_centroid=np.repeat(self.doppler_freq, self.N),
            wavelength=self.wavelength,
            az_initial_time_guesses=self.init_guess,
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
        np.testing.assert_allclose(rng_times, np.repeat(self.range_res, self.N), self.rng_abs_tolerance)

    def test_inverse_geocoding_monostatic_case3c(self) -> None:
        """Testing inverse_geocoding_monostatic, case 3c"""

        # case3c: N ground points (N, 3), N doppler freqs, N init guesses
        az_times, rng_times = inverse_geocoding_monostatic(
            trajectory=self.trajectory,
            ground_points=np.full((self.N, 3), self.ground_point),
            frequencies_doppler_centroid=np.repeat(self.doppler_freq, self.N),
            wavelength=self.wavelength,
            az_initial_time_guesses=np.repeat(self.init_guess, self.N),
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
        np.testing.assert_allclose(rng_times, np.repeat(self.range_res, self.N), self.rng_abs_tolerance)

    def test_inverse_geocoding_monostatic_error(self) -> None:
        """Testing inverse_geocoding_monostatic, error for not specifying guesses or time step"""

        with self.assertRaises(RuntimeError):
            inverse_geocoding_monostatic(
                trajectory=self.trajectory,
                ground_points=np.full((self.N, 3), self.ground_point),
                frequencies_doppler_centroid=np.repeat(self.doppler_freq, self.N),
                wavelength=self.wavelength,
            )


class InverseGeocodingMonostaticInitTest(unittest.TestCase):
    """Testing inverse geocoding monostatic init"""

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
        self.result = PreciseDateTime.from_utc_string("13-FEB-2023 09:33:58.000000000000")

    def test_inverse_geocoding_monostatic_init_case0a(self) -> None:
        """Testing inverse geocoding monostatic init, case 0a"""

        # case0a: 1 ground point (3,), 1 freq
        az_times = inverse_geocoding_monostatic_init(
            trajectory=self.trajectory,
            ground_points=self.ground_point,
            time_axis=self.trajectory.times,
            frequencies_doppler_centroid=self.doppler_freq,
            wavelength=self.wavelength,
        )

        self.assertTrue(isinstance(az_times, PreciseDateTime))
        delta_az = np.array(az_times - self.result, dtype=float)
        np.testing.assert_allclose(delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0)

    def test_inverse_geocoding_monostatic_init_case0b(self) -> None:
        """Testing inverse geocoding monostatic init, case 0b"""

        # case0b: 1 ground point (1, 3), 1 freq
        az_times = inverse_geocoding_monostatic_init(
            trajectory=self.trajectory,
            ground_points=self.ground_point.reshape(1, 3),
            time_axis=self.trajectory.times,
            frequencies_doppler_centroid=self.doppler_freq,
            wavelength=self.wavelength,
        )

        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(az_times.size == 1)
        delta_az = np.array(az_times - self.result, dtype=float)
        np.testing.assert_allclose(delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0)

    def test_inverse_geocoding_monostatic_init_case0c(self) -> None:
        """Testing inverse geocoding monostatic init, case 0c"""

        # case0c: 1 ground point (1, 3), 1 freq (array)
        az_times = inverse_geocoding_monostatic_init(
            trajectory=self.trajectory,
            ground_points=self.ground_point.reshape(1, 3),
            time_axis=self.trajectory.times,
            frequencies_doppler_centroid=np.array([self.doppler_freq]),
            wavelength=self.wavelength,
        )

        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(az_times.size == 1)
        delta_az = np.array(az_times - self.result, dtype=float)
        np.testing.assert_allclose(delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0)

    def test_inverse_geocoding_monostatic_init_case1(self) -> None:
        """Testing inverse geocoding monostatic init, case 1"""

        # case1: N ground point (N, 3), 1 freq
        az_times = inverse_geocoding_monostatic_init(
            trajectory=self.trajectory,
            ground_points=np.full((self.N, 3), self.ground_point),
            time_axis=self.trajectory.times,
            frequencies_doppler_centroid=self.doppler_freq,
            wavelength=self.wavelength,
        )

        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(az_times.size == self.N)
        delta_az = np.array(az_times - self.result, dtype=float)
        np.testing.assert_allclose(delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0)

    def test_inverse_geocoding_monostatic_init_case2(self) -> None:
        """Testing inverse geocoding monostatic init, case 2"""

        # case2: N ground point (N, 3), 1 freq (array)
        az_times = inverse_geocoding_monostatic_init(
            trajectory=self.trajectory,
            ground_points=np.full((self.N, 3), self.ground_point),
            time_axis=self.trajectory.times,
            frequencies_doppler_centroid=np.array([self.doppler_freq]),
            wavelength=self.wavelength,
        )

        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(az_times.size == self.N)
        delta_az = np.array(az_times - self.result, dtype=float)
        np.testing.assert_allclose(delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0)

    def test_inverse_geocoding_monostatic_init_case3(self) -> None:
        """Testing inverse geocoding monostatic init, case 3"""

        # case3: 1 ground point (3,), M freq
        az_times = inverse_geocoding_monostatic_init(
            trajectory=self.trajectory,
            ground_points=self.ground_point,
            time_axis=self.trajectory.times,
            frequencies_doppler_centroid=np.repeat(self.doppler_freq, self.M),
            wavelength=self.wavelength,
        )

        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(az_times.size == self.M)
        delta_az = np.array(az_times - self.result, dtype=float)
        np.testing.assert_allclose(delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0)

    def test_inverse_geocoding_monostatic_init_case4(self) -> None:
        """Testing inverse geocoding monostatic init, case 4"""

        # case4: N ground point (N, 3), N freq
        az_times = inverse_geocoding_monostatic_init(
            trajectory=self.trajectory,
            ground_points=np.full((self.N, 3), self.ground_point),
            time_axis=self.trajectory.times,
            frequencies_doppler_centroid=np.repeat(self.doppler_freq, self.N),
            wavelength=self.wavelength,
        )

        self.assertTrue(isinstance(az_times, np.ndarray))
        self.assertTrue(az_times.size == self.N)
        delta_az = np.array(az_times - self.result, dtype=float)
        np.testing.assert_allclose(delta_az, np.zeros_like(delta_az), atol=self.az_abs_tolerance, rtol=0)


if __name__ == "__main__":
    unittest.main()
