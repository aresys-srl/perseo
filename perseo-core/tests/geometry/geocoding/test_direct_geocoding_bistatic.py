# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for geometry/geocoding/direct_geocoding.py and direct_geocoding_core.py bistatic functionalities"""

from __future__ import annotations

import itertools
import unittest
from dataclasses import dataclass

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


@dataclass
class BistaticTestCase:
    sensor_positions_rx: npt.NDArray
    sensor_velocities_rx: npt.NDArray
    sensor_positions_tx: npt.NDArray
    sensor_velocities_tx: npt.NDArray
    range_times: float | npt.NDArray
    doppler_frequencies: float | npt.NDArray
    initial_guesses: npt.NDArray
    expected_shape: tuple[int, ...]


class DirectGeocodingBistaticTest(unittest.TestCase):
    """Testing all direct_geocoding_bistatic case 1* scenarios"""

    def setUp(self):
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

    def _check_residuals(self, case: BistaticTestCase, out: npt.NDArray[np.floating]):
        doppler_residual = doppler_equation_bistatic_residuals(
            sensor_pos_rx=case.sensor_positions_rx,
            sensor_pos_tx=case.sensor_positions_tx,
            sensor_vel_rx=case.sensor_velocities_rx,
            sensor_vel_tx=case.sensor_velocities_tx,
            ground_points=out,
            doppler_freq=self.doppler_frequency,
            wavelength=self.wavelength,
        )
        range_residual = _range_equation_residual_bistatic(
            sensor_pos_rx=case.sensor_positions_rx,
            sensor_pos_tx=case.sensor_positions_tx,
            ground_points=out,
            range_time=case.range_times,
        )
        ellipse_residual = _ellipse_equation_residual(ground_points=out)

        np.testing.assert_allclose(doppler_residual, 0, atol=self.residual_tolerance, rtol=0)
        np.testing.assert_allclose(range_residual, 0, atol=self.residual_tolerance, rtol=0)
        np.testing.assert_allclose(ellipse_residual, 0, atol=self.residual_tolerance, rtol=0)

    def _check_output(self, case: BistaticTestCase, out: npt.NDArray[np.floating]):
        self.assertEqual(out.shape, case.expected_shape)
        np.testing.assert_allclose(
            out,
            np.full(case.expected_shape, self.results),
            atol=self.tolerance,
            rtol=0,
        )

    def _generate_cases_1_point(self):
        """Use itertools.product to generate all valid combinations"""
        positions = [self.position, self.position.reshape(1, 3)]
        velocities = [self.velocity, self.velocity.reshape(1, 3)]
        initial_guesses = [None, self.initial_guess, self.initial_guess.reshape(1, 3)]

        for pos_rx, vel_rx, pos_tx, vel_tx, init_guess in itertools.product(
            positions, velocities, positions, velocities, initial_guesses
        ):
            if init_guess is not None:
                if init_guess.ndim > 1 and pos_tx.ndim == 1 and pos_rx.ndim == 1:
                    continue

            expected_shape = (1, 3) if (pos_rx.ndim > 1 or pos_tx.ndim > 1) else (3,)

            yield BistaticTestCase(
                sensor_positions_rx=pos_rx,
                sensor_velocities_rx=vel_rx,
                sensor_positions_tx=pos_tx,
                sensor_velocities_tx=vel_tx,
                range_times=self.range_times[0],
                doppler_frequencies=self.doppler_frequency,
                initial_guesses=init_guess,
                expected_shape=expected_shape,
            )

    def test_direct_geocoding_bistatic_1_point(self):
        for case in self._generate_cases_1_point():
            out = direct_geocoding_bistatic(
                sensor_positions_rx=case.sensor_positions_rx,
                sensor_velocities_rx=case.sensor_velocities_rx,
                sensor_positions_tx=case.sensor_positions_tx,
                sensor_velocities_tx=case.sensor_velocities_tx,
                range_times=case.range_times,
                look_direction=self.look_direction,
                altitude=self.altitude,
                doppler_frequencies=case.doppler_frequencies,
                wavelength=self.wavelength,
                initial_guesses=case.initial_guesses,
            )

            self._check_residuals(case, out)
            self._check_output(case, out)

    def _generate_cases_N_points(self):
        """Generate all case 1a, 1b, 1c"""

        pos_rx = np.full((self.N, 3), self.position)
        vel_rx = np.full((self.N, 3), self.velocity)

        # TX variants
        tx_variants = [
            (self.position, self.velocity),  # (3,)
            (self.position.reshape(1, 3), self.velocity.reshape(1, 3)),  # (1,3)
        ]

        # Initial guess variants
        init_variants = [
            self.initial_guess,  # (3,) -> case 1a
            np.full((self.N, 3), self.initial_guess),  # (N,3) -> case 1b
        ]

        for (pos_tx, vel_tx), init in itertools.product(tx_variants, init_variants):
            yield BistaticTestCase(
                sensor_positions_rx=pos_rx,
                sensor_velocities_rx=vel_rx,
                sensor_positions_tx=pos_tx,
                sensor_velocities_tx=vel_tx,
                range_times=self.range_times,
                doppler_frequencies=self.doppler_frequency,
                initial_guesses=init,
                expected_shape=(self.N, 3),
            )

    def test_direct_geocoding_bistatic_N_points(self):
        for case in self._generate_cases_N_points():
            out = direct_geocoding_bistatic(
                sensor_positions_rx=case.sensor_positions_rx,
                sensor_velocities_rx=case.sensor_velocities_rx,
                sensor_positions_tx=case.sensor_positions_tx,
                sensor_velocities_tx=case.sensor_velocities_tx,
                range_times=self.range_times,
                look_direction=self.look_direction,
                altitude=self.altitude,
                doppler_frequencies=case.doppler_frequencies,
                wavelength=self.wavelength,
                initial_guesses=case.initial_guesses,
            )

            self._check_residuals(case, out)
            self._check_output(case, out)

    def _generate_cases_M_points(self):
        """Generate all case 2a, 2b, 2c"""

        # RX variants (scalar or (1,3))
        rx_variants = [
            (self.position, self.velocity),  # (3,)
            (self.position.reshape(1, 3), self.velocity.reshape(1, 3)),  # (1,3)
        ]

        # TX always (M,3)
        pos_tx = np.full((self.M, 3), self.position)
        vel_tx = np.full((self.M, 3), self.velocity)

        # Initial guess variants
        init_variants = [
            self.initial_guess,  # (3,) -> case 2a / 2b
            self.initial_guess.reshape(1, 3),  # (1,3) -> case 2c
        ]

        # Doppler variants
        doppler_variants = [
            self.doppler_frequency,  # scalar -> 2a / 2c
            np.repeat(self.doppler_frequency, self.M),  # (M,) -> 2b
        ]

        for (pos_rx, vel_rx), init, doppler in itertools.product(rx_variants, init_variants, doppler_variants):
            yield BistaticTestCase(
                sensor_positions_rx=pos_rx,
                sensor_velocities_rx=vel_rx,
                sensor_positions_tx=pos_tx,
                sensor_velocities_tx=vel_tx,
                initial_guesses=init,
                range_times=np.repeat(self.range_times, self.M),
                doppler_frequencies=doppler,
                expected_shape=(self.M, 3),
            )

    def test_direct_geocoding_bistatic_M_points(self):
        for case in self._generate_cases_M_points():
            out = direct_geocoding_bistatic(
                sensor_positions_rx=case.sensor_positions_rx,
                sensor_velocities_rx=case.sensor_velocities_rx,
                sensor_positions_tx=case.sensor_positions_tx,
                sensor_velocities_tx=case.sensor_velocities_tx,
                range_times=case.range_times,
                look_direction=self.look_direction,
                altitude=self.altitude,
                doppler_frequencies=case.doppler_frequencies,
                wavelength=self.wavelength,
                initial_guesses=case.initial_guesses,
            )

            self._check_residuals(case, out)
            self._check_output(case, out)

    def _generate_cases_3_points(self):
        """Generate all case 3a, 3b, 3c"""

        pos_rx = np.full((self.N, 3), self.position)
        vel_rx = np.full((self.N, 3), self.velocity)

        pos_tx = np.full((self.M, 3), self.position)
        vel_tx = np.full((self.M, 3), self.velocity)

        range_times = np.repeat(self.range_times[0], self.M)

        # Initial guess variants (3a, 3b, 3c)
        init_variants = [
            self.initial_guess,  # (3,) -> 3a
            self.initial_guess.reshape(1, 3),  # (1,3) -> 3b
            np.full((self.N, 3), self.initial_guess),  # (N,3) -> 3c
        ]

        # Doppler variants (scalar vs array)
        doppler_variants = [
            self.doppler_frequency,  # scalar -> 3a, 3b
            np.repeat(self.doppler_frequency, self.M),  # (M,) -> 3c
        ]

        for init, doppler in itertools.product(init_variants, doppler_variants):
            # Skip invalid combos to match original cases exactly
            if isinstance(doppler, np.ndarray) and init.ndim != 2:
                continue  # only valid for 3c

            yield BistaticTestCase(
                sensor_positions_rx=pos_rx,
                sensor_velocities_rx=vel_rx,
                sensor_positions_tx=pos_tx,
                sensor_velocities_tx=vel_tx,
                range_times=range_times,
                doppler_frequencies=doppler,
                initial_guesses=init,
                expected_shape=(self.N, self.M, 3),
            )

    def test_direct_geocoding_bistatic_3_points(self):
        for case in self._generate_cases_3_points():
            out = direct_geocoding_bistatic(
                sensor_positions_rx=case.sensor_positions_rx,
                sensor_velocities_rx=case.sensor_velocities_rx,
                sensor_positions_tx=case.sensor_positions_tx,
                sensor_velocities_tx=case.sensor_velocities_tx,
                range_times=case.range_times,
                look_direction=self.look_direction,
                altitude=self.altitude,
                doppler_frequencies=case.doppler_frequencies,
                wavelength=self.wavelength,
                initial_guesses=case.initial_guesses,
            )

            for range_index in range(self.M):
                case_range = BistaticTestCase(
                    sensor_positions_rx=case.sensor_positions_rx,
                    sensor_velocities_rx=case.sensor_velocities_rx,
                    sensor_positions_tx=case.sensor_positions_tx[range_index, :],
                    sensor_velocities_tx=case.sensor_velocities_tx[range_index, :],
                    range_times=case.range_times[range_index] if np.size(case.range_times) > 1 else case.range_times,
                    doppler_frequencies=case.doppler_frequencies[range_index]
                    if np.size(case.doppler_frequencies) > 1
                    else case.doppler_frequencies,
                    initial_guesses=case.initial_guesses,
                    expected_shape=case.expected_shape,
                )
                self._check_residuals(case_range, out[:, range_index, :])

            self._check_output(case, out)


class DirectGeocodingBistaticCoreTest(unittest.TestCase):
    """Testing direct geocoding bistatic core"""

    def setUp(self):
        self.position = np.array(
            [4387348.749948771, 762123.3489877012, 4553067.931912004],
        )
        self.velocity = np.array(
            [-856.1384108174528, -329.7629775067583, 398.55830806407346],
        )
        self.initial_guess = np.array([4385932.628762595, 764443.4718341012, 4551945.624046889])
        self.range_times = np.array([2.05624579e-05])
        self.doppler_freqs = 0.0
        self.geodetic_altitude = 0.0
        self.wavelength = 1.0
        self.N = 4
        self.M = 5

        self.tolerance = 1e-6

        self.results = np.array([4385882.195057692, 764600.9869913795, 4551967.6143934])

    def test_direct_geocoding_bistatic_core_case0a(self) -> None:
        """Testing direct_geocoding_bistatic_core, case 0a"""

        # case 0a: 1 pos rx (3,), 1 vel rx (3,), 1 init guess (3,), 1 pos tx (3,), 1 vel tx (3,), 1 rng time
        out = direct_geocoding_bistatic_core(
            sensor_positions_rx=self.position,
            sensor_velocities_rx=self.velocity,
            initial_guesses=self.initial_guess,
            sensor_positions_tx=self.position,
            sensor_velocities_tx=self.velocity,
            range_times=self.range_times[0],
            altitude=self.geodetic_altitude,
            doppler_frequencies=self.doppler_freqs,
            wavelength=self.wavelength,
        )

        self.assertEqual(out.ndim, 1)
        self.assertEqual(out.shape, (3,))
        np.testing.assert_allclose(out, self.results, atol=self.tolerance, rtol=0)

    def test_direct_geocoding_bistatic_core_case0b(self) -> None:
        """Testing direct_geocoding_bistatic_core, case 0b"""

        # case 0b: 1 pos rx (1,3), 1 vel rx (1,3), 1 init guess (1,3), 1 pos tx (1,3), 1 vel tx (1,3), 1 rng time
        out = direct_geocoding_bistatic_core(
            sensor_positions_rx=self.position.reshape(1, 3),
            sensor_velocities_rx=self.velocity.reshape(1, 3),
            initial_guesses=self.initial_guess.reshape(1, 3),
            sensor_positions_tx=self.position.reshape(1, 3),
            sensor_velocities_tx=self.velocity.reshape(1, 3),
            range_times=self.range_times[0],
            altitude=self.geodetic_altitude,
            doppler_frequencies=self.doppler_freqs,
            wavelength=self.wavelength,
        )

        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape, (1, 3))
        np.testing.assert_allclose(out, self.results.reshape(1, 3), atol=self.tolerance, rtol=0)

    def test_direct_geocoding_bistatic_core_case0c(self) -> None:
        """Testing direct_geocoding_bistatic_core, case 0c"""

        # case 0c: 1 pos rx (3,), 1 vel rx (3,), 1 init guess (3,), 1 pos tx (1,3), 1 vel tx (1,3), 1 rng time
        out = direct_geocoding_bistatic_core(
            sensor_positions_rx=self.position,
            sensor_velocities_rx=self.velocity,
            initial_guesses=self.initial_guess,
            sensor_positions_tx=self.position.reshape(1, 3),
            sensor_velocities_tx=self.velocity.reshape(1, 3),
            range_times=self.range_times[0],
            altitude=self.geodetic_altitude,
            doppler_frequencies=self.doppler_freqs,
            wavelength=self.wavelength,
        )

        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape, (1, 3))
        np.testing.assert_allclose(out, self.results.reshape(1, 3), atol=self.tolerance, rtol=0)

    def test_direct_geocoding_bistatic_core_case1a(self) -> None:
        """Testing direct_geocoding_bistatic_core, case 1a"""

        # case 1a: N pos rx (N,3), N vel rx (N,3), 1 init guess (3,), 1 pos tx (3,), 1 vel tx (3,), 1 rng time
        out = direct_geocoding_bistatic_core(
            sensor_positions_rx=np.full((self.N, 3), self.position),
            sensor_velocities_rx=np.full((self.N, 3), self.velocity),
            initial_guesses=self.initial_guess,
            sensor_positions_tx=self.position,
            sensor_velocities_tx=self.velocity,
            range_times=self.range_times[0],
            altitude=self.geodetic_altitude,
            doppler_frequencies=self.doppler_freqs,
            wavelength=self.wavelength,
        )

        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape, (self.N, 3))
        np.testing.assert_allclose(out, np.full((self.N, 3), self.results), atol=self.tolerance, rtol=0)

    def test_direct_geocoding_bistatic_core_case1b(self) -> None:
        """Testing direct_geocoding_bistatic_core, case 1b"""

        # case 1b: N pos rx (N,3), N vel rx (N,3), N init guess (N,3), 1 pos tx (3,), 1 vel tx (3,), 1 rng time
        out = direct_geocoding_bistatic_core(
            sensor_positions_rx=np.full((self.N, 3), self.position),
            sensor_velocities_rx=np.full((self.N, 3), self.velocity),
            initial_guesses=np.full((self.N, 3), self.initial_guess),
            sensor_positions_tx=self.position,
            sensor_velocities_tx=self.velocity,
            range_times=self.range_times[0],
            altitude=self.geodetic_altitude,
            doppler_frequencies=self.doppler_freqs,
            wavelength=self.wavelength,
        )

        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape, (self.N, 3))
        np.testing.assert_allclose(out, np.full((self.N, 3), self.results), atol=self.tolerance, rtol=0)

    def test_direct_geocoding_bistatic_core_case1c(self) -> None:
        """Testing direct_geocoding_bistatic_core, case 1c"""

        # case 1c: N pos rx (N,3), N vel rx (N,3), 1 init guess (1,3), 1 pos tx (1,3), 1 vel tx (1,3), 1 rng time
        out = direct_geocoding_bistatic_core(
            sensor_positions_rx=np.full((self.N, 3), self.position),
            sensor_velocities_rx=np.full((self.N, 3), self.velocity),
            initial_guesses=self.initial_guess.reshape(1, 3),
            sensor_positions_tx=self.position.reshape(1, 3),
            sensor_velocities_tx=self.velocity.reshape(1, 3),
            range_times=self.range_times[0],
            altitude=self.geodetic_altitude,
            doppler_frequencies=self.doppler_freqs,
            wavelength=self.wavelength,
        )

        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape, (self.N, 3))
        np.testing.assert_allclose(out, np.full((self.N, 3), self.results), atol=self.tolerance, rtol=0)

    def test_direct_geocoding_bistatic_core_case2a(self) -> None:
        """Testing direct_geocoding_bistatic_core, case 2a"""

        # case 2a: 1 pos rx (3,), 1 vel rx (3,), 1 init guess (3,), M pos tx (M,3), M vel tx (M,3), M rng times
        out = direct_geocoding_bistatic_core(
            sensor_positions_rx=self.position,
            sensor_velocities_rx=self.velocity,
            initial_guesses=self.initial_guess,
            sensor_positions_tx=np.full((self.M, 3), self.position),
            sensor_velocities_tx=np.full((self.M, 3), self.velocity),
            range_times=np.repeat(self.range_times[0], self.M),
            altitude=self.geodetic_altitude,
            doppler_frequencies=self.doppler_freqs,
            wavelength=self.wavelength,
        )

        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape, (self.M, 3))
        np.testing.assert_allclose(out, np.full((self.M, 3), self.results), atol=self.tolerance, rtol=0)

    def test_direct_geocoding_bistatic_core_case2b(self) -> None:
        """Testing direct_geocoding_bistatic_core, case 2a"""

        # case 2a: 1 pos rx (1,3), 1 vel rx (1,3), 1 init guess (1,3), M pos tx (M,3), M vel tx (M,3), M rng times
        out = direct_geocoding_bistatic_core(
            sensor_positions_rx=self.position.reshape(1, 3),
            sensor_velocities_rx=self.velocity.reshape(1, 3),
            initial_guesses=self.initial_guess.reshape(1, 3),
            sensor_positions_tx=np.full((self.M, 3), self.position),
            sensor_velocities_tx=np.full((self.M, 3), self.velocity),
            range_times=np.repeat(self.range_times[0], self.M),
            altitude=self.geodetic_altitude,
            doppler_frequencies=self.doppler_freqs,
            wavelength=self.wavelength,
        )

        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape, (self.M, 3))
        np.testing.assert_allclose(out, np.full((self.M, 3), self.results), atol=self.tolerance, rtol=0)

    def test_direct_geocoding_bistatic_core_case2c(self) -> None:
        """Testing direct_geocoding_bistatic_core, case 2c"""

        # case 2c: 1 pos rx (3,), 1 vel rx (3,), 1 init guess (3,),
        # M pos tx (M,3), M vel tx (M,3), M rng times, M doppler freqs
        out = direct_geocoding_bistatic_core(
            sensor_positions_rx=self.position,
            sensor_velocities_rx=self.velocity,
            initial_guesses=self.initial_guess,
            sensor_positions_tx=np.full((self.M, 3), self.position),
            sensor_velocities_tx=np.full((self.M, 3), self.velocity),
            range_times=np.repeat(self.range_times[0], self.M),
            altitude=self.geodetic_altitude,
            doppler_frequencies=np.repeat(self.doppler_freqs, self.M),
            wavelength=self.wavelength,
        )

        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape, (self.M, 3))
        np.testing.assert_allclose(out, np.full((self.M, 3), self.results), atol=self.tolerance, rtol=0)

    def test_direct_geocoding_bistatic_core_case3a(self) -> None:
        """Testing direct_geocoding_bistatic_core, case 3a"""

        # case 3a: N pos rx (N,3), N vel rx (N,3), 1 init guess (3,),
        # M pos tx (M,3), M vel tx (M,3), M rng times
        out = direct_geocoding_bistatic_core(
            sensor_positions_rx=np.full((self.N, 3), self.position),
            sensor_velocities_rx=np.full((self.N, 3), self.velocity),
            initial_guesses=self.initial_guess,
            sensor_positions_tx=np.full((self.M, 3), self.position),
            sensor_velocities_tx=np.full((self.M, 3), self.velocity),
            range_times=np.repeat(self.range_times[0], self.M),
            altitude=self.geodetic_altitude,
            doppler_frequencies=self.doppler_freqs,
            wavelength=self.wavelength,
        )

        self.assertEqual(out.ndim, 3)
        self.assertEqual(out.shape, (self.N, self.M, 3))
        np.testing.assert_allclose(out, np.full((self.N, self.M, 3), self.results), atol=self.tolerance, rtol=0)

    def test_direct_geocoding_bistatic_core_case3b(self) -> None:
        """Testing direct_geocoding_bistatic_core, case 3b"""

        # case 3b: N pos rx (N,3), N vel rx (N,3), 1 init guess (1,3),
        # M pos tx (M,3), M vel tx (M,3), M rng times, M doppler freqs
        out = direct_geocoding_bistatic_core(
            sensor_positions_rx=np.full((self.N, 3), self.position),
            sensor_velocities_rx=np.full((self.N, 3), self.velocity),
            initial_guesses=self.initial_guess.reshape(1, 3),
            sensor_positions_tx=np.full((self.M, 3), self.position),
            sensor_velocities_tx=np.full((self.M, 3), self.velocity),
            range_times=np.repeat(self.range_times[0], self.M),
            altitude=self.geodetic_altitude,
            doppler_frequencies=np.repeat(self.doppler_freqs, self.M),
            wavelength=self.wavelength,
        )

        self.assertEqual(out.ndim, 3)
        self.assertEqual(out.shape, (self.N, self.M, 3))
        np.testing.assert_allclose(out, np.full((self.N, self.M, 3), self.results), atol=self.tolerance, rtol=0)

    def test_direct_geocoding_bistatic_core_case3c(self) -> None:
        """Testing direct_geocoding_bistatic_core, case 3c"""

        # case 3c: N pos rx (N,3), N vel rx (N,3), N init guess (N,3),
        # M pos tx (M,3), M vel tx (M,3), M rng times
        out = direct_geocoding_bistatic_core(
            sensor_positions_rx=np.full((self.N, 3), self.position),
            sensor_velocities_rx=np.full((self.N, 3), self.velocity),
            initial_guesses=np.full((self.N, 3), self.initial_guess),
            sensor_positions_tx=np.full((self.M, 3), self.position),
            sensor_velocities_tx=np.full((self.M, 3), self.velocity),
            range_times=np.repeat(self.range_times[0], self.M),
            altitude=self.geodetic_altitude,
            doppler_frequencies=self.doppler_freqs,
            wavelength=self.wavelength,
        )

        self.assertEqual(out.ndim, 3)
        self.assertEqual(out.shape, (self.N, self.M, 3))
        np.testing.assert_allclose(out, np.full((self.N, self.M, 3), self.results), atol=self.tolerance, rtol=0)

    def test_direct_geocoding_bistatic_core_case4(self) -> None:
        """Testing direct_geocoding_bistatic_core, case 4"""

        # case 4: N pos rx (N,3), N vel rx (N,3), N init guess (N,3),
        # M pos tx (M,3), M vel tx (M,3), M rng times, M doppler freqs
        out = direct_geocoding_bistatic_core(
            sensor_positions_rx=np.full((self.N, 3), self.position),
            sensor_velocities_rx=np.full((self.N, 3), self.velocity),
            initial_guesses=np.full((self.N, 3), self.initial_guess),
            sensor_positions_tx=np.full((self.M, 3), self.position),
            sensor_velocities_tx=np.full((self.M, 3), self.velocity),
            range_times=np.repeat(self.range_times[0], self.M),
            altitude=self.geodetic_altitude,
            doppler_frequencies=np.repeat(self.doppler_freqs, self.M),
            wavelength=self.wavelength,
        )

        self.assertEqual(out.ndim, 3)
        self.assertEqual(out.shape, (self.N, self.M, 3))
        np.testing.assert_allclose(out, np.full((self.N, self.M, 3), self.results), atol=self.tolerance, rtol=0)

    def test_direct_geocoding_bistatic_core_case5(self) -> None:
        """Testing direct_geocoding_bistatic_core, case 5"""

        # case 5: N pos rx (N,3), N vel rx (N,3), N init guess (N,3),
        # M pos tx (M,3), M vel tx (M,3), M rng times, N doppler freqs
        # raising ambiguous input correlation error: mismatch range times / frequencies
        with self.assertRaises(RuntimeError):
            direct_geocoding_bistatic_core(
                sensor_positions_rx=np.full((self.N, 3), self.position),
                sensor_velocities_rx=np.full((self.N, 3), self.velocity),
                initial_guesses=np.full((self.N, 3), self.initial_guess),
                sensor_positions_tx=np.full((self.M, 3), self.position),
                sensor_velocities_tx=np.full((self.M, 3), self.velocity),
                range_times=np.repeat(self.range_times[0], self.M),
                altitude=self.geodetic_altitude,
                doppler_frequencies=np.repeat(self.doppler_freqs, self.N),
                wavelength=self.wavelength,
            )


class NewtonForDirectGeocodingBistaticTest(unittest.TestCase):
    """Testing Newton for direct geocoding bistatic core"""

    def setUp(self):
        self.position = np.array(
            [4387348.749948771, 762123.3489877012, 4553067.931912004],
        )
        self.velocity = np.array(
            [-856.1384108174528, -329.7629775067583, 398.55830806407346],
        )
        self.initial_guess = np.array([4385932.628762595, 764443.4718341012, 4551945.624046889])
        self.range_times = 2.05624579e-05
        self.doppler_freqs = 0.0
        self.geodetic_altitude = 0.0
        self.wavelength = 1.0
        self.N = 4
        self.M = 5

        self.tolerance = 1e-6

        self.results = np.array([4385882.195057692, 764600.9869913795, 4551967.6143934])

    def test_newton_for_direct_geocoding_bistatic_case0(self) -> None:
        """Testing Newton for direct geocoding bistatic, case 0"""

        # case 0: 1 pos (3,), 1 vel (3,), 1 init guess (3,), 1 rng time
        out = _direct_geocoding_bistatic_newton(
            sensor_positions_rx=self.position,
            sensor_velocities_rx=self.velocity,
            initial_guesses=self.initial_guess,
            sensor_position_tx=self.position,
            sensor_velocity_tx=self.velocity,
            range_time=self.range_times,
            doppler_frequency=self.doppler_freqs,
            wavelength=self.wavelength,
            altitude=self.geodetic_altitude,
        )

        self.assertTrue(out.ndim == 1)
        np.testing.assert_allclose(out, self.results, atol=self.tolerance, rtol=0)

    def test_newton_for_direct_geocoding_bistatic_case1(self) -> None:
        """Testing Newton for direct geocoding bistatic, case 1"""

        # case 1: 1 pos (1, 3), 1 vel (1, 3), 1 init guess (1, 3), 1 rng time
        out = _direct_geocoding_bistatic_newton(
            sensor_positions_rx=self.position.reshape(1, 3),
            sensor_velocities_rx=self.velocity.reshape(1, 3),
            initial_guesses=self.initial_guess.reshape(1, 3),
            sensor_position_tx=self.position,
            sensor_velocity_tx=self.velocity,
            range_time=self.range_times,
            doppler_frequency=self.doppler_freqs,
            wavelength=self.wavelength,
            altitude=self.geodetic_altitude,
        )

        self.assertTrue(out.ndim == 2)
        np.testing.assert_allclose(out, self.results.reshape(1, 3), atol=self.tolerance, rtol=0)

    def test_newton_for_direct_geocoding_bistatic_case2a(self) -> None:
        """Testing Newton for direct geocoding bistatic, case 2a"""

        # case 2a: N pos (N, 3), N vel (N, 3), 1 init guess (3,), 1 rng time
        out = _direct_geocoding_bistatic_newton(
            sensor_positions_rx=np.full((self.N, 3), self.position),
            sensor_velocities_rx=np.full((self.N, 3), self.velocity),
            initial_guesses=self.initial_guess,
            sensor_position_tx=self.position,
            sensor_velocity_tx=self.velocity,
            range_time=self.range_times,
            doppler_frequency=self.doppler_freqs,
            wavelength=self.wavelength,
            altitude=self.geodetic_altitude,
        )

        self.assertTrue(out.ndim == 2)
        np.testing.assert_allclose(out, np.full((self.N, 3), self.results), atol=self.tolerance, rtol=0)

    def test_newton_for_direct_geocoding_bistatic_case2b(self) -> None:
        """Testing Newton for direct geocoding bistatic, case 2b"""

        # case 2b: N pos (N, 3), N vel (N, 3), 1 init guess (1, 3), 1 rng time
        out = _direct_geocoding_bistatic_newton(
            sensor_positions_rx=np.full((self.N, 3), self.position),
            sensor_velocities_rx=np.full((self.N, 3), self.velocity),
            initial_guesses=self.initial_guess.reshape(1, 3),
            sensor_position_tx=self.position,
            sensor_velocity_tx=self.velocity,
            range_time=self.range_times,
            doppler_frequency=self.doppler_freqs,
            wavelength=self.wavelength,
            altitude=self.geodetic_altitude,
        )

        self.assertTrue(out.ndim == 2)
        np.testing.assert_allclose(out, np.full((self.N, 3), self.results), atol=self.tolerance, rtol=0)

    def test_newton_for_direct_geocoding_bistatic_case3(self) -> None:
        """Testing Newton for direct geocoding bistatic, case 3"""

        # case 3: N pos (N, 3), N vel (N, 3), N init guess (N, 3), 1 rng time
        out = _direct_geocoding_bistatic_newton(
            sensor_positions_rx=np.full((self.N, 3), self.position),
            sensor_velocities_rx=np.full((self.N, 3), self.velocity),
            initial_guesses=self.initial_guess.reshape(1, 3),
            sensor_position_tx=self.position,
            sensor_velocity_tx=self.velocity,
            range_time=self.range_times,
            doppler_frequency=self.doppler_freqs,
            wavelength=self.wavelength,
            altitude=self.geodetic_altitude,
        )

        self.assertTrue(out.ndim == 2)
        np.testing.assert_allclose(out, np.full((self.N, 3), self.results), atol=self.tolerance, rtol=0)


if __name__ == "__main__":
    unittest.main()
