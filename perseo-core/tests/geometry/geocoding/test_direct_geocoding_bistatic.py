# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for geometry/geocoding/direct_geocoding.py and direct_geocoding_core.py bistatic functionalities"""

from __future__ import annotations

import unittest

import numpy as np
from pyproj import Geod
from scipy.constants import speed_of_light

from perseo_core.geometry.doppler import doppler_equation_bistatic_residuals
from perseo_core.geometry.geocoding.direct_geocoding import direct_geocoding_bistatic
from perseo_core.geometry.geocoding.direct_geocoding_core import (
    AmbiguousInputCorrelation,
    _ellipse_equation,
    _newton_for_direct_geocoding_bistatic,
    direct_geocoding_bistatic_core,
)
from perseo_core.models.enums import SensorLookDirection

WGS84 = Geod(ellps="WGS84")


def _range_equation_residual_bistatic(
    sensor_pos_rx: np.ndarray,
    sensor_pos_tx: np.ndarray,
    ground_points: np.ndarray,
    range_time: float,
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


class DirectGeocodingBistaticTest(unittest.TestCase):
    """Testing direct geocoding bistatic"""

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
        self.look_direction = SensorLookDirection.RIGHT_LOOKING
        self.wavelength = 1.0
        self.N = 4
        self.M = 5

        self.tolerance = 1e-6
        self.residual_tolerance = 1e-8

        self.results = np.array([4385882.195057692, 764600.9869913795, 4551967.6143934])

    def test_direct_geocoding_bistatic_case0a(self) -> None:
        """Testing direct_geocoding_bistatic, case 0a"""

        # case 0a: 1 pos rx (3,), 1 vel rx (3,), 1 init guess (3,), 1 pos tx (3,), 1 vel tx (3,), 1 rng time
        out = direct_geocoding_bistatic(
            sensor_positions_rx=self.position,
            sensor_velocities_rx=self.velocity,
            sensor_positions_tx=self.position,
            sensor_velocities_tx=self.velocity,
            range_times=self.range_times[0],
            look_direction=self.look_direction,
            geodetic_altitude=self.geodetic_altitude,
            frequencies_doppler_centroid=self.doppler_freqs,
            wavelength=self.wavelength,
            initial_guesses=self.initial_guess,
        )

        doppler_residual = doppler_equation_bistatic_residuals(
            sensor_pos_rx=self.position,
            sensor_pos_tx=self.position,
            sensor_vel_rx=self.velocity,
            sensor_vel_tx=self.velocity,
            ground_points=out,
            doppler_freq=self.doppler_freqs,
            wavelength=self.wavelength,
        )
        range_residual = _range_equation_residual_bistatic(
            sensor_pos_rx=self.position,
            sensor_pos_tx=self.position,
            ground_points=out,
            range_time=self.range_times,
        )
        ellipse_residual = _ellipse_equation_residual(ground_points=out)

        self.assertEqual(out.ndim, 1)
        self.assertEqual(out.shape, (3,))
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
        np.testing.assert_allclose(out, self.results, atol=self.tolerance, rtol=0)

    def test_direct_geocoding_bistatic_case0b(self) -> None:
        """Testing direct_geocoding_bistatic, case 0b"""

        # case 0b: 1 pos rx (3,), 1 vel rx (3,), no init guess, 1 pos tx (3,), 1 vel tx (3,), 1 rng time
        out = direct_geocoding_bistatic(
            sensor_positions_rx=self.position,
            sensor_velocities_rx=self.velocity,
            sensor_positions_tx=self.position,
            sensor_velocities_tx=self.velocity,
            range_times=self.range_times[0],
            look_direction=self.look_direction,
            geodetic_altitude=self.geodetic_altitude,
            frequencies_doppler_centroid=self.doppler_freqs,
            wavelength=self.wavelength,
            initial_guesses=None,
        )

        self.assertEqual(out.ndim, 1)
        self.assertEqual(out.shape, (3,))
        np.testing.assert_allclose(out, self.results, atol=self.tolerance, rtol=0)

    def test_direct_geocoding_bistatic_case0c(self) -> None:
        """Testing direct_geocoding_bistatic, case 0c"""

        # case 0c: 1 pos rx (1,3), 1 vel rx (1,3), 1 init guess (1,3), 1 pos tx (3,), 1 vel tx (3,), 1 rng time
        out = direct_geocoding_bistatic(
            sensor_positions_rx=self.position.reshape(1, 3),
            sensor_velocities_rx=self.velocity.reshape(1, 3),
            sensor_positions_tx=self.position,
            sensor_velocities_tx=self.velocity,
            range_times=self.range_times[0],
            look_direction=self.look_direction,
            geodetic_altitude=self.geodetic_altitude,
            frequencies_doppler_centroid=self.doppler_freqs,
            wavelength=self.wavelength,
            initial_guesses=self.initial_guess.reshape(1, 3),
        )

        doppler_residual = doppler_equation_bistatic_residuals(
            sensor_pos_rx=self.position,
            sensor_pos_tx=self.position,
            sensor_vel_rx=self.velocity,
            sensor_vel_tx=self.velocity,
            ground_points=out,
            doppler_freq=self.doppler_freqs,
            wavelength=self.wavelength,
        )
        range_residual = _range_equation_residual_bistatic(
            sensor_pos_rx=self.position,
            sensor_pos_tx=self.position,
            ground_points=out,
            range_time=self.range_times,
        )
        ellipse_residual = _ellipse_equation_residual(ground_points=out)

        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape, (1, 3))
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
        np.testing.assert_allclose(out, self.results.reshape(1, 3), atol=self.tolerance, rtol=0)

    def test_direct_geocoding_bistatic_case0d(self) -> None:
        """Testing direct_geocoding_bistatic, case 0d"""

        # case 0d: 1 pos rx (3,), 1 vel rx (3,), 1 init guess (3,), 1 pos tx (1,3), 1 vel tx (1,3), 1 rng time
        out = direct_geocoding_bistatic(
            sensor_positions_rx=self.position,
            sensor_velocities_rx=self.velocity,
            sensor_positions_tx=self.position.reshape(1, 3),
            sensor_velocities_tx=self.velocity.reshape(1, 3),
            range_times=self.range_times,
            look_direction=self.look_direction,
            geodetic_altitude=self.geodetic_altitude,
            frequencies_doppler_centroid=self.doppler_freqs,
            wavelength=self.wavelength,
            initial_guesses=self.initial_guess,
        )

        doppler_residual = doppler_equation_bistatic_residuals(
            sensor_pos_rx=self.position,
            sensor_pos_tx=self.position,
            sensor_vel_rx=self.velocity,
            sensor_vel_tx=self.velocity,
            ground_points=out,
            doppler_freq=self.doppler_freqs,
            wavelength=self.wavelength,
        )
        range_residual = _range_equation_residual_bistatic(
            sensor_pos_rx=self.position,
            sensor_pos_tx=self.position,
            ground_points=out,
            range_time=self.range_times,
        )
        ellipse_residual = _ellipse_equation_residual(ground_points=out)

        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape, (1, 3))
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
        np.testing.assert_allclose(out, self.results.reshape(1, 3), atol=self.tolerance, rtol=0)

    def test_direct_geocoding_bistatic_case0e(self) -> None:
        """Testing direct_geocoding_bistatic, case 0e"""

        # case 0e: 1 pos rx (3,), 1 vel rx (3,), 1 init guess (1, 3), 1 pos tx (1,3), 1 vel tx (1,3), 1 rng time
        out = direct_geocoding_bistatic(
            sensor_positions_rx=self.position,
            sensor_velocities_rx=self.velocity,
            sensor_positions_tx=self.position.reshape(1, 3),
            sensor_velocities_tx=self.velocity.reshape(1, 3),
            range_times=self.range_times,
            look_direction=self.look_direction,
            geodetic_altitude=self.geodetic_altitude,
            frequencies_doppler_centroid=self.doppler_freqs,
            wavelength=self.wavelength,
            initial_guesses=self.initial_guess.reshape(1, 3),
        )

        doppler_residual = doppler_equation_bistatic_residuals(
            sensor_pos_rx=self.position,
            sensor_pos_tx=self.position,
            sensor_vel_rx=self.velocity,
            sensor_vel_tx=self.velocity,
            ground_points=out,
            doppler_freq=self.doppler_freqs,
            wavelength=self.wavelength,
        )
        range_residual = _range_equation_residual_bistatic(
            sensor_pos_rx=self.position,
            sensor_pos_tx=self.position,
            ground_points=out,
            range_time=self.range_times,
        )
        ellipse_residual = _ellipse_equation_residual(ground_points=out)

        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape, (1, 3))
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
        np.testing.assert_allclose(out, self.results.reshape(1, 3), atol=self.tolerance, rtol=0)

    def test_direct_geocoding_bistatic_case1a(self) -> None:
        """Testing direct_geocoding_bistatic, case 1a"""

        # case 1a: N pos rx (N,3), N vel rx (N,3), 1 init guess (3,), 1 pos tx (3,), 1 vel tx (3,), 1 rng time
        out = direct_geocoding_bistatic(
            sensor_positions_rx=np.full((self.N, 3), self.position),
            sensor_velocities_rx=np.full((self.N, 3), self.velocity),
            sensor_positions_tx=self.position,
            sensor_velocities_tx=self.velocity,
            range_times=self.range_times,
            look_direction=self.look_direction,
            geodetic_altitude=self.geodetic_altitude,
            frequencies_doppler_centroid=self.doppler_freqs,
            wavelength=self.wavelength,
            initial_guesses=self.initial_guess,
        )

        doppler_residual = doppler_equation_bistatic_residuals(
            sensor_pos_rx=np.full((self.N, 3), self.position),
            sensor_pos_tx=np.full((self.N, 3), self.position),
            sensor_vel_rx=np.full((self.N, 3), self.velocity),
            sensor_vel_tx=np.full((self.N, 3), self.velocity),
            ground_points=out,
            doppler_freq=self.doppler_freqs,
            wavelength=self.wavelength,
        )
        range_residual = _range_equation_residual_bistatic(
            sensor_pos_rx=np.full((self.N, 3), self.position),
            sensor_pos_tx=np.full((self.N, 3), self.position),
            ground_points=out,
            range_time=self.range_times,
        )
        ellipse_residual = _ellipse_equation_residual(ground_points=out)

        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape, (self.N, 3))
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
        np.testing.assert_allclose(out, np.full((self.N, 3), self.results), atol=self.tolerance, rtol=0)

    def test_direct_geocoding_bistatic_case1b(self) -> None:
        """Testing direct_geocoding_bistatic, case 1b"""

        # case 1b: N pos rx (N,3), N vel rx (N,3), N init guess (N,3), 1 pos tx (3,), 1 vel tx (3,), 1 rng time
        out = direct_geocoding_bistatic(
            sensor_positions_rx=np.full((self.N, 3), self.position),
            sensor_velocities_rx=np.full((self.N, 3), self.velocity),
            sensor_positions_tx=self.position,
            sensor_velocities_tx=self.velocity,
            range_times=self.range_times,
            look_direction=self.look_direction,
            geodetic_altitude=self.geodetic_altitude,
            frequencies_doppler_centroid=self.doppler_freqs,
            wavelength=self.wavelength,
            initial_guesses=np.full((self.N, 3), self.initial_guess),
        )

        doppler_residual = doppler_equation_bistatic_residuals(
            sensor_pos_rx=np.full((self.N, 3), self.position),
            sensor_pos_tx=np.full((self.N, 3), self.position),
            sensor_vel_rx=np.full((self.N, 3), self.velocity),
            sensor_vel_tx=np.full((self.N, 3), self.velocity),
            ground_points=out,
            doppler_freq=self.doppler_freqs,
            wavelength=self.wavelength,
        )
        range_residual = _range_equation_residual_bistatic(
            sensor_pos_rx=np.full((self.N, 3), self.position),
            sensor_pos_tx=np.full((self.N, 3), self.position),
            ground_points=out,
            range_time=self.range_times,
        )
        ellipse_residual = _ellipse_equation_residual(ground_points=out)

        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape, (self.N, 3))
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
        np.testing.assert_allclose(out, np.full((self.N, 3), self.results), atol=self.tolerance, rtol=0)

    def test_direct_geocoding_bistatic_case1c(self) -> None:
        """Testing direct_geocoding_bistatic, case 1c"""

        # case 1c: N pos rx (N,3), N vel rx (N,3), N init guess (N,3), 1 pos tx (1,3), 1 vel tx (1,3), 1 rng time
        out = direct_geocoding_bistatic(
            sensor_positions_rx=np.full((self.N, 3), self.position),
            sensor_velocities_rx=np.full((self.N, 3), self.velocity),
            sensor_positions_tx=self.position.reshape(1, 3),
            sensor_velocities_tx=self.velocity.reshape(1, 3),
            range_times=self.range_times,
            look_direction=self.look_direction,
            geodetic_altitude=self.geodetic_altitude,
            frequencies_doppler_centroid=self.doppler_freqs,
            wavelength=self.wavelength,
            initial_guesses=np.full((self.N, 3), self.initial_guess),
        )

        doppler_residual = doppler_equation_bistatic_residuals(
            sensor_pos_rx=np.full((self.N, 3), self.position),
            sensor_pos_tx=np.full((self.N, 3), self.position),
            sensor_vel_rx=np.full((self.N, 3), self.velocity),
            sensor_vel_tx=np.full((self.N, 3), self.velocity),
            ground_points=out,
            doppler_freq=self.doppler_freqs,
            wavelength=self.wavelength,
        )
        range_residual = _range_equation_residual_bistatic(
            sensor_pos_rx=np.full((self.N, 3), self.position),
            sensor_pos_tx=np.full((self.N, 3), self.position),
            ground_points=out,
            range_time=self.range_times,
        )
        ellipse_residual = _ellipse_equation_residual(ground_points=out)

        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape, (self.N, 3))
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
        np.testing.assert_allclose(out, np.full((self.N, 3), self.results), atol=self.tolerance, rtol=0)

    def test_direct_geocoding_bistatic_case2a(self) -> None:
        """Testing direct_geocoding_bistatic, case 2a"""

        # case 2a: 1 pos rx (3,), 1 vel rx (3,), 1 init guess (3,), M pos tx (M,3), M vel tx (M,3), M rng times
        out = direct_geocoding_bistatic(
            sensor_positions_rx=self.position,
            sensor_velocities_rx=self.velocity,
            sensor_positions_tx=np.full((self.M, 3), self.position),
            sensor_velocities_tx=np.full((self.M, 3), self.velocity),
            range_times=np.repeat(self.range_times[0], self.M),
            look_direction=self.look_direction,
            geodetic_altitude=self.geodetic_altitude,
            frequencies_doppler_centroid=self.doppler_freqs,
            wavelength=self.wavelength,
            initial_guesses=self.initial_guess,
        )

        doppler_residual = doppler_equation_bistatic_residuals(
            sensor_pos_rx=self.position,
            sensor_pos_tx=self.position,
            sensor_vel_rx=self.velocity,
            sensor_vel_tx=self.velocity,
            ground_points=out,
            doppler_freq=self.doppler_freqs,
            wavelength=self.wavelength,
        )
        range_residual = _range_equation_residual_bistatic(
            sensor_pos_rx=self.position,
            sensor_pos_tx=self.position,
            ground_points=out,
            range_time=self.range_times,
        )
        ellipse_residual = _ellipse_equation_residual(ground_points=out)

        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape, (self.M, 3))
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
        np.testing.assert_allclose(out, np.full((self.M, 3), self.results), atol=self.tolerance, rtol=0)

    def test_direct_geocoding_bistatic_case2b(self) -> None:
        """Testing direct_geocoding_bistatic, case 2b"""

        # case 2b: 1 pos rx (3,), 1 vel rx (3,), 1 init guess (3,),
        # M pos tx (M,3), M vel tx (M,3), M rng times, M doppler freqs
        out = direct_geocoding_bistatic(
            sensor_positions_rx=self.position,
            sensor_velocities_rx=self.velocity,
            sensor_positions_tx=np.full((self.M, 3), self.position),
            sensor_velocities_tx=np.full((self.M, 3), self.velocity),
            range_times=np.repeat(self.range_times[0], self.M),
            look_direction=self.look_direction,
            geodetic_altitude=self.geodetic_altitude,
            frequencies_doppler_centroid=np.repeat(self.doppler_freqs, self.M),
            wavelength=self.wavelength,
            initial_guesses=self.initial_guess,
        )

        doppler_residual = doppler_equation_bistatic_residuals(
            sensor_pos_rx=self.position,
            sensor_pos_tx=self.position,
            sensor_vel_rx=self.velocity,
            sensor_vel_tx=self.velocity,
            ground_points=out,
            doppler_freq=self.doppler_freqs,
            wavelength=self.wavelength,
        )
        range_residual = _range_equation_residual_bistatic(
            sensor_pos_rx=self.position,
            sensor_pos_tx=self.position,
            ground_points=out,
            range_time=self.range_times,
        )
        ellipse_residual = _ellipse_equation_residual(ground_points=out)

        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape, (self.M, 3))
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
        np.testing.assert_allclose(out, np.full((self.M, 3), self.results), atol=self.tolerance, rtol=0)

    def test_direct_geocoding_bistatic_case2c(self) -> None:
        """Testing direct_geocoding_bistatic, case 2c"""

        # case 2c: 1 pos rx (1,3), 1 vel rx (1,3), 1 init guess (1,3), M pos tx (M,3), M vel tx (M,3), M rng times
        out = direct_geocoding_bistatic(
            sensor_positions_rx=self.position.reshape(1, 3),
            sensor_velocities_rx=self.velocity.reshape(1, 3),
            sensor_positions_tx=np.full((self.M, 3), self.position),
            sensor_velocities_tx=np.full((self.M, 3), self.velocity),
            range_times=np.repeat(self.range_times[0], self.M),
            look_direction=self.look_direction,
            geodetic_altitude=self.geodetic_altitude,
            frequencies_doppler_centroid=self.doppler_freqs,
            wavelength=self.wavelength,
            initial_guesses=self.initial_guess.reshape(1, 3),
        )

        doppler_residual = doppler_equation_bistatic_residuals(
            sensor_pos_rx=self.position,
            sensor_pos_tx=self.position,
            sensor_vel_rx=self.velocity,
            sensor_vel_tx=self.velocity,
            ground_points=out,
            doppler_freq=self.doppler_freqs,
            wavelength=self.wavelength,
        )
        range_residual = _range_equation_residual_bistatic(
            sensor_pos_rx=self.position,
            sensor_pos_tx=self.position,
            ground_points=out,
            range_time=self.range_times,
        )
        ellipse_residual = _ellipse_equation_residual(ground_points=out)

        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape, (self.M, 3))
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
        np.testing.assert_allclose(out, np.full((self.M, 3), self.results), atol=self.tolerance, rtol=0)

    def test_direct_geocoding_bistatic_case3a(self) -> None:
        """Testing direct_geocoding_bistatic, case 3a"""

        # case 3a: N pos rx (N,3), N vel rx (N,3), 1 init guess (3,), M pos tx (M,3), M vel tx (M,3), M rng times
        out = direct_geocoding_bistatic(
            sensor_positions_rx=np.full((self.N, 3), self.position),
            sensor_velocities_rx=np.full((self.N, 3), self.velocity),
            sensor_positions_tx=np.full((self.M, 3), self.position),
            sensor_velocities_tx=np.full((self.M, 3), self.velocity),
            range_times=np.repeat(self.range_times[0], self.M),
            look_direction=self.look_direction,
            geodetic_altitude=self.geodetic_altitude,
            frequencies_doppler_centroid=self.doppler_freqs,
            wavelength=self.wavelength,
            initial_guesses=self.initial_guess,
        )

        doppler_residual = doppler_equation_bistatic_residuals(
            sensor_pos_rx=np.full((self.N, 3), self.position),
            sensor_pos_tx=np.full((self.N, 3), self.position),
            sensor_vel_rx=np.full((self.N, 3), self.velocity),
            sensor_vel_tx=np.full((self.N, 3), self.velocity),
            ground_points=out[:, 0, :],
            doppler_freq=self.doppler_freqs,
            wavelength=self.wavelength,
        )
        range_residual = _range_equation_residual_bistatic(
            sensor_pos_rx=np.full((self.N, 3), self.position),
            sensor_pos_tx=np.full((self.N, 3), self.position),
            ground_points=out[:, 0, :],
            range_time=self.range_times,
        )
        ellipse_residual = _ellipse_equation_residual(ground_points=out[:, 0, :])

        self.assertEqual(out.ndim, 3)
        self.assertEqual(out.shape, (self.N, self.M, 3))
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
        np.testing.assert_allclose(out, np.full((self.N, self.M, 3), self.results), atol=self.tolerance, rtol=0)

    def test_direct_geocoding_bistatic_case3b(self) -> None:
        """Testing direct_geocoding_bistatic, case 3b"""

        # case 3b: N pos rx (N,3), N vel rx (N,3), 1 init guess (1,3), M pos tx (M,3), M vel tx (M,3), M rng times
        out = direct_geocoding_bistatic(
            sensor_positions_rx=np.full((self.N, 3), self.position),
            sensor_velocities_rx=np.full((self.N, 3), self.velocity),
            sensor_positions_tx=np.full((self.M, 3), self.position),
            sensor_velocities_tx=np.full((self.M, 3), self.velocity),
            range_times=np.repeat(self.range_times[0], self.M),
            look_direction=self.look_direction,
            geodetic_altitude=self.geodetic_altitude,
            frequencies_doppler_centroid=self.doppler_freqs,
            wavelength=self.wavelength,
            initial_guesses=self.initial_guess.reshape(1, 3),
        )

        doppler_residual = doppler_equation_bistatic_residuals(
            sensor_pos_rx=np.full((self.N, 3), self.position),
            sensor_pos_tx=np.full((self.N, 3), self.position),
            sensor_vel_rx=np.full((self.N, 3), self.velocity),
            sensor_vel_tx=np.full((self.N, 3), self.velocity),
            ground_points=out[:, 0, :],
            doppler_freq=self.doppler_freqs,
            wavelength=self.wavelength,
        )
        range_residual = _range_equation_residual_bistatic(
            sensor_pos_rx=np.full((self.N, 3), self.position),
            sensor_pos_tx=np.full((self.N, 3), self.position),
            ground_points=out[:, 0, :],
            range_time=self.range_times,
        )
        ellipse_residual = _ellipse_equation_residual(ground_points=out[:, 0, :])

        self.assertEqual(out.ndim, 3)
        self.assertEqual(out.shape, (self.N, self.M, 3))
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
        np.testing.assert_allclose(out, np.full((self.N, self.M, 3), self.results), atol=self.tolerance, rtol=0)

    def test_direct_geocoding_bistatic_case3c(self) -> None:
        """Testing direct_geocoding_bistatic, case 3c"""

        # case 3c: N pos rx (N,3), N vel rx (N,3), 1 init guess (N,3),
        # M pos tx (M,3), M vel tx (M,3), M rng times, M doppler freqs
        out = direct_geocoding_bistatic(
            sensor_positions_rx=np.full((self.N, 3), self.position),
            sensor_velocities_rx=np.full((self.N, 3), self.velocity),
            sensor_positions_tx=np.full((self.M, 3), self.position),
            sensor_velocities_tx=np.full((self.M, 3), self.velocity),
            range_times=np.repeat(self.range_times[0], self.M),
            look_direction=self.look_direction,
            geodetic_altitude=self.geodetic_altitude,
            frequencies_doppler_centroid=np.repeat(self.doppler_freqs, self.M),
            wavelength=self.wavelength,
            initial_guesses=np.full((self.N, 3), self.initial_guess),
        )

        doppler_residual = doppler_equation_bistatic_residuals(
            sensor_pos_rx=np.full((self.N, 3), self.position),
            sensor_pos_tx=np.full((self.N, 3), self.position),
            sensor_vel_rx=np.full((self.N, 3), self.velocity),
            sensor_vel_tx=np.full((self.N, 3), self.velocity),
            ground_points=out[:, 0, :],
            doppler_freq=self.doppler_freqs,
            wavelength=self.wavelength,
        )
        range_residual = _range_equation_residual_bistatic(
            sensor_pos_rx=np.full((self.N, 3), self.position),
            sensor_pos_tx=np.full((self.N, 3), self.position),
            ground_points=out[:, 0, :],
            range_time=self.range_times,
        )
        ellipse_residual = _ellipse_equation_residual(ground_points=out[:, 0, :])

        self.assertEqual(out.ndim, 3)
        self.assertEqual(out.shape, (self.N, self.M, 3))
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
        np.testing.assert_allclose(out, np.full((self.N, self.M, 3), self.results), atol=self.tolerance, rtol=0)


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
            geodetic_altitude=self.geodetic_altitude,
            frequencies_doppler_centroid=self.doppler_freqs,
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
            geodetic_altitude=self.geodetic_altitude,
            frequencies_doppler_centroid=self.doppler_freqs,
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
            geodetic_altitude=self.geodetic_altitude,
            frequencies_doppler_centroid=self.doppler_freqs,
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
            geodetic_altitude=self.geodetic_altitude,
            frequencies_doppler_centroid=self.doppler_freqs,
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
            geodetic_altitude=self.geodetic_altitude,
            frequencies_doppler_centroid=self.doppler_freqs,
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
            geodetic_altitude=self.geodetic_altitude,
            frequencies_doppler_centroid=self.doppler_freqs,
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
            geodetic_altitude=self.geodetic_altitude,
            frequencies_doppler_centroid=self.doppler_freqs,
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
            geodetic_altitude=self.geodetic_altitude,
            frequencies_doppler_centroid=self.doppler_freqs,
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
            geodetic_altitude=self.geodetic_altitude,
            frequencies_doppler_centroid=np.repeat(self.doppler_freqs, self.M),
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
            geodetic_altitude=self.geodetic_altitude,
            frequencies_doppler_centroid=self.doppler_freqs,
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
            geodetic_altitude=self.geodetic_altitude,
            frequencies_doppler_centroid=np.repeat(self.doppler_freqs, self.M),
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
            geodetic_altitude=self.geodetic_altitude,
            frequencies_doppler_centroid=self.doppler_freqs,
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
            geodetic_altitude=self.geodetic_altitude,
            frequencies_doppler_centroid=np.repeat(self.doppler_freqs, self.M),
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
        with self.assertRaises(AmbiguousInputCorrelation):
            direct_geocoding_bistatic_core(
                sensor_positions_rx=np.full((self.N, 3), self.position),
                sensor_velocities_rx=np.full((self.N, 3), self.velocity),
                initial_guesses=np.full((self.N, 3), self.initial_guess),
                sensor_positions_tx=np.full((self.M, 3), self.position),
                sensor_velocities_tx=np.full((self.M, 3), self.velocity),
                range_times=np.repeat(self.range_times[0], self.M),
                geodetic_altitude=self.geodetic_altitude,
                frequencies_doppler_centroid=np.repeat(self.doppler_freqs, self.N),
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
        out = _newton_for_direct_geocoding_bistatic(
            sensor_positions_rx=self.position,
            sensor_velocities_rx=self.velocity,
            initial_guesses=self.initial_guess,
            sensor_position_tx=self.position,
            sensor_velocity_tx=self.velocity,
            range_time=self.range_times,
            frequency_doppler_centroid=self.doppler_freqs,
            wavelength=self.wavelength,
            geodetic_altitude=self.geodetic_altitude,
        )

        self.assertTrue(out.ndim == 1)
        np.testing.assert_allclose(out, self.results, atol=self.tolerance, rtol=0)

    def test_newton_for_direct_geocoding_bistatic_case1(self) -> None:
        """Testing Newton for direct geocoding bistatic, case 1"""

        # case 1: 1 pos (1, 3), 1 vel (1, 3), 1 init guess (1, 3), 1 rng time
        out = _newton_for_direct_geocoding_bistatic(
            sensor_positions_rx=self.position.reshape(1, 3),
            sensor_velocities_rx=self.velocity.reshape(1, 3),
            initial_guesses=self.initial_guess.reshape(1, 3),
            sensor_position_tx=self.position,
            sensor_velocity_tx=self.velocity,
            range_time=self.range_times,
            frequency_doppler_centroid=self.doppler_freqs,
            wavelength=self.wavelength,
            geodetic_altitude=self.geodetic_altitude,
        )

        self.assertTrue(out.ndim == 2)
        np.testing.assert_allclose(out, self.results.reshape(1, 3), atol=self.tolerance, rtol=0)

    def test_newton_for_direct_geocoding_bistatic_case2a(self) -> None:
        """Testing Newton for direct geocoding bistatic, case 2a"""

        # case 2a: N pos (N, 3), N vel (N, 3), 1 init guess (3,), 1 rng time
        out = _newton_for_direct_geocoding_bistatic(
            sensor_positions_rx=np.full((self.N, 3), self.position),
            sensor_velocities_rx=np.full((self.N, 3), self.velocity),
            initial_guesses=self.initial_guess,
            sensor_position_tx=self.position,
            sensor_velocity_tx=self.velocity,
            range_time=self.range_times,
            frequency_doppler_centroid=self.doppler_freqs,
            wavelength=self.wavelength,
            geodetic_altitude=self.geodetic_altitude,
        )

        self.assertTrue(out.ndim == 2)
        np.testing.assert_allclose(out, np.full((self.N, 3), self.results), atol=self.tolerance, rtol=0)

    def test_newton_for_direct_geocoding_bistatic_case2b(self) -> None:
        """Testing Newton for direct geocoding bistatic, case 2b"""

        # case 2b: N pos (N, 3), N vel (N, 3), 1 init guess (1, 3), 1 rng time
        out = _newton_for_direct_geocoding_bistatic(
            sensor_positions_rx=np.full((self.N, 3), self.position),
            sensor_velocities_rx=np.full((self.N, 3), self.velocity),
            initial_guesses=self.initial_guess.reshape(1, 3),
            sensor_position_tx=self.position,
            sensor_velocity_tx=self.velocity,
            range_time=self.range_times,
            frequency_doppler_centroid=self.doppler_freqs,
            wavelength=self.wavelength,
            geodetic_altitude=self.geodetic_altitude,
        )

        self.assertTrue(out.ndim == 2)
        np.testing.assert_allclose(out, np.full((self.N, 3), self.results), atol=self.tolerance, rtol=0)

    def test_newton_for_direct_geocoding_bistatic_case3(self) -> None:
        """Testing Newton for direct geocoding bistatic, case 3"""

        # case 3: N pos (N, 3), N vel (N, 3), N init guess (N, 3), 1 rng time
        out = _newton_for_direct_geocoding_bistatic(
            sensor_positions_rx=np.full((self.N, 3), self.position),
            sensor_velocities_rx=np.full((self.N, 3), self.velocity),
            initial_guesses=self.initial_guess.reshape(1, 3),
            sensor_position_tx=self.position,
            sensor_velocity_tx=self.velocity,
            range_time=self.range_times,
            frequency_doppler_centroid=self.doppler_freqs,
            wavelength=self.wavelength,
            geodetic_altitude=self.geodetic_altitude,
        )

        self.assertTrue(out.ndim == 2)
        np.testing.assert_allclose(out, np.full((self.N, 3), self.results), atol=self.tolerance, rtol=0)


if __name__ == "__main__":
    unittest.main()
