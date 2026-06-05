# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for geometry/geocoding/direct_geocoding.py and direct_geocoding_core.py bistatic functionalities"""

from __future__ import annotations

import numpy as np
import pytest
from numpy import typing as npt
from scipy.constants import speed_of_light

from perseo_core.geometry.doppler import doppler_equation_bistatic_residuals
from perseo_core.geometry.ellipsoid import WGS84
from perseo_core.geometry.geocoding.direct_geocoding import direct_geocoding_bistatic
from perseo_core.geometry.geocoding.direct_geocoding_core import (
    _direct_geocoding_bistatic_newton,
    _ellipse_equation,
    direct_geocoding_bistatic_core,
)


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

    return _ellipse_equation(ground_points, r_ee2, r_ep2)


def _reshape_output(res: float, shape: tuple[int, ...]) -> npt.NDArray[np.floating]:
    return np.full(shape, res)


class TestDirectGeocodingBistatic:
    """Testing direct_geocoding_bistatic with consolidated parametrize."""

    @pytest.fixture(autouse=True)
    def setup(self, direct_geocoding_test_data: dict) -> None:
        self.position = direct_geocoding_test_data["sensor_position"]
        self.velocity = direct_geocoding_test_data["sensor_velocity"]
        self.initial_guess = direct_geocoding_test_data["initial_guess"]
        self.range_times = direct_geocoding_test_data["range_time"]
        self.doppler_frequency = direct_geocoding_test_data["doppler_frequency"]
        self.altitude = direct_geocoding_test_data["geodetic_altitude"]
        self.look_direction = direct_geocoding_test_data["look_direction"]
        self.wavelength = direct_geocoding_test_data["wavelength"]
        self.N = direct_geocoding_test_data["az_reps"]
        self.M = direct_geocoding_test_data["rng_reps"]
        self.tolerance = direct_geocoding_test_data["tolerance"]
        self.residual_tolerance = direct_geocoding_test_data["residual_tolerance"]
        self.expected_results = direct_geocoding_test_data["expected_ground_points"]

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
            doppler_frequency=self.doppler_frequency,
            wavelength=self.wavelength,
        )
        range_residual = _range_equation_residual_bistatic(
            sensor_pos_rx=sensor_positions_rx,
            sensor_pos_tx=sensor_positions_tx,
            ground_points=out,
            range_time=range_times,
        )
        ellipse_residual = _ellipse_equation_residual(ground_points=out)

        np.testing.assert_allclose(
            doppler_residual, 0, atol=self.residual_tolerance["atol"], rtol=self.residual_tolerance["rtol"]
        )
        np.testing.assert_allclose(
            range_residual, 0, atol=self.residual_tolerance["atol"], rtol=self.residual_tolerance["rtol"]
        )
        np.testing.assert_allclose(
            ellipse_residual, 0, atol=self.residual_tolerance["atol"], rtol=self.residual_tolerance["rtol"]
        )

    def _check_output(self, expected_shape: tuple[int, ...], out: npt.NDArray[np.floating]) -> None:
        assert out.shape == expected_shape
        np.testing.assert_allclose(
            out,
            _reshape_output(self.expected_results, expected_shape),
            atol=self.tolerance["atol"],
            rtol=self.tolerance["rtol"],
        )

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(
                {
                    "name": "1-point rx_pos=1d rx_vel=1d tx_pos=1d tx_vel=1d init=none",
                    "sensor_positions_rx": "position",
                    "sensor_velocities_rx": "velocity",
                    "sensor_positions_tx": "position",
                    "sensor_velocities_tx": "velocity",
                    "range_times": "range_times_0",
                    "doppler_frequencies": "doppler_frequency",
                    "initial_guesses": None,
                    "expected_shape": (3,),
                },
                id="1-point-init-none",
            ),
            pytest.param(
                {
                    "name": "1-point rx_pos=1d rx_vel=1d tx_pos=1d tx_vel=1d init=1d",
                    "sensor_positions_rx": "position",
                    "sensor_velocities_rx": "velocity",
                    "sensor_positions_tx": "position",
                    "sensor_velocities_tx": "velocity",
                    "range_times": "range_times_0",
                    "doppler_frequencies": "doppler_frequency",
                    "initial_guesses": "initial_guess",
                    "expected_shape": (3,),
                },
                id="1-point-init-1d",
            ),
            pytest.param(
                {
                    "name": "1-point rx_pos=1d rx_vel=1d tx_pos=1d tx_vel=1d init=2d",
                    "sensor_positions_rx": "position",
                    "sensor_velocities_rx": "velocity",
                    "sensor_positions_tx": "position",
                    "sensor_velocities_tx": "velocity",
                    "range_times": "range_times_0",
                    "doppler_frequencies": "doppler_frequency",
                    "initial_guesses": "initial_guess_1_3",
                    "expected_shape": (3,),
                },
                id="1-point-init-2d",
            ),
            pytest.param(
                {
                    "name": "1-point rx_pos=1d rx_vel=1d tx_pos=2d tx_vel=2d init=none",
                    "sensor_positions_rx": "position",
                    "sensor_velocities_rx": "velocity",
                    "sensor_positions_tx": "position_1_3",
                    "sensor_velocities_tx": "velocity_1_3",
                    "range_times": "range_times_0",
                    "doppler_frequencies": "doppler_frequency",
                    "initial_guesses": None,
                    "expected_shape": (1, 3),
                },
                id="1-point-tx-2d-init-none",
            ),
            pytest.param(
                {
                    "name": "1-point rx_pos=1d rx_vel=1d tx_pos=2d tx_vel=2d init=1d",
                    "sensor_positions_rx": "position",
                    "sensor_velocities_rx": "velocity",
                    "sensor_positions_tx": "position_1_3",
                    "sensor_velocities_tx": "velocity_1_3",
                    "range_times": "range_times_0",
                    "doppler_frequencies": "doppler_frequency",
                    "initial_guesses": "initial_guess",
                    "expected_shape": (1, 3),
                },
                id="1-point-tx-2d-init-1d",
            ),
            pytest.param(
                {
                    "name": "1-point rx_pos=1d rx_vel=1d tx_pos=2d tx_vel=2d init=2d",
                    "sensor_positions_rx": "position",
                    "sensor_velocities_rx": "velocity",
                    "sensor_positions_tx": "position_1_3",
                    "sensor_velocities_tx": "velocity_1_3",
                    "range_times": "range_times_0",
                    "doppler_frequencies": "doppler_frequency",
                    "initial_guesses": "initial_guess_1_3",
                    "expected_shape": (1, 3),
                },
                id="1-point-tx-2d-init-2d",
            ),
            pytest.param(
                {
                    "name": "1-point rx_pos=2d rx_vel=2d tx_pos=1d tx_vel=1d init=none",
                    "sensor_positions_rx": "position_1_3",
                    "sensor_velocities_rx": "velocity_1_3",
                    "sensor_positions_tx": "position",
                    "sensor_velocities_tx": "velocity",
                    "range_times": "range_times_0",
                    "doppler_frequencies": "doppler_frequency",
                    "initial_guesses": None,
                    "expected_shape": (1, 3),
                },
                id="1-point-rx-2d-tx-1d-init-none",
            ),
            pytest.param(
                {
                    "name": "1-point rx_pos=2d rx_vel=2d tx_pos=1d tx_vel=1d init=1d",
                    "sensor_positions_rx": "position_1_3",
                    "sensor_velocities_rx": "velocity_1_3",
                    "sensor_positions_tx": "position",
                    "sensor_velocities_tx": "velocity",
                    "range_times": "range_times_0",
                    "doppler_frequencies": "doppler_frequency",
                    "initial_guesses": "initial_guess",
                    "expected_shape": (1, 3),
                },
                id="1-point-rx-2d-tx-1d-init-1d",
            ),
            pytest.param(
                {
                    "name": "1-point rx_pos=2d rx_vel=2d tx_pos=2d tx_vel=2d init=none",
                    "sensor_positions_rx": "position_1_3",
                    "sensor_velocities_rx": "velocity_1_3",
                    "sensor_positions_tx": "position_1_3",
                    "sensor_velocities_tx": "velocity_1_3",
                    "range_times": "range_times_0",
                    "doppler_frequencies": "doppler_frequency",
                    "initial_guesses": None,
                    "expected_shape": (1, 3),
                },
                id="1-point-rx-2d-tx-2d-init-none",
            ),
            pytest.param(
                {
                    "name": "1-point rx_pos=2d rx_vel=2d tx_pos=2d tx_vel=2d init=2d",
                    "sensor_positions_rx": "position_1_3",
                    "sensor_velocities_rx": "velocity_1_3",
                    "sensor_positions_tx": "position_1_3",
                    "sensor_velocities_tx": "velocity_1_3",
                    "range_times": "range_times_0",
                    "doppler_frequencies": "doppler_frequency",
                    "initial_guesses": "initial_guess_1_3",
                    "expected_shape": (1, 3),
                },
                id="1-point-rx-2d-tx-2d-init-2d",
            ),
        ],
    )
    def test_direct_geocoding_bistatic_1_point(self, case: dict) -> None:
        """Testing direct_geocoding_bistatic with 1-point cases."""
        value_map = {
            "position": self.position,
            "position_1_3": self.position.reshape(1, 3),
            "velocity": self.velocity,
            "velocity_1_3": self.velocity.reshape(1, 3),
            "initial_guess": self.initial_guess,
            "initial_guess_1_3": self.initial_guess.reshape(1, 3),
            "range_times_0": self.range_times[0],
            "doppler_frequency": self.doppler_frequency,
        }

        sensor_positions_rx = value_map[case["sensor_positions_rx"]]
        sensor_velocities_rx = value_map[case["sensor_velocities_rx"]]
        sensor_positions_tx = value_map[case["sensor_positions_tx"]]
        sensor_velocities_tx = value_map[case["sensor_velocities_tx"]]
        range_times = value_map[case["range_times"]]
        doppler_frequencies = value_map[case["doppler_frequencies"]]
        initial_guesses = value_map.get(case["initial_guesses"], case["initial_guesses"])

        out = direct_geocoding_bistatic(
            sensor_positions_rx=sensor_positions_rx,
            sensor_velocities_rx=sensor_velocities_rx,
            sensor_positions_tx=sensor_positions_tx,
            sensor_velocities_tx=sensor_velocities_tx,
            range_times=range_times,
            look_direction=self.look_direction,
            altitude=self.altitude,
            doppler_frequencies=doppler_frequencies,
            wavelength=self.wavelength,
            initial_guesses=initial_guesses,
        )

        self._check_residuals(
            sensor_positions_rx,
            sensor_velocities_rx,
            sensor_positions_tx,
            sensor_velocities_tx,
            range_times,
            out,
        )
        self._check_output(case["expected_shape"], out)

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(
                {
                    "name": "N-points tx_pos=1d tx_vel=1d init=1d",
                    "sensor_positions_rx": "position_N_3",
                    "sensor_velocities_rx": "velocity_N_3",
                    "sensor_positions_tx": "position",
                    "sensor_velocities_tx": "velocity",
                    "range_times": "range_times",
                    "doppler_frequencies": "doppler_frequency",
                    "initial_guesses": "initial_guess",
                    "expected_shape": ("N", 3),
                },
                id="N-points-tx-1d-init-1d",
            ),
            pytest.param(
                {
                    "name": "N-points tx_pos=1d tx_vel=1d init=2d",
                    "sensor_positions_rx": "position_N_3",
                    "sensor_velocities_rx": "velocity_N_3",
                    "sensor_positions_tx": "position",
                    "sensor_velocities_tx": "velocity",
                    "range_times": "range_times",
                    "doppler_frequencies": "doppler_frequency",
                    "initial_guesses": "initial_guess_N_3",
                    "expected_shape": ("N", 3),
                },
                id="N-points-tx-1d-init-2d",
            ),
            pytest.param(
                {
                    "name": "N-points tx_pos=2d tx_vel=2d init=1d",
                    "sensor_positions_rx": "position_N_3",
                    "sensor_velocities_rx": "velocity_N_3",
                    "sensor_positions_tx": "position_1_3",
                    "sensor_velocities_tx": "velocity_1_3",
                    "range_times": "range_times",
                    "doppler_frequencies": "doppler_frequency",
                    "initial_guesses": "initial_guess",
                    "expected_shape": ("N", 3),
                },
                id="N-points-tx-2d-init-1d",
            ),
            pytest.param(
                {
                    "name": "N-points tx_pos=2d tx_vel=2d init=2d",
                    "sensor_positions_rx": "position_N_3",
                    "sensor_velocities_rx": "velocity_N_3",
                    "sensor_positions_tx": "position_1_3",
                    "sensor_velocities_tx": "velocity_1_3",
                    "range_times": "range_times",
                    "doppler_frequencies": "doppler_frequency",
                    "initial_guesses": "initial_guess_1_3",
                    "expected_shape": ("N", 3),
                },
                id="N-points-tx-2d-init-2d",
            ),
        ],
    )
    def test_direct_geocoding_bistatic_n_points(self, case: dict) -> None:
        """Testing direct_geocoding_bistatic with N-point cases."""
        N = self.N
        value_map = {
            "position": self.position,
            "position_N_3": np.full((N, 3), self.position),
            "position_1_3": self.position.reshape(1, 3),
            "velocity": self.velocity,
            "velocity_N_3": np.full((N, 3), self.velocity),
            "velocity_1_3": self.velocity.reshape(1, 3),
            "initial_guess": self.initial_guess,
            "initial_guess_N_3": np.full((N, 3), self.initial_guess),
            "initial_guess_1_3": self.initial_guess.reshape(1, 3),
            "range_times": self.range_times,
            "doppler_frequency": self.doppler_frequency,
        }

        sensor_positions_rx = value_map[case["sensor_positions_rx"]]
        sensor_velocities_rx = value_map[case["sensor_velocities_rx"]]
        sensor_positions_tx = value_map[case["sensor_positions_tx"]]
        sensor_velocities_tx = value_map[case["sensor_velocities_tx"]]
        range_times = value_map[case["range_times"]]
        doppler_frequencies = value_map[case["doppler_frequencies"]]
        initial_guesses = value_map[case["initial_guesses"]]

        out = direct_geocoding_bistatic(
            sensor_positions_rx=sensor_positions_rx,
            sensor_velocities_rx=sensor_velocities_rx,
            sensor_positions_tx=sensor_positions_tx,
            sensor_velocities_tx=sensor_velocities_tx,
            range_times=range_times,
            look_direction=self.look_direction,
            altitude=self.altitude,
            doppler_frequencies=doppler_frequencies,
            wavelength=self.wavelength,
            initial_guesses=initial_guesses,
        )

        self._check_residuals(
            sensor_positions_rx,
            sensor_velocities_rx,
            sensor_positions_tx,
            sensor_velocities_tx,
            range_times,
            out,
        )
        self._check_output((N, 3), out)

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(
                {
                    "name": "M-points rx_pos=1d rx_vel=1d init=1d doppler=scalar",
                    "sensor_positions_rx": "position",
                    "sensor_velocities_rx": "velocity",
                    "sensor_positions_tx": "position_M_3",
                    "sensor_velocities_tx": "velocity_M_3",
                    "range_times": "range_times_M",
                    "doppler_frequencies": "doppler_frequency",
                    "initial_guesses": "initial_guess",
                    "expected_shape": ("M", 3),
                },
                id="M-points-init-1d-doppler-scalar",
            ),
            pytest.param(
                {
                    "name": "M-points rx_pos=1d rx_vel=1d init=1d doppler=array",
                    "sensor_positions_rx": "position",
                    "sensor_velocities_rx": "velocity",
                    "sensor_positions_tx": "position_M_3",
                    "sensor_velocities_tx": "velocity_M_3",
                    "range_times": "range_times_M",
                    "doppler_frequencies": "doppler_frequency_M",
                    "initial_guesses": "initial_guess",
                    "expected_shape": ("M", 3),
                },
                id="M-points-init-1d-doppler-array",
            ),
            pytest.param(
                {
                    "name": "M-points rx_pos=1d rx_vel=1d init=2d doppler=scalar",
                    "sensor_positions_rx": "position",
                    "sensor_velocities_rx": "velocity",
                    "sensor_positions_tx": "position_M_3",
                    "sensor_velocities_tx": "velocity_M_3",
                    "range_times": "range_times_M",
                    "doppler_frequencies": "doppler_frequency",
                    "initial_guesses": "initial_guess_1_3",
                    "expected_shape": ("M", 3),
                },
                id="M-points-init-2d-doppler-scalar",
            ),
            pytest.param(
                {
                    "name": "M-points rx_pos=2d rx_vel=2d init=1d doppler=scalar",
                    "sensor_positions_rx": "position_1_3",
                    "sensor_velocities_rx": "velocity_1_3",
                    "sensor_positions_tx": "position_M_3",
                    "sensor_velocities_tx": "velocity_M_3",
                    "range_times": "range_times_M",
                    "doppler_frequencies": "doppler_frequency",
                    "initial_guesses": "initial_guess",
                    "expected_shape": ("M", 3),
                },
                id="M-points-rx-2d-init-1d-doppler-scalar",
            ),
            pytest.param(
                {
                    "name": "M-points rx_pos=1d rx_vel=1d init=2d doppler=scalar",
                    "sensor_positions_rx": "position",
                    "sensor_velocities_rx": "velocity",
                    "sensor_positions_tx": "position_M_3",
                    "sensor_velocities_tx": "velocity_M_3",
                    "range_times": "range_times_M",
                    "doppler_frequencies": "doppler_frequency",
                    "initial_guesses": "initial_guess_1_3",
                    "expected_shape": ("M", 3),
                },
                id="M-points-rx-1d-init-2d-doppler-scalar",
            ),
            pytest.param(
                {
                    "name": "M-points rx_pos=1d rx_vel=1d init=2d doppler=array",
                    "sensor_positions_rx": "position",
                    "sensor_velocities_rx": "velocity",
                    "sensor_positions_tx": "position_M_3",
                    "sensor_velocities_tx": "velocity_M_3",
                    "range_times": "range_times_M",
                    "doppler_frequencies": "doppler_frequency_M",
                    "initial_guesses": "initial_guess_1_3",
                    "expected_shape": ("M", 3),
                },
                id="M-points-init-2d-doppler-array",
            ),
            pytest.param(
                {
                    "name": "M-points rx_pos=2d rx_vel=2d init=1d doppler=scalar",
                    "sensor_positions_rx": "position_1_3",
                    "sensor_velocities_rx": "velocity_1_3",
                    "sensor_positions_tx": "position_M_3",
                    "sensor_velocities_tx": "velocity_M_3",
                    "range_times": "range_times_M",
                    "doppler_frequencies": "doppler_frequency",
                    "initial_guesses": "initial_guess",
                    "expected_shape": ("M", 3),
                },
                id="M-points-rx-2d-init-1d-doppler-scalar-2",
            ),
            pytest.param(
                {
                    "name": "M-points rx_pos=2d rx_vel=2d init=1d doppler=array",
                    "sensor_positions_rx": "position_1_3",
                    "sensor_velocities_rx": "velocity_1_3",
                    "sensor_positions_tx": "position_M_3",
                    "sensor_velocities_tx": "velocity_M_3",
                    "range_times": "range_times_M",
                    "doppler_frequencies": "doppler_frequency_M",
                    "initial_guesses": "initial_guess",
                    "expected_shape": ("M", 3),
                },
                id="M-points-rx-2d-init-1d-doppler-array",
            ),
            pytest.param(
                {
                    "name": "M-points rx_pos=2d rx_vel=2d init=2d doppler=array",
                    "sensor_positions_rx": "position_1_3",
                    "sensor_velocities_rx": "velocity_1_3",
                    "sensor_positions_tx": "position_M_3",
                    "sensor_velocities_tx": "velocity_M_3",
                    "range_times": "range_times_M",
                    "doppler_frequencies": "doppler_frequency_M",
                    "initial_guesses": "initial_guess_1_3",
                    "expected_shape": ("M", 3),
                },
                id="M-points-rx-2d-init-2d-doppler-array",
            ),
        ],
    )
    def test_direct_geocoding_bistatic_m_points(self, case: dict) -> None:
        """Testing direct_geocoding_bistatic with M-point cases."""
        M = self.M
        value_map = {
            "position": self.position,
            "position_1_3": self.position.reshape(1, 3),
            "position_M_3": np.full((M, 3), self.position),
            "velocity": self.velocity,
            "velocity_1_3": self.velocity.reshape(1, 3),
            "velocity_M_3": np.full((M, 3), self.velocity),
            "initial_guess": self.initial_guess,
            "initial_guess_1_3": self.initial_guess.reshape(1, 3),
            "range_times_M": np.repeat(self.range_times, M),
            "doppler_frequency": self.doppler_frequency,
            "doppler_frequency_M": np.repeat(self.doppler_frequency, M),
        }

        sensor_positions_rx = value_map[case["sensor_positions_rx"]]
        sensor_velocities_rx = value_map[case["sensor_velocities_rx"]]
        sensor_positions_tx = value_map[case["sensor_positions_tx"]]
        sensor_velocities_tx = value_map[case["sensor_velocities_tx"]]
        range_times = value_map[case["range_times"]]
        doppler_frequencies = value_map[case["doppler_frequencies"]]
        initial_guesses = value_map[case["initial_guesses"]]

        out = direct_geocoding_bistatic(
            sensor_positions_rx=sensor_positions_rx,
            sensor_velocities_rx=sensor_velocities_rx,
            sensor_positions_tx=sensor_positions_tx,
            sensor_velocities_tx=sensor_velocities_tx,
            range_times=range_times,
            look_direction=self.look_direction,
            altitude=self.altitude,
            doppler_frequencies=doppler_frequencies,
            wavelength=self.wavelength,
            initial_guesses=initial_guesses,
        )

        if out.ndim == 2:
            self._check_residuals(
                sensor_positions_rx,
                sensor_velocities_rx,
                sensor_positions_tx,
                sensor_velocities_tx,
                range_times,
                out,
            )
        self._check_output((M, 3), out)

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(
                {
                    "name": "3-points init=1d doppler=scalar",
                    "sensor_positions_rx": "position_N_3",
                    "sensor_velocities_rx": "velocity_N_3",
                    "sensor_positions_tx": "position_M_3",
                    "sensor_velocities_tx": "velocity_M_3",
                    "range_times": "range_times_0_M",
                    "doppler_frequencies": "doppler_frequency",
                    "initial_guesses": "initial_guess",
                    "expected_shape": ("N", "M", 3),
                },
                id="3-points-init-1d-doppler-scalar",
            ),
            pytest.param(
                {
                    "name": "3-points init=2d doppler=scalar",
                    "sensor_positions_rx": "position_N_3",
                    "sensor_velocities_rx": "velocity_N_3",
                    "sensor_positions_tx": "position_M_3",
                    "sensor_velocities_tx": "velocity_M_3",
                    "range_times": "range_times_0_M",
                    "doppler_frequencies": "doppler_frequency",
                    "initial_guesses": "initial_guess_1_3",
                    "expected_shape": ("N", "M", 3),
                },
                id="3-points-init-2d-doppler-scalar",
            ),
            pytest.param(
                {
                    "name": "3-points init=2d doppler=array",
                    "sensor_positions_rx": "position_N_3",
                    "sensor_velocities_rx": "velocity_N_3",
                    "sensor_positions_tx": "position_M_3",
                    "sensor_velocities_tx": "velocity_M_3",
                    "range_times": "range_times_0_M",
                    "doppler_frequencies": "doppler_frequency_M",
                    "initial_guesses": "initial_guess_1_3",
                    "expected_shape": ("N", "M", 3),
                },
                id="3-points-init-2d-doppler-array",
            ),
            pytest.param(
                {
                    "name": "3-points init=3d doppler=scalar",
                    "sensor_positions_rx": "position_N_3",
                    "sensor_velocities_rx": "velocity_N_3",
                    "sensor_positions_tx": "position_M_3",
                    "sensor_velocities_tx": "velocity_M_3",
                    "range_times": "range_times_0_M",
                    "doppler_frequencies": "doppler_frequency",
                    "initial_guesses": "initial_guess_N_3",
                    "expected_shape": ("N", "M", 3),
                },
                id="3-points-init-3d-doppler-scalar",
            ),
            pytest.param(
                {
                    "name": "3-points init=3d doppler=array",
                    "sensor_positions_rx": "position_N_3",
                    "sensor_velocities_rx": "velocity_N_3",
                    "sensor_positions_tx": "position_M_3",
                    "sensor_velocities_tx": "velocity_M_3",
                    "range_times": "range_times_0_M",
                    "doppler_frequencies": "doppler_frequency_M",
                    "initial_guesses": "initial_guess_N_3",
                    "expected_shape": ("N", "M", 3),
                },
                id="3-points-init-3d-doppler-array",
            ),
        ],
    )
    def test_direct_geocoding_bistatic_3_points(self, case: dict) -> None:
        """Testing direct_geocoding_bistatic with 3-point cases (N x M)."""
        N, M = self.N, self.M
        value_map = {
            "position_N_3": np.full((N, 3), self.position),
            "velocity_N_3": np.full((N, 3), self.velocity),
            "position_M_3": np.full((M, 3), self.position),
            "velocity_M_3": np.full((M, 3), self.velocity),
            "initial_guess": self.initial_guess,
            "initial_guess_1_3": self.initial_guess.reshape(1, 3),
            "initial_guess_N_3": np.full((N, 3), self.initial_guess),
            "range_times_0_M": np.repeat(self.range_times[0], M),
            "doppler_frequency": self.doppler_frequency,
            "doppler_frequency_M": np.repeat(self.doppler_frequency, M),
        }

        sensor_positions_rx = value_map[case["sensor_positions_rx"]]
        sensor_velocities_rx = value_map[case["sensor_velocities_rx"]]
        sensor_positions_tx = value_map[case["sensor_positions_tx"]]
        sensor_velocities_tx = value_map[case["sensor_velocities_tx"]]
        range_times = value_map[case["range_times"]]
        doppler_frequencies = value_map[case["doppler_frequencies"]]
        initial_guesses = value_map[case["initial_guesses"]]

        out = direct_geocoding_bistatic(
            sensor_positions_rx=sensor_positions_rx,
            sensor_velocities_rx=sensor_velocities_rx,
            sensor_positions_tx=sensor_positions_tx,
            sensor_velocities_tx=sensor_velocities_tx,
            range_times=range_times,
            look_direction=self.look_direction,
            altitude=self.altitude,
            doppler_frequencies=doppler_frequencies,
            wavelength=self.wavelength,
            initial_guesses=initial_guesses,
        )

        if out.ndim == 3:
            for range_index in range(M):
                self._check_residuals(
                    sensor_positions_rx,
                    sensor_velocities_rx,
                    sensor_positions_tx[range_index, :],
                    sensor_velocities_tx[range_index, :],
                    range_times[range_index] if np.size(range_times) > 1 else range_times,
                    out[:, range_index, :],
                )
        else:
            self._check_residuals(
                sensor_positions_rx,
                sensor_velocities_rx,
                sensor_positions_tx,
                sensor_velocities_tx,
                range_times,
                out,
            )

        self._check_output((N, M, 3), out)


class TestDirectGeocodingBistaticCore:
    """Testing direct geocoding bistatic core with consolidated parametrize."""

    @pytest.fixture(autouse=True)
    def setup(self, direct_geocoding_test_data: dict) -> None:
        self.position = direct_geocoding_test_data["sensor_position"]
        self.velocity = direct_geocoding_test_data["sensor_velocity"]
        self.initial_guess = direct_geocoding_test_data["initial_guess"]
        self.range_times = direct_geocoding_test_data["range_time"]
        self.doppler_frequency = direct_geocoding_test_data["doppler_frequency"]
        self.altitude = direct_geocoding_test_data["geodetic_altitude"]
        self.look_direction = direct_geocoding_test_data["look_direction"]
        self.wavelength = direct_geocoding_test_data["wavelength"]
        self.N = direct_geocoding_test_data["az_reps"]
        self.M = direct_geocoding_test_data["rng_reps"]
        self.tolerance = direct_geocoding_test_data["tolerance"]
        self.residual_tolerance = direct_geocoding_test_data["residual_tolerance"]
        self.expected_results = direct_geocoding_test_data["expected_ground_points"]

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(
                {
                    "name": "case0a: 1 pos (3,), 1 vel (3,), 1 rng time, 1 initial guess (3,)",
                    "sensor_positions_rx": "position",
                    "sensor_velocities_rx": "velocity",
                    "sensor_positions_tx": "position",
                    "sensor_velocities_tx": "velocity",
                    "range_times": "range_times_0",
                    "doppler_frequencies": "doppler_frequency",
                    "initial_guesses": "initial_guess",
                    "expected_shape": (3,),
                },
                id="case0a",
            ),
            pytest.param(
                {
                    "name": "case0b: 1 pos (1, 3), 1 vel (1, 3), 1 rng time, 1 initial guess (1, 3)",
                    "sensor_positions_rx": "position_1_3",
                    "sensor_velocities_rx": "velocity_1_3",
                    "sensor_positions_tx": "position_1_3",
                    "sensor_velocities_tx": "velocity_1_3",
                    "range_times": "range_times_0",
                    "doppler_frequencies": "doppler_frequency",
                    "initial_guesses": "initial_guess_1_3",
                    "expected_shape": (1, 3),
                },
                id="case0b",
            ),
            pytest.param(
                {
                    "name": "case0c: 1 pos (3,) + (1, 3), 1 vel (3,) + (1, 3), 1 rng time, 1 initial guess (3,)",
                    "sensor_positions_rx": "position",
                    "sensor_velocities_rx": "velocity",
                    "sensor_positions_tx": "position_1_3",
                    "sensor_velocities_tx": "velocity_1_3",
                    "range_times": "range_times_0",
                    "doppler_frequencies": "doppler_frequency",
                    "initial_guesses": "initial_guess",
                    "expected_shape": (1, 3),
                },
                id="case0c",
            ),
            pytest.param(
                {
                    "name": "case1a: N pos (N, 3), N vel (N, 3), 1 rng time, 1 initial guess (3,)",
                    "sensor_positions_rx": "position_N_3",
                    "sensor_velocities_rx": "velocity_N_3",
                    "sensor_positions_tx": "position",
                    "sensor_velocities_tx": "velocity",
                    "range_times": "range_times_0",
                    "doppler_frequencies": "doppler_frequency",
                    "initial_guesses": "initial_guess",
                    "expected_shape": ("N", 3),
                },
                id="case1a",
            ),
            pytest.param(
                {
                    "name": "case1b: N pos (N, 3), N vel (N, 3), 1 rng time, N initial guess (N, 3)",
                    "sensor_positions_rx": "position_N_3",
                    "sensor_velocities_rx": "velocity_N_3",
                    "sensor_positions_tx": "position",
                    "sensor_velocities_tx": "velocity",
                    "range_times": "range_times_0",
                    "doppler_frequencies": "doppler_frequency",
                    "initial_guesses": "initial_guess_N_3",
                    "expected_shape": ("N", 3),
                },
                id="case1b",
            ),
            pytest.param(
                {
                    "name": "case1c: N pos (N, 3), N vel (N, 3), 1 rng time, 1 initial guess (1, 3)",
                    "sensor_positions_rx": "position_N_3",
                    "sensor_velocities_rx": "velocity_N_3",
                    "sensor_positions_tx": "position_1_3",
                    "sensor_velocities_tx": "velocity_1_3",
                    "range_times": "range_times_0",
                    "doppler_frequencies": "doppler_frequency",
                    "initial_guesses": "initial_guess_1_3",
                    "expected_shape": ("N", 3),
                },
                id="case1c",
            ),
            pytest.param(
                {
                    "name": "case2a: 1 pos (3,), 1 vel (3,), M rng times, 1 initial guess (3,)",
                    "sensor_positions_rx": "position",
                    "sensor_velocities_rx": "velocity",
                    "sensor_positions_tx": "position_M_3",
                    "sensor_velocities_tx": "velocity_M_3",
                    "range_times": "range_times_0_M",
                    "doppler_frequencies": "doppler_frequency",
                    "initial_guesses": "initial_guess",
                    "expected_shape": ("M", 3),
                },
                id="case2a",
            ),
            pytest.param(
                {
                    "name": "case2b: 1 pos (1, 3), 1 vel (1, 3), M rng times, 1 initial guess (1, 3)",
                    "sensor_positions_rx": "position_1_3",
                    "sensor_velocities_rx": "velocity_1_3",
                    "sensor_positions_tx": "position_M_3",
                    "sensor_velocities_tx": "velocity_M_3",
                    "range_times": "range_times_0_M",
                    "doppler_frequencies": "doppler_frequency",
                    "initial_guesses": "initial_guess_1_3",
                    "expected_shape": ("M", 3),
                },
                id="case2b",
            ),
            pytest.param(
                {
                    "name": "case2c: 1 pos (3,), 1 vel (3,), M rng times, M doppler freq",
                    "sensor_positions_rx": "position",
                    "sensor_velocities_rx": "velocity",
                    "sensor_positions_tx": "position_M_3",
                    "sensor_velocities_tx": "velocity_M_3",
                    "range_times": "range_times_0_M",
                    "doppler_frequencies": "doppler_frequency_M",
                    "initial_guesses": "initial_guess",
                    "expected_shape": ("M", 3),
                },
                id="case2c",
            ),
            pytest.param(
                {
                    "name": "case3a: N pos (N, 3), N vel (N, 3), M rng times, 1 initial guess (3,)",
                    "sensor_positions_rx": "position_N_3",
                    "sensor_velocities_rx": "velocity_N_3",
                    "sensor_positions_tx": "position_M_3",
                    "sensor_velocities_tx": "velocity_M_3",
                    "range_times": "range_times_0_M",
                    "doppler_frequencies": "doppler_frequency",
                    "initial_guesses": "initial_guess",
                    "expected_shape": ("N", "M", 3),
                },
                id="case3a",
            ),
            pytest.param(
                {
                    "name": "case3b: N pos (N, 3), N vel (N, 3), M rng times, M doppler freq, 1 init (1, 3)",
                    "sensor_positions_rx": "position_N_3",
                    "sensor_velocities_rx": "velocity_N_3",
                    "sensor_positions_tx": "position_M_3",
                    "sensor_velocities_tx": "velocity_M_3",
                    "range_times": "range_times_0_M",
                    "doppler_frequencies": "doppler_frequency_M",
                    "initial_guesses": "initial_guess_1_3",
                    "expected_shape": ("N", "M", 3),
                },
                id="case3b",
            ),
            pytest.param(
                {
                    "name": "case3c: N pos (N, 3), N vel (N, 3), M rng times, 1 doppler, N init (N, 3)",
                    "sensor_positions_rx": "position_N_3",
                    "sensor_velocities_rx": "velocity_N_3",
                    "sensor_positions_tx": "position_M_3",
                    "sensor_velocities_tx": "velocity_M_3",
                    "range_times": "range_times_0_M",
                    "doppler_frequencies": "doppler_frequency",
                    "initial_guesses": "initial_guess_N_3",
                    "expected_shape": ("N", "M", 3),
                },
                id="case3c",
            ),
            pytest.param(
                {
                    "name": "case4: N pos (N, 3), N vel (N, 3), M rng times, M doppler freq, N init (N, 3)",
                    "sensor_positions_rx": "position_N_3",
                    "sensor_velocities_rx": "velocity_N_3",
                    "sensor_positions_tx": "position_M_3",
                    "sensor_velocities_tx": "velocity_M_3",
                    "range_times": "range_times_0_M",
                    "doppler_frequencies": "doppler_frequency_M",
                    "initial_guesses": "initial_guess_N_3",
                    "expected_shape": ("N", "M", 3),
                },
                id="case4",
            ),
        ],
    )
    def test_direct_geocoding_bistatic_core_cases(self, case: dict) -> None:
        """Testing direct_geocoding_bistatic_core success cases."""
        N, M = self.N, self.M
        value_map = {
            "position": self.position,
            "position_1_3": self.position.reshape(1, 3),
            "position_N_3": np.full((N, 3), self.position),
            "position_M_3": np.full((M, 3), self.position),
            "velocity": self.velocity,
            "velocity_1_3": self.velocity.reshape(1, 3),
            "velocity_N_3": np.full((N, 3), self.velocity),
            "velocity_M_3": np.full((M, 3), self.velocity),
            "initial_guess": self.initial_guess,
            "initial_guess_1_3": self.initial_guess.reshape(1, 3),
            "initial_guess_N_3": np.full((N, 3), self.initial_guess),
            "range_times_0": self.range_times[0],
            "range_times_0_M": np.repeat(self.range_times[0], M),
            "doppler_frequency": self.doppler_frequency,
            "doppler_frequency_M": np.repeat(self.doppler_frequency, M),
        }

        sensor_positions_rx = value_map[case["sensor_positions_rx"]]
        sensor_velocities_rx = value_map[case["sensor_velocities_rx"]]
        sensor_positions_tx = value_map[case["sensor_positions_tx"]]
        sensor_velocities_tx = value_map[case["sensor_velocities_tx"]]
        range_times = value_map[case["range_times"]]
        doppler_frequencies = value_map[case["doppler_frequencies"]]
        initial_guesses = value_map[case["initial_guesses"]]

        expected_shape = tuple(N if dim == "N" else M if dim == "M" else dim for dim in case["expected_shape"])

        out = direct_geocoding_bistatic_core(
            sensor_positions_rx=sensor_positions_rx,
            sensor_velocities_rx=sensor_velocities_rx,
            initial_guesses=initial_guesses,
            sensor_positions_tx=sensor_positions_tx,
            sensor_velocities_tx=sensor_velocities_tx,
            range_times=range_times,
            altitude=self.altitude,
            doppler_frequencies=doppler_frequencies,
            wavelength=self.wavelength,
        )

        assert out.shape == expected_shape
        np.testing.assert_allclose(
            out,
            _reshape_output(self.expected_results, expected_shape),
            atol=self.tolerance["atol"],
            rtol=self.tolerance["rtol"],
        )

    def test_direct_geocoding_bistatic_core_error_cases(self) -> None:
        """Testing direct_geocoding_bistatic_core error cases."""
        # case5: N doppler (N,) with M doppler (M,) - size mismatch
        with pytest.raises(RuntimeError):
            direct_geocoding_bistatic_core(
                sensor_positions_rx=np.full((self.N, 3), self.position),
                sensor_velocities_rx=np.full((self.N, 3), self.velocity),
                initial_guesses=np.full((self.N, 3), self.initial_guess),
                sensor_positions_tx=np.full((self.M, 3), self.position),
                sensor_velocities_tx=np.full((self.M, 3), self.velocity),
                range_times=np.repeat(self.range_times[0], self.M),
                altitude=self.altitude,
                doppler_frequencies=np.repeat(self.doppler_frequency, self.N),
                wavelength=self.wavelength,
            )


class TestNewtonForDirectGeocodingBistatic:
    """Testing Newton for direct geocoding bistatic core with consolidated parametrize."""

    @pytest.fixture(autouse=True)
    def setup(self, direct_geocoding_test_data: dict) -> None:
        self.position = direct_geocoding_test_data["sensor_position"]
        self.velocity = direct_geocoding_test_data["sensor_velocity"]
        self.initial_guess = direct_geocoding_test_data["initial_guess"]
        self.range_times = direct_geocoding_test_data["range_time"]
        self.doppler_frequency = direct_geocoding_test_data["doppler_frequency"]
        self.altitude = direct_geocoding_test_data["geodetic_altitude"]
        self.look_direction = direct_geocoding_test_data["look_direction"]
        self.wavelength = direct_geocoding_test_data["wavelength"]
        self.N = direct_geocoding_test_data["az_reps"]
        self.M = direct_geocoding_test_data["rng_reps"]
        self.tolerance = direct_geocoding_test_data["tolerance"]
        self.residual_tolerance = direct_geocoding_test_data["residual_tolerance"]
        self.expected_results = direct_geocoding_test_data["expected_ground_points"]

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(
                {
                    "name": "case0: 1 pos (3,), 1 vel (3,), 1 init guess (3,)",
                    "sensor_positions_rx": "position",
                    "sensor_velocities_rx": "velocity",
                    "initial_guesses": "initial_guess",
                    "expected_shape": (3,),
                },
                id="case0",
            ),
            pytest.param(
                {
                    "name": "case1: 1 pos (1, 3), 1 vel (1, 3), 1 init guess (1, 3)",
                    "sensor_positions_rx": "position_1_3",
                    "sensor_velocities_rx": "velocity_1_3",
                    "initial_guesses": "initial_guess_1_3",
                    "expected_shape": (1, 3),
                },
                id="case1",
            ),
            pytest.param(
                {
                    "name": "case2a: N pos (N, 3), N vel (N, 3), 1 init guess (3,)",
                    "sensor_positions_rx": "position_N_3",
                    "sensor_velocities_rx": "velocity_N_3",
                    "initial_guesses": "initial_guess",
                    "expected_shape": ("N", 3),
                },
                id="case2a",
            ),
            pytest.param(
                {
                    "name": "case2b: N pos (N, 3), N vel (N, 3), 1 init guess (1, 3)",
                    "sensor_positions_rx": "position_N_3",
                    "sensor_velocities_rx": "velocity_N_3",
                    "initial_guesses": "initial_guess_1_3",
                    "expected_shape": ("N", 3),
                },
                id="case2b",
            ),
            pytest.param(
                {
                    "name": "case3: N pos (N, 3), N vel (N, 3), 1 init guess (1, 3)",
                    "sensor_positions_rx": "position_N_3",
                    "sensor_velocities_rx": "velocity_N_3",
                    "initial_guesses": "initial_guess_1_3",
                    "expected_shape": ("N", 3),
                },
                id="case3",
            ),
        ],
    )
    def test_newton_for_direct_geocoding_bistatic_cases(self, case: dict) -> None:
        """Testing _direct_geocoding_bistatic_newton cases."""
        N = self.N
        value_map = {
            "position": self.position,
            "position_1_3": self.position.reshape(1, 3),
            "position_N_3": np.full((N, 3), self.position),
            "velocity": self.velocity,
            "velocity_1_3": self.velocity.reshape(1, 3),
            "velocity_N_3": np.full((N, 3), self.velocity),
            "initial_guess": self.initial_guess,
            "initial_guess_1_3": self.initial_guess.reshape(1, 3),
        }

        sensor_positions_rx = value_map[case["sensor_positions_rx"]]
        sensor_velocities_rx = value_map[case["sensor_velocities_rx"]]
        initial_guesses = value_map[case["initial_guesses"]]
        expected_shape = tuple(N if dim == "N" else dim for dim in case["expected_shape"])

        out = _direct_geocoding_bistatic_newton(
            sensor_positions_rx=sensor_positions_rx,
            sensor_velocities_rx=sensor_velocities_rx,
            initial_guesses=initial_guesses,
            sensor_position_tx=self.position,
            sensor_velocity_tx=self.velocity,
            range_times=self.range_times[0],
            doppler_frequencies=self.doppler_frequency,
            wavelength=self.wavelength,
            altitude=self.altitude,
        )

        assert out.shape == expected_shape
        np.testing.assert_allclose(
            out,
            _reshape_output(self.expected_results, expected_shape),
            atol=self.tolerance["atol"],
            rtol=self.tolerance["rtol"],
        )
