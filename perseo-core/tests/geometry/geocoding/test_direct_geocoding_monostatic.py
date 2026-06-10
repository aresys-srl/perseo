# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for geometry/geocoding/direct_geocoding.py and direct_geocoding_core.py monostatic functionalities"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import speed_of_light

from perseo_core.geometry.coordinates.ellipsoid import WGS84
from perseo_core.geometry.doppler import doppler_equation
from perseo_core.geometry.geocoding.core.direct import (
    _direct_geocoding_monostatic_newton,
    _ellipse_equation,
    direct_geocoding_monostatic_core,
    direct_geocoding_monostatic_core_range_vectorized,
)
from perseo_core.geometry.geocoding.direct import direct_geocoding_init, direct_geocoding_monostatic


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

    doppler_residual, _ = doppler_equation(
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

    return _ellipse_equation(ground_points, r_ee2, r_ep2)


class TestDirectGeocodingMonostatic:
    """Test direct_geocoding_monostatic with various input dimension combinations using parametrize."""

    @pytest.fixture(autouse=True)
    def setup(self, direct_geocoding_test_data: dict) -> None:
        """Load test data from fixtures."""
        self.sensor_position = direct_geocoding_test_data["sensor_position"]
        self.sensor_velocity = direct_geocoding_test_data["sensor_velocity"]
        self.initial_guess = direct_geocoding_test_data["initial_guess"]
        self.range_time = direct_geocoding_test_data["range_time"]
        self.doppler_freq = direct_geocoding_test_data["doppler_frequency"]
        self.geodetic_altitude = direct_geocoding_test_data["geodetic_altitude"]
        self.wavelength = direct_geocoding_test_data["wavelength"]
        self.look_direction = direct_geocoding_test_data["look_direction"]
        self.N = direct_geocoding_test_data["az_reps"]
        self.M = direct_geocoding_test_data["rng_reps"]
        self.Q = direct_geocoding_test_data["high_rng_reps"]
        self.tolerance = direct_geocoding_test_data["tolerance"]
        self.residual_tolerance = direct_geocoding_test_data["residual_tolerance"]
        self.results = direct_geocoding_test_data["expected_ground_points"]

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(
                {
                    "name": "case0a: 1 pos (3,), 1 vel (3,), 1 rng time, 1 initial guess (3,)",
                    "positions": "sensor_position",
                    "velocities": "sensor_velocity",
                    "initial_guesses": "initial_guess",
                    "range_times": "range_time_0",
                    "doppler_frequencies": "doppler_freq",
                    "expected_ndim": 1,
                    "expected_shape": (3,),
                    "check_residuals": True,
                },
                id="case0a",
            ),
            pytest.param(
                {
                    "name": "case0b: 1 pos (1,3), 1 vel (1,3), 1 rng time, 1 initial guess (1,3)",
                    "positions": "sensor_position_1_3",
                    "velocities": "sensor_velocity_1_3",
                    "initial_guesses": "initial_guess_1_3",
                    "range_times": "range_time_0",
                    "doppler_frequencies": "doppler_freq",
                    "expected_ndim": 2,
                    "expected_shape": (1, 3),
                    "check_residuals": True,
                },
                id="case0b",
            ),
            pytest.param(
                {
                    "name": "case0c: 1 pos (3,), 1 vel (3,), 1 rng time, no initial guess",
                    "positions": "sensor_position",
                    "velocities": "sensor_velocity",
                    "initial_guesses": None,
                    "range_times": "range_time_0",
                    "doppler_frequencies": "doppler_freq",
                    "expected_ndim": 1,
                    "expected_shape": (3,),
                    "check_residuals": True,
                },
                id="case0c",
            ),
            pytest.param(
                {
                    "name": "case0d: 1 pos (1,3), 1 vel (1,3), 1 rng time, no initial guess",
                    "positions": "sensor_position_1_3",
                    "velocities": "sensor_velocity_1_3",
                    "initial_guesses": None,
                    "range_times": "range_time_0",
                    "doppler_frequencies": "doppler_freq",
                    "expected_ndim": 2,
                    "expected_shape": (1, 3),
                    "check_residuals": True,
                },
                id="case0d",
            ),
            pytest.param(
                {
                    "name": "case1a: N pos (N,3), N vel (N,3), 1 rng time, 1 initial guess (3,)",
                    "positions": "sensor_position_N_3",
                    "velocities": "sensor_velocity_N_3",
                    "initial_guesses": "initial_guess",
                    "range_times": "range_time_0",
                    "doppler_frequencies": "doppler_freq",
                    "expected_ndim": 2,
                    "expected_shape": ("N", 3),
                    "check_residuals": True,
                },
                id="case1a",
            ),
            pytest.param(
                {
                    "name": "case1b: N pos (N,3), N vel (N,3), 1 rng time, 1 initial guess (1,3)",
                    "positions": "sensor_position_N_3",
                    "velocities": "sensor_velocity_N_3",
                    "initial_guesses": "initial_guess_1_3",
                    "range_times": "range_time_0",
                    "doppler_frequencies": "doppler_freq",
                    "expected_ndim": 2,
                    "expected_shape": ("N", 3),
                    "check_residuals": True,
                },
                id="case1b",
            ),
            pytest.param(
                {
                    "name": "case1c: N pos (N,3), N vel (N,3), 1 rng time, N initial guesses (N,3)",
                    "positions": "sensor_position_N_3",
                    "velocities": "sensor_velocity_N_3",
                    "initial_guesses": "initial_guess_N_3",
                    "range_times": "range_time_0",
                    "doppler_frequencies": "doppler_freq",
                    "expected_ndim": 2,
                    "expected_shape": ("N", 3),
                    "check_residuals": True,
                },
                id="case1c",
            ),
            pytest.param(
                {
                    "name": "case1d: N pos (N,3), N vel (N,3), 1 rng time, no initial guess",
                    "positions": "sensor_position_N_3",
                    "velocities": "sensor_velocity_N_3",
                    "initial_guesses": None,
                    "range_times": "range_time_0",
                    "doppler_frequencies": "doppler_freq",
                    "expected_ndim": 2,
                    "expected_shape": ("N", 3),
                    "check_residuals": True,
                },
                id="case1d",
            ),
            pytest.param(
                {
                    "name": "case2a: 1 pos (3,), 1 vel (3,), M rng times (M,), 1 initial guess (3,)",
                    "positions": "sensor_position",
                    "velocities": "sensor_velocity",
                    "initial_guesses": "initial_guess",
                    "range_times": "range_time_M",
                    "doppler_frequencies": "doppler_freq",
                    "expected_ndim": 2,
                    "expected_shape": ("M", 3),
                    "check_residuals": True,
                },
                id="case2a",
            ),
            pytest.param(
                {
                    "name": "case2b: 1 pos (1,3), 1 vel (1,3), M rng times (M,), 1 initial guess (1,3)",
                    "positions": "sensor_position_1_3",
                    "velocities": "sensor_velocity_1_3",
                    "initial_guesses": "initial_guess_1_3",
                    "range_times": "range_time_M",
                    "doppler_frequencies": "doppler_freq",
                    "expected_ndim": 2,
                    "expected_shape": ("M", 3),
                    "check_residuals": True,
                },
                id="case2b",
            ),
            pytest.param(
                {
                    "name": "case2c: 1 pos (3,), 1 vel (3,), M rng times (M,), no initial guess",
                    "positions": "sensor_position",
                    "velocities": "sensor_velocity",
                    "initial_guesses": None,
                    "range_times": "range_time_M",
                    "doppler_frequencies": "doppler_freq",
                    "expected_ndim": 2,
                    "expected_shape": ("M", 3),
                    "check_residuals": True,
                },
                id="case2c",
            ),
            pytest.param(
                {
                    "name": "case2d: 1 pos (3,), 1 vel (3,), M rng times (M,), 1 initial guess (3,), M doppler freqs",
                    "positions": "sensor_position",
                    "velocities": "sensor_velocity",
                    "initial_guesses": "initial_guess",
                    "range_times": "range_time_M",
                    "doppler_frequencies": "doppler_freq_M",
                    "expected_ndim": 2,
                    "expected_shape": ("M", 3),
                    "check_residuals": True,
                },
                id="case2d",
            ),
            pytest.param(
                {
                    "name": "case2e: 1 pos (1,3), 1 vel (1,3), M rng times (M,), 1 init guess (1,3), M doppler freqs",
                    "positions": "sensor_position_1_3",
                    "velocities": "sensor_velocity_1_3",
                    "initial_guesses": "initial_guess_1_3",
                    "range_times": "range_time_M",
                    "doppler_frequencies": "doppler_freq_M",
                    "expected_ndim": 2,
                    "expected_shape": ("M", 3),
                    "check_residuals": True,
                },
                id="case2e",
            ),
            pytest.param(
                {
                    "name": "case2f: 1 pos (3,), 1 vel (3,), Q rng times (M,), 1 initial guess (3,)",
                    "positions": "sensor_position",
                    "velocities": "sensor_velocity",
                    "initial_guesses": "initial_guess",
                    "range_times": "range_time_Q",
                    "doppler_frequencies": "doppler_freq",
                    "expected_ndim": 2,
                    "expected_shape": ("Q", 3),
                    "check_residuals": True,
                },
                id="case2f",
            ),
            pytest.param(
                {
                    "name": "case2g: 1 pos (1,3), 1 vel (1,3), Q rng times (Q,), 1 initial guess (1,3)",
                    "positions": "sensor_position_1_3",
                    "velocities": "sensor_velocity_1_3",
                    "initial_guesses": "initial_guess_1_3",
                    "range_times": "range_time_Q",
                    "doppler_frequencies": "doppler_freq",
                    "expected_ndim": 2,
                    "expected_shape": ("Q", 3),
                    "check_residuals": True,
                },
                id="case2g",
            ),
            pytest.param(
                {
                    "name": "case2h: 1 pos (3,), 1 vel (3,), Q rng times (Q,), no initial guess",
                    "positions": "sensor_position",
                    "velocities": "sensor_velocity",
                    "initial_guesses": None,
                    "range_times": "range_time_Q",
                    "doppler_frequencies": "doppler_freq",
                    "expected_ndim": 2,
                    "expected_shape": ("Q", 3),
                    "check_residuals": True,
                },
                id="case2h",
            ),
            pytest.param(
                {
                    "name": "case2i: 1 pos (3,), 1 vel (3,), Q rng times (Q,), 1 initial guess (3,), Q doppler freqs",
                    "positions": "sensor_position",
                    "velocities": "sensor_velocity",
                    "initial_guesses": "initial_guess",
                    "range_times": "range_time_Q",
                    "doppler_frequencies": "doppler_freq_Q",
                    "expected_ndim": 2,
                    "expected_shape": ("Q", 3),
                    "check_residuals": True,
                },
                id="case2i",
            ),
            pytest.param(
                {
                    "name": "case2j: 1 pos (1,3), 1 vel (1,3), Q rng times (Q,), 1 init guess (1,3), Q doppler freqs",
                    "positions": "sensor_position_1_3",
                    "velocities": "sensor_velocity_1_3",
                    "initial_guesses": "initial_guess_1_3",
                    "range_times": "range_time_Q",
                    "doppler_frequencies": "doppler_freq_Q",
                    "expected_ndim": 2,
                    "expected_shape": ("Q", 3),
                    "check_residuals": True,
                },
                id="case2j",
            ),
            pytest.param(
                {
                    "name": "case2k: 1 pos (1,3), 1 vel (1,3), Q rng times (Q,), Q init guesses (Q,3), Q doppler freqs",
                    "positions": "sensor_position_1_3",
                    "velocities": "sensor_velocity_1_3",
                    "initial_guesses": "initial_guess_Q_3",
                    "range_times": "range_time_Q",
                    "doppler_frequencies": "doppler_freq_Q",
                    "expected_ndim": 2,
                    "expected_shape": ("Q", 3),
                    "check_residuals": True,
                },
                id="case2k",
            ),
            pytest.param(
                {
                    "name": "case3a: N pos (N,3), N vel (N,3), M rng times (M,), 1 initial guess (3,)",
                    "positions": "sensor_position_N_3",
                    "velocities": "sensor_velocity_N_3",
                    "initial_guesses": "initial_guess",
                    "range_times": "range_time_M",
                    "doppler_frequencies": "doppler_freq",
                    "expected_ndim": 3,
                    "expected_shape": ("N", "M", 3),
                    "check_residuals": True,
                },
                id="case3a",
            ),
            pytest.param(
                {
                    "name": "case3b: N pos (N,3), N vel (N,3), M rng times (M,), no initial guess",
                    "positions": "sensor_position_N_3",
                    "velocities": "sensor_velocity_N_3",
                    "initial_guesses": None,
                    "range_times": "range_time_M",
                    "doppler_frequencies": "doppler_freq",
                    "expected_ndim": 3,
                    "expected_shape": ("N", "M", 3),
                    "check_residuals": True,
                },
                id="case3b",
            ),
            pytest.param(
                {
                    "name": "case3c: N pos (N,3), N vel (N,3), M rng times (M,), 1 initial guess (3,), M doppler freqs",
                    "positions": "sensor_position_N_3",
                    "velocities": "sensor_velocity_N_3",
                    "initial_guesses": "initial_guess",
                    "range_times": "range_time_M",
                    "doppler_frequencies": "doppler_freq_M",
                    "expected_ndim": 3,
                    "expected_shape": ("N", "M", 3),
                    "check_residuals": True,
                },
                id="case3c",
            ),
            pytest.param(
                {
                    "name": "case3d: N pos (N,3), N vel (N,3), Q rng times (Q,), 1 initial guess (3,)",
                    "positions": "sensor_position_N_3",
                    "velocities": "sensor_velocity_N_3",
                    "initial_guesses": "initial_guess",
                    "range_times": "range_time_Q",
                    "doppler_frequencies": "doppler_freq",
                    "expected_ndim": 3,
                    "expected_shape": ("N", "Q", 3),
                    "check_residuals": True,
                },
                id="case3d",
            ),
            pytest.param(
                {
                    "name": "case3e: N pos (N,3), N vel (N,3), Q rng times (Q,), no initial guess",
                    "positions": "sensor_position_N_3",
                    "velocities": "sensor_velocity_N_3",
                    "initial_guesses": None,
                    "range_times": "range_time_Q",
                    "doppler_frequencies": "doppler_freq",
                    "expected_ndim": 3,
                    "expected_shape": ("N", "Q", 3),
                    "check_residuals": True,
                },
                id="case3e",
            ),
            pytest.param(
                {
                    "name": "case3f: N pos (N,3), N vel (N,3), Q rng times (Q,), 1 initial guess (3,), Q doppler freqs",
                    "positions": "sensor_position_N_3",
                    "velocities": "sensor_velocity_N_3",
                    "initial_guesses": "initial_guess",
                    "range_times": "range_time_Q",
                    "doppler_frequencies": "doppler_freq_Q",
                    "expected_ndim": 3,
                    "expected_shape": ("N", "Q", 3),
                    "check_residuals": True,
                },
                id="case3f",
            ),
            pytest.param(
                {
                    "name": "case3g: N pos (N,3), N vel (N,3), Q rng times (Q,), Q initial guesses (Q,3)",
                    "positions": "sensor_position_N_3",
                    "velocities": "sensor_velocity_N_3",
                    "initial_guesses": "initial_guess_Q_3",
                    "range_times": "range_time_Q",
                    "doppler_frequencies": "doppler_freq_Q",
                    "expected_ndim": 3,
                    "expected_shape": ("N", "Q", 3),
                    "check_residuals": True,
                },
                id="case3g",
            ),
        ],
    )
    def test_direct_geocoding_monostatic_cases(self, case: dict) -> None:
        """Test direct_geocoding_monostatic with all input dimension combinations."""
        # Resolve dynamic values
        N, M, Q = self.N, self.M, self.Q

        value_map = {
            "sensor_position": self.sensor_position,
            "sensor_position_1_3": self.sensor_position.reshape(1, 3),
            "sensor_position_N_3": np.full((N, 3), self.sensor_position),
            "sensor_velocity": self.sensor_velocity,
            "sensor_velocity_1_3": self.sensor_velocity.reshape(1, 3),
            "sensor_velocity_N_3": np.full((N, 3), self.sensor_velocity),
            "initial_guess": self.initial_guess,
            "initial_guess_1_3": self.initial_guess.reshape(1, 3),
            "initial_guess_N_3": np.full((N, 3), self.initial_guess),
            "initial_guess_Q_3": np.full((Q, 3), self.initial_guess),
            "range_time_0": self.range_time[0],
            "range_time_M": np.repeat(self.range_time[0], M),
            "range_time_Q": np.repeat(self.range_time[0], Q),
            "doppler_freq": self.doppler_freq,
            "doppler_freq_M": np.repeat(self.doppler_freq, M),
            "doppler_freq_Q": np.repeat(self.doppler_freq, Q),
        }

        positions = value_map[case["positions"]]
        velocities = value_map[case["velocities"]]
        initial_guesses = value_map[case["initial_guesses"]] if case["initial_guesses"] is not None else None
        range_times = value_map[case["range_times"]]
        doppler_frequencies = value_map[case["doppler_frequencies"]]

        expected_shape = tuple(
            N if dim == "N" else M if dim == "M" else Q if dim == "Q" else dim for dim in case["expected_shape"]
        )

        out = direct_geocoding_monostatic(
            sensor_positions=positions,
            sensor_velocities=velocities,
            initial_guesses=initial_guesses,
            range_times=range_times,
            doppler_frequencies=doppler_frequencies,
            altitude=self.geodetic_altitude,
            look_direction=self.look_direction,
            wavelength=self.wavelength,
        )

        assert out.ndim == case["expected_ndim"]
        assert out.shape == expected_shape

        if case["check_residuals"]:
            doppler_residual = _doppler_equation_residual(
                sensor_pos=self.sensor_position,
                sensor_vel=self.sensor_velocity,
                ground_points=out,
                doppler_freq=self.doppler_freq,
                wavelength=self.wavelength,
            )
            range_residual = _range_equation_residual(
                sensor_pos=self.sensor_position, ground_points=out, range_time=self.range_time[0]
            )
            ellipse_residual = _ellipse_equation_residual(ground_points=out)

            np.testing.assert_allclose(
                doppler_residual,
                np.zeros_like(doppler_residual),
                atol=self.residual_tolerance["atol"],
                rtol=self.residual_tolerance["rtol"],
            )
            np.testing.assert_allclose(
                range_residual,
                np.zeros_like(range_residual),
                atol=self.residual_tolerance["atol"],
                rtol=self.residual_tolerance["rtol"],
            )
            np.testing.assert_allclose(
                ellipse_residual,
                np.zeros_like(ellipse_residual),
                atol=self.residual_tolerance["atol"],
                rtol=self.residual_tolerance["rtol"],
            )

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(
                {
                    "name": "case4: N pos (N,3), M vel (M,3), mismatch position/velocity",
                    "positions": "sensor_position_N_3",
                    "velocities": "sensor_velocity_M_3",
                    "initial_guesses": "initial_guess_N_3",
                    "range_times": "range_time_M",
                    "doppler_frequencies": "doppler_freq",
                },
                id="case4",
            ),
            pytest.param(
                {
                    "name": "case5: N pos (N,3), M init guesses (M,3), mismatch position/init guesses",
                    "positions": "sensor_position_N_3",
                    "velocities": "sensor_velocity_N_3",
                    "initial_guesses": "initial_guess_M_3",
                    "range_times": "range_time_M",
                    "doppler_frequencies": "doppler_freq",
                },
                id="case5",
            ),
            pytest.param(
                {
                    "name": "case6: N range (N,), M freqs (M,), mismatch frequency/ranges",
                    "positions": "sensor_position_N_3",
                    "velocities": "sensor_velocity_N_3",
                    "initial_guesses": "initial_guess_N_3",
                    "range_times": "range_time_N",
                    "doppler_frequencies": "doppler_freq_M",
                },
                id="case6",
            ),
        ],
    )
    def test_direct_geocoding_monostatic_error_cases(self, case: dict) -> None:
        """Test direct_geocoding_monostatic error handling for mismatched dimensions."""
        N, M = self.N, self.M

        value_map = {
            "sensor_position_N_3": np.full((N, 3), self.sensor_position),
            "sensor_velocity_M_3": np.full((M, 3), self.sensor_velocity),
            "sensor_velocity_N_3": np.full((N, 3), self.sensor_velocity),
            "initial_guess_N_3": np.full((N, 3), self.initial_guess),
            "initial_guess_M_3": np.full((M, 3), self.initial_guess)
            if hasattr(self, "M")
            else np.full((N // 2, 3), self.initial_guess),
            "range_time_M": np.repeat(self.range_time[0], M),
            "range_time_N": np.repeat(self.range_time[0], N),
            "doppler_freq": self.doppler_freq,
            "doppler_freq_M": np.repeat(self.doppler_freq, M),
        }

        positions = value_map[case["positions"]]
        velocities = value_map[case["velocities"]]
        initial_guesses = value_map[case["initial_guesses"]]
        range_times = value_map[case["range_times"]]
        doppler_frequencies = value_map[case["doppler_frequencies"]]

        with pytest.raises(RuntimeError):
            direct_geocoding_monostatic(
                sensor_positions=positions,
                sensor_velocities=velocities,
                initial_guesses=initial_guesses,
                range_times=range_times,
                doppler_frequencies=doppler_frequencies,
                altitude=self.geodetic_altitude,
                look_direction=self.look_direction,
                wavelength=self.wavelength,
            )


class TestDirectGeocodingMonostaticCore:
    """Testing direct geocoding monostatic core with various input combinations using parametrize."""

    @pytest.fixture(autouse=True)
    def setup(self, direct_geocoding_test_data: dict) -> None:
        """Setting up variables for testing"""
        self.sensor_position = direct_geocoding_test_data["sensor_position"]
        self.sensor_velocity = direct_geocoding_test_data["sensor_velocity"]
        self.initial_guess = direct_geocoding_test_data["initial_guess"]
        self.range_time = direct_geocoding_test_data["range_time"]
        self.doppler_freq = direct_geocoding_test_data["doppler_frequency"]
        self.geodetic_altitude = direct_geocoding_test_data["geodetic_altitude"]
        self.wavelength = direct_geocoding_test_data["wavelength"]
        self.N = direct_geocoding_test_data["az_reps"]
        self.M = direct_geocoding_test_data["rng_reps"]
        self.tolerance = direct_geocoding_test_data["tolerance"]
        self.results = direct_geocoding_test_data["expected_ground_points"]

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(
                {
                    "name": "case0a: 1 pos (3,), 1 vel (3,), 1 guess (3,), 1 rng time",
                    "positions": "sensor_position",
                    "velocities": "sensor_velocity",
                    "initial_guesses": "initial_guess",
                    "range_times": "range_time_0",
                    "doppler_frequencies": "doppler_freq",
                    "expected_ndim": 1,
                    "expected_shape": (3,),
                },
                id="case0a",
            ),
            pytest.param(
                {
                    "name": "case0b: 1 pos (1,3), 1 vel (1,3), 1 guess (1,3), 1 rng time",
                    "positions": "sensor_position_1_3",
                    "velocities": "sensor_velocity_1_3",
                    "initial_guesses": "initial_guess_1_3",
                    "range_times": "range_time_0",
                    "doppler_frequencies": "doppler_freq",
                    "expected_ndim": 2,
                    "expected_shape": (1, 3),
                },
                id="case0b",
            ),
            pytest.param(
                {
                    "name": "case0c: 1 pos (3,), 1 vel (3,), 1 guess (3,), M range times",
                    "positions": "sensor_position",
                    "velocities": "sensor_velocity",
                    "initial_guesses": "initial_guess",
                    "range_times": "range_time_M",
                    "doppler_frequencies": "doppler_freq",
                    "expected_ndim": 2,
                    "expected_shape": ("M", 3),
                },
                id="case0c",
            ),
            pytest.param(
                {
                    "name": "case0d: 1 pos (1,3), 1 vel (1,3), 1 guess (1,3), M range times",
                    "positions": "sensor_position_1_3",
                    "velocities": "sensor_velocity_1_3",
                    "initial_guesses": "initial_guess_1_3",
                    "range_times": "range_time_M",
                    "doppler_frequencies": "doppler_freq",
                    "expected_ndim": 2,
                    "expected_shape": ("M", 3),
                },
                id="case0d",
            ),
            pytest.param(
                {
                    "name": "case0e: 1 pos (3,), 1 vel (3,), 1 guess (3,), M range times, M doppler freqs",
                    "positions": "sensor_position",
                    "velocities": "sensor_velocity",
                    "initial_guesses": "initial_guess",
                    "range_times": "range_time_M",
                    "doppler_frequencies": "doppler_freq_M",
                    "expected_ndim": 2,
                    "expected_shape": ("M", 3),
                },
                id="case0e",
            ),
            pytest.param(
                {
                    "name": "case0f: 1 pos (1,3), 1 vel (1,3), 1 guess (1,3), M range times, M doppler freqs",
                    "positions": "sensor_position_1_3",
                    "velocities": "sensor_velocity_1_3",
                    "initial_guesses": "initial_guess_1_3",
                    "range_times": "range_time_M",
                    "doppler_frequencies": "doppler_freq_M",
                    "expected_ndim": 2,
                    "expected_shape": ("M", 3),
                },
                id="case0f",
            ),
            pytest.param(
                {
                    "name": "case1a: N pos (N,3), N vel (N,3), N guess (N,3), 1 rng time",
                    "positions": "sensor_position_N_3",
                    "velocities": "sensor_velocity_N_3",
                    "initial_guesses": "initial_guess_N_3",
                    "range_times": "range_time_0",
                    "doppler_frequencies": "doppler_freq",
                    "expected_ndim": 2,
                    "expected_shape": ("N", 3),
                },
                id="case1a",
            ),
            pytest.param(
                {
                    "name": "case1b: N pos (N,3), N vel (N,3), N guess (N,3), M rng times",
                    "positions": "sensor_position_N_3",
                    "velocities": "sensor_velocity_N_3",
                    "initial_guesses": "initial_guess_N_3",
                    "range_times": "range_time_M",
                    "doppler_frequencies": "doppler_freq",
                    "expected_ndim": 3,
                    "expected_shape": ("N", "M", 3),
                },
                id="case1b",
            ),
            pytest.param(
                {
                    "name": "case1c: N pos (N,3), N vel (N,3), N guess (N,3), M rng times, M doppler freqs",
                    "positions": "sensor_position_N_3",
                    "velocities": "sensor_velocity_N_3",
                    "initial_guesses": "initial_guess_N_3",
                    "range_times": "range_time_M",
                    "doppler_frequencies": "doppler_freq_M",
                    "expected_ndim": 3,
                    "expected_shape": ("N", "M", 3),
                },
                id="case1c",
            ),
        ],
    )
    def test_monostatic_core_cases(self, case: dict) -> None:
        """Test direct_geocoding_monostatic_core with various input combinations."""
        N, M = self.N, self.M

        value_map = {
            "sensor_position": self.sensor_position,
            "sensor_position_1_3": self.sensor_position.reshape(1, 3),
            "sensor_position_N_3": np.full((N, 3), self.sensor_position),
            "sensor_velocity": self.sensor_velocity,
            "sensor_velocity_1_3": self.sensor_velocity.reshape(1, 3),
            "sensor_velocity_N_3": np.full((N, 3), self.sensor_velocity),
            "initial_guess": self.initial_guess,
            "initial_guess_1_3": self.initial_guess.reshape(1, 3),
            "initial_guess_N_3": np.full((N, 3), self.initial_guess),
            "range_time_0": self.range_time[0],
            "range_time_M": np.repeat(self.range_time[0], M),
            "doppler_freq": self.doppler_freq,
            "doppler_freq_M": np.repeat(self.doppler_freq, M),
        }

        positions = value_map[case["positions"]]
        velocities = value_map[case["velocities"]]
        initial_guesses = value_map[case["initial_guesses"]]
        range_times = value_map[case["range_times"]]
        doppler_frequencies = value_map[case["doppler_frequencies"]]

        expected_shape = tuple(N if dim == "N" else M if dim == "M" else dim for dim in case["expected_shape"])

        out = direct_geocoding_monostatic_core(
            sensor_positions=positions,
            sensor_velocities=velocities,
            initial_guesses=initial_guesses,
            range_times=range_times,
            doppler_frequencies=doppler_frequencies,
            wavelength=self.wavelength,
            altitude=self.geodetic_altitude,
        )

        assert out.ndim == case["expected_ndim"]
        assert out.shape == expected_shape

        # Determine expected result shape
        if expected_shape == (3,):
            expected_result = self.results
        elif expected_shape == (1, 3):
            expected_result = self.results.reshape(1, 3)
        elif expected_shape == (M, 3):
            expected_result = np.full((M, 3), self.results)
        elif expected_shape == (N, 3):
            expected_result = np.full((N, 3), self.results)
        elif expected_shape == (N, M, 3):
            expected_result = np.full((N, M, 3), self.results)
        else:
            expected_result = self.results

        np.testing.assert_allclose(out, expected_result, atol=self.tolerance["atol"], rtol=self.tolerance["rtol"])

    def test_monostatic_core_error_case(self) -> None:
        """Test direct_geocoding_monostatic_core error handling for mismatched dimensions."""
        # case: N range (M,), N doppler freqs (N,), mismatched frequencies/ranges
        with pytest.raises(RuntimeError):
            direct_geocoding_monostatic_core(
                sensor_positions=np.full((self.N, 3), self.sensor_position),
                sensor_velocities=np.full((self.N, 3), self.sensor_velocity),
                initial_guesses=np.full((self.N, 3), self.initial_guess),
                range_times=np.repeat(self.range_time[0], self.M),
                doppler_frequencies=np.repeat(self.doppler_freq, self.N),
                wavelength=self.wavelength,
                altitude=self.geodetic_altitude,
            )


class TestDirectGeocodingRangeVectorizedMonostaticCore:
    """Testing direct geocoding monostatic range vectorized core with various input combinations using parametrize."""

    @pytest.fixture(autouse=True)
    def setup(self, direct_geocoding_test_data: dict) -> None:
        """Setting up variables for testing"""
        self.sensor_position = direct_geocoding_test_data["sensor_position"]
        self.sensor_velocity = direct_geocoding_test_data["sensor_velocity"]
        self.initial_guess = direct_geocoding_test_data["initial_guess"]
        self.range_time = direct_geocoding_test_data["range_time"]
        self.doppler_freq = direct_geocoding_test_data["doppler_frequency"]
        self.geodetic_altitude = direct_geocoding_test_data["geodetic_altitude"]
        self.wavelength = direct_geocoding_test_data["wavelength"]
        self.N = direct_geocoding_test_data["az_reps"]
        self.M = direct_geocoding_test_data["high_rng_reps"]
        self.tolerance = direct_geocoding_test_data["tolerance"]
        self.results = direct_geocoding_test_data["expected_ground_points"]

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(
                {
                    "name": "case0a: 1 pos (3,), 1 vel (3,), 1 guess (3,), 1 rng time",
                    "positions": "sensor_position",
                    "velocities": "sensor_velocity",
                    "initial_guesses": "initial_guess",
                    "range_times": "range_time_0",
                    "doppler_frequencies": "doppler_freq",
                    "expected_ndim": 1,
                    "expected_shape": (3,),
                },
                id="case0a",
            ),
            pytest.param(
                {
                    "name": "case0b: 1 pos (1,3), 1 vel (1,3), 1 guess (1,3), 1 rng time",
                    "positions": "sensor_position_1_3",
                    "velocities": "sensor_velocity_1_3",
                    "initial_guesses": "initial_guess_1_3",
                    "range_times": "range_time_0",
                    "doppler_frequencies": "doppler_freq",
                    "expected_ndim": 2,
                    "expected_shape": (1, 3),
                },
                id="case0b",
            ),
            pytest.param(
                {
                    "name": "case0c: 1 pos (3,), 1 vel (3,), 1 guess (3,), M range times",
                    "positions": "sensor_position",
                    "velocities": "sensor_velocity",
                    "initial_guesses": "initial_guess",
                    "range_times": "range_time_M",
                    "doppler_frequencies": "doppler_freq",
                    "expected_ndim": 2,
                    "expected_shape": ("M", 3),
                },
                id="case0c",
            ),
            pytest.param(
                {
                    "name": "case0d: 1 pos (1,3), 1 vel (1,3), 1 guess (1,3), M range times",
                    "positions": "sensor_position_1_3",
                    "velocities": "sensor_velocity_1_3",
                    "initial_guesses": "initial_guess_1_3",
                    "range_times": "range_time_M",
                    "doppler_frequencies": "doppler_freq",
                    "expected_ndim": 2,
                    "expected_shape": ("M", 3),
                },
                id="case0d",
            ),
            pytest.param(
                {
                    "name": "case0e: 1 pos (1,3), 1 vel (1,3), M guess (M,3), M range times",
                    "positions": "sensor_position_1_3",
                    "velocities": "sensor_velocity_1_3",
                    "initial_guesses": "initial_guess_M_3",
                    "range_times": "range_time_M",
                    "doppler_frequencies": "doppler_freq",
                    "expected_ndim": 2,
                    "expected_shape": ("M", 3),
                },
                id="case0e",
            ),
            pytest.param(
                {
                    "name": "case0f: 1 pos (3,), 1 vel (3,), 1 guess (3,), M range times, M doppler freqs",
                    "positions": "sensor_position",
                    "velocities": "sensor_velocity",
                    "initial_guesses": "initial_guess",
                    "range_times": "range_time_M",
                    "doppler_frequencies": "doppler_freq_M",
                    "expected_ndim": 2,
                    "expected_shape": ("M", 3),
                },
                id="case0f",
            ),
            pytest.param(
                {
                    "name": "case0g: 1 pos (1,3), 1 vel (1,3), 1 guess (1,3), M range times, M doppler freqs",
                    "positions": "sensor_position_1_3",
                    "velocities": "sensor_velocity_1_3",
                    "initial_guesses": "initial_guess_1_3",
                    "range_times": "range_time_M",
                    "doppler_frequencies": "doppler_freq_M",
                    "expected_ndim": 2,
                    "expected_shape": ("M", 3),
                },
                id="case0g",
            ),
            pytest.param(
                {
                    "name": "case0h: 1 pos (1,3), 1 vel (1,3), M guess (M,3), M range times, M doppler freqs",
                    "positions": "sensor_position_1_3",
                    "velocities": "sensor_velocity_1_3",
                    "initial_guesses": "initial_guess_M_3",
                    "range_times": "range_time_M",
                    "doppler_frequencies": "doppler_freq_M",
                    "expected_ndim": 2,
                    "expected_shape": ("M", 3),
                },
                id="case0h",
            ),
            pytest.param(
                {
                    "name": "case1a: N pos (N,3), N vel (N,3), N guess (N,3), 1 rng time",
                    "positions": "sensor_position_N_3",
                    "velocities": "sensor_velocity_N_3",
                    "initial_guesses": "initial_guess_N_3",
                    "range_times": "range_time_0",
                    "doppler_frequencies": "doppler_freq",
                    "expected_ndim": 2,
                    "expected_shape": ("N", 3),
                },
                id="case1a",
            ),
            pytest.param(
                {
                    "name": "case1b: N pos (N,3), N vel (N,3), N guess (N,3), M rng times",
                    "positions": "sensor_position_N_3",
                    "velocities": "sensor_velocity_N_3",
                    "initial_guesses": "initial_guess_N_3",
                    "range_times": "range_time_M",
                    "doppler_frequencies": "doppler_freq",
                    "expected_ndim": 3,
                    "expected_shape": ("N", "M", 3),
                },
                id="case1b",
            ),
            pytest.param(
                {
                    "name": "case1c: N pos (N,3), N vel (N,3), M guess (M,3), M rng times",
                    "positions": "sensor_position_N_3",
                    "velocities": "sensor_velocity_N_3",
                    "initial_guesses": "initial_guess_M_3",
                    "range_times": "range_time_M",
                    "doppler_frequencies": "doppler_freq",
                    "expected_ndim": 3,
                    "expected_shape": ("N", "M", 3),
                },
                id="case1c",
            ),
            pytest.param(
                {
                    "name": "case1d: N pos (N,3), N vel (N,3), N guess (N,3), M rng times, M doppler freqs",
                    "positions": "sensor_position_N_3",
                    "velocities": "sensor_velocity_N_3",
                    "initial_guesses": "initial_guess_N_3",
                    "range_times": "range_time_M",
                    "doppler_frequencies": "doppler_freq_M",
                    "expected_ndim": 3,
                    "expected_shape": ("N", "M", 3),
                },
                id="case1d",
            ),
            pytest.param(
                {
                    "name": "case1e: N pos (N,3), N vel (N,3), M guess (M,3), M rng times, M doppler freqs",
                    "positions": "sensor_position_N_3",
                    "velocities": "sensor_velocity_N_3",
                    "initial_guesses": "initial_guess_M_3",
                    "range_times": "range_time_M",
                    "doppler_frequencies": "doppler_freq_M",
                    "expected_ndim": 3,
                    "expected_shape": ("N", "M", 3),
                },
                id="case1e",
            ),
        ],
    )
    def test_monostatic_core_cases(self, case: dict) -> None:
        """Test direct_geocoding_monostatic_core_range_vectorized with various input combinations."""
        N, M = self.N, self.M

        value_map = {
            "sensor_position": self.sensor_position,
            "sensor_position_1_3": self.sensor_position.reshape(1, 3),
            "sensor_position_N_3": np.full((N, 3), self.sensor_position),
            "sensor_velocity": self.sensor_velocity,
            "sensor_velocity_1_3": self.sensor_velocity.reshape(1, 3),
            "sensor_velocity_N_3": np.full((N, 3), self.sensor_velocity),
            "initial_guess": self.initial_guess,
            "initial_guess_1_3": self.initial_guess.reshape(1, 3),
            "initial_guess_N_3": np.full((N, 3), self.initial_guess),
            "initial_guess_M_3": np.full((M, 3), self.initial_guess),
            "range_time_0": self.range_time[0],
            "range_time_M": np.repeat(self.range_time[0], M),
            "doppler_freq": self.doppler_freq,
            "doppler_freq_M": np.repeat(self.doppler_freq, M),
        }

        positions = value_map[case["positions"]]
        velocities = value_map[case["velocities"]]
        initial_guesses = value_map[case["initial_guesses"]]
        range_times = value_map[case["range_times"]]
        doppler_frequencies = value_map[case["doppler_frequencies"]]

        expected_shape = tuple(N if dim == "N" else M if dim == "M" else dim for dim in case["expected_shape"])

        out = direct_geocoding_monostatic_core_range_vectorized(
            sensor_positions=positions,
            sensor_velocities=velocities,
            initial_guesses=initial_guesses,
            range_times=range_times,
            doppler_frequencies=doppler_frequencies,
            wavelength=self.wavelength,
            altitude=self.geodetic_altitude,
        )

        assert out.ndim == case["expected_ndim"]
        assert out.shape == expected_shape

        # Determine expected result shape
        if expected_shape == (3,):
            expected_result = self.results
        elif expected_shape == (1, 3):
            expected_result = self.results.reshape(1, 3)
        elif expected_shape == (M, 3):
            expected_result = np.full((M, 3), self.results)
        elif expected_shape == (N, 3):
            expected_result = np.full((N, 3), self.results)
        elif expected_shape == (N, M, 3):
            expected_result = np.full((N, M, 3), self.results)
        else:
            expected_result = self.results

        np.testing.assert_allclose(out, expected_result, atol=self.tolerance["atol"], rtol=self.tolerance["rtol"])

    def test_monostatic_core_error_case(self) -> None:
        """Test direct_geocoding_monostatic_core error handling for mismatched dimensions."""
        # case: M range (M,), N doppler freqs (N,), mismatched frequencies/ranges
        with pytest.raises(RuntimeError):
            direct_geocoding_monostatic_core(
                sensor_positions=np.full((self.N, 3), self.sensor_position),
                sensor_velocities=np.full((self.N, 3), self.sensor_velocity),
                initial_guesses=np.full((self.N, 3), self.initial_guess),
                range_times=np.repeat(self.range_time[0], self.M),
                doppler_frequencies=np.repeat(self.doppler_freq, self.N),
                wavelength=self.wavelength,
                altitude=self.geodetic_altitude,
            )

        # case: M range (M,), Q initial guesses (Q,), mismatched initial guesses/ranges
        with pytest.raises(RuntimeError):
            direct_geocoding_monostatic_core(
                sensor_positions=np.full((self.N, 3), self.sensor_position),
                sensor_velocities=np.full((self.N, 3), self.sensor_velocity),
                initial_guesses=np.full((17, 3), self.initial_guess),
                range_times=np.repeat(self.range_time[0], self.M),
                doppler_frequencies=np.repeat(self.doppler_freq, self.N),
                wavelength=self.wavelength,
                altitude=self.geodetic_altitude,
            )


class TestNewtonForDirectGeocodingMonostatic:
    """Testing Newton method for direct geocoding monostatic using parametrize."""

    @pytest.fixture(autouse=True)
    def setup(self, direct_geocoding_test_data: dict) -> None:
        self.sensor_position = direct_geocoding_test_data["sensor_position"]
        self.sensor_velocity = direct_geocoding_test_data["sensor_velocity"]
        self.initial_guess = direct_geocoding_test_data["initial_guess"]
        self.range_time = float(direct_geocoding_test_data["range_time"][0])
        self.doppler_freq = direct_geocoding_test_data["doppler_frequency"]
        self.geodetic_altitude = direct_geocoding_test_data["geodetic_altitude"]
        self.wavelength = direct_geocoding_test_data["wavelength"]
        self.tolerance = direct_geocoding_test_data["tolerance"]
        self.results = direct_geocoding_test_data["expected_ground_points"]

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(
                {
                    "name": "case0a: 1 pos (3,), 1 vel (3,), 1 init guess (3,)",
                    "positions": "sensor_position",
                    "velocities": "sensor_velocity",
                    "initial_guesses": "initial_guess",
                    "expected_ndim": 1,
                    "expected_shape": (3,),
                },
                id="case0a",
            ),
            pytest.param(
                {
                    "name": "case0b: 1 pos (1,3), 1 vel (1,3), 1 init guess (1,3)",
                    "positions": "sensor_position_1_3",
                    "velocities": "sensor_velocity_1_3",
                    "initial_guesses": "initial_guess_1_3",
                    "expected_ndim": 2,
                    "expected_shape": (1, 3),
                },
                id="case0b",
            ),
            pytest.param(
                {
                    "name": "case1: N pos (N,3), N vel (N,3), N init guess (N,3)",
                    "positions": "sensor_position_4_3",
                    "velocities": "sensor_velocity_4_3",
                    "initial_guesses": "initial_guess_4_3",
                    "expected_ndim": 2,
                    "expected_shape": (4, 3),
                },
                id="case1",
            ),
        ],
    )
    def test_newton_for_geocoding_array_cases(self, case: dict) -> None:
        """Testing Newton for geocoding with array inputs."""
        value_map = {
            "sensor_position": self.sensor_position,
            "sensor_position_1_3": self.sensor_position.reshape(1, 3),
            "sensor_position_4_3": np.full((4, 3), self.sensor_position),
            "sensor_velocity": self.sensor_velocity,
            "sensor_velocity_1_3": self.sensor_velocity.reshape(1, 3),
            "sensor_velocity_4_3": np.full((4, 3), self.sensor_velocity),
            "initial_guess": self.initial_guess,
            "initial_guess_1_3": self.initial_guess.reshape(1, 3),
            "initial_guess_4_3": np.full((4, 3), self.initial_guess),
        }

        positions = value_map[case["positions"]]
        velocities = value_map[case["velocities"]]
        initial_guesses = value_map[case["initial_guesses"]]
        expected_shape = case["expected_shape"]

        out = _direct_geocoding_monostatic_newton(
            sensor_positions=positions,
            sensor_velocities=velocities,
            initial_guesses=initial_guesses,
            range_times=self.range_time,
            altitude=self.geodetic_altitude,
            wavelength=self.wavelength,
            doppler_frequencies=self.doppler_freq,
        )

        assert out.ndim == case["expected_ndim"]
        assert out.shape == expected_shape

        if expected_shape == (3,):
            expected_result = self.results
        elif expected_shape == (1, 3):
            expected_result = self.results.reshape(1, 3)
        elif expected_shape == (4, 3):
            expected_result = np.full((4, 3), self.results)
        else:
            expected_result = self.results

        np.testing.assert_allclose(out, expected_result, atol=self.tolerance["atol"], rtol=self.tolerance["rtol"])


class TestDirectGeocodingMonostaticInit:
    """Testing direct_geocoding_monostatic_init with various input combinations using parametrize."""

    @pytest.fixture(autouse=True)
    def setup(self, direct_geocoding_test_data: dict) -> None:
        self.sensor_position = direct_geocoding_test_data["sensor_position"]
        self.sensor_velocity = direct_geocoding_test_data["sensor_velocity"]
        self.initial_guess = direct_geocoding_test_data["initial_guess"]
        self.range_distance = direct_geocoding_test_data["range_distance"]
        self.look_direction = direct_geocoding_test_data["look_direction"]
        self.doppler_freq = direct_geocoding_test_data["doppler_frequency"]
        self.geodetic_altitude = direct_geocoding_test_data["geodetic_altitude"]
        self.wavelength = direct_geocoding_test_data["wavelength"]
        self.N = direct_geocoding_test_data["az_reps"]
        self.tolerance = direct_geocoding_test_data["tolerance"]
        self.results = direct_geocoding_test_data["expected_monostatic_init"]

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(
                {
                    "name": "case0a: 1 sensor pos (3,), 1 sensor vel (3,)",
                    "positions": "sensor_position",
                    "velocities": "sensor_velocity",
                    "expected_ndim": 1,
                    "expected_shape": (3,),
                    "expected_result": "results",
                },
                id="case0a",
            ),
            pytest.param(
                {
                    "name": "case0b: 1 sensor pos (1,3), 1 sensor vel (3,)",
                    "positions": "sensor_position_1_3",
                    "velocities": "sensor_velocity",
                    "expected_ndim": 2,
                    "expected_shape": (1, 3),
                    "expected_result": "results_1_3",
                },
                id="case0b",
            ),
            pytest.param(
                {
                    "name": "case0c: 1 sensor pos (3,), 1 sensor vel (1,3)",
                    "positions": "sensor_position",
                    "velocities": "sensor_velocity_1_3",
                    "expected_ndim": 2,
                    "expected_shape": (1, 3),
                    "expected_result": "results_1_3",
                },
                id="case0c",
            ),
            pytest.param(
                {
                    "name": "case0d: 1 sensor pos (1,3), 1 sensor vel (1,3)",
                    "positions": "sensor_position_1_3",
                    "velocities": "sensor_velocity_1_3",
                    "expected_ndim": 2,
                    "expected_shape": (1, 3),
                    "expected_result": "results_1_3",
                },
                id="case0d",
            ),
            pytest.param(
                {
                    "name": "case1a: N sensor pos (N,3), 1 sensor vel (3,)",
                    "positions": "sensor_position_N_3",
                    "velocities": "sensor_velocity",
                    "expected_ndim": 2,
                    "expected_shape": ("N", 3),
                    "expected_result": "results_N_3",
                },
                id="case1a",
            ),
            pytest.param(
                {
                    "name": "case1b: N sensor pos (N,3), 1 sensor vel (1,3)",
                    "positions": "sensor_position_N_3",
                    "velocities": "sensor_velocity_1_3",
                    "expected_ndim": 2,
                    "expected_shape": ("N", 3),
                    "expected_result": "results_N_3",
                },
                id="case1b",
            ),
            pytest.param(
                {
                    "name": "case1c: 1 sensor pos (3,), N sensor vel (N,3)",
                    "positions": "sensor_position",
                    "velocities": "sensor_velocity_N_3",
                    "expected_ndim": 2,
                    "expected_shape": ("N", 3),
                    "expected_result": "results_N_3",
                },
                id="case1c",
            ),
            pytest.param(
                {
                    "name": "case1d: 1 sensor pos (1,3), N sensor vel (N,3)",
                    "positions": "sensor_position_1_3",
                    "velocities": "sensor_velocity_N_3",
                    "expected_ndim": 2,
                    "expected_shape": ("N", 3),
                    "expected_result": "results_N_3",
                },
                id="case1d",
            ),
            pytest.param(
                {
                    "name": "case1e: N sensor pos (N,3), N sensor vel (N,3)",
                    "positions": "sensor_position_N_3",
                    "velocities": "sensor_velocity_N_3",
                    "expected_ndim": 2,
                    "expected_shape": ("N", 3),
                    "expected_result": "results_N_3",
                },
                id="case1e",
            ),
        ],
    )
    def test_direct_geocoding_monostatic_init_cases(self, case: dict) -> None:
        """Testing direct_geocoding_monostatic_init with various dimension combinations."""
        N = self.N

        value_map = {
            "sensor_position": self.sensor_position,
            "sensor_position_1_3": self.sensor_position.reshape(1, 3),
            "sensor_position_N_3": np.full((N, 3), self.sensor_position),
            "sensor_velocity": self.sensor_velocity,
            "sensor_velocity_1_3": self.sensor_velocity.reshape(1, 3),
            "sensor_velocity_N_3": np.full((N, 3), self.sensor_velocity),
            "results": self.results,
            "results_1_3": self.results.reshape(1, 3),
            "results_N_3": np.full((N, 3), self.results),
        }

        positions = value_map[case["positions"]]
        velocities = value_map[case["velocities"]]
        expected_shape = tuple(N if dim == "N" else dim for dim in case["expected_shape"])
        expected_result = value_map[case["expected_result"]]

        out = direct_geocoding_init(
            sensor_positions=positions,
            sensor_velocities=velocities,
            range_distance=self.range_distance,
            look_direction=self.look_direction,
        )
        assert out.ndim == case["expected_ndim"]
        assert out.shape == expected_shape
        np.testing.assert_allclose(out, expected_result, atol=self.tolerance["atol"], rtol=self.tolerance["rtol"])
