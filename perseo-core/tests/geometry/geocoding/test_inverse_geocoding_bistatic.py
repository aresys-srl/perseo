# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for geometry/geocoding/inverse_geocoding.py and inverse_geocoding_core.py bistatic functionalities"""

from __future__ import annotations

import numpy as np
import pytest

from perseo_core.geometry.geocoding.inverse_geocoding import inverse_geocoding_bistatic
from perseo_core.geometry.geocoding.inverse_geocoding_core import (
    inverse_geocoding_bistatic_init_core,
)
from perseo_core.timing.precise_datetime import PreciseDateTime


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


class TestInverseGeocodingBistatic:
    """Testing inverse geocoding bistatic functionality using parametrize."""

    @pytest.fixture(autouse=True)
    def setup(self, inverse_geocoding_test_data):
        self.trajectory = inverse_geocoding_test_data["trajectory"]
        self.wavelength = inverse_geocoding_test_data["wavelength"]
        self.doppler_freq = inverse_geocoding_test_data["doppler_frequency"]
        self.init_guess = inverse_geocoding_test_data["init_guess"]
        self.ground_point = inverse_geocoding_test_data["ground_point"]
        self.tolerance = inverse_geocoding_test_data["tolerance"]
        self.residual_tolerance = inverse_geocoding_test_data["residual_tolerance"]
        self.N = inverse_geocoding_test_data["az_reps"]
        self.M = inverse_geocoding_test_data["rng_reps"]
        self.azimuth_res = inverse_geocoding_test_data["expected_azimuth"]
        self.range_res = inverse_geocoding_test_data["expected_range"]

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(
                {
                    "name": "case0a: scalar",
                    "ground_points": "ground_point",
                    "doppler_freqs": "doppler_freq",
                    "init_guess": "init_guess",
                    "init_guess_time_step": None,
                    "expected_az_size": None,
                    "expected_rng_size": None,
                },
                id="case0a",
            ),
            pytest.param(
                {
                    "name": "case0b: 1x3 array",
                    "ground_points": "ground_point_1_3",
                    "doppler_freqs": "doppler_freq",
                    "init_guess": "init_guess",
                    "init_guess_time_step": None,
                    "expected_az_size": 1,
                    "expected_rng_size": 1,
                },
                id="case0b",
            ),
            pytest.param(
                {
                    "name": "case0c: 1x3 array init array",
                    "ground_points": "ground_point_1_3",
                    "doppler_freqs": "doppler_freq",
                    "init_guess": "init_guess_1",
                    "init_guess_time_step": None,
                    "expected_az_size": 1,
                    "expected_rng_size": 1,
                },
                id="case0c",
            ),
            pytest.param(
                {
                    "name": "case0d: scalar init array",
                    "ground_points": "ground_point",
                    "doppler_freqs": "doppler_freq",
                    "init_guess": "init_guess_1",
                    "init_guess_time_step": None,
                    "expected_az_size": 1,
                    "expected_rng_size": 1,
                },
                id="case0d",
            ),
            pytest.param(
                {
                    "name": "case1a: Nx3 array",
                    "ground_points": "ground_point_N_3",
                    "doppler_freqs": "doppler_freq",
                    "init_guess": "init_guess",
                    "init_guess_time_step": None,
                    "expected_az_size": "N",
                    "expected_rng_size": "N",
                },
                id="case1a",
            ),
            pytest.param(
                {
                    "name": "case1b: Nx3 init array",
                    "ground_points": "ground_point_N_3",
                    "doppler_freqs": "doppler_freq",
                    "init_guess": "init_guess_1",
                    "init_guess_time_step": None,
                    "expected_az_size": "N",
                    "expected_rng_size": "N",
                },
                id="case1b",
            ),
            pytest.param(
                {
                    "name": "case1c: Nx3 N init",
                    "ground_points": "ground_point_N_3",
                    "doppler_freqs": "doppler_freq",
                    "init_guess": "init_guess_N",
                    "init_guess_time_step": None,
                    "expected_az_size": "N",
                    "expected_rng_size": "N",
                },
                id="case1c",
            ),
            pytest.param(
                {
                    "name": "case2a: scalar N init",
                    "ground_points": "ground_point",
                    "doppler_freqs": "doppler_freq",
                    "init_guess": "init_guess_N",
                    "init_guess_time_step": None,
                    "expected_az_size": "N",
                    "expected_rng_size": "N",
                },
                id="case2a",
            ),
            pytest.param(
                {
                    "name": "case2b: 1x3 N init",
                    "ground_points": "ground_point_1_3",
                    "doppler_freqs": "doppler_freq",
                    "init_guess": "init_guess_N",
                    "init_guess_time_step": None,
                    "expected_az_size": "N",
                    "expected_rng_size": "N",
                },
                id="case2b",
            ),
            pytest.param(
                {
                    "name": "case3a: scalar M doppler",
                    "ground_points": "ground_point",
                    "doppler_freqs": "doppler_freq_M",
                    "init_guess": "init_guess",
                    "init_guess_time_step": None,
                    "expected_az_size": "M",
                    "expected_rng_size": "M",
                },
                id="case3a",
            ),
            pytest.param(
                {
                    "name": "case3b: Nx3 N doppler",
                    "ground_points": "ground_point_N_3",
                    "doppler_freqs": "doppler_freq",
                    "init_guess": "init_guess",
                    "init_guess_time_step": None,
                    "expected_az_size": "N",
                    "expected_rng_size": "N",
                },
                id="case3b",
            ),
            pytest.param(
                {
                    "name": "case3c: Nx3 N doppler N init",
                    "ground_points": "ground_point_N_3",
                    "doppler_freqs": "doppler_freq_N",
                    "init_guess": "init_guess_N",
                    "init_guess_time_step": None,
                    "expected_az_size": "N",
                    "expected_rng_size": "N",
                },
                id="case3c",
            ),
            pytest.param(
                {
                    "name": "case3d: scalar M doppler M init",
                    "ground_points": "ground_point",
                    "doppler_freqs": "doppler_freq_M",
                    "init_guess": "init_guess_M",
                    "init_guess_time_step": None,
                    "expected_az_size": "M",
                    "expected_rng_size": "M",
                },
                id="case3d",
            ),
            pytest.param(
                {
                    "name": "case3e: 1x3 M doppler M init",
                    "ground_points": "ground_point_1_3",
                    "doppler_freqs": "doppler_freq_M",
                    "init_guess": "init_guess_M",
                    "init_guess_time_step": None,
                    "expected_az_size": "M",
                    "expected_rng_size": "M",
                },
                id="case3e",
            ),
            pytest.param(
                {
                    "name": "case4c: Nx3 time step",
                    "ground_points": "ground_point_N_3",
                    "doppler_freqs": "doppler_freq",
                    "init_guess": None,
                    "init_guess_time_step": 1,
                    "expected_az_size": "N",
                    "expected_rng_size": "N",
                },
                id="case4c",
            ),
            pytest.param(
                {
                    "name": "case4d: Nx3 N doppler time step",
                    "ground_points": "ground_point_N_3",
                    "doppler_freqs": "doppler_freq_N",
                    "init_guess": None,
                    "init_guess_time_step": "time_step",
                    "expected_az_size": "N",
                    "expected_rng_size": "N",
                },
                id="case4d",
            ),
            pytest.param(
                {
                    "name": "case4e: scalar M doppler time step",
                    "ground_points": "ground_point",
                    "doppler_freqs": "doppler_freq_M",
                    "init_guess": None,
                    "init_guess_time_step": 1,
                    "expected_az_size": "M",
                    "expected_rng_size": "M",
                },
                id="case4e",
            ),
            pytest.param(
                {
                    "name": "case4f: 1x3 M doppler time step",
                    "ground_points": "ground_point_1_3",
                    "doppler_freqs": "doppler_freq_M",
                    "init_guess": None,
                    "init_guess_time_step": "time_step",
                    "expected_az_size": "M",
                    "expected_rng_size": "M",
                },
                id="case4f",
            ),
        ],
    )
    def test_inverse_geocoding_bistatic_cases(self, case) -> None:
        """Testing inverse_geocoding_bistatic success cases."""
        N, M = self.N, self.M
        time_step = self.trajectory.times[1] - self.trajectory.times[0]

        value_map = {
            "ground_point": self.ground_point,
            "ground_point_1_3": self.ground_point.reshape(1, 3),
            "ground_point_N_3": np.full((N, 3), self.ground_point),
            "doppler_freq": self.doppler_freq,
            "doppler_freq_M": np.repeat(self.doppler_freq, M),
            "doppler_freq_N": np.repeat(self.doppler_freq, N),
            "init_guess": self.init_guess,
            "init_guess_1": np.array([self.init_guess]),
            "init_guess_N": np.repeat(self.init_guess, N),
            "init_guess_M": np.repeat(self.init_guess, M),
            "time_step": time_step,
        }

        ground_points = value_map[case["ground_points"]]
        doppler_freqs = value_map[case["doppler_freqs"]]
        init_guess = value_map.get(case["init_guess"], case["init_guess"])
        init_guess_time_step = value_map.get(case["init_guess_time_step"], case["init_guess_time_step"])
        expected_az_size = (
            N if case["expected_az_size"] == "N" else M if case["expected_az_size"] == "M" else case["expected_az_size"]
        )
        expected_rng_size = (
            N
            if case["expected_rng_size"] == "N"
            else M
            if case["expected_rng_size"] == "M"
            else case["expected_rng_size"]
        )

        if init_guess is not None:
            az_times, rng_times = inverse_geocoding_bistatic(
                trajectory_rx=self.trajectory,
                trajectory_tx=self.trajectory,
                ground_points=ground_points,
                frequencies_doppler_centroid=doppler_freqs,
                az_initial_time_guesses=init_guess,
                wavelength=self.wavelength,
            )
        else:
            az_times, rng_times = inverse_geocoding_bistatic(
                trajectory_rx=self.trajectory,
                trajectory_tx=self.trajectory,
                ground_points=ground_points,
                frequencies_doppler_centroid=doppler_freqs,
                wavelength=self.wavelength,
                init_guess_search_time_step=init_guess_time_step,
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
            ground_points=ground_points,
            wavelength=self.wavelength,
            freq_doppler=self.doppler_freq,
        )

        np.testing.assert_allclose(
            doppler_residual_new,
            np.zeros_like(doppler_residual_new),
            atol=self.residual_tolerance["atol"],
            rtol=self.residual_tolerance["rtol"],
        )

        if expected_az_size is None:
            assert isinstance(az_times, PreciseDateTime)
            assert isinstance(rng_times, (float, np.floating))
            assert abs(az_times - self.azimuth_res) < self.tolerance["atol"]
            assert abs(float(rng_times) - self.range_res) < self.tolerance["atol"]
        else:
            assert isinstance(az_times, np.ndarray)
            assert isinstance(rng_times, np.ndarray)
            assert az_times.ndim == 1
            assert rng_times.ndim == 1
            assert az_times.size == expected_az_size
            assert rng_times.size == expected_rng_size
            assert all(isinstance(value, PreciseDateTime) for value in az_times)
            assert all(isinstance(value, (float, np.floating)) for value in rng_times)

            delta_az = np.array(az_times - self.azimuth_res, dtype=float)
            np.testing.assert_allclose(
                delta_az, np.zeros_like(delta_az), atol=self.tolerance["atol"], rtol=self.tolerance["rtol"]
            )
            np.testing.assert_allclose(
                rng_times,
                np.repeat(self.range_res, expected_rng_size),
                atol=self.tolerance["atol"],
                rtol=self.tolerance["rtol"],
            )

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(
                {
                    "name": "case4a: error Nx3 M init",
                    "ground_points": "ground_point_N_3",
                    "doppler_freqs": "doppler_freq",
                    "init_guess": "init_guess_M",
                    "init_guess_time_step": None,
                },
                id="case4a",
            ),
            pytest.param(
                {
                    "name": "case4b: error Nx3 M doppler",
                    "ground_points": "ground_point_N_3",
                    "doppler_freqs": "doppler_freq_M",
                    "init_guess": "init_guess",
                    "init_guess_time_step": None,
                },
                id="case4b",
            ),
            pytest.param(
                {
                    "name": "error: no init guess no time step",
                    "ground_points": "ground_point_1_3",
                    "doppler_freqs": "doppler_freq_M",
                    "init_guess": None,
                    "init_guess_time_step": None,
                },
                id="no-init-no-time-step",
            ),
        ],
    )
    def test_inverse_geocoding_bistatic_error_cases(self, case) -> None:
        """Testing inverse_geocoding_bistatic error cases."""
        N, M = self.N, self.M
        value_map = {
            "ground_point_N_3": np.full((N, 3), self.ground_point),
            "ground_point_1_3": self.ground_point.reshape(1, 3),
            "doppler_freq": self.doppler_freq,
            "doppler_freq_M": np.repeat(self.doppler_freq, M),
            "init_guess": self.init_guess,
            "init_guess_M": np.repeat(self.init_guess, M),
        }

        ground_points = value_map[case["ground_points"]]
        doppler_freqs = value_map[case["doppler_freqs"]]
        init_guess = value_map.get(case["init_guess"], case["init_guess"])
        init_guess_time_step = case["init_guess_time_step"]

        with pytest.raises(RuntimeError):
            if init_guess is not None:
                inverse_geocoding_bistatic(
                    trajectory_rx=self.trajectory,
                    trajectory_tx=self.trajectory,
                    ground_points=ground_points,
                    frequencies_doppler_centroid=doppler_freqs,
                    az_initial_time_guesses=init_guess,
                    wavelength=self.wavelength,
                )
            else:
                inverse_geocoding_bistatic(
                    trajectory_rx=self.trajectory,
                    trajectory_tx=self.trajectory,
                    ground_points=ground_points,
                    frequencies_doppler_centroid=doppler_freqs,
                    wavelength=self.wavelength,
                    init_guess_search_time_step=init_guess_time_step,
                )


class TestInverseGeocodingBistaticInit:
    """Testing inverse geocoding bistatic initialization using parametrize."""

    @pytest.fixture(autouse=True)
    def setup(self, inverse_geocoding_test_data):
        self.trajectory = inverse_geocoding_test_data["trajectory"]
        self.wavelength = inverse_geocoding_test_data["wavelength"]
        self.doppler_freq = inverse_geocoding_test_data["doppler_frequency"]
        self.init_guess = inverse_geocoding_test_data["init_guess"]
        self.ground_point = inverse_geocoding_test_data["ground_point"]
        self.tolerance = inverse_geocoding_test_data["tolerance"]
        self.N = inverse_geocoding_test_data["az_reps"]
        self.M = inverse_geocoding_test_data["rng_reps"]
        self.result = inverse_geocoding_test_data["expected_init_guess"]

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(
                {
                    "name": "case0a: scalar",
                    "ground_points": "ground_point",
                    "doppler_freqs": "doppler_freq",
                    "expected_size": None,
                    "should_raise": False,
                },
                id="case0a",
            ),
            pytest.param(
                {
                    "name": "case0b: 1x3",
                    "ground_points": "ground_point_1_3",
                    "doppler_freqs": "doppler_freq",
                    "expected_size": 1,
                    "should_raise": False,
                },
                id="case0b",
            ),
            pytest.param(
                {
                    "name": "case0c: scalar doppler array",
                    "ground_points": "ground_point",
                    "doppler_freqs": "doppler_freq_1",
                    "expected_size": 1,
                    "should_raise": False,
                },
                id="case0c",
            ),
            pytest.param(
                {
                    "name": "case1a: Nx3",
                    "ground_points": "ground_point_N_3",
                    "doppler_freqs": "doppler_freq",
                    "expected_size": "N",
                    "should_raise": False,
                },
                id="case1a",
            ),
            pytest.param(
                {
                    "name": "case1b: Nx3 N doppler",
                    "ground_points": "ground_point_N_3",
                    "doppler_freqs": "doppler_freq_N",
                    "expected_size": "N",
                    "should_raise": False,
                },
                id="case1b",
            ),
            pytest.param(
                {
                    "name": "case2a: scalar M doppler",
                    "ground_points": "ground_point",
                    "doppler_freqs": "doppler_freq_M",
                    "expected_size": "M",
                    "should_raise": False,
                },
                id="case2a",
            ),
            pytest.param(
                {
                    "name": "case2b: 1x3 M doppler",
                    "ground_points": "ground_point_1_3",
                    "doppler_freqs": "doppler_freq_M",
                    "expected_size": "M",
                    "should_raise": False,
                },
                id="case2b",
            ),
            pytest.param(
                {
                    "name": "case3: error Nx3 M doppler",
                    "ground_points": "ground_point_N_3",
                    "doppler_freqs": "doppler_freq_M",
                    "expected_size": None,
                    "should_raise": True,
                },
                id="case3",
            ),
        ],
    )
    def test_inverse_geocoding_bistatic_init_core_cases(self, case) -> None:
        """Parameterized test with 8 cases (7 valid cases + 1 error case)."""
        N, M = self.N, self.M
        value_map = {
            "ground_point": self.ground_point,
            "ground_point_1_3": self.ground_point.reshape(1, 3),
            "ground_point_N_3": np.full((N, 3), self.ground_point),
            "doppler_freq": self.doppler_freq,
            "doppler_freq_1": np.array([self.doppler_freq]),
            "doppler_freq_N": np.repeat(self.doppler_freq, N),
            "doppler_freq_M": np.repeat(self.doppler_freq, M),
        }

        ground_points = value_map[case["ground_points"]]
        doppler_freqs = value_map[case["doppler_freqs"]]
        expected_size = (
            N if case["expected_size"] == "N" else M if case["expected_size"] == "M" else case["expected_size"]
        )

        if case["should_raise"]:
            with pytest.raises(RuntimeError):
                inverse_geocoding_bistatic_init_core(
                    trajectory_rx=self.trajectory,
                    trajectory_tx=self.trajectory,
                    time_axis_rx=self.trajectory.times,
                    time_axis_tx=self.trajectory.times,
                    ground_points=ground_points,
                    frequencies_doppler_centroid=doppler_freqs,
                    wavelength=self.wavelength,
                )
        else:
            az_times = inverse_geocoding_bistatic_init_core(
                trajectory_rx=self.trajectory,
                trajectory_tx=self.trajectory,
                time_axis_rx=self.trajectory.times,
                time_axis_tx=self.trajectory.times,
                ground_points=ground_points,
                frequencies_doppler_centroid=doppler_freqs,
                wavelength=self.wavelength,
            )

            if expected_size is None:
                assert isinstance(az_times, PreciseDateTime)
                assert np.abs(az_times - self.result) < self.tolerance["atol"]
            else:
                assert isinstance(az_times, np.ndarray)
                assert az_times.size == expected_size
                delta_az = np.array(az_times - self.result, dtype=float)
                np.testing.assert_allclose(delta_az, np.zeros_like(delta_az), atol=self.tolerance["atol"], rtol=0)
