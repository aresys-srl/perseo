# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for geometry/geocoding/inverse_geocoding.py and inverse_geocoding_core.py monostatic functionalities"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from perseo_core.geometry.geocoding.inverse_geocoding import (
    inverse_geocoding_monostatic,
    inverse_geocoding_monostatic_init,
)
from perseo_core.geometry.geocoding.inverse_geocoding_core import inverse_geocoding_monostatic_core
from perseo_core.models.trajectory import Trajectory
from perseo_core.timing.precise_datetime import PreciseDateTime


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
    az_times: PreciseDateTime | np.ndarray,
    expected_size: int | None,
    expected_value: PreciseDateTime,
    tolerance: float,
) -> None:
    if expected_size is None:
        assert isinstance(az_times, PreciseDateTime)
        assert abs(az_times - expected_value) < tolerance
        return

    assert isinstance(az_times, np.ndarray)
    assert az_times.ndim == 1
    assert az_times.size == expected_size
    assert all(isinstance(value, PreciseDateTime) for value in az_times)

    delta_az = np.array(az_times - expected_value, dtype=float)
    np.testing.assert_allclose(delta_az, np.zeros_like(delta_az), atol=tolerance, rtol=0)


def _assert_range_output(
    rng_times: float | np.ndarray,
    expected_size: int | None,
    expected_value: float,
    tolerance: float,
) -> None:
    if expected_size is None:
        assert isinstance(rng_times, (float, np.floating))
        assert abs(float(rng_times) - expected_value) < tolerance
        return

    assert isinstance(rng_times, np.ndarray)
    assert rng_times.ndim == 1
    assert rng_times.size == expected_size
    assert all(isinstance(value, (float, np.floating)) for value in rng_times)
    np.testing.assert_allclose(rng_times, np.repeat(expected_value, expected_size), atol=tolerance, rtol=0)


def _assert_inverse_geocoding_output(
    az_times: PreciseDateTime | np.ndarray,
    rng_times: float | np.ndarray,
    expected_size: int | None,
    expected_azimuth: PreciseDateTime,
    expected_range: float,
    az_tolerance: float,
    rng_tolerance: float,
) -> None:
    _assert_azimuth_output(az_times, expected_size, expected_azimuth, az_tolerance)
    _assert_range_output(rng_times, expected_size, expected_range, rng_tolerance)


def _assert_init_output(
    az_times: PreciseDateTime | np.ndarray,
    expected_size: int | None,
    expected_value: PreciseDateTime,
    tolerance: float,
) -> None:
    _assert_azimuth_output(az_times, expected_size, expected_value, tolerance)


def _reshape_ground_points(res: np.ndarray, size: int) -> np.ndarray:
    return np.full((size, 3), res)


class TestInverseGeocodingMonostaticCore:
    """Testing inverse geocoding monostatic core using parametrize."""

    @pytest.fixture(autouse=True)
    def setup(self, inverse_geocoding_test_data):
        self.trajectory = inverse_geocoding_test_data["trajectory"]
        self.wavelength = inverse_geocoding_test_data["wavelength"]
        self.doppler_freq = inverse_geocoding_test_data["doppler_frequency"]
        self.init_guess = inverse_geocoding_test_data["init_guess"]
        self.ground_point = inverse_geocoding_test_data["ground_point"]
        self.N = inverse_geocoding_test_data["az_reps"]
        self.M = inverse_geocoding_test_data["rng_reps"]
        self.azimuth_res = inverse_geocoding_test_data["expected_azimuth_mono"]
        self.range_res = inverse_geocoding_test_data["expected_range_mono"]
        self.tolerance = inverse_geocoding_test_data["tolerance"]
        self.residual_tolerance = inverse_geocoding_test_data["residual_tolerance"]

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(
                {
                    "name": "case0a: 1 ground point (3,), 1 doppler freq, 1 init guess PDT",
                    "ground_points": "ground_point",
                    "frequencies": "doppler_freq",
                    "initial_guesses": "init_guess",
                    "expected_size": None,
                    "residual_ground_points": "ground_point",
                },
                id="case0a",
            ),
            pytest.param(
                {
                    "name": "case0b: 1 ground point (1,3), 1 doppler freq, 1 init guess PDT",
                    "ground_points": "ground_point_1_3",
                    "frequencies": "doppler_freq",
                    "initial_guesses": "init_guess",
                    "expected_size": 1,
                    "residual_ground_points": "ground_point_1_3",
                },
                id="case0b",
            ),
            pytest.param(
                {
                    "name": "case0c: 1 ground point (1,3), 1 doppler freq, 1 init guess PDT",
                    "ground_points": "ground_point_1_3",
                    "frequencies": "doppler_freq",
                    "initial_guesses": "init_guess",
                    "expected_size": 1,
                    "residual_ground_points": "ground_point_1_3",
                },
                id="case0c",
            ),
            pytest.param(
                {
                    "name": "case1a: N ground points (N, 3), 1 doppler freq, 1 init guess PDT",
                    "ground_points": "ground_point_N_3",
                    "frequencies": "doppler_freq",
                    "initial_guesses": "init_guess",
                    "expected_size": "N",
                    "residual_ground_points": "ground_point_N_3",
                },
                id="case1a",
            ),
            pytest.param(
                {
                    "name": "case1b: N ground points (N, 3), 1 doppler freq, N init guesses (N,)",
                    "ground_points": "ground_point_N_3",
                    "frequencies": "doppler_freq",
                    "initial_guesses": "init_guess_N",
                    "expected_size": "N",
                    "residual_ground_points": "ground_point_N_3",
                },
                id="case1b",
            ),
            pytest.param(
                {
                    "name": "case1c: N ground points (N, 3), N doppler freqs (N,), N init guesses (N,)",
                    "ground_points": "ground_point_N_3",
                    "frequencies": "doppler_freq_N",
                    "initial_guesses": "init_guess_N",
                    "expected_size": "N",
                    "residual_ground_points": "ground_point_N_3",
                },
                id="case1c",
            ),
            pytest.param(
                {
                    "name": "case2a: 1 ground point (3,), 1 doppler freq, N init guesses (N,)",
                    "ground_points": "ground_point",
                    "frequencies": "doppler_freq",
                    "initial_guesses": "init_guess_N",
                    "expected_size": "N",
                    "residual_ground_points": "ground_point_N_3",
                },
                id="case2a",
            ),
            pytest.param(
                {
                    "name": "case2b: 1 ground point (1,3), 1 doppler freq, N init guesses (N,)",
                    "ground_points": "ground_point_1_3",
                    "frequencies": "doppler_freq",
                    "initial_guesses": "init_guess_N",
                    "expected_size": "N",
                    "residual_ground_points": "ground_point_N_3",
                },
                id="case2b",
            ),
            pytest.param(
                {
                    "name": "case3: 1 ground point (3,), N doppler freqs (N,), 1 init guess PDT",
                    "ground_points": "ground_point",
                    "frequencies": "doppler_freq_N",
                    "initial_guesses": "init_guess",
                    "expected_size": "N",
                    "residual_ground_points": "ground_point_N_3",
                },
                id="case3",
            ),
        ],
    )
    def test_inverse_geocoding_monostatic_core_cases(self, case) -> None:
        """Testing inverse_geocoding_monostatic_core with parametrize."""
        N = self.N
        value_map = {
            "ground_point": self.ground_point,
            "ground_point_1_3": self.ground_point.reshape(1, 3),
            "ground_point_N_3": _reshape_ground_points(self.ground_point, N),
            "doppler_freq": self.doppler_freq,
            "doppler_freq_N": np.repeat(self.doppler_freq, N),
            "init_guess": self.init_guess,
            "init_guess_N": np.repeat(self.init_guess, N),
        }

        ground_points = value_map[case["ground_points"]]
        frequencies = value_map[case["frequencies"]]
        initial_guesses = value_map[case["initial_guesses"]]
        expected_size = N if case["expected_size"] == "N" else case["expected_size"]
        residual_ground_points = value_map[case["residual_ground_points"]]

        az_times, rng_times = inverse_geocoding_monostatic_core(
            trajectory=self.trajectory,
            ground_points=ground_points,
            frequencies_doppler_centroid=frequencies,
            initial_guesses=initial_guesses,
            wavelength=self.wavelength,
        )
        doppler_residual = _doppler_equation_residual(
            trajectory=self.trajectory,
            ground_points=residual_ground_points,
            az_times=az_times,
            frequency_doppler=self.doppler_freq,
            wavelength=self.wavelength,
        )

        _assert_inverse_geocoding_output(
            az_times,
            rng_times,
            expected_size,
            self.azimuth_res,
            self.range_res,
            self.tolerance["atol"],
            self.tolerance["atol"],
        )
        np.testing.assert_allclose(
            doppler_residual,
            np.zeros_like(doppler_residual),
            atol=self.residual_tolerance["atol"],
            rtol=self.residual_tolerance["rtol"],
        )

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(
                {
                    "name": "case4a: N ground points (N, 3), 1 doppler freq, M init guesses (M,)",
                    "ground_points": "ground_point_N_3",
                    "frequencies": "doppler_freq",
                    "initial_guesses": "init_guess_M",
                },
                id="case4a",
            ),
            pytest.param(
                {
                    "name": "case4b: N ground points (N, 3), M doppler freqs (M,), 1 init guess",
                    "ground_points": "ground_point_N_3",
                    "frequencies": "doppler_freq_M",
                    "initial_guesses": "init_guess",
                },
                id="case4b",
            ),
        ],
    )
    def test_inverse_geocoding_monostatic_core_error_cases(self, case) -> None:
        """Testing inverse_geocoding_monostatic_core error cases with parametrize."""
        N, M = self.N, self.M
        value_map = {
            "ground_point_N_3": _reshape_ground_points(self.ground_point, N),
            "doppler_freq": self.doppler_freq,
            "doppler_freq_M": np.repeat(self.doppler_freq, M),
            "init_guess": self.init_guess,
            "init_guess_M": np.repeat(self.init_guess, M),
        }

        ground_points = value_map[case["ground_points"]]
        frequencies = value_map[case["frequencies"]]
        initial_guesses = value_map[case["initial_guesses"]]

        with pytest.raises(RuntimeError):
            inverse_geocoding_monostatic_core(
                trajectory=self.trajectory,
                ground_points=ground_points,
                frequencies_doppler_centroid=frequencies,
                initial_guesses=initial_guesses,
                wavelength=self.wavelength,
            )


class TestInverseGeocodingMonostatic:
    """Testing inverse geocoding monostatic using parametrize."""

    @pytest.fixture(autouse=True)
    def setup(self, inverse_geocoding_test_data):
        self.trajectory = inverse_geocoding_test_data["trajectory"]
        self.time_step = self.trajectory.times[1] - self.trajectory.times[0]
        self.wavelength = inverse_geocoding_test_data["wavelength"]
        self.doppler_freq = inverse_geocoding_test_data["doppler_frequency"]
        self.init_guess = inverse_geocoding_test_data["init_guess"]
        self.ground_point = inverse_geocoding_test_data["ground_point"]
        self.N = inverse_geocoding_test_data["az_reps"]
        self.M = inverse_geocoding_test_data["rng_reps"]
        self.azimuth_res = inverse_geocoding_test_data["expected_azimuth_mono"]
        self.range_res = inverse_geocoding_test_data["expected_range_mono"]
        self.tolerance = inverse_geocoding_test_data["tolerance"]
        self.residual_tolerance = inverse_geocoding_test_data["residual_tolerance"]
        self.rng_tolerance = inverse_geocoding_test_data["low_tolerance"]

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(
                {
                    "name": "case0a: 1 ground point (3,), 1 doppler freq, no init guess",
                    "ground_points": "ground_point",
                    "frequencies": "doppler_freq",
                    "search_time_step": 1,
                    "initial_guesses": None,
                    "expected_size": None,
                },
                id="case0a",
            ),
            pytest.param(
                {
                    "name": "case0b: 1 ground point (1,3), 1 doppler freq, no init guess",
                    "ground_points": "ground_point_1_3",
                    "frequencies": "doppler_freq",
                    "search_time_step": "time_step",
                    "initial_guesses": None,
                    "expected_size": 1,
                },
                id="case0b",
            ),
            pytest.param(
                {
                    "name": "case0c: 1 ground point (3,), 1 doppler freq, 1 init guess",
                    "ground_points": "ground_point",
                    "frequencies": "doppler_freq",
                    "search_time_step": None,
                    "initial_guesses": "init_guess",
                    "expected_size": None,
                },
                id="case0c",
            ),
            pytest.param(
                {
                    "name": "case0d: 1 ground point (3,), 1 doppler freq, 1 init guess (1,)",
                    "ground_points": "ground_point",
                    "frequencies": "doppler_freq",
                    "search_time_step": None,
                    "initial_guesses": "init_guess_1",
                    "expected_size": 1,
                },
                id="case0d",
            ),
            pytest.param(
                {
                    "name": "case1a: 1 ground point (3,), M doppler freqs",
                    "ground_points": "ground_point",
                    "frequencies": "doppler_freq_M",
                    "search_time_step": 1,
                    "initial_guesses": None,
                    "expected_size": "M",
                },
                id="case1a",
            ),
            pytest.param(
                {
                    "name": "case1b: 1 ground point (3,), M doppler freqs, 1 init guess PDT",
                    "ground_points": "ground_point",
                    "frequencies": "doppler_freq_M",
                    "search_time_step": None,
                    "initial_guesses": "init_guess",
                    "expected_size": "M",
                },
                id="case1b",
            ),
            pytest.param(
                {
                    "name": "case1c: 1 ground point (1,3), M doppler freqs",
                    "ground_points": "ground_point_1_3",
                    "frequencies": "doppler_freq_M",
                    "search_time_step": "time_step",
                    "initial_guesses": None,
                    "expected_size": "M",
                },
                id="case1c",
            ),
            pytest.param(
                {
                    "name": "case2a: N ground points (N, 3), 1 doppler freq",
                    "ground_points": "ground_point_N_3",
                    "frequencies": "doppler_freq",
                    "search_time_step": 1,
                    "initial_guesses": None,
                    "expected_size": "N",
                },
                id="case2a",
            ),
            pytest.param(
                {
                    "name": "case2b: N ground points (N, 3), 1 doppler freq, 1 init guess",
                    "ground_points": "ground_point_N_3",
                    "frequencies": "doppler_freq",
                    "search_time_step": None,
                    "initial_guesses": "init_guess",
                    "expected_size": "N",
                },
                id="case2b",
            ),
            pytest.param(
                {
                    "name": "case2c: N ground points (N, 3), 1 doppler freq, N init guesses",
                    "ground_points": "ground_point_N_3",
                    "frequencies": "doppler_freq",
                    "search_time_step": None,
                    "initial_guesses": "init_guess_N",
                    "expected_size": "N",
                },
                id="case2c",
            ),
            pytest.param(
                {
                    "name": "case3a: N ground points (N, 3), N doppler freqs",
                    "ground_points": "ground_point_N_3",
                    "frequencies": "doppler_freq_N",
                    "search_time_step": "time_step",
                    "initial_guesses": None,
                    "expected_size": "N",
                },
                id="case3a",
            ),
            pytest.param(
                {
                    "name": "case3b: N ground points (N, 3), N doppler freqs, 1 init guess",
                    "ground_points": "ground_point_N_3",
                    "frequencies": "doppler_freq_N",
                    "search_time_step": None,
                    "initial_guesses": "init_guess",
                    "expected_size": "N",
                },
                id="case3b",
            ),
            pytest.param(
                {
                    "name": "case3c: N ground points (N, 3), N doppler freqs, N init guesses",
                    "ground_points": "ground_point_N_3",
                    "frequencies": "doppler_freq_N",
                    "search_time_step": None,
                    "initial_guesses": "init_guess_N",
                    "expected_size": "N",
                },
                id="case3c",
            ),
        ],
    )
    def test_inverse_geocoding_monostatic_cases(self, case) -> None:
        """Testing inverse_geocoding_monostatic with parametrize."""
        N, M = self.N, self.M
        value_map = {
            "ground_point": self.ground_point,
            "ground_point_1_3": self.ground_point.reshape(1, 3),
            "ground_point_N_3": _reshape_ground_points(self.ground_point, N),
            "doppler_freq": self.doppler_freq,
            "doppler_freq_M": np.repeat(self.doppler_freq, M),
            "doppler_freq_N": np.repeat(self.doppler_freq, N),
            "init_guess": self.init_guess,
            "init_guess_1": np.array([self.init_guess]),
            "init_guess_N": np.repeat(self.init_guess, N),
            "time_step": self.time_step,
        }

        ground_points = value_map[case["ground_points"]]
        frequencies = value_map[case["frequencies"]]
        search_time_step = value_map.get(case["search_time_step"], case["search_time_step"])
        initial_guesses = value_map.get(case["initial_guesses"], case["initial_guesses"])
        expected_size = (
            N if case["expected_size"] == "N" else M if case["expected_size"] == "M" else case["expected_size"]
        )

        kwargs = {
            "trajectory": self.trajectory,
            "ground_points": ground_points,
            "frequencies_doppler_centroid": frequencies,
            "wavelength": self.wavelength,
        }
        if search_time_step is not None:
            kwargs["init_guess_search_time_step"] = search_time_step
        if initial_guesses is not None:
            kwargs["az_initial_time_guesses"] = initial_guesses

        az_times, rng_times = inverse_geocoding_monostatic(**kwargs)
        _assert_inverse_geocoding_output(
            az_times,
            rng_times,
            expected_size,
            self.azimuth_res,
            self.range_res,
            self.tolerance["atol"],
            self.rng_tolerance["atol"],
        )

    def test_inverse_geocoding_monostatic_error_cases(self) -> None:
        """Testing inverse_geocoding_monostatic error cases."""
        with pytest.raises(RuntimeError):
            inverse_geocoding_monostatic(
                trajectory=self.trajectory,
                ground_points=_reshape_ground_points(self.ground_point, self.N),
                frequencies_doppler_centroid=np.repeat(self.doppler_freq, self.N),
                wavelength=self.wavelength,
            )


class TestInverseGeocodingMonostaticInit:
    """Testing inverse geocoding monostatic init using parametrize."""

    @pytest.fixture(autouse=True)
    def setup(self, inverse_geocoding_test_data):
        self.trajectory = inverse_geocoding_test_data["trajectory"]
        self.wavelength = inverse_geocoding_test_data["wavelength"]
        self.doppler_freq = inverse_geocoding_test_data["doppler_frequency"]
        self.init_guess = inverse_geocoding_test_data["init_guess"]
        self.ground_point = inverse_geocoding_test_data["ground_point"]
        self.N = inverse_geocoding_test_data["az_reps"]
        self.M = inverse_geocoding_test_data["rng_reps"]
        self.azimuth_res = inverse_geocoding_test_data["expected_azimuth_mono"]
        self.range_res = inverse_geocoding_test_data["expected_range_mono"]
        self.tolerance = inverse_geocoding_test_data["tolerance"]
        self.residual_tolerance = inverse_geocoding_test_data["residual_tolerance"]
        self.result = inverse_geocoding_test_data["expected_init_guess_mono"]

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(
                {
                    "name": "case0a: 1 ground point (3,), 1 freq",
                    "ground_points": "ground_point",
                    "frequencies": "doppler_freq",
                    "expected_size": None,
                },
                id="case0a",
            ),
            pytest.param(
                {
                    "name": "case0b: 1 ground point (1, 3), 1 freq",
                    "ground_points": "ground_point_1_3",
                    "frequencies": "doppler_freq",
                    "expected_size": 1,
                },
                id="case0b",
            ),
            pytest.param(
                {
                    "name": "case0c: 1 ground point (1, 3), 1 freq (array)",
                    "ground_points": "ground_point_1_3",
                    "frequencies": "doppler_freq_1",
                    "expected_size": 1,
                },
                id="case0c",
            ),
            pytest.param(
                {
                    "name": "case1: N ground points (N, 3), 1 freq",
                    "ground_points": "ground_point_N_3",
                    "frequencies": "doppler_freq",
                    "expected_size": "N",
                },
                id="case1",
            ),
            pytest.param(
                {
                    "name": "case2: N ground points (N, 3), 1 freq (array)",
                    "ground_points": "ground_point_N_3",
                    "frequencies": "doppler_freq_1",
                    "expected_size": "N",
                },
                id="case2",
            ),
            pytest.param(
                {
                    "name": "case3: 1 ground point (3,), M freq",
                    "ground_points": "ground_point",
                    "frequencies": "doppler_freq_M",
                    "expected_size": "M",
                },
                id="case3",
            ),
            pytest.param(
                {
                    "name": "case4: N ground points (N, 3), N freq",
                    "ground_points": "ground_point_N_3",
                    "frequencies": "doppler_freq_N",
                    "expected_size": "N",
                },
                id="case4",
            ),
        ],
    )
    def test_inverse_geocoding_monostatic_init_cases(self, case) -> None:
        """Testing inverse_geocoding_monostatic_init with parametrize."""
        N, M = self.N, self.M
        value_map = {
            "ground_point": self.ground_point,
            "ground_point_1_3": self.ground_point.reshape(1, 3),
            "ground_point_N_3": _reshape_ground_points(self.ground_point, N),
            "doppler_freq": self.doppler_freq,
            "doppler_freq_1": np.array([self.doppler_freq]),
            "doppler_freq_M": np.repeat(self.doppler_freq, M),
            "doppler_freq_N": np.repeat(self.doppler_freq, N),
        }

        ground_points = value_map[case["ground_points"]]
        frequencies = value_map[case["frequencies"]]
        expected_size = (
            N if case["expected_size"] == "N" else M if case["expected_size"] == "M" else case["expected_size"]
        )

        az_times = inverse_geocoding_monostatic_init(
            trajectory=self.trajectory,
            ground_points=ground_points,
            time_axis=self.trajectory.times,
            frequencies_doppler_centroid=frequencies,
            wavelength=self.wavelength,
        )
        _assert_init_output(
            az_times,
            expected_size,
            self.result,
            self.tolerance["atol"],
        )
