# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for geographical orientation, footprint and ground-cell subsampling of quick-look data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr
from scipy.constants import speed_of_light

from perseo_core.geometry.angles import compute_incidence_angles, compute_look_angles
from perseo_core.geometry.coordinates.conversions import xyz2llh
from perseo_core.geometry.geocoding.direct import direct_geocoding_monostatic
from perseo_core.geometry.velocities import compute_ground_velocity
from perseo_core.imaging.geospatial import orientation
from perseo_core.imaging.geospatial.orientation import (
    compute_sar_data_model_footprint,
    compute_square_ground_cell_subsampling_factors,
    orient_geographically,
)
from perseo_core.imaging.models import AZIMUTH_DIM, RANGE_DIM, SARDataModel, create_sar_data_model
from perseo_core.timing.precise_datetime import PreciseDateTime

if TYPE_CHECKING:
    from perseo_core.geometry.navigation import CubicSplineTrajectory


def _make_data() -> SARDataModel:
    raster = np.arange(4, dtype=float).reshape(2, 2)
    azimuth_times = [
        PreciseDateTime.from_numeric_datetime(2024, 1, 1),
        PreciseDateTime.from_numeric_datetime(2024, 1, 2),
    ]
    return create_sar_data_model(raster, azimuth_times, np.array([0.0, 1.0]))


class TestOrientGeographically:
    def test_north_up_requires_no_flip(self) -> None:
        data = _make_data()
        corners = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
        result = orient_geographically(data, corners)
        np.testing.assert_allclose(result.values, data.values)
        assert result.azimuth.values[0] == data.azimuth.values[0]
        assert result.range.values[0] == data.range.values[0]

    def test_south_up_flips_azimuth(self) -> None:
        data = _make_data()
        corners = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
        result = orient_geographically(data, corners)
        np.testing.assert_allclose(result.values, np.flipud(data.values))
        assert result.azimuth.values[0] == data.azimuth.values[-1]
        assert result.azimuth.values[-1] == data.azimuth.values[0]

    def test_westward_right_edge_flips_range(self) -> None:
        data = _make_data()
        corners = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
        result = orient_geographically(data, corners)
        np.testing.assert_allclose(result.values, np.fliplr(data.values))
        assert result.range.values[0] == data.range.values[-1]
        assert result.range.values[-1] == data.range.values[0]

    def test_both_flips(self) -> None:
        data = _make_data()
        corners = np.array([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        result = orient_geographically(data, corners)
        np.testing.assert_allclose(result.values, np.flipud(np.fliplr(data.values)))
        assert result.azimuth.values[0] == data.azimuth.values[-1]
        assert result.range.values[0] == data.range.values[-1]

    def test_east_heading_requires_no_flip(self) -> None:
        data = _make_data()
        # image "up" points east (azimuth 90 degrees): north component is zero
        corners = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
        result = orient_geographically(data, corners)
        np.testing.assert_allclose(result.values, data.values)

    def test_invalid_corner_shape(self) -> None:
        with pytest.raises(ValueError, match=r"\(4, 2\)"):
            orient_geographically(_make_data(), np.zeros((3, 2)))

    def test_invalid_latitude(self) -> None:
        corners = np.array([[95.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
        with pytest.raises(ValueError, match="Latitudes"):
            orient_geographically(_make_data(), corners)

    def test_invalid_longitude(self) -> None:
        corners = np.array([[1.0, 0.0], [1.0, 190.0], [0.0, 1.0], [0.0, 0.0]])
        with pytest.raises(ValueError, match="Longitudes"):
            orient_geographically(_make_data(), corners)

    def test_rejects_generic_data_array(self) -> None:
        generic = xr.DataArray(np.zeros((2, 2)), dims=("x", "y"))
        corners = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
        with pytest.raises(ValueError, match="dimensions"):
            orient_geographically(generic, corners)


def _make_footprint_data(trajectory: CubicSplineTrajectory) -> SARDataModel:
    azimuth_times = [
        trajectory.times[0],
        trajectory.times[trajectory.times.size // 2],
        trajectory.times[-1],
    ]
    return create_sar_data_model(np.zeros((3, 2)), azimuth_times, np.array([0.0043, 0.0045]))


class TestComputeSarDataModelFootprint:
    """Test compute_sar_data_model_footprint ground corner computation."""

    @pytest.fixture(autouse=True)
    def setup_footprint_data(self, testing_trajectory: CubicSplineTrajectory) -> None:
        """Load a SAR data model defined over the testing trajectory domain."""
        self._trajectory = testing_trajectory
        self._data = _make_footprint_data(testing_trajectory)

    def test_ecef_footprint_shape(self) -> None:
        """Default output is a (4, 3) floating ECEF array, for both look sides."""
        for side_looking in ("LEFT", "RIGHT"):
            footprint = compute_sar_data_model_footprint(self._data, self._trajectory, side_looking)
            assert footprint.shape == (4, 3)
            assert np.issubdtype(footprint.dtype, np.floating)

    def test_corners_match_direct_geocoding_reference(self) -> None:
        """Corners equal the direct geocoding of the extreme azimuth and range coordinates."""
        first_azimuth_time = self._data[AZIMUTH_DIM].values[0]
        last_azimuth_time = self._data[AZIMUTH_DIM].values[-1]
        first_range_time = self._data[RANGE_DIM].values[0]
        last_range_time = self._data[RANGE_DIM].values[-1]
        expected_upper_corners = direct_geocoding_monostatic(
            sensor_positions=self._trajectory.position(first_azimuth_time),
            sensor_velocities=self._trajectory.velocity(first_azimuth_time),
            range_times=np.array([first_range_time, last_range_time]),
            look_direction="RIGHT",
            doppler_frequencies=0,
            wavelength=1,
            altitude=0,
        )
        expected_lower_corners = direct_geocoding_monostatic(
            sensor_positions=self._trajectory.position(last_azimuth_time),
            sensor_velocities=self._trajectory.velocity(last_azimuth_time),
            range_times=np.array([last_range_time, first_range_time]),
            look_direction="RIGHT",
            doppler_frequencies=0,
            wavelength=1,
            altitude=0,
        )
        expected_footprint = np.r_[expected_upper_corners, expected_lower_corners]
        footprint = compute_sar_data_model_footprint(self._data, self._trajectory, "RIGHT")
        np.testing.assert_allclose(footprint, expected_footprint, rtol=1e-10)

    def test_to_llh_returns_degrees(self) -> None:
        """With to_llh, output corners are (lat, lon) [deg] over the WGS84 ellipsoid."""
        footprint = compute_sar_data_model_footprint(self._data, self._trajectory, "RIGHT", to_llh=True)
        latitudes, longitudes, heights = footprint[:, 0], footprint[:, 1], footprint[:, 2]
        assert np.all((latitudes >= -90.0) & (latitudes <= 90.0))
        assert np.all((longitudes >= -180.0) & (longitudes <= 180.0))
        np.testing.assert_allclose(heights, 0.0, atol=1e-6)
        ecef_footprint = compute_sar_data_model_footprint(self._data, self._trajectory, "RIGHT")
        np.testing.assert_allclose(footprint, xyz2llh(ecef_footprint, radians=False), rtol=1e-10)

    def test_left_right_footprints_differ(self) -> None:
        """LEFT and RIGHT look directions produce different footprints."""
        right_footprint = compute_sar_data_model_footprint(self._data, self._trajectory, "RIGHT")
        left_footprint = compute_sar_data_model_footprint(self._data, self._trajectory, "LEFT")
        assert right_footprint.shape == left_footprint.shape == (4, 3)
        assert not np.allclose(right_footprint, left_footprint)

    def test_reversed_azimuth_swaps_corner_pairs(self) -> None:
        """Reversing the azimuth axis swaps the upper/lower corner pairs."""
        reversed_data = self._data.isel({AZIMUTH_DIM: slice(None, None, -1)})
        footprint = compute_sar_data_model_footprint(self._data, self._trajectory, "RIGHT")
        reversed_footprint = compute_sar_data_model_footprint(reversed_data, self._trajectory, "RIGHT")
        np.testing.assert_allclose(footprint[:2], reversed_footprint[2:][::-1], rtol=1e-10)
        np.testing.assert_allclose(footprint[2:], reversed_footprint[:2][::-1], rtol=1e-10)

    def test_geocoding_call_arguments(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Upper corners use the first azimuth time, lower corners the last one."""
        calls: list[dict] = []

        def fake_geocoding(**kwargs: object) -> np.ndarray:
            calls.append(kwargs)
            return np.zeros((2, 3))

        monkeypatch.setattr(orientation, "direct_geocoding_monostatic", fake_geocoding)
        compute_sar_data_model_footprint(self._data, self._trajectory, "LEFT")

        assert len(calls) == 2
        first_azimuth_time = self._data[AZIMUTH_DIM].values[0]
        last_azimuth_time = self._data[AZIMUTH_DIM].values[-1]
        first_range_time = self._data[RANGE_DIM].values[0]
        last_range_time = self._data[RANGE_DIM].values[-1]
        upper_kwargs, lower_kwargs = calls
        np.testing.assert_allclose(upper_kwargs["sensor_positions"], self._trajectory.position(first_azimuth_time))
        np.testing.assert_allclose(upper_kwargs["sensor_velocities"], self._trajectory.velocity(first_azimuth_time))
        np.testing.assert_allclose(upper_kwargs["range_times"], [first_range_time, last_range_time])
        np.testing.assert_allclose(lower_kwargs["sensor_positions"], self._trajectory.position(last_azimuth_time))
        np.testing.assert_allclose(lower_kwargs["sensor_velocities"], self._trajectory.velocity(last_azimuth_time))
        np.testing.assert_allclose(lower_kwargs["range_times"], [last_range_time, first_range_time])
        for kwargs in calls:
            assert kwargs["look_direction"] == "LEFT"
            assert kwargs["doppler_frequencies"] == 0
            assert kwargs["wavelength"] == 1
            assert kwargs["altitude"] == 0

    def test_to_llh_invokes_xyz2llh_with_degrees(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With to_llh, both corner sets are converted to degrees through xyz2llh."""
        conversions: list[tuple[np.ndarray, bool]] = []

        def fake_geocoding(**_kwargs: object) -> np.ndarray:
            return np.zeros((2, 3))

        def fake_xyz2llh(coordinates: np.ndarray, *, radians: bool = True) -> np.ndarray:
            conversions.append((coordinates, radians))
            return np.zeros((2, 3))

        monkeypatch.setattr(orientation, "direct_geocoding_monostatic", fake_geocoding)
        monkeypatch.setattr(orientation, "xyz2llh", fake_xyz2llh)
        compute_sar_data_model_footprint(self._data, self._trajectory, "RIGHT", to_llh=True)
        assert [radians for _, radians in conversions] == [False, False]

    def test_ecef_output_skips_llh_conversion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """By default no xyz2llh conversion is performed."""
        conversions: list[tuple[np.ndarray, bool]] = []

        def fake_geocoding(**_kwargs: object) -> np.ndarray:
            return np.zeros((2, 3))

        def fake_xyz2llh(coordinates: np.ndarray, *, radians: bool = True) -> np.ndarray:
            conversions.append((coordinates, radians))
            return np.zeros((2, 3))

        monkeypatch.setattr(orientation, "direct_geocoding_monostatic", fake_geocoding)
        monkeypatch.setattr(orientation, "xyz2llh", fake_xyz2llh)
        footprint = compute_sar_data_model_footprint(self._data, self._trajectory, "RIGHT")
        assert conversions == []
        np.testing.assert_allclose(footprint, np.zeros((4, 3)))


class TestComputeSquareGroundCellSubsamplingFactors:
    """Test compute_square_ground_cell_subsampling_factors formula."""

    @pytest.fixture(autouse=True)
    def setup_subsampling_data(self, testing_trajectory: CubicSplineTrajectory) -> None:
        """Load the testing trajectory and mid-swath acquisition parameters."""
        self._trajectory = testing_trajectory
        self._azimuth_time = testing_trajectory.times[testing_trajectory.times.size // 2]
        self._range_time = 0.0044
        self._lines_step = 1.5
        self._samples_step = 2.0
        self._subsampling_factor = 2.0

    def _patch_dependencies(
        self,
        monkeypatch: pytest.MonkeyPatch,
        look_angle: float = 0.5,
        incidence_angle: float = 0.6,
        ground_velocity: float = 7.0,
    ) -> None:
        """Replace the angle and velocity dependencies with fixed values."""
        monkeypatch.setattr(orientation, "compute_look_angles", lambda **_kwargs: look_angle)
        monkeypatch.setattr(orientation, "compute_incidence_angles", lambda **_kwargs: incidence_angle)
        monkeypatch.setattr(orientation, "compute_ground_velocity", lambda **_kwargs: ground_velocity)

    def _compute_factors(self, projection: str) -> tuple[float, float]:
        return compute_square_ground_cell_subsampling_factors(
            self._subsampling_factor,
            self._trajectory,
            self._azimuth_time,
            self._range_time,
            self._lines_step,
            self._samples_step,
            "RIGHT",
            projection,
        )

    def test_ground_range_projection_formula(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GROUND RANGE projection uses the samples step directly as range sampling step."""
        self._patch_dependencies(monkeypatch)
        az_factor, rng_factor = self._compute_factors("GROUND RANGE")
        expected_rng_factor = 7.0 * self._lines_step / self._samples_step * self._subsampling_factor
        assert az_factor == self._subsampling_factor
        assert rng_factor == pytest.approx(expected_rng_factor)

    def test_slant_range_projection_formula(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SLANT RANGE projection converts the samples step to meters via the incidence angle."""
        self._patch_dependencies(monkeypatch)
        az_factor, rng_factor = self._compute_factors("SLANT RANGE")
        rng_sampling_step_m = self._samples_step / np.sin(0.6) * speed_of_light / 2
        expected_rng_factor = 7.0 * self._lines_step / rng_sampling_step_m * self._subsampling_factor
        assert az_factor == self._subsampling_factor
        assert rng_factor == pytest.approx(expected_rng_factor)

    def test_first_factor_equals_requested_subsampling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The azimuth factor is the requested subsampling factor, unchanged."""
        self._patch_dependencies(monkeypatch)
        az_factor, _ = self._compute_factors("GROUND RANGE")
        assert az_factor == self._subsampling_factor

    def test_dependencies_receive_forwarded_arguments(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Look/incidence angles and ground velocity receive the mid-swath parameters."""
        look_calls: list[dict] = []
        incidence_calls: list[dict] = []
        velocity_calls: list[dict] = []

        def fake_look_angles(**kwargs: object) -> float:
            look_calls.append(kwargs)
            return 0.5

        def fake_incidence_angles(**kwargs: object) -> float:
            incidence_calls.append(kwargs)
            return 0.6

        def fake_ground_velocity(**kwargs: object) -> float:
            velocity_calls.append(kwargs)
            return 7.0

        monkeypatch.setattr(orientation, "compute_look_angles", fake_look_angles)
        monkeypatch.setattr(orientation, "compute_incidence_angles", fake_incidence_angles)
        monkeypatch.setattr(orientation, "compute_ground_velocity", fake_ground_velocity)
        self._compute_factors("GROUND RANGE")

        assert len(look_calls) == len(incidence_calls) == len(velocity_calls) == 1
        for kwargs in (*look_calls, *incidence_calls):
            assert kwargs["trajectory"] is self._trajectory
            assert kwargs["azimuth_time"] is self._azimuth_time
            assert kwargs["range_times"] == self._range_time
            assert kwargs["look_direction"] == "RIGHT"
        velocity_kwargs = velocity_calls[0]
        assert velocity_kwargs["trajectory"] is self._trajectory
        assert velocity_kwargs["azimuth_time"] is self._azimuth_time
        np.testing.assert_allclose(velocity_kwargs["look_angles_rad"], 0.5)

    def test_scaling_with_steps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The range factor scales linearly with the azimuth step and inversely with the range step."""
        self._patch_dependencies(monkeypatch)
        base_factor = self._compute_factors("GROUND RANGE")[1]
        doubled_lines_step = compute_square_ground_cell_subsampling_factors(
            self._subsampling_factor,
            self._trajectory,
            self._azimuth_time,
            self._range_time,
            2 * self._lines_step,
            self._samples_step,
            "RIGHT",
            "GROUND RANGE",
        )[1]
        doubled_samples_step = compute_square_ground_cell_subsampling_factors(
            self._subsampling_factor,
            self._trajectory,
            self._azimuth_time,
            self._range_time,
            self._lines_step,
            2 * self._samples_step,
            "RIGHT",
            "GROUND RANGE",
        )[1]
        assert doubled_lines_step == pytest.approx(2 * base_factor)
        assert doubled_samples_step == pytest.approx(base_factor / 2)

    def test_returns_tuple_of_two_floats(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Output is a (azimuth, range) tuple of floats."""
        self._patch_dependencies(monkeypatch)
        factors = self._compute_factors("GROUND RANGE")
        assert isinstance(factors, tuple)
        assert len(factors) == 2
        assert all(isinstance(factor, float) for factor in factors)

    def test_matches_manual_composition_with_real_trajectory(self) -> None:
        """Factors match the manual composition of look/incidence angles and ground velocity."""
        look_angle = compute_look_angles(self._trajectory, self._azimuth_time, self._range_time, "RIGHT")
        incidence_angle = compute_incidence_angles(self._trajectory, self._azimuth_time, self._range_time, "RIGHT")
        ground_velocity = compute_ground_velocity(self._trajectory, self._azimuth_time, look_angle)
        az_sampling_step_m = ground_velocity * self._lines_step
        for projection, rng_sampling_step_m in (
            ("GROUND RANGE", self._samples_step),
            ("SLANT RANGE", self._samples_step / np.sin(incidence_angle) * speed_of_light / 2),
        ):
            az_factor, rng_factor = self._compute_factors(projection)
            expected_rng_factor = float(az_sampling_step_m / rng_sampling_step_m) * self._subsampling_factor
            assert az_factor == self._subsampling_factor
            assert rng_factor == pytest.approx(expected_rng_factor, rel=1e-12)
