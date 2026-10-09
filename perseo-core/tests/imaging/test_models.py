# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for the SAR data model."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from perseo_core.imaging.models import (
    SARDataModel,
    _interpolate_axis_coords,
    create_sar_data_model,
    validate_sar_data_model,
)
from perseo_core.timing.precise_datetime import PreciseDateTime


def _azimuth_times(count: int) -> list[PreciseDateTime]:
    return [PreciseDateTime.from_numeric_datetime(2024, 1, 1) + index * 1.0 for index in range(count)]


class TestCreateSarDataModel:
    def test_creates_sar_data_model(self) -> None:
        raster = np.arange(6, dtype=float).reshape(2, 3)
        azimuth_times = _azimuth_times(2)
        range_coords = np.array([0.0, 2.5, 5.0])
        data = create_sar_data_model(raster, azimuth_times, range_coords)
        assert isinstance(data, SARDataModel)
        assert data.dims == ("azimuth", "range")
        assert data.shape == (2, 3)
        assert data.azimuth.dtype == object
        assert data.azimuth.values[1] == azimuth_times[1]
        np.testing.assert_allclose(data.range.values, range_coords)

    def test_rejects_non_2d_raster(self) -> None:
        with pytest.raises(ValueError, match="two-dimensional"):
            create_sar_data_model(np.zeros(3), _azimuth_times(3), np.zeros(3))

    def test_rejects_mismatched_azimuth(self) -> None:
        with pytest.raises(ValueError, match="Azimuth times count"):
            create_sar_data_model(np.zeros((2, 3)), _azimuth_times(3), np.zeros(3))

    def test_rejects_mismatched_range(self) -> None:
        with pytest.raises(ValueError, match="Range coordinates count"):
            create_sar_data_model(np.zeros((2, 3)), _azimuth_times(2), np.zeros(2))


class TestValidateSarDataModel:
    def test_accepts_valid_model(self) -> None:
        data = create_sar_data_model(np.zeros((2, 2)), _azimuth_times(2), np.zeros(2))
        validate_sar_data_model(data)

    def test_rejects_wrong_dims(self) -> None:
        data = xr.DataArray(np.zeros((2, 2)), dims=("x", "y"))
        with pytest.raises(ValueError, match="dimensions"):
            validate_sar_data_model(data)

    def test_rejects_missing_azimuth_coord(self) -> None:
        data = xr.DataArray(np.zeros((2, 2)), dims=("azimuth", "range"), coords={"range": np.zeros(2)})
        with pytest.raises(ValueError, match="Missing azimuth"):
            validate_sar_data_model(data)

    def test_rejects_wrong_azimuth_dtype(self) -> None:
        data = xr.DataArray(
            np.zeros((2, 2)),
            dims=("azimuth", "range"),
            coords={"azimuth": np.zeros(2), "range": np.zeros(2)},
        )
        with pytest.raises(ValueError, match="object dtype"):
            validate_sar_data_model(data)

    def test_rejects_wrong_range_dtype(self) -> None:
        data = xr.DataArray(
            np.zeros((2, 2)),
            dims=("azimuth", "range"),
            coords={
                "azimuth": np.array(["a", "b"], dtype=object),
                "range": np.array([0, 1]),
            },
        )
        with pytest.raises(ValueError, match="floating dtype"):
            validate_sar_data_model(data)


class TestInterpolateAxisCoords:
    def test_float_interpolation(self) -> None:
        coords = np.array([0.0, 10.0])
        positions = np.array([0.0, 0.25, 0.5, 1.0])
        np.testing.assert_allclose(_interpolate_axis_coords(coords, positions), [0.0, 2.5, 5.0, 10.0])

    def test_precise_datetime_interpolation(self) -> None:
        start = PreciseDateTime.from_numeric_datetime(2024, 1, 1)
        end = PreciseDateTime.from_numeric_datetime(2024, 1, 2)
        coords = np.array([start, end], dtype=object)
        result = _interpolate_axis_coords(coords, np.array([0.5]))
        assert result[0] == start + 0.5 * (end - start)

    def test_single_coordinate(self) -> None:
        coords = np.array([5.0])
        np.testing.assert_allclose(_interpolate_axis_coords(coords, np.array([0.0])), [5.0])

    def test_non_monotonic_burst_overlap(self) -> None:
        coords = np.array([0.0, 2.0, 1.0], dtype=float)
        positions = np.array([0.5, 1.5])
        np.testing.assert_allclose(_interpolate_axis_coords(coords, positions), [1.0, 1.5])
