# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for geographical orientation of quick-look data."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from perseo_core.imaging.geospatial.orientation import orient_geographically
from perseo_core.imaging.models import SARDataModel, create_sar_data_model
from perseo_core.timing.precise_datetime import PreciseDateTime


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
