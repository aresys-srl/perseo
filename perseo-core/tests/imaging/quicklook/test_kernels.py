# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for quicklook kernels."""

from __future__ import annotations

import numpy as np

from perseo_core.imaging.models import SARDataModel, create_sar_data_model
from perseo_core.imaging.quicklook.kernels import BoxKernel, DecimationKernel, GaussianKernel
from perseo_core.timing.precise_datetime import PreciseDateTime

DATA_4X4 = np.arange(1, 17, dtype=float).reshape(4, 4)


def _make_data(
    raster: np.ndarray,
    azimuth_step: float = 1.0,
    range_step: float = 1.0,
) -> SARDataModel:
    azimuth_times = [
        PreciseDateTime.from_numeric_datetime(2024, 1, 1) + index * azimuth_step for index in range(raster.shape[0])
    ]
    range_coords = np.arange(raster.shape[1], dtype=float) * range_step
    return create_sar_data_model(raster, azimuth_times, range_coords)


class TestBoxKernel:
    def test_known_values(self) -> None:
        result = BoxKernel().apply(_make_data(DATA_4X4), 2.0, 2.0)
        # window covers 3 pixels with nominal normalization, edges are attenuated
        np.testing.assert_allclose(result.values, [[14 / 9, 10 / 3], [19 / 3, 11]])

    def test_integer_positions_keep_exact_coordinates(self) -> None:
        data = _make_data(DATA_4X4, azimuth_step=10.0, range_step=5.0)
        result = BoxKernel().apply(data, 2.0, 2.0)
        assert result.azimuth.values[0] == data.azimuth.values[0]
        assert result.azimuth.values[1] == data.azimuth.values[2]
        np.testing.assert_allclose(result.range.values, [0.0, 10.0])

    def test_fractional_positions_interpolate_coordinates(self) -> None:
        data = _make_data(np.ones((5, 5)), azimuth_step=10.0, range_step=5.0)
        result = BoxKernel().apply(data, 2.5, 2.5)
        # position 2.5 interpolates between pixels 2 and 3
        expected_azimuth = data.azimuth.values[2] + 0.5 * (data.azimuth.values[3] - data.azimuth.values[2])
        assert result.azimuth.values[1] == expected_azimuth
        np.testing.assert_allclose(result.range.values, [0.0, 12.5])

    def test_constant_raster_is_preserved_in_interior(self) -> None:
        result = BoxKernel().apply(_make_data(np.ones((8, 8))), 2.0, 2.0)
        np.testing.assert_allclose(result.values[1:-1, 1:-1], 1.0)

    def test_float_subsampling_factor_shape(self) -> None:
        result = BoxKernel().apply(_make_data(np.ones((5, 5))), 2.5, 2.5)
        assert result.shape == (2, 2)

    def test_attrs_are_preserved(self) -> None:
        data = _make_data(DATA_4X4)
        data.attrs["units"] = "sigma0"
        result = BoxKernel().apply(data, 2.0, 2.0)
        assert result.attrs["units"] == "sigma0"


class TestDecimationKernel:
    def test_selects_every_nth_sample(self) -> None:
        result = DecimationKernel().apply(_make_data(DATA_4X4), 2.0, 2.0)
        np.testing.assert_allclose(result.values, [[1.0, 3.0], [9.0, 11.0]])

    def test_rounds_fractional_positions(self) -> None:
        data = _make_data(np.arange(100, dtype=float).reshape(10, 10))
        result = DecimationKernel().apply(data, 2.4, 2.4)
        positions = np.array([0, 2, 5, 7, 9])
        np.testing.assert_allclose(result.values, data.values[positions][:, positions])

    def test_coordinates_are_interpolated(self) -> None:
        data = _make_data(np.arange(100, dtype=float).reshape(10, 10), azimuth_step=10.0)
        result = DecimationKernel().apply(data, 2.4, 2.4)
        # position 4.8 interpolates between pixels 4 and 5
        position = np.arange(10, step=2.4)[2]
        index = int(np.floor(position))
        fraction = position - index
        expected = data.azimuth.values[index] + fraction * (data.azimuth.values[index + 1] - data.azimuth.values[index])
        assert result.azimuth.values[2] == expected


class TestGaussianKernel:
    def test_constant_raster_is_preserved(self) -> None:
        result = GaussianKernel().apply(_make_data(np.ones((8, 8))), 2.0, 2.0)
        np.testing.assert_allclose(result.values, 1.0)

    def test_float_subsampling_factor_shape(self) -> None:
        result = GaussianKernel().apply(_make_data(np.ones((10, 10))), 2.5, 2.5)
        assert result.shape == (4, 4)

    def test_concentrates_weight_at_center(self) -> None:
        raster = np.zeros((5, 5))
        raster[2, 2] = 1.0
        gaussian = GaussianKernel().apply(_make_data(raster), 2.0, 2.0).values
        box = BoxKernel().apply(_make_data(raster), 2.0, 2.0).values
        # output position 1 corresponds to full resolution position 2 (the delta)
        assert gaussian[1, 1] > box[1, 1]
