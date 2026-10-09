# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for quicklook output transformations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from perseo_core.imaging.models import create_sar_data_model
from perseo_core.imaging.quicklook.transforms import (
    DEFAULT_TRANSFORM,
    TRANSFORMS,
    ClipPercentileTransform,
    PhaseTransform,
    ToDbTransform,
    abs_percentiles_threshold_filtering,
    clip_percentile,
    get_transform,
    to_db,
    to_phase,
)
from perseo_core.timing.precise_datetime import PreciseDateTime

if TYPE_CHECKING:
    from perseo_core.imaging.models import SARDataModel


def _make_data(raster: np.ndarray) -> SARDataModel:
    azimuth_times = [
        PreciseDateTime.from_numeric_datetime(2024, 1, 1) + index * 1.0 for index in range(raster.shape[0])
    ]
    range_coords = np.arange(raster.shape[1], dtype=float)
    return create_sar_data_model(raster, azimuth_times, range_coords)


class TestToDb:
    def test_known_values(self) -> None:
        np.testing.assert_allclose(to_db([1.0, 10.0, 100.0]), [0.0, 20.0, 40.0])

    def test_zero_gives_minus_inf(self) -> None:
        assert to_db([0.0])[0] == -np.inf

    def test_uses_magnitude(self) -> None:
        np.testing.assert_allclose(to_db([-10.0]), to_db([10.0]))


class TestToPhase:
    def test_known_values(self) -> None:
        np.testing.assert_allclose(to_phase(np.array([1 + 0j, 1j])), [0.0, np.pi / 2])


class TestClipPercentile:
    def test_clips_between_percentiles(self) -> None:
        data = np.arange(101, dtype=float)
        result = clip_percentile(data, min_percentile=10.0, max_percentile=90.0)
        np.testing.assert_allclose(result, np.clip(data, 10.0, 90.0))

    def test_invalid_percentiles_raise(self) -> None:
        with pytest.raises(ValueError, match="Percentiles must satisfy"):
            clip_percentile(np.arange(10.0), min_percentile=50.0, max_percentile=10.0)


class TestAbsPercentilesThresholdFiltering:
    def test_defaults(self) -> None:
        data = np.arange(101, dtype=float)
        result = abs_percentiles_threshold_filtering(data)
        expected = np.clip(data, *np.nanpercentile(data, [1.0, 99.0]))
        np.testing.assert_allclose(result, expected)


class TestToDbTransform:
    def test_pre_filter_extracts_magnitude(self) -> None:
        data = _make_data(np.array([[3 + 4j, 0j], [0j, 0j]]))
        prepared = ToDbTransform().pre_filter(data)
        np.testing.assert_allclose(prepared.values, np.abs(data.values))

    def test_post_filter_converts_to_db(self) -> None:
        data = _make_data(np.array([[10.0, 100.0]]))
        result = ToDbTransform().post_filter(data)
        np.testing.assert_allclose(result.values, to_db(np.array([[10.0, 100.0]])))

    def test_preserves_coordinates(self) -> None:
        data = _make_data(np.arange(4, dtype=float).reshape(2, 2))
        result = ToDbTransform().pre_filter(data)
        assert result.azimuth.values[0] == data.azimuth.values[0]
        np.testing.assert_allclose(result.range.values, data.range.values)


class TestPhaseTransform:
    def test_pre_filter_keeps_data(self) -> None:
        data = _make_data(np.array([[1 + 1j, 1 - 1j]]))
        assert PhaseTransform().pre_filter(data) is data

    def test_post_filter_extracts_phase(self) -> None:
        data = _make_data(np.array([[1 + 0j, 1j]]))
        result = PhaseTransform().post_filter(data)
        np.testing.assert_allclose(result.values, to_phase(np.array([[1 + 0j, 1j]])))


class TestClipPercentileTransform:
    def test_rejects_invalid_percentiles(self) -> None:
        with pytest.raises(ValueError, match="Percentiles must satisfy"):
            ClipPercentileTransform(min_percentile=50.0, max_percentile=10.0)

    def test_pre_filter_extracts_magnitude(self) -> None:
        data = _make_data(np.array([[3 + 4j, 0j]]))
        prepared = ClipPercentileTransform().pre_filter(data)
        np.testing.assert_allclose(prepared.values, np.abs(data.values))

    def test_post_filter_clips(self) -> None:
        data = _make_data(np.arange(101, dtype=float).reshape(1, 101))
        transform = ClipPercentileTransform(min_percentile=10.0, max_percentile=90.0)
        result = transform.post_filter(data)
        expected = np.clip(np.arange(101, dtype=float), 10.0, 90.0).reshape(1, 101)
        np.testing.assert_allclose(result.values, expected)


class TestGetTransform:
    def test_db(self) -> None:
        assert isinstance(get_transform("db"), ToDbTransform)

    def test_phase(self) -> None:
        assert isinstance(get_transform("phase"), PhaseTransform)

    def test_clip_percentile_with_params(self) -> None:
        transform = get_transform("clip_percentile", min_percentile=10.0, max_percentile=90.0)
        assert isinstance(transform, ClipPercentileTransform)
        data = _make_data(np.arange(101, dtype=float).reshape(1, 101))
        expected = np.clip(np.arange(101, dtype=float), 10.0, 90.0).reshape(1, 101)
        np.testing.assert_allclose(transform.post_filter(data).values, expected)

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown transform"):
            get_transform("nonexistent")

    def test_registry_contents(self) -> None:
        assert set(TRANSFORMS) == {"db", "phase", "clip_percentile"}


class TestDefaultTransform:
    def test_is_clip_percentile(self) -> None:
        assert isinstance(DEFAULT_TRANSFORM, ClipPercentileTransform)
