# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for io/layout.py core functionalities"""

from __future__ import annotations

import numpy as np
import pytest
from perseo_core.timing import PreciseDateTime

from perseo_quality.core.custom_errors import (
    AzimuthExceedsBoundariesError,
    InvalidBurstIdError,
    RangeExceedsBoundariesError,
)
from perseo_quality.io.layout import L1BurstLayout, L1RasterLayout


class TestBurstLayout:
    """Testing layout.py BurstLayout dataclass"""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.azimuth_start_time = PreciseDateTime.from_utc_string("03-OCT-2019 14:14:24.625760000000")
        self.range_start_time_s = 0.005276013758430800300
        self.lines = 5181
        self.samples = 23706
        self.lines_step = 0.00213030157265797
        self.samples_step = 1.55411747884846e-8
        self.expected_mid_range_time = 0.005460215532611314
        self.expected_mid_azimuth_time = PreciseDateTime.from_utc_string("03-OCT-2019 14:14:30.143241073184")

    def test_init(self) -> None:
        """Testing class init"""
        burst_layout = L1BurstLayout(
            burst_id=1,
            lines=self.lines,
            samples=self.samples,
            lines_step=self.lines_step,
            samples_step=self.samples_step,
            lines_start=self.azimuth_start_time,
            samples_start=self.range_start_time_s,
        )
        assert isinstance(burst_layout, L1BurstLayout)

    def test_input_variables_accessibility(self) -> None:
        """Testing retrieval of input variables"""
        burst_layout = L1BurstLayout(
            burst_id=1,
            lines=self.lines,
            samples=self.samples,
            lines_step=self.lines_step,
            samples_step=self.samples_step,
            lines_start=self.azimuth_start_time,
            samples_start=self.range_start_time_s,
        )
        assert burst_layout.burst_id == 1
        assert burst_layout.lines == self.lines
        assert burst_layout.samples == self.samples
        assert burst_layout._lines_step == self.lines_step
        assert burst_layout._samples_step == self.samples_step
        assert burst_layout.azimuth_axis[0] == self.azimuth_start_time
        assert burst_layout.range_axis[0] == self.range_start_time_s

    def test_computed_azimuth_axis(self) -> None:
        """Testing azimuth axis computation"""
        burst_layout = L1BurstLayout(
            burst_id=1,
            lines=self.lines,
            samples=self.samples,
            lines_step=self.lines_step,
            samples_step=self.samples_step,
            lines_start=self.azimuth_start_time,
            samples_start=self.range_start_time_s,
        )
        np.testing.assert_allclose(burst_layout.azimuth_axis[0] - self.azimuth_start_time, 0, atol=1e-9, rtol=0)
        np.testing.assert_allclose(
            np.diff(burst_layout.azimuth_axis).mean() - self.lines_step,
            0,
            atol=1e-9,
            rtol=0,
        )
        np.testing.assert_allclose(
            burst_layout.azimuth_axis[-1] - (self.lines_step * (self.lines - 1) + self.azimuth_start_time),
            0,
            atol=1e-9,
            rtol=0,
        )

    def test_computed_range_axis(self) -> None:
        """Testing range axis computation"""
        burst_layout = L1BurstLayout(
            burst_id=1,
            lines=self.lines,
            samples=self.samples,
            lines_step=self.lines_step,
            samples_step=self.samples_step,
            lines_start=self.azimuth_start_time,
            samples_start=self.range_start_time_s,
        )
        np.testing.assert_allclose(
            burst_layout.range_axis[0],
            self.range_start_time_s,
            atol=1e-9,
            rtol=0,
        )
        np.testing.assert_allclose(
            np.diff(burst_layout.range_axis).mean(),
            self.samples_step,
            atol=1e-12,
            rtol=0,
        )
        np.testing.assert_allclose(
            burst_layout.range_axis[-1] - (self.samples_step * (self.samples - 1) + self.range_start_time_s),
            0,
            atol=1e-9,
            rtol=0,
        )

    def test_computed_mid_burst_azimuth_time(self) -> None:
        """Testing mid burst azimuth time computation"""
        burst_layout = L1BurstLayout(
            burst_id=1,
            lines=self.lines,
            samples=self.samples,
            lines_step=self.lines_step,
            samples_step=self.samples_step,
            lines_start=self.azimuth_start_time,
            samples_start=self.range_start_time_s,
        )
        np.testing.assert_allclose(
            burst_layout.mid_burst_azimuth - self.expected_mid_azimuth_time,
            0,
            atol=1e-9,
            rtol=0,
        )

    def test_computed_mid_burst_range_time(self) -> None:
        """Testing mid burst range time computation"""
        burst_layout = L1BurstLayout(
            burst_id=1,
            lines=self.lines,
            samples=self.samples,
            lines_step=self.lines_step,
            samples_step=self.samples_step,
            lines_start=self.azimuth_start_time,
            samples_start=self.range_start_time_s,
        )
        np.testing.assert_allclose(
            burst_layout.mid_burst_range,
            self.expected_mid_range_time,
            atol=1e-9,
            rtol=0,
        )

    def test_is_azimuth_in_burst_true(self) -> None:
        """Testing is_azimuth_in_burst method, with positive outcome"""
        burst_layout = L1BurstLayout(
            burst_id=1,
            lines=self.lines,
            samples=self.samples,
            lines_step=self.lines_step,
            samples_step=self.samples_step,
            lines_start=self.azimuth_start_time,
            samples_start=self.range_start_time_s,
        )
        assert burst_layout.is_azimuth_in_burst(self.azimuth_start_time + self.lines // 5 * self.lines_step)

    def test_is_azimuth_in_burst_false(self) -> None:
        """Testing is_azimuth_in_burst method, with negative outcome"""
        burst_layout = L1BurstLayout(
            burst_id=1,
            lines=self.lines,
            samples=self.samples,
            lines_step=self.lines_step,
            samples_step=self.samples_step,
            lines_start=self.azimuth_start_time,
            samples_start=self.range_start_time_s,
        )
        for t in [-self.lines_step, self.lines * self.lines_step, (self.lines + 1) * self.lines_step]:
            assert not burst_layout.is_azimuth_in_burst(self.azimuth_start_time + t)

    def test_is_azimuth_relative_pixel_in_burst_true(self) -> None:
        """Testing is_azimuth_relative_pixel_in_burst method, with positive outcome"""
        burst_layout = L1BurstLayout(
            burst_id=1,
            lines=self.lines,
            samples=self.samples,
            lines_step=self.lines_step,
            samples_step=self.samples_step,
            lines_start=self.azimuth_start_time,
            samples_start=self.range_start_time_s,
        )
        assert burst_layout.is_azimuth_relative_pixel_in_burst(850)

    def test_is_azimuth_relative_pixel_in_burst_false(self) -> None:
        """Testing is_azimuth_relative_pixel_in_burst method, with negative outcome"""
        burst_layout = L1BurstLayout(
            burst_id=1,
            lines=self.lines,
            samples=self.samples,
            lines_step=self.lines_step,
            samples_step=self.samples_step,
            lines_start=self.azimuth_start_time,
            samples_start=self.range_start_time_s,
        )
        for t in [-1, 8500, self.lines]:
            assert not burst_layout.is_azimuth_relative_pixel_in_burst(t)

    def test_is_range_in_burst_true(self) -> None:
        """Testing is_range_in_burst method, with positive outcome"""
        burst_layout = L1BurstLayout(
            burst_id=1,
            lines=self.lines,
            samples=self.samples,
            lines_step=self.lines_step,
            samples_step=self.samples_step,
            lines_start=self.azimuth_start_time,
            samples_start=self.range_start_time_s,
        )
        assert burst_layout.is_range_in_burst(self.range_start_time_s + self.samples // 5 * self.samples_step)

    def test_is_range_in_burst_false(self) -> None:
        """Testing is_range_in_burst method, with negative outcome"""
        burst_layout = L1BurstLayout(
            burst_id=1,
            lines=self.lines,
            samples=self.samples,
            lines_step=self.lines_step,
            samples_step=self.samples_step,
            lines_start=self.azimuth_start_time,
            samples_start=self.range_start_time_s,
        )
        assert not burst_layout.is_range_in_burst(self.range_start_time_s + (self.samples + 1) * self.samples_step)

    def test_is_range_relative_pixel_in_burst_true(self) -> None:
        """Testing is_range_relative_pixel_in_burst method, with positive outcome"""
        burst_layout = L1BurstLayout(
            burst_id=1,
            lines=self.lines,
            samples=self.samples,
            lines_step=self.lines_step,
            samples_step=self.samples_step,
            lines_start=self.azimuth_start_time,
            samples_start=self.range_start_time_s,
        )
        assert burst_layout.is_range_relative_pixel_in_burst(850)

    def test_is_range_relative_pixel_in_burst_false(self) -> None:
        """Testing is_range_relative_pixel_in_burst method, with negative outcome"""
        burst_layout = L1BurstLayout(
            burst_id=1,
            lines=self.lines,
            samples=self.samples,
            lines_step=self.lines_step,
            samples_step=self.samples_step,
            lines_start=self.azimuth_start_time,
            samples_start=self.range_start_time_s,
        )
        for t in [-1, 95000, self.samples]:
            assert not burst_layout.is_range_relative_pixel_in_burst(t)

    def test_azimuth_to_burst_pixel(self) -> None:
        """Testing azimuth_to_burst_pixel method"""
        burst_layout = L1BurstLayout(
            burst_id=1,
            lines=self.lines,
            samples=self.samples,
            lines_step=self.lines_step,
            samples_step=self.samples_step,
            lines_start=self.azimuth_start_time,
            samples_start=self.range_start_time_s,
        )
        indexes = [0, 300, self.lines - 1]
        times = [self.azimuth_start_time + idx * self.lines_step for idx in indexes]
        for idx, t in zip(indexes, times, strict=True):
            np.testing.assert_allclose(
                burst_layout.azimuth_to_burst_pixel(t),
                idx,
                atol=1e-9,
                rtol=0,
            )

    def test_azimuth_to_burst_pixel_error(self) -> None:
        """Testing azimuth_to_burst_pixel method, raising error"""
        burst_layout = L1BurstLayout(
            burst_id=1,
            lines=self.lines,
            samples=self.samples,
            lines_step=self.lines_step,
            samples_step=self.samples_step,
            lines_start=self.azimuth_start_time,
            samples_start=self.range_start_time_s,
        )
        indexes = [-1, self.lines + 5]
        times = [self.azimuth_start_time + idx * self.lines_step for idx in indexes]
        for t in times:
            with pytest.raises(AzimuthExceedsBoundariesError):
                burst_layout.azimuth_to_burst_pixel(t)

    def test_range_to_burst_pixel(self) -> None:
        """Testing range_to_burst_pixel method"""
        burst_layout = L1BurstLayout(
            burst_id=1,
            lines=self.lines,
            samples=self.samples,
            lines_step=self.lines_step,
            samples_step=self.samples_step,
            lines_start=self.azimuth_start_time,
            samples_start=self.range_start_time_s,
        )
        indexes = [0, 300, self.samples - 1]
        times = [self.range_start_time_s + idx * self.samples_step for idx in indexes]
        for idx, t in zip(indexes, times, strict=True):
            np.testing.assert_allclose(
                burst_layout.range_to_burst_pixel(t),
                idx,
                atol=1e-9,
                rtol=0,
            )

    def test_range_to_burst_pixel_error(self) -> None:
        """Testing range_to_burst_pixel method, raising error"""
        burst_layout = L1BurstLayout(
            burst_id=1,
            lines=self.lines,
            samples=self.samples,
            lines_step=self.lines_step,
            samples_step=self.samples_step,
            lines_start=self.azimuth_start_time,
            samples_start=self.range_start_time_s,
        )
        indexes = [-1, self.samples + 5]
        times = [self.range_start_time_s + idx * self.samples_step for idx in indexes]
        for t in times:
            with pytest.raises(RangeExceedsBoundariesError):
                burst_layout.range_to_burst_pixel(t)

    def test_burst_pixel_to_azimuth_1(self) -> None:
        """Testing burst_pixel_to_azimuth method, version 1"""
        burst_layout = L1BurstLayout(
            burst_id=1,
            lines=self.lines,
            samples=self.samples,
            lines_step=self.lines_step,
            samples_step=self.samples_step,
            lines_start=self.azimuth_start_time,
            samples_start=self.range_start_time_s,
        )
        az_time = burst_layout.burst_pixel_to_azimuth(800)
        np.testing.assert_allclose((az_time - self.azimuth_start_time) / self.lines_step, 800, atol=1e-9, rtol=0)

    def test_burst_pixel_to_azimuth_2(self) -> None:
        """Testing burst_pixel_to_azimuth method, version 2"""
        burst_layout = L1BurstLayout(
            burst_id=1,
            lines=self.lines,
            samples=self.samples,
            lines_step=self.lines_step,
            samples_step=self.samples_step,
            lines_start=self.azimuth_start_time,
            samples_start=self.range_start_time_s,
        )
        index = burst_layout.azimuth_to_burst_pixel(burst_layout.burst_pixel_to_azimuth(800.36))
        np.testing.assert_allclose(index, 800.36, atol=1e-9, rtol=0)

    def test_burst_pixel_to_azimuth_error(self) -> None:
        """Testing burst_pixel_to_azimuth method, raising error"""
        burst_layout = L1BurstLayout(
            burst_id=1,
            lines=self.lines,
            samples=self.samples,
            lines_step=self.lines_step,
            samples_step=self.samples_step,
            lines_start=self.azimuth_start_time,
            samples_start=self.range_start_time_s,
        )
        for t in [-5, 8000]:
            with pytest.raises(AzimuthExceedsBoundariesError):
                burst_layout.burst_pixel_to_azimuth(t)

    def test_burst_pixel_to_range_1(self) -> None:
        """Testing burst_pixel_to_range method, version 1"""
        burst_layout = L1BurstLayout(
            burst_id=1,
            lines=self.lines,
            samples=self.samples,
            lines_step=self.lines_step,
            samples_step=self.samples_step,
            lines_start=self.azimuth_start_time,
            samples_start=self.range_start_time_s,
        )
        rng = burst_layout.burst_pixel_to_range(800)
        np.testing.assert_allclose((rng - self.range_start_time_s) / self.samples_step, 800, atol=1e-9, rtol=0)

    def test_burst_pixel_to_range_2(self) -> None:
        """Testing burst_pixel_to_range method, version 2"""
        burst_layout = L1BurstLayout(
            burst_id=1,
            lines=self.lines,
            samples=self.samples,
            lines_step=self.lines_step,
            samples_step=self.samples_step,
            lines_start=self.azimuth_start_time,
            samples_start=self.range_start_time_s,
        )
        index = burst_layout.range_to_burst_pixel(burst_layout.burst_pixel_to_range(800.36))
        np.testing.assert_allclose(index, 800.36, atol=1e-9, rtol=0)

    def test_burst_pixel_to_range_error(self) -> None:
        """Testing burst_pixel_to_range method, raising error"""
        burst_layout = L1BurstLayout(
            burst_id=1,
            lines=self.lines,
            samples=self.samples,
            lines_step=self.lines_step,
            samples_step=self.samples_step,
            lines_start=self.azimuth_start_time,
            samples_start=self.range_start_time_s,
        )
        for t in [-5, 90000]:
            with pytest.raises(RangeExceedsBoundariesError):
                burst_layout.burst_pixel_to_range(t)

    def test_burst_pixel_to_coordinates(self) -> None:
        """Testing burst_pixels_to_coordinates method"""
        burst_layout = L1BurstLayout(
            burst_id=1,
            lines=self.lines,
            samples=self.samples,
            lines_step=self.lines_step,
            samples_step=self.samples_step,
            lines_start=self.azimuth_start_time,
            samples_start=self.range_start_time_s,
        )
        az, rng = burst_layout.burst_pixels_to_coordinates(az_burst_pixel=800.36, rng_burst_pixel=15000.12)
        np.testing.assert_allclose((az - self.azimuth_start_time) / self.lines_step, 800.36, atol=1e-9, rtol=0)
        np.testing.assert_allclose((rng - self.range_start_time_s) / self.samples_step, 15000.12, atol=1e-9, rtol=0)

    def test_burst_pixel_to_coordinates_error_1(self) -> None:
        """Testing burst_pixels_to_coordinates method, raising error 1"""
        burst_layout = L1BurstLayout(
            burst_id=1,
            lines=self.lines,
            samples=self.samples,
            lines_step=self.lines_step,
            samples_step=self.samples_step,
            lines_start=self.azimuth_start_time,
            samples_start=self.range_start_time_s,
        )
        with pytest.raises(AzimuthExceedsBoundariesError):
            burst_layout.burst_pixels_to_coordinates(az_burst_pixel=5800.36, rng_burst_pixel=15000.12)

    def test_burst_pixel_to_coordinates_error_2(self) -> None:
        """Testing burst_pixels_to_coordinates method, raising error 2"""
        burst_layout = L1BurstLayout(
            burst_id=1,
            lines=self.lines,
            samples=self.samples,
            lines_step=self.lines_step,
            samples_step=self.samples_step,
            lines_start=self.azimuth_start_time,
            samples_start=self.range_start_time_s,
        )
        with pytest.raises(RangeExceedsBoundariesError):
            burst_layout.burst_pixels_to_coordinates(az_burst_pixel=800, rng_burst_pixel=90000)

    def test_coordinates_to_burst_pixels(self) -> None:
        """Testing coordinates_to_burst_pixels method"""
        burst_layout = L1BurstLayout(
            burst_id=1,
            lines=self.lines,
            samples=self.samples,
            lines_step=self.lines_step,
            samples_step=self.samples_step,
            lines_start=self.azimuth_start_time,
            samples_start=self.range_start_time_s,
        )
        az_px, rng_px = burst_layout.coordinates_to_burst_pixels(
            az=self.azimuth_start_time + self.lines_step * 954.521,
            rng=self.range_start_time_s + self.samples_step * 13520.634,
        )
        np.testing.assert_allclose(az_px, 954.521, atol=1e-9, rtol=0)
        np.testing.assert_allclose(rng_px, 13520.634, atol=1e-9, rtol=0)

    def test_coordinates_to_burst_pixels_error_1(self) -> None:
        """Testing coordinates_to_burst_pixels method, raising error 1"""
        burst_layout = L1BurstLayout(
            burst_id=1,
            lines=self.lines,
            samples=self.samples,
            lines_step=self.lines_step,
            samples_step=self.samples_step,
            lines_start=self.azimuth_start_time,
            samples_start=self.range_start_time_s,
        )
        with pytest.raises(AzimuthExceedsBoundariesError):
            burst_layout.coordinates_to_burst_pixels(
                az=self.azimuth_start_time + self.lines_step * 10000,
                rng=self.range_start_time_s + self.samples_step * 13520.634,
            )

    def test_coordinates_to_burst_pixels_error_2(self) -> None:
        """Testing coordinates_to_burst_pixels method, raising error 2"""
        burst_layout = L1BurstLayout(
            burst_id=1,
            lines=self.lines,
            samples=self.samples,
            lines_step=self.lines_step,
            samples_step=self.samples_step,
            lines_start=self.azimuth_start_time,
            samples_start=self.range_start_time_s,
        )
        with pytest.raises(RangeExceedsBoundariesError):
            burst_layout.coordinates_to_burst_pixels(
                az=self.azimuth_start_time + self.lines_step * 800,
                rng=self.range_start_time_s + self.samples_step * 103520.634,
            )


class TestRasterLayoutOverlappingBursts:
    """Testing layout.py RasterLayout dataclass with 3 overlapping bursts"""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.base_azimuth_start_time = PreciseDateTime.from_utc_string("03-MAR-2021 11:06:56.329529000000")
        self.base_range_start_time = 0.00436789112663711
        self.burst_lines = 1200
        self.samples = 20000
        self.overlapping_burst_num = 3
        self.samples_step = 2.8e-09
        self.lines_step = 0.05
        self.burst_start_times = [self.base_azimuth_start_time + delta for delta in (0, 25, 50)]
        self.overlapping_bursts_layout = [
            L1BurstLayout(
                burst_id=b_id + 1,
                lines=self.burst_lines,
                samples=self.samples,
                lines_start=az_start,
                samples_start=self.base_range_start_time,
                lines_step=self.lines_step,
                samples_step=self.samples_step,
            )
            for b_id, az_start in enumerate(self.burst_start_times)
        ]
        self.overlapping_bursts_layout_lines = self.burst_lines * self.overlapping_burst_num

        self.expected_mid_azimuth_time = PreciseDateTime.from_utc_string("03-MAR-2021 11:07:51.304529000000")
        self.expected_mid_ground_range = 250000
        self.expected_mid_range = 0.00439588972663711

    def test_init(self) -> None:
        """Testing class init"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        assert isinstance(raster_layout, L1RasterLayout)

    def test_init_error(self) -> None:
        """Testing class init, raising error"""
        with pytest.raises(ValueError):
            L1RasterLayout(
                lines=self.overlapping_bursts_layout_lines,
                samples=self.samples,
                bursts=[],
            )

    def test_input_variables_accessibility(self) -> None:
        """Testing retrieval of input variables"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        assert raster_layout.lines == self.overlapping_bursts_layout_lines
        assert raster_layout.samples == self.samples
        assert len(raster_layout.bursts) == len(self.overlapping_bursts_layout)
        for burst_id, burst in enumerate(self.overlapping_bursts_layout):
            assert burst == raster_layout.bursts[burst_id]

    def test_computed_burst_ids(self) -> None:
        """Testing computed properties, burst_ids"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        assert raster_layout.burst_ids == [b + 1 for b in range(self.overlapping_burst_num)]

    def test_computed_burst_starting_line_offset(self) -> None:
        """Testing computed properties, burst_starting_line_offset"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        np.testing.assert_array_equal(
            raster_layout.burst_starting_line_offsets,
            np.array([b * self.burst_lines for b in range(self.overlapping_burst_num)]),
        )

    def test_computed_azimuth_start_time(self) -> None:
        """Testing computed properties, azimuth_start_time"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        np.testing.assert_allclose(
            raster_layout.raster_azimuth_axis[0] - self.base_azimuth_start_time,
            0,
            atol=1e-9,
            rtol=0,
        )

    def test_computed_range_start_time(self) -> None:
        """Testing computed properties, range_start_time"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        np.testing.assert_allclose(
            raster_layout.raster_range_axis[0],
            self.base_range_start_time,
            atol=1e-12,
            rtol=0,
        )

    def test_computed_mid_azimuth_time(self) -> None:
        """Testing computed properties, mid_azimuth_time"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        np.testing.assert_allclose(
            raster_layout.mid_swath_azimuth - self.expected_mid_azimuth_time,
            0,
            atol=1e-9,
            rtol=0,
        )

    def test_computed_mid_range_time(self) -> None:
        """Testing computed properties, mid_range_time"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        np.testing.assert_allclose(
            raster_layout.mid_swath_range,
            self.expected_mid_range,
            atol=1e-12,
            rtol=0,
        )

    def test_computed_raster_azimuth_axis(self) -> None:
        """Testing computed properties, raster azimuth axis"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        delta_times = raster_layout.raster_azimuth_axis - np.concatenate(
            [b.azimuth_axis for b in self.overlapping_bursts_layout]
        )
        np.testing.assert_allclose(
            delta_times.astype(float),
            np.zeros(raster_layout.raster_azimuth_axis.size),
            atol=1e-9,
            rtol=0,
        )

    def test_computed_raster_range_axis(self) -> None:
        """Testing computed properties, raster range axis"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        np.testing.assert_allclose(
            raster_layout.raster_range_axis,
            self.overlapping_bursts_layout[0].range_axis,
            atol=1e-9,
            rtol=0,
        )

    def test_get_burst_layout(self) -> None:
        """Testing get_burst_layout method"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        burst = raster_layout.get_burst_layout(burst_id=1)
        assert burst.burst_id == 1

    def test_get_burst_layout_error(self) -> None:
        """Testing get_burst_layout method, raising error"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        with pytest.raises(InvalidBurstIdError):
            raster_layout.get_burst_layout(burst_id=7)

    def test_get_burst_start_times(self) -> None:
        """Testing get_burst_start_times method"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        for b_id, burst_start_time in enumerate(self.burst_start_times):
            az_time, rng_time = raster_layout.get_burst_start_coordinates(burst_id=b_id + 1)
            np.testing.assert_allclose(
                az_time - burst_start_time,
                0,
                atol=1e-9,
                rtol=0,
            )
            np.testing.assert_allclose(
                rng_time - self.base_range_start_time,
                0,
                atol=1e-12,
                rtol=0,
            )

    def test_get_burst_start_times_error(self) -> None:
        """Testing get_burst_start_times method, raising error"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        with pytest.raises(InvalidBurstIdError):
            raster_layout.get_burst_start_coordinates(burst_id=16)

    def test_get_burst_lines(self) -> None:
        """Testing get_burst_lines method"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        for b_id in range(self.overlapping_burst_num):
            np.testing.assert_allclose(
                raster_layout.get_burst_lines(burst_id=b_id + 1),
                self.burst_lines,
                atol=1e-9,
                rtol=0,
            )

    def test_get_burst_lines_error(self) -> None:
        """Testing get_burst_lines method, raising error"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        with pytest.raises(InvalidBurstIdError):
            raster_layout.get_burst_lines(burst_id=16)

    def test_get_burst_samples(self) -> None:
        """Testing get_burst_samples method"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        for b_id in range(self.overlapping_burst_num):
            np.testing.assert_allclose(
                raster_layout.get_burst_lines(burst_id=b_id + 1),
                self.burst_lines,
                atol=1e-9,
                rtol=0,
            )

    def test_get_burst_samples_error(self) -> None:
        """Testing get_burst_samples method, raising error"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        with pytest.raises(InvalidBurstIdError):
            raster_layout.get_burst_samples(burst_id=16)

    def test_check_azimuth_time_in_raster_true(self) -> None:
        """Testing check_azimuth_time_in_raster, with positive outcome"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        assert raster_layout.is_azimuth_in_raster(
            az=self.base_azimuth_start_time + self.overlapping_bursts_layout_lines // 5 * self.lines_step
        )

    def test_check_azimuth_time_in_raster_false(self) -> None:
        """Testing check_azimuth_time_in_raster, with negative outcome"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        assert not raster_layout.is_azimuth_in_raster(
            az=self.base_azimuth_start_time + self.overlapping_bursts_layout_lines * 2 * self.lines_step
        )

    def test_check_range_time_in_raster_true(self) -> None:
        """Testing check_range_time_in_raster, with positive outcome"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        assert raster_layout.is_range_in_raster(rng=self.base_range_start_time + self.samples // 5 * self.samples_step)

    def test_check_range_time_in_raster_false(self) -> None:
        """Testing check_range_time_in_raster, with negative outcome"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        assert not raster_layout.is_range_in_raster(
            rng=self.base_range_start_time + self.samples * 2 * self.samples_step
        )

    def test_azimuth_time_to_bursts_association_error(self) -> None:
        """Testing azimuth_time_to_bursts_association, raising error"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        with pytest.raises(AzimuthExceedsBoundariesError):
            raster_layout.azimuth_to_bursts_association(az=self.base_azimuth_start_time - 1)

    def test_azimuth_time_to_bursts_association_1(self) -> None:
        """Testing azimuth_time_to_bursts_association, with 1 match"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        assert raster_layout.azimuth_to_bursts_association(az=self.burst_start_times[0] + self.lines_step) == [1]

    def test_azimuth_time_to_bursts_association_2(self) -> None:
        """Testing azimuth_time_to_bursts_association, with 2 matches"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        assert raster_layout.azimuth_to_bursts_association(az=self.burst_start_times[1] + self.lines_step) == [1, 2]

    def test_azimuth_time_to_bursts_association_3(self) -> None:
        """Testing azimuth_time_to_bursts_association, with 3 matches"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        assert raster_layout.azimuth_to_bursts_association(az=self.burst_start_times[2] + self.lines_step) == [1, 2, 3]

    def test_range_time_to_bursts_association_error(self) -> None:
        """Testing range_time_to_bursts_association, raising error"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        with pytest.raises(RangeExceedsBoundariesError):
            raster_layout.range_to_bursts_association(rng=self.base_range_start_time - self.samples_step)

    def test_range_time_to_bursts_association(self) -> None:
        """Testing range_time_to_bursts_association"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        assert raster_layout.range_to_bursts_association(rng=self.base_range_start_time + self.samples_step) == [
            1,
            2,
            3,
        ]

    def test_azimuth_time_to_pixel_conversion_single(self) -> None:
        """Testing azimuth_time_to_pixel_conversion"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        az_pixel = raster_layout.azimuth_to_pixel_conversion(az=self.base_azimuth_start_time + self.lines_step * 150)
        assert len(az_pixel) == 1
        assert az_pixel[0][0] == 1
        np.testing.assert_allclose(az_pixel[0][1], 150, atol=1e-9, rtol=0)

    def test_azimuth_time_to_pixel_conversion_single_error(self) -> None:
        """Testing azimuth_time_to_pixel_conversion, single return, raising error"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        with pytest.raises(AzimuthExceedsBoundariesError):
            raster_layout.azimuth_to_pixel_conversion(az=self.base_azimuth_start_time + self.lines_step * 15000)

    def test_azimuth_time_to_pixel_conversion_multiple(self) -> None:
        """Testing azimuth_time_to_pixel_conversion, multiple returns"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        az_pixel = raster_layout.azimuth_to_pixel_conversion(az=self.base_azimuth_start_time + self.lines_step * 1350)
        assert len(az_pixel) == 2
        assert az_pixel[0][0] == 2
        np.testing.assert_allclose(az_pixel[0][1], 2050, atol=1e-9, rtol=0)
        assert az_pixel[1][0] == 3
        np.testing.assert_allclose(az_pixel[1][1], 2750, atol=1e-9, rtol=0)

    def test_azimuth_time_to_pixel_conversion_with_burst_id(self) -> None:
        """Testing azimuth_time_to_pixel_conversion, providing burst id"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        az_pixel = raster_layout.azimuth_to_pixel_conversion(
            az=self.base_azimuth_start_time + self.lines_step * 1350, burst_id=2
        )
        assert len(az_pixel) == 1
        assert az_pixel[0][0] == 2
        np.testing.assert_allclose(az_pixel[0][1], 2050, atol=1e-9, rtol=0)

    def test_azimuth_time_to_pixel_conversion_with_burst_id_error(self) -> None:
        """Testing azimuth_time_to_pixel_conversion, raising error"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        with pytest.raises(InvalidBurstIdError):
            raster_layout.azimuth_to_pixel_conversion(
                az=self.base_azimuth_start_time + self.lines_step * 1350, burst_id=5
            )

    def test_range_time_to_pixel_conversion_multiple(self) -> None:
        """Testing range_time_to_pixel_conversion, multiple returns"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        rng_pixel = raster_layout.range_to_pixel_conversion(rng=self.base_range_start_time + self.samples_step * 150)
        assert len(rng_pixel) == 3
        assert rng_pixel[0][0] == 1
        np.testing.assert_allclose(rng_pixel[0][1], 150, atol=1e-9, rtol=0)
        assert rng_pixel[1][0] == 2
        np.testing.assert_allclose(rng_pixel[1][1], 150, atol=1e-9, rtol=0)
        assert rng_pixel[2][0] == 3
        np.testing.assert_allclose(rng_pixel[2][1], 150, atol=1e-9, rtol=0)

    def test_range_time_to_pixel_conversion_multiple_error(self) -> None:
        """Testing range_time_to_pixel_conversion, multiple return, raising error"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        with pytest.raises(RangeExceedsBoundariesError):
            raster_layout.range_to_pixel_conversion(rng=self.base_range_start_time + self.samples_step * 30000)

    def test_range_time_to_pixel_conversion_with_burst_id(self) -> None:
        """Testing range_time_to_pixel_conversion, providing burst id"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        rng_pixel = raster_layout.range_to_pixel_conversion(
            rng=self.base_range_start_time + self.samples_step * 150, burst_id=2
        )
        assert len(rng_pixel) == 1
        assert rng_pixel[0][0] == 2
        np.testing.assert_allclose(rng_pixel[0][1], 150, atol=1e-9, rtol=0)

    def test_range_time_to_pixel_conversion_with_burst_id_error(self) -> None:
        """Testing range_time_to_pixel_conversion, raising error"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        with pytest.raises(InvalidBurstIdError):
            raster_layout.range_to_pixel_conversion(
                rng=self.base_range_start_time + self.samples_step * 150, burst_id=5
            )

    def test_azimuth_pixel_to_time_conversion(self) -> None:
        """Testing test_azimuth_pixel_to_time_conversion"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        t_az = raster_layout.pixel_to_azimuth_conversion(az_pixel_index=150)
        time_delta = t_az - (self.base_azimuth_start_time + 150 * self.lines_step)
        np.testing.assert_allclose(time_delta, 0, atol=1e9, rtol=0)

    def test_azimuth_pixel_to_time_conversion_sub_pixel(self) -> None:
        """Testing test_azimuth_pixel_to_time_conversion"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        t_az = raster_layout.pixel_to_azimuth_conversion(az_pixel_index=150.84)
        time_delta = t_az - (self.base_azimuth_start_time + 150.84 * self.lines_step)
        np.testing.assert_allclose(time_delta, 0, atol=1e9, rtol=0)

    def test_azimuth_pixel_to_time_conversion_error(self) -> None:
        """Testing test_azimuth_pixel_to_time_conversion, raising error"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        for t in [-1, self.overlapping_bursts_layout_lines, self.overlapping_bursts_layout_lines + 10]:
            with pytest.raises(AzimuthExceedsBoundariesError):
                raster_layout.pixel_to_azimuth_conversion(az_pixel_index=t)

    def test_range_pixel_to_time_conversion(self) -> None:
        """Testing range_pixel_to_time_conversion"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        t_rng = raster_layout.pixel_to_range_conversion(rng_pixel_index=150)
        np.testing.assert_allclose(t_rng, self.base_range_start_time + 150 * self.samples_step, atol=1e9, rtol=0)

    def test_range_pixel_to_time_conversion_sub_pixel(self) -> None:
        """Testing range_pixel_to_time_conversion"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        t_rng = raster_layout.pixel_to_range_conversion(rng_pixel_index=150.87)
        np.testing.assert_allclose(t_rng, self.base_range_start_time + 150.87 * self.samples_step, atol=1e9, rtol=0)

    def test_range_pixel_to_time_conversion_error(self) -> None:
        """Testing range_pixel_to_time_conversion, raising error"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        for t in [-1, self.samples, self.samples + 10]:
            with pytest.raises(RangeExceedsBoundariesError):
                raster_layout.pixel_to_range_conversion(rng_pixel_index=t)

    def test_is_roi_in_raster_true(self) -> None:
        """Testing is_roi_in_raster, true"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        assert raster_layout.is_roi_in_raster(roi=[100, 1500, 850, 1653])

    def test_is_roi_in_raster_error_false(self) -> None:
        """Testing is_roi_in_raster, false"""
        raster_layout = L1RasterLayout(
            lines=self.overlapping_bursts_layout_lines,
            samples=self.samples,
            bursts=self.overlapping_bursts_layout,
        )
        assert not raster_layout.is_roi_in_raster(roi=[-5, 1500, 850, 1653])


class TestRasterLayoutContiguousBursts:
    """Testing layout.py RasterLayout dataclass with 2 contiguous bursts"""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.base_azimuth_start_time = PreciseDateTime.from_utc_string("03-MAR-2021 11:06:56.329529000000")
        self.base_range_start_time = 0.00436789112663711
        self.burst_lines = 1200
        self.samples = 20000
        self.ground_range_start_m = 0
        self.ground_range_step_m = 25
        self.burst_num = 2
        self.samples_step = 2.8e-09
        self.lines_step = 0.05
        self.burst_start_times = [self.base_azimuth_start_time + delta for delta in (0, 59.95)]
        self.bursts_layout = [
            L1BurstLayout(
                burst_id=b_id + 1,
                lines=self.burst_lines,
                samples=self.samples,
                lines_start=az_start,
                samples_start=self.base_range_start_time,
                lines_step=self.lines_step,
                samples_step=self.samples_step,
            )
            for b_id, az_start in enumerate(self.burst_start_times)
        ]
        self.bursts_layout_lines = self.burst_lines * self.burst_num

        self.expected_mid_azimuth_time = PreciseDateTime.from_utc_string("03-MAR-2021 11:07:56.279529000000")
        self.expected_mid_ground_range = 250000
        self.expected_mid_range = 0.00439589112663711
        self.expected_ground_range_axis = (
            np.arange(0, self.samples, 1) * self.ground_range_step_m + self.ground_range_start_m
        )

    def test_init(self) -> None:
        """Testing class init"""
        raster_layout = L1RasterLayout(
            lines=self.bursts_layout_lines,
            samples=self.samples,
            bursts=self.bursts_layout,
        )
        assert isinstance(raster_layout, L1RasterLayout)

    def test_computed_burst_starting_line_offset(self) -> None:
        """Testing computed properties, burst_starting_line_offset"""
        raster_layout = L1RasterLayout(
            lines=self.bursts_layout_lines,
            samples=self.samples,
            bursts=self.bursts_layout,
        )
        np.testing.assert_array_equal(
            raster_layout.burst_starting_line_offsets,
            np.array([b * self.burst_lines for b in range(self.burst_num)]),
        )

    def test_computed_mid_azimuth_time(self) -> None:
        """Testing computed properties, mid_azimuth_time"""
        raster_layout = L1RasterLayout(
            lines=self.bursts_layout_lines,
            samples=self.samples,
            bursts=self.bursts_layout,
        )
        np.testing.assert_allclose(
            raster_layout.mid_swath_azimuth - self.expected_mid_azimuth_time,
            0,
            atol=1e-9,
            rtol=0,
        )

    def test_computed_raster_azimuth_axis(self) -> None:
        """Testing computed properties, raster azimuth axis"""
        raster_layout = L1RasterLayout(
            lines=self.bursts_layout_lines,
            samples=self.samples,
            bursts=self.bursts_layout,
        )
        delta_times = raster_layout.raster_azimuth_axis - np.concatenate([b.azimuth_axis for b in self.bursts_layout])
        np.testing.assert_allclose(
            delta_times.astype(float),
            np.zeros(raster_layout.raster_azimuth_axis.size),
            atol=1e-9,
            rtol=0,
        )

    def test_azimuth_time_to_bursts_association_1(self) -> None:
        """Testing azimuth_time_to_bursts_association, with 1 match"""
        raster_layout = L1RasterLayout(
            lines=self.bursts_layout_lines,
            samples=self.samples,
            bursts=self.bursts_layout,
        )
        assert raster_layout.azimuth_to_bursts_association(az=self.burst_start_times[0] + self.lines_step) == [1]

    def test_azimuth_time_to_bursts_association_2(self) -> None:
        """Testing azimuth_time_to_bursts_association, with 2 matches"""
        raster_layout = L1RasterLayout(
            lines=self.bursts_layout_lines,
            samples=self.samples,
            bursts=self.bursts_layout,
        )
        assert raster_layout.azimuth_to_bursts_association(az=self.burst_start_times[1]) == [1, 2]

    def test_azimuth_time_to_pixel_conversion_single(self) -> None:
        """Testing azimuth_time_to_pixel_conversion"""
        raster_layout = L1RasterLayout(
            lines=self.bursts_layout_lines,
            samples=self.samples,
            bursts=self.bursts_layout,
        )
        az_pixel = raster_layout.azimuth_to_pixel_conversion(az=self.base_azimuth_start_time + self.lines_step * 150)
        assert len(az_pixel) == 1
        assert az_pixel[0][0] == 1
        np.testing.assert_allclose(az_pixel[0][1], 150, atol=1e-9, rtol=0)

    def test_azimuth_time_to_pixel_conversion_single_error(self) -> None:
        """Testing azimuth_time_to_pixel_conversion, single return, raising error"""
        raster_layout = L1RasterLayout(
            lines=self.bursts_layout_lines,
            samples=self.samples,
            bursts=self.bursts_layout,
        )
        with pytest.raises(AzimuthExceedsBoundariesError):
            raster_layout.azimuth_to_pixel_conversion(az=self.base_azimuth_start_time + self.lines_step * 15000)

    def test_azimuth_time_to_pixel_conversion_multiple(self) -> None:
        """Testing azimuth_time_to_pixel_conversion, multiple returns"""
        raster_layout = L1RasterLayout(
            lines=self.bursts_layout_lines,
            samples=self.samples,
            bursts=self.bursts_layout,
        )
        az_pixel = raster_layout.azimuth_to_pixel_conversion(az=self.burst_start_times[1])
        assert len(az_pixel) == 2
        assert az_pixel[0][0] == 1
        np.testing.assert_allclose(az_pixel[0][1], 1199, atol=1e-9, rtol=0)
        assert az_pixel[1][0] == 2
        np.testing.assert_allclose(az_pixel[1][1], 1200, atol=1e-9, rtol=0)

    def test_azimuth_pixel_to_time_conversion(self) -> None:
        """Testing test_azimuth_pixel_to_time_conversion"""
        raster_layout = L1RasterLayout(
            lines=self.bursts_layout_lines,
            samples=self.samples,
            bursts=self.bursts_layout,
        )
        t_az = raster_layout.pixel_to_azimuth_conversion(az_pixel_index=150)
        time_delta = t_az - (self.base_azimuth_start_time + 150 * self.lines_step)
        np.testing.assert_allclose(time_delta, 0, atol=1e9, rtol=0)

    def test_azimuth_pixel_to_time_conversion_error(self) -> None:
        """Testing test_azimuth_pixel_to_time_conversion, raising error"""
        raster_layout = L1RasterLayout(
            lines=self.bursts_layout_lines,
            samples=self.samples,
            bursts=self.bursts_layout,
        )
        for t in [-1, self.bursts_layout_lines, self.bursts_layout_lines + 10]:
            with pytest.raises(AzimuthExceedsBoundariesError):
                raster_layout.pixel_to_azimuth_conversion(az_pixel_index=t)


class TestRasterLayoutSeparateBursts:
    """Testing layout.py RasterLayout dataclass with 2 separate bursts"""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.base_azimuth_start_time = PreciseDateTime.from_utc_string("03-MAR-2021 11:06:56.329529000000")
        self.base_range_start_time = 0.00436789112663711
        self.burst_lines = 1200
        self.samples = 20000
        self.ground_range_start_m = 0
        self.ground_range_step_m = 25
        self.burst_num = 2
        self.samples_step = 2.8e-09
        self.lines_step = 0.05
        self.burst_start_times = [self.base_azimuth_start_time + delta for delta in (0, 65)]
        self.bursts_layout = [
            L1BurstLayout(
                burst_id=b_id + 1,
                lines=self.burst_lines,
                samples=self.samples,
                lines_start=az_start,
                samples_start=self.base_range_start_time,
                lines_step=self.lines_step,
                samples_step=self.samples_step,
            )
            for b_id, az_start in enumerate(self.burst_start_times)
        ]
        self.bursts_layout_lines = self.burst_lines * self.burst_num

        self.expected_mid_azimuth_time = PreciseDateTime.from_utc_string("03-MAR-2021 11:07:58.804529000000")
        self.expected_mid_ground_range = 250000
        self.expected_mid_range = 0.00439589112663711
        self.expected_ground_range_axis = (
            np.arange(0, self.samples, 1) * self.ground_range_step_m + self.ground_range_start_m
        )

    def test_init(self) -> None:
        """Testing class init"""
        raster_layout = L1RasterLayout(
            lines=self.bursts_layout_lines,
            samples=self.samples,
            bursts=self.bursts_layout,
        )
        assert isinstance(raster_layout, L1RasterLayout)

    def test_computed_burst_starting_line_offset(self) -> None:
        """Testing computed properties, burst_starting_line_offset"""
        raster_layout = L1RasterLayout(
            lines=self.bursts_layout_lines,
            samples=self.samples,
            bursts=self.bursts_layout,
        )
        np.testing.assert_array_equal(
            raster_layout.burst_starting_line_offsets,
            np.array([b * self.burst_lines for b in range(self.burst_num)]),
        )

    def test_computed_mid_azimuth_time(self) -> None:
        """Testing computed properties, mid_azimuth_time"""
        raster_layout = L1RasterLayout(
            lines=self.bursts_layout_lines,
            samples=self.samples,
            bursts=self.bursts_layout,
        )
        np.testing.assert_allclose(
            raster_layout.mid_swath_azimuth - self.expected_mid_azimuth_time,
            0,
            atol=1e-9,
            rtol=0,
        )

    def test_computed_raster_azimuth_axis(self) -> None:
        """Testing computed properties, raster azimuth axis"""
        raster_layout = L1RasterLayout(
            lines=self.bursts_layout_lines,
            samples=self.samples,
            bursts=self.bursts_layout,
        )
        delta_times = raster_layout.raster_azimuth_axis - np.concatenate([b.azimuth_axis for b in self.bursts_layout])
        np.testing.assert_allclose(
            delta_times.astype(float),
            np.zeros(raster_layout.raster_azimuth_axis.size),
            atol=1e-9,
            rtol=0,
        )

    def test_azimuth_time_to_bursts_association_1(self) -> None:
        """Testing azimuth_time_to_bursts_association, with 1 match"""
        raster_layout = L1RasterLayout(
            lines=self.bursts_layout_lines,
            samples=self.samples,
            bursts=self.bursts_layout,
        )
        assert raster_layout.azimuth_to_bursts_association(az=self.burst_start_times[0] + self.lines_step) == [1]

    def test_azimuth_time_to_pixel_conversion_single(self) -> None:
        """Testing azimuth_time_to_pixel_conversion"""
        raster_layout = L1RasterLayout(
            lines=self.bursts_layout_lines,
            samples=self.samples,
            bursts=self.bursts_layout,
        )
        az_pixel = raster_layout.azimuth_to_pixel_conversion(az=self.base_azimuth_start_time + self.lines_step * 150)
        assert len(az_pixel) == 1
        assert az_pixel[0][0] == 1
        np.testing.assert_allclose(az_pixel[0][1], 150, atol=1e-9, rtol=0)

    def test_azimuth_time_to_pixel_conversion_single_error(self) -> None:
        """Testing azimuth_time_to_pixel_conversion, single return, raising error"""
        raster_layout = L1RasterLayout(
            lines=self.bursts_layout_lines,
            samples=self.samples,
            bursts=self.bursts_layout,
        )
        with pytest.raises(AzimuthExceedsBoundariesError):
            raster_layout.azimuth_to_pixel_conversion(az=self.base_azimuth_start_time + self.lines_step * 15000)

    def test_azimuth_pixel_to_time_conversion(self) -> None:
        """Testing test_azimuth_pixel_to_time_conversion"""
        raster_layout = L1RasterLayout(
            lines=self.bursts_layout_lines,
            samples=self.samples,
            bursts=self.bursts_layout,
        )
        t_az = raster_layout.pixel_to_azimuth_conversion(az_pixel_index=150)
        time_delta = t_az - (self.base_azimuth_start_time + 150 * self.lines_step)
        np.testing.assert_allclose(time_delta, 0, atol=1e9, rtol=0)

    def test_azimuth_pixel_to_time_conversion_error(self) -> None:
        """Testing test_azimuth_pixel_to_time_conversion, raising error"""
        raster_layout = L1RasterLayout(
            lines=self.bursts_layout_lines,
            samples=self.samples,
            bursts=self.bursts_layout,
        )
        for t in [-1, self.bursts_layout_lines, self.bursts_layout_lines + 10]:
            with pytest.raises(AzimuthExceedsBoundariesError):
                raster_layout.pixel_to_azimuth_conversion(az_pixel_index=t)
