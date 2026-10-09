# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for quicklook engine."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from perseo_core.imaging.models import (
    AZIMUTH_DIM,
    RANGE_DIM,
    SARDataModel,
    _interpolate_axis_coords,
    create_sar_data_model,
)
from perseo_core.imaging.quicklook.engine import quicklook_rendering
from perseo_core.imaging.quicklook.kernels import (
    BoxKernel,
    DecimationKernel,
    QuicklookKernel,
)
from perseo_core.imaging.quicklook.transforms import (
    DEFAULT_TRANSFORM,
    ClipPercentileTransform,
    PhaseTransform,
    ToDbTransform,
    to_db,
    to_phase,
)
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


class AllMeanKernel:
    """Test kernel: every output pixel is the mean of all input pixels."""

    def apply(
        self,
        data: SARDataModel,
        lines_subsampling_factor: float,
        samples_subsampling_factor: float,
    ) -> SARDataModel:
        low_res_lines = np.arange(data.shape[0], step=lines_subsampling_factor)
        low_res_samples = np.arange(data.shape[1], step=samples_subsampling_factor)
        values = np.full((low_res_lines.size, low_res_samples.size), data.values.mean())
        return SARDataModel(
            values,
            dims=(AZIMUTH_DIM, RANGE_DIM),
            coords={
                AZIMUTH_DIM: _interpolate_axis_coords(data[AZIMUTH_DIM].values, low_res_lines),
                RANGE_DIM: _interpolate_axis_coords(data[RANGE_DIM].values, low_res_samples),
            },
            attrs=data.attrs,
        )


class TestQuicklookGeneratorCore:
    def test_default_kernel_is_box(self) -> None:
        data = _make_data(DATA_4X4)
        result = quicklook_rendering(data, 2.0, 2.0, transform=PhaseTransform())
        np.testing.assert_allclose(result.values, to_phase(BoxKernel().apply(data, 2.0, 2.0).values))

    def test_default_transform_is_percentile_clipping(self) -> None:
        data = _make_data(DATA_4X4)
        filtered = BoxKernel().apply(data, 2.0, 2.0)
        expected = DEFAULT_TRANSFORM.post_filter(filtered)
        np.testing.assert_allclose(quicklook_rendering(data, 2.0, 2.0).values, expected.values)

    def test_db_transform_injection(self) -> None:
        data = _make_data(DATA_4X4)
        filtered = BoxKernel().apply(data, 2.0, 2.0)
        result = quicklook_rendering(data, 2.0, 2.0, transform=ToDbTransform())
        np.testing.assert_allclose(result.values, to_db(filtered.values))

    def test_phase_transform_injection(self) -> None:
        data = _make_data(DATA_4X4)
        filtered = BoxKernel().apply(data, 2.0, 2.0)
        result = quicklook_rendering(data, 2.0, 2.0, transform=PhaseTransform())
        np.testing.assert_allclose(result.values, to_phase(filtered.values))

    def test_percentile_transform_with_params(self) -> None:
        data = _make_data(DATA_4X4)
        filtered = BoxKernel().apply(data, 2.0, 2.0)
        transform = ClipPercentileTransform(min_percentile=10.0, max_percentile=90.0)
        expected = transform.post_filter(filtered)
        result = quicklook_rendering(data, 2.0, 2.0, transform=transform)
        np.testing.assert_allclose(result.values, expected.values)

    def test_decimation_kernel_injection(self) -> None:
        data = _make_data(DATA_4X4)
        result = quicklook_rendering(data, 2.0, 2.0, kernel=DecimationKernel(), transform=PhaseTransform())
        np.testing.assert_allclose(result.values, to_phase(np.array([[1.0, 3.0], [9.0, 11.0]])))

    def test_custom_kernel_and_transform(self) -> None:
        data = _make_data(DATA_4X4)
        result = quicklook_rendering(data, 2.0, 2.0, kernel=AllMeanKernel(), transform=ToDbTransform())
        np.testing.assert_allclose(result.values, to_db(np.full((2, 2), DATA_4X4.mean())))

    def test_float_subsampling_factors(self) -> None:
        data = _make_data(np.arange(25, dtype=float).reshape(5, 5))
        result = quicklook_rendering(data, 2.5, 2.5, transform=PhaseTransform())
        assert result.shape == (2, 2)

    def test_coordinates_and_attrs_are_preserved(self) -> None:
        data = _make_data(DATA_4X4, azimuth_step=10.0, range_step=5.0)
        data.attrs["mission"] = "test"
        result = quicklook_rendering(data, 2.0, 2.0, transform=ToDbTransform())
        assert result.attrs["mission"] == "test"
        assert result.azimuth.values[1] == data.azimuth.values[2]
        np.testing.assert_allclose(result.range.values, data.range.values[[0, 2]])

    def test_kernel_satisfies_contract(self) -> None:
        kernel: QuicklookKernel = AllMeanKernel()
        assert kernel.apply(_make_data(DATA_4X4), 2.0, 2.0).shape == (2, 2)

    def test_rejects_generic_data_array(self) -> None:
        generic = xr.DataArray(np.zeros((2, 2)), dims=("x", "y"))
        with pytest.raises(ValueError, match="dimensions"):
            quicklook_rendering(generic, 2.0, 2.0)


class TestAmplitudeVsPhaseFiltering:
    """Amplitude transformations filter magnitudes, the phase transformation filters complex data."""

    COMPLEX_RASTER = np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])
    # box kernel with factor 2 weights each of the 4 pixels by 1/9

    def test_amplitude_modes_filter_magnitudes(self) -> None:
        data = _make_data(self.COMPLEX_RASTER)
        result = quicklook_rendering(data, 2.0, 2.0, transform=ToDbTransform())
        # magnitude route: each |pixel| = sqrt(2), filtered = 4*sqrt(2)/9
        expected = to_db(np.full((1, 1), 4 * np.sqrt(2.0) / 9))
        np.testing.assert_allclose(result.values, expected)

    def test_phase_mode_filters_complex_data(self) -> None:
        data = _make_data(self.COMPLEX_RASTER)
        result = quicklook_rendering(data, 2.0, 2.0, transform=PhaseTransform())
        # complex route: filtered = (4+0j)/9, phase = 0
        np.testing.assert_allclose(result.values, [[0.0]])

    def test_clip_mode_filters_magnitudes(self) -> None:
        data = _make_data(self.COMPLEX_RASTER)
        transform = ClipPercentileTransform(min_percentile=0.0, max_percentile=100.0)
        result = quicklook_rendering(data, 2.0, 2.0, transform=transform)
        # magnitude route: 4*sqrt(2)/9
        np.testing.assert_allclose(result.values, np.full((1, 1), 4 * np.sqrt(2.0) / 9))


class TestNonMonotonicAzimuth:
    def test_burst_overlap_keeps_positional_linkage(self) -> None:
        raster = np.arange(10, dtype=float).reshape(5, 2)
        base = PreciseDateTime.from_numeric_datetime(2024, 1, 1)
        # burst overlap: the time goes back at index 3
        azimuth_times = [
            base + 0.0,
            base + 1.0,
            base + 2.0,
            base + 1.5,
            base + 2.5,
        ]
        data = create_sar_data_model(raster, azimuth_times, np.array([0.0, 1.0]))
        result = DecimationKernel().apply(data, 2.0, 1.0)
        # positions [0, 2, 4] keep the pixel-to-time linkage
        assert result.azimuth.values[0] == azimuth_times[0]
        assert result.azimuth.values[1] == azimuth_times[2]
        assert result.azimuth.values[2] == azimuth_times[4]
        np.testing.assert_allclose(result.values, raster[[0, 2, 4]])

    def test_engine_handles_burst_overlap(self) -> None:
        raster = np.arange(10, dtype=float).reshape(5, 2)
        base = PreciseDateTime.from_numeric_datetime(2024, 1, 1)
        azimuth_times = [base + 0.0, base + 1.0, base + 2.0, base + 1.5, base + 2.5]
        data = create_sar_data_model(raster, azimuth_times, np.array([0.0, 1.0]))
        result = quicklook_rendering(data, 2.0, 1.0, transform=PhaseTransform())
        assert result.shape == (3, 2)
        assert result.azimuth.values[2] == azimuth_times[4]
