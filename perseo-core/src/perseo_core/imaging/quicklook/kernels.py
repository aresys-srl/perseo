# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Filtering kernels for quick-look generation.

Kernels honor the QuicklookKernel contract: they filter and downsample a SAR data
model along the azimuth (lines) and range (samples) dimensions, returning a new
data model with subsampled axis coordinates. How each kernel accomplishes this is an
implementation detail; the kernels shipped here happen to be separable and share a
private helper, but any filtering strategy is allowed as long as the apply() contract
is honored.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np
from scipy import sparse
from scipy.ndimage import uniform_filter1d

from perseo_core.imaging.models import (
    AZIMUTH_DIM,
    RANGE_DIM,
    SARDataModel,
    _interpolate_axis_coords,
    validate_sar_data_model,
)

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["BoxKernel", "DecimationKernel", "GaussianKernel", "QuicklookKernel", "Uniform1DKernel"]


class QuicklookKernel(Protocol):
    """Contract for quick-look filtering kernels.

    A kernel filters and downsamples a SAR data model along the azimuth (lines) and
    range (samples) dimensions. The implementation is free to choose the filtering
    strategy and is responsible for honoring the contract, i.e. returning the filtered
    and downsampled data model with its axis coordinates subsampled accordingly.
    """

    def apply(
        self,
        data: SARDataModel,
        lines_subsampling_factor: float,
        samples_subsampling_factor: float,
    ) -> SARDataModel:
        """Filter and downsample a SAR data model.

        Parameters
        ----------
        data : SARDataModel
            two-dimensional SAR data model, with ``azimuth`` and ``range`` dimensions
        lines_subsampling_factor : float
            subsampling factor along the azimuth (line) dimension
        samples_subsampling_factor : float
            subsampling factor along the range (sample) dimension

        Returns
        -------
        SARDataModel
            filtered and downsampled data model

        """
        ...


def _per_axis_filtering(
    data: SARDataModel,
    lines_subsampling_factor: float,
    samples_subsampling_factor: float,
    build_axis_filter: Callable[[np.ndarray, np.ndarray, float], sparse.csc_matrix],
) -> SARDataModel:
    """Apply a separable filter along both raster dimensions.

    The axis filter is built independently for the azimuth and the range dimension,
    then applied through the matrix product ``kernel_azimuth @ raster @ kernel_range.T``.
    The output axis coordinates are linearly interpolated at the subsampled positions,
    so that the pixel-to-coordinate linkage is preserved.
    """
    validate_sar_data_model(data)
    raster_data = data.values
    n_lines, n_samples = raster_data.shape
    low_res_lines = np.arange(n_lines, step=lines_subsampling_factor)
    low_res_samples = np.arange(n_samples, step=samples_subsampling_factor)
    filtering_kernel_azimuth = build_axis_filter(np.arange(n_lines), low_res_lines, lines_subsampling_factor)
    filtering_kernel_range = build_axis_filter(np.arange(n_samples), low_res_samples, samples_subsampling_factor)
    filtered = filtering_kernel_azimuth @ raster_data @ filtering_kernel_range.T
    return SARDataModel(
        filtered,
        dims=(AZIMUTH_DIM, RANGE_DIM),
        coords={
            AZIMUTH_DIM: _interpolate_axis_coords(data[AZIMUTH_DIM].values, low_res_lines),
            RANGE_DIM: _interpolate_axis_coords(data[RANGE_DIM].values, low_res_samples),
        },
        attrs=data.attrs,
    )


def _box_axis_filter(
    full_res_indexes: np.ndarray, low_res_indexes: np.ndarray, filter_length: float
) -> sparse.csc_matrix:
    """Build a sparse averaging matrix mapping full resolution to subsampled positions."""
    threshold = np.floor(filter_length / 2) + 1
    t_mesh_out, t_mesh_in = np.meshgrid(full_res_indexes.flatten(), low_res_indexes.flatten())
    ql_filter = sparse.csc_matrix(np.abs(t_mesh_in - t_mesh_out) < threshold)
    return ql_filter / (threshold * 2 - 1)


def _decimation_axis_filter(
    full_res_indexes: np.ndarray,
    low_res_indexes: np.ndarray,
    filter_length: float,  # noqa: ARG001  # part of the axis filter signature
) -> sparse.csc_matrix:
    """Build a sparse matrix selecting the nearest full resolution pixel for each output position."""
    full_res_indexes = np.asarray(full_res_indexes).flatten()
    low_res_indexes = np.asarray(low_res_indexes).flatten()
    # sparse matrices require integer column indexes, so fractional positions
    # (coming from float subsampling factors) are rounded to the nearest pixel
    selected = np.clip(np.rint(low_res_indexes).astype(int), 0, full_res_indexes.size - 1)
    rows = np.arange(selected.size)
    data = np.ones(selected.size)
    return sparse.csc_matrix((data, (rows, selected)), shape=(selected.size, full_res_indexes.size))


def _gaussian_axis_filter(
    full_res_indexes: np.ndarray, low_res_indexes: np.ndarray, filter_length: float
) -> sparse.csc_matrix:
    """Build a sparse gaussian-weighted matrix mapping full resolution to subsampled positions."""
    threshold = np.floor(filter_length / 2) + 1
    sigma = filter_length / 2
    t_mesh_out, t_mesh_in = np.meshgrid(full_res_indexes.flatten(), low_res_indexes.flatten())
    distances = np.abs(t_mesh_in - t_mesh_out)
    weights = np.exp(-0.5 * (distances / sigma) ** 2) * (distances < threshold)
    row_sums = weights.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    return sparse.csc_matrix(weights / row_sums[:, None])


def _uniform_window_size(subsampling_factor: float) -> int:
    """Window size of the uniform filter matching the box kernel width."""
    return 2 * int(np.floor(subsampling_factor / 2)) + 1


class BoxKernel:
    """Averaging kernel: each output pixel is the mean of the input pixels within a centered window."""

    def apply(
        self,
        data: SARDataModel,
        lines_subsampling_factor: float,
        samples_subsampling_factor: float,
    ) -> SARDataModel:
        """Filter and downsample the data model by separable averaging along both dimensions."""
        return _per_axis_filtering(data, lines_subsampling_factor, samples_subsampling_factor, _box_axis_filter)


class Uniform1DKernel:
    """Averaging kernel based on ``scipy.ndimage.uniform_filter1d``.

    Each output pixel is the mean of the input pixels within a centered
    window of ``2 * floor(factor / 2) + 1`` pixels, matching the BoxKernel
    window width. The raster edges are zero-padded, so edge pixels are
    attenuated like BoxKernel. Filtering is performed by two
    ``uniform_filter1d`` passes, one per axis with the second in-place,
    which is faster and uses less memory than building sparse filtering
    matrices. Fractional subsampling factors are rounded to the nearest
    pixel position, like DecimationKernel.
    """

    def apply(
        self,
        data: SARDataModel,
        lines_subsampling_factor: float,
        samples_subsampling_factor: float,
    ) -> SARDataModel:
        """Filter and downsample the data model by separable uniform averaging."""
        validate_sar_data_model(data)
        raster_data = data.values
        low_res_lines = np.arange(raster_data.shape[0], step=lines_subsampling_factor)
        low_res_samples = np.arange(raster_data.shape[1], step=samples_subsampling_factor)
        filtered = uniform_filter1d(
            raster_data,
            size=_uniform_window_size(lines_subsampling_factor),
            axis=0,
            mode="constant",
            cval=0.0,
        )
        uniform_filter1d(
            filtered,
            size=_uniform_window_size(samples_subsampling_factor),
            axis=1,
            mode="constant",
            cval=0.0,
            output=filtered,
        )
        line_indexes = np.clip(np.rint(low_res_lines).astype(int), 0, raster_data.shape[0] - 1)
        sample_indexes = np.clip(np.rint(low_res_samples).astype(int), 0, raster_data.shape[1] - 1)
        return SARDataModel(
            filtered[line_indexes][:, sample_indexes],
            dims=(AZIMUTH_DIM, RANGE_DIM),
            coords={
                AZIMUTH_DIM: _interpolate_axis_coords(data[AZIMUTH_DIM].values, low_res_lines),
                RANGE_DIM: _interpolate_axis_coords(data[RANGE_DIM].values, low_res_samples),
            },
            attrs=data.attrs,
        )


class DecimationKernel:
    """Decimation kernel: each output pixel selects the nearest full resolution pixel, without averaging."""

    def apply(
        self,
        data: SARDataModel,
        lines_subsampling_factor: float,
        samples_subsampling_factor: float,
    ) -> SARDataModel:
        """Filter and downsample the data model by separable decimation along both dimensions."""
        return _per_axis_filtering(data, lines_subsampling_factor, samples_subsampling_factor, _decimation_axis_filter)


class GaussianKernel:
    """Gaussian kernel: weighted average within a centered gaussian window.

    Weights decay as a gaussian of the distance from the output position. Each row is normalized
    by its actual sum so that the filtered values keep their scale also at the raster edges.
    """

    def apply(
        self,
        data: SARDataModel,
        lines_subsampling_factor: float,
        samples_subsampling_factor: float,
    ) -> SARDataModel:
        """Filter and downsample the data model by separable gaussian averaging along both dimensions."""
        return _per_axis_filtering(data, lines_subsampling_factor, samples_subsampling_factor, _gaussian_axis_filter)
