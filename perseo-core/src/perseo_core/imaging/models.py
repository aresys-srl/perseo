# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""General SAR data model based on xarray.

The model associates a two-dimensional raster with its azimuth and range axes. The
azimuth axis holds one PreciseDateTime coordinate per pixel, while the range axis
holds float coordinates. The azimuth axis may be non-monotonically increasing, e.g.
in TOPS/ScanSAR acquisitions where overlapping bursts restart the time axis: only
positional operations (e.g. ``isel``, slicing) are safe on such axes, label-based
selection and interpolation must be avoided.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Sequence

    from perseo_core.timing.precise_datetime import PreciseDateTime

__all__ = ["SARDataModel", "create_sar_data_model", "validate_sar_data_model"]

AZIMUTH_DIM = "azimuth"
RANGE_DIM = "range"


class SARDataModel(xr.DataArray):
    """SAR data model with azimuth and range axes.

    The azimuth axis holds one PreciseDateTime coordinate per pixel, while the range
    axis holds float coordinates. The azimuth axis may be non-monotonically
    increasing, e.g. in TOPS/ScanSAR acquisitions where overlapping bursts restart
    the time axis: only positional operations (e.g. ``isel``, slicing) are safe on
    such axes, label-based selection and interpolation must be avoided.
    """

    __slots__ = ()


def create_sar_data_model(
    raster_data: np.ndarray,
    azimuth_times: Sequence[PreciseDateTime],
    range_coords: npt.NDArray[np.floating],
) -> SARDataModel:
    """Create the SAR data model from a raster and its axis coordinates.

    Parameters
    ----------
    raster_data : np.ndarray
        two-dimensional raster data, with shape (azimuth, range)
    azimuth_times : Sequence[PreciseDateTime]
        acquisition time of each azimuth pixel, one per raster row
    range_coords : npt.NDArray[np.floating]
        range coordinate of each pixel, one per raster column

    Returns
    -------
    SARDataModel
        data model with ``azimuth`` and ``range`` dimensions and matching coordinates

    Raises
    ------
    ValueError
        if the raster is not two-dimensional or the coordinate counts do not match
        the raster shape

    """
    raster_data = np.asarray(raster_data)
    azimuth_coords = np.asarray(azimuth_times, dtype=object)
    range_coords = np.asarray(range_coords, dtype=float)

    if raster_data.ndim != 2:
        msg = f"Raster data must be two-dimensional, got shape {raster_data.shape}."
        raise ValueError(msg)
    if azimuth_coords.size != raster_data.shape[0]:
        msg = (
            f"Azimuth times count ({azimuth_coords.size}) must match the raster azimuth "
            f"dimension ({raster_data.shape[0]})."
        )
        raise ValueError(msg)
    if range_coords.size != raster_data.shape[1]:
        msg = (
            f"Range coordinates count ({range_coords.size}) must match the raster range "
            f"dimension ({raster_data.shape[1]})."
        )
        raise ValueError(msg)

    return SARDataModel(
        raster_data,
        dims=(AZIMUTH_DIM, RANGE_DIM),
        coords={AZIMUTH_DIM: azimuth_coords, RANGE_DIM: range_coords},
    )


def validate_sar_data_model(data: xr.DataArray) -> None:
    """Validate the SAR data model structure.

    Checks that the data has the ``azimuth`` and ``range`` dimensions, with an
    object-dtype azimuth coordinate (e.g. PreciseDateTime) and a floating-dtype
    range coordinate. Only the coordinate dtypes are checked, not their values.

    Parameters
    ----------
    data : xr.DataArray
        data to validate

    Raises
    ------
    ValueError
        if the data does not match the SARDataModel structure

    """
    if data.dims != (AZIMUTH_DIM, RANGE_DIM):
        msg = f"Data must have dimensions ({AZIMUTH_DIM}, {RANGE_DIM}), got {data.dims}."
        raise ValueError(msg)
    if AZIMUTH_DIM not in data.coords:
        msg = f"Missing {AZIMUTH_DIM} coordinate."
        raise ValueError(msg)
    if data[AZIMUTH_DIM].dtype != object:
        msg = f"Azimuth coordinate must have object dtype, got {data[AZIMUTH_DIM].dtype}."
        raise ValueError(msg)
    if RANGE_DIM not in data.coords:
        msg = f"Missing {RANGE_DIM} coordinate."
        raise ValueError(msg)
    if not np.issubdtype(data[RANGE_DIM].dtype, np.floating):
        msg = f"Range coordinate must have floating dtype, got {data[RANGE_DIM].dtype}."
        raise ValueError(msg)


def _interpolate_axis_coords(coords: np.ndarray, positions: npt.NDArray[np.floating]) -> np.ndarray:
    """Interpolate axis coordinates at fractional pixel positions.

    Positions are linearly interpolated between adjacent pixels, so that the
    pixel-to-coordinate linkage is preserved when subsampling with float factors.
    Positions falling exactly on the last pixel are returned unchanged. Object-dtype
    coordinates (e.g. PreciseDateTime) are interpolated through their own arithmetic,
    i.e. ``coords[i] + fraction * (coords[j] - coords[i])``.

    When the axis is non-monotonic (burst overlaps), the interpolated value linearly
    bridges the overlap: this is an approximation, but the positional linkage between
    pixels and coordinates is preserved either way.

    Parameters
    ----------
    coords : np.ndarray
        axis coordinates, one per pixel
    positions : npt.NDArray[np.floating]
        fractional pixel positions within ``[0, coords.size - 1]``

    Returns
    -------
    np.ndarray
        coordinates interpolated at the input positions

    """
    positions = np.asarray(positions, dtype=float)
    if coords.size <= 1:
        return coords.copy()
    indexes = np.floor(positions).astype(int)
    fractions = positions - indexes
    next_indexes = np.minimum(indexes + 1, coords.size - 1)
    if coords.dtype == object:
        return np.array(
            [
                coords[i] + fraction * (coords[j] - coords[i])
                for i, j, fraction in zip(indexes, next_indexes, fractions, strict=True)
            ],
            dtype=object,
        )
    return np.interp(positions, np.arange(coords.size), coords)  # pyrefly: ignore [bad-return]
