# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Geographical orientation of quick-look data based on corner coordinates."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from perseo_core.geometry.coordinates.ellipsoid import WGS84
from perseo_core.imaging.models import (
    AZIMUTH_DIM,
    RANGE_DIM,
    SARDataModel,
    validate_sar_data_model,
)

__all__ = ["orient_geographically"]

# corner indexes in the documented [top-left, top-right, bottom-right, bottom-left] order
_TOP_LEFT, _TOP_RIGHT, _BOTTOM_RIGHT, _BOTTOM_LEFT = 0, 1, 2, 3


def orient_geographically(data: SARDataModel, corners: npt.NDArray[np.floating]) -> SARDataModel:
    """Orient a SAR data model along cardinal points using its corner coordinates.

    The raster keeps its original azimuth and range axes, oriented along the orbit
    heading; only the flips needed to align the axes with the cardinal points are
    applied, so that the image remains oriented as the orbit heading. The actual
    geographical display is delegated to georeferencing tools (e.g. KML/KMZ
    viewers) using the corner coordinates.

    Parameters
    ----------
    data : SARDataModel
        SAR data model with ``azimuth`` and ``range`` dimensions
    corners : npt.NDArray[np.floating]
        (lat, lon) [deg] pairs of the pixels at image image order positions
        ``[top-left, top-right, bottom-right, bottom-left]``

    Returns
    -------
    SARDataModel
        data with the required flips applied to both values and axis coordinates

    Raises
    ------
    ValueError
        if the data does not match the SARDataModel structure or the corners
        are not a (4, 2) array of valid (lat, lon) pairs

    """
    validate_sar_data_model(data)
    corners = np.asarray(corners, dtype=float)
    _validate_corners(corners)

    flip_azimuth, flip_range = _required_flips(corners)
    if flip_azimuth:
        data = data.isel({AZIMUTH_DIM: slice(None, None, -1)})
    if flip_range:
        data = data.isel({RANGE_DIM: slice(None, None, -1)})
    return data


def _validate_corners(corners: npt.NDArray[np.floating]) -> None:
    """Validate the corner coordinates."""
    if corners.shape != (4, 2):
        msg = f"Corners must be a (4, 2) array of (lat, lon) pairs, got shape {corners.shape}."
        raise ValueError(msg)
    latitudes, longitudes = corners[:, 0], corners[:, 1]
    if not np.all((latitudes >= -90.0) & (latitudes <= 90.0)):
        msg = f"Latitudes must be within [-90, 90], got {latitudes}."
        raise ValueError(msg)
    if not np.all((longitudes >= -180.0) & (longitudes <= 180.0)):
        msg = f"Longitudes must be within [-180, 180], got {longitudes}."
        raise ValueError(msg)


def _required_flips(corners: npt.NDArray[np.floating]) -> tuple[bool, bool]:
    """Derive the flips needed to align the raster axes with the cardinal points.

    The image ``up`` direction is the average azimuth of the two vertical edges, while
    the image ``right`` direction is the average azimuth of the two horizontal edges.
    A flip is needed when the up direction has a southward component (negative north
    component) or the right direction has a westward component (negative east component).
    """
    top_left = corners[_TOP_LEFT]
    top_right = corners[_TOP_RIGHT]
    bottom_left = corners[_BOTTOM_LEFT]
    bottom_right = corners[_BOTTOM_RIGHT]

    up_azimuth = _vector_mean_azimuth((bottom_left, top_left), (bottom_right, top_right))
    right_azimuth = _vector_mean_azimuth((top_left, top_right), (bottom_left, bottom_right))

    flip_azimuth = np.cos(np.deg2rad(up_azimuth)) < 0.0
    flip_range = np.sin(np.deg2rad(right_azimuth)) < 0.0
    return flip_azimuth, flip_range


def _vector_mean_azimuth(
    *segments: tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]],
) -> float:
    """Average the forward azimuths of the input segments, robust to the 0/360 wrap-around."""
    sin_sum = 0.0
    cos_sum = 0.0
    for start, end in segments:
        # Geod.inv takes (lon, lat) and returns the forward azimuth in degrees,
        # measured clockwise from north
        azimuth = WGS84.inv(start[1], start[0], end[1], end[0])[0]
        sin_sum += np.sin(np.deg2rad(azimuth))
        cos_sum += np.cos(np.deg2rad(azimuth))
    return np.rad2deg(np.arctan2(sin_sum, cos_sum)) % 360.0
