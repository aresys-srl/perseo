# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Ellipsoid models and utilities for geodetic computations."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from pyproj import Geod

WGS84 = Geod(ellps="WGS84")


def create_inflated_wgs84_ellipsoid(height: float) -> Geod:
    """Create an inflated WGS84 ellipsoid.

    Parameters
    ----------
    height : float
        height above WGS84 ellipsoid

    Returns
    -------
    Geod
        inflated WGS84-like ellipsoid

    """
    return Geod(a=WGS84.a + height, b=WGS84.b + height)


def compute_line_ellipsoid_intersections(
    line_directions: npt.NDArray[np.floating],
    line_origins: npt.NDArray[np.floating],
    ellipsoid: Geod,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Compute the intersections between lines and an ellipsoid.

    For each line, returns two intersection arrays. When a line does not intersect the ellipsoid,
    both arrays contain NaN. When a line is tangent, both arrays contain the same point.
    The first array always contains the intersection closest to line_origin.

    Parameters
    ----------
    line_directions : npt.NDArray[np.floating]
        (3,), (N, 3) one or more line directions, not necessarily normalized
    line_origins : npt.NDArray[np.floating]
        (3,), (N, 3) one or more line origins
    ellipsoid : Geod
        ellipsoid

    Returns
    -------
    npt.NDArray[np.floating]
        closest intersections to line_origins, with shape (3,) for a single line or (N, 3) for multiple lines, np.nan
        is used as a placeholder where no intersection exists
    npt.NDArray[np.floating]
        farthest intersections to line_origins, with shape (3,) for a single line or (N, 3) for multiple lines, np.nan
        is used as a placeholder where no intersection exists

    Raises
    ------
    ValueError
        In case of invalid input

    Examples
    --------
    no intersection

    >>> first, second = compute_line_ellipsoid_intersections([0, 0, 100], [-5, 0, 2], Geod(a=2, b=1))
    >>> print(first, second)
    [nan nan nan] [nan nan nan]

    tangent intersection (first == second)

    >>> first, second = compute_line_ellipsoid_intersections([100, 0, 0], [-5, 0, 1], Geod(a=2, b=1))
    >>> print(first, second)
    [0. 0. 1.] [0. 0. 1.]

    two intersections, first one is closer to line_origin

    >>> first, second = compute_line_ellipsoid_intersections([100, 0, 0], [-5, 0, 0], Geod(a=2, b=1))
    >>> print(first, second)
    [-2.  0.  0.] [2. 0. 0.]

    multiple lines

    >>> line_origins = np.array([[-5, 0, 2], [-5, 0, 1], [-5, 0, 0]])
    >>> line_directions = np.array([[0, 0, 100], [100, 0, 0], [100, 0, 0]])
    >>> first, second = compute_line_ellipsoid_intersections(line_directions, line_origins, Geod(a=2, b=1))
    >>> print(first)
    [[nan nan nan]
     [ 0.  0.  1.]
     [-2.  0.  0.]]
    >>> print(second)
    [[nan nan nan]
     [ 0.  0.  1.]
     [ 2.  0.  0.]]

    extract validity mask for multiple lines

    >>> line_origins = np.array([[-5, 0, 2], [-5, 0, 1], [-5, 0, 0]])
    >>> line_directions = np.array([[0, 0, 100], [100, 0, 0], [100, 0, 0]])
    >>> first, second = compute_line_ellipsoid_intersections(line_directions, line_origins, Geod(a=2, b=1))
    >>> valid_mask = ~np.isnan(first[:, 0])
    >>> print(first[valid_mask], second[valid_mask])
    [[ 0.  0.  1.]
     [-2.  0.  0.]] [[0. 0. 1.]
     [2. 0. 0.]]

    """
    line_directions = np.asarray(line_directions)
    line_origins = np.asarray(line_origins)

    ndim = max(line_origins.ndim, line_directions.ndim)

    if line_directions.shape[-1] != 3:
        msg = f"Invalid line_direction shape: {line_directions.shape} should be (3,) or (N,3)"
        raise ValueError(msg)

    if line_origins.shape[-1] != 3:
        msg = f"Invalid line_origin shape: {line_origins.shape} should be (3,) or (N,3)"
        raise ValueError(msg)

    line_directions = line_directions / np.linalg.norm(line_directions, axis=-1, keepdims=True)

    # line: x = line_origin + t times line_direction
    # equation: t ** 2 + b t + c  = 0

    assert isinstance(line_directions, np.ndarray)
    line_directions_scaled = np.stack(
        [
            line_directions[..., 0] / ellipsoid.a,
            line_directions[..., 1] / ellipsoid.a,
            line_directions[..., 2] / ellipsoid.b,
        ],
        axis=-1,
    )

    line_origins_scaled = np.stack(
        [
            line_origins[..., 0] / ellipsoid.a,
            line_origins[..., 1] / ellipsoid.a,
            line_origins[..., 2] / ellipsoid.b,
        ],
        axis=-1,
    )

    num_lines = max(line_directions.size // 3, line_origins.size // 3)

    quadratic_terms = np.sum(line_directions_scaled * line_directions_scaled, axis=-1)
    linear_terms = 2 * np.sum(line_directions_scaled * line_origins_scaled, axis=-1)
    constant_terms = np.sum(line_origins_scaled * line_origins_scaled, axis=-1) - 1

    quadratic_terms = np.broadcast_to(quadratic_terms, (num_lines,))
    linear_terms = np.broadcast_to(linear_terms, (num_lines,))
    constant_terms = np.broadcast_to(constant_terms, (num_lines,))

    # Normalize the quadratic so the leading coefficient is 1, then use the half-b form:
    #   t^2 + 2*b*t + c = 0  with  b = linear_terms / (2 * quadratic_terms),  c = constant_terms / quadratic_terms
    # Discriminant is d = b^2 - c
    # Stable root (larger |t|, avoids cancellation): r_stable = -b - sign(b)*sqrt(d)  (sign chosen so terms add)
    # Other root via Vieta's formula (smaller |t|, closest to line_origin): r_other = c / r_stable
    # Special case b=0: r_stable = sqrt(d), r_other = -sqrt(d)  (equal absolute values)
    b = linear_terms / (2 * quadratic_terms)
    c_norm = constant_terms / quadratic_terms

    d = b * b - c_norm
    sqrt_d = np.sqrt(np.maximum(d, 0))

    r_stable = -b - np.copysign(sqrt_d, b)
    r_other = np.where(r_stable != 0, c_norm / r_stable, -sqrt_d)

    no_intersection = d < 0
    r_first = np.where(no_intersection, np.nan, r_other)
    r_second = np.where(no_intersection, np.nan, r_stable)

    line_origins = np.broadcast_to(line_origins, (num_lines, 3))
    line_directions = np.broadcast_to(line_directions, (num_lines, 3))

    first_intersections = line_origins + r_first[:, np.newaxis] * line_directions
    second_intersections = line_origins + r_second[:, np.newaxis] * line_directions

    if ndim == 1:
        return first_intersections[0], second_intersections[0]
    return first_intersections, second_intersections


__all__ = ["WGS84", "compute_line_ellipsoid_intersections", "create_inflated_wgs84_ellipsoid"]
