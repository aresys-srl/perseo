# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Ellipsoids"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from pyproj import Geod

WGS84 = Geod(ellps="WGS84")


def create_inflated_WGS84_ellipsoid(height: float) -> Geod:
    """Creating an inflated WGS84 ellipsoid.

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
    line_directions: npt.NDArray[np.floating], line_origins: npt.NDArray[np.floating], ellipsoid: Geod
) -> tuple[tuple[np.ndarray]] | tuple[np.ndarray]:
    """Compute the intersections between lines and an ellipsoid

    For each line it returns the intersections.

    When two intersections are found they are sorted by the closest to line_origin.

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
    tuple[tuple[np.ndarray]] | tuple[np.ndarray]
        The intersections or a tuple of intersections, depending on N.
        Intersections are stored as a tuple of points (np.array (3,)).
        The number of points depends on the intersection results can be 0, 1 or 2.

    Raises
    ------
    ValueError
        In case of invalid input

    Examples
    --------

    empty intersection

    >>> intersections = compute_line_ellipsoid_intersections(
                    [0, 0, 100], [-5, 0, 2], Ellipsoid(1, 2)
                )
    >>> print(intersections)
    ()

    single intersection

    >>> intersections = compute_line_ellipsoid_intersections(
                    [100, 0, 0], [-5, 0, 1], Ellipsoid(1, 2)
                )
    >>> print(intersections)
    (array([-8.8817842e-16,  0.0000000e+00,  1.0000000e+00]),)

    two intersections, first one is closer to line_origin

    >>> intersections = compute_line_ellipsoid_intersections(
                    [100, 0, 0], [-5, 0, 0], Ellipsoid(1, 2)
                )
    >>> print(intersections)
    (array([-2.,  0.,  0.]), array([2., 0., 0.]))

    multiple lines

    >>> line_origins = np.array([[-5, 0, 2], [-5, 0, 1], [-5, 0, 0]])
    >>> line_directions = np.array([[0, 0, 100], [100, 0, 0], [100, 0, 0]])
    >>> intersections = compute_line_ellipsoid_intersections(
                line_directions, line_origins, Ellipsoid(1, 2)
            )
    >>> print(intersections)
    ((), (array([-8.8817842e-16,  0.0000000e+00,  1.0000000e+00]),),
         (array([-2.,  0.,  0.]), array([2., 0., 0.])))
    """

    # TODO: return always two solutions or None and check where this is used to update code accordingly

    line_directions = np.asarray(line_directions)
    line_origins = np.asarray(line_origins)

    ndim = max(line_origins.ndim, line_directions.ndim)

    if line_directions.shape[-1] != 3:
        raise ValueError(f"Invalid line_direction shape: {line_directions.shape} should be (3,) or (N,3)")

    if line_origins.shape[-1] != 3:
        raise ValueError(f"Invalid line_origin shape: {line_origins.shape} should be (3,) or (N,3)")

    line_directions = line_directions / np.linalg.norm(line_directions, axis=-1, keepdims=True)

    # line: x = line_origin + t * line_direction
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

    polynomials = (
        np.polynomial.Polynomial(coeffs) for coeffs in zip(constant_terms, linear_terms, quadratic_terms, strict=False)
    )

    line_origins = np.broadcast_to(line_origins, (num_lines, 3))
    line_directions = np.broadcast_to(line_directions, (num_lines, 3))

    def solve_equation(poly: np.polynomial.Polynomial) -> tuple:
        return tuple(sorted(np.unique([root for root in poly.roots() if np.isreal(root)]), key=abs))

    assert isinstance(line_directions, np.ndarray)
    assert isinstance(line_origins, np.ndarray)
    intersections_tuple = tuple(
        tuple(origin + root * direction for root in solve_equation(poly))
        for poly, origin, direction in zip(polynomials, line_origins, line_directions, strict=False)
    )

    return intersections_tuple[0] if ndim == 1 else intersections_tuple


__all__ = ["WGS84", "create_inflated_WGS84_ellipsoid", "compute_line_ellipsoid_intersections"]
