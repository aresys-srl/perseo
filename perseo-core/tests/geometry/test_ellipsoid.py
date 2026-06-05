# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Testing geometry/ellipsoid.py functionalities"""

import numpy as np
import pytest
from pyproj import Geod

from perseo_core.geometry.ellipsoid import (
    WGS84,
    compute_line_ellipsoid_intersections,
    create_inflated_wgs84_ellipsoid,
)


class TestCreateInflatedWGS84Ellipsoid:
    def test_inflate_positive(self):
        inflated = create_inflated_wgs84_ellipsoid(5)
        assert inflated.a == pytest.approx(WGS84.a + 5)
        assert inflated.b == pytest.approx(WGS84.b + 5)

    def test_inflate_zero(self):
        inflated = create_inflated_wgs84_ellipsoid(0)
        assert inflated.a == pytest.approx(WGS84.a)
        assert inflated.b == pytest.approx(WGS84.b)

    def test_inflate_negative(self):
        inflated = create_inflated_wgs84_ellipsoid(-0.5)
        assert inflated.a == pytest.approx(WGS84.a - 0.5)
        assert inflated.b == pytest.approx(WGS84.b - 0.5)


class TestLineEllipsoidIntersections:
    # Ellipsoid with a=2 (equatorial, x/y) and b=1 (polar, z)
    ellipsoid = Geod(a=2, b=1)

    def test_no_intersections(self):
        line_origin = np.array([-5, 0, 2], dtype=float)
        line_direction = np.array([0, 0, 100], dtype=float)

        first, second = compute_line_ellipsoid_intersections(line_direction, line_origin, self.ellipsoid)

        assert first.shape == (3,)
        assert second.shape == (3,)
        assert np.all(np.isnan(first))
        assert np.all(np.isnan(second))

    def test_one_intersection(self):
        line_origin = np.array([-5, 0, 1], dtype=float)
        line_direction = np.array([100, 0, 0], dtype=float)

        first, second = compute_line_ellipsoid_intersections(line_direction, line_origin, self.ellipsoid)

        np.testing.assert_allclose(first, np.array([0, 0, 1]), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(second, np.array([0, 0, 1]), rtol=1e-10, atol=1e-10)

    def test_two_intersections_x_axis(self):
        line_origin = np.array([-5, 0, 0], dtype=float)
        line_direction = np.array([100, 0, 0], dtype=float)

        first, second = compute_line_ellipsoid_intersections(line_direction, line_origin, self.ellipsoid)

        assert np.linalg.norm(first - line_origin) < np.linalg.norm(second - line_origin)
        np.testing.assert_allclose(first, np.array([-2, 0, 0]), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(second, np.array([2, 0, 0]), rtol=1e-10, atol=1e-10)

    def test_two_intersections_y_axis(self):
        line_origin = np.array([0, 2, 0], dtype=float)
        line_direction = np.array([0, 1, 0], dtype=float)

        first, second = compute_line_ellipsoid_intersections(line_direction, line_origin, self.ellipsoid)

        assert np.linalg.norm(first - line_origin) < np.linalg.norm(second - line_origin)
        np.testing.assert_allclose(first, np.array([0, 2, 0]), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(second, np.array([0, -2, 0]), rtol=1e-10, atol=1e-10)

    def test_two_intersections_z_axis_vectorized_single(self):
        line_origin = np.array([[0, 0, 10]], dtype=float)
        line_direction = np.array([[0, 0, 5]], dtype=float)

        first, second = compute_line_ellipsoid_intersections(line_direction, line_origin, self.ellipsoid)

        assert first.shape == (1, 3)
        assert second.shape == (1, 3)
        assert np.linalg.norm(first[0] - line_origin[0]) < np.linalg.norm(second[0] - line_origin[0])
        np.testing.assert_allclose(first[0], np.array([0, 0, 1]), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(second[0], np.array([0, 0, -1]), rtol=1e-10, atol=1e-10)

    def test_vectorized(self):
        line_origins = np.array([[-5, 0, 2], [-5, 0, 1], [-5, 0, 0]])
        line_directions = np.array([[0, 0, 100], [100, 0, 0], [100, 0, 0]])

        first, second = compute_line_ellipsoid_intersections(line_directions, line_origins, self.ellipsoid)

        assert first.shape == (3, 3)
        assert second.shape == (3, 3)

        # first line: no intersection → both NaN
        assert np.all(np.isnan(first[0]))
        assert np.all(np.isnan(second[0]))

        # second line: tangent → first and second are the same point
        np.testing.assert_allclose(first[1], np.array([0, 0, 1]), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(second[1], np.array([0, 0, 1]), rtol=1e-10, atol=1e-10)

        # third line: two distinct intersections
        assert np.linalg.norm(first[2] - line_origins[2]) < np.linalg.norm(second[2] - line_origins[2])
        np.testing.assert_allclose(first[2], np.array([-2, 0, 0]), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(second[2], np.array([2, 0, 0]), rtol=1e-10, atol=1e-10)

    def test_invalid_input_origin_wrong_shape(self):
        line_origin = np.array([-5, 0], dtype=float)
        line_direction = np.array([0, 0, 100], dtype=float)

        with pytest.raises(ValueError):
            compute_line_ellipsoid_intersections(line_direction, line_origin, self.ellipsoid)

    def test_invalid_input_direction_wrong_shape(self):
        line_origin = np.array([-5, 0, 2], dtype=float)
        line_direction = np.array([0, 0, 100, 0], dtype=float)

        with pytest.raises(ValueError):
            compute_line_ellipsoid_intersections(line_direction, line_origin, self.ellipsoid)


class TestWGS84:
    def test_semi_major_axis(self):
        assert WGS84.a == pytest.approx(6378137.0)

    def test_semi_minor_axis(self):
        assert WGS84.b == pytest.approx(6356752.314245)
