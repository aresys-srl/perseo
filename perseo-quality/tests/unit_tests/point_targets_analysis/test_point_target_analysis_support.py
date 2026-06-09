# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for point_target_analysis/support.py core functionalities"""

from __future__ import annotations

import numpy as np
import pytest

import perseo_quality.point_targets_analysis.support as support


class TestGetSquintAngle:
    """Testing point_target_analysis/support.py get_squint_angle function"""

    @pytest.fixture(autouse=True)
    def _setup(self, mock_channel_data, ref_time, ref_ground_point) -> None:
        self.channel_data = mock_channel_data
        self.ref_time = ref_time
        self.ref_ground_point = ref_ground_point
        self.tolerance = 1e-9
        self.expected_res = -0.5964450004813256

    def test_get_squint_angle(self):
        """Testing get_squint_angle function"""
        squint = support.get_squint_angle(
            channel_data=self.channel_data, azimuth_time=self.ref_time, ground_point=self.ref_ground_point
        )
        np.testing.assert_allclose(squint, self.expected_res, atol=self.tolerance, rtol=0)

    def test_get_squint_angle_vectorized_0(self):
        """Testing get_squint_angle function, vectorized ground points"""
        squint = support.get_squint_angle(
            channel_data=self.channel_data,
            azimuth_time=self.ref_time,
            ground_point=np.tile(self.ref_ground_point, 4).reshape(-1, 3),
        )
        np.testing.assert_allclose(squint, np.repeat(self.expected_res, 4), atol=self.tolerance, rtol=0)

    def test_get_squint_angle_vectorized_1(self):
        """Testing get_squint_angle function, vectorized times"""
        squint = support.get_squint_angle(
            channel_data=self.channel_data,
            azimuth_time=np.array([self.ref_time, self.ref_time]),
            ground_point=self.ref_ground_point,
        )
        np.testing.assert_allclose(squint, np.repeat(self.expected_res, 2), atol=self.tolerance, rtol=0)

    def test_get_squint_angle_vectorized_2(self):
        """Testing get_squint_angle function, vectorized all"""
        squint = support.get_squint_angle(
            channel_data=self.channel_data,
            azimuth_time=np.array([self.ref_time, self.ref_time, self.ref_time, self.ref_time]),
            ground_point=np.tile(self.ref_ground_point, 4).reshape(-1, 3),
        )
        np.testing.assert_allclose(squint, np.repeat(self.expected_res, 4), atol=self.tolerance, rtol=0)


class TestGetDopplerCentroid:
    """Testing point_target_analysis/support.py get_doppler_centroid function"""

    @pytest.fixture(autouse=True)
    def _setup(self, mock_channel_data, ref_time, ref_ground_point) -> None:
        self.channel_data = mock_channel_data
        self.ref_time = ref_time
        self.ref_ground_point = ref_ground_point
        self.tolerance = 1e-9
        self.expected_res = -153549.19005492592

    def test_get_doppler_centroid(self):
        """Testing get_doppler_centroid function"""
        dc = support.get_doppler_centroid(
            channel_data=self.channel_data, azimuth_time=self.ref_time, ground_point=self.ref_ground_point
        )
        np.testing.assert_allclose(dc, self.expected_res, atol=self.tolerance, rtol=0)

    def test_get_doppler_centroid_vectorized_0(self):
        """Testing get_doppler_centroid function, vectorized ground points"""
        dc = support.get_doppler_centroid(
            channel_data=self.channel_data,
            azimuth_time=self.ref_time,
            ground_point=np.tile(self.ref_ground_point, 4).reshape(-1, 3),
        )
        np.testing.assert_allclose(dc, np.repeat(self.expected_res, 4), atol=self.tolerance, rtol=0)

    def test_get_doppler_centroid_vectorized_1(self):
        """Testing get_doppler_centroid function, vectorized times"""
        dc = support.get_doppler_centroid(
            channel_data=self.channel_data,
            azimuth_time=np.array([self.ref_time, self.ref_time]),
            ground_point=self.ref_ground_point,
        )
        np.testing.assert_allclose(dc, np.repeat(self.expected_res, 2), atol=self.tolerance, rtol=0)

    def test_get_doppler_centroid_vectorized_2(self):
        """Testing get_doppler_centroid function, vectorized all"""
        dc = support.get_doppler_centroid(
            channel_data=self.channel_data,
            azimuth_time=np.array([self.ref_time, self.ref_time, self.ref_time, self.ref_time]),
            ground_point=np.tile(self.ref_ground_point, 4).reshape(-1, 3),
        )
        np.testing.assert_allclose(dc, np.repeat(self.expected_res, 4), atol=self.tolerance, rtol=0)
