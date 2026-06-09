# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for io/protocol_utilities.py"""

from __future__ import annotations

import pytest

from perseo_quality.core.custom_errors import AzimuthExceedsBoundariesError, RangeExceedsBoundariesError
from perseo_quality.io.protocol_utilities import roi_validation


class TestROIValidation:
    """Testing roi_validation function"""

    def test_valid_roi_no_burst(self):
        """Testing valid ROI without burst boundaries"""
        roi_validation(roi=[0, 0, 100, 200], raster_boundaries=[0, 0, 500, 1000])

    def test_valid_roi_with_burst(self):
        """Testing valid ROI with burst boundaries"""
        roi_validation(
            roi=[50, 0, 100, 200],
            raster_boundaries=[0, 0, 500, 1000],
            burst_boundaries=[50, 0, 200, 1000],
        )

    def test_start_azimuth_exceeds_raster(self):
        """Testing start azimuth >= raster azimuth lines raises error"""
        with pytest.raises(AzimuthExceedsBoundariesError, match="exceeds azimuth swath boundaries"):
            roi_validation(roi=[500, 0, 100, 200], raster_boundaries=[0, 0, 500, 1000])

    def test_start_azimuth_negative(self):
        """Testing negative start azimuth raises error"""
        with pytest.raises(AzimuthExceedsBoundariesError, match="exceeds azimuth swath boundaries"):
            roi_validation(roi=[-1, 0, 100, 200], raster_boundaries=[0, 0, 500, 1000])

    def test_end_azimuth_exceeds_raster(self):
        """Testing end azimuth > raster azimuth lines raises error"""
        with pytest.raises(AzimuthExceedsBoundariesError, match="exceeds azimuth swath boundaries"):
            roi_validation(roi=[400, 0, 200, 200], raster_boundaries=[0, 0, 500, 1000])

    def test_start_range_exceeds_raster(self):
        """Testing start range >= raster range samples raises error"""
        with pytest.raises(RangeExceedsBoundariesError, match="exceeds range swath boundaries"):
            roi_validation(roi=[0, 1000, 100, 200], raster_boundaries=[0, 0, 500, 1000])

    def test_start_range_negative(self):
        """Testing negative start range raises error"""
        with pytest.raises(RangeExceedsBoundariesError, match="exceeds range swath boundaries"):
            roi_validation(roi=[0, -1, 100, 200], raster_boundaries=[0, 0, 500, 1000])

    def test_end_range_exceeds_raster(self):
        """Testing end range > raster range samples raises error"""
        with pytest.raises(RangeExceedsBoundariesError, match="exceeds range swath boundaries"):
            roi_validation(roi=[0, 800, 100, 300], raster_boundaries=[0, 0, 500, 1000])

    def test_burst_start_azimuth_exceeds(self):
        """Testing start azimuth before burst start raises error"""
        with pytest.raises(AzimuthExceedsBoundariesError, match="exceeds azimuth burst boundaries"):
            roi_validation(
                roi=[30, 0, 100, 200],
                raster_boundaries=[0, 0, 500, 1000],
                burst_boundaries=[50, 0, 200, 1000],
            )

    def test_burst_end_azimuth_exceeds(self):
        """Testing end azimuth after burst end raises error"""
        with pytest.raises(AzimuthExceedsBoundariesError, match="exceeds azimuth burst boundaries"):
            roi_validation(
                roi=[50, 0, 250, 200],
                raster_boundaries=[0, 0, 500, 1000],
                burst_boundaries=[50, 0, 200, 1000],
            )
