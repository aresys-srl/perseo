# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for timing.conversions module"""

import numpy as np
import pytest

from perseo_core.timing import conversions


class TestGPSWeekConversion:
    """Test date_to_gps_week conversion function with valid and invalid inputs."""

    @pytest.fixture(autouse=True)
    def setup_gps_week_data(self, gps_week_conversion_test_data: dict) -> None:
        """Load test data from fixtures."""
        self.input_date = gps_week_conversion_test_data["input_date"]
        self.input_date_2 = gps_week_conversion_test_data["input_date_2"]
        self.ref_results = gps_week_conversion_test_data["ref_results"]

    def test_date_to_gps_week_conversion(self) -> None:
        """Test that date_to_gps_week returns correct GPS week and day of week."""
        gps_week, day_of_week = conversions.date_to_gps_week(self.input_date)

        assert isinstance(gps_week, int)
        assert isinstance(day_of_week, int)
        assert gps_week == self.ref_results[0]
        assert day_of_week == self.ref_results[1]

    def test_date_to_gps_week_conversion_with_error(self) -> None:
        """Test that date_to_gps_week raises ValueError for dates outside valid GPS epoch range."""
        with pytest.raises(ValueError, match="cannot be before"):
            conversions.date_to_gps_week(self.input_date_2)


class TestPreciseDateTimeToNumpy:
    """Test precise_datetime_to_numpy conversion with scalar and vectorized inputs."""

    @pytest.fixture(autouse=True)
    def setup_numpy_conversion_data(self, precise_datetime_to_numpy_test_data: dict) -> None:
        """Load test data from fixtures."""
        self.input_date = precise_datetime_to_numpy_test_data["input_date"]
        self.time_deltas = precise_datetime_to_numpy_test_data["time_deltas"]
        self.ref_results = precise_datetime_to_numpy_test_data["ref_results"]

    def test_single_time_conversion(self) -> None:
        """Test that precise_datetime_to_numpy correctly converts single PreciseDateTime to numpy datetime64."""
        conv_time = conversions.precise_datetime_to_numpy(times=self.input_date + self.time_deltas[0])
        np.testing.assert_equal(float(conv_time - self.ref_results[0]), 0.0)

    def test_multiple_times_conversion(self) -> None:
        """Test that precise_datetime_to_numpy correctly converts list of PreciseDateTime objects."""
        conv_times = conversions.precise_datetime_to_numpy(times=[self.input_date + t for t in self.time_deltas])
        np.testing.assert_equal(
            (conv_times - self.ref_results).astype(float), np.zeros_like(self.ref_results, dtype=float)
        )
