# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for timing.conversion module"""

import unittest

import numpy as np

import perseo_core.timing.conversions as conv
from tests.fixtures.timing_data import (
    get_gps_week_conversion_test_data,
    get_precise_datetime_to_numpy_test_data,
)


class GPSWeekConversionTest(unittest.TestCase):
    """Test date_to_gps_week conversion function with valid and invalid inputs."""

    def setUp(self) -> None:
        # Load test data from fixtures
        data = get_gps_week_conversion_test_data()
        self.input_date = data["input_date"]
        self.input_date_2 = data["input_date_2"]
        self.ref_results = data["ref_results"]

    def test_date_to_gps_week_conversion(self) -> None:
        """Test that date_to_gps_week returns correct GPS week and day of week."""
        gps_week, day_of_week = conv.date_to_gps_week(self.input_date)

        self.assertIsInstance(gps_week, int)
        self.assertIsInstance(day_of_week, int)
        self.assertEqual(gps_week, self.ref_results[0])
        self.assertEqual(day_of_week, self.ref_results[1])

    def test_date_to_gps_week_conversion_with_error(self) -> None:
        """Test that date_to_gps_week raises ValueError for dates outside valid GPS epoch range."""
        with self.assertRaises(ValueError):
            conv.date_to_gps_week(self.input_date_2)


class PreciseDateTimeToNumpyTest(unittest.TestCase):
    """Test precise_datetime_to_numpy conversion with scalar and vectorized inputs."""

    def setUp(self) -> None:
        # Load test data from fixtures
        data = get_precise_datetime_to_numpy_test_data()
        self.input_date = data["input_date"]
        self.time_deltas = data["time_deltas"]
        self.ref_results = data["ref_results"]

    def test_single_time_conversion(self) -> None:
        """Test that precise_datetime_to_numpy correctly converts single PreciseDateTime to numpy datetime64."""
        conv_time = conv.precise_datetime_to_numpy(times=self.input_date + self.time_deltas[0])
        np.testing.assert_equal(float(conv_time - self.ref_results[0]), 0.0)

    def test_multiple_times_conversion(self) -> None:
        """Test that precise_datetime_to_numpy correctly converts list of PreciseDateTime objects."""
        conv_times = conv.precise_datetime_to_numpy(times=[self.input_date + t for t in self.time_deltas])
        np.testing.assert_equal(
            (conv_times - self.ref_results).astype(float), np.zeros_like(self.ref_results, dtype=float)
        )


if __name__ == "__main__":
    unittest.main()
