# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for timing.conversion module"""

import unittest
from datetime import datetime

import numpy as np

import perseo_core.timing.conversions as conv
from perseo_core.timing.precise_datetime import PreciseDateTime


class GPSWeekConversionTest(unittest.TestCase):
    """Testing date_to_gps_week function"""

    def setUp(self) -> None:
        self.input_date = PreciseDateTime.from_numeric_datetime(2012, 6, 15, 17, 30)
        self.input_date_2 = datetime(1979, 6, 15, 17, 30)
        self.ref_results = (1692, 5)

    def test_date_to_gps_week_conversion(self) -> None:
        """Testing date_to_gps_week conversion function"""
        gps_week, day_of_week = conv.date_to_gps_week(self.input_date)

        self.assertIsInstance(gps_week, int)
        self.assertIsInstance(day_of_week, int)
        self.assertEqual(gps_week, self.ref_results[0])
        self.assertEqual(day_of_week, self.ref_results[1])

    def test_date_to_gps_week_conversion_with_error(self) -> None:
        """Testing date_to_gps_week conversion function, with error"""
        with self.assertRaises(ValueError):
            conv.date_to_gps_week(self.input_date_2)


class PreciseDateTimeToNumpyTest(unittest.TestCase):
    """Testing precise_datetime_to_numpy function"""

    def setUp(self) -> None:
        self.input_date = PreciseDateTime.from_numeric_datetime(2012, 6, 15, 17, 30)
        self.time_deltas = [115e-12, 115e-9, 115.000854]
        self.ref_results = [
            np.datetime64("2012-06-15T17:30:00.000000000"),
            np.datetime64("2012-06-15T17:30:00.000000115"),
            np.datetime64("2012-06-15T17:31:55.000854000"),
        ]

    def test_single_time_conversion(self) -> None:
        """Testing single PreciseDateTime conversion"""
        conv_time = conv.precise_datetime_to_numpy(times=self.input_date + self.time_deltas[0])
        np.testing.assert_equal(float(conv_time - self.ref_results[0]), 0.0)

    def test_multiple_times_conversion(self) -> None:
        """Testing vectorized PreciseDateTime conversion"""
        conv_times = conv.precise_datetime_to_numpy(times=[self.input_date + t for t in self.time_deltas])
        np.testing.assert_equal(
            (conv_times - self.ref_results).astype(float), np.zeros_like(self.ref_results, dtype=float)
        )


if __name__ == "__main__":
    unittest.main()
