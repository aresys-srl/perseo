# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for timing.precise_datetime module"""

import unittest
from dataclasses import dataclass

import numpy as np

from perseo_core.timing.precise_datetime import PreciseDateTime


class PreciseDateTimeTestCase(unittest.TestCase):
    """Test PreciseDateTime initialization, properties, and conversion methods."""

    def test_init_invalid_seconds(self):
        """Test that PreciseDateTime raises ValueError with negative seconds."""
        with self.assertRaises(ValueError):
            PreciseDateTime(-1)

    def test_init_invalid_picoseconds(self):
        """Test that PreciseDateTime raises ValueError with negative picoseconds."""
        with self.assertRaises(ValueError):
            PreciseDateTime(0, -1)

    def test_properties_accessor(self):
        """Test that PreciseDateTime properties correctly return date/time components."""
        date = PreciseDateTime.from_numeric_datetime(2021, 7, 29, 14, 6, 12, 113_324)

        self.assertEqual(date.year, 2021)
        self.assertEqual(date.month, 7)
        self.assertEqual(date.day_of_the_month, 29)
        self.assertEqual(date.hour_of_day, 14)
        self.assertEqual(date.minute_of_hour, 6)
        self.assertEqual(date.second_of_minute, 12)
        self.assertEqual(date.picosecond_of_second, 113_324)

        self.assertEqual(date.day_of_the_year, 210)
        self.assertAlmostEqual(date.fraction_of_day, 0.5876388889)

    def test_fromisoformat_isoformat(self):
        """Test that fromisoformat and isoformat are inverse operations."""
        date = PreciseDateTime.from_numeric_datetime(1989, 11, 18, 23, 51, 20)
        self.assertEqual(date, PreciseDateTime.fromisoformat(date.isoformat()))

    def test_isoformat_timespec_auto(self):
        """Test that isoformat automatically chooses appropriate precision."""
        date = PreciseDateTime.from_numeric_datetime(1989, 11, 18, 23, 51, 20)
        self.assertEqual(date.isoformat(), "1989-11-18T23:51:20Z")
        self.assertEqual(date.isoformat(timespec="picoseconds"), "1989-11-18T23:51:20.000000000000Z")

        date = PreciseDateTime.from_numeric_datetime(1989, 11, 18, 23, 51, 20, 12)
        self.assertEqual(date.isoformat(), "1989-11-18T23:51:20.000000000012Z")

    def test_isoformat_timespec(self):
        """Test that isoformat respects specified timespec precision levels."""
        date = PreciseDateTime.from_numeric_datetime(1989, 11, 18, 23, 51, 20, 123456789012)
        self.assertEqual(date.isoformat(), "1989-11-18T23:51:20.123456789012Z")
        self.assertEqual(date.isoformat(timespec="nanoseconds"), "1989-11-18T23:51:20.123456789Z")
        self.assertEqual(date.isoformat(timespec="microseconds"), "1989-11-18T23:51:20.123456Z")
        self.assertEqual(date.isoformat(timespec="milliseconds"), "1989-11-18T23:51:20.123Z")
        self.assertEqual(date.isoformat(timespec="seconds"), "1989-11-18T23:51:20Z")
        self.assertEqual(date.isoformat(timespec="minutes"), "1989-11-18T23:51Z")
        self.assertEqual(date.isoformat(timespec="hours"), "1989-11-18T23Z")

    def test_isoformat_invalid_timespec(self):
        """Test that isoformat raises ValueError for invalid timespec values."""
        date = PreciseDateTime.from_numeric_datetime(1989, 11, 18, 23, 51, 20, 123456789012)
        with self.assertRaises(ValueError):
            date.isoformat(timespec="days")

    def test_isoformat_change_separator(self):
        """Test that isoformat respects custom date/time separator."""
        date = PreciseDateTime.from_numeric_datetime(1989, 11, 18, 23, 51, 20)
        self.assertEqual(date.isoformat(sep=" "), "1989-11-18 23:51:20Z")

    def test_fromisoformat(self):
        """Test that fromisoformat correctly parses ISO format strings with varying precision."""
        self.assertEqual(
            PreciseDateTime.fromisoformat("1989"),
            PreciseDateTime.from_numeric_datetime(1989),
        )

        self.assertEqual(
            PreciseDateTime.fromisoformat("1989-11"),
            PreciseDateTime.from_numeric_datetime(1989, 11),
        )

        self.assertEqual(
            PreciseDateTime.fromisoformat("1989-11-18"),
            PreciseDateTime.from_numeric_datetime(1989, 11, 18),
        )

        self.assertEqual(
            PreciseDateTime.fromisoformat("1989-11-18T23"),
            PreciseDateTime.from_numeric_datetime(1989, 11, 18, 23),
        )

        self.assertEqual(
            PreciseDateTime.fromisoformat("1989-11-18T23Z"),
            PreciseDateTime.from_numeric_datetime(1989, 11, 18, 23),
        )

        self.assertEqual(
            PreciseDateTime.fromisoformat("1989-11-18T23:51"),
            PreciseDateTime.from_numeric_datetime(1989, 11, 18, 23, 51),
        )

        self.assertEqual(
            PreciseDateTime.fromisoformat("1989-11-18T23:51Z"),
            PreciseDateTime.from_numeric_datetime(1989, 11, 18, 23, 51),
        )

        self.assertEqual(
            PreciseDateTime.fromisoformat("1989-11-18T23:51:20"),
            PreciseDateTime.from_numeric_datetime(1989, 11, 18, 23, 51, 20),
        )

        self.assertEqual(
            PreciseDateTime.fromisoformat("1989-11-18T23:51:20Z"),
            PreciseDateTime.from_numeric_datetime(1989, 11, 18, 23, 51, 20),
        )

        self.assertEqual(
            PreciseDateTime.fromisoformat("1989-11-18T23:51:20.123"),
            PreciseDateTime.from_numeric_datetime(1989, 11, 18, 23, 51, 20, 123000000000),
        )

        self.assertEqual(
            PreciseDateTime.fromisoformat("1989-11-18T23:51:20.123Z"),
            PreciseDateTime.from_numeric_datetime(1989, 11, 18, 23, 51, 20, 123000000000),
        )

        self.assertEqual(
            PreciseDateTime.fromisoformat("1989-11-18T23:51:20.123456"),
            PreciseDateTime.from_numeric_datetime(1989, 11, 18, 23, 51, 20, 123456000000),
        )

        self.assertEqual(
            PreciseDateTime.fromisoformat("1989-11-18T23:51:20.123456Z"),
            PreciseDateTime.from_numeric_datetime(1989, 11, 18, 23, 51, 20, 123456000000),
        )

        self.assertEqual(
            PreciseDateTime.fromisoformat("1989-11-18T23:51:20.123456789"),
            PreciseDateTime.from_numeric_datetime(1989, 11, 18, 23, 51, 20, 123456789000),
        )

        self.assertEqual(
            PreciseDateTime.fromisoformat("1989-11-18T23:51:20.123456789Z"),
            PreciseDateTime.from_numeric_datetime(1989, 11, 18, 23, 51, 20, 123456789000),
        )

        self.assertEqual(
            PreciseDateTime.fromisoformat("1989-11-18T23:51:20.123456789012"),
            PreciseDateTime.from_numeric_datetime(1989, 11, 18, 23, 51, 20, 123456789012),
        )

        self.assertEqual(
            PreciseDateTime.fromisoformat("1989-11-18T23:51:20.123456789012Z"),
            PreciseDateTime.from_numeric_datetime(1989, 11, 18, 23, 51, 20, 123456789012),
        )

    def test_fromisoformat_change_separator(self):
        """Test that fromisoformat respects custom date/time separator."""
        self.assertEqual(
            PreciseDateTime.fromisoformat("1989-11-18 23:51:20.123456789012Z", sep=" "),
            PreciseDateTime.from_numeric_datetime(1989, 11, 18, 23, 51, 20, 123456789012),
        )

    def test_fromisoformat_compact(self):
        """Test that fromisoformat accepts compact format without separators."""
        self.assertEqual(
            PreciseDateTime.fromisoformat("19891118"),
            PreciseDateTime.fromisoformat("1989-11-18"),
        )

        self.assertEqual(
            PreciseDateTime.fromisoformat("1989-11-18T2351"),
            PreciseDateTime.fromisoformat("1989-11-18T23:51"),
        )

        self.assertEqual(
            PreciseDateTime.fromisoformat("1989-11-18T235120"),
            PreciseDateTime.fromisoformat("1989-11-18T23:51:20"),
        )

        self.assertEqual(
            PreciseDateTime.fromisoformat("1989-11-18T235120.123"),
            PreciseDateTime.fromisoformat("1989-11-18T23:51:20.123"),
        )

    def test_fromisoformat_invalid_year(self):
        """Test that fromisoformat raises ValueError for year with insufficient digits."""
        with self.assertRaises(ValueError):
            PreciseDateTime.fromisoformat("189")

    def test_fromisoformat_invalid_year_month_sep(self):
        """Test that fromisoformat raises ValueError for invalid year/month separator."""
        with self.assertRaises(ValueError):
            PreciseDateTime.fromisoformat("1989/11")

    def test_fromisoformat_invalid_month(self):
        """Test that fromisoformat raises ValueError for month with insufficient digits."""
        with self.assertRaises(ValueError):
            PreciseDateTime.fromisoformat("1989-1")

    def test_fromisoformat_invalid_year_month_date(self):
        """Test that fromisoformat raises ValueError for compact format with missing separator."""
        with self.assertRaises(ValueError):
            PreciseDateTime.fromisoformat("198911")

    def test_fromisoformat_incomplete_day(self):
        """Test that fromisoformat raises ValueError for day with insufficient digits."""
        with self.assertRaises(ValueError):
            PreciseDateTime.fromisoformat("1989-11-1")

    def test_fromisoformat_invalid_month_day_sep(self):
        with self.assertRaises(ValueError):
            PreciseDateTime.fromisoformat("1989-11 18")

    def test_fromisoformat_unexpected_month_day_sep(self):
        with self.assertRaises(ValueError):
            PreciseDateTime.fromisoformat("198911-18")

    def test_fromisoformat_invalid_hour(self):
        with self.assertRaises(ValueError):
            PreciseDateTime.fromisoformat("1989-11-18T2")

    def test_fromisoformat_invalid_hour_minute_sep(self):
        with self.assertRaises(ValueError):
            PreciseDateTime.fromisoformat("1989-11-18T23.51")

    def test_fromisoformat_invalid_minute(self):
        with self.assertRaises(ValueError):
            PreciseDateTime.fromisoformat("1989-11-18T23:5")

    def test_fromisoformat_invalid_minute_second_sep(self):
        with self.assertRaises(ValueError):
            PreciseDateTime.fromisoformat("1989-11-18T23:51.30")

    def test_fromisoformat_unexpected_minute_second_sep(self):
        with self.assertRaises(ValueError):
            PreciseDateTime.fromisoformat("1989-11-18T2351:30")

    def test_fromisoformat_invalid_second(self):
        with self.assertRaises(ValueError):
            PreciseDateTime.fromisoformat("1989-11-18T23:51:3")

    def test_fromisoformat_invalid_picosecond(self):
        with self.assertRaises(ValueError):
            PreciseDateTime.fromisoformat("1989-11-18T23:51:30.")

    def test_fromisoformat_invalid_separtor(self):
        with self.assertRaises(ValueError):
            PreciseDateTime.fromisoformat("1989-11-18T23:51:30", sep=" ")

    def test_fromisoformat_unsupported_timezone(self):
        with self.assertRaises(ValueError):
            PreciseDateTime.fromisoformat("1989-11-18T23:51:30+01")


@dataclass(frozen=True)
class ArithmeticParams:
    seconds: float
    picoseconds: float
    delta: float
    reference_seconds: float
    reference_picoseconds: float


class PreciseDateTimeArithmeticTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.arithmetic_tests_params_add = [
            ArithmeticParams(
                seconds=100,
                picoseconds=158973000000.0,
                delta=-1.158973,
                reference_seconds=99,
                reference_picoseconds=0,
            ),
            ArithmeticParams(
                seconds=100,
                picoseconds=100,
                delta=1.0 + 5.0e-10,
                reference_seconds=101,
                reference_picoseconds=600,
            ),
            ArithmeticParams(
                seconds=100,
                picoseconds=100,
                delta=-0.8,
                reference_seconds=99,
                reference_picoseconds=200000000100,
            ),
            ArithmeticParams(
                seconds=100,
                picoseconds=100,
                delta=-2.8,
                reference_seconds=97,
                reference_picoseconds=200000000100,
            ),
            ArithmeticParams(
                seconds=0,
                picoseconds=0,
                delta=0.0,
                reference_seconds=0,
                reference_picoseconds=0.0,
            ),
        ]

        self.arithmetic_tests_params_sub = [
            ArithmeticParams(
                seconds=100,
                picoseconds=158973000000.0,
                delta=1.158973,
                reference_seconds=99,
                reference_picoseconds=0,
            ),
            ArithmeticParams(
                seconds=100,
                picoseconds=800,
                delta=1.0 + 5.0e-10,
                reference_seconds=99,
                reference_picoseconds=300,
            ),
            ArithmeticParams(
                seconds=100,
                picoseconds=100,
                delta=0.8,
                reference_seconds=99,
                reference_picoseconds=200000000100,
            ),
            ArithmeticParams(
                seconds=100,
                picoseconds=100,
                delta=-0.8,
                reference_seconds=100,
                reference_picoseconds=800000000100,
            ),
            ArithmeticParams(
                seconds=0,
                picoseconds=0,
                delta=0.0,
                reference_seconds=0,
                reference_picoseconds=0.0,
            ),
        ]
        return super().setUp()

    def test_add(self):
        for param in self.arithmetic_tests_params_add:
            with self.subTest(msg=f"{param}"):
                t0 = PreciseDateTime(seconds=param.seconds, picoseconds=param.picoseconds)
                t = t0 + param.delta

                self.assertEqual(t._seconds, param.reference_seconds)
                self.assertEqual(round(t._picoseconds), param.reference_picoseconds)

    def test_iadd(self):
        for param in self.arithmetic_tests_params_add:
            with self.subTest(msg=f"{param}"):
                t = PreciseDateTime(seconds=param.seconds, picoseconds=param.picoseconds)
                idx_reference = id(t)
                t += param.delta
                idx = id(t)
                self.assertEqual(idx, idx_reference)
                self.assertEqual(t._seconds, param.reference_seconds)
                self.assertEqual(round(t._picoseconds), param.reference_picoseconds)

    def test_sub(self):
        for param in self.arithmetic_tests_params_sub:
            with self.subTest(msg=f"{param}"):
                t0 = PreciseDateTime(seconds=param.seconds, picoseconds=param.picoseconds)
                t = t0 - param.delta

                self.assertEqual(t._seconds, param.reference_seconds)
                self.assertEqual(round(t._picoseconds), param.reference_picoseconds)

    def test_isub(self):
        for param in self.arithmetic_tests_params_sub:
            with self.subTest(msg=f"{param}"):
                t = PreciseDateTime(seconds=param.seconds, picoseconds=param.picoseconds)
                idx_reference = id(t)
                t -= param.delta
                idx = id(t)
                self.assertEqual(idx, idx_reference)
                self.assertEqual(t._seconds, param.reference_seconds)
                self.assertEqual(round(t._picoseconds), param.reference_picoseconds)

    def test_isub_pdt(self):
        t = PreciseDateTime(100, 100)
        b = PreciseDateTime(101, 100)
        b -= t
        self.assertEqual(b, 1.0)

    def test_sub_pdt(self):
        delta = PreciseDateTime(seconds=100, picoseconds=800) - PreciseDateTime(seconds=98, picoseconds=500)
        self.assertEqual(delta, 2.0 + 3.0e-10)

        delta = PreciseDateTime(seconds=100, picoseconds=800) - PreciseDateTime(seconds=98, picoseconds=900)
        self.assertEqual(delta, 2 - 1.0e-10)

    def test_add_array(self):
        t0 = PreciseDateTime(seconds=100, picoseconds=100)
        array = np.ones((5, 2))
        reference = PreciseDateTime(seconds=101, picoseconds=100)

        t_array = t0 + array
        self.assertEqual(t_array.shape, (5, 2))
        for time in np.nditer(t_array, flags=["refs_ok"]):
            self.assertEqual(time, reference)

        t_array = array + t0
        self.assertEqual(t_array.shape, (5, 2))
        for time in np.nditer(t_array, flags=["refs_ok"]):
            self.assertEqual(time, reference)

    def test_sub_array(self):
        t0 = PreciseDateTime(seconds=100, picoseconds=100)
        array = np.ones((5, 2))
        reference = PreciseDateTime(seconds=99, picoseconds=100)

        t_array = t0 - array
        self.assertEqual(t_array.shape, (5, 2))
        for time in np.nditer(t_array, flags=["refs_ok"]):
            self.assertEqual(time, reference)

    def test_add_invalid(self):
        t0 = PreciseDateTime(seconds=100, picoseconds=100)
        invalid_element = [PreciseDateTime(seconds=100, picoseconds=100), "a string"]

        for other in invalid_element:
            with self.subTest(msg=f"{type(other)}"):
                with self.assertRaises(TypeError):
                    t0 + other  # type:ignore

    def test_iadd_invalid(self):
        t0 = PreciseDateTime(seconds=100, picoseconds=100)
        invalid_element = [PreciseDateTime(seconds=100, picoseconds=100), "a string"]

        for other in invalid_element:
            with self.subTest(msg=f"{type(other)}"):
                with self.assertRaises(TypeError):
                    t0 += other

    def test_sub_invalid(self):
        t0 = PreciseDateTime(seconds=100, picoseconds=100)

        with self.assertRaises(TypeError):
            t0 - "string"  # type:ignore

        with self.assertRaises(TypeError):
            1.0 - t0  # type:ignore

        with self.assertRaises(TypeError):
            np.ones((5, 2)) - t0  # type:ignore

    def test_isub_invalid(self):
        t0 = PreciseDateTime(seconds=100, picoseconds=100)
        invalid_element = ["a string"]
        for other in invalid_element:
            with self.subTest(msg=f"{type(other)}"):
                with self.assertRaises(TypeError):
                    t0 -= other  # type: ignore

    def test_total_ordering(self):
        self.assertGreater(
            PreciseDateTime(seconds=100, picoseconds=200),
            PreciseDateTime(seconds=100, picoseconds=100),
        )
        self.assertGreaterEqual(
            PreciseDateTime(seconds=100, picoseconds=200),
            PreciseDateTime(seconds=100, picoseconds=100),
        )

        self.assertLess(
            PreciseDateTime(seconds=100, picoseconds=100),
            PreciseDateTime(seconds=100, picoseconds=200),
        )
        self.assertLessEqual(
            PreciseDateTime(seconds=100, picoseconds=100),
            PreciseDateTime(seconds=100, picoseconds=200),
        )

        self.assertNotEqual(
            PreciseDateTime(seconds=100, picoseconds=200),
            PreciseDateTime(seconds=100, picoseconds=100),
        )

        self.assertEqual(
            PreciseDateTime(seconds=100, picoseconds=100),
            PreciseDateTime(seconds=100, picoseconds=100),
        )
        self.assertLessEqual(
            PreciseDateTime(seconds=100, picoseconds=100),
            PreciseDateTime(seconds=100, picoseconds=100),
        )
        self.assertGreaterEqual(
            PreciseDateTime(seconds=100, picoseconds=100),
            PreciseDateTime(seconds=100, picoseconds=100),
        )


if __name__ == "__main__":
    unittest.main()
