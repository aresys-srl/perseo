# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for timing module"""

from dataclasses import dataclass

import numpy as np
import pytest

from perseo_core.timing import PreciseDateTime


@dataclass(frozen=True)
class ArithmeticParams:
    """Parameters for PreciseDateTime arithmetic tests."""

    seconds: float
    picoseconds: float
    delta: float
    reference_seconds: float
    reference_picoseconds: float


def get_arithmetic_add_test_params() -> list[ArithmeticParams]:
    """Return fixture data for PreciseDateTime addition tests.

    Returns
    -------
    list[ArithmeticParams]
        List of parameters for addition test cases.
    """
    return [
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


def get_arithmetic_sub_test_params() -> list[ArithmeticParams]:
    """Return fixture data for PreciseDateTime subtraction tests.

    Returns
    -------
    list[ArithmeticParams]
        List of parameters for subtraction test cases.
    """
    return [
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


class TestPreciseDateTime:
    """Test PreciseDateTime initialization, properties, and conversion methods."""

    def test_init_invalid_seconds(self) -> None:
        """Test that PreciseDateTime raises ValueError with negative seconds."""
        with pytest.raises(ValueError, match="The specified time is before the reference date"):
            PreciseDateTime(-1)

    def test_init_invalid_picoseconds(self) -> None:
        """Test that PreciseDateTime raises ValueError with negative picoseconds."""
        with pytest.raises(ValueError, match="The specified time is before the reference date"):
            PreciseDateTime(0, -1)

    def test_properties_accessor(self) -> None:
        """Test that PreciseDateTime properties correctly return date/time components."""
        date = PreciseDateTime.from_numeric_datetime(2021, 7, 29, 14, 6, 12, 113_324)

        assert date.year == 2021
        assert date.month == 7
        assert date.day_of_the_month == 29
        assert date.hour_of_day == 14
        assert date.minute_of_hour == 6
        assert date.second_of_minute == 12
        assert date.picosecond_of_second == 113_324

        assert date.day_of_the_year == 210
        assert date.fraction_of_day == pytest.approx(0.5876388889)

    def test_fromisoformat_isoformat(self) -> None:
        """Test that fromisoformat and isoformat are inverse operations."""
        date = PreciseDateTime.from_numeric_datetime(1989, 11, 18, 23, 51, 20)
        assert date == PreciseDateTime.fromisoformat(date.isoformat())

    def test_isoformat_timespec_auto(self) -> None:
        """Test that isoformat automatically chooses appropriate precision."""
        date = PreciseDateTime.from_numeric_datetime(1989, 11, 18, 23, 51, 20)
        assert date.isoformat() == "1989-11-18T23:51:20Z"
        assert date.isoformat(timespec="picoseconds") == "1989-11-18T23:51:20.000000000000Z"

        date = PreciseDateTime.from_numeric_datetime(1989, 11, 18, 23, 51, 20, 12)
        assert date.isoformat() == "1989-11-18T23:51:20.000000000012Z"

    def test_isoformat_timespec(self) -> None:
        """Test that isoformat respects specified timespec precision levels."""
        date = PreciseDateTime.from_numeric_datetime(1989, 11, 18, 23, 51, 20, 123456789012)
        assert date.isoformat() == "1989-11-18T23:51:20.123456789012Z"
        assert date.isoformat(timespec="nanoseconds") == "1989-11-18T23:51:20.123456789Z"
        assert date.isoformat(timespec="microseconds") == "1989-11-18T23:51:20.123456Z"
        assert date.isoformat(timespec="milliseconds") == "1989-11-18T23:51:20.123Z"
        assert date.isoformat(timespec="seconds") == "1989-11-18T23:51:20Z"
        assert date.isoformat(timespec="minutes") == "1989-11-18T23:51Z"
        assert date.isoformat(timespec="hours") == "1989-11-18T23Z"

    def test_isoformat_invalid_timespec(self) -> None:
        """Test that isoformat raises ValueError for invalid timespec values."""
        date = PreciseDateTime.from_numeric_datetime(1989, 11, 18, 23, 51, 20, 123456789012)
        with pytest.raises(ValueError, match="Unknown timespec value"):
            date.isoformat(timespec="days")

    def test_isoformat_change_separator(self) -> None:
        """Test that isoformat respects custom date/time separator."""
        date = PreciseDateTime.from_numeric_datetime(1989, 11, 18, 23, 51, 20)
        assert date.isoformat(sep=" ") == "1989-11-18 23:51:20Z"

    def test_fromisoformat(self) -> None:
        """Test that fromisoformat correctly parses ISO format strings with varying precision."""
        assert PreciseDateTime.fromisoformat("1989") == PreciseDateTime.from_numeric_datetime(1989)

        assert PreciseDateTime.fromisoformat("1989-11") == PreciseDateTime.from_numeric_datetime(1989, 11)

        assert PreciseDateTime.fromisoformat("1989-11-18") == PreciseDateTime.from_numeric_datetime(1989, 11, 18)

        assert PreciseDateTime.fromisoformat("1989-11-18T23") == PreciseDateTime.from_numeric_datetime(1989, 11, 18, 23)

        assert PreciseDateTime.fromisoformat("1989-11-18T23Z") == PreciseDateTime.from_numeric_datetime(
            1989, 11, 18, 23
        )

        assert PreciseDateTime.fromisoformat("1989-11-18T23:51") == PreciseDateTime.from_numeric_datetime(
            1989, 11, 18, 23, 51
        )

        assert PreciseDateTime.fromisoformat("1989-11-18T23:51Z") == PreciseDateTime.from_numeric_datetime(
            1989, 11, 18, 23, 51
        )

        assert PreciseDateTime.fromisoformat("1989-11-18T23:51:20") == PreciseDateTime.from_numeric_datetime(
            1989, 11, 18, 23, 51, 20
        )

        assert PreciseDateTime.fromisoformat("1989-11-18T23:51:20Z") == PreciseDateTime.from_numeric_datetime(
            1989, 11, 18, 23, 51, 20
        )

        assert PreciseDateTime.fromisoformat("1989-11-18T23:51:20.123") == PreciseDateTime.from_numeric_datetime(
            1989, 11, 18, 23, 51, 20, 123000000000
        )

        assert PreciseDateTime.fromisoformat("1989-11-18T23:51:20.123Z") == PreciseDateTime.from_numeric_datetime(
            1989, 11, 18, 23, 51, 20, 123000000000
        )

        assert PreciseDateTime.fromisoformat("1989-11-18T23:51:20.123456") == PreciseDateTime.from_numeric_datetime(
            1989, 11, 18, 23, 51, 20, 123456000000
        )

        assert PreciseDateTime.fromisoformat("1989-11-18T23:51:20.123456Z") == PreciseDateTime.from_numeric_datetime(
            1989, 11, 18, 23, 51, 20, 123456000000
        )

        assert PreciseDateTime.fromisoformat("1989-11-18T23:51:20.123456789") == PreciseDateTime.from_numeric_datetime(
            1989, 11, 18, 23, 51, 20, 123456789000
        )

        assert PreciseDateTime.fromisoformat("1989-11-18T23:51:20.123456789Z") == PreciseDateTime.from_numeric_datetime(
            1989, 11, 18, 23, 51, 20, 123456789000
        )

        assert PreciseDateTime.fromisoformat(
            "1989-11-18T23:51:20.123456789012"
        ) == PreciseDateTime.from_numeric_datetime(1989, 11, 18, 23, 51, 20, 123456789012)

        assert PreciseDateTime.fromisoformat(
            "1989-11-18T23:51:20.123456789012Z"
        ) == PreciseDateTime.from_numeric_datetime(1989, 11, 18, 23, 51, 20, 123456789012)

    def test_fromisoformat_change_separator(self) -> None:
        """Test that fromisoformat respects custom date/time separator."""
        assert PreciseDateTime.fromisoformat(
            "1989-11-18 23:51:20.123456789012Z", sep=" "
        ) == PreciseDateTime.from_numeric_datetime(1989, 11, 18, 23, 51, 20, 123456789012)

    def test_fromisoformat_compact(self) -> None:
        """Test that fromisoformat accepts compact format without separators."""
        assert PreciseDateTime.fromisoformat("19891118") == PreciseDateTime.fromisoformat("1989-11-18")

        assert PreciseDateTime.fromisoformat("1989-11-18T2351") == PreciseDateTime.fromisoformat("1989-11-18T23:51")

        assert PreciseDateTime.fromisoformat("1989-11-18T235120") == PreciseDateTime.fromisoformat(
            "1989-11-18T23:51:20"
        )

        assert PreciseDateTime.fromisoformat("1989-11-18T235120.123") == PreciseDateTime.fromisoformat(
            "1989-11-18T23:51:20.123"
        )

    def test_fromisoformat_invalid_year(self) -> None:
        """Test that fromisoformat raises ValueError for year with insufficient digits."""
        with pytest.raises(ValueError, match="Invalid isoformat string"):
            PreciseDateTime.fromisoformat("189")

    def test_fromisoformat_invalid_year_month_sep(self) -> None:
        """Test that fromisoformat raises ValueError for invalid year/month separator."""
        with pytest.raises(ValueError, match="Invalid isoformat string"):
            PreciseDateTime.fromisoformat("1989/11")

    def test_fromisoformat_invalid_month(self) -> None:
        """Test that fromisoformat raises ValueError for month with insufficient digits."""
        with pytest.raises(ValueError, match="Invalid isoformat string"):
            PreciseDateTime.fromisoformat("1989-1")

    def test_fromisoformat_invalid_year_month_date(self) -> None:
        """Test that fromisoformat raises ValueError for compact format with missing separator."""
        with pytest.raises(ValueError, match="Invalid isoformat string"):
            PreciseDateTime.fromisoformat("198911")

    def test_fromisoformat_incomplete_day(self) -> None:
        """Test that fromisoformat raises ValueError for day with insufficient digits."""
        with pytest.raises(ValueError, match="Invalid isoformat string"):
            PreciseDateTime.fromisoformat("1989-11-1")

    def test_fromisoformat_invalid_month_day_sep(self) -> None:
        with pytest.raises(ValueError, match="Invalid isoformat string"):
            PreciseDateTime.fromisoformat("1989-11 18")

    def test_fromisoformat_unexpected_month_day_sep(self) -> None:
        with pytest.raises(ValueError, match="Invalid isoformat string"):
            PreciseDateTime.fromisoformat("198911-18")

    def test_fromisoformat_invalid_hour(self) -> None:
        with pytest.raises(ValueError, match="Invalid isoformat string"):
            PreciseDateTime.fromisoformat("1989-11-18T2")

    def test_fromisoformat_invalid_hour_minute_sep(self) -> None:
        with pytest.raises(ValueError, match="Invalid isoformat string"):
            PreciseDateTime.fromisoformat("1989-11-18T23.51")

    def test_fromisoformat_invalid_minute(self) -> None:
        with pytest.raises(ValueError, match="Invalid isoformat string"):
            PreciseDateTime.fromisoformat("1989-11-18T23:5")

    def test_fromisoformat_invalid_minute_second_sep(self) -> None:
        with pytest.raises(ValueError, match="Invalid isoformat string"):
            PreciseDateTime.fromisoformat("1989-11-18T23:51.30")

    def test_fromisoformat_unexpected_minute_second_sep(self) -> None:
        with pytest.raises(ValueError, match="Invalid isoformat string"):
            PreciseDateTime.fromisoformat("1989-11-18T2351:30")

    def test_fromisoformat_invalid_second(self) -> None:
        with pytest.raises(ValueError, match="Invalid isoformat string"):
            PreciseDateTime.fromisoformat("1989-11-18T23:51:3")

    def test_fromisoformat_invalid_picosecond(self) -> None:
        with pytest.raises(ValueError, match="Invalid isoformat string"):
            PreciseDateTime.fromisoformat("1989-11-18T23:51:30.")

    def test_fromisoformat_invalid_separtor(self) -> None:
        with pytest.raises(ValueError, match="Invalid isoformat string"):
            PreciseDateTime.fromisoformat("1989-11-18T23:51:30", sep=" ")

    def test_fromisoformat_unsupported_timezone(self) -> None:
        with pytest.raises(ValueError, match="Invalid isoformat string"):
            PreciseDateTime.fromisoformat("1989-11-18T23:51:30+01")


class TestPreciseDateTimeArithmetic:
    """Test PreciseDateTime arithmetic operations."""

    @pytest.mark.parametrize("params", get_arithmetic_add_test_params())
    def test_add(self, params: ArithmeticParams) -> None:
        """Test addition with parametrized cases."""
        t0 = PreciseDateTime(seconds=params.seconds, picoseconds=params.picoseconds)
        t = t0 + params.delta
        assert t._seconds == params.reference_seconds
        assert round(t._picoseconds) == params.reference_picoseconds

    @pytest.mark.parametrize("params", get_arithmetic_add_test_params())
    def test_iadd(self, params: ArithmeticParams) -> None:
        """Test in-place addition with parametrized cases."""
        t = PreciseDateTime(seconds=params.seconds, picoseconds=params.picoseconds)
        idx_reference = id(t)
        t += params.delta
        idx = id(t)
        assert idx == idx_reference
        assert t._seconds == params.reference_seconds
        assert round(t._picoseconds) == params.reference_picoseconds

    @pytest.mark.parametrize("params", get_arithmetic_sub_test_params())
    def test_sub(self, params: ArithmeticParams) -> None:
        """Test subtraction with parametrized cases."""
        t0 = PreciseDateTime(seconds=params.seconds, picoseconds=params.picoseconds)
        t = t0 - params.delta
        assert t._seconds == params.reference_seconds
        assert round(t._picoseconds) == params.reference_picoseconds

    @pytest.mark.parametrize("params", get_arithmetic_sub_test_params())
    def test_isub(self, params: ArithmeticParams) -> None:
        """Test in-place subtraction with parametrized cases."""
        t = PreciseDateTime(seconds=params.seconds, picoseconds=params.picoseconds)
        idx_reference = id(t)
        t -= params.delta
        idx = id(t)
        assert idx == idx_reference
        assert t._seconds == params.reference_seconds
        assert round(t._picoseconds) == params.reference_picoseconds

    def test_isub_pdt(self) -> None:
        t = PreciseDateTime(100, 100)
        b = PreciseDateTime(101, 100)
        b -= t
        assert b == 1.0

    def test_sub_pdt(self) -> None:
        delta = PreciseDateTime(seconds=100, picoseconds=800) - PreciseDateTime(seconds=98, picoseconds=500)
        assert delta == 2.0 + 3.0e-10

        delta = PreciseDateTime(seconds=100, picoseconds=800) - PreciseDateTime(seconds=98, picoseconds=900)
        assert delta == 2 - 1.0e-10

    def test_add_array(self) -> None:
        t0 = PreciseDateTime(seconds=100, picoseconds=100)
        array = np.ones((5, 2))
        reference = PreciseDateTime(seconds=101, picoseconds=100)

        t_array = t0 + array
        assert t_array.shape == (5, 2)
        for time in np.nditer(t_array, flags=["refs_ok"]):
            assert time == reference

        t_array = array + t0
        assert t_array.shape == (5, 2)
        for time in np.nditer(t_array, flags=["refs_ok"]):
            assert time == reference

    def test_sub_array(self) -> None:
        t0 = PreciseDateTime(seconds=100, picoseconds=100)
        array = np.ones((5, 2))
        reference = PreciseDateTime(seconds=99, picoseconds=100)

        t_array = t0 - array
        assert t_array.shape == (5, 2)
        for time in np.nditer(t_array, flags=["refs_ok"]):
            assert time == reference

    def test_add_invalid(self) -> None:
        t0 = PreciseDateTime(seconds=100, picoseconds=100)
        invalid_element = [PreciseDateTime(seconds=100, picoseconds=100), "a string"]

        for other in invalid_element:
            with pytest.raises(TypeError):
                t0 + other  # type:ignore

    def test_iadd_invalid(self) -> None:
        t0 = PreciseDateTime(seconds=100, picoseconds=100)
        invalid_element = [PreciseDateTime(seconds=100, picoseconds=100), "a string"]

        for other in invalid_element:
            with pytest.raises(TypeError):
                t0 += other

    def test_sub_invalid(self) -> None:
        t0 = PreciseDateTime(seconds=100, picoseconds=100)

        with pytest.raises(TypeError):
            t0 - "string"  # type:ignore

        with pytest.raises(TypeError):
            1.0 - t0  # type:ignore

        with pytest.raises(TypeError):
            np.ones((5, 2)) - t0  # type:ignore

    def test_isub_invalid(self) -> None:
        t0 = PreciseDateTime(seconds=100, picoseconds=100)
        invalid_element = ["a string"]
        for other in invalid_element:
            with pytest.raises(TypeError):
                t0 -= other  # type: ignore

    def test_total_ordering(self) -> None:
        assert PreciseDateTime(seconds=100, picoseconds=200) > PreciseDateTime(seconds=100, picoseconds=100)
        assert PreciseDateTime(seconds=100, picoseconds=200) >= PreciseDateTime(seconds=100, picoseconds=100)

        assert PreciseDateTime(seconds=100, picoseconds=100) < PreciseDateTime(seconds=100, picoseconds=200)
        assert PreciseDateTime(seconds=100, picoseconds=100) <= PreciseDateTime(seconds=100, picoseconds=200)

        assert PreciseDateTime(seconds=100, picoseconds=200) != PreciseDateTime(seconds=100, picoseconds=100)

        assert PreciseDateTime(seconds=100, picoseconds=100) == PreciseDateTime(seconds=100, picoseconds=100)
        assert PreciseDateTime(seconds=100, picoseconds=100) <= PreciseDateTime(seconds=100, picoseconds=100)
        assert PreciseDateTime(seconds=100, picoseconds=100) >= PreciseDateTime(seconds=100, picoseconds=100)
