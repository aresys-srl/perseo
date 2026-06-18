# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Precise date-time object with picosecond timing accuracy."""

from __future__ import annotations

import datetime
import functools
import math
import re
from typing import Self, overload

from dateutil import parser, tz  # pyrefly: ignore

_ISO_DATE_SEPARATOR = "-"
_ISO_TIME_SEPARATOR = ":"
_ISO_FRACTION_REGEX = re.compile("[\\.,]([0-9]+)")


def _parse_isoformat_date(date_string: str) -> tuple[tuple[int, int, int], str]:
    """Parse string formatted according to ISO, returns date components and remaining chars to parse."""
    year_length = 4
    if len(date_string) < year_length:
        msg = "Incomplete year component"
        raise ValueError(msg)
    year = int(date_string[:year_length])
    date_string = date_string[year_length:]
    if not date_string:
        return (year, 1, 1), date_string

    # skip separator if present
    has_separator = date_string[0] == _ISO_DATE_SEPARATOR
    if has_separator:
        date_string = date_string[1:]

    month_length = 2
    if len(date_string) < month_length:
        msg = "Incomplete month component"
        raise ValueError(msg)
    month = int(date_string[:month_length])
    date_string = date_string[month_length:]
    if not date_string:
        if has_separator:
            return (year, month, 1), date_string
        msg = "Invalid ISO date"
        raise ValueError(msg)

    # skip separator if present
    if has_separator:
        if date_string[0] != _ISO_DATE_SEPARATOR:
            msg = "Unexpected date separator"
            raise ValueError(msg)
        date_string = date_string[1:]

    day_length = 2
    if len(date_string) < day_length:
        msg = "Incomplete day component"
        raise ValueError(msg)
    day = int(date_string[:day_length])
    date_string = date_string[day_length:]

    return (year, month, day), date_string


def _isoformat_date(year: int, month: int, day: int) -> str:
    return f"{year:04d}-{month:02d}-{day:02d}"


def _parse_isoformat_tz(time_string: str) -> tuple[datetime.tzinfo | None, str]:
    if not time_string:
        return None, time_string

    if time_string[0] == "z" or time_string[0] == "Z":
        return tz.UTC, time_string[1:]

    if time_string[0] == "+" or time_string[0] == "-":
        msg = "Unsupported timezone"
        raise ValueError(msg)

    return None, time_string


def _strip_timezone(time: tuple[int, int, int, int, datetime.tzinfo | None]) -> tuple[int, int, int, int]:
    timezone = time[-1]
    if timezone is not None and timezone != tz.UTC:
        msg = f"Unsupported timezone: {timezone}"
        raise ValueError(msg)
    return time[:-1]


def _parse_isoformat_section(time_string: str, tag: str, length: int = 2) -> tuple[int, str]:
    if len(time_string) < length:
        msg = f"Incomplete {tag} component"
        raise ValueError(msg)
    value = int(time_string[:length])
    time_string = time_string[length:]
    return value, time_string


def _parse_isoformat_hour(time_string: str) -> tuple[int, str]:
    return _parse_isoformat_section(time_string, "hour")


def _parse_isoformat_minute(time_string: str) -> tuple[int, str]:
    return _parse_isoformat_section(time_string, "minute")


def _parse_isoformat_second(time_string: str) -> tuple[int, str]:
    return _parse_isoformat_section(time_string, "second")


def _skip_separator(time_string: str, separator: str = _ISO_TIME_SEPARATOR) -> tuple[str, bool]:
    if time_string and time_string[0] == separator:
        return time_string[1:], True
    return time_string, False


def _parse_isoformat_time(time_string: str) -> tuple[tuple[int, int, int, int, datetime.tzinfo | None], str]:
    """Parse string formatted according to ISO, returns time components and remaining chars to parse."""
    minute, second, picosecond, timezone = 0, 0, 0, None

    hour, time_string = _parse_isoformat_hour(time_string)
    if not time_string:
        return (hour, minute, second, picosecond, timezone), time_string

    timezone, time_string = _parse_isoformat_tz(time_string)
    if timezone is not None:
        return (hour, minute, second, picosecond, timezone), time_string

    time_string, has_separator = _skip_separator(time_string)

    minute, time_string = _parse_isoformat_minute(time_string)
    if not time_string:
        return (hour, minute, second, picosecond, timezone), time_string

    timezone, time_string = _parse_isoformat_tz(time_string)
    if timezone is not None:
        return (hour, minute, second, picosecond, timezone), time_string

    if has_separator:
        if time_string[0] != _ISO_TIME_SEPARATOR:
            msg = "Unexpected time separator"
            raise ValueError(msg)
        time_string = time_string[1:]

    second, time_string = _parse_isoformat_second(time_string)
    if not time_string:
        return (hour, minute, second, picosecond, timezone), time_string

    timezone, time_string = _parse_isoformat_tz(time_string)
    if timezone is not None:
        return (hour, minute, second, picosecond, timezone), time_string

    # parse fraction of second
    fraction_match = _ISO_FRACTION_REGEX.match(time_string)
    picosecond = 0
    if fraction_match:
        fraction = fraction_match.group(1)
        digits_count = len(fraction)

        scale_factor = 10 ** (12 - digits_count)
        picosecond = int(fraction) * scale_factor

        time_string = time_string[len(fraction_match.group()) :]

    timezone, time_string = _parse_isoformat_tz(time_string)

    return (hour, minute, second, picosecond, timezone), time_string


def _isoformat_time(hour: int, minute: int, second: int, picosecond: int, timespec: str = "auto") -> str:
    fmt_specs = {
        "hours": "{:02d}",
        "minutes": "{:02d}:{:02d}",
        "seconds": "{:02d}:{:02d}:{:02d}",
        "milliseconds": "{:02d}:{:02d}:{:02d}.{:03d}",
        "microseconds": "{:02d}:{:02d}:{:02d}.{:06d}",
        "nanoseconds": "{:02d}:{:02d}:{:02d}.{:09d}",
        "picoseconds": "{:02d}:{:02d}:{:02d}.{:012d}",
    }

    fraction_of_second = picosecond
    if timespec == "auto":
        timespec = "picoseconds" if picosecond else "seconds"
    elif timespec == "milliseconds":
        fraction_of_second //= 1_000_000_000
    elif timespec == "microseconds":
        fraction_of_second //= 1_000_000
    elif timespec == "nanoseconds":
        fraction_of_second //= 1_000

    try:
        fmt = fmt_specs[timespec]
    except KeyError as exc:
        msg = "Unknown timespec value"
        raise ValueError(msg) from exc

    return fmt.format(hour, minute, second, fraction_of_second)


@functools.total_ordering
class PreciseDateTime:
    """Precise Date Time format specification.

    | Field | Value / Format | Description |
    | --- | --- | --- |
    | Precision | 1e-12 s | Picosecond precision |
    | Standard format | `"DD-MMM-YYYY hh:mm:ss.pppppppppppp"` | Default string representation |
    | Standard reference date | `PRECISEDATETIME_REFERENCE_TIME` | Reference epoch |
    | DD | — | Day |
    | MMM | `JAN`, `FEB`, `MAR`, `APR`, `MAY`, `JUN`, `JUL`, `AUG`, `SEP`, `OCT`, `NOV`, `DEC` |\
    Month (three-letter uppercase abbreviation) |
    | YYYY | — | Year (4 digits) |
    | hh | 00-23 | Hours (24-hour format) |
    | mm | 00-59 | Minutes |
    | ss | 00-59 | Seconds |
    | pppppppppppp | 12 digits | Picoseconds |
    """

    _MONTH_ABBREVIATED_NAME_DIRECTIVE = "%b"
    _MONTH_ABBREVIATED_NAMES = (
        "JAN",
        "FEB",
        "MAR",
        "APR",
        "MAY",
        "JUN",
        "JUL",
        "AUG",
        "SEP",
        "OCT",
        "NOV",
        "DEC",
    )
    _STRING_FORMAT = "%d-%b-%Y %H:%M:%S."
    _REFERENCE_DATETIME = datetime.datetime(year=1985, month=1, day=1)
    _TIME_DIFF_REFERENCE_FROM_1985 = _REFERENCE_DATETIME - datetime.datetime(year=1985, month=1, day=1)
    _PRECISION = 1e-12  # Precision of the decimal part
    _SECONDS_IN_A_DAY = 24 * 60 * 60

    def __init__(self, seconds: float = 0.0, picoseconds: float = 0.0) -> None:
        """Initialize the PreciseDateTime object with the specified time point.

        Parameters
        ----------
        seconds : float, optional
            number of seconds since reference date, by default 0.0
        picoseconds : float, optional
            number of picoseconds since reference date, by default 0.0

        Raises
        ------
        ValueError
            Raised when negative values are passed as arguments

        """
        self._set_state(seconds, picoseconds)

    __hash__ = None  # type: ignore[assignment]

    def _set_state(self, seconds: float, picoseconds: float) -> None:
        """Set the object at the required time point.

        Parameters
        ----------
        seconds : float
            number of seconds from the reference date
        picoseconds : float
            number of picoseconds to add to the specified seconds

        Raises
        ------
        ValueError
            if the time point would be prior to PRECISEDATETIME_REFERENCE_TIME

        """
        seconds_fraction = seconds - int(seconds)
        picoseconds_in_seconds_fraction = seconds_fraction / self._PRECISION
        tot_picoseconds = picoseconds + picoseconds_in_seconds_fraction

        # if 'tot_picoseconds' is too small the "normalized_picoseconds" may round up
        # to 1e12 that is not allowed as an internal state.
        ndigits = 2
        picoseconds_threhsold = 10**-ndigits
        if -picoseconds_threhsold < tot_picoseconds < picoseconds_threhsold:
            tot_picoseconds = round(tot_picoseconds, ndigits=ndigits)

        seconds_adj = math.floor(tot_picoseconds * self._PRECISION)
        normalized_seconds = int(seconds) + seconds_adj
        normalized_picoseconds = tot_picoseconds % (1 / self._PRECISION)

        if normalized_seconds < 0:
            msg = "The specified time is before the reference date"
            raise ValueError(msg)

        assert normalized_seconds >= 0
        assert 0 <= normalized_picoseconds < 1 / self._PRECISION

        self._seconds = normalized_seconds
        self._picoseconds = normalized_picoseconds

        assert isinstance(self._seconds, int)

    def __iadd__(self, seconds: float) -> Self:
        """Add the input seconds to the current time point.

        Parameters
        ----------
        seconds : float
            number of seconds to add to the current time point

        Returns
        -------
        PreciseDateTime
            self

        """
        if not isinstance(seconds, (int, float)):
            return NotImplemented

        seconds_fraction = seconds - int(seconds)
        self._set_state(
            self._seconds + int(seconds),
            self._picoseconds + seconds_fraction / self._PRECISION,
        )
        return self

    def __isub__(self, seconds: float) -> Self:  # type: ignore[misc]
        """Subtract the input seconds from the current time point.

        Parameters
        ----------
        seconds : float
            number of seconds to subtract from the current time point

        Returns
        -------
        PreciseDateTime
            self

        """
        if not isinstance(seconds, (int, float)):
            return NotImplemented

        seconds_fraction = seconds - int(seconds)
        self._set_state(
            self._seconds - int(seconds),
            self._picoseconds - seconds_fraction / self._PRECISION,
        )
        return self

    def __add__(self, seconds: float) -> PreciseDateTime:
        """Return the sum between the current time point and the specified input seconds.

        Parameters
        ----------
        seconds : float
            number of seconds to add to the current time point

        Returns
        -------
        PreciseDateTime
            a new PreciseDateTime object initialized to the resulting time point

        """
        if not isinstance(seconds, (int, float)):
            return NotImplemented

        seconds_fraction = seconds - int(seconds)
        return PreciseDateTime(
            self._seconds + int(seconds),
            self._picoseconds + seconds_fraction / self._PRECISION,
        )

    __radd__ = __add__

    @overload
    def __sub__(self, other: float) -> PreciseDateTime: ...

    @overload
    def __sub__(self, other: PreciseDateTime) -> float: ...

    def __sub__(self, other: float | PreciseDateTime) -> float | PreciseDateTime:
        """Return the difference between the current time point and a float or a PreciseDateTime object.

        Parameters
        ----------
        other : float | PreciseDateTime
            number of seconds or PreciseDateTime object to subtract from the current time point

        Returns
        -------
        float | PreciseDateTime:
            if the input parameter is a PreciseDateTime object, the difference in seconds between the two time
            points; otherwise, a new PreciseDateTime object initialized to the resulting time point

        """
        if isinstance(other, PreciseDateTime):
            seconds_fraction = (self._picoseconds - other._picoseconds) * self._PRECISION
            return self._seconds - other._seconds + seconds_fraction

        if isinstance(other, (int, float)):
            seconds_fraction = other - int(other)
            return PreciseDateTime(
                self._seconds - int(other),
                self._picoseconds - seconds_fraction / self._PRECISION,
            )

        return NotImplemented

    def __repr__(self) -> str:
        """Return a string representation of the current time point."""
        assert isinstance(self._seconds, int)
        assert 0 <= self._picoseconds < 1 / self._PRECISION
        absolute_datetime = self._REFERENCE_DATETIME + datetime.timedelta(0, self._seconds)
        tmp_str = f"{int(self._picoseconds):0>12d}"

        # Replacing month abbreviated name directive with english abbreviated month name
        # to be locale independent
        month_id = int(absolute_datetime.strftime("%m")) - 1
        month_name = self._MONTH_ABBREVIATED_NAMES[month_id]
        updated_string_format = self._STRING_FORMAT.replace(self._MONTH_ABBREVIATED_NAME_DIRECTIVE, month_name)

        return absolute_datetime.strftime(updated_string_format) + tmp_str

    def __eq__(self, other: object) -> bool:
        """Compare the current time point with another PreciseDateTime object for equality."""
        if isinstance(other, self.__class__):
            return (other._seconds == self._seconds) and (other._picoseconds == self._picoseconds)

        return NotImplemented

    def __lt__(self, other: object) -> bool:
        """Compare the current time point with another PreciseDateTime object."""
        if isinstance(other, self.__class__):
            return self._seconds < other._seconds or (
                self._seconds == other._seconds and self._picoseconds < other._picoseconds
            )

        return NotImplemented

    @classmethod
    def get_precision(cls) -> float:
        """Date-time representation precision.

        Returns
        -------
        float
            precision of the date-time representation in seconds

        """
        return cls._PRECISION

    @classmethod
    def get_reference_datetime(cls) -> datetime.datetime:
        """Get reference date-time.

        Returns
        -------
        datetime.datetime
            reference date-time as a datetime object

        """
        return cls._REFERENCE_DATETIME

    @property
    def year(self) -> int:
        """Year associated to the current time point."""
        assert 0 <= self._picoseconds < 1 / self._PRECISION
        absolute_datetime = self._REFERENCE_DATETIME + datetime.timedelta(0, self._seconds)
        return int(absolute_datetime.strftime("%Y"))

    @property
    def month(self) -> int:
        """Month associated to the current time point."""
        assert 0 <= self._picoseconds < 1 / self._PRECISION
        absolute_datetime = self._REFERENCE_DATETIME + datetime.timedelta(0, self._seconds)
        return int(absolute_datetime.strftime("%m"))

    @property
    def day_of_the_month(self) -> int:
        """Day of the month associated to the current time point."""
        assert 0 <= self._picoseconds < 1 / self._PRECISION
        absolute_datetime = self._REFERENCE_DATETIME + datetime.timedelta(0, self._seconds)
        return int(absolute_datetime.strftime("%d"))

    @property
    def hour_of_day(self) -> int:
        """Hour of the day associated to the current time point."""
        assert 0 <= self._picoseconds < 1 / self._PRECISION
        absolute_datetime = self._REFERENCE_DATETIME + datetime.timedelta(0, self._seconds)
        return int(absolute_datetime.strftime("%H"))

    @property
    def minute_of_hour(self) -> int:
        """Minute of the hour associated to the current time point."""
        assert 0 <= self._picoseconds < 1 / self._PRECISION
        absolute_datetime = self._REFERENCE_DATETIME + datetime.timedelta(0, self._seconds)
        return int(absolute_datetime.strftime("%M"))

    @property
    def second_of_minute(self) -> int:
        """Second of the minute associated to the current time point."""
        assert 0 <= self._picoseconds < 1 / self._PRECISION
        absolute_datetime = self._REFERENCE_DATETIME + datetime.timedelta(0, self._seconds)
        return int(absolute_datetime.strftime("%S"))

    @property
    def picosecond_of_second(self) -> float:
        """Picosecond of the second associated to the current time point."""
        assert isinstance(self._seconds, int)
        assert 0 <= self._picoseconds < 1 / self._PRECISION
        return self._picoseconds

    @property
    def fraction_of_day(self) -> float:
        """Fraction of the day associated to the current time point."""
        assert 0 <= self._picoseconds < 1 / self._PRECISION
        seconds_from_day_start = self._seconds % self._SECONDS_IN_A_DAY
        return (seconds_from_day_start + self._picoseconds * self._PRECISION) / self._SECONDS_IN_A_DAY

    @property
    def day_of_the_year(self) -> int:
        """Day from the first day of the year associated to the current time point."""
        assert 0 <= self._picoseconds < 1 / self._PRECISION
        absolute_datetime_first_day_of_year = datetime.datetime(self.year, 1, 1)
        absolute_datetime = self._REFERENCE_DATETIME + datetime.timedelta(0, self._seconds)
        time_diff_from_first_day_of_year = absolute_datetime - absolute_datetime_first_day_of_year
        return time_diff_from_first_day_of_year.days + 1

    @property
    def sec85(self) -> float:
        """Time distance in seconds from PRECISEDATETIME_1985 to the current time point."""
        return self._TIME_DIFF_REFERENCE_FROM_1985.total_seconds() + self._seconds + self._picoseconds * self._PRECISION

    @classmethod
    def now(cls) -> PreciseDateTime:
        """Create an object with the current time (local timezone)."""
        absolute_datetime_now = datetime.datetime.now()
        time_diff_from_reference_date = absolute_datetime_now - cls._REFERENCE_DATETIME
        seconds = time_diff_from_reference_date.total_seconds()
        return cls(seconds, 0)

    @classmethod
    def from_sec85(cls, seconds: float) -> PreciseDateTime:
        """Create an object with the time point defined by adding the specified input seconds to PRECISEDATETIME_1985.

        Parameters
        ----------
        seconds : float
            number of seconds from PRECISEDATETIME_1985

        Returns
        -------
        PreciseDateTime
            the time point

        """
        return cls(seconds - cls._TIME_DIFF_REFERENCE_FROM_1985.total_seconds(), 0)

    @classmethod
    def from_utc_string(cls, utc_str: str) -> PreciseDateTime:
        """Create an object with the time point specified by the input UTC string.

        Parameters
        ----------
        utc_str : str
            UTC string

        Returns
        -------
        PreciseDateTime
            the time point

        """
        try:
            seconds_fraction = float("0." + utc_str.split(".")[1].strip())
            absolute_datetime = parser.parse(utc_str).replace(microsecond=0)
        except Exception as exc:
            msg = f"Invalid utc string: {utc_str}"
            raise ValueError(msg) from exc

        time_diff_from_reference_date = absolute_datetime - cls._REFERENCE_DATETIME

        seconds = time_diff_from_reference_date.total_seconds()
        picoseconds = seconds_fraction / cls._PRECISION

        return cls(seconds, picoseconds)

    @classmethod
    def from_numeric_datetime(
        cls,
        year: int,
        month: int = 1,
        day: int = 1,
        hours: int = 0,
        minutes: int = 0,
        seconds: int = 0,
        picoseconds: float = 0.0,
    ) -> PreciseDateTime:
        """Create an object with the time point specified by the input date and time parameters.

        Parameters
        ----------
        year : int
            year
        month : int, optional
            from 1 to 12, by default 1
        day : int, optional
            from 1 to 28-31 (depending on month), by default 1
        hours : int, optional
            from 0 to 23, by default 0
        minutes : int, optional
            from 0 to 59, by default 0
        seconds : int, optional
            from 0 to 59, by default 0
        picoseconds : float, optional
            non-negative and less than 1e12, by default 0.0

        Returns
        -------
        PreciseDateTime
            the time point

        Raises
        ------
        ValueError
            in case of invalid picoseconds

        """
        absolute_datetime = datetime.datetime(year, month, day, hours, minutes, seconds)
        if not 0 <= picoseconds < 1 / cls._PRECISION:
            msg = f"Picoseconds must be non-negative and less than {1 / cls._PRECISION}"
            raise ValueError(msg)

        time_diff_from_reference_date = absolute_datetime - cls._REFERENCE_DATETIME
        total_seconds = time_diff_from_reference_date.total_seconds()

        return cls(total_seconds, picoseconds)

    @classmethod
    def fromisoformat(cls, datetime_string: str, sep: str = "T") -> PreciseDateTime:
        """Create an object with the time specified by the input ISO string.

        Parameters
        ----------
        datetime_string : str
            time specified as ISO string
        sep : str, optional
            separator between date and time, by default "T"

        Returns
        -------
        PreciseDateTime
            the time point

        Raises
        ------
        ValueError
            in case of an invalid input datetime string

        """
        date_string = datetime_string
        try:
            date, time_string = _parse_isoformat_date(date_string)
        except ValueError as exc:
            msg = f"Invalid isoformat string: {datetime_string}"
            raise ValueError(msg) from exc

        time: tuple[int, ...] = ()
        if time_string:
            if time_string[0] == sep:
                time_string = time_string[1:]

            try:
                time_tz, unparsed_string = _parse_isoformat_time(time_string)
                time = _strip_timezone(time_tz)
            except ValueError as exc:
                msg = f"Invalid isoformat string: {datetime_string}"
                raise ValueError(msg) from exc

            if unparsed_string:
                msg = f"Invalid isoformat string: {datetime_string}"
                raise ValueError(msg)

        return cls.from_numeric_datetime(*(date + time))

    def isoformat(self, sep: str = "T", timespec: str = "auto") -> str:
        """ISO formatting of the time point.

        Parameters
        ----------
        sep : str, optional
            separator between date and time, by default "T"
        timespec : str, optional
            number of extra terms to include in the string, by default ``auto``.

            Valid options are: ``auto``, ``hours``, ``minutes``, ``seconds``, ``milliseconds``, ``microseconds``,
            ``nanoseconds`` and ``picoseconds``.

        Returns
        -------
        str
            time formatted according to ISO

        """
        absolute_datetime = self._REFERENCE_DATETIME + datetime.timedelta(seconds=self._seconds)
        date = _isoformat_date(absolute_datetime.year, absolute_datetime.month, absolute_datetime.day)
        time = _isoformat_time(
            absolute_datetime.hour,
            absolute_datetime.minute,
            absolute_datetime.second,
            int(self._picoseconds),
            timespec=timespec,
        )
        return f"{date}{sep:s}{time}Z"


__all__ = ["PreciseDateTime"]
