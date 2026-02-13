# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Timing - Conversions Utilities
------------------------------
"""

from __future__ import annotations

import warnings
from datetime import datetime

import numpy as np

from perseo_core.models.types import NumpyDateTimeArrayType, PreciseDateTimeArrayType
from perseo_core.timing.precise_datetime import PreciseDateTime

_GPS_START_DATE = datetime(year=1980, month=1, day=6)


def date_to_gps_week(date: PreciseDateTime | datetime) -> tuple[int, int]:
    """Convert input date to GPS week.
    GPS weeks are counted since 06-January-1980.

    Parameters
    ----------
    date : PreciseDateTime | datetime
        date to be converted to GPS week, in PreciseDateTime or datetime format

    Returns
    -------
    int
        GPS week
    int
        GPS day of the week

    Raises
    ------
    ValueError
        if input date is before 06-January-1980
    """
    if isinstance(date, PreciseDateTime):
        # converting to datetime
        date = datetime(
            year=date.year,
            month=date.month,
            day=date.day_of_the_month,
            hour=date.hour_of_day,
            minute=date.minute_of_hour,
        )
    delta = date - _GPS_START_DATE

    if delta.days < 0:
        raise ValueError(f"Invalid date: {date} cannot be before {_GPS_START_DATE}")

    weeks = delta.days // 7
    day_of_week = delta.days % 7

    return weeks, day_of_week


def precise_datetime_to_numpy(
    times: PreciseDateTime | PreciseDateTimeArrayType,
) -> np.datetime64 | NumpyDateTimeArrayType:
    """Converting Perseo internal timing object PreciseDateTime (picoseconds precision) to numpy.datetime64 data type,
    truncated to nanoseconds precision.

    Parameters
    ----------
    date : PreciseDateTime | PreciseDateTimeArrayType
        times to be converted from PreciseDateTime format to numpy.datetime64[ns]

    Returns
    -------
    np.datetime64 | NumpyDateTimeArrayType
        numpy.datetime64[ns] at nanoseconds precision corresponding to the input times
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        times_ = np.atleast_1d(times)
        times_ = np.array([np.datetime64(t.isoformat(), "ns") for t in times_])
        if isinstance(times, PreciseDateTime):
            return times_[0]
        return times_
