# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Test data fixtures for timing module tests."""

from datetime import datetime

import numpy as np

from perseo_core.timing import PreciseDateTime


def get_gps_week_conversion_test_data():
    """Return fixture data for ``date_to_gps_week`` tests.

    Returns
    -------
    dict[str, object]
        Valid/invalid input dates and expected tuple result.
    """
    return {
        "input_date": PreciseDateTime.from_numeric_datetime(2012, 6, 15, 17, 30),
        "input_date_2": datetime(1979, 6, 15, 17, 30),
        "ref_results": (1692, 5),
    }


def get_precise_datetime_to_numpy_test_data():
    """Return fixture data for ``precise_datetime_to_numpy`` tests.

    Returns
    -------
    dict[str, object]
        Base time, time deltas, and expected ``datetime64`` values.
    """
    input_date = PreciseDateTime.from_numeric_datetime(2012, 6, 15, 17, 30)
    time_deltas = [115e-12, 115e-9, 115.000854]
    ref_results = [
        np.datetime64("2012-06-15T17:30:00.000000000"),
        np.datetime64("2012-06-15T17:30:00.000000115"),
        np.datetime64("2012-06-15T17:31:55.000854000"),
    ]

    return {
        "input_date": input_date,
        "time_deltas": time_deltas,
        "ref_results": ref_results,
    }
