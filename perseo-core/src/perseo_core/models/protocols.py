# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Models - Custom Protocols
-------------------------
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from perseo_core.models.types import ExtendedDatetimeType


# TODO: check this, from quality protocols
@runtime_checkable
class SARCoordinatesFunction(Protocol):
    """Protocol to define a function taking SAR coordinates (Azimuth, Range) as inputs and returns a float
    This can be any generic f: SAR Times -> R.
    """

    def evaluate(self, azimuth_time: ExtendedDatetimeType, range_time: float) -> float:
        """Evaluate the wrapped function at given azimuth and range times.

        Parameters
        ----------
        azimuth_time : ExtendedDatetimeType
            azimuth time at which evaluate the function
        range_time : float
            range time at which evaluate the function

        Returns
        -------
        float
            output of the wrapped function
        """
