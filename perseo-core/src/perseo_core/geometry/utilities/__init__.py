# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Geometry - Utilities
--------------------
"""

from enum import Enum
from typing import Union


class RotationOrder(Enum):
    """Yaw / Pitch / Roll rotation orders"""

    ypr = "YPR"
    yrp = "YRP"
    pry = "PRY"
    pyr = "PYR"
    ryp = "RYP"
    rpy = "RPY"


RotationOrderLike = Union[str, RotationOrder]


class ReferenceFrame(Enum):
    """Available reference frames"""

    GEOCENTRIC = "GEOCENTRIC"
    GEODETIC = "GEODETIC"
    ZERO_DOPPLER = "ZERODOPPLER"


ReferenceFrameLike = Union[str, ReferenceFrame]
