# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Models - Custom Enums
---------------------
"""

from __future__ import annotations

from enum import Enum


class SensorLookDirection(Enum):
    """Looking direction of sensor with respect to sensor velocity"""

    RIGHT_LOOKING = "RIGHT"
    LEFT_LOOKING = "LEFT"
