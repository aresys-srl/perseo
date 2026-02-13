# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Models - Custom Enums
---------------------
"""

from __future__ import annotations

from enum import Enum, auto


# TODO: change with Literal?
class SensorLookDirection(Enum):
    """Looking direction of sensor with respect to sensor velocity"""

    RIGHT_LOOKING = "RIGHT"
    LEFT_LOOKING = "LEFT"


class SARRadiometricQuantity(Enum):
    """Enum class for radiometric analysis input/output quantity types"""

    BETA_NOUGHT = auto()
    SIGMA_NOUGHT = auto()
    GAMMA_NOUGHT = auto()


class GetFrequencyMethod(Enum):
    """Enum class for get local frequency settings"""

    AUTOCORRELATION = auto()
    FFT = auto()
    POWER_BALANCE = auto()


class SARAcquisitionMode(Enum):
    """Acquisition mode enum class"""

    SCANSAR = auto()
    SPOTLIGHT = auto()
    STRIPMAP = auto()
    TOPSAR = auto()
    WAVE = auto()
