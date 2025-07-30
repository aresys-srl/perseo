# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Definition of Radiometric Analysis specific dataclasses"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from arepytools.timing.precisedatetime import PreciseDateTime

from perseo_quality.core.generic_dataclasses import SARPolarization, SARProjection, SARRadiometricQuantity


class RadiometricAnalysisDirection(Enum):
    """Enum class for radiometric analysis direction"""

    RANGE = auto()
    AZIMUTH = auto()
    ALL = auto()


class RadiometricAnalysisValue(Enum):
    """Enum class for radiometric analysis value to be represented"""

    AMPLITUDE = auto()
    PHASE = auto()


class RadiometricAnalysisAxes(Enum):
    """Enum class for radiometric analysis output axes to represent data versus"""

    NATURAL = auto()  # Time/Distance
    INCIDENCE_ANGLE = auto()
    LOOK_ANGLE = auto()


@dataclass
class RadiometricProfilesOutput:
    """Dataclass to collect Radiometric Profiles output"""

    swath: str | None = None
    channel: str | int | None = None
    polarization: SARPolarization | None = None
    direction: RadiometricAnalysisDirection | None = None
    output_radiometric_quantity: SARRadiometricQuantity | None = None
    azimuth_block_centers: np.ndarray | None = None
    range_block_centers: np.ndarray | None = None
    blocks_num: int | None = None
    azimuth_start_time: PreciseDateTime | None = None
    profiles: np.ndarray | None = None
    look_angles: np.ndarray | None = None
    incidence_angles: np.ndarray | None = None
    block_azimuth_times: np.ndarray | None = None
    hist_2d: np.ndarray | None = None
    hist_x_bins_axis: np.ndarray | None = None
    hist_y_bins_axis: np.ndarray | None = None


@dataclass
class PointWiseRadiometricAnalysisOutput:
    """Dataclass to collect generic output from Radiometric Analysis"""

    swath: str | None = None
    burst: int | None = None
    channel: str | int | None = None
    polarization: SARPolarization | None = None
    projection: SARProjection | None = None
    original_profile_db: np.ndarray | None = None
    smoothed_profile_db: np.ndarray | None = None
    axis: np.ndarray | None = None
    time: float | str | np.ndarray | None = None
    direction: RadiometricAnalysisDirection | None = None
    value_type: RadiometricAnalysisValue | None = None
    axis_type: RadiometricAnalysisAxes | None = None
    radiometric_quantity: SARRadiometricQuantity | None = None
