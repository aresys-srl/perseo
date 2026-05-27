# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Definition of Radiometric Analysis specific dataclasses"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto

import numpy as np
import numpy.typing as npt
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

    general_info: RadiometricOutputProductGeneralInfo | None = None
    direction: RadiometricAnalysisDirection | None = None
    kpi: list[AverageElevationRadiometricKPI] | list[NESZRadiometricKPI] | list[ScallopingRadiometricKPI] | None = None
    azimuth_block_centers: npt.NDArray[np.floating] | None = None
    range_block_centers: npt.NDArray[np.floating] | None = None
    blocks_num: int | None = None
    azimuth_start_time: PreciseDateTime | None = None
    profiles: npt.NDArray[np.floating] | None = None
    noise_vectors: npt.NDArray[np.floating] | None = None
    look_angles: npt.NDArray[np.floating] | None = None
    incidence_angles: npt.NDArray[np.floating] | None = None
    block_azimuth_times: np.ndarray | None = None
    hist_2d: npt.NDArray[np.floating] | None = None
    hist_x_bins_axis: npt.NDArray[np.floating] | None = None
    hist_y_bins_axis: npt.NDArray[np.floating] | None = None


@dataclass
class PointWiseRadiometricAnalysisOutput:
    """Dataclass to collect generic output from Radiometric Analysis"""

    swath: str | None = None
    burst: int | None = None
    channel: str | int | None = None
    polarization: SARPolarization | None = None
    projection: SARProjection | None = None
    original_profile_db: npt.NDArray[np.floating] | None = None
    smoothed_profile_db: npt.NDArray[np.floating] | None = None
    axis: npt.NDArray[np.floating] | None = None
    time: float | str | np.ndarray | None = None
    direction: RadiometricAnalysisDirection | None = None
    value_type: RadiometricAnalysisValue | None = None
    axis_type: RadiometricAnalysisAxes | None = None
    radiometric_quantity: SARRadiometricQuantity | None = None


@dataclass
class BaseRadiometricKPI:
    """Basic Radiometric Block info"""

    block_num: int | None = None
    first_az_line_block: int | None = None
    lines_block: int | None = None


@dataclass
class AverageElevationRadiometricKPI(BaseRadiometricKPI):
    """Block-wise Radiometry Block info"""

    mid_incidence_angle_deg: float | None = None
    mid_look_angle_deg: float | None = None
    enl_block: float | None = None
    mean_level_db: float | None = None
    std_level_db: float | None = None
    slope_wrt_look_angle_db_deg: float | None = None
    variability_index_db: float | None = None


@dataclass
class NESZRadiometricKPI(BaseRadiometricKPI):
    """NESZ Radiometric Block info"""

    min_nesz_profile_db: float | None = None
    min_nesz_incidence_angle_deg: float | None = None
    min_nesz_range_position: float | None = None
    min_nesz_look_angle_deg: float | None = None
    max_nesz_profile_db: float | None = None
    max_nesz_incidence_angle_deg: float | None = None
    max_nesz_range_position: float | None = None
    max_nesz_look_angle_deg: float | None = None
    mean_nesz_profile_dB: float | None = None
    std_nesz_profile_dB: float | None = None
    mode_nesz_profile_dB: float | None = None
    skewness_profile: float | None = None
    kurtosis_profile: float | None = None
    mean_block_dB: float | None = None
    std_block_dB: float | None = None
    mode_block_dB: float | None = None
    skewness_block: float | None = None
    kurtosis_block: float | None = None


@dataclass
class ScallopingRadiometricKPI(BaseRadiometricKPI):
    """Scalloping Radiometric Block info"""

    mean_level_db: float | None = None
    min_level_db: float | None = None
    max_level_db: float | None = None
    std_level_db: float | None = None


@dataclass
class RadiometricOutputProductGeneralInfo:
    """Block-wise Radiometric analysis product related generic info"""

    product: str
    channel: str
    swath: str
    polarization: str
    sensor: str
    product_type: str
    acquisition_mode: str
    radiometric_quantity: str
    orbit_direction: str
    acquisition_start_time: datetime


@dataclass
class RadiometricProfileAxes:
    """Axes corresponding to the computed radiometric profile"""

    incidence_angles_deg: npt.NDArray[np.floating]
    look_angles_deg: npt.NDArray[np.floating]
    slant_range: npt.NDArray[np.floating]
    azimuth: np.ndarray
