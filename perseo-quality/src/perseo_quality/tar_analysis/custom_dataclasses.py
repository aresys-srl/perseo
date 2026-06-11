# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Definition of Target Ambiguity Analysis specific dataclasses"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass
class AmbiguityRatioProductGeneralInfo:
    """General info for the analyzed product"""

    product: str
    channel: str
    swath: str
    polarization: str
    sensor: str
    product_type: str
    acquisition_mode: str
    orbit_direction: str
    acquisition_start_time: datetime


@dataclass
class AmbiguityRatioCoreInfo:
    """Output core results for single Target Ambiguity Ratio Analysis"""

    burst: int | None = None
    roi_size_azimuth: int | None = None
    roi_size_range: int | None = None
    azimuth_time_delta: float | None = None
    range_time_delta: float | None = None
    left_ambiguity_azimuth_pixel: np.ndarray | None = None
    left_ambiguity_range_pixel: np.ndarray | None = None
    right_ambiguity_azimuth_pixel: np.ndarray | None = None
    right_ambiguity_range_pixel: np.ndarray | None = None
    ambiguity_ratio_db: float | None = None
    target_image: np.ndarray | None = None
    right_ambiguity_image: np.ndarray | None = None
    left_ambiguity_image: np.ndarray | None = None


@dataclass
class AmbiguityRatioTargetInfo(AmbiguityRatioCoreInfo):
    """Output results for single Point Target Ambiguity Ratio Analysis"""

    target_name: str | None = None
    target_nominal_coordinates: np.ndarray | None = None
    target_azimuth_pixel: float | None = None
    target_range_pixel: float | None = None


@dataclass
class AmbiguityRatioROIInfo(AmbiguityRatioCoreInfo):
    """Output results for single Distributed Target Ambiguity Ratio Analysis"""

    roi_name: str | None = None
    roi_center_ground_point_coordinates: np.ndarray | None = None
    roi_center_azimuth_pixel: float | None = None
    roi_center_range_pixel: float | None = None


@dataclass
class PointTargetAmbiguityRatioDataOutput:
    """Storing data for Point Target Ambiguity Ratio Analysis graphs"""

    general_info: AmbiguityRatioProductGeneralInfo | None = None
    targets_info: list[AmbiguityRatioTargetInfo] | None = None


@dataclass
class DistributedTargetAmbiguityRatioDataOutput:
    """Storing data for Distributed Target Ambiguity Ratio Analysis graphs"""

    general_info: AmbiguityRatioProductGeneralInfo | None = None
    roi_info: list[AmbiguityRatioROIInfo] | None = None
