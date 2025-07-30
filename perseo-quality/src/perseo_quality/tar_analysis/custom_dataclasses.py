# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Definition of Target Ambiguity Analysis specific dataclasses"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from perseo_quality.core.generic_dataclasses import SARPolarization


@dataclass
class AmbiguityRatioOutput:
    """Output results for single Point or Distributed Target Ambiguity Ratio Analysis"""

    target_name: str | None = None
    product_name: str | None = None
    channel: str | None = None
    swath: str | None = None
    polarization: SARPolarization | None = None
    burst: int | None = None
    roi_size_azimuth: int | None = None
    roi_size_range: int | None = None
    target_nominal_coordinates: np.ndarray | None = None
    target_azimuth_pixel: float | None = None
    target_range_pixel: float | None = None
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
