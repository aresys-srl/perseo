# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Definition of Elevation Notch Analysis specific dataclasses"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from perseo_quality.core.generic_dataclasses import SARPolarization


@dataclass
class ElevationNotchOutput:
    product_name: str | None = None
    channel: str | None = None
    swath: str | None = None
    polarization: SARPolarization | None = None
    blocks_info: list[ElevationNotchBlockInfo] | None = None


@dataclass
class ElevationNotchBlockInfo:
    block_num: int | None = None
    first_az_line_block: int | None = None
    lines_block: int | None = None
    samples_block: int | None = None
    altitude_m: float | None = None
    annotated_roll_deg: float | None = None
    estimated_roll_deg: float | None = None
    antenna_profile_from_data_db: np.ndarray | None = None
    antenna_profile_from_model_db: np.ndarray | None = None
    antenna_profile_parabolic_fit_db: np.ndarray | None = None
    parabolic_fit_axis_deg: np.ndarray | None = None
    parabola_minimum_deg: float | None = None
    parabola_coefficients: np.ndarray | None = None
    antenna_angles_deg: np.ndarray | None = None
    mispointing_error_deg: float | None = None
    calibration_constant: float | None = None
    noise_floor: float | None = None
    notch_minimum_position_deg: float | None = None
