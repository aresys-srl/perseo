# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Definition of Spectral Analysis specific dataclasses"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arepytools.timing.precisedatetime import PreciseDateTime
from numpy.polynomial import Polynomial

from perseo_quality.core.generic_dataclasses import SARPolarization


@dataclass
class SpectraDataOutput:
    """Storing data for Spectral Analysis graphs"""

    target_name: str | None = None
    product_name: str | None = None
    channel: str | None = None
    swath: str | None = None
    polarization: SARPolarization | None = None
    burst: int | None = None
    roi_size_azimuth: int | None = None
    roi_size_range: int | None = None
    target_azimuth_pixel: float | None = None
    target_range_pixel: float | None = None
    azimuth_time: PreciseDateTime | None = None
    range_time: float | None = None
    azimuth_frequency_axis: np.ndarray | None = None
    range_frequency_axis: np.ndarray | None = None
    spectrum_db: np.ndarray | None = None
    spectrum_deg: np.ndarray | None = None
    spectrogram_db: np.ndarray | None = None
    spectrogram_frequencies: np.ndarray | None = None
    spectrogram_times: np.ndarray | None = None
    range_profiles_db: list[np.ndarray] | None = None
    azimuth_profiles_db: list[np.ndarray] | None = None
    range_profiles_deg: list[np.ndarray] | None = None
    azimuth_profiles_deg: list[np.ndarray] | None = None
    range_polynomial_fit: Polynomial | None = None
    azimuth_polynomial_fit: Polynomial | None = None
