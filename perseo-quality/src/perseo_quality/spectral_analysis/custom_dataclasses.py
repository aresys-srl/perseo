# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Definition of Spectral Analysis specific dataclasses"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from arepytools.timing.precisedatetime import PreciseDateTime
from numpy.polynomial import Polynomial


@dataclass
class SpectralAnalysisProductGeneralInfo:
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
class PointTargetSpectraDataOutput:
    """Storing data for Point Target Spectral Analysis graphs"""

    general_info: SpectralAnalysisProductGeneralInfo | None = None
    targets_info: list[SpectralAnalysisTargetInfo] | None = None


@dataclass
class DistributedSpectraDataOutput:
    """Storing data for Distributed Spectral Analysis graphs"""

    general_info: SpectralAnalysisProductGeneralInfo | None = None
    blocks_info: list[SpectralAnalysisBlockInfo] | None = None


@dataclass
class SpectralAnalysisTargetInfo:
    """Storing data for Point TargetSpectral Analysis graphs"""

    target_name: str | None = None
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
    range_profiles_norm_deg: list[np.ndarray] | None = None
    azimuth_profiles_deg: list[np.ndarray] | None = None
    azimuth_profiles_norm_deg: list[np.ndarray] | None = None
    target_phase_value_deg: float | None = None
    target_doppler_centroid_Hz: float | None = None
    range_polynomial_fit: Polynomial | None = None
    azimuth_polynomial_fit: Polynomial | None = None
    rng_spectrum_boundaries: tuple[float, float] | None = None
    az_spectrum_boundaries: tuple[float, float] | None = None


@dataclass
class SpectralAnalysisBlockInfo:
    """Block-wise Spectral Analysis info"""

    block_num: int | None = None
    first_az_line_block: int | None = None
    lines_block: int | None = None
    samples_block: int | None = None
    doppler_centroid_mid_block: float | None = None
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
