# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Definition of Interferometric Coherence Analysis specific dataclasses"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt

from perseo_quality.core.generic_dataclasses import SARPolarization


class CoherenceGraphMode(Enum):
    """Coherence graphs complex coherence plot method"""

    MAGNITUDE = "magnitude"
    PHASE = "phase"


@dataclass
class InterferometricCoherenceOutput:
    """Interferometric Coherence computation output"""

    channel_name: str
    swath: str
    burst: int
    polarization: SARPolarization
    coherence: npt.NDArray[np.floating]
    coherence_histograms: InterferometricCoherence2DHistograms


@dataclass
class InterferometricCoherence2DHistograms:
    """Interferometric Coherence 2D histograms output"""

    coherence_bin_edges: npt.NDArray[np.floating]
    azimuth_histogram: npt.NDArray[np.floating]
    range_histogram: npt.NDArray[np.floating]
