# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Definition of ENL specific dataclasses"""

from __future__ import annotations

from dataclasses import dataclass

from perseo_quality.core.generic_dataclasses import SARPolarization


@dataclass
class ENLOutput:
    """Output results for Equivalent Number of Looks analysis"""

    product_name: str | None = None
    channel: str | None = None
    swath: str | None = None
    polarization: SARPolarization | None = None
    roi_center: tuple[int, int] | None = None
    roi_size_azimuth: int | None = None
    roi_size_range: int | None = None
