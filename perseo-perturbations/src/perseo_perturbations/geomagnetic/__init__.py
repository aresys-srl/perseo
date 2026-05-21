# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
IGRF - Geomagnetic Reference Field Computation v14
--------------------------------------------------

The code is based on the ``IAGA-VMOD/ppigrf: 2.0.0`` https://github.com/IAGA-VMOD/ppigrf/tree/main project by
Karl M. Laundal, Santiago Soler, Ashley Smith, Andreas S. Skeidsvoll and Daniel Billett
(https://doi.org/10.5281/zenodo.14231854)
"""

from perseo_perturbations.geomagnetic.core.utils import get_magnetic_declination, get_magnetic_inclination
from perseo_perturbations.geomagnetic.igrf import get_geocentric_igrf, get_geocentric_igrf_potential, get_geodetic_igrf

__all__ = [
    "get_geocentric_igrf",
    "get_geocentric_igrf_potential",
    "get_geodetic_igrf",
    "get_magnetic_declination",
    "get_magnetic_inclination",
]
