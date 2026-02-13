# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Geocoding errors definitions for troubleshooting"""


class AmbiguousInputCorrelation(RuntimeError):
    """Ambiguous correlation between input init times number and number of ground points. Operation not supported."""


class NewtonMethodConvergenceError(RuntimeError):
    """Newton method could not converge to a root solution."""


class OrbitsNotOverlappedError(RuntimeError):
    """Orbits provided for bistatic inverse geocoding are not overlapped"""


class EmptyEllipsoidIntersection(RuntimeError):
    """Ellipsoid intersection cannot be found"""
