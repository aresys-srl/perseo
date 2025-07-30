# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Custom errors definition for better troubleshooting"""


class AzimuthExceedsBoundariesError(ValueError):
    """Selected Azimuth index in ROI extraction from Swath exceeds boundaries"""


class RangeExceedsBoundariesError(ValueError):
    """Selected Range index in ROI extraction from Swath exceeds boundaries"""


class SideLobesDirectionsEstimationError(RuntimeError):
    """Could not evaluate the side lobes directions values"""


class CoordinatesOutOfBounds(RuntimeError):
    """Input pixel/time coordinate is out of swath bounds"""


class TargetAreaRecenteringError(RuntimeError):
    """Error raised when recentering operation on input data fails"""


class InvalidBurstIdError(ValueError):
    """Provided burst id is invalid"""


known_errors = [AzimuthExceedsBoundariesError, RangeExceedsBoundariesError]
