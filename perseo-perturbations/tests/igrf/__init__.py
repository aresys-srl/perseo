# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
IGRF tests
-----------
"""

from .test_conversions import TestConversions
from .test_igrf import TestMainEdgeCases, TestNonRegression, TestPrepareCoefficients, TestValidateDegreeRange
from .test_shc_reader import TestShcReader
from .test_utils import TestGetInclinationDeclination, TestGetLegendre

__all__ = [
    "TestConversions",
    "TestMainEdgeCases",
    "TestPrepareCoefficients",
    "TestValidateDegreeRange",
    "TestNonRegression",
    "TestShcReader",
    "TestGetLegendre",
    "TestGetInclinationDeclination",
]
