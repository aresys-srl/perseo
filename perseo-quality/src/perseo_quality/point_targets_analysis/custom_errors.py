# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Definition of custom errors for troubleshooting of Point Target Analysis"""


class PointTargetComputationError(RuntimeError):
    """Error raised while computing Point Target Analysis"""
