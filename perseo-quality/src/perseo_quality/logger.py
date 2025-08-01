# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
PERSEO Quality: Package logger
------------------------------
"""

import logging

quality_logger = logging.getLogger("perseo-quality")
quality_logger.addHandler(logging.NullHandler())
quality_logger.setLevel(logging.DEBUG)
