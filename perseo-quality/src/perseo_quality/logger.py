# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Quality Logger."""

import logging

quality_logger = logging.getLogger("perseo-quality")
quality_logger.addHandler(logging.NullHandler())
quality_logger.setLevel(logging.DEBUG)
