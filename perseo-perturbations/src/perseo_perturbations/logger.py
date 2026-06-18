# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""PERSEO Perturbations: Package logger."""

import logging

perturb_logger = logging.getLogger("perseo-perturbations")
perturb_logger.addHandler(logging.NullHandler())
perturb_logger.setLevel(logging.DEBUG)
