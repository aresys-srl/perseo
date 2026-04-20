# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Models - Custom Types
---------------------
"""

from __future__ import annotations

import numpy as np
from numpydantic import NDArray, Shape

from perseo_core.timing.precise_datetime import PreciseDateTime

ExtendedDatetimeType = PreciseDateTime | np.datetime64
ExtendedDatetimeArrayType = NDArray[Shape["*"], ExtendedDatetimeType]  # type: ignore
CoordinatesArrayType = NDArray[Shape["* x, * y, * z"], float] | NDArray[Shape["3"], float]  # type: ignore
FloatArrayType = NDArray[Shape["*"], float]  # type: ignore
