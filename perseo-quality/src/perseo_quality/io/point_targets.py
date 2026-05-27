# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Definition of Point Targets required info and utilities for usage inside PERSEO Quality"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, field_validator


class PointTarget(BaseModel):
    """Point Target info needed by PERSEO-Quality analyses"""

    name: str | None = None
    xyz_coordinates: npt.NDArray[np.floating] | None = None
    rcs_hh: complex | float | None = None
    rcs_vv: complex | float | None = None
    rcs_vh: complex | float | None = None
    rcs_hv: complex | float | None = None
    delay: float | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("xyz_coordinates", mode="before")
    @classmethod
    def validate_coords(cls, coords: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        if coords.ndim != 1 and coords.size != 3:
            raise ValueError("xyz coordinates must be an array of floats with shape (3,)")
        return coords
