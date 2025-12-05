# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Definition of Elevation Notch Analysis configuration"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass
class ElevationNotchConfig:
    """Configuration for Elevation Notch Analysis"""

    azimuth_block_size: int = 5000
    range_pixel_margin: int = 150

    @classmethod
    def from_dict(cls, arg: dict) -> ElevationNotchConfig:
        """Creating a ElevationNotchConfig object by conversion from a dictionary.

        Args:
            arg (dict): dictionary with keys equal to the ElevationNotchConfig ones

        Returns:
            ElevationNotchConfig: ElevationNotchConfig object
        """
        ar_obj = cls()
        dict_in = arg.copy()

        try:
            dtc_fields = [f.name for f in fields(ar_obj)]
            for key, value in dict_in.items():
                if key in dtc_fields:
                    setattr(ar_obj, key, value)

            return ar_obj

        except Exception as err:
            raise ValueError("Invalid dictionary structure.") from err
