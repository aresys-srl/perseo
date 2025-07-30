# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Definition of Target Ambiguity Ratio Analysis configuration"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass
class AmbiguityRatioConfig:
    """Configuration for Target Ambiguity Ratio computation"""

    interpolation_factor: int = 8
    cropping_size: tuple[int, int] = (128, 128)  # (number of samples, number of lines)

    @classmethod
    def from_dict(cls, arg: dict) -> AmbiguityRatioConfig:
        """Creating a AmbiguityRatioConfig object by conversion from a dictionary.

        Args:
            arg (dict): dictionary with keys equal to the AmbiguityRatioConfig ones

        Returns:
            AmbiguityRatioConfig: AmbiguityRatioConfig object
        """
        ar_obj = cls()
        dict_in = arg.copy()

        try:
            dtc_fields = [f.name for f in fields(ar_obj)]
            for key, value in dict_in.items():
                if key in dtc_fields:
                    if isinstance(value, list):
                        setattr(ar_obj, key, tuple(value))
                    else:
                        setattr(ar_obj, key, value)

            return ar_obj

        except Exception as err:
            raise ValueError("Invalid dictionary structure.") from err
