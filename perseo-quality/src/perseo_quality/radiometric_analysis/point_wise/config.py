# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Definition of Point-Wise Radiometric Analysis configuration"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

from perseo_quality.core.generic_dataclasses import SARRadiometricQuantity, convert_to_enum_field
from perseo_quality.radiometric_analysis.custom_dataclasses import (
    RadiometricAnalysisAxes,
    RadiometricAnalysisDirection,
    RadiometricAnalysisValue,
)


@dataclass
class RadiometricAnalysisParameters:
    """Dataclass to store configuration parameters for Radiometric Analysis functions"""

    smoothening_order: int = 3
    smoothening_window_length: int = 71
    outliers_kernel_size: tuple[int, int] = (5, 5)
    outliers_filter_kernel_size: tuple[int, int] = (10, 10)
    outliers_percentile_boundaries: tuple[int, int] = (20, 90)

    @staticmethod
    def from_dict(arg: dict) -> RadiometricAnalysisParameters:
        """Creating a RadiometricAnalysisParameters object by conversion from a dictionary.

        Parameters
        ----------
        arg : dict
            dictionary with keys equal to the RadiometricAnalysisParameters ones

        Returns
        -------
        RadiometricAnalysisParameters
            RadiometricAnalysisParameters object

        Raises
        ------
        ValueError
            invalid dictionary structure
        """
        rap_obj = RadiometricAnalysisParameters()

        try:
            for fld in fields(rap_obj):
                if fld.name in arg.keys():
                    if isinstance(arg[fld.name], list):
                        setattr(rap_obj, fld.name, tuple(arg[fld.name]))
                    else:
                        setattr(rap_obj, fld.name, arg[fld.name])

            return rap_obj

        except Exception as err:
            raise ValueError("Invalid dictionary structure.") from err


@dataclass
class PointWiseRadiometricAnalysisConfig:
    """Point Wise Radiometric Analysis configuration setup dataclass"""

    output_quantity: SARRadiometricQuantity = SARRadiometricQuantity.BETA_NOUGHT
    value: RadiometricAnalysisValue = RadiometricAnalysisValue.AMPLITUDE
    direction: RadiometricAnalysisDirection = RadiometricAnalysisDirection.RANGE
    axis: RadiometricAnalysisAxes = RadiometricAnalysisAxes.NATURAL
    az_average_lines: int = 1000
    rng_average_samples: int = 1000
    outlier_removal: bool = False
    smoothening_filter: bool = True
    parameters: RadiometricAnalysisParameters = field(default_factory=RadiometricAnalysisParameters)

    @staticmethod
    def from_dict(arg: dict) -> PointWiseRadiometricAnalysisConfig:
        """Creating a PointWiseRadiometricAnalysisConfig object by conversion from a dictionary.

        Parameters
        ----------
        arg : dict
            dictionary with keys equal to the PointWiseRadiometricAnalysisConfig ones

        Returns
        -------
        PointWiseRadiometricAnalysisConfig
            PointWiseRadiometricAnalysisConfig object

        Raises
        ------
        ValueError
            invalid dictionary structure
        """
        ra_obj = PointWiseRadiometricAnalysisConfig()

        try:
            if "parameters" in arg:
                ra_obj.parameters = RadiometricAnalysisParameters.from_dict(arg["parameters"])

            if "output_quantity" in arg:
                ra_obj.output_quantity = convert_to_enum_field(arg["output_quantity"], enum_type=SARRadiometricQuantity)
            if "value" in arg:
                ra_obj.value = convert_to_enum_field(arg["value"], enum_type=RadiometricAnalysisValue)
            if "direction" in arg:
                ra_obj.direction = convert_to_enum_field(arg["direction"], enum_type=RadiometricAnalysisDirection)
            if "axis" in arg:
                ra_obj.axis = convert_to_enum_field(arg["axis"], enum_type=RadiometricAnalysisAxes)
            if "outlier_removal" in arg:
                ra_obj.outlier_removal = arg["outlier_removal"]
            if "smoothening_filter" in arg:
                ra_obj.smoothening_filter = arg["smoothening_filter"]
            if "az_average_lines" in arg:
                ra_obj.az_average_lines = arg["az_average_lines"]
            if "rng_average_samples" in arg:
                ra_obj.rng_average_samples = arg["rng_average_samples"]

            return ra_obj

        except Exception as err:
            raise ValueError("Invalid dictionary structure.") from err
