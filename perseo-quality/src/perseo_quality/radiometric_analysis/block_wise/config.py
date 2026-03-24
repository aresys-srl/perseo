# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Definition of Block-Wise Radiometric Analysis configuration"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum, auto


@dataclass
class Radiometric2DHistogramParameters:
    """Radiometric 2D Histogram configuration parameters"""

    x_bins_step: int | None = None  # resampling step of x axis
    y_bins_num: int | None = None  # number of bins along y axis
    y_bins_center_margin: float | None = (
        None  # +- margin defining the extension of the y binning axis around y center value
    )

    @classmethod
    def from_dict(cls, arg: dict) -> Radiometric2DHistogramParameters:
        """Creating a Radiometric2DHistogramParameters object by conversion from a dictionary.

        Parameters
        ----------
        arg : dict
            dictionary with keys equal to the ProfileExtractionParameters ones

        Returns
        -------
        Radiometric2DHistogramParameters
            Radiometric2DHistogramParameters object

        Raises
        ------
        ValueError
            invalid dictionary structure
        """
        h_obj = cls()

        try:
            for fld in fields(cls):
                if fld.name in arg.keys():
                    setattr(h_obj, fld.name, arg[fld.name])

            return h_obj

        except Exception as err:
            raise ValueError("Invalid dictionary structure.") from err


class RiverMaskingMode(Enum):
    """Describe how river masking is applied
    DISABLED: no river masking is applied
    FULL: full river masking algorithm is applied, it may take a while
    FAST: faster but less accurate river masking is applied
    """

    DISABLED = auto()
    FULL = auto()
    FAST = auto()


@dataclass
class RiverMaskingConfig:
    """Tunable parameters for river masking algorithm"""

    river_masking_mode: RiverMaskingMode = RiverMaskingMode.DISABLED
    local_stats_window: int = 10
    backscatter_threshold_percentile: float = 25
    cv_lower_threshold_percentile: float = 20
    cv_upper_threshold_percentile: float = 90
    morph_opening_radius: int = 3
    min_river_area_px_percentile: float = 99
    region_grow_iterations: int = 13
    relaxed_backscatter_threshold_percentile: float = 35

    @classmethod
    def from_dict(cls, arg: dict) -> RiverMaskingConfig:
        """Creating a RiverMaskingConfig object by conversion from a dictionary.

        Parameters
        ----------
        arg : dict
            dictionary with keys equal to the RiverMaskingConfig ones

        Returns
        -------
        RiverMaskingConfig
            RiverMaskingConfig object

        Raises
        ------
        ValueError
            invalid dictionary structure
        """
        pf_obj = cls()

        try:
            for fld in fields(cls):
                if fld.name in arg.keys():
                    if fld.name == "river_masking_mode":
                        setattr(pf_obj, fld.name, RiverMaskingMode[arg[fld.name]])
                    else:
                        setattr(pf_obj, fld.name, arg[fld.name])

            return pf_obj

        except Exception as err:
            raise ValueError("Invalid dictionary structure.") from err


@dataclass
class ProfileExtractionParameters:
    """Dataclass to store configuration parameters for Radiometric Analysis functions"""

    outlier_removal: bool = False
    smoothening_filter: bool = False
    river_masking: RiverMaskingConfig = field(default_factory=RiverMaskingConfig)
    filtering_kernel_size: tuple[int, int] | None = None
    outliers_percentile_boundaries: tuple[int, int] = (20, 90)
    outliers_kernel_size: tuple[int, int] = (5, 5)

    @classmethod
    def from_dict(cls, arg: dict) -> ProfileExtractionParameters:
        """Creating a ProfileExtractionParameters object by conversion from a dictionary.

        Parameters
        ----------
        arg : dict
            dictionary with keys equal to the ProfileExtractionParameters ones

        Returns
        -------
        ProfileExtractionParameters
            ProfileExtractionParameters object

        Raises
        ------
        ValueError
            invalid dictionary structure
        """
        pf_obj = cls()

        try:
            for fld in fields(cls):
                if fld.name in arg.keys():
                    if fld.name == "river_masking":
                        pf_obj.river_masking = RiverMaskingConfig.from_dict(arg[fld.name])
                    elif isinstance(arg[fld.name], list):
                        setattr(pf_obj, fld.name, tuple(arg[fld.name]))
                    else:
                        setattr(pf_obj, fld.name, arg[fld.name])

            return pf_obj

        except Exception as err:
            raise ValueError("Invalid dictionary structure.") from err


@dataclass
class RadiometricProfilesConfig:
    """Radiometric Profiles configuration setup dataclass"""

    azimuth_block_size: int = 2000
    range_pixel_margin: int = 150
    histogram_parameters: Radiometric2DHistogramParameters = field(default_factory=Radiometric2DHistogramParameters)
    profile_extraction_parameters: ProfileExtractionParameters = field(default_factory=ProfileExtractionParameters)

    @classmethod
    def from_dict(cls, arg: dict) -> RadiometricProfilesConfig:
        """Creating a RadiometricProfilesConfig object by conversion from a dictionary.

        Parameters
        ----------
        arg : dict
            dictionary with keys equal to the RadiometricProfilesConfig ones

        Returns
        -------
        RadiometricProfilesConfig
            RadiometricProfilesConfig object

        Raises
        ------
        ValueError
            invalid dictionary structure
        """
        ra_obj = cls()

        try:
            if "histogram_parameters" in arg:
                ra_obj.histogram_parameters = Radiometric2DHistogramParameters.from_dict(arg["histogram_parameters"])

            if "profile_extraction_parameters" in arg:
                ra_obj.profile_extraction_parameters = ProfileExtractionParameters.from_dict(
                    arg["profile_extraction_parameters"]
                )

            if "azimuth_block_size" in arg:
                ra_obj.azimuth_block_size = arg["azimuth_block_size"]
            if "range_pixel_margin" in arg:
                ra_obj.range_pixel_margin = arg["range_pixel_margin"]

            return ra_obj

        except Exception as err:
            raise ValueError("Invalid dictionary structure.") from err
