# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Block-Wise Radiometric Analysis: NESZ, Average Radiometric Profiles, Scalloping"""

from __future__ import annotations

import logging

import perseo_quality.core.generic_dataclasses as gdt
import perseo_quality.radiometric_analysis.custom_dataclasses as rdt
from perseo_quality.io.quality_input_protocol import QualityInputProduct
from perseo_quality.radiometric_analysis.block_wise.config import RadiometricProfilesConfig
from perseo_quality.radiometric_analysis.block_wise.core.profile_extractors import (
    average_elevation_profiles_extractor,
    nesz_profiles_extractor,
    scalloping_profiles_extractor,
)
from perseo_quality.radiometric_analysis.block_wise.core.radiometric_profiles import radiometric_profiles

# syncing with logger
log = logging.getLogger("quality_analysis")


def nesz_profiles(
    product: QualityInputProduct,
    output_quantity: gdt.SARRadiometricQuantity = gdt.SARRadiometricQuantity.SIGMA_NOUGHT,
    config: RadiometricProfilesConfig | None = None,
) -> list[rdt.RadiometricProfilesOutput]:
    """Noise Equivalent Sigma-Zero (NESZ) radiometric profiles computation. Profiles along RANGE direction.

    Parameters
    ----------
    product : QualityInputProduct
        object containing product information and data satisfying the QualityInputProduct protocol
    output_quantity : gdt.SARRadiometricQuantity, optional
        desired radiometric output quantity, by default gdt.SARRadiometricQuantity.SIGMA_NOUGHT
    config : RadiometricProfilesConfig | None, optional
        RadiometricProfiles configuration, by default None

    Returns
    -------
    list[rdt.RadiometricProfilesOutput]
        a RadiometricProfilesOutput dataclass for each channel
    """
    config = _nesz_config_manager(config=config)

    log.info(f"Performing NESZ Analysis on {product.name}")

    return radiometric_profiles(
        product=product,
        direction=rdt.RadiometricAnalysisDirection.RANGE,
        profile_extractor_func=nesz_profiles_extractor,
        output_quantity=output_quantity,
        config=config,
    )


def average_elevation_profiles(
    product: QualityInputProduct,
    output_quantity: gdt.SARRadiometricQuantity,
    config: RadiometricProfilesConfig | None = None,
) -> list[rdt.RadiometricProfilesOutput]:
    """Average elevation radiometric profiles computation. Profiles along RANGE direction.

    Parameters
    ----------
    product : QualityInputProduct
        object containing product information and data satisfying the QualityInputProduct protocol
    output_quantity : gdt.SARRadiometricQuantity
        desired radiometric output quantity
    config : RadiometricProfilesConfig | None, optional
        RadiometricProfiles configuration, by default None

    Returns
    -------
    list[rdt.RadiometricProfilesOutput]
        a RadiometricProfilesOutput dataclass for each channel
    """
    config = _average_elevation_config_manager(config=config)

    log.info(f"Performing Average Elevation Profiles Analysis on {product.name}")
    log.info(f"Requested output radiometric quantity is: {output_quantity.name}")

    return radiometric_profiles(
        product=product,
        direction=rdt.RadiometricAnalysisDirection.RANGE,
        profile_extractor_func=average_elevation_profiles_extractor,
        output_quantity=output_quantity,
        config=config,
    )


def scalloping_profiles(
    product: QualityInputProduct,
    output_quantity: gdt.SARRadiometricQuantity = gdt.SARRadiometricQuantity.GAMMA_NOUGHT,
    config: RadiometricProfilesConfig | None = None,
) -> list[rdt.RadiometricProfilesOutput]:
    """Scalloping radiometric profiles computation. Profiles along AZIMUTH direction.

    Parameters
    ----------
    product : QualityInputProduct
        object containing product information and data satisfying the QualityInputProduct protocol
    output_quantity : gdt.SARRadiometricQuantity, optional
        desired radiometric output quantity, by default gdt.SARRadiometricQuantity.GAMMA_NOUGHT
    config : RadiometricProfilesConfig | None, optional
        RadiometricProfiles configuration, by default None

    Returns
    -------
    list[rdt.RadiometricProfilesOutput]
        a RadiometricProfilesOutput dataclass for each channel
    """
    config = _scalloping_config_manager(config=config)

    log.info(f"Performing Scalloping Profiles Analysis on {product.name}")

    return radiometric_profiles(
        product=product,
        direction=rdt.RadiometricAnalysisDirection.AZIMUTH,
        output_quantity=output_quantity,
        profile_extractor_func=scalloping_profiles_extractor,
        config=config,
    )


def _nesz_config_manager(config: RadiometricProfilesConfig | None) -> RadiometricProfilesConfig:
    """Initializing default NESZ Profiles config if None is provided and check that histogram parameters are not None.

    Parameters
    ----------
    config : RadiometricProfilesConfig | None
        input NESZ configuration

    Returns
    -------
    RadiometricProfilesConfig
        updated/default/checked NESZ configuration
    """
    if config is None:
        log.info("Configuration not provided. Using default NESZ Profiles configuration...")
        config = RadiometricProfilesConfig()

    if config.profile_extraction_parameters.filtering_kernel_size is None:
        config.profile_extraction_parameters.filtering_kernel_size = (7, 7)

    if config.histogram_parameters.x_bins_step is None:
        config.histogram_parameters.x_bins_step = 5

    if config.histogram_parameters.y_bins_num is None:
        config.histogram_parameters.y_bins_num = 301

    if config.histogram_parameters.y_bins_center_margin is None:
        config.histogram_parameters.y_bins_center_margin = 20

    return config


def _average_elevation_config_manager(config: RadiometricProfilesConfig | None) -> RadiometricProfilesConfig:
    """Initializing default Average Elevation Profiles config if None is provided and check that histogram parameters
    are not None.

    Parameters
    ----------
    config : RadiometricProfilesConfig | None
        input average elevation profiles configuration

    Returns
    -------
    RadiometricProfilesConfig
        updated/default/checked average elevation profiles configuration
    """
    if config is None:
        log.info("Configuration not provided. Using default Average Elevation Profiles configuration...")
        config = RadiometricProfilesConfig()

    if config.profile_extraction_parameters.filtering_kernel_size is None:
        config.profile_extraction_parameters.filtering_kernel_size = (11, 11)

    if config.histogram_parameters.x_bins_step is None:
        config.histogram_parameters.x_bins_step = 5

    if config.histogram_parameters.y_bins_num is None:
        config.histogram_parameters.y_bins_num = 101

    if config.histogram_parameters.y_bins_center_margin is None:
        config.histogram_parameters.y_bins_center_margin = 3

    return config


def _scalloping_config_manager(config: RadiometricProfilesConfig | None) -> RadiometricProfilesConfig:
    """Initializing default Scalloping Profiles config.

    Parameters
    ----------
    config : RadiometricProfilesConfig | None
        input Scalloping configuration

    Returns
    -------
    RadiometricProfilesConfig
        updated/default/checked Scalloping configuration
    """
    if config is None:
        log.info("Configuration not provided. Using default Scalloping Profiles configuration...")
        config = RadiometricProfilesConfig()

    if config.profile_extraction_parameters.filtering_kernel_size is None:
        config.profile_extraction_parameters.filtering_kernel_size = (11, 11)

    if config.histogram_parameters.x_bins_step is None:
        config.histogram_parameters.x_bins_step = 3

    if config.histogram_parameters.y_bins_num is None:
        config.histogram_parameters.y_bins_num = 51

    if config.histogram_parameters.y_bins_center_margin is None:
        config.histogram_parameters.y_bins_center_margin = 1.5

    return config
