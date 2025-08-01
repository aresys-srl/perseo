# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Equivalent Number of Looks (ENL) Analysis"""

from __future__ import annotations

from perseo_quality.core.signal_processing import compute_equivalent_number_of_looks
from perseo_quality.enl_analysis.custom_dataclasses import ENLOutput
from perseo_quality.io.quality_input_protocol import QualityInputProduct
from perseo_quality.logger import quality_logger as log


def equivalent_number_of_looks_analysis(
    product: QualityInputProduct, roi_centers: list[tuple[int, int]], cropping_size: tuple[int, int]
) -> list[ENLOutput]:
    """Performing Equivalent Number of Looks Analysis on input product at each ROI center location.

    Parameters
    ----------
    product : QualityInputProduct
        object satisfying the QualityInputProduct protocol
    roi_centers : list[tuple[int, int]]
        list of ROI centers where to perform the ENL computation, (range pixel, azimuth pixel)
    cropping_size : tuple[int, int]
        size of the ROI to be extracted, (number of range samples, number of azimuth lines)

    Returns
    -------
    list[ENLOutput]
        Equivalent Number of Looks results for each product channel and each ROI of interest
    """

    log.info("Performing Equivalent Number of Looks (ENL) analysis on given ROIs.")

    output_results = []
    for channel in product.channels_list:
        channel_data = product.get_channel_data(channel_id=channel)
        log.info(
            f"Analyzing channel {channel}, swath {channel_data.swath_name} and "
            + f"polarization {channel_data.polarization.value}..."
        )
        for roi in roi_centers:
            roi_data = channel_data.read_data(
                azimuth_index=roi[1],
                range_index=roi[0],
                cropping_size=cropping_size,
                output_radiometric_quantity=channel_data.radiometric_quantity,
            )
            enl = compute_equivalent_number_of_looks(roi_data=roi_data)
            log.info(f"ENL for current ROI center {roi}, size {cropping_size}: {enl}")
            output_results.append(
                ENLOutput(
                    product_name=product.name,
                    channel=channel,
                    swath=channel_data.swath_name,
                    polarization=channel_data.polarization,
                    roi_center=roi,
                    roi_size_azimuth=cropping_size[1],
                    roi_size_range=cropping_size[0],
                )
            )

    return output_results
