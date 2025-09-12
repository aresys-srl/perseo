# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Spectral analysis testing script"""

import logging
import sys
from pathlib import Path

from sct_aresys_reader.protocol_implementation import ProductFolderManagerExtended

from perseo_quality.logger import quality_logger
from perseo_quality.spectral_analysis.analysis import (
    point_target_spectral_analysis,
)
from perseo_quality.spectral_analysis.graphical_output import spectral_graphs


if __name__ == "__main__":
    # setup custom logger
    quality_logger.addHandler(logging.StreamHandler(sys.stdout))

    product_path = r"C:\Users\giorgio.parma\Aresys_DATA\quality_data\demo_topsar\SLC"
    point_targets_file = r"C:\Users\giorgio.parma\Aresys_DATA\quality_data\demo_topsar\PointTargetsBinary"
    output_dir = r"C:\Users\giorgio.parma\Desktop\temporary_outputs"

    point_targets_locations = ...
    product = ProductFolderManagerExtended(product_path)

    print("")
    quality_logger.info("PT Spectral Analysis")
    print("")

    spectral_data = point_target_spectral_analysis(
        product=product, point_targets=point_targets_locations, cropping_size=(256, 256)
    )
    spectral_graphs(data=spectral_data, output_dir=output_dir, graph_type="POINT")

    # print("")
    # log.info("Distributed Spectral Analysis")
    # print("")

    # crop = (2661, 112)  # rng, az
    # rois = [(10632 + crop[0] // 2, 2414 + crop[1] // 2)]  # rng, az
    # spectral_data = distributed_target_spectral_analysis(product=product, roi_centers=rois, cropping_size=crop)
    # spectral_graphs(data=spectral_data, output_dir=output_dir, graph_type="DISTRIBUTED")
