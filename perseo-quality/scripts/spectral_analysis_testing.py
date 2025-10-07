# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Spectral analysis testing script"""

import logging
import sys
from pathlib import Path

from sct_aresys_reader.protocol_implementation import ProductFolderManagerExtended

from perseo_quality.io.point_targets import PointTarget
from perseo_quality.logger import quality_logger
from perseo_quality.spectral_analysis.analysis import (
    block_wise_distributed_spectral_analysis,
    point_target_spectral_analysis,
)
from perseo_quality.spectral_analysis.graphical_output import spectral_graphs

from arepytools.io.point_target_binary import PointSetProduct

if __name__ == "__main__":
    # setup custom logger
    quality_logger.addHandler(logging.StreamHandler(sys.stdout))

    product_path = r"C:\Users\giorgio.parma\Aresys_DATA\quality_data\demo_topsar\SLC"
    point_targets_file = r"C:\Users\giorgio.parma\Aresys_DATA\quality_data\demo_topsar\demo_topsar_point_targets_binary"
    output_dir = r"C:\Users\giorgio.parma\Desktop\temporary_outputs\test"

    pts = PointSetProduct(point_targets_file)
    coords, rcs = pts.read_data()
    pt_list = []
    for pt in range(coords.shape[0]):
        pt_list.append(PointTarget(
            delay=0,
            name=f"pt{pt}",
            xyz_coordinates=coords[pt, :],
            rcs_hh=rcs[pt, 0],
            rcs_hv=rcs[pt, 1],
            rcs_vh=rcs[pt, 2],
            rcs_vv=rcs[pt, 3],
        ))

    product = ProductFolderManagerExtended(product_path)

    print("")
    quality_logger.info("PT Spectral Analysis")
    print("")

    spectral_data = point_target_spectral_analysis(
        product=product, point_targets=pt_list, cropping_size=(256, 256)
    )
    spectral_graphs(data=spectral_data, output_dir=output_dir)

    # print("")
    # print("Distributed Spectral Analysis")
    # print("")

    # product = ProductFolderManagerExtended(product_path)

    # spectral_data =block_wise_distributed_spectral_analysis(product=product)
    # spectral_graphs(data=spectral_data, output_dir=output_dir)
