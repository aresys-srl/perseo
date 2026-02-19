# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Spectral analysis testing script"""

import logging
import sys
from pathlib import Path

from sct_aresys_reader.protocol_implementation import ProductFolderManagerExtended
from sct_sentinel1_reader.protocol_implementation import Sentinel1ProductManager
from sct.io.point_target_manager import extract_point_target_data_from_source, convert_df_to_nominal_point_target
from perseo_quality.io.point_targets import PointTarget
from perseo_quality.logger import quality_logger
from perseo_quality.spectral_analysis.analysis import (
    block_wise_distributed_spectral_analysis,
    point_target_spectral_analysis,
)
from perseo_quality.spectral_analysis.graphical_output import spectral_graphs

from arepytools.io.point_target_binary import PointSetProduct

from perseo_quality.spectral_analysis.support import spectral_analysis_profiles_to_netcdf

if __name__ == "__main__":
    # setup custom logger
    quality_logger.addHandler(logging.StreamHandler(sys.stdout))

    product_path = r"C:\Users\giorgio.parma\Aresys_DATA\sct_data\sentinel1\SLC_PT_SPECTRAL.SAFE"
    point_targets_file = r"C:\Users\giorgio.parma\Aresys_DATA\sct_data\sentinel1\dlr_calibration_site_germany.csv"
    output_dir = r"C:\ARESYS_PROJ\perseo\perseo-quality\scripts\out"

    product = Sentinel1ProductManager(product_path)
    pt = extract_point_target_data_from_source(source=point_targets_file)
    pt_list = convert_df_to_nominal_point_target(data_df=pt)

    print("")
    quality_logger.info("PT Spectral Analysis")
    print("")

    spectral_data = point_target_spectral_analysis(
        product=product, point_targets=pt_list, cropping_size=(256, 256)
    )
    # spectral_data = block_wise_distributed_spectral_analysis(product=product)
    spectral_graphs(data=spectral_data, output_dir=output_dir)
    spectral_analysis_profiles_to_netcdf(data=spectral_data, out_path=output_dir)
