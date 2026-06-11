# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Point and Distributed Target Ambiguity Ratio analysis testing script"""

import logging
import sys
from pathlib import Path

from sct_aresys_reader.protocol_implementation import ProductFolderManagerExtended

from perseo_quality.logger import quality_logger
from perseo_quality.tar_analysis.analysis import (
    distributed_target_ambiguity_ratio_analysis,
    point_target_ambiguity_ratio_analysis,
)
from perseo_quality.tar_analysis.config import (
    AmbiguityRatioConfig,
)
from perseo_quality.tar_analysis.graphical_output import (
    ambiguities_graphs,
)
from sct.io.point_target_manager import extract_point_target_data_from_source, convert_df_to_nominal_point_target



if __name__ == "__main__":
    # setup custom logger
    quality_logger.addHandler(logging.StreamHandler(sys.stdout))

    product_path = r"C:\Users\giorgio.parma\Aresys_DATA\quality_data\demo_topsar\SLC"
    point_targets_file = r"C:\Users\giorgio.parma\Aresys_DATA\quality_data\demo_topsar\demo_topsar_pt_dataset.csv"
    output_dir = r"C:\Users\giorgio.parma\Desktop\temporary_outputs"

    pt = extract_point_target_data_from_source(source=point_targets_file)
    pt_list = convert_df_to_nominal_point_target(data_df=pt)
    product = ProductFolderManagerExtended(product_path)

    print("")
    quality_logger.info("PTAR")
    print("")
    # PTAR
    pt_ambiguities = point_target_ambiguity_ratio_analysis(product=product, point_targets=pt_list)
    quality_logger.info("PTAR graphs generation...")
    ambiguities_graphs(data=pt_ambiguities, output_dir=output_dir)

    print("")
    quality_logger.info("DTAR")
    print("")
    # DTAR
    rois = [(9829, 875)]
    config = AmbiguityRatioConfig(cropping_size=(828, 74))
    dist_ambiguities = distributed_target_ambiguity_ratio_analysis(product=product, roi_centers=rois, config=config)
    quality_logger.info("DTAR graphs generation...")
    ambiguities_graphs(data=dist_ambiguities, output_dir=output_dir)
