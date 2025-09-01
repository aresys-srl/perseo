# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Point and Distributed Target Ambiguity Ratio analysis testing script"""

import logging
import sys
from pathlib import Path

from arepytools.io import (
    PointSetProduct,
    convert_array_to_point_target_structure,
    read_point_targets_file,
)
from arepytools.io.io_support import NominalPointTarget
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


def pt_loader(point_targets_path: str | Path) -> dict[str, NominalPointTarget]:
    """Loading point targets data from file.

    Parameters
    ----------
    point_targets_path : str | Path
        Path to point targets file or binary folder

    Returns
    -------
    dict[str, NominalPointTarget]
        point targets locations
    """

    point_targets_path = Path(point_targets_path)
    # loading point target information from xml file
    if point_targets_path.is_dir():
        coords, rcs = PointSetProduct(point_targets_path).read_data()
        return convert_array_to_point_target_structure(coords=coords, rcs=rcs)

    return read_point_targets_file(point_targets_path)


if __name__ == "__main__":
    # setup custom logger
    quality_logger.addHandler(logging.StreamHandler(sys.stdout))

    product_path = r"C:\Users\giorgio.parma\Aresys_DATA\quality_data\demo_topsar\SLC"
    point_targets_file = r"C:\Users\giorgio.parma\Aresys_DATA\quality_data\demo_topsar\PointTargetsBinary"
    output_dir = r"C:\Users\giorgio.parma\Desktop\temporary_outputs"

    point_targets_locations = pt_loader(point_targets_path=point_targets_file)
    product = ProductFolderManagerExtended(product_path)

    print("")
    quality_logger.info("PTAR")
    print("")
    # PTAR
    pt_ambiguities = point_target_ambiguity_ratio_analysis(product=product, point_targets=point_targets_locations)
    quality_logger.info("PTAR graphs generation...")
    ambiguities_graphs(data=pt_ambiguities, output_dir=output_dir, graph_type="PTAR")

    print("")
    quality_logger.info("DTAR")
    print("")
    # DTAR
    rois = [(9829, 875)]
    config = AmbiguityRatioConfig(cropping_size=(828, 74))
    dist_ambiguities = distributed_target_ambiguity_ratio_analysis(product=product, roi_centers=rois, config=config)
    quality_logger.info("DTAR graphs generation...")
    ambiguities_graphs(data=dist_ambiguities, output_dir=output_dir, graph_type="DTAR")
