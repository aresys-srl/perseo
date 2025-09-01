# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Spectral analysis testing script"""

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
from perseo_quality.spectral_analysis.analysis import (
    point_target_spectral_analysis,
)
from perseo_quality.spectral_analysis.graphical_output import spectral_graphs


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
