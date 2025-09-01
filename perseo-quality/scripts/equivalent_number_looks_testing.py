# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Secondary analyses testing script"""

import logging
import sys

from sct_aresys_reader.protocol_implementation import ProductFolderManagerExtended

from perseo_quality.enl_analysis.analysis import (
    equivalent_number_of_looks_analysis,
)
from perseo_quality.logger import quality_logger

if __name__ == "__main__":
    # setup custom logger
    quality_logger.addHandler(logging.StreamHandler(sys.stdout))

    pf_path = r"C:\Users\giorgio.parma\Aresys_DATA\quality_data\demo_topsar\SLC"
    product = ProductFolderManagerExtended(pf_path)

    # ENL
    crop = [2528, 164]
    roi_centers = [(13129 + crop[0] // 2, 423 + crop[1] // 2)]
    enl_results = equivalent_number_of_looks_analysis(product=product, roi_centers=roi_centers, cropping_size=crop)
