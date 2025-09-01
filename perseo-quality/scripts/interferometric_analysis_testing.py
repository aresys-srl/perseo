# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Script to test the interferometric analysis functionalities"""

import logging
import sys

from sct_aresys_reader.protocol_implementation import ProductFolderManagerExtended

from perseo_quality.interferometric_analysis.analysis import (
    interferometric_analysis,
)
from perseo_quality.interferometric_analysis.config import InterferometricConfig
from perseo_quality.interferometric_analysis.graphical_output import (
    generate_coherence_graphs,
)
from perseo_quality.interferometric_analysis.support import (
    coherence_histograms_to_netcdf,
)
from perseo_quality.logger import quality_logger

if __name__ == "__main__":
    # setup custom logger
    quality_logger.addHandler(logging.StreamHandler(sys.stdout))

    out_dir = r"C:\tdir"

    prd_path_1 = r"C:\Users\giorgio.parma\Aresys_DATA\quality_data\interferometry\S1A_IW_SLC__0S_DV_20210611T053528_PF"
    # prd_path_1 = r"C:\Users\giorgio.parma\Aresys_DATA\quality_data\interferometry\S1B_IW_SLC__0S_DV_20210605T053455_PF_Cor_SCI_CM"  # noqa: E501
    product_1 = ProductFolderManagerExtended(path=prd_path_1)

    prd_path_2 = (
        r"C:\Users\giorgio.parma\Aresys_DATA\quality_data\interferometry\S1B_IW_SLC__0S_DV_20210605T053455_PF_Cor"
    )
    product_2 = ProductFolderManagerExtended(path=prd_path_2)
    # product_2 = None

    config = InterferometricConfig()
    config.enable_coherence_computation = True
    # config.coherence_kernel = 5

    output = interferometric_analysis(product=product_1, second_product=product_2, config=config)
    for out in output:
        generate_coherence_graphs(out, output_dir=out_dir, mode="magnitude")
        generate_coherence_graphs(out, output_dir=out_dir, mode="phase")
        coherence_histograms_to_netcdf(out, output_dir=out_dir)
