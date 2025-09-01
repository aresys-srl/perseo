# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Testing coherence from interferogram products"""

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

    pf_path = r"C:\Users\giorgio.parma\Aresys_DATA\quality_data\interferometry\S1A_IW_SLC__1SDH_20240205T113611_20240205T113638_052423_065701_FDAB_PF_RORB_Cor_SCI_Mrg"  # noqa: E501
    out_dir = r"C:\Users\giorgio.parma\Desktop\temporary_outputs"

    config = InterferometricConfig()
    config.enable_coherence_computation = True

    product = ProductFolderManagerExtended(pf_path)
    coherence_results = interferometric_analysis(product=product, config=config)

    for res in coherence_results:
        coherence_histograms_to_netcdf(data=res, output_dir=out_dir)
        generate_coherence_graphs(data=res, output_dir=out_dir, config=config)
