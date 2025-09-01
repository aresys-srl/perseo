# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Script to test the point wise radiometric analysis functionalities"""

import logging
import sys

from sct_aresys_reader.protocol_implementation import ProductFolderManagerExtended

from perseo_quality.logger import quality_logger
from perseo_quality.radiometric_analysis.custom_dataclasses import RadiometricAnalysisDirection
from perseo_quality.radiometric_analysis.point_wise.analysis import point_wise_radiometric_analysis
from perseo_quality.radiometric_analysis.point_wise.config import PointWiseRadiometricAnalysisConfig
from perseo_quality.radiometric_analysis.point_wise.graphical_output import radiometric_analysis_graphs

if __name__ == "__main__":
    # setup custom logger
    quality_logger.addHandler(logging.StreamHandler(sys.stdout))

    # analysis
    out_dir = r"C:\Users\giorgio.parma\Desktop\temporary_outputs"
    prd_path = r"C:\Users\giorgio.parma\Aresys_DATA\quality_data\SLC"
    product = ProductFolderManagerExtended(path=prd_path)
    out = point_wise_radiometric_analysis(
        product=product,
        azimuth_times=[2000],
        range_times=None,
        swath_name=None,
        selected_polarization=None,
        is_pixel=True,
        analysis_config=PointWiseRadiometricAnalysisConfig(direction=RadiometricAnalysisDirection.RANGE),
    )
    radiometric_analysis_graphs(data=out, out_dir=out_dir)
