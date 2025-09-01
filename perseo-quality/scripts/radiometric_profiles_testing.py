# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Block-wise radiometric analysis testing script"""

import logging
import sys
from pathlib import Path

from sct_aresys_reader.protocol_implementation import ProductFolderManagerExtended

from perseo_quality.core.generic_dataclasses import SARRadiometricQuantity
from perseo_quality.logger import quality_logger
from perseo_quality.radiometric_analysis.block_wise.analysis import (
    average_elevation_profiles,
    scalloping_profiles,
)
from perseo_quality.radiometric_analysis.block_wise.graphical_output import (
    radiometric_2D_hist_plot,
)
from perseo_quality.radiometric_analysis.block_wise.support import (
    radiometric_profiles_to_netcdf,
    radiometric_statistical_analysis_to_df,
)

if __name__ == "__main__":
    # setup custom logger
    quality_logger.addHandler(logging.StreamHandler(sys.stdout))

    out_fldr = Path(r"C:\Users\giorgio.parma\Aresys_DATA\quality_data\radiometry\reports")

    prod = r"C:\Users\giorgio.parma\Aresys_DATA\quality_data\real_ICEYE_slc\SLH_45599_SLC"
    prod = ProductFolderManagerExtended(prod)

    # performing NESZ analysis
    # res = nesz_profiles(product=prod)
    # tag = "nesz"
    # plot_mode = "min"

    # stats_df = radiometric_statistical_analysis_to_df(res)

    # performing NESZ analysis
    res = scalloping_profiles(product=prod)
    tag = "scalloping"
    plot_mode = "mean"

    stats_df = radiometric_statistical_analysis_to_df(res)

    # performing Rain Forest analysis
    res = average_elevation_profiles(product=prod, output_quantity=SARRadiometricQuantity.GAMMA_NOUGHT)
    tag = "rain_forest"
    plot_mode = "mean"

    stats_df = radiometric_statistical_analysis_to_df(res)

    # graphs and netcdf saving
    for item in res:
        radiometric_2D_hist_plot(
            data=item,
            out_dir=out_fldr,
            title=f"{tag.upper()} Profiles {item.general_info.swath} {item.general_info.polarization}",
            plot_mode=plot_mode,
        )
        radiometric_profiles_to_netcdf(data=item, out_path=out_fldr, tag=tag)
