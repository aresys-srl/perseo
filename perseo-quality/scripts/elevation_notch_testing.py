# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Elevation Notch analysis testing script"""

import logging
import sys
from pathlib import Path
import xarray as xr

from sct_sentinel1_reader.protocol_implementation import Sentinel1ProductManager
from perseo_quality.elevation_notch_analysis.config import ElevationNotchConfig
from perseo_quality.elevation_notch_analysis.graphical_output import plot_elevation_notch_analysis
from perseo_quality.elevation_notch_analysis.support import elevation_notch_profiles_to_netcdf
from perseo_quality.logger import quality_logger

from perseo_quality.elevation_notch_analysis.analysis import (
    elevation_notch_analysis,
)
from netCDF4 import Dataset
from sct.io.antenna_pattern_manager import read_antenna_pattern_netcdf

if __name__ == "__main__":
    # setup custom logger
    quality_logger.addHandler(logging.StreamHandler(sys.stdout))
    output_dir = Path(r"C:\ARESYS_PROJ\perseo\perseo-quality\scripts\out")
    graphs_dir = output_dir.joinpath("graphs")
    graphs_dir.mkdir(exist_ok=True)

    # from netCDF4 import Dataset
    # import matplotlib.pyplot as plt
    # root_200 = Dataset(r"C:\Users\giorgio.parma\Desktop\temporary_outputs\notch\elevation_notch_results.nc")
    # root_0 = Dataset(r"C:\Users\giorgio.parma\Desktop\temporary_outputs\notch\zero\elevation_notch_results.nc")
    # ds_0, ds_200 = root_0["EN"]["HH"], root_200["EN"]["HH"]
    # ...

    product_path = r"C:\Users\giorgio.parma\Aresys_DATA\sct_data\sentinel1\SLC_EN_25.SAFE"
    product = Sentinel1ProductManager(product_path)
    config = ElevationNotchConfig(range_pixel_margin=2927)
    am_pattern = read_antenna_pattern_netcdf(Path(r"C:\Users\giorgio.parma\Aresys_DATA\sct_data\sentinel1\antenna_pattern_TW_EN.nc"))
    # am_pattern = None
    results = elevation_notch_analysis(product, config=config, antenna_pattern=am_pattern)
    # import pickle
    # with open("data_no_antenna.pkl", "rb") as f:
    #     results = pickle.load(f)
    output_netcdf = elevation_notch_profiles_to_netcdf(results, output_dir)
    plot_elevation_notch_analysis(results, graphs_dir)
    ...
