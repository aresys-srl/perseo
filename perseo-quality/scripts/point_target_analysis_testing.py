# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Script to test the point target analysis functionalities"""

import logging
import sys
from pathlib import Path

from sct_aresys_reader.protocol_implementation import ProductFolderManagerExtended

from perseo_quality.logger import quality_logger
from perseo_quality.point_targets_analysis.analysis import point_target_analysis
from perseo_quality.point_targets_analysis.config import PointTargetAnalysisConfig

if __name__ == "__main__":
    # setup custom logger
    quality_logger.addHandler(logging.StreamHandler(sys.stdout))

    # product folder
    prd_base_path = Path(r"C:\Users\giorgio.parma\Aresys_DATA\quality_data")
    # prd_path = prd_base_path.joinpath("demo_topsar", "SLC")
    # prd_path = prd_base_path.joinpath("biomass_multichannel", "iSLC")
    # prd_path = prd_base_path.joinpath("biomass_multichannel", "iGRD")
    prd_path = prd_base_path.joinpath("real_ICEYE_slc", "SLH_45599_SLC")
    # prd_path = prd_base_path.joinpath("squinted_data", "50deg", "SLC")

    # point target
    pt_base_path = Path(r"C:\Users\giorgio.parma\Aresys_DATA\quality_data")
    # pt_path = pt_base_path.joinpath("demo_topsar", "PointTargetsBinary")
    # pt_path = pt_base_path.joinpath("biomass_multichannel", "PointTargets_File.xml")
    pt_path = pt_base_path.joinpath("real_ICEYE_slc", "PointTargets_File__ICEYE.xml")
    # pt_path = pt_base_path.joinpath("squinted_data", "50deg", "PointTargetsBinary")

    # config file
    config = PointTargetAnalysisConfig()
    graphs = True

    # output path
    out = Path(r"C:\Users\giorgio.parma\Desktop\temporary_outputs")

    point_targets = ...

    # analysis
    product = ProductFolderManagerExtended(path=prd_path)
    df, graph_data = point_target_analysis(product=product, point_targets=point_targets, config=config)
    df.to_csv(out.joinpath("results").with_suffix(".csv"), index=False)

    # graphs
    if graphs:
        import perseo_quality.point_targets_analysis.graphical_output as ptgpo

        grph_out_dir = out.joinpath("graphs")
        Path.mkdir(grph_out_dir, exist_ok=True)

        for item in graph_data:
            this_graph = f"Target {item.target}, Channel {item.channel}"
            quality_logger.info("Generating graphical output for " + this_graph)
            data_val = df.query(f"target == {item.target} & channel == {item.channel}").to_dict("records")[0]
            label = (
                f"target_{data_val['target']}_{data_val['swath']}_"
                + f"polarization_{data_val['polarization'].replace('/', '')}"
            )

            if config.generate_static_graphs:
                try:
                    # IRF graphs
                    ptgpo.irf_graphs(
                        data_graph=item.irf,
                        data_values=data_val,
                        label=label,
                        out_dir=grph_out_dir,
                    )
                except Exception:
                    quality_logger.error("Could not generate IRF images for " + this_graph)

                if config.perform_rcs:
                    # RCS graphs
                    try:
                        ptgpo.rcs_graphs(data_graph=item.rcs, label=label, out_dir=grph_out_dir)
                    except Exception:
                        quality_logger.error("Could not generate RCS images for " + this_graph)
