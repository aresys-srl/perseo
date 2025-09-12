# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Reference for numerical output content for Point Target Analysis"""

from __future__ import annotations

from dataclasses import fields

from perseo_quality.point_targets_analysis.custom_dataclasses import (
    GenericInfoOutput,
    IRFDataOutput,
    PTAdditionalInfo,
    RCSDataOutput,
)

PROD_INFO_COLS = ["target", "channel", "product", "sensor"]
GENERIC_INFO_OUTPUT_COLS = [f.name for f in fields(GenericInfoOutput)]
ADDITIONAL_INFO_OUTPUT_COLS = [f.name for f in fields(PTAdditionalInfo)]
IRF_OUTPUT_COLS = [f.name for f in fields(IRFDataOutput)]
RCS_OUTPUT_COLS = [f.name for f in fields(RCSDataOutput)]
REMOVED_OUTPUT_COLS = ["peak_value_complex"]

_PTA_OUTPUT_COLUMNS_DF = (
    PROD_INFO_COLS + GENERIC_INFO_OUTPUT_COLS + ADDITIONAL_INFO_OUTPUT_COLS + IRF_OUTPUT_COLS + RCS_OUTPUT_COLS
)
_PTA_OUTPUT_COLUMNS_DF = [p for p in _PTA_OUTPUT_COLUMNS_DF if p not in REMOVED_OUTPUT_COLS]

UNIT_OF_MEASURE_REFERENCE_MAP = {
    "incidence_angle": "[deg]",
    "look_angle": "[deg]",
    "ground_velocity": "[ms]",
    "doppler_frequency": "[Hz]",
    "steering_doppler_frequency": "[Hz]",
    "doppler_rate_real": "[Hzs]",
    "doppler_rate_theoretical": "[Hzs]",
    "peak_azimuth_time": "[UTC]",
    "peak_range_time": "[s]",
    "squint_angle": "[rad]",
    "azimuth_resolution": "[m]",
    "azimuth_pslr": "[dB]",
    "azimuth_islr": "[dB]",
    "azimuth_sslr": "[dB]",
    "azimuth_localization_error": "[m]",
    "range_resolution": "[m]",
    "ground_range_resolution": "[m]",
    "range_pslr": "[dB]",
    "range_islr": "[dB]",
    "range_sslr": "[dB]",
    "pslr_2d": "[dB]",
    "islr_2d": "[dB]",
    "sslr_2d": "[dB]",
    "ground_range_localization_error": "[m]",
    "slant_range_localization_error": "[m]",
    "rcs": "[dB]",
    "rcs_error": "[dB]",
    "peak_phase_error": "[deg]",
    "clutter": "[dB]",
    "scr": "[dB]",
}


def add_unit_of_measure_to_df_columns(columns: list[str]) -> list:
    """Attributing unit of measure to dataframe column names.

    Parameters
    ----------
    columns :  list[str]
        output results dataframe column names

    Returns
    -------
    list
        new names with unit of measure added
    """

    return [(c + "_" + UNIT_OF_MEASURE_REFERENCE_MAP[c] if c in UNIT_OF_MEASURE_REFERENCE_MAP else c) for c in columns]


PTA_OUTPUT_COLUMNS_DF_UM = add_unit_of_measure_to_df_columns(_PTA_OUTPUT_COLUMNS_DF)
