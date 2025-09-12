# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Reference for numerical output content for Block-Wise Radiometric Analysis KPI"""

from __future__ import annotations

from dataclasses import fields

from perseo_quality.radiometric_analysis.custom_dataclasses import (
    AverageElevationRadiometricKPI,
    NESZRadiometricKPI,
    RadiometricOutputProductGeneralInfo,
    ScallopingRadiometricKPI,
)

GENERAL_INFO_COLS = [f.name for f in fields(RadiometricOutputProductGeneralInfo)]
AVG_PROFILE_KPI_COLS = [f.name for f in fields(AverageElevationRadiometricKPI)]
NESZ_KPI_COLS = [f.name for f in fields(NESZRadiometricKPI)]
SCALLOPING_KPI_COLS = [f.name for f in fields(ScallopingRadiometricKPI)]

AVERAGE_PROFILES_KPI_COLUMNS = GENERAL_INFO_COLS + AVG_PROFILE_KPI_COLS
NESZ_KPI_COLUMNS = GENERAL_INFO_COLS + NESZ_KPI_COLS
SCALLOPING_KPI_COLUMNS = GENERAL_INFO_COLS + SCALLOPING_KPI_COLS
