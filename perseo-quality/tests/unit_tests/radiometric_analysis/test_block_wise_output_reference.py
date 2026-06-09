# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for radiometric_analysis/block_wise/output_reference.py"""

from __future__ import annotations

from dataclasses import fields

from perseo_quality.radiometric_analysis.block_wise import output_reference as ref
from perseo_quality.radiometric_analysis.custom_dataclasses import (
    AverageElevationRadiometricKPI,
    NESZRadiometricKPI,
    RadiometricOutputProductGeneralInfo,
    ScallopingRadiometricKPI,
)


class TestBlockWiseOutputReference:
    """Testing block_wise output reference column lists"""

    def test_general_info_cols_match_dataclass(self):
        """Testing GENERAL_INFO_COLS matches RadiometricOutputProductGeneralInfo fields"""
        expected = [f.name for f in fields(RadiometricOutputProductGeneralInfo)]
        assert ref.GENERAL_INFO_COLS == expected

    def test_avg_profile_kpi_cols_match_dataclass(self):
        """Testing AVG_PROFILE_KPI_COLS matches AverageElevationRadiometricKPI fields"""
        expected = [f.name for f in fields(AverageElevationRadiometricKPI)]
        assert ref.AVG_PROFILE_KPI_COLS == expected

    def test_nesz_kpi_cols_match_dataclass(self):
        """Testing NESZ_KPI_COLS matches NESZRadiometricKPI fields"""
        expected = [f.name for f in fields(NESZRadiometricKPI)]
        assert ref.NESZ_KPI_COLS == expected

    def test_scalloping_kpi_cols_match_dataclass(self):
        """Testing SCALLOPING_KPI_COLS matches ScallopingRadiometricKPI fields"""
        expected = [f.name for f in fields(ScallopingRadiometricKPI)]
        assert ref.SCALLOPING_KPI_COLS == expected

    def test_average_profiles_kpi_columns_is_concatenation(self):
        """Testing AVERAGE_PROFILES_KPI_COLUMNS is GENERAL_INFO_COLS + AVG_PROFILE_KPI_COLS"""
        assert ref.AVERAGE_PROFILES_KPI_COLUMNS == ref.GENERAL_INFO_COLS + ref.AVG_PROFILE_KPI_COLS

    def test_nesz_kpi_columns_is_concatenation(self):
        """Testing NESZ_KPI_COLUMNS is GENERAL_INFO_COLS + NESZ_KPI_COLS"""
        assert ref.NESZ_KPI_COLUMNS == ref.GENERAL_INFO_COLS + ref.NESZ_KPI_COLS

    def test_scalloping_kpi_columns_is_concatenation(self):
        """Testing SCALLOPING_KPI_COLUMNS is GENERAL_INFO_COLS + SCALLOPING_KPI_COLS"""
        assert ref.SCALLOPING_KPI_COLUMNS == ref.GENERAL_INFO_COLS + ref.SCALLOPING_KPI_COLS
