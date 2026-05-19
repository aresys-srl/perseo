# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for atmospheric/ionosphere map name builder"""

from arepytools.timing.precisedatetime import PreciseDateTime

from perseo_perturbations.atmospheric.ionosphere import (
    IonosphericAnalysisCenters,
    TECMapSolutionType,
    TECMapTimeResolution,
    generate_ionospheric_map_filename,
)
from perseo_perturbations.atmospheric.troposphere import (
    TroposphericMapType,
    generate_tropospheric_map_name_for_vmf_data,
)


class IonosphereMapNameBuildTest:
    """Testing atmospheric/ionosphere.py generate_ionospheric_map_filename functionality"""

    test_date_precise_old_method = PreciseDateTime.from_numeric_datetime(2018, 10, 26, 15, 12, 33, 15236)
    test_date_precise_new_method = PreciseDateTime.from_numeric_datetime(2023, 10, 26, 15, 12, 33, 15236)
    centers = [IonosphericAnalysisCenters.COD, IonosphericAnalysisCenters.ESA]
    solution_type = TECMapSolutionType.RAPID
    time_resolution = TECMapTimeResolution.HALF_HOUR
    expected_results_old = ["codg2990.18i", "esag2990.18i"]
    expected_results_new = ["COD0OPSFIN_20232990000_01D_01H_GIM.INX", "ESA0OPSFIN_20232990000_01D_01H_GIM.INX"]

    def test_generate_ionospheric_map_name_from_precisedatetime_old_method(self) -> None:
        """Testing ionospheric map name generation from PreciseDateTime, pre format change"""
        for item_id, center in enumerate(self.centers):
            map_name = generate_ionospheric_map_filename(acq_time=self.test_date_precise_old_method, center=center)
            assert map_name == self.expected_results_old[item_id]

    def test_generate_ionospheric_map_name_from_precisedatetime_new_method(self) -> None:
        """Testing ionospheric map name generation from PreciseDateTime, post format change"""
        for item_id, center in enumerate(self.centers):
            map_name = generate_ionospheric_map_filename(acq_time=self.test_date_precise_new_method, center=center)
            assert map_name == self.expected_results_new[item_id]

    def test_generate_ionospheric_map_name_from_precisedatetime_new_method_sol_type(self) -> None:
        """Testing ionospheric map name generation from PreciseDateTime, post format change, with solution type"""
        for item_id, center in enumerate(self.centers):
            map_name = generate_ionospheric_map_filename(
                acq_time=self.test_date_precise_new_method, center=center, solution_type=self.solution_type
            )
            assert map_name == self.expected_results_new[item_id].replace("FIN", "RAP")

    def test_generate_ionospheric_map_name_from_precisedatetime_new_method_time_res(self) -> None:
        """Testing ionospheric map name generation from PreciseDateTime, post format change, with time resolution"""
        for item_id, center in enumerate(self.centers):
            map_name = generate_ionospheric_map_filename(
                acq_time=self.test_date_precise_new_method, center=center, time_resolution=self.time_resolution
            )
            assert map_name == self.expected_results_new[item_id].replace("01H", "30M")


class TroposphereMapNameBuildTest:
    """Testing atmospheric/ionosphere.py generate_ionospheric_map_filename functionality"""

    test_date = PreciseDateTime.from_numeric_datetime(2023, 10, 26, 15, 12, 33, 15236)
    expected_map_names = ["VMF3_20231026.H06", "VMF3_20231026.H12", "VMF3_20231026.H18", "VMF3_20231027.H00"]
    expected_times = [
        PreciseDateTime.from_numeric_datetime(2023, 10, 26, 6),
        PreciseDateTime.from_numeric_datetime(2023, 10, 26, 12),
        PreciseDateTime.from_numeric_datetime(2023, 10, 26, 18),
        PreciseDateTime.from_numeric_datetime(2023, 10, 27, 0),
    ]

    def test_troposphere_map_generator_vmf3(self) -> None:
        """Testing map name generator for Troposphere VMF3 data"""
        map_names, map_times = generate_tropospheric_map_name_for_vmf_data(
            acq_time=self.test_date, map_type=TroposphericMapType.VMF3
        )

        assert map_names == self.expected_map_names
        assert map_times == self.expected_times
