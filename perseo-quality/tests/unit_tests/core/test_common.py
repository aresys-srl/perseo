# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for core/common.py functionalities"""

from __future__ import annotations

import unittest

import numpy as np
import numpy.typing as npt
import pandas as pd
from arepytools.timing.precisedatetime import PreciseDateTime

from perseo_quality.core.common import angles_computation_setup, blocks_partitioning, check_targets_visibility
from tests.unit_tests.test_utils import REF_POINTS, MockProduct


class MockTrajectory:
    """Mocking trajectory class"""

    def evaluate(self, time) -> npt.NDArray[np.floating]:
        """Mocking position interpolation"""
        return np.array([3319265.6853109375, -6203680.762160135, -768545.9597902696])

    def evaluate_first_derivatives(self, time) -> npt.NDArray[np.floating]:
        """Mocking velocity interpolation"""
        return np.array([-1775.9112802854143, -44.50034452228635, -7385.436916417019])


class CheckTargetsVisibilityTest(unittest.TestCase):
    """Testing point_target_analysis/support.py check_targets_visibility function"""

    def setUp(self) -> None:
        # creating test data
        data_dict = {
            "id": [0, 1, 2, 0, 1, 2],
            "channel": [1, 1, 1, 2, 2, 2],
            "burst": [[0], None, [1], None, [0], None],
            "swath": ["S1", "S1", "S1", "S2", "S2", "S2"],
            "polarization": ["HH"] * 6,
        }
        self.reference_df = pd.DataFrame(data=data_dict)

    def test_check_targets_visibility(self):
        """Testing check_targets_visibility"""
        targets_visibility = check_targets_visibility(product=MockProduct(), point_targets=REF_POINTS)
        targets_visibility["id"] = targets_visibility["id"].astype("int64")
        pd.testing.assert_frame_equal(targets_visibility, self.reference_df)


class AnglesComputationSetupTest(unittest.TestCase):
    """Testing radiometric_analysis/support.py angles_computation_setup function"""

    def setUp(self) -> None:
        # reference results
        self.tolerance = 1e-7
        self._trajectory = MockTrajectory()
        self._az_time = PreciseDateTime.from_utc_string("17-DEC-2025 10:04:09.272879327571")
        self._rng_values = np.array(
            [
                5.410155494173790255e-03,
                5.413152718964230233e-03,
                5.416149943754669342e-03,
                5.419147168545109319e-03,
                5.422144393335549296e-03,
                5.425141618125989273e-03,
                5.428138842916428383e-03,
                5.431136067706868360e-03,
                5.434133292497308337e-03,
                5.437130517287748314e-03,
                5.440127742078188292e-03,
                5.443124966868627401e-03,
                5.446122191659067378e-03,
                5.449119416449507355e-03,
                5.452116641239947332e-03,
                5.455113866030387310e-03,
                5.458111090820826419e-03,
                5.461108315611266396e-03,
                5.464105540401706373e-03,
                5.467102765192146351e-03,
                5.470099989982586328e-03,
                5.473097214773025437e-03,
                5.476094439563465414e-03,
                5.479091664353905392e-03,
                5.482088889144345369e-03,
                5.485086113934785346e-03,
                5.488083338725224455e-03,
                5.491080563515664432e-03,
                5.494077788306104410e-03,
                5.497075013096544387e-03,
                5.500072237886984364e-03,
                5.503069462677423473e-03,
                5.506066687467863451e-03,
                5.509063912258303428e-03,
                5.512061137048743405e-03,
                5.515058361839183382e-03,
                5.518055586629622492e-03,
                5.521052811420062469e-03,
                5.524050036210502446e-03,
                5.527047261000942423e-03,
                5.530044485791381532e-03,
                5.533041710581821510e-03,
                5.536038935372261487e-03,
                5.539036160162701464e-03,
                5.542033384953141441e-03,
                5.545030609743580551e-03,
                5.548027834534020528e-03,
                5.551025059324460505e-03,
                5.554022284114900482e-03,
                5.557019508905340459e-03,
                5.560016733695779569e-03,
                5.563013958486219546e-03,
                5.566011183276659523e-03,
                5.569008408067099500e-03,
                5.572005632857539477e-03,
                5.575002857647978587e-03,
                5.578000082438418564e-03,
                5.580997307228858541e-03,
                5.583994532019298518e-03,
                5.586991756809738495e-03,
                5.589988981600177605e-03,
                5.592986206390617582e-03,
                5.595983431181057559e-03,
                5.598980655971497536e-03,
                5.601977880761936646e-03,
                5.604975105552376623e-03,
                5.607972330342816600e-03,
            ]
        )
        self._altitude_m = 30
        # expected results
        self._ref_sensor_pos = [5634298.570491991, -4277813.834855013, 183850.74790036504]
        self._ref_ground_points = np.array(
            [
                [2654147.00288691, -5767181.69446901, -611240.78948044],
                [2653405.81450105, -5767541.98773266, -611060.39148846],
                [2652665.76868802, -5767901.60236494, -610880.27232999],
                [2651926.85800814, -5768260.54257257, -610700.43019073],
                [2651189.07509733, -5768618.81251951, -610520.86327481],
                [2650452.41266602, -5768976.4163276, -610341.56980454],
                [2649716.86349815, -5769333.35807712, -610162.54802016],
                [2648982.42045014, -5769689.64180735, -609983.79617956],
                [2648249.07644986, -5770045.27151719, -609805.31255808],
                [2647516.82449566, -5770400.25116567, -609627.09544824],
                [2646785.6576554, -5770754.58467252, -609449.14315952],
                [2646055.56906553, -5771108.27591869, -609271.4540181],
                [2645326.55193009, -5771461.32874691, -609094.02636667],
                [2644598.59951985, -5771813.74696219, -608916.85856419],
                [2643871.70517138, -5772165.5343323, -608739.94898565],
                [2643145.86228616, -5772516.69458834, -608563.29602189],
                [2642421.06432973, -5772867.23142518, -608386.89807936],
                [2641697.30483081, -5773217.14850196, -608210.75357992],
                [2640974.57738048, -5773566.44944256, -608034.86096063],
                [2640252.87563132, -5773915.13783609, -607859.21867356],
                [2639532.19329664, -5774263.21723731, -607683.8251856],
                [2638812.52414964, -5774610.69116712, -607508.67897821],
                [2638093.86202264, -5774957.56311301, -607333.7785473],
                [2637376.20080631, -5775303.83652944, -607159.12240299],
                [2636659.5344489, -5775649.51483834, -606984.70906946],
                [2635943.85695551, -5775994.60142949, -606810.53708474],
                [2635229.16238733, -5776339.09966095, -606636.60500054],
                [2634515.44486093, -5776683.01285946, -606462.91138209],
                [2633802.69854755, -5777026.34432085, -606289.45480792],
                [2633090.9176724, -5777369.09731045, -606116.23386976],
                [2632380.09651397, -5777711.27506344, -605943.24717231],
                [2631670.22940333, -5778052.88078528, -605770.49333309],
                [2630961.31072352, -5778393.91765205, -605597.97098231],
                [2630253.33490883, -5778734.38881083, -605425.67876267],
                [2629546.29644422, -5779074.29738007, -605253.61532923],
                [2628840.18986461, -5779413.64644997, -605081.77934922],
                [2628135.00975432, -5779752.43908277, -604910.16950193],
                [2627430.75074644, -5780090.67831317, -604738.78447855],
                [2626727.4075222, -5780428.36714863, -604567.62298199],
                [2626024.97481039, -5780765.50856971, -604396.68372677],
                [2625323.4473868, -5781102.10553042, -604225.96543888],
                [2624622.82007361, -5781438.16095852, -604055.46685561],
                [2623923.08773881, -5781773.67775587, -603885.18672542],
                [2623224.2452957, -5782108.6587987, -603715.12380783],
                [2622526.28770228, -5782443.10693799, -603545.27687325],
                [2621829.20996074, -5782777.02499971, -603375.64470289],
                [2621133.00711692, -5783110.41578515, -603206.22608857],
                [2620437.67425979, -5783443.28207123, -603037.01983265],
                [2619743.20652093, -5783775.62661077, -602868.02474788],
                [2619049.599074, -5784107.45213278, -602699.23965727],
                [2618356.84713427, -5784438.76134276, -602530.66339398],
                [2617664.94595813, -5784769.55692295, -602362.29480118],
                [2616973.89084257, -5785099.84153263, -602194.13273196],
                [2616283.67712472, -5785429.6178084, -602026.1760492],
                [2615594.3001814, -5785758.8883644, -601858.42362544],
                [2614905.75542859, -5786087.65579263, -601690.87434281],
                [2614218.03832107, -5786415.92266316, -601523.52709284],
                [2613531.14435188, -5786743.69152442, -601356.38077646],
                [2612845.06905191, -5787070.96490343, -601189.43430377],
                [2612159.8079895, -5787397.74530605, -601022.68659404],
                [2611475.35676994, -5787724.03521724, -600856.13657553],
                [2610791.71103509, -5788049.83710128, -600689.78318545],
                [2610108.86646294, -5788375.15340202, -600523.62536977],
                [2609426.81876723, -5788699.9865431, -600357.66208323],
                [2608745.563697, -5789024.33892819, -600191.89228915],
                [2608065.09703623, -5789348.21294122, -600026.31495937],
                [2607385.41460342, -5789671.61094659, -599860.92907415],
            ]
        )
        self._ref_nadir = np.array([-328173.0560301547, 613352.7916620681, 76446.66955969587])

    def test_angles_computation_setup_with_altitude(self):
        """Testing angles_computation_setup"""
        sensor_pos, ground_points, nadir = angles_computation_setup(
            trajectory=self._trajectory,
            azimuth_time=self._az_time,
            look_direction="RIGHT",
            range_values=self._rng_values,
            altitude=self._altitude_m,
        )
        np.testing.assert_allclose(sensor_pos, self._trajectory.evaluate(self._az_time), atol=self.tolerance, rtol=0)
        np.testing.assert_allclose(ground_points, self._ref_ground_points, atol=self.tolerance, rtol=0)
        np.testing.assert_allclose(nadir, self._ref_nadir, atol=self.tolerance, rtol=0)


class BlocksDefinitionTest(unittest.TestCase):
    """Testing radiometric_analysis/support.py blocks_definition function"""

    def setUp(self) -> None:
        # creating test data
        self._az_axis = np.zeros(2500)
        self._rng_axis = np.zeros(4500)
        self._lines_per_burst = np.array([300] * 5)
        self._default_block_size = 2000
        self.expected_res_0 = [
            np.array([300] * 5),
            5,
            [(150, 2250), (450, 2250), (750, 2250), (1050, 2250), (1350, 2250)],
        ]
        self.expected_res_1 = [2000, 1, [(1000, 2250)]]

    def test_blocks_definition_0(self):
        """Testing blocks_definition function"""
        blocks_data = blocks_partitioning(
            azimuth_axis=self._az_axis,
            default_block_size=self._default_block_size,
            lines_per_burst=self._lines_per_burst,
            range_axis=self._rng_axis,
        )
        np.testing.assert_array_equal(blocks_data[0], self.expected_res_0[0])
        self.assertEqual(blocks_data[1], self.expected_res_0[1])
        np.testing.assert_array_equal(blocks_data[2], self.expected_res_0[2])

    def test_blocks_definition_1(self):
        """Testing blocks_definition function"""
        blocks_data = blocks_partitioning(
            azimuth_axis=self._az_axis,
            default_block_size=self._default_block_size,
            lines_per_burst=np.array([self._lines_per_burst[0]]),
            range_axis=self._rng_axis,
        )
        self.assertListEqual(list(blocks_data), self.expected_res_1)


if __name__ == "__main__":
    unittest.main()
