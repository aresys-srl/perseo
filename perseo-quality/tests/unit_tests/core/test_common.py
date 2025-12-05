# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for core/common.py functionalities"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from arepytools.timing.precisedatetime import PreciseDateTime

from perseo_quality.core.common import angles_computation_setup, blocks_partitioning, check_targets_visibility
from tests.unit_tests.test_utils import REF_POINTS, MockProduct, MockTrajectory


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
        self.tolerance = 1e-9
        self._trajectory = MockTrajectory()
        self._az_time = PreciseDateTime.from_utc_string("05-JAN-2017 08:29:41.068885794433")
        self._rng_values = np.array(
            [
                0.004975715993854,
                0.004975729022631,
                0.004975742051408,
                0.004975755080185,
                0.004975768108962,
                0.004975781137739,
                0.004975794166516,
                0.004975807195293,
                0.00497582022407,
                0.004975833252847,
                0.004975846281624,
                0.004975859310401,
                0.004975872339178,
                0.004975885367955,
                0.004975898396732,
                0.004975911425509,
                0.004975924454286,
                0.004975937483063001,
                0.004975950511840001,
                0.004975963540617001,
                0.004975976569394001,
            ]
        )
        # expected results
        self._ref_sensor_pos = [5634298.570491991, -4277813.834855013, 183850.74790036504]
        self._ref_ground_points = np.array(
            [
                4.926391951508208178e06,
                -4.045258062462532893e06,
                2.164839266398744367e05,
                4.926388638358334079e06,
                -4.045262038216231391e06,
                2.164850228497198550e05,
                4.926385325266964734e06,
                -4.045266013892872725e06,
                2.164861190389312687e05,
                4.926382012234100141e06,
                -4.045269989492459223e06,
                2.164872152075095219e05,
                4.926378699259732850e06,
                -4.045273965014998335e06,
                2.164883113554565352e05,
                4.926375386343861930e06,
                -4.045277940460492857e06,
                2.164894074827730074e05,
                4.926372073486481793e06,
                -4.045281915828950703e06,
                2.164905035894609464e05,
                4.926368760687589645e06,
                -4.045285891120372806e06,
                2.164915996755208762e05,
                4.926365447947181761e06,
                -4.045289866334765218e06,
                2.164926957409543102e05,
                4.926362135265253484e06,
                -4.045293841472133063e06,
                2.164937917857626453e05,
                4.926358822641801089e06,
                -4.045297816532485187e06,
                2.164948878099481226e05,
                4.926355510076822713e06,
                -4.045301791515820194e06,
                2.164959838135105674e05,
                4.926352197570312768e06,
                -4.045305766422146000e06,
                2.164970797964519879e05,
                4.926348885122268461e06,
                -4.045309741251465864e06,
                2.164981757587734028e05,
                4.926345572732684202e06,
                -4.045313716003788635e06,
                2.164992717004768783e05,
                4.926342260401559062e06,
                -4.045317690679115243e06,
                2.165003676215629093e05,
                4.926338948128886521e06,
                -4.045321665277452208e06,
                2.165014635220333003e05,
                4.926335635914664716e06,
                -4.045325639798806049e06,
                2.165025594018894772e05,
                4.926332323758888990e06,
                -4.045329614243176300e06,
                2.165036552611317602e05,
                4.926329011661556549e06,
                -4.045333588610572275e06,
                2.165047510997624195e05,
                4.926325699622661807e06,
                -4.045337562901000027e06,
                2.165058469177829102e05,
            ]
        ).reshape(-1, 3)
        self._ref_nadir = np.array([-556144.218795416, 422249.79801344173, -18257.49818938377])

    def test_angles_computation_setup(self):
        """Testing angles_computation_setup"""
        sensor_pos, ground_points, nadir = angles_computation_setup(
            trajectory=self._trajectory,
            azimuth_time=self._az_time,
            look_direction="RIGHT",
            range_values=self._rng_values,
        )
        np.testing.assert_allclose(sensor_pos, self._ref_sensor_pos, atol=self.tolerance, rtol=0)
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
