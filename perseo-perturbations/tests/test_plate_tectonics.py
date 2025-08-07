# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for geodynamics/plate_tectonics.py core functionalities"""

import unittest

import numpy as np

from perseo_perturbations.geodynamics.plate_tectonics import compute_displacement


class PlateTectonics(unittest.TestCase):
    """Testing plate_tectonics.py core functionalities"""

    def setUp(self) -> None:
        # creating test data
        self.pt_pos = np.array(
            [
                (-2468789.77437779, -4626148.4320329, 3620025.27093258),
                (-2467963.52819618, -4626181.75345945, 3620542.41415808),
                (-2467068.51440511, -4626651.66776616, 3620552.45331852),
                (-2466645.20366593, -4626877.04176162, 3620552.92942237),
                (-2466140.52981391, -4627136.54745231, 3620565.21049583),
            ]
        )
        self.time = 2.5 * 3.154e7  # 2 and a half year in sec
        self.displacement_ref = np.array(
            [
                [0.07495684784999701, -0.09378870416470346, -0.06873647242749047],
                [0.07495657874266105, -0.09378148063923704, -0.0687355770332182],
                [0.07496478195082092, -0.0937659653845648, -0.06874067151215248],
                [0.07496872341131613, -0.09375856748311671, -0.06874312464983566],
                [0.07497324245580272, -0.09374991139958834, -0.06874592087558197],
            ]
        )

    def test_compute_displacement(self):
        """Testing compute_displacement function"""
        displacement = compute_displacement(xyz_coords=self.pt_pos, plate_ref="ARAB", time_delta=self.time)

        np.testing.assert_allclose(displacement, self.displacement_ref, atol=1e-10, rtol=0)

    def test_compute_displacement_drift_vel(self):
        """Testing compute_displacement function with drift velocities"""
        drift_vel = np.ones_like(self.pt_pos) * 2e-7
        displacement = compute_displacement(xyz_coords=self.pt_pos, time_delta=self.time, drift_vel=drift_vel)

        np.testing.assert_allclose(displacement, self.time * drift_vel, atol=1e-10, rtol=0)

    def test_compute_displacement_error(self):
        """Testing compute_displacement function with drift velocities"""
        with self.assertRaises(RuntimeError):
            compute_displacement(xyz_coords=self.pt_pos, time_delta=self.time)


if __name__ == "__main__":
    unittest.main()
