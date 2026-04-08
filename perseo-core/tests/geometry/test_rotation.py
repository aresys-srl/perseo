# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Testing geometry/utilities/rotations.py functionalities"""

import unittest
from typing import get_args

import numpy as np

from perseo_core.geometry.utilities.rotations import (
    RotationOrder,
    euler_angles_to_rotation,
    rotation_to_euler_angles,
)


class RotatedFramesTestCase(unittest.TestCase):
    """Testing euler_angles_to_rotation function"""

    def setUp(self):
        self.yaw = np.deg2rad(10)
        self.pitch = np.deg2rad(15)
        self.roll = np.deg2rad(20)
        self._euler_angles = np.stack([self.yaw, self.pitch, self.roll], axis=-1)
        self._tolerance = 1e-9

    def test_euler_angles_to_rotation_0(self):
        """Testing compute rotation, scalar inputs"""
        reference_ypr = np.asarray(
            [
                [0.951251242564198, -0.075999422127131, 0.298906609756981],
                [0.167731259496521, 0.940788145499406, -0.294591055321609],
                [-0.258819045102521, 0.330366089549352, 0.907673371190369],
            ],
            dtype=float,
        )
        rotation_matrix_ypr = euler_angles_to_rotation(
            euler_angles_rad=self._euler_angles,
            order="YPR",
        )
        np.testing.assert_allclose(rotation_matrix_ypr.as_matrix(), reference_ypr, rtol=0, atol=self._tolerance)

    def test_euler_angles_to_rotation_1(self):
        """Testing compute rotation, scalar inputs"""
        reference_yrp = np.asarray(
            [
                [0.935879675463115, -0.163175911166535, 0.312254471657374],
                [0.254907748535925, 0.925416578398323, -0.280403630792980],
                [-0.243210346801694, 0.342020143325669, 0.907673371190369],
            ],
            dtype=float,
        )
        rotation_matrix_yrp = euler_angles_to_rotation(
            euler_angles_rad=self._euler_angles,
            order="YRP",
        )
        np.testing.assert_allclose(rotation_matrix_yrp.as_matrix(), reference_yrp, rtol=0, atol=self._tolerance)

    def test_euler_angles_to_rotation_2(self):
        """Testing compute rotation, scalar inputs"""
        reference_rpy = np.asarray(
            [
                [0.951251242564198, -0.167731259496521, 0.258819045102521],
                [0.250352400205939, 0.910045011297241, -0.330366089549352],
                [-0.180124260529211, 0.379057122345321, 0.907673371190369],
            ],
            dtype=float,
        )
        rotation_matrix_rpy = euler_angles_to_rotation(
            euler_angles_rad=self._euler_angles,
            order="RPY",
        )
        np.testing.assert_allclose(rotation_matrix_rpy.as_matrix(), reference_rpy, rtol=0, atol=self._tolerance)

    def test_euler_angles_to_rotation_3(self):
        """Testing compute rotation, scalar inputs"""
        reference_ryp = np.asarray(
            [
                [0.951251242564198, -0.173648177666930, 0.254887002244179],
                [0.246137153725384, 0.925416578398323, -0.288133056037496],
                [-0.185842877388499, 0.336824088833465, 0.923044938291451],
            ],
            dtype=float,
        )
        rotation_matrix_ryp = euler_angles_to_rotation(
            euler_angles_rad=self._euler_angles,
            order="RYP",
        )
        np.testing.assert_allclose(rotation_matrix_ryp.as_matrix(), reference_ryp, rtol=0, atol=self._tolerance)

    def test_euler_angles_to_rotation_4(self):
        """Testing compute rotation, scalar inputs"""
        reference_pry = np.asarray(
            [
                [0.966622809665280, -0.080554770457117, 0.243210346801694],
                [0.163175911166535, 0.925416578398323, -0.342020143325669],
                [-0.197519532830984, 0.370290541848075, 0.907673371190369],
            ],
            dtype=float,
        )
        rotation_matrix_pry = euler_angles_to_rotation(
            euler_angles_rad=self._euler_angles,
            order="PRY",
        )
        np.testing.assert_allclose(rotation_matrix_pry.as_matrix(), reference_pry, rtol=0, atol=self._tolerance)

    def test_euler_angles_to_rotation_5(self):
        """Testing compute rotation, scalar inputs"""
        reference_pyr = np.asarray(
            [
                [0.951251242564198, -0.069094499922630, 0.300577816214889],
                [0.173648177666930, 0.925416578398323, -0.336824088833465],
                [-0.254887002244179, 0.372599123061208, 0.892301804089286],
            ],
            dtype=float,
        )
        rotation_matrix_pyr = euler_angles_to_rotation(
            euler_angles_rad=self._euler_angles,
            order="PYR",
        )
        np.testing.assert_allclose(rotation_matrix_pyr.as_matrix(), reference_pyr, rtol=0, atol=self._tolerance)

    def test_euler_angles_to_rotation_vectorized(self):
        """Testing euler_angles_to_rotation, array inputs"""
        reference_pyr = np.asarray(
            [
                [0.951251242564198, -0.069094499922630, 0.300577816214889],
                [0.173648177666930, 0.925416578398323, -0.336824088833465],
                [-0.254887002244179, 0.372599123061208, 0.892301804089286],
            ],
            dtype=float,
        )

        rotation_matrix_pyr = euler_angles_to_rotation(order="PYR", euler_angles_rad=self._euler_angles.reshape(1, -1))

        np.testing.assert_allclose(
            rotation_matrix_pyr.as_matrix(), reference_pyr.reshape((1, 3, 3)), rtol=0, atol=self._tolerance
        )

        rotation_matrix_pyr = euler_angles_to_rotation(
            euler_angles_rad=np.tile(self._euler_angles, (2, 1)),
            order="PYR",
        )

        np.testing.assert_allclose(
            rotation_matrix_pyr.as_matrix(), np.tile(reference_pyr, (2, 1, 1)), rtol=0, atol=self._tolerance
        )

    def test_euler_angles_to_rotation_invalid_orders(self):
        """Testing euler_angles_to_rotation, invalid rotation orders"""
        with self.assertRaises(ValueError):
            euler_angles_to_rotation(order="PPP", euler_angles_rad=self._euler_angles)

        with self.assertRaises(AttributeError):
            euler_angles_to_rotation(order=None, euler_angles_rad=self._euler_angles)

        with self.assertRaises(ValueError):
            euler_angles_to_rotation(order="xyz", euler_angles_rad=self._euler_angles)


class EulerAnglesTestCase(unittest.TestCase):
    """Testing rotation_to_euler_angles function"""

    def setUp(self):
        self.yaw = [np.deg2rad(10), np.deg2rad(20), np.deg2rad(30)]
        self.pitch = [np.deg2rad(15), np.deg2rad(25), np.deg2rad(35)]
        self.roll = [np.deg2rad(0), np.deg2rad(30), np.deg2rad(60)]
        self._euler_angles = np.stack([self.yaw, self.pitch, self.roll], axis=-1)
        self._tolerance = 1e-9

    def test_compute_euler_angles_scalar(self):
        """Testing compute_euler_angles for single values of yaw, pitch and roll"""
        rotation = euler_angles_to_rotation(order="YPR", euler_angles_rad=self._euler_angles[0, :])
        euler_angles = rotation_to_euler_angles(order="YPR", rotation=rotation)

        np.testing.assert_allclose(euler_angles, self._euler_angles[0, :], atol=self._tolerance, rtol=0)

    def test_compute_euler_angles_vectorized(self):
        """Testing compute_euler_angles for arrays of yaw, pitch and roll"""
        rotation = euler_angles_to_rotation(order="YPR", euler_angles_rad=self._euler_angles)
        euler_angles = rotation_to_euler_angles(order="YPR", rotation=rotation)

        np.testing.assert_allclose(euler_angles, self._euler_angles, atol=self._tolerance, rtol=0)

    def test_compute_euler_angles_vectorized_all_rotations(self):
        """Testing compute_euler_angles for arrays of yaw, pitch and roll, all rotation orders"""
        for order in get_args(RotationOrder):
            rotation = euler_angles_to_rotation(order=order, euler_angles_rad=self._euler_angles)
            euler_angles = rotation_to_euler_angles(order=order, rotation=rotation)
            np.testing.assert_allclose(
                self.yaw,
                euler_angles[:, order.find("Y")],
                atol=self._tolerance,
                rtol=0,
            )
            np.testing.assert_allclose(
                self.pitch,
                euler_angles[:, order.find("P")],
                atol=self._tolerance,
                rtol=0,
            )
            np.testing.assert_allclose(
                self.roll,
                euler_angles[:, order.find("R")],
                atol=self._tolerance,
                rtol=0,
            )

    def test_compute_euler_angles_invalid_orders(self):
        """Testing compute_euler_angles with invalid rotation orders"""
        rotation = euler_angles_to_rotation(order="YPR", euler_angles_rad=self._euler_angles)
        with self.assertRaises(ValueError):
            rotation_to_euler_angles(order="PPP", rotation=rotation)

        with self.assertRaises(ValueError):
            rotation_to_euler_angles(order=None, rotation=rotation)

        with self.assertRaises(ValueError):
            rotation_to_euler_angles(order="xyz", rotation=rotation)


if __name__ == "__main__":
    unittest.main()
