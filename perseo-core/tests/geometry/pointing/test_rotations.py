# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Testing geometry/utilities/rotations.py functionalities"""

from typing import get_args

import numpy as np
import pytest

from perseo_core.geometry.pointing import (
    RotationOrder,
)
from perseo_core.geometry.pointing.rotations import (
    euler_angles_to_rotation,
    rotation_to_euler_angles,
)


class TestRotation:
    """Test euler_angles_to_rotation and rotation_to_euler_angles with various angle orders."""

    @pytest.fixture(autouse=True)
    def setup_rotation_data(self, rotation_test_data):
        data = rotation_test_data
        self.yaw = data["yaw"]
        self.pitch = data["pitch"]
        self.roll = data["roll"]
        self._euler_angles = data["euler_angles"]
        self._tolerance = data["tolerance"]

    def test_euler_angles_to_rotation_roll(self):
        """Test euler_angles_to_rotation with roll only."""
        cos_roll = np.cos(self.roll)
        sin_roll = np.sin(self.roll)
        rotation_matrix = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, cos_roll, -sin_roll],
                [0.0, sin_roll, cos_roll],
            ],
        )
        rotation = euler_angles_to_rotation(
            ypr_rad=np.array([0, 0, self.roll]),
            order="YPR",
        )
        np.testing.assert_allclose(rotation.as_matrix(), rotation_matrix, rtol=0, atol=self._tolerance)

    def test_euler_angles_to_rotation_pitch(self):
        """Test euler_angles_to_rotation with pitch only."""
        cos_pitch = np.cos(self.pitch)
        sin_pitch = np.sin(self.pitch)
        rotation_matrix = np.asarray(
            [
                [cos_pitch, 0.0, sin_pitch],
                [0.0, 1.0, 0.0],
                [-sin_pitch, 0.0, cos_pitch],
            ],
        )
        rotation = euler_angles_to_rotation(
            ypr_rad=np.array([0, self.pitch, 0]),
            order="YPR",
        )
        np.testing.assert_allclose(rotation.as_matrix(), rotation_matrix, rtol=0, atol=self._tolerance)

    def test_euler_angles_to_rotation_yaw(self):
        """Test euler_angles_to_rotation with yaw only."""
        cos_yaw = np.cos(self.yaw)
        sin_yaw = np.sin(self.yaw)
        rotation_matrix = np.asarray(
            [
                [cos_yaw, -sin_yaw, 0.0],
                [sin_yaw, cos_yaw, 0.0],
                [0.0, 0.0, 1.0],
            ],
        )
        rotation = euler_angles_to_rotation(
            ypr_rad=np.array([self.yaw, 0, 0]),
            order="YPR",
        )
        np.testing.assert_allclose(rotation.as_matrix(), rotation_matrix, rtol=0, atol=self._tolerance)

    def test_euler_angles_to_rotation_ypr(self):
        """Test euler_angles_to_rotation with YPR order."""
        rotation_matrix = np.asarray(
            [
                [0.951251242564198, -0.075999422127131, 0.298906609756981],
                [0.167731259496521, 0.940788145499406, -0.294591055321609],
                [-0.258819045102521, 0.330366089549352, 0.907673371190369],
            ],
        )
        rotation = euler_angles_to_rotation(
            ypr_rad=self._euler_angles,
            order="YPR",
        )
        np.testing.assert_allclose(rotation.as_matrix(), rotation_matrix, rtol=0, atol=self._tolerance)

    def test_euler_angles_to_rotation_yrp(self):
        """Test euler_angles_to_rotation with YRP order."""
        rotation_matrix = np.asarray(
            [
                [0.935879675463115, -0.163175911166535, 0.312254471657374],
                [0.254907748535925, 0.925416578398323, -0.280403630792980],
                [-0.243210346801694, 0.342020143325669, 0.907673371190369],
            ],
        )
        rotation = euler_angles_to_rotation(
            ypr_rad=self._euler_angles,
            order="YRP",
        )
        np.testing.assert_allclose(rotation.as_matrix(), rotation_matrix, rtol=0, atol=self._tolerance)

    def test_euler_angles_to_rotation_rpy(self):
        """Test euler_angles_to_rotation with RPY order."""
        rotation_matrix = np.asarray(
            [
                [0.951251242564198, -0.167731259496521, 0.258819045102521],
                [0.250352400205939, 0.910045011297241, -0.330366089549352],
                [-0.180124260529211, 0.379057122345321, 0.907673371190369],
            ],
        )
        rotation = euler_angles_to_rotation(
            ypr_rad=self._euler_angles,
            order="RPY",
        )
        np.testing.assert_allclose(rotation.as_matrix(), rotation_matrix, rtol=0, atol=self._tolerance)

    def test_euler_angles_to_rotation_ryp(self):
        """Test euler_angles_to_rotation with RYP order."""
        rotation_matrix = np.asarray(
            [
                [0.951251242564198, -0.173648177666930, 0.254887002244179],
                [0.246137153725384, 0.925416578398323, -0.288133056037496],
                [-0.185842877388499, 0.336824088833465, 0.923044938291451],
            ],
        )
        rotation = euler_angles_to_rotation(
            ypr_rad=self._euler_angles,
            order="RYP",
        )
        np.testing.assert_allclose(rotation.as_matrix(), rotation_matrix, rtol=0, atol=self._tolerance)

    def test_euler_angles_to_rotation_pry(self):
        """Test euler_angles_to_rotation with PRY order."""
        rotation_matrix = np.asarray(
            [
                [0.966622809665280, -0.080554770457117, 0.243210346801694],
                [0.163175911166535, 0.925416578398323, -0.342020143325669],
                [-0.197519532830984, 0.370290541848075, 0.907673371190369],
            ],
        )
        rotation = euler_angles_to_rotation(
            ypr_rad=self._euler_angles,
            order="PRY",
        )
        np.testing.assert_allclose(rotation.as_matrix(), rotation_matrix, rtol=0, atol=self._tolerance)

    def test_euler_angles_to_rotation_pyr(self):
        """Testing compute rotation, scalar inputs"""
        rotation_matrix = np.asarray(
            [
                [0.951251242564198, -0.069094499922630, 0.300577816214889],
                [0.173648177666930, 0.925416578398323, -0.336824088833465],
                [-0.254887002244179, 0.372599123061208, 0.892301804089286],
            ],
        )
        rotation = euler_angles_to_rotation(
            ypr_rad=self._euler_angles,
            order="PYR",
        )
        np.testing.assert_allclose(rotation.as_matrix(), rotation_matrix, rtol=0, atol=self._tolerance)

    def test_euler_angles_to_rotation_vectorized(self):
        """Testing euler_angles_to_rotation, array inputs"""
        rotation_matrix = np.asarray(
            [
                [0.951251242564198, -0.069094499922630, 0.300577816214889],
                [0.173648177666930, 0.925416578398323, -0.336824088833465],
                [-0.254887002244179, 0.372599123061208, 0.892301804089286],
            ],
        )

        rotation = euler_angles_to_rotation(order="PYR", ypr_rad=self._euler_angles.reshape(1, -1))

        np.testing.assert_allclose(
            rotation.as_matrix(), rotation_matrix.reshape((1, 3, 3)), rtol=0, atol=self._tolerance
        )

        rotation = euler_angles_to_rotation(
            ypr_rad=np.tile(self._euler_angles, (2, 1)),
            order="PYR",
        )

        np.testing.assert_allclose(
            rotation.as_matrix(), np.tile(rotation_matrix, (2, 1, 1)), rtol=0, atol=self._tolerance
        )

    def test_euler_angles_to_rotation_invalid_orders(self):
        """Testing euler_angles_to_rotation, invalid rotation orders"""
        with pytest.raises(ValueError):
            euler_angles_to_rotation(order="PPP", ypr_rad=self._euler_angles)

        with pytest.raises(ValueError):
            euler_angles_to_rotation(order=None, ypr_rad=self._euler_angles)

        with pytest.raises(ValueError):
            euler_angles_to_rotation(order="xyz", ypr_rad=self._euler_angles)


class TestEulerAngles:
    """Testing rotation_to_euler_angles function"""

    @pytest.fixture(autouse=True)
    def setup_euler_angles_data(self):
        self.yaw = [np.deg2rad(10), np.deg2rad(20), np.deg2rad(30)]
        self.pitch = [np.deg2rad(15), np.deg2rad(25), np.deg2rad(35)]
        self.roll = [np.deg2rad(0), np.deg2rad(30), np.deg2rad(60)]
        self._euler_angles = np.stack([self.yaw, self.pitch, self.roll], axis=-1)
        self._tolerance = 1e-9

    @pytest.mark.parametrize("order", get_args(RotationOrder))
    def test_compute_euler_angles_roll(self, order):
        """Testing compute_euler_angles for roll only"""
        rotation = euler_angles_to_rotation(order=order, ypr_rad=np.array([0, 0, self.roll[0]]))
        euler_angles = rotation_to_euler_angles(order=order, rotation=rotation)

        np.testing.assert_allclose(euler_angles, np.array([0, 0, self.roll[0]]), atol=self._tolerance, rtol=0)

    @pytest.mark.parametrize("order", get_args(RotationOrder))
    def test_compute_euler_angles_pitch(self, order):
        """Testing compute_euler_angles for pitch only"""
        rotation = euler_angles_to_rotation(order=order, ypr_rad=np.array([0, self.pitch[0], 0]))
        euler_angles = rotation_to_euler_angles(order=order, rotation=rotation)

        np.testing.assert_allclose(euler_angles, np.array([0, self.pitch[0], 0]), atol=self._tolerance, rtol=0)

    @pytest.mark.parametrize("order", get_args(RotationOrder))
    def test_compute_euler_angles_yaw(self, order):
        """Testing compute_euler_angles for yaw only"""
        rotation = euler_angles_to_rotation(order=order, ypr_rad=np.array([self.yaw[0], 0, 0]))
        euler_angles = rotation_to_euler_angles(order=order, rotation=rotation)

        np.testing.assert_allclose(euler_angles, np.array([self.yaw[0], 0, 0]), atol=self._tolerance, rtol=0)

    def test_compute_euler_angles_scalar(self):
        """Testing compute_euler_angles for single values of yaw, pitch and roll"""
        rotation = euler_angles_to_rotation(order="YPR", ypr_rad=self._euler_angles[0, :])
        euler_angles = rotation_to_euler_angles(order="YPR", rotation=rotation)

        np.testing.assert_allclose(euler_angles, self._euler_angles[0, :], atol=self._tolerance, rtol=0)

    def test_compute_euler_angles_vectorized(self):
        """Testing compute_euler_angles for arrays of yaw, pitch and roll"""
        rotation = euler_angles_to_rotation(order="YPR", ypr_rad=self._euler_angles)
        euler_angles = rotation_to_euler_angles(order="YPR", rotation=rotation)

        np.testing.assert_allclose(euler_angles, self._euler_angles, atol=self._tolerance, rtol=0)

    @pytest.mark.parametrize("order", get_args(RotationOrder))
    def test_compute_euler_angles_vectorized_all_rotations(self, order):
        """Testing compute_euler_angles for arrays of yaw, pitch and roll, all rotation orders"""
        rotation = euler_angles_to_rotation(order=order, ypr_rad=self._euler_angles)
        euler_angles = rotation_to_euler_angles(order=order, rotation=rotation)
        np.testing.assert_allclose(
            self.yaw,
            euler_angles[:, 0],
            atol=self._tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            self.pitch,
            euler_angles[:, 1],
            atol=self._tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(
            self.roll,
            euler_angles[:, 2],
            atol=self._tolerance,
            rtol=0,
        )

    def test_compute_euler_angles_invalid_orders(self):
        """Testing compute_euler_angles with invalid rotation orders"""
        rotation = euler_angles_to_rotation(order="YPR", ypr_rad=self._euler_angles)
        with pytest.raises(ValueError):
            rotation_to_euler_angles(order="PPP", rotation=rotation)

        with pytest.raises(ValueError):
            rotation_to_euler_angles(order=None, rotation=rotation)

        with pytest.raises(ValueError):
            rotation_to_euler_angles(order="xyz", rotation=rotation)
