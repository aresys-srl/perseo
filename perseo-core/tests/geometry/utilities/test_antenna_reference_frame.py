# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Testing geometry/utilities/antenna_reference_frame.py functionalities"""

import unittest
from typing import get_args

import numpy as np

from perseo_core.geometry.utilities.antenna_reference_frame import (
    compute_antenna_reference_frame_from_euler_angles,
    compute_euler_angles_from_antenna_reference_frame,
)
from perseo_core.geometry.utilities.rotations import RotationOrder
from tests.fixtures.geometry_utilities_data import get_rotation_test_data


class ComputeAntennaReferenceFrameFromEulerAnglesTest(unittest.TestCase):
    """Test compute_antenna_reference_frame_from_euler_angles with various input dimensions and rotation orders."""

    def setUp(self):
        """Load test data from fixtures."""
        data = get_rotation_test_data()
        self.yaw = data["yaw"]
        self.pitch = data["pitch"]
        self.roll = data["roll"]
        self.euler_angles = data["euler_angles"]
        self.arf_from_eye = data["arf_from_eye"]
        self.tolerance = data["tolerance"]

    def test_basic_computation_single_angles_identity_frame(self):
        """Test basic ARF computation with single angles and identity initial frame."""
        initial_ref_frame = np.eye(3)
        euler_angles = np.array([self.yaw, self.pitch, self.roll])

        arf = compute_antenna_reference_frame_from_euler_angles(
            ypr_rad=euler_angles,
            rotation_order="YPR",
            initial_reference_frame_axis=initial_ref_frame,
        )

        np.testing.assert_allclose(arf, self.arf_from_eye, atol=self.tolerance)

    def test_input_shape_single_angles_as_1d_array(self):
        """Test (1, 3) array input produces proper output shape."""
        initial_ref_frame = np.eye(3)
        euler_angles = np.array([[self.yaw, self.pitch, self.roll]])

        arf = compute_antenna_reference_frame_from_euler_angles(
            ypr_rad=euler_angles,
            rotation_order="YPR",
            initial_reference_frame_axis=initial_ref_frame,
        )

        self.assertEqual(arf.shape, (1, 3, 3))

    def test_input_shape_multiple_rotations(self):
        """Test (N, 3) input angles produce (N, 3, 3) output."""
        initial_ref_frame = np.eye(3)
        num_rotations = 5
        euler_angles_array = np.tile(self.euler_angles, (num_rotations, 1))

        arf = compute_antenna_reference_frame_from_euler_angles(
            ypr_rad=euler_angles_array,
            rotation_order="YPR",
            initial_reference_frame_axis=initial_ref_frame,
        )

        self.assertEqual(arf.shape, (num_rotations, 3, 3))

    def test_rotation_order_all_supported_orders(self):
        """Test all supported rotation orders (YPR, YRP, PRY, PYR, RYP, RPY)."""
        initial_ref_frame = np.eye(3)
        euler_angles = np.array([self.yaw, self.pitch, self.roll])

        for order in get_args(RotationOrder):
            with self.subTest(rotation_order=order):
                arf = compute_antenna_reference_frame_from_euler_angles(
                    ypr_rad=euler_angles,
                    rotation_order=order,
                    initial_reference_frame_axis=initial_ref_frame,
                )
                self.assertEqual(arf.shape, (3, 3))

    def test_edge_case_zero_angles(self):
        """Test zero angles with identity frame produce identity rotation."""
        initial_ref_frame = np.eye(3)
        euler_angles = np.array([0, 0, 0])

        arf = compute_antenna_reference_frame_from_euler_angles(
            ypr_rad=euler_angles,
            rotation_order="YPR",
            initial_reference_frame_axis=initial_ref_frame,
        )

        np.testing.assert_allclose(arf, initial_ref_frame, atol=self.tolerance)

    def test_compose_with_non_identity_initial_frame(self):
        """Test composition with non-identity initial frame."""
        initial_ref_frame = self.arf_from_eye
        euler_angles = np.array([-self.yaw, -self.pitch, -self.roll])

        arf = compute_antenna_reference_frame_from_euler_angles(
            ypr_rad=euler_angles,
            rotation_order="RPY",
            initial_reference_frame_axis=initial_ref_frame,
        )

        np.testing.assert_allclose(arf, np.eye(3), atol=1e-7)

    def test_broadcasting_single_angles_multiple_initial_frames(self):
        """Test broadcasting scalar angles with multiple initial frames."""
        num_frames = 5
        initial_ref_frame = np.tile(np.eye(3), (num_frames, 1, 1))

        arf = compute_antenna_reference_frame_from_euler_angles(
            ypr_rad=self.euler_angles,
            rotation_order="YPR",
            initial_reference_frame_axis=initial_ref_frame,
        )

        self.assertEqual(arf.shape, (num_frames, 3, 3))

    def test_broadcasting_multiple_angles_multiple_initial_frames(self):
        """Test broadcasting with matching multiple angles and frames."""
        num_frames = 3
        initial_ref_frames = np.tile(np.eye(3), (num_frames, 1, 1))
        euler_angles_array = np.tile(self.euler_angles, (num_frames, 1))

        arf = compute_antenna_reference_frame_from_euler_angles(
            ypr_rad=euler_angles_array,
            rotation_order="YPR",
            initial_reference_frame_axis=initial_ref_frames,
        )

        self.assertEqual(arf.shape, (num_frames, 3, 3))
        np.testing.assert_allclose(arf, np.tile(self.arf_from_eye, (num_frames, 1, 1)), atol=self.tolerance)


class ComputeEulerAnglesFromAntennaReferenceFrameTest(unittest.TestCase):
    """Test compute_euler_angles_from_antenna_reference_frame with various input dimensions."""

    def setUp(self):
        """Load test data from fixtures."""
        data = get_rotation_test_data()
        self.yaw = data["yaw"]
        self.pitch = data["pitch"]
        self.roll = data["roll"]
        self.euler_angles = data["euler_angles"]
        self.tolerance = data["tolerance"]
        self.arf_from_eye = data["arf_from_eye"]

    def test_single_rotation_recovery(self):
        """Test recovery of single euler angles from antenna reference frame."""
        euler_angles_out = compute_euler_angles_from_antenna_reference_frame(
            antenna_reference_frame=self.arf_from_eye,
            initial_reference_frame_axis=np.eye(3),
            rotation_order="YPR",
        )

        np.testing.assert_allclose(euler_angles_out, np.array([self.yaw, self.pitch, self.roll]), atol=self.tolerance)

    def test_multiple_rotations_recovery(self):
        """Test recovery of multiple euler angles from antenna reference frames."""
        num_rotations = 5
        euler_angles_out = compute_euler_angles_from_antenna_reference_frame(
            antenna_reference_frame=np.tile(self.arf_from_eye, (num_rotations, 1, 1)),
            initial_reference_frame_axis=np.tile(np.eye(3), (num_rotations, 1, 1)),
            rotation_order="YPR",
        )

        np.testing.assert_allclose(euler_angles_out, np.tile(self.euler_angles, (num_rotations, 1)), atol=1e-7)

    def test_non_identity_initial_frame_recovery(self):
        """Test recovery with non-identity initial reference frame."""
        euler_angles_out = compute_euler_angles_from_antenna_reference_frame(
            antenna_reference_frame=np.eye(3),
            initial_reference_frame_axis=self.arf_from_eye,
            rotation_order="RPY",
        )
        np.testing.assert_allclose(
            euler_angles_out, np.array([-self.yaw, -self.pitch, -self.roll]), atol=self.tolerance
        )

    def test_all_rotation_orders_recovery(self):
        """Test round trip."""
        initial_ref_frame = np.eye(3)
        euler_angles_in = np.array([self.yaw, self.pitch, self.roll])

        for order in get_args(RotationOrder):
            with self.subTest(rotation_order=order):
                arf = compute_antenna_reference_frame_from_euler_angles(
                    ypr_rad=euler_angles_in,
                    rotation_order=order,
                    initial_reference_frame_axis=initial_ref_frame,
                )

                euler_angles_out = compute_euler_angles_from_antenna_reference_frame(
                    antenna_reference_frame=arf,
                    initial_reference_frame_axis=initial_ref_frame,
                    rotation_order=order,
                )

                np.testing.assert_allclose(euler_angles_out, euler_angles_in, atol=self.tolerance)

    def test_round_trip_multiple_angles(self):
        """Test round-trip conversion for multiple different euler angles."""
        initial_ref_frame = np.eye(3)

        yaw_values = np.array([0, np.deg2rad(30), np.deg2rad(75), -np.deg2rad(40)])
        pitch_values = np.array([0, np.deg2rad(15), np.deg2rad(35), -np.deg2rad(25)])
        roll_values = np.array([0, np.deg2rad(60), -np.deg2rad(20), np.deg2rad(70)])

        euler_angles_in = np.column_stack([yaw_values, pitch_values, roll_values])

        for order in get_args(RotationOrder):
            with self.subTest(rotation_order=order):
                arfs = compute_antenna_reference_frame_from_euler_angles(
                    ypr_rad=euler_angles_in,
                    rotation_order=order,
                    initial_reference_frame_axis=initial_ref_frame,
                )

                euler_angles_out = compute_euler_angles_from_antenna_reference_frame(
                    antenna_reference_frame=arfs,
                    initial_reference_frame_axis=initial_ref_frame,
                    rotation_order=order,
                )

                np.testing.assert_allclose(euler_angles_out, euler_angles_in, atol=1e-7)

    def test_shape_mismatch_error_multiple_initial_frames_multiple_arfs(self):
        """Test error when initial frames and arfs have mismatched multiple shapes."""
        with self.assertRaises(RuntimeError) as context:
            compute_euler_angles_from_antenna_reference_frame(
                antenna_reference_frame=np.tile(self.arf_from_eye, (3, 1, 1)),
                initial_reference_frame_axis=np.tile(np.eye(3), (2, 1, 1)),
                rotation_order="YPR",
            )

        self.assertIn("input shape mismatch", str(context.exception))

    def test_broadcasting_single_initial_frame_multiple_arfs(self):
        """Test broadcasting with single initial frame and multiple arfs."""
        num_rotations = 4

        euler_angles_out = compute_euler_angles_from_antenna_reference_frame(
            antenna_reference_frame=np.tile(self.arf_from_eye, (num_rotations, 1, 1)),
            initial_reference_frame_axis=np.tile(np.eye(3), (num_rotations, 1, 1)),
            rotation_order="YPR",
        )

        np.testing.assert_allclose(euler_angles_out, np.tile(self.euler_angles, (num_rotations, 1)), atol=1e-7)

    def test_broadcasting_multiple_initial_frames_single_arf(self):
        """Test broadcasting with multiple initial frames and single arf."""
        num_frames = 3

        euler_angles_out = compute_euler_angles_from_antenna_reference_frame(
            antenna_reference_frame=self.arf_from_eye,
            initial_reference_frame_axis=np.tile(self.arf_from_eye, (num_frames, 1, 1)),
            rotation_order="YPR",
        )

        # Output should have shape (num_frames, 3)
        self.assertEqual(euler_angles_out.shape, (num_frames, 3))

    def test_output_shape_single_input(self):
        """Test output shape for single input."""
        initial_ref_frame = np.eye(3)

        euler_angles_out = compute_euler_angles_from_antenna_reference_frame(
            antenna_reference_frame=np.eye(3),
            initial_reference_frame_axis=initial_ref_frame,
            rotation_order="YPR",
        )

        self.assertEqual(euler_angles_out.shape, (3,))

    def test_output_shape_multiple_inputs(self):
        """Test output shape for multiple inputs."""
        initial_ref_frame = np.eye(3)
        num_rotations = 5

        euler_angles_out = compute_euler_angles_from_antenna_reference_frame(
            antenna_reference_frame=np.tile(self.arf_from_eye, (num_rotations, 1, 1)),
            initial_reference_frame_axis=initial_ref_frame,
            rotation_order="YPR",
        )

        self.assertEqual(euler_angles_out.shape, (num_rotations, 3))


if __name__ == "__main__":
    unittest.main()
