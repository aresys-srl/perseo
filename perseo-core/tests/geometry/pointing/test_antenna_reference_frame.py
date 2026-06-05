# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Testing geometry/utilities/antenna_reference_frame.py functionalities"""

import itertools
from typing import get_args

import numpy as np
import pytest

from perseo_core.geometry.pointing import (
    RotationOrder,
    compute_antenna_reference_frame_from_euler_angles,
    compute_euler_angles_from_antenna_reference_frame,
    compute_pointing_directions,
)
from tests.common import compute_antenna_angles_a_posteriori


class TestComputeAntennaReferenceFrameFromEulerAngles:
    """Test compute_antenna_reference_frame_from_euler_angles with various input dimensions and rotation orders."""

    @pytest.fixture(autouse=True)
    def setup_data(self, rotation_test_data: dict) -> None:
        self.yaw = rotation_test_data["yaw"]
        self.pitch = rotation_test_data["pitch"]
        self.roll = rotation_test_data["roll"]
        self.euler_angles = rotation_test_data["euler_angles"]
        self.arf_from_eye = rotation_test_data["arf_from_eye"]
        self.tolerance = rotation_test_data["tolerance"]

    def test_basic_computation_single_angles_identity_frame(self) -> None:
        """Test basic ARF computation with single angles and identity initial frame."""
        initial_ref_frame = np.eye(3)
        euler_angles = np.array([self.yaw, self.pitch, self.roll])

        arf = compute_antenna_reference_frame_from_euler_angles(
            ypr_rad=euler_angles,
            rotation_order="YPR",
            initial_reference_frame=initial_ref_frame,
        )

        np.testing.assert_allclose(arf, self.arf_from_eye, atol=self.tolerance)

    def test_input_shape_single_angles_as_1d_array(self) -> None:
        """Test (1, 3) array input produces proper output shape."""
        initial_ref_frame = np.eye(3)
        euler_angles = np.array([[self.yaw, self.pitch, self.roll]])

        arf = compute_antenna_reference_frame_from_euler_angles(
            ypr_rad=euler_angles,
            rotation_order="YPR",
            initial_reference_frame=initial_ref_frame,
        )

        assert arf.shape == (1, 3, 3)

    def test_input_shape_multiple_rotations(self) -> None:
        """Test (N, 3) input angles produce (N, 3, 3) output."""
        initial_ref_frame = np.eye(3)
        num_rotations = 5
        euler_angles_array = np.tile(self.euler_angles, (num_rotations, 1))

        arf = compute_antenna_reference_frame_from_euler_angles(
            ypr_rad=euler_angles_array,
            rotation_order="YPR",
            initial_reference_frame=initial_ref_frame,
        )

        assert arf.shape == (num_rotations, 3, 3)

    @pytest.mark.parametrize("order", get_args(RotationOrder))
    def test_rotation_order_all_supported_orders(self, order: RotationOrder) -> None:
        """Test all supported rotation orders (YPR, YRP, PRY, PYR, RYP, RPY)."""
        initial_ref_frame = np.eye(3)
        euler_angles = np.array([self.yaw, self.pitch, self.roll])

        arf = compute_antenna_reference_frame_from_euler_angles(
            ypr_rad=euler_angles,
            rotation_order=order,
            initial_reference_frame=initial_ref_frame,
        )
        assert arf.shape == (3, 3)

    def test_edge_case_zero_angles(self) -> None:
        """Test zero angles with identity frame produce identity rotation."""
        initial_ref_frame = np.eye(3)
        euler_angles = np.array([0, 0, 0])

        arf = compute_antenna_reference_frame_from_euler_angles(
            ypr_rad=euler_angles,
            rotation_order="YPR",
            initial_reference_frame=initial_ref_frame,
        )

        np.testing.assert_allclose(arf, initial_ref_frame, atol=self.tolerance)

    def test_compose_with_non_identity_initial_frame(self) -> None:
        """Test composition with non-identity initial frame."""
        initial_ref_frame = self.arf_from_eye
        euler_angles = np.array([-self.yaw, -self.pitch, -self.roll])

        arf = compute_antenna_reference_frame_from_euler_angles(
            ypr_rad=euler_angles,
            rotation_order="RPY",
            initial_reference_frame=initial_ref_frame,
        )

        np.testing.assert_allclose(arf, np.eye(3), atol=1e-7)

    def test_broadcasting_single_angles_multiple_initial_frames(self) -> None:
        """Test broadcasting scalar angles with multiple initial frames."""
        num_frames = 5
        initial_ref_frame = np.tile(np.eye(3), (num_frames, 1, 1))

        arf = compute_antenna_reference_frame_from_euler_angles(
            ypr_rad=self.euler_angles,
            rotation_order="YPR",
            initial_reference_frame=initial_ref_frame,
        )

        assert arf.shape == (num_frames, 3, 3)

    def test_broadcasting_multiple_angles_multiple_initial_frames(self) -> None:
        """Test broadcasting with matching multiple angles and frames."""
        num_frames = 3
        initial_ref_frames = np.tile(np.eye(3), (num_frames, 1, 1))
        euler_angles_array = np.tile(self.euler_angles, (num_frames, 1))

        arf = compute_antenna_reference_frame_from_euler_angles(
            ypr_rad=euler_angles_array,
            rotation_order="YPR",
            initial_reference_frame=initial_ref_frames,
        )

        assert arf.shape == (num_frames, 3, 3)
        np.testing.assert_allclose(arf, np.tile(self.arf_from_eye, (num_frames, 1, 1)), atol=self.tolerance)


class TestComputeEulerAnglesFromAntennaReferenceFrame:
    """Test compute_euler_angles_from_antenna_reference_frame with various input dimensions."""

    @pytest.fixture(autouse=True)
    def setup_data(self, rotation_test_data: dict) -> None:
        self.yaw = rotation_test_data["yaw"]
        self.pitch = rotation_test_data["pitch"]
        self.roll = rotation_test_data["roll"]
        self.euler_angles = rotation_test_data["euler_angles"]
        self.tolerance = rotation_test_data["tolerance"]
        self.arf_from_eye = rotation_test_data["arf_from_eye"]

    def test_single_rotation_recovery(self) -> None:
        """Test recovery of single euler angles from antenna reference frame."""
        euler_angles_out = compute_euler_angles_from_antenna_reference_frame(
            antenna_reference_frame=self.arf_from_eye,
            initial_reference_frame=np.eye(3),
            rotation_order="YPR",
        )

        np.testing.assert_allclose(euler_angles_out, np.array([self.yaw, self.pitch, self.roll]), atol=self.tolerance)

    def test_multiple_rotations_recovery(self) -> None:
        """Test recovery of multiple euler angles from antenna reference frames."""
        num_rotations = 5
        euler_angles_out = compute_euler_angles_from_antenna_reference_frame(
            antenna_reference_frame=np.tile(self.arf_from_eye, (num_rotations, 1, 1)),
            initial_reference_frame=np.tile(np.eye(3), (num_rotations, 1, 1)),
            rotation_order="YPR",
        )

        np.testing.assert_allclose(euler_angles_out, np.tile(self.euler_angles, (num_rotations, 1)), atol=1e-7)

    def test_non_identity_initial_frame_recovery(self) -> None:
        """Test recovery with non-identity initial reference frame."""
        euler_angles_out = compute_euler_angles_from_antenna_reference_frame(
            antenna_reference_frame=np.eye(3),
            initial_reference_frame=self.arf_from_eye,
            rotation_order="RPY",
        )
        np.testing.assert_allclose(
            euler_angles_out, np.array([-self.yaw, -self.pitch, -self.roll]), atol=self.tolerance
        )

    @pytest.mark.parametrize("order", get_args(RotationOrder))
    def test_all_rotation_orders_recovery(self, order: RotationOrder) -> None:
        """Test round trip."""
        initial_ref_frame = np.eye(3)
        euler_angles_in = np.array([self.yaw, self.pitch, self.roll])

        arf = compute_antenna_reference_frame_from_euler_angles(
            ypr_rad=euler_angles_in,
            rotation_order=order,
            initial_reference_frame=initial_ref_frame,
        )

        euler_angles_out = compute_euler_angles_from_antenna_reference_frame(
            antenna_reference_frame=arf,
            initial_reference_frame=initial_ref_frame,
            rotation_order=order,
        )

        np.testing.assert_allclose(euler_angles_out, euler_angles_in, atol=self.tolerance)

    @pytest.mark.parametrize("order", get_args(RotationOrder))
    def test_round_trip_multiple_angles(self, order: RotationOrder) -> None:
        """Test round-trip conversion for multiple different euler angles."""
        initial_ref_frame = np.eye(3)

        yaw_values = np.array([0, np.deg2rad(30), np.deg2rad(75), -np.deg2rad(40)])
        pitch_values = np.array([0, np.deg2rad(15), np.deg2rad(35), -np.deg2rad(25)])
        roll_values = np.array([0, np.deg2rad(60), -np.deg2rad(20), np.deg2rad(70)])

        euler_angles_in = np.column_stack([yaw_values, pitch_values, roll_values])

        arfs = compute_antenna_reference_frame_from_euler_angles(
            ypr_rad=euler_angles_in,
            rotation_order=order,
            initial_reference_frame=initial_ref_frame,
        )

        euler_angles_out = compute_euler_angles_from_antenna_reference_frame(
            antenna_reference_frame=arfs,
            initial_reference_frame=initial_ref_frame,
            rotation_order=order,
        )

        np.testing.assert_allclose(euler_angles_out, euler_angles_in, atol=1e-7)

    def test_shape_mismatch_error_multiple_initial_frames_multiple_arfs(self) -> None:
        """Test error when initial frames and arfs have mismatched multiple shapes."""
        with pytest.raises(ValueError, match="broadcast"):
            compute_euler_angles_from_antenna_reference_frame(
                antenna_reference_frame=np.tile(self.arf_from_eye, (3, 1, 1)),
                initial_reference_frame=np.tile(np.eye(3), (2, 1, 1)),
                rotation_order="YPR",
            )

    def test_broadcasting_single_initial_frame_multiple_arfs(self) -> None:
        """Test broadcasting with single initial frame and multiple arfs."""
        num_rotations = 4

        euler_angles_out = compute_euler_angles_from_antenna_reference_frame(
            antenna_reference_frame=np.tile(self.arf_from_eye, (num_rotations, 1, 1)),
            initial_reference_frame=np.tile(np.eye(3), (num_rotations, 1, 1)),
            rotation_order="YPR",
        )

        np.testing.assert_allclose(euler_angles_out, np.tile(self.euler_angles, (num_rotations, 1)), atol=1e-7)

    def test_broadcasting_multiple_initial_frames_single_arf(self) -> None:
        """Test broadcasting with multiple initial frames and single arf."""
        num_frames = 3

        euler_angles_out = compute_euler_angles_from_antenna_reference_frame(
            antenna_reference_frame=self.arf_from_eye,
            initial_reference_frame=np.tile(self.arf_from_eye, (num_frames, 1, 1)),
            rotation_order="YPR",
        )

        # Output should have shape (num_frames, 3)
        assert euler_angles_out.shape == (num_frames, 3)

    def test_output_shape_single_input(self) -> None:
        """Test output shape for single input."""
        initial_ref_frame = np.eye(3)

        euler_angles_out = compute_euler_angles_from_antenna_reference_frame(
            antenna_reference_frame=np.eye(3),
            initial_reference_frame=initial_ref_frame,
            rotation_order="YPR",
        )

        assert euler_angles_out.shape == (3,)

    def test_output_shape_multiple_inputs(self) -> None:
        """Test output shape for multiple inputs."""
        initial_ref_frame = np.eye(3)
        num_rotations = 5

        euler_angles_out = compute_euler_angles_from_antenna_reference_frame(
            antenna_reference_frame=np.tile(self.arf_from_eye, (num_rotations, 1, 1)),
            initial_reference_frame=initial_ref_frame,
            rotation_order="YPR",
        )

        assert euler_angles_out.shape == (num_rotations, 3)


class TestPointingDirections:
    """Testing compute pointing directions function"""

    @pytest.fixture(autouse=True)
    def setup_pointing_data(self) -> None:
        self._arf_in = np.array(
            [
                [-0.604580222175426, -0.564852727629684, -0.561626344684451],
                [-0.304829433093801, 0.815474732550541, -0.492016236816769],
                [0.735908806628936, -0.126263045507894, -0.665203631728697],
            ]
        )
        self._tolerance = 1e-8

    def test_compute_pointing_directions(self) -> None:
        boresight_dir = compute_pointing_directions(
            antenna_reference_frame=self._arf_in,
            azimuth_antenna_angles=0,
            elevation_antenna_angles=0,
        )
        np.testing.assert_allclose(boresight_dir, self._arf_in[:, 2], rtol=0, atol=self._tolerance)

    def test_compute_pointing_directions_vectorized(self) -> None:
        """Testing compute pointing directions, vectorized"""

        num_elements = 10
        arf_inputs = [self._arf_in, np.tile(self._arf_in, (num_elements, 1, 1))]

        az_angles_in = np.deg2rad(np.linspace(-5, 5, 10))
        el_angles_in = np.deg2rad(np.linspace(-3, 2, 10))

        azimuth_angles_inputs = [az_angles_in, az_angles_in[0]]
        elevation_angles_inputs = [el_angles_in, el_angles_in[4]]

        for arf, azimuth_angles, elevation_angles in itertools.product(
            arf_inputs, azimuth_angles_inputs, elevation_angles_inputs
        ):
            directions = compute_pointing_directions(arf, azimuth_angles, elevation_angles)

            expected_shape = (3,)
            if arf.ndim == 3 or np.size(azimuth_angles) > 1 or np.size(elevation_angles) > 1:
                expected_shape = (num_elements, *expected_shape)

            assert directions.shape == expected_shape

            np.testing.assert_allclose(
                np.linalg.norm(directions, axis=-1),
                np.ones_like(azimuth_angles),
                rtol=0,
                atol=self._tolerance,
            )

            azimuth_out, elevation_out = compute_antenna_angles_a_posteriori(self._arf_in, directions)

            assert np.max(np.abs(azimuth_out - azimuth_angles)) < self._tolerance
            assert np.max(np.abs(elevation_out - elevation_angles)) < self._tolerance

    def test_compute_pointing_directions_invalid_inputs(self) -> None:
        arf_inputs = [
            np.ones((10, 3, 3)),
            np.ones((10, 3, 3)),
            np.ones((10, 3, 3)),
            np.ones((10, 3, 3)),
            np.ones((3, 3)),
            np.ones((3, 2)),
            np.ones((10, 3, 2)),
        ]
        azimuth_angles_inputs = [
            np.arange(5),
            np.arange(5),
            1,
            np.arange(10),
            np.arange(5),
            1,
            1,
        ]
        elevation_angles_inputs = [
            np.arange(5),
            1,
            np.arange(5),
            np.arange(3),
            np.arange(3),
            1,
            np.arange(10),
        ]

        for arf, azimuth_angles, elevation_angles in zip(
            arf_inputs,
            azimuth_angles_inputs,
            elevation_angles_inputs,
            strict=True,
        ):
            with pytest.raises(ValueError, match="broadcast"):
                compute_pointing_directions(arf, azimuth_angles, elevation_angles)
