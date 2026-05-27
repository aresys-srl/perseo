# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for point_target_analysis/rcs_geometric_computation.py core functionalities"""

from __future__ import annotations

import numpy as np
import pytest

from perseo_quality.point_targets_analysis.rcs_geometric_computation import compute_rcs_trihedral_corner_reflector


class TestRCSComputation:
    """Testing rcs computation functionalities"""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.cr_arm = 0.7
        self.wavelength = 0.055
        self.elevation_rad = np.array([0, np.pi / 4, np.pi / 3])
        self.azimuth_rad = np.array([0, np.pi / 4, np.pi / 3])

        # expected results
        self.rcs_expected_array = np.array([0.0, 286.0556891916635, 66.34793846512898])
        self.rcs_expected_invalid = np.array([0.0, np.nan, np.nan])

    def test_rcs_trihedral_scalar(self):
        rcs = compute_rcs_trihedral_corner_reflector(
            cr_arm_length_m=self.cr_arm,
            wavelength_m=self.wavelength,
            elevation_rad=self.elevation_rad[1],
            azimuth_rad=self.azimuth_rad[1],
        )
        assert isinstance(rcs, float)
        np.testing.assert_allclose(rcs, self.rcs_expected_array[1], atol=1e-8, rtol=0)

    def test_rcs_trihedral_array(self):
        rcs = compute_rcs_trihedral_corner_reflector(
            cr_arm_length_m=self.cr_arm,
            wavelength_m=self.wavelength,
            elevation_rad=self.elevation_rad,
            azimuth_rad=self.azimuth_rad,
        )
        assert isinstance(rcs, np.ndarray)
        assert rcs.size == self.elevation_rad.size
        assert rcs.shape == self.elevation_rad.shape
        np.testing.assert_allclose(rcs, self.rcs_expected_array, atol=1e-8, rtol=0)

    def test_rcs_trihedral_invalid(self):
        rcs = compute_rcs_trihedral_corner_reflector(
            cr_arm_length_m=self.cr_arm,
            wavelength_m=self.wavelength,
            elevation_rad=-self.elevation_rad,
            azimuth_rad=-self.azimuth_rad,
        )
        assert isinstance(rcs, np.ndarray)
        assert rcs.size == self.elevation_rad.size
        assert rcs.shape == self.elevation_rad.shape
        np.testing.assert_allclose(rcs, self.rcs_expected_invalid, atol=1e-8, rtol=0)
