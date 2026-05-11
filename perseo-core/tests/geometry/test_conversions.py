# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Testing geometry/coords_conversions.py functionalities"""

import numpy as np
import pytest

from perseo_core.geometry.coords_conversions import llh2xyz, xyz2llh


class TestCoordsConversions:
    """Test coordinate conversion functions xyz2llh and llh2xyz with various input types and options."""

    @pytest.fixture(autouse=True)
    def setup_coords_conversions_data(self, coords_conversions_test_data):
        """Load test data from fixtures."""
        self.xyz = coords_conversions_test_data["xyz"]
        self.llh = coords_conversions_test_data["llh"]
        self.llh_deg = coords_conversions_test_data["llh_deg"]
        self.xyz_vec = coords_conversions_test_data["xyz_vec"]
        self.llh_vec = coords_conversions_test_data["llh_vec"]
        self.llh_vec_deg = coords_conversions_test_data["llh_vec_deg"]
        self.atol = coords_conversions_test_data["tolerance"]["atol"]
        self.rtol = coords_conversions_test_data["tolerance"]["rtol"]

    def test_xyz2llh_from_list(self) -> None:
        """Test xyz2llh with list input in radians mode."""
        llh = xyz2llh(self.xyz)
        assert llh.shape == (3,)
        np.testing.assert_allclose(llh, np.array(self.llh), atol=self.atol, rtol=self.rtol)

    def test_xyz2llh_from_list_deg(self) -> None:
        """Test xyz2llh with list input in degrees mode."""
        llh = xyz2llh(self.xyz, radians=False)
        llh[:2] = np.deg2rad(llh[:2])
        assert llh.shape == (3,)
        np.testing.assert_allclose(llh, np.array(self.llh), atol=self.atol, rtol=self.rtol)

    def test_xyz2llh_from_array(self) -> None:
        """Test xyz2llh with numpy array (1D) input in radians mode."""
        llh = xyz2llh(np.array(self.xyz))
        assert llh.shape == (3,)
        np.testing.assert_allclose(llh, np.array(self.llh), atol=self.atol, rtol=self.rtol)

    def test_xyz2llh_from_array_deg(self) -> None:
        """Test xyz2llh with numpy array (1D) input in degrees mode."""
        llh = xyz2llh(np.array(self.xyz), radians=False)
        llh[:2] = np.deg2rad(llh[:2])
        assert llh.shape == (3,)
        np.testing.assert_allclose(llh, np.array(self.llh), atol=self.atol, rtol=self.rtol)

    def test_llh2xyz_from_list(self) -> None:
        """Test llh2xyz with list input in radians mode."""
        xyz = llh2xyz(self.llh)
        assert xyz.shape == (3,)
        np.testing.assert_allclose(xyz, np.array(self.xyz), atol=self.atol, rtol=self.rtol)

    def test_llh2xyz_from_list_deg(self) -> None:
        """Test llh2xyz with list input in degrees mode."""
        xyz = llh2xyz(self.llh_deg, radians=False)
        assert xyz.shape == (3,)
        np.testing.assert_allclose(xyz, np.array(self.xyz), atol=self.atol, rtol=self.rtol)

    def test_llh2xyz_from_array(self) -> None:
        """Test llh2xyz with numpy array (1D) input in radians mode."""
        xyz = llh2xyz(np.array(self.llh))
        assert xyz.shape == (3,)
        np.testing.assert_allclose(xyz, np.array(self.xyz), atol=self.atol, rtol=self.rtol)

    def test_llh2xyz_from_array_deg(self) -> None:
        """Test llh2xyz with numpy array (1D) input in degrees mode."""
        xyz = llh2xyz(np.array(self.llh_deg), radians=False)
        assert xyz.shape == (3,)
        np.testing.assert_allclose(xyz, np.array(self.xyz), atol=self.atol, rtol=self.rtol)

    def test_xyz2llh_from_array_2d(self) -> None:
        """Test xyz2llh with numpy array (2D/vectorized) input in radians mode."""
        llh = xyz2llh(self.xyz_vec)
        assert llh.shape == self.xyz_vec.shape
        np.testing.assert_allclose(llh, self.llh_vec, atol=self.atol, rtol=self.rtol)

    def test_xyz2llh_from_array_2d_deg(self) -> None:
        """Test xyz2llh with numpy array (2D/vectorized) input in degrees mode."""
        llh = xyz2llh(self.xyz_vec, radians=False)
        llh[:, :2] = np.deg2rad(llh[:, :2])
        assert llh.shape == self.xyz_vec.shape
        np.testing.assert_allclose(llh, self.llh_vec, atol=self.atol, rtol=self.rtol)

    def test_llh2xyz_from_array_2d(self) -> None:
        """Test llh2xyz with numpy array (2D/vectorized) input in radians mode."""
        xyz = llh2xyz(self.llh_vec)
        assert xyz.shape == self.llh_vec.shape
        np.testing.assert_allclose(xyz, self.xyz_vec, atol=self.atol, rtol=self.rtol)

    def test_llh2xyz_from_array_2d_deg(self) -> None:
        """Test llh2xyz with numpy array (2D/vectorized) input in degrees mode."""

        xyz = llh2xyz(self.llh_vec_deg, radians=False)
        assert xyz.shape == self.llh_vec.shape
        np.testing.assert_allclose(xyz, self.xyz_vec, atol=self.atol, rtol=self.rtol)

    def test_multiple_application_1(self) -> None:
        """Testing application of xyz2llh and llh2xyz from array 2D"""
        llh = xyz2llh(self.xyz_vec)
        xyz = llh2xyz(llh)
        np.testing.assert_allclose(xyz, self.xyz_vec, atol=self.atol, rtol=self.rtol)

    def test_multiple_application_1_deg(self) -> None:
        """Testing application of xyz2llh and llh2xyz from array 2D, output in deg"""
        llh = xyz2llh(self.xyz_vec, radians=False)
        xyz = llh2xyz(llh, radians=False)
        np.testing.assert_allclose(xyz, self.xyz_vec, atol=self.atol, rtol=self.rtol)

    def test_multiple_application_2(self) -> None:
        """Testing application of llh2xyz and xyz2llh from array 2D"""
        xyz = llh2xyz(self.llh_vec)
        llh = xyz2llh(xyz)
        np.testing.assert_allclose(llh, self.llh_vec, atol=self.atol, rtol=self.rtol)

    def test_multiple_application_2_deg(self) -> None:
        """Testing application of llh2xyz and xyz2llh from array 2D, input in deg"""
        xyz = llh2xyz(self.llh_vec_deg, radians=False)
        llh = xyz2llh(xyz)
        np.testing.assert_allclose(llh, self.llh_vec, atol=self.atol, rtol=self.rtol)
