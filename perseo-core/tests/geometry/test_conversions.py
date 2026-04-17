# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Testing geometry/coords_conversions.py functionalities"""

import unittest

import numpy as np

from perseo_core.geometry.coords_conversions import llh2xyz, xyz2llh
from tests.fixtures.geometry_data import get_coords_conversions_test_data


class CoordsConversionsTest(unittest.TestCase):
    """Test coordinate conversion functions xyz2llh and llh2xyz with various input types and options."""

    def setUp(self) -> None:
        # Load test data from fixtures
        data = get_coords_conversions_test_data()
        self.xyz = data["xyz"]
        self.llh = data["llh"]
        self.llh_deg = data["llh_deg"]
        self.xyz_vec = data["xyz_vec"]
        self.llh_vec = data["llh_vec"]
        self.llh_vec_deg = data["llh_vec_deg"]
        self.atol = data["tolerance"]["atol"]
        self.rtol = data["tolerance"]["rtol"]

    def test_xyz2llh_from_list(self) -> None:
        """Test xyz2llh with list input in radians mode."""
        llh = xyz2llh(self.xyz)
        self.assertEqual(llh.shape, (3,))
        np.testing.assert_allclose(llh, np.array(self.llh), atol=self.atol, rtol=self.rtol)

    def test_xyz2llh_from_list_deg(self) -> None:
        """Test xyz2llh with list input in degrees mode."""
        llh = xyz2llh(self.xyz, radians=False)
        llh[:2] = np.deg2rad(llh[:2])
        self.assertEqual(llh.shape, (3,))
        np.testing.assert_allclose(llh, np.array(self.llh), atol=self.atol, rtol=self.rtol)

    def test_xyz2llh_from_array(self) -> None:
        """Test xyz2llh with numpy array (1D) input in radians mode."""
        llh = xyz2llh(np.array(self.xyz))
        self.assertEqual(llh.shape, (3,))
        np.testing.assert_allclose(llh, np.array(self.llh), atol=self.atol, rtol=self.rtol)

    def test_xyz2llh_from_array_deg(self) -> None:
        """Test xyz2llh with numpy array (1D) input in degrees mode."""
        llh = xyz2llh(np.array(self.xyz), radians=False)
        llh[:2] = np.deg2rad(llh[:2])
        self.assertEqual(llh.shape, (3,))
        np.testing.assert_allclose(llh, np.array(self.llh), atol=self.atol, rtol=self.rtol)

    def test_llh2xyz_from_list(self) -> None:
        """Test llh2xyz with list input in radians mode."""
        xyz = llh2xyz(self.llh)
        self.assertEqual(xyz.shape, (3,))
        np.testing.assert_allclose(xyz, np.array(self.xyz), atol=self.atol, rtol=self.rtol)

    def test_llh2xyz_from_list_deg(self) -> None:
        """Test llh2xyz with list input in degrees mode."""
        xyz = llh2xyz(self.llh_deg, radians=False)
        self.assertEqual(xyz.shape, (3,))
        np.testing.assert_allclose(xyz, np.array(self.xyz), atol=self.atol, rtol=self.rtol)

    def test_llh2xyz_from_array(self) -> None:
        """Test llh2xyz with numpy array (1D) input in radians mode."""
        xyz = llh2xyz(np.array(self.llh))
        self.assertEqual(xyz.shape, (3,))
        np.testing.assert_allclose(xyz, np.array(self.xyz), atol=self.atol, rtol=self.rtol)

    def test_llh2xyz_from_array_deg(self) -> None:
        """Test llh2xyz with numpy array (1D) input in degrees mode."""
        xyz = llh2xyz(np.array(self.llh_deg), radians=False)
        self.assertEqual(xyz.shape, (3,))
        np.testing.assert_allclose(xyz, np.array(self.xyz), atol=self.atol, rtol=self.rtol)

    def test_xyz2llh_from_array_2d(self) -> None:
        """Test xyz2llh with numpy array (2D/vectorized) input in radians mode."""
        llh = xyz2llh(self.xyz_vec)
        self.assertEqual(llh.shape, self.xyz_vec.shape)
        np.testing.assert_allclose(llh, self.llh_vec, atol=self.atol, rtol=self.rtol)

    def test_xyz2llh_from_array_2d_deg(self) -> None:
        """Test xyz2llh with numpy array (2D/vectorized) input in degrees mode."""
        llh = xyz2llh(self.xyz_vec, radians=False)
        llh[:, :2] = np.deg2rad(llh[:, :2])
        self.assertEqual(llh.shape, self.xyz_vec.shape)
        np.testing.assert_allclose(llh, self.llh_vec, atol=self.atol, rtol=self.rtol)

    def test_llh2xyz_from_array_2d(self) -> None:
        """Test llh2xyz with numpy array (2D/vectorized) input in radians mode."""
        xyz = llh2xyz(self.llh_vec)
        self.assertEqual(xyz.shape, self.llh_vec.shape)
        np.testing.assert_allclose(xyz, self.xyz_vec, atol=self.atol, rtol=self.rtol)

    def test_llh2xyz_from_array_2d_deg(self) -> None:
        """Test llh2xyz with numpy array (2D/vectorized) input in degrees mode."""

        xyz = llh2xyz(self.llh_vec_deg, radians=False)
        self.assertEqual(xyz.shape, self.llh_vec.shape)
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


if __name__ == "__main__":
    unittest.main()
