# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Testing geometry/coords_conversions.py functionalities"""

import unittest

import numpy as np
from pyproj import Geod

from perseo_core.geometry.coords_conversions import llh2xyz, xyz2llh

WGS84 = Geod(ellps="WGS84")


class CoordsConversionsTest(unittest.TestCase):
    """Testing conversions xyz2llh and llh2xyz"""

    def setUp(self) -> None:
        self.xyz = [2.354828227500000e4, 9.457755947560000e05, 6.286558297154000e06]
        self.llh = [1.4224117467026256, 1.5459030877183293, 123.45599971152842]
        self.llh_deg = [np.rad2deg(self.llh[0]), np.rad2deg(self.llh[1]), self.llh[2]]
        self.xyz_vec = np.stack(
            [
                [5336078.7743305163, WGS84.a, 0.0, -WGS84.a, 0.0],
                [2346746.5942683504, 0.0, WGS84.a, -WGS84.a, 0.0],
                [4033846.0446414836, 0.0, 0.0, 0.0, WGS84.b],
            ],
            axis=-1,
        )
        self.llh_vec = np.stack(
            [
                [0.608159140099359, 0.0, 0.0, 0.0, np.pi / 2],
                [0.414329746479487, 0.0, np.pi / 2, -2.35619449019234, 0.0],
                [717733.676999941, 0.0, 0.0, 2641910.84807364, 0.0],
            ],
            axis=-1,
        )
        self.llh_vec_deg = self.llh_vec.copy()
        self.llh_vec_deg[:, :2] = np.rad2deg(self.llh_vec_deg[:, :2])

        self.atol = 1e-8
        self.rtol = 1e-8

    def test_xyz2llh_from_list(self) -> None:
        """Testing xyz2llh from list"""
        llh = xyz2llh(self.xyz)
        self.assertEqual(llh.shape, (3,))
        np.testing.assert_allclose(llh, np.array(self.llh), atol=self.atol, rtol=self.rtol)

    def test_xyz2llh_from_list_deg(self) -> None:
        """Testing xyz2llh from list, output in deg"""
        llh = xyz2llh(self.xyz, radians=False)
        llh[:2] = np.deg2rad(llh[:2])
        self.assertEqual(llh.shape, (3,))
        np.testing.assert_allclose(llh, np.array(self.llh), atol=self.atol, rtol=self.rtol)

    def test_xyz2llh_from_array(self) -> None:
        """Testing xyz2llh from array"""
        llh = xyz2llh(np.array(self.xyz))
        self.assertEqual(llh.shape, (3,))
        np.testing.assert_allclose(llh, np.array(self.llh), atol=self.atol, rtol=self.rtol)

    def test_xyz2llh_from_array_deg(self) -> None:
        """Testing xyz2llh from array, output in deg"""
        llh = xyz2llh(np.array(self.xyz), radians=False)
        llh[:2] = np.deg2rad(llh[:2])
        self.assertEqual(llh.shape, (3,))
        np.testing.assert_allclose(llh, np.array(self.llh), atol=self.atol, rtol=self.rtol)

    def test_llh2xyz_from_list(self) -> None:
        """Testing llh2xyz from list"""
        xyz = llh2xyz(self.llh)
        self.assertEqual(xyz.shape, (3,))
        np.testing.assert_allclose(xyz, np.array(self.xyz), atol=self.atol, rtol=self.rtol)

    def test_llh2xyz_from_list_deg(self) -> None:
        """Testing llh2xyz from list, input in deg"""
        xyz = llh2xyz(self.llh_deg, radians=False)
        self.assertEqual(xyz.shape, (3,))
        np.testing.assert_allclose(xyz, np.array(self.xyz), atol=self.atol, rtol=self.rtol)

    def test_llh2xyz_from_array(self) -> None:
        """Testing llh2xyz from array"""
        xyz = llh2xyz(np.array(self.llh))
        self.assertEqual(xyz.shape, (3,))
        np.testing.assert_allclose(xyz, np.array(self.xyz), atol=self.atol, rtol=self.rtol)

    def test_llh2xyz_from_array_deg(self) -> None:
        """Testing llh2xyz from array, input in deg"""
        xyz = llh2xyz(np.array(self.llh_deg), radians=False)
        self.assertEqual(xyz.shape, (3,))
        np.testing.assert_allclose(xyz, np.array(self.xyz), atol=self.atol, rtol=self.rtol)

    def test_xyz2llh_from_array_2d(self) -> None:
        """Testing xyz2llh from array 2D"""
        llh = xyz2llh(self.xyz_vec)
        self.assertEqual(llh.shape, self.xyz_vec.shape)
        np.testing.assert_allclose(llh, self.llh_vec, atol=self.atol, rtol=self.rtol)

    def test_xyz2llh_from_array_2d_deg(self) -> None:
        """Testing xyz2llh from array 2D, output in deg"""
        llh = xyz2llh(self.xyz_vec, radians=False)
        llh[:, :2] = np.deg2rad(llh[:, :2])
        self.assertEqual(llh.shape, self.xyz_vec.shape)
        np.testing.assert_allclose(llh, self.llh_vec, atol=self.atol, rtol=self.rtol)

    def test_llh2xyz_from_array_2d(self) -> None:
        """Testing llh2xyz from array 2D"""
        xyz = llh2xyz(self.llh_vec)
        self.assertEqual(xyz.shape, self.llh_vec.shape)
        np.testing.assert_allclose(xyz, self.xyz_vec, atol=self.atol, rtol=self.rtol)

    def test_llh2xyz_from_array_2d_deg(self) -> None:
        """Testing llh2xyz from array 2D, input in deg"""
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
