# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Testing geometry/coords_conversions.py functionalities"""

import numpy as np
import pytest

from perseo_core.geometry.coords_conversions import ecef2eci, eci2ecef, llh2xyz, xyz2llh


class TestXYZ_LLHCoordsConversions:
    """Test coordinate conversion functions xyz2llh and llh2xyz with various input types and options."""

    @pytest.fixture(autouse=True)
    def setup_coords_conversions_data(self, xyz2llh_coords_conversions_test_data):
        """Load test data from fixtures."""
        self.xyz = xyz2llh_coords_conversions_test_data["xyz"]
        self.llh = xyz2llh_coords_conversions_test_data["llh"]
        self.llh_deg = xyz2llh_coords_conversions_test_data["llh_deg"]
        self.xyz_vec = xyz2llh_coords_conversions_test_data["xyz_vec"]
        self.llh_vec = xyz2llh_coords_conversions_test_data["llh_vec"]
        self.llh_vec_deg = xyz2llh_coords_conversions_test_data["llh_vec_deg"]
        self.atol = xyz2llh_coords_conversions_test_data["tolerance"]["atol"]
        self.rtol = xyz2llh_coords_conversions_test_data["tolerance"]["rtol"]

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


class TestECEF_ECICoordsConversions:
    """Test coordinate conversion functions ecef2eci and eci2ecef with various input types and options"""

    def test_ecef2eci_single(self, ecef_eci_coords_conversions_test_data) -> None:
        """Testing ecef2eci, single coordinate"""
        positions_eci, velocities_eci = ecef2eci(
            positions=ecef_eci_coords_conversions_test_data["ecef_positions"][0],
            velocities=ecef_eci_coords_conversions_test_data["ecef_velocities"][0],
            times=ecef_eci_coords_conversions_test_data["ref_times"][0],
        )

        assert positions_eci.shape == ecef_eci_coords_conversions_test_data["ecef_positions"][0].shape
        assert velocities_eci.shape == ecef_eci_coords_conversions_test_data["ecef_velocities"][0].shape
        np.testing.assert_allclose(
            positions_eci,
            ecef_eci_coords_conversions_test_data["eci_positions"][0],
            atol=ecef_eci_coords_conversions_test_data["tolerance"]["atol"],
            rtol=ecef_eci_coords_conversions_test_data["tolerance"]["rtol"],
        )
        np.testing.assert_allclose(
            velocities_eci,
            ecef_eci_coords_conversions_test_data["eci_velocities"][0],
            atol=ecef_eci_coords_conversions_test_data["tolerance"]["atol"],
            rtol=ecef_eci_coords_conversions_test_data["tolerance"]["rtol"],
        )

    def test_ecef2eci_single_1_3(self, ecef_eci_coords_conversions_test_data) -> None:
        """Testing ecef2eci, single coordinate with (1, 3) shape"""
        positions_eci, velocities_eci = ecef2eci(
            positions=ecef_eci_coords_conversions_test_data["ecef_positions"][:1],
            velocities=ecef_eci_coords_conversions_test_data["ecef_velocities"][:1],
            times=ecef_eci_coords_conversions_test_data["ref_times"][:1],
        )

        assert positions_eci.shape == (1, 3)
        assert velocities_eci.shape == (1, 3)
        np.testing.assert_allclose(
            positions_eci,
            ecef_eci_coords_conversions_test_data["eci_positions"][:1],
            atol=ecef_eci_coords_conversions_test_data["tolerance"]["atol"],
            rtol=ecef_eci_coords_conversions_test_data["tolerance"]["rtol"],
        )
        np.testing.assert_allclose(
            velocities_eci,
            ecef_eci_coords_conversions_test_data["eci_velocities"][:1],
            atol=ecef_eci_coords_conversions_test_data["tolerance"]["atol"],
            rtol=ecef_eci_coords_conversions_test_data["tolerance"]["rtol"],
        )

    def test_ecef2eci_vectorized(self, ecef_eci_coords_conversions_test_data) -> None:
        """Testing ecef2eci, vectorized coordinates"""
        positions_eci, velocities_eci = ecef2eci(
            positions=ecef_eci_coords_conversions_test_data["ecef_positions"],
            velocities=ecef_eci_coords_conversions_test_data["ecef_velocities"],
            times=ecef_eci_coords_conversions_test_data["ref_times"],
        )

        assert positions_eci.shape == ecef_eci_coords_conversions_test_data["ecef_positions"].shape
        assert velocities_eci.shape == ecef_eci_coords_conversions_test_data["ecef_velocities"].shape
        np.testing.assert_allclose(
            positions_eci,
            ecef_eci_coords_conversions_test_data["eci_positions"],
            atol=ecef_eci_coords_conversions_test_data["tolerance"]["atol"],
            rtol=ecef_eci_coords_conversions_test_data["tolerance"]["rtol"],
        )
        np.testing.assert_allclose(
            velocities_eci,
            ecef_eci_coords_conversions_test_data["eci_velocities"],
            atol=ecef_eci_coords_conversions_test_data["tolerance"]["atol"],
            rtol=ecef_eci_coords_conversions_test_data["tolerance"]["rtol"],
        )

    def test_eci2ecef_single(self, ecef_eci_coords_conversions_test_data) -> None:
        """Testing eci2ecef, single coordinate"""
        positions_ecef, velocities_ecef = eci2ecef(
            positions=ecef_eci_coords_conversions_test_data["eci_positions"][0],
            velocities=ecef_eci_coords_conversions_test_data["eci_velocities"][0],
            times=ecef_eci_coords_conversions_test_data["ref_times"][0],
        )

        assert positions_ecef.shape == ecef_eci_coords_conversions_test_data["eci_positions"][0].shape
        assert velocities_ecef.shape == ecef_eci_coords_conversions_test_data["eci_velocities"][0].shape
        np.testing.assert_allclose(
            positions_ecef,
            ecef_eci_coords_conversions_test_data["ecef_positions"][0],
            atol=ecef_eci_coords_conversions_test_data["tolerance"]["atol"],
            rtol=ecef_eci_coords_conversions_test_data["tolerance"]["rtol"],
        )
        np.testing.assert_allclose(
            velocities_ecef,
            ecef_eci_coords_conversions_test_data["ecef_velocities"][0],
            atol=ecef_eci_coords_conversions_test_data["tolerance"]["atol"],
            rtol=ecef_eci_coords_conversions_test_data["tolerance"]["rtol"],
        )

    def test_eci2ecef_single_1_3(self, ecef_eci_coords_conversions_test_data) -> None:
        """Testing eci2ecef, single coordinate with (1, 3) shape"""
        positions_ecef, velocities_ecef = eci2ecef(
            positions=ecef_eci_coords_conversions_test_data["eci_positions"][:1],
            velocities=ecef_eci_coords_conversions_test_data["eci_velocities"][:1],
            times=ecef_eci_coords_conversions_test_data["ref_times"][:1],
        )

        assert positions_ecef.shape == (1, 3)
        assert velocities_ecef.shape == (1, 3)
        np.testing.assert_allclose(
            positions_ecef,
            ecef_eci_coords_conversions_test_data["ecef_positions"][:1],
            atol=ecef_eci_coords_conversions_test_data["tolerance"]["atol"],
            rtol=ecef_eci_coords_conversions_test_data["tolerance"]["rtol"],
        )
        np.testing.assert_allclose(
            velocities_ecef,
            ecef_eci_coords_conversions_test_data["ecef_velocities"][:1],
            atol=ecef_eci_coords_conversions_test_data["tolerance"]["atol"],
            rtol=ecef_eci_coords_conversions_test_data["tolerance"]["rtol"],
        )

    def test_eci2ecef_vectorized(self, ecef_eci_coords_conversions_test_data) -> None:
        """Testing eci2ecef, vectorized coordinates"""
        positions_ecef, velocities_ecef = eci2ecef(
            positions=ecef_eci_coords_conversions_test_data["eci_positions"],
            velocities=ecef_eci_coords_conversions_test_data["eci_velocities"],
            times=ecef_eci_coords_conversions_test_data["ref_times"],
        )

        assert positions_ecef.shape == ecef_eci_coords_conversions_test_data["eci_positions"].shape
        assert velocities_ecef.shape == ecef_eci_coords_conversions_test_data["eci_velocities"].shape
        np.testing.assert_allclose(
            positions_ecef,
            ecef_eci_coords_conversions_test_data["ecef_positions"],
            atol=ecef_eci_coords_conversions_test_data["tolerance"]["atol"],
            rtol=ecef_eci_coords_conversions_test_data["tolerance"]["rtol"],
        )
        np.testing.assert_allclose(
            velocities_ecef,
            ecef_eci_coords_conversions_test_data["ecef_velocities"],
            atol=ecef_eci_coords_conversions_test_data["tolerance"]["atol"],
            rtol=ecef_eci_coords_conversions_test_data["tolerance"]["rtol"],
        )

    def test_ecef2eci_and_back(self, ecef_eci_coords_conversions_test_data) -> None:
        """Testing ecef2eci and back to ecef with eci2ecef, vectorized coordinates"""
        positions_eci, velocities_eci = ecef2eci(
            positions=ecef_eci_coords_conversions_test_data["ecef_positions"],
            velocities=ecef_eci_coords_conversions_test_data["ecef_velocities"],
            times=ecef_eci_coords_conversions_test_data["ref_times"],
        )
        positions_ecef, velocities_ecef = eci2ecef(
            positions=positions_eci,
            velocities=velocities_eci,
            times=ecef_eci_coords_conversions_test_data["ref_times"],
        )

        assert positions_ecef.shape == ecef_eci_coords_conversions_test_data["ecef_positions"].shape
        assert velocities_ecef.shape == ecef_eci_coords_conversions_test_data["ecef_velocities"].shape
        np.testing.assert_allclose(
            positions_ecef,
            ecef_eci_coords_conversions_test_data["ecef_positions"],
            atol=ecef_eci_coords_conversions_test_data["tolerance"]["atol"],
        )
        np.testing.assert_allclose(
            velocities_ecef,
            ecef_eci_coords_conversions_test_data["ecef_velocities"],
            rtol=ecef_eci_coords_conversions_test_data["tolerance"]["rtol"],
        )

    def test_eci2ecef_and_back(self, ecef_eci_coords_conversions_test_data) -> None:
        """Testing eci2ecef and back to ecef with ecef2eci, vectorized coordinates"""
        positions_ecef, velocities_ecef = eci2ecef(
            positions=ecef_eci_coords_conversions_test_data["eci_positions"],
            velocities=ecef_eci_coords_conversions_test_data["eci_velocities"],
            times=ecef_eci_coords_conversions_test_data["ref_times"],
        )
        positions_eci, velocities_eci = ecef2eci(
            positions=positions_ecef,
            velocities=velocities_ecef,
            times=ecef_eci_coords_conversions_test_data["ref_times"],
        )

        assert positions_eci.shape == ecef_eci_coords_conversions_test_data["eci_positions"].shape
        assert velocities_eci.shape == ecef_eci_coords_conversions_test_data["eci_velocities"].shape
        np.testing.assert_allclose(
            positions_eci,
            ecef_eci_coords_conversions_test_data["eci_positions"],
            atol=ecef_eci_coords_conversions_test_data["tolerance"]["atol"],
        )
        np.testing.assert_allclose(
            velocities_eci,
            ecef_eci_coords_conversions_test_data["eci_velocities"],
            rtol=ecef_eci_coords_conversions_test_data["tolerance"]["rtol"],
        )
