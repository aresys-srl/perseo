# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Testing geometry/coordinates/conversions.py functionalities"""

import numpy as np
import pytest

from perseo_core.geometry.coordinates.conversions import ecef2eci, eci2ecef, llh2utm, llh2xyz, utm2llh, xyz2llh


class TestXYZLLHCoordsConversions:
    """Test coordinate conversion functions xyz2llh and llh2xyz with various input types and options."""

    @pytest.fixture(autouse=True)
    def setup_coords_conversions_data(self, xyz2llh_coords_conversions_test_data: dict) -> None:
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

    def test_xyz2llh_from_array_1_3(self) -> None:
        """Test xyz2llh with numpy array (1, 3) input in radians mode."""
        llh = xyz2llh(self.xyz_vec[:1])
        assert llh.shape == (1, 3)
        np.testing.assert_allclose(llh, self.llh_vec[:1], atol=self.atol, rtol=self.rtol)

    def test_xyz2llh_from_array_1_3_deg(self) -> None:
        """Test xyz2llh with numpy array (1, 3) input in degrees mode."""
        llh = xyz2llh(self.xyz_vec[:1], radians=False)
        llh[:, :2] = np.deg2rad(llh[:, :2])
        assert llh.shape == (1, 3)
        np.testing.assert_allclose(llh, self.llh_vec[:1], atol=self.atol, rtol=self.rtol)

    def test_llh2xyz_from_array_1_3(self) -> None:
        """Test llh2xyz with numpy array (1, 3) input in radians mode."""
        xyz = llh2xyz(self.llh_vec[:1])
        assert xyz.shape == (1, 3)
        np.testing.assert_allclose(xyz, self.xyz_vec[:1], atol=self.atol, rtol=self.rtol)

    def test_llh2xyz_from_array_1_3_deg(self) -> None:
        """Test llh2xyz with numpy array (1, 3) input in degrees mode."""
        xyz = llh2xyz(self.llh_vec_deg[:1], radians=False)
        assert xyz.shape == (1, 3)
        np.testing.assert_allclose(xyz, self.xyz_vec[:1], atol=self.atol, rtol=self.rtol)

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


class TestECEFECICoordsConversions:
    """Test coordinate conversion functions ecef2eci and eci2ecef with various input types and options"""

    def test_ecef2eci_single(self, ecef_eci_coords_conversions_test_data: dict) -> None:
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

    def test_ecef2eci_single_1_3(self, ecef_eci_coords_conversions_test_data: dict) -> None:
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

    def test_ecef2eci_vectorized(self, ecef_eci_coords_conversions_test_data: dict) -> None:
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

    def test_eci2ecef_single(self, ecef_eci_coords_conversions_test_data: dict) -> None:
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

    def test_eci2ecef_single_1_3(self, ecef_eci_coords_conversions_test_data: dict) -> None:
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

    def test_eci2ecef_vectorized(self, ecef_eci_coords_conversions_test_data: dict) -> None:
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

    def test_ecef2eci_and_back(self, ecef_eci_coords_conversions_test_data: dict) -> None:
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

    def test_eci2ecef_and_back(self, ecef_eci_coords_conversions_test_data: dict) -> None:
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


class TestUTMLLHCoordsConversions:
    """Test coordinate conversion functions utm2llh and llh2utm with various input types and options."""

    CASE_NAMES = ("case_1", "case_2", "case_3", "case_4")

    @pytest.fixture(autouse=True)
    def setup_tolerances(self, utm2llh_coords_conversions_test_data: dict) -> None:
        self.tol_llh = utm2llh_coords_conversions_test_data["tolerance_llh"]
        self.tol_utm = utm2llh_coords_conversions_test_data["tolerance_utm"]

    @pytest.mark.parametrize("case_name", CASE_NAMES)
    def test_utm2llh_scalar_deg(self, case_name: str, utm2llh_coords_conversions_test_data: dict) -> None:
        """Test utm2llh, scalar, output in degrees."""
        case = utm2llh_coords_conversions_test_data[case_name]
        llh_deg = utm2llh(coordinates=case["utm"][0, :], zone=case["zone"], radians=False)
        assert llh_deg.shape == (3,)
        np.testing.assert_allclose(llh_deg, case["llh_deg"][0, :], atol=self.tol_llh["atol"], rtol=self.tol_llh["rtol"])

    @pytest.mark.parametrize("case_name", CASE_NAMES)
    def test_utm2llh_scalar_rad(self, case_name: str, utm2llh_coords_conversions_test_data: dict) -> None:
        """Test utm2llh, scalar, output in radians."""
        case = utm2llh_coords_conversions_test_data[case_name]
        llh = utm2llh(coordinates=case["utm"][0, :], zone=case["zone"], radians=True)
        llh[:2] = np.rad2deg(llh[:2])
        assert llh.shape == (3,)
        np.testing.assert_allclose(llh, case["llh_deg"][0, :], atol=self.tol_llh["atol"], rtol=self.tol_llh["rtol"])

    @pytest.mark.parametrize("case_name", CASE_NAMES)
    def test_utm2llh_vect_deg(self, case_name: str, utm2llh_coords_conversions_test_data: dict) -> None:
        """Test utm2llh, vectorized, output in degrees."""
        case = utm2llh_coords_conversions_test_data[case_name]
        llh_deg = utm2llh(coordinates=case["utm"], zone=case["zone"], radians=False)
        assert llh_deg.shape == case["utm"].shape
        np.testing.assert_allclose(llh_deg, case["llh_deg"], atol=self.tol_llh["atol"], rtol=self.tol_llh["rtol"])

    @pytest.mark.parametrize("case_name", CASE_NAMES)
    def test_utm2llh_vect_rad(self, case_name: str, utm2llh_coords_conversions_test_data: dict) -> None:
        """Test utm2llh, vectorized, output in radians."""
        case = utm2llh_coords_conversions_test_data[case_name]
        llh = utm2llh(coordinates=case["utm"], zone=case["zone"], radians=True)
        llh[:, :2] = np.rad2deg(llh[:, :2])
        assert llh.shape == case["utm"].shape
        np.testing.assert_allclose(llh, case["llh_deg"], atol=self.tol_llh["atol"], rtol=self.tol_llh["rtol"])

    @pytest.mark.parametrize("case_name", CASE_NAMES)
    def test_llh2utm_scalar_deg(self, case_name: str, utm2llh_coords_conversions_test_data: dict) -> None:
        """Test llh2utm, scalar, output in degrees."""
        case = utm2llh_coords_conversions_test_data[case_name]
        utm = llh2utm(coordinates=case["llh_deg"][0, :], zone=case["zone"], radians=False)
        assert utm.shape == (3,)
        np.testing.assert_allclose(utm, case["utm"][0, :], atol=self.tol_utm["atol"], rtol=self.tol_utm["rtol"])

    @pytest.mark.parametrize("case_name", CASE_NAMES)
    def test_llh2utm_scalar_rad(self, case_name: str, utm2llh_coords_conversions_test_data: dict) -> None:
        """Test llh2utm, scalar, output in radians."""
        case = utm2llh_coords_conversions_test_data[case_name]
        coords = case["llh_deg"][0, :].copy()
        coords[:2] = np.deg2rad(coords[:2])
        utm = llh2utm(coordinates=coords, zone=case["zone"], radians=True)
        assert utm.shape == (3,)
        np.testing.assert_allclose(utm, case["utm"][0, :], atol=self.tol_utm["atol"], rtol=self.tol_utm["rtol"])

    @pytest.mark.parametrize("case_name", CASE_NAMES)
    def test_llh2utm_vect_deg(self, case_name: str, utm2llh_coords_conversions_test_data: dict) -> None:
        """Test llh2utm, vectorized, output in degrees."""
        case = utm2llh_coords_conversions_test_data[case_name]
        utm = llh2utm(coordinates=case["llh_deg"], zone=case["zone"], radians=False)
        assert utm.shape == case["llh_deg"].shape
        np.testing.assert_allclose(utm, case["utm"], atol=self.tol_utm["atol"], rtol=self.tol_utm["rtol"])

    @pytest.mark.parametrize("case_name", CASE_NAMES)
    def test_llh2utm_vect_rad(self, case_name: str, utm2llh_coords_conversions_test_data: dict) -> None:
        """Test llh2utm, vectorized, output in radians."""
        case = utm2llh_coords_conversions_test_data[case_name]
        coords = case["llh_deg"].copy()
        coords[:, :2] = np.deg2rad(coords[:, :2])
        utm = llh2utm(coordinates=coords, zone=case["zone"], radians=True)
        assert utm.shape == case["llh_deg"].shape
        np.testing.assert_allclose(utm, case["utm"], atol=self.tol_utm["atol"], rtol=self.tol_utm["rtol"])

    @pytest.mark.parametrize("case_name", CASE_NAMES)
    def test_utm2llh_vect_1_3_deg(self, case_name: str, utm2llh_coords_conversions_test_data: dict) -> None:
        """Test utm2llh, (1, 3) vectorized, output in degrees."""
        case = utm2llh_coords_conversions_test_data[case_name]
        llh_deg = utm2llh(coordinates=case["utm"][:1], zone=case["zone"], radians=False)
        assert llh_deg.shape == (1, 3)
        np.testing.assert_allclose(llh_deg, case["llh_deg"][:1], atol=self.tol_llh["atol"], rtol=self.tol_llh["rtol"])

    @pytest.mark.parametrize("case_name", CASE_NAMES)
    def test_utm2llh_vect_1_3_rad(self, case_name: str, utm2llh_coords_conversions_test_data: dict) -> None:
        """Test utm2llh, (1, 3) vectorized, output in radians."""
        case = utm2llh_coords_conversions_test_data[case_name]
        llh = utm2llh(coordinates=case["utm"][:1], zone=case["zone"], radians=True)
        llh[:, :2] = np.rad2deg(llh[:, :2])
        assert llh.shape == (1, 3)
        np.testing.assert_allclose(llh, case["llh_deg"][:1], atol=self.tol_llh["atol"], rtol=self.tol_llh["rtol"])

    @pytest.mark.parametrize("case_name", CASE_NAMES)
    def test_llh2utm_vect_1_3_deg(self, case_name: str, utm2llh_coords_conversions_test_data: dict) -> None:
        """Test llh2utm, (1, 3) vectorized, output in degrees."""
        case = utm2llh_coords_conversions_test_data[case_name]
        utm = llh2utm(coordinates=case["llh_deg"][:1], zone=case["zone"], radians=False)
        assert utm.shape == (1, 3)
        np.testing.assert_allclose(utm, case["utm"][:1], atol=self.tol_utm["atol"], rtol=self.tol_utm["rtol"])

    @pytest.mark.parametrize("case_name", CASE_NAMES)
    def test_llh2utm_vect_1_3_rad(self, case_name: str, utm2llh_coords_conversions_test_data: dict) -> None:
        """Test llh2utm, (1, 3) vectorized, output in radians."""
        case = utm2llh_coords_conversions_test_data[case_name]
        coords = case["llh_deg"][:1].copy()
        coords[:, :2] = np.deg2rad(coords[:, :2])
        utm = llh2utm(coordinates=coords, zone=case["zone"], radians=True)
        assert utm.shape == (1, 3)
        np.testing.assert_allclose(utm, case["utm"][:1], atol=self.tol_utm["atol"], rtol=self.tol_utm["rtol"])

    @pytest.mark.parametrize("case_name", CASE_NAMES)
    def test_round_trip_vectorized_deg(self, case_name: str, utm2llh_coords_conversions_test_data: dict) -> None:
        """Test utm2llh and llh2utm round trip for vectorized input, degrees."""
        case = utm2llh_coords_conversions_test_data[case_name]
        llh_deg = utm2llh(coordinates=case["utm"], zone=case["zone"], radians=False)
        utm = llh2utm(coordinates=llh_deg, zone=case["zone"], radians=False)
        np.testing.assert_allclose(utm, case["utm"], atol=self.tol_utm["atol"], rtol=self.tol_utm["rtol"])

    @pytest.mark.parametrize("case_name", CASE_NAMES)
    def test_round_trip_vectorized_rad(self, case_name: str, utm2llh_coords_conversions_test_data: dict) -> None:
        """Test utm2llh and llh2utm round trip for vectorized input, radians."""
        case = utm2llh_coords_conversions_test_data[case_name]
        coords = case["llh_deg"].copy()
        coords[:, :2] = np.deg2rad(coords[:, :2])
        utm = llh2utm(coordinates=coords, zone=case["zone"], radians=True)
        llh_deg = utm2llh(coordinates=utm, zone=case["zone"], radians=False)
        np.testing.assert_allclose(llh_deg, case["llh_deg"], atol=self.tol_llh["atol"], rtol=self.tol_llh["rtol"])

    def test_invalid_zone_format(self, utm2llh_coords_conversions_test_data: dict) -> None:
        """Test that invalid zone format raises ValueError."""
        case1 = utm2llh_coords_conversions_test_data["case_1"]
        with pytest.raises(TypeError):
            utm2llh(case1["utm"], zone=33)
        with pytest.raises(TypeError):
            llh2utm(case1["llh_deg"], zone=33, radians=False)

    def test_invalid_zone_format_2(self, utm2llh_coords_conversions_test_data: dict) -> None:
        """Test that invalid zone format raises ValueError."""
        case1 = utm2llh_coords_conversions_test_data["case_1"]
        with pytest.raises(ValueError, match="Zone must be string format"):
            utm2llh(case1["utm"], zone="33 N")
        with pytest.raises(ValueError, match="Zone must be string format"):
            llh2utm(case1["llh_deg"], zone="33 N", radians=False)

    def test_invalid_zone_number(self, utm2llh_coords_conversions_test_data: dict) -> None:
        """Test that invalid zone number raises ValueError."""
        case1 = utm2llh_coords_conversions_test_data["case_1"]
        with pytest.raises(ValueError, match="zone number must be between 1 and 60"):
            utm2llh(case1["utm"], zone="61N")
        with pytest.raises(ValueError, match="zone number must be between 1 and 60"):
            llh2utm(case1["llh_deg"], zone="61N", radians=False)

    def test_invalid_hemisphere(self, utm2llh_coords_conversions_test_data: dict) -> None:
        """Test that invalid hemisphere raises ValueError."""
        case1 = utm2llh_coords_conversions_test_data["case_1"]
        with pytest.raises(ValueError, match="Hemisphere must be 'N' or 'S'"):
            utm2llh(case1["utm"], zone="33X")
        with pytest.raises(ValueError, match="Hemisphere must be 'N' or 'S'"):
            llh2utm(case1["llh_deg"], zone="33X", radians=False)
