# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Testing geometry/doppler.py functionalities"""

import numpy as np
import pytest

from perseo_core.geometry.doppler import (
    compute_steering_doppler_frequency,
    compute_theoretical_doppler_rate,
    doppler_equation,
    doppler_equation_bistatic_residuals,
    doppler_equation_monostatic_residuals,
    get_geometric_doppler_centroid,
)


class TestDopplerEquation:
    """Test doppler_equation with scalar and vectorized inputs."""

    def test_doppler_equation_scalar(self, doppler_test_data: dict) -> None:
        """Test doppler_equation with 1D inputs (single point)."""

        doppler, gradient = doppler_equation(
            wavelength=doppler_test_data["wavelength"],
            pv_scalar=doppler_test_data["pv_scalar"],
            distance=doppler_test_data["distance"],
            doppler_frequency=doppler_test_data["frequency_doppler_centroid"],
            sensor_velocity=doppler_test_data["sensor_velocity"],
            los=doppler_test_data["los"],
        )

        assert isinstance(doppler, float)
        assert isinstance(gradient, np.ndarray)
        assert gradient.shape == (3,)
        np.testing.assert_allclose(doppler, doppler_test_data["doppler_result"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(
            gradient,
            doppler_test_data["gradient_result"],
            atol=doppler_test_data["tolerance"]["atol"],
            rtol=doppler_test_data["tolerance"]["rtol"],
        )

    def test_doppler_equation_vectorized(self, doppler_test_data: dict) -> None:
        """Test doppler_equation with 2D inputs (multiple points)."""
        doppler, gradient = doppler_equation(
            wavelength=doppler_test_data["wavelength"],
            pv_scalar=np.tile(doppler_test_data["pv_scalar"], (doppler_test_data["N"])),
            distance=np.tile(doppler_test_data["distance"], (doppler_test_data["N"])),
            doppler_frequency=np.tile(doppler_test_data["frequency_doppler_centroid"], (doppler_test_data["N"])),
            sensor_velocity=np.tile(doppler_test_data["sensor_velocity"], (doppler_test_data["N"], 1)),
            los=np.tile(doppler_test_data["los"], (doppler_test_data["N"], 1)),
        )

        assert isinstance(doppler, np.ndarray)
        assert isinstance(gradient, np.ndarray)
        assert doppler.shape == (doppler_test_data["N"],)
        assert gradient.shape == (doppler_test_data["N"], 3)
        np.testing.assert_allclose(
            doppler, np.tile(doppler_test_data["doppler_result"], doppler_test_data["N"]), atol=1e-12, rtol=0
        )
        np.testing.assert_allclose(
            gradient,
            np.tile(doppler_test_data["gradient_result"], (doppler_test_data["N"], 1)),
            atol=doppler_test_data["tolerance"]["atol"],
            rtol=doppler_test_data["tolerance"]["rtol"],
        )


class TestDopplerEquationMonostaticResiduals:
    """Test doppler_equation_monostatic_residuals with various input shapes."""

    def test_monostatic_residuals_single_position(self, doppler_test_data: dict) -> None:
        """Test with a single sensor position."""
        positions = doppler_test_data["trajectory"].position(doppler_test_data["azimuth_time"])
        velocities = doppler_test_data["trajectory"].velocity(doppler_test_data["azimuth_time"])

        residual = doppler_equation_monostatic_residuals(
            ground_point=doppler_test_data["ground_point"],
            sensor_positions=positions,
            sensor_velocities=velocities,
            doppler_frequency=doppler_test_data["frequency_doppler_centroid"],
            wavelength=doppler_test_data["wavelength"],
        )

        assert isinstance(residual, float)
        np.testing.assert_allclose(
            residual,
            doppler_test_data["residual_monostatic_result"],
            atol=doppler_test_data["tolerance"]["atol"],
            rtol=doppler_test_data["tolerance"]["rtol"],
        )

    def test_monostatic_residuals_single_position_vect(self, doppler_test_data: dict) -> None:
        """Test with multiple sensor positions (vectorized)."""
        positions = np.stack(
            [
                doppler_test_data["trajectory"].position(doppler_test_data["azimuth_time"]),
                doppler_test_data["trajectory"].position(doppler_test_data["azimuth_time"]),
            ]
        )
        velocities = np.stack(
            [
                doppler_test_data["trajectory"].velocity(doppler_test_data["azimuth_time"]),
                doppler_test_data["trajectory"].velocity(doppler_test_data["azimuth_time"]),
            ]
        )
        residual = doppler_equation_monostatic_residuals(
            ground_point=doppler_test_data["ground_point"],
            sensor_positions=positions,
            sensor_velocities=velocities,
            doppler_frequency=doppler_test_data["frequency_doppler_centroid"],
            wavelength=doppler_test_data["wavelength"],
        )

        assert isinstance(residual, np.ndarray)
        assert residual.shape == (2,)
        np.testing.assert_allclose(
            residual,
            np.repeat(doppler_test_data["residual_monostatic_result"], 2),
            atol=doppler_test_data["tolerance"]["atol"],
            rtol=doppler_test_data["tolerance"]["rtol"],
        )

    def test_monostatic_residuals_invalid_sensor_positions_shape(self) -> None:
        """Test that invalid sensor_positions shapes raise ValueError."""
        with pytest.raises(ValueError, match=r"sensor_positions has invalid shape"):
            doppler_equation_monostatic_residuals(
                np.arange(3),
                np.arange(5),
                np.arange(5),
                0,
                1,
            )

        with pytest.raises(ValueError, match=r"sensor_positions has invalid shape"):
            doppler_equation_monostatic_residuals(
                np.arange(3),
                np.arange(12).reshape(2, 2, 3),
                np.arange(6).reshape(2, 3),
                0,
                1,
            )

        with pytest.raises(ValueError, match=r"cannot reshape"):
            doppler_equation_monostatic_residuals(
                np.arange(3),
                np.arange(6).reshape(2, 2),
                np.arange(6).reshape(2, 3),
                0,
                1,
            )

    def test_monostatic_residuals_invalid_sensor_velocities_shape(self) -> None:
        """Test that invalid sensor_velocities shapes raise ValueError."""
        with pytest.raises(ValueError, match=r"sensor_velocities has invalid shape"):
            doppler_equation_monostatic_residuals(
                np.arange(3),
                np.arange(6).reshape(2, 3),
                np.arange(5),
                0,
                1,
            )

        with pytest.raises(ValueError, match=r"sensor_velocities has invalid shape"):
            doppler_equation_monostatic_residuals(
                np.arange(3),
                np.arange(6).reshape(2, 3),
                np.arange(12).reshape(2, 2, 3),
                0,
                1,
            )

        with pytest.raises(ValueError, match=r"cannot reshape"):
            doppler_equation_monostatic_residuals(
                np.arange(3),
                np.arange(6).reshape(2, 3),
                np.arange(6).reshape(2, 2),
                0,
                1,
            )


class TestDopplerEquationBistaticResiduals:
    """Test doppler_equation_bistatic_residuals."""

    def test_bistatic_residuals_monostatic_degenerate(self, doppler_test_data: dict) -> None:
        """Test with identical rx and tx (degenerate to monostatic), single position."""
        positions = doppler_test_data["trajectory"].position(doppler_test_data["azimuth_time"])
        velocities = doppler_test_data["trajectory"].velocity(doppler_test_data["azimuth_time"])
        point = doppler_test_data["ground_point"]
        residual_bistatic = doppler_equation_bistatic_residuals(
            sensor_pos_rx=positions,
            sensor_pos_tx=positions,
            sensor_vel_rx=velocities,
            sensor_vel_tx=velocities,
            ground_points=point,
            wavelength=doppler_test_data["wavelength"],
            doppler_frequency=doppler_test_data["frequency_doppler_centroid"],
        )

        assert isinstance(residual_bistatic, float)
        np.testing.assert_allclose(
            residual_bistatic,
            -2 * doppler_test_data["residual_monostatic_result"],
            atol=doppler_test_data["tolerance"]["atol"],
            rtol=doppler_test_data["tolerance"]["rtol"],
        )

    def test_bistatic_residuals_monostatic_degenerate_vect(self, doppler_test_data: dict) -> None:
        """Test with identical rx and tx (degenerate to monostatic), multiple position (vectorized)."""
        positions = np.stack(
            [
                doppler_test_data["trajectory"].position(doppler_test_data["azimuth_time"]),
                doppler_test_data["trajectory"].position(doppler_test_data["azimuth_time"]),
            ]
        )
        velocities = np.stack(
            [
                doppler_test_data["trajectory"].velocity(doppler_test_data["azimuth_time"]),
                doppler_test_data["trajectory"].velocity(doppler_test_data["azimuth_time"]),
            ]
        )
        point = np.stack([doppler_test_data["ground_point"], doppler_test_data["ground_point"]])
        residual_bistatic = doppler_equation_bistatic_residuals(
            sensor_pos_rx=positions,
            sensor_pos_tx=positions,
            sensor_vel_rx=velocities,
            sensor_vel_tx=velocities,
            ground_points=point,
            wavelength=doppler_test_data["wavelength"],
            doppler_frequency=doppler_test_data["frequency_doppler_centroid"],
        )

        assert isinstance(residual_bistatic, np.ndarray)
        np.testing.assert_allclose(
            residual_bistatic,
            -2 * np.repeat(doppler_test_data["residual_monostatic_result"], 2),
            atol=doppler_test_data["tolerance"]["atol"],
            rtol=doppler_test_data["tolerance"]["rtol"],
        )


class TestGeometricDopplerCentroid:
    """Test get_geometric_doppler_centroid."""

    def test_geometric_doppler_centroid_single(self, doppler_test_data: dict) -> None:
        """Test with a single position-velocity-point set."""
        centroid = get_geometric_doppler_centroid(
            sensor_positions=doppler_test_data["trajectory"].position(doppler_test_data["azimuth_time"]),
            sensor_velocities=doppler_test_data["trajectory"].velocity(doppler_test_data["azimuth_time"]),
            ground_points=doppler_test_data["ground_point"],
            wavelength=doppler_test_data["wavelength"],
        )

        assert isinstance(centroid, float)
        np.testing.assert_allclose(
            centroid,
            doppler_test_data["doppler_centroid_result"],
            atol=doppler_test_data["tolerance"]["atol"],
            rtol=doppler_test_data["tolerance"]["rtol"],
        )

    def test_geometric_doppler_centroid_vect(self, doppler_test_data: dict) -> None:
        """Test with multiple position-velocity-point sets (vectorized)."""
        positions = np.tile(
            doppler_test_data["trajectory"].position(doppler_test_data["azimuth_time"]), (doppler_test_data["N"], 1)
        )
        velocities = np.tile(
            doppler_test_data["trajectory"].velocity(doppler_test_data["azimuth_time"]), (doppler_test_data["N"], 1)
        )
        points = np.tile(doppler_test_data["ground_point"], (doppler_test_data["N"], 1))

        centroid = get_geometric_doppler_centroid(
            sensor_positions=positions,
            sensor_velocities=velocities,
            ground_points=points,
            wavelength=doppler_test_data["wavelength"],
        )

        assert isinstance(centroid, np.ndarray)
        assert centroid.shape == (doppler_test_data["N"],)
        np.testing.assert_allclose(
            centroid,
            np.tile(doppler_test_data["doppler_centroid_result"], doppler_test_data["N"]),
            atol=doppler_test_data["tolerance"]["atol"],
            rtol=doppler_test_data["tolerance"]["rtol"],
        )


class TestTheoreticalDopplerRate:
    """Test compute_theoretical_doppler_rate."""

    def test_compute_theoretical_doppler_rate_scalar(self, doppler_test_data: dict) -> None:
        """Test compute_theoretical_doppler_rate with single ground point."""
        doppler_rate = compute_theoretical_doppler_rate(
            trajectory=doppler_test_data["trajectory"],
            azimuth_time=doppler_test_data["azimuth_time"],
            ground_points=doppler_test_data["ground_point"],
            carrier_frequency=doppler_test_data["carrier_frequency"],
        )

        assert isinstance(doppler_rate, float)
        np.testing.assert_allclose(
            doppler_rate,
            doppler_test_data["doppler_rate_result"],
            atol=doppler_test_data["tolerance"]["atol"],
            rtol=doppler_test_data["tolerance"]["rtol"],
        )

    def test_compute_theoretical_doppler_rate_vect(self, doppler_test_data: dict) -> None:
        """Test compute_theoretical_doppler_rate with multiple ground points."""
        doppler_rate = compute_theoretical_doppler_rate(
            trajectory=doppler_test_data["trajectory"],
            azimuth_time=doppler_test_data["azimuth_time"],
            ground_points=np.tile(doppler_test_data["ground_point"], (doppler_test_data["N"], 1)),
            carrier_frequency=doppler_test_data["carrier_frequency"],
        )

        np.testing.assert_allclose(
            doppler_rate,
            np.full(doppler_test_data["N"], doppler_test_data["doppler_rate_result"]),
            atol=doppler_test_data["tolerance"]["atol"],
            rtol=doppler_test_data["tolerance"]["rtol"],
        )


class TestSteeringDopplerFrequency:
    """Test compute_steering_doppler_frequency."""

    def test_compute_steering_doppler_frequency_before_mid_burst(self, doppler_test_data: dict) -> None:
        """Test compute_steering_doppler_frequency, before mid burst."""
        steering_freq = compute_steering_doppler_frequency(
            trajectory=doppler_test_data["trajectory"],
            azimuth_time=doppler_test_data["azimuth_time"] - 1,
            az_mid_burst_time=doppler_test_data["azimuth_time"],
            doppler_rate=float(doppler_test_data["doppler_rate"]),
            az_steering_rate=doppler_test_data["az_steering_rate_rad_s"],
            carrier_frequency=doppler_test_data["carrier_frequency"],
        )

        np.testing.assert_allclose(
            steering_freq,
            doppler_test_data["steering_doppler_frequency_result"][0],
            atol=doppler_test_data["tolerance"]["atol"],
            rtol=doppler_test_data["tolerance"]["rtol"],
        )

    def test_compute_steering_doppler_frequency_after_mid_burst(self, doppler_test_data: dict) -> None:
        """Test compute_steering_doppler_frequency, after mid burst."""
        steering_freq = compute_steering_doppler_frequency(
            trajectory=doppler_test_data["trajectory"],
            azimuth_time=doppler_test_data["azimuth_time"] + 1,
            az_mid_burst_time=doppler_test_data["azimuth_time"],
            doppler_rate=float(doppler_test_data["doppler_rate"]),
            az_steering_rate=doppler_test_data["az_steering_rate_rad_s"],
            carrier_frequency=doppler_test_data["carrier_frequency"],
        )

        np.testing.assert_allclose(
            steering_freq,
            doppler_test_data["steering_doppler_frequency_result"][1],
            atol=doppler_test_data["tolerance"]["atol"],
            rtol=doppler_test_data["tolerance"]["rtol"],
        )
