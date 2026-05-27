# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Tests for target_ambiguity_ratio_analysis core functionalities"""

import numpy as np
import numpy.typing as npt
import pytest
from arepytools.timing.precisedatetime import PreciseDateTime

from perseo_quality.core.common import detect_burst_from_pixel
from perseo_quality.tar_analysis import support

TIMESTAMP = PreciseDateTime.from_utc_string("15-JAN-2019 17:14:00.177514769177")


class ChannelDataMock:
    """Mocking ChannelData class"""

    def __init__(self) -> None:
        self.trajectory = TrajectoryMock()

    def times_to_pixel_conversion(
        self, azimuth_time: PreciseDateTime, range_time: float, burst: int = None
    ) -> tuple[float, float]:
        """Mocking times to pixel method"""
        if azimuth_time > TIMESTAMP:
            return (1219.181615461248, 11859.28856588589)
        return (508.55123828424587, 11859.28856588589)


class TrajectoryMock:
    """Mocking trajectory class"""

    def evaluate(self, time: PreciseDateTime) -> npt.NDArray[np.floating]:
        """Mocking evaluate method"""
        if time == TIMESTAMP:
            return np.array([5097170.677138778, 382444.54075094324, 4883009.133504967])
        return np.array([5093441.435722766, 380760.62909382704, 4887041.911669481])


class TestTargetAmbiguityRatioFunctions:
    """Testing Target Ambiguity Ratio Analysis support functionalities"""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        """Testing setup"""
        self.lines_per_burst = np.array([150, 150, 150, 150, 150])
        self.tolerance = 1e-6

    def test_are_ambiguities_inside_scene_true(self) -> None:
        """Testing are_ambiguities_inside_scene check function, true case"""
        assert support.are_ambiguities_inside_scene(r_amb=(1219, 11860), l_amb=(508, 11860), lines=5180, samples=23000)

    def test_are_ambiguities_inside_scene_false(self) -> None:
        """Testing are_ambiguities_inside_scene check function, false case"""
        assert not support.are_ambiguities_inside_scene(
            r_amb=(12190, 11860), l_amb=(508, 11860), lines=5180, samples=23000
        )

    def test_detect_burst_from_pixel_case0(self) -> None:
        """Testing detect_burst_from_pixel function, case 0"""
        burst = detect_burst_from_pixel(lines_per_burst=self.lines_per_burst, azimuth_px=301)
        assert burst == 2

    def test_detect_burst_from_pixel_case1(self) -> None:
        """Testing detect_burst_from_pixel function, case 0"""
        burst = detect_burst_from_pixel(lines_per_burst=self.lines_per_burst[0], azimuth_px=146)
        assert burst == 0

    def test_compute_ambiguities_location(self) -> None:
        """Testing compute_ambiguities_location function"""
        channel_data = ChannelDataMock()
        l_amb_location_px, r_amb_location_px, az_delta, rng_delta = support.compute_ambiguities_locations(
            channel_data=channel_data,
            point_target_xyz_coords=np.array([4485484.70898501, 725439.6916493, 4461017.28690889]),
            burst=1,
            point_target_azimuth_time=TIMESTAMP,
            point_target_range_time=0.005460200797481148,
            doppler_rate=-2268.548,
            prf=1717,
        )
        np.testing.assert_allclose(
            np.array(l_amb_location_px), np.array((508.55123828424587, 11859.28856588589)), atol=self.tolerance, rtol=0
        )
        np.testing.assert_allclose(
            np.array(r_amb_location_px), np.array((1219.181615461248, 11859.28856588589)), atol=self.tolerance, rtol=0
        )
        np.testing.assert_allclose(az_delta, 0.7568717964089806, atol=self.tolerance, rtol=0)
        np.testing.assert_allclose(rng_delta * 1000, 0.00012023741916089479, atol=self.tolerance, rtol=0)
