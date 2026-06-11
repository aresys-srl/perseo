# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Shared fixtures and utilities for unit tests."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
import xarray as xr
from arepytools.timing.precisedatetime import PreciseDateTime
from scipy.fft import fft2, ifft2

from perseo_quality.core.generic_dataclasses import SARPolarization, SARSideLooking
from perseo_quality.core.signal_processing import locate_max_2d_interp
from perseo_quality.io.point_targets import PointTarget
from perseo_quality.point_targets_analysis.custom_dataclasses import IRFDataOutput

_REF_DATA_IRF_RESULTS = IRFDataOutput(
    range_resolution=0.9831505486028405,
    azimuth_resolution=0.9831505486028405,
    pslr_2d=-13.272021869964892,
    islr_2d=-7.208789038956241,
    sslr_2d=-23.005687315280902,
    azimuth_pslr=-13.272021869964906,
    azimuth_islr=-10.219088995596044,
    azimuth_sslr=-23.005687315280902,
    range_pslr=-13.272021869964892,
    range_islr=-10.219088995596062,
    range_sslr=-23.005687315280916,
    azimuth_localization_error=-1.1392089049877541e-07,
    ground_range_localization_error=-1.1392089049877541e-07,
    slant_range_localization_error=-1.1392089049877541e-07,
)

_REF_DATA_RCS_RESULTS: dict = {"clutter": -91.7106822164182, "rcs": 77.228675375588, "scr": 91.712713592383}

_DEFAULT_INPUT_DATA_GENERATION: dict = {
    "lines": 128,
    "samples": 128,
    "lines_step": 0.00016420361247947455,
    "samples_step": 3.57142857142857e-09,
    "samples_start": 0.00400438831877932,
    "fc_hz": 9.6e9,
}

_REF_TIME = PreciseDateTime.from_utc_string("15-JAN-2019 16:37:12.051461300098")
_REF_GROUND_POINT = np.array([-4989394.044, 2746844.389, -2862070.09])
_REF_POINTS = [
    PointTarget(
        name="0",
        xyz_coordinates=np.array([4921229.04081908, -4051559.15884936, 216078.76707954]),
        rcs_hh=(100000 + 0j),
        rcs_vv=0j,
        rcs_vh=(100000 + 0j),
        rcs_hv=0j,
        delay=None,
    ),
    PointTarget(
        name="1",
        xyz_coordinates=np.array([4832296.19624738, -4155847.75546086, 241004.24360898]),
        rcs_hh=(100000 + 0j),
        rcs_vv=0j,
        rcs_vh=(100000 + 0j),
        rcs_hv=0j,
        delay=None,
    ),
    PointTarget(
        name="2",
        xyz_coordinates=np.array([4891219.45186627, -4087200.87719583, 225939.83847657]),
        rcs_hh=(100000 + 0j),
        rcs_vv=0j,
        rcs_vh=(100000 + 0j),
        rcs_hv=0j,
        delay=None,
    ),
]


class MockTrajectory:
    """Mocking trajectory class"""

    def evaluate(self, time) -> npt.NDArray[np.floating]:
        out = [-381087.525550857, 932485.770149446, -7007146.93083064]
        if np.size(time) == 1:
            return np.array(out)
        return np.stack([out] * np.size(time))

    def evaluate_first_derivatives(self, time) -> npt.NDArray[np.floating]:
        out = [7057.60934660782, 2768.35602191122, -0.259400938909807]
        if np.size(time) == 1:
            return np.array(out)
        return np.stack([out] * np.size(time))


class MockChannelData:
    """Mocking ChannelData class"""

    def __init__(self, channel_id: int = 1) -> None:
        self.channel_id = channel_id
        self._trajectory = MockTrajectory()

    @property
    def swath_name(self) -> str:
        return f"S{self.channel_id}"

    @property
    def polarization(self) -> str:
        return SARPolarization.HH

    @property
    def trajectory(self) -> MockTrajectory:
        return self._trajectory

    @property
    def looking_side(self) -> SARSideLooking:
        return SARSideLooking.RIGHT_LOOKING

    @property
    def carrier_frequency(self) -> float:
        return 5405000000

    def ground_points_to_burst_association(self, coordinates: npt.NDArray[np.floating]) -> list:
        if self.channel_id == 1:
            return [[0], None, [1]]
        return [None, [0], None]


class MockProduct:
    """Mocking Product class"""

    @property
    def channels_list(self) -> list[int]:
        return [1, 2]

    def get_channel_data(self, channel_id: int) -> MockChannelData:
        return MockChannelData(channel_id=channel_id)


def generate_target_data(
    bandwidths: tuple[float, float],
    axes: tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]],
    target_relative_pos: tuple[float, float],
    swath_range_start: float,
    fc_hz: float,
    window: npt.NDArray[np.floating] | None = None,
) -> npt.NDArray[np.floating]:
    array = np.outer(
        np.sinc(bandwidths[0] * (axes[0] - target_relative_pos[0])),
        np.sinc(bandwidths[1] * (axes[1] - target_relative_pos[1])),
    )
    array = array * np.exp(2j * np.pi * fc_hz * (target_relative_pos[0] + swath_range_start))
    if window is not None:
        return ifft2(fft2(array) * window)
    return array


def generate_data_for_test(
    lines: int,
    samples: int,
    lines_step: float,
    samples_step: float,
    samples_start: float,
    fc_hz: float,
    perc: float = 0.9,
    window: npt.NDArray[np.floating] | None = None,
) -> tuple[npt.NDArray[np.floating], tuple[float, float], tuple[float, float]]:
    azimuth_bandwidth = perc / lines_step
    range_bandwidth = perc / samples_step
    az_axis = np.arange(0, lines) * lines_step
    rng_axis = np.arange(0, samples) * samples_step
    target_az_rel = (np.ceil(lines / 2) + 0.5) * lines_step
    target_rng_rel = (np.ceil(samples / 2) + 0.5) * samples_step
    data = generate_target_data(
        bandwidths=(range_bandwidth, azimuth_bandwidth),
        axes=(rng_axis, az_axis),
        target_relative_pos=(target_rng_rel, target_az_rel),
        swath_range_start=samples_start,
        fc_hz=fc_hz,
        window=window,
    )
    _, *peak_pos = locate_max_2d_interp(data)
    target_pos = np.array([data.shape[0] / 2 + 0.5, data.shape[1] / 2 + 0.5])
    return data, peak_pos, target_pos


def generate_antenna_pattern() -> xr.Dataset:
    ds = xr.Dataset(
        {
            "gain": (
                ["azimuth_angles", "elevation_angles"],
                np.array(
                    [
                        [
                            129.55548635,
                            131.98112506,
                            134.00269966,
                            135.66136098,
                            136.98268344,
                            137.97979327,
                            138.6547115,
                            138.99814962,
                            138.98226901,
                            138.55844256,
                            137.63525623,
                            136.03710036,
                            133.38852027,
                            128.74301175,
                            122.004593,
                            128.77782281,
                            134.29660263,
                            137.68592647,
                            139.97862457,
                            141.58056386,
                            142.68263238,
                            143.38540311,
                            143.74504021,
                            143.79198111,
                            143.53937783,
                            142.98666975,
                            142.12101083,
                            140.91753215,
                            139.33689206,
                            137.3220495,
                            134.79195725,
                            131.63019337,
                            127.66477974,
                            122.62788755,
                            116.0590059,
                            106.84216924,
                            85.35376321,
                            102.61457528,
                            111.60922528,
                            119.38783371,
                            126.02720055,
                            131.56590119,
                            136.19595571,
                            140.09860106,
                            143.41386699,
                            146.23757924,
                            148.63745705,
                            150.66070905,
                            152.33813984,
                            153.68709803,
                            154.71504397,
                            155.41198259,
                            155.74919307,
                            155.6721423,
                            155.07652174,
                            153.75132897,
                            151.19797042,
                            145.71083774,
                            128.25449709,
                            149.10000134,
                            155.36731254,
                            159.29839639,
                            162.17073892,
                            164.40609626,
                            166.1948974,
                            167.63884446,
                            168.79697968,
                            169.70492193,
                            170.3839497,
                            170.84554837,
                            171.09300563,
                            171.12273753,
                            170.92340758,
                            170.47350839,
                            169.7366555,
                            168.65211745,
                            167.11124925,
                            164.90579968,
                            161.55878356,
                            155.51145707,
                            127.44201,
                            156.17233495,
                            161.90502188,
                            165.15057292,
                            167.30472911,
                            168.81169259,
                            169.86962641,
                            170.58406283,
                            171.01452544,
                            171.19702364,
                            171.15290449,
                            170.89261409,
                            170.41776353,
                            169.72240459,
                            168.7901917,
                            167.59372321,
                            166.0893158,
                            164.20610783,
                            161.82165728,
                            158.69962022,
                            154.27846941,
                            146.53932337,
                            133.81182438,
                            148.3358975,
                            152.5583119,
                            154.65625305,
                            155.73965825,
                            156.1844266,
                            156.1592389,
                            155.74878653,
                            154.99626552,
                            153.92253185,
                            152.53226328,
                            150.81529593,
                            148.75265584,
                            146.31343354,
                            143.45165862,
                            140.10195782,
                            136.17204786,
                            131.52746573,
                            125.99673087,
                            119.40478756,
                            111.7090518,
                            102.32726097,
                            92.14673025,
                            109.60470543,
                            117.94643843,
                            124.05993472,
                            128.82210406,
                            132.60850062,
                            135.64669794,
                            138.08929106,
                            140.04349868,
                            141.58586443,
                            142.77042263,
                            143.63310546,
                            144.19382195,
                            144.458037,
                            144.4163507,
                            144.04277869,
                            143.29010889,
                            142.07840346,
                            140.26663061,
                            137.57442167,
                            133.30890459,
                            124.69490363,
                            122.03198226,
                            131.33762459,
                            135.20372477,
                            137.33085381,
                            138.54149047,
                            139.15188094,
                            139.31439676,
                            139.10938121,
                            138.57734862,
                            137.73688908,
                            136.58506852,
                            135.10499318,
                            133.2662743,
                            131.02434365,
                            128.31808763,
                        ]
                    ]
                ),
            )
        },
        coords={
            "azimuth_angles": np.array([0]),
            "elevation_angles": np.array(
                [
                    -15.0,
                    -14.8125,
                    -14.625,
                    -14.4375,
                    -14.25,
                    -14.0625,
                    -13.875,
                    -13.6875,
                    -13.5,
                    -13.3125,
                    -13.125,
                    -12.9375,
                    -12.75,
                    -12.5625,
                    -12.375,
                    -12.1875,
                    -12.0,
                    -11.8125,
                    -11.625,
                    -11.4375,
                    -11.25,
                    -11.0625,
                    -10.875,
                    -10.6875,
                    -10.5,
                    -10.3125,
                    -10.125,
                    -9.9375,
                    -9.75,
                    -9.5625,
                    -9.375,
                    -9.1875,
                    -9.0,
                    -8.8125,
                    -8.625,
                    -8.4375,
                    -8.25,
                    -8.0625,
                    -7.875,
                    -7.6875,
                    -7.5,
                    -7.3125,
                    -7.125,
                    -6.9375,
                    -6.75,
                    -6.5625,
                    -6.375,
                    -6.1875,
                    -6.0,
                    -5.8125,
                    -5.625,
                    -5.4375,
                    -5.25,
                    -5.0625,
                    -4.875,
                    -4.6875,
                    -4.5,
                    -4.3125,
                    -4.125,
                    -3.9375,
                    -3.75,
                    -3.5625,
                    -3.375,
                    -3.1875,
                    -3.0,
                    -2.8125,
                    -2.625,
                    -2.4375,
                    -2.25,
                    -2.0625,
                    -1.875,
                    -1.6875,
                    -1.5,
                    -1.3125,
                    -1.125,
                    -0.9375,
                    -0.75,
                    -0.5625,
                    -0.375,
                    -0.1875,
                    0.0,
                    0.1875,
                    0.375,
                    0.5625,
                    0.75,
                    0.9375,
                    1.125,
                    1.3125,
                    1.5,
                    1.6875,
                    1.875,
                    2.0625,
                    2.25,
                    2.4375,
                    2.625,
                    2.8125,
                    3.0,
                    3.1875,
                    3.375,
                    3.5625,
                    3.75,
                    3.9375,
                    4.125,
                    4.3125,
                    4.5,
                    4.6875,
                    4.875,
                    5.0625,
                    5.25,
                    5.4375,
                    5.625,
                    5.8125,
                    6.0,
                    6.1875,
                    6.375,
                    6.5625,
                    6.75,
                    6.9375,
                    7.125,
                    7.3125,
                    7.5,
                    7.6875,
                    7.875,
                    8.0625,
                    8.25,
                    8.4375,
                    8.625,
                    8.8125,
                    9.0,
                    9.1875,
                    9.375,
                    9.5625,
                    9.75,
                    9.9375,
                    10.125,
                    10.3125,
                    10.5,
                    10.6875,
                    10.875,
                    11.0625,
                    11.25,
                    11.4375,
                    11.625,
                    11.8125,
                    12.0,
                    12.1875,
                    12.375,
                    12.5625,
                    12.75,
                    12.9375,
                    13.125,
                    13.3125,
                    13.5,
                    13.6875,
                    13.875,
                    14.0625,
                    14.25,
                    14.4375,
                    14.625,
                    14.8125,
                    15.0,
                ]
            ),
        },
    )
    ds["gain"].attrs["units"] = "dB"
    ds["azimuth_angles"].attrs["units"] = "deg"
    ds["elevation_angles"].attrs["units"] = "deg"
    return ds


@pytest.fixture
def mock_trajectory() -> MockTrajectory:
    return MockTrajectory()


@pytest.fixture
def mock_channel_data() -> MockChannelData:
    return MockChannelData()


@pytest.fixture
def mock_product() -> MockProduct:
    return MockProduct()


@pytest.fixture
def antenna_pattern() -> xr.Dataset:
    return generate_antenna_pattern()


@pytest.fixture
def test_data_128(default_input_data_gen):
    data, peak_pos, target_pos = generate_data_for_test(
        lines=default_input_data_gen["lines"],
        samples=default_input_data_gen["samples"],
        samples_start=default_input_data_gen["samples_start"],
        lines_step=default_input_data_gen["lines_step"],
        samples_step=default_input_data_gen["samples_step"],
        fc_hz=default_input_data_gen["fc_hz"],
    )
    return data, peak_pos, target_pos


@pytest.fixture
def test_data_256(default_input_data_gen):
    data, peak_pos, _ = generate_data_for_test(
        lines=256,
        samples=256,
        samples_start=default_input_data_gen["samples_start"],
        lines_step=default_input_data_gen["lines_step"],
        samples_step=default_input_data_gen["samples_step"],
        fc_hz=default_input_data_gen["fc_hz"],
        perc=0.9,
    )
    return data, peak_pos


@pytest.fixture(scope="session")
def ref_data_irf_results():
    return _REF_DATA_IRF_RESULTS


@pytest.fixture(scope="session")
def ref_data_rcs_results():
    return _REF_DATA_RCS_RESULTS


@pytest.fixture(scope="session")
def default_input_data_gen():
    return _DEFAULT_INPUT_DATA_GENERATION


@pytest.fixture(scope="session")
def ref_time():
    return _REF_TIME


@pytest.fixture(scope="session")
def ref_ground_point():
    return _REF_GROUND_POINT


@pytest.fixture(scope="session")
def ref_points():
    return _REF_POINTS
