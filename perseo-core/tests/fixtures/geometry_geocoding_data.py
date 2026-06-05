# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Test data fixtures for geometry/geocoding utilities module tests."""

import numpy as np
from scipy.constants import speed_of_light

from perseo_core.timing import PreciseDateTime
from tests.fixtures.trajectory_attitude_data import get_testing_attitude, get_testing_trajectory


def get_direct_geocoding_with_looks_test_data() -> dict[str, object]:
    """Return fixture data for direct_geocoding_with_looks functionalities.

    Returns
    -------
    dict[str, object]
        Test data for direct_geocoding_with_looks functionalities.
    """
    sensor_positions = np.array(
        [
            [5317606.94350283, 610603.985945038, 4577936.89859885],
            [5313024.53547427, 608285.563877273, 4583547.15708167],
            [5308435.7651548, 605967.120830312, 4589152.18047604],
            [5303840.63790599, 603648.660435838, 4594751.96221552],
            [5299239.15894225, 601330.18624638, 4600346.49592944],
            [5294631.33350784, 599011.701824865, 4605935.7752263],
            [5290017.16682646, 596693.210719223, 4611519.79375494],
        ]
    )
    sensor_velocity = np.array([-4.579225059127757, -2.318410347817750, 5.612873789290531])
    arf = np.array(
        [
            [-0.6045802221754875, -0.5648527276296573, -0.5616263446844129],
            [-0.304829433093796, 0.8154747325505556, -0.49201623681674933],
            [0.7359088066288877, -0.1262630455079271, -0.6652036317287437],
        ]
    )
    expected_results_pointing = np.array(
        [
            [4928985.040167449, 206139.93335815019, 4029123.540450799],
        ]
    )
    reference_point = np.array([4809369.353321205, 552244.754877962, 4140394.297540082])
    looking_direction = np.array([5317606.0, 610603.0, 4577936.0])
    wrong_looking_direction = np.array([-4.617335568769981, -2.318493268811102, 5.581386037561345])
    expected_results_looks = np.array(
        [
            [4759710.115562946, 723739.860905043, 4169511.582485821],
            [4740767.822131609, 785940.591895703, 4179749.794811581],
            [4719765.088454115, 852956.178108469, 4190295.797874668],
            [4695901.593234693, 926811.645389169, 4201333.096521494],
            [4668001.681937613, 1010339.507423213, 4213072.517214957],
            [4634228.769201612, 1107766.560515324, 4225761.582103892],
            [4591484.972384566, 1225899.798385040, 4239684.548934286],
        ]
    )
    return {
        "sensor_positions": sensor_positions,
        "sensor_velocity": sensor_velocity,
        "looking_direction": looking_direction,
        "wrong_looking_direction": wrong_looking_direction,
        "arf": arf,
        "reference_point": reference_point,
        "expected_ground_points_with_pointing": expected_results_pointing,
        "expected_ground_points_for_look_angles": expected_results_looks,
        "tolerance": {"atol": 1e-7, "rtol": 0},
    }


def get_direct_geocoding_test_data() -> dict[str, object]:
    """Return fixture data for direct_geocoding_monostatic and bistatic functionalities.

    Returns
    -------
    dict[str, object]
        Test data for direct_geocoding_monostatic and bistatic functionalities.
    """
    sensor_position = np.array(
        [4387348.749948771, 762123.3489877012, 4553067.931912004],
    )
    sensor_velocity = np.array(
        [-856.1384108174528, -329.7629775067583, 398.55830806407346],
    )
    initial_guess = np.array([4385932.628762595, 764443.4718341012, 4551945.624046889])
    range_time = np.array([2.05624579e-05])
    range_distance = range_time * speed_of_light / 2
    expected_ground_points = np.array([4385882.195057692, 764600.9869913795, 4551967.6143934])
    expected_monostatic_init = np.array([4385882.165361054, 764600.91441278, 4551967.49055163])
    return {
        "sensor_position": sensor_position,
        "sensor_velocity": sensor_velocity,
        "initial_guess": initial_guess,
        "range_time": range_time,
        "range_distance": range_distance,
        "doppler_frequency": 0,
        "geodetic_altitude": 0,
        "wavelength": 1,
        "look_direction": "RIGHT",
        "az_reps": 4,
        "rng_reps": 5,
        "high_rng_reps": 120,
        "expected_ground_points": expected_ground_points,
        "expected_monostatic_init": expected_monostatic_init,
        "tolerance": {"atol": 1e-6, "rtol": 0},
        "residual_tolerance": {"atol": 1e-8, "rtol": 0},
    }


def get_inverse_geocoding_test_data() -> dict[str, object]:
    """Return fixture data for inverse_geocoding_monostatic and bistatic functionalities.

    Returns
    -------
    dict[str, object]
        Test data for inverse_geocoding_monostatic and bistatic functionalities.
    """
    trajectory = get_testing_trajectory()
    wavelength = 1
    doppler_freq = 0

    # inputs
    init_guess = PreciseDateTime.from_utc_string("13-FEB-2023 09:34:01.500000000000")
    ground_point = np.array(
        [-2243618.48435212, -4728341.28615007, 3633267.229522297],
    )
    az_reps = 5
    rng_reps = 7

    # expected results
    azimuth_res = PreciseDateTime.from_utc_string("13-FEB-2023 09:33:58.482637823016")
    azimuth_res_mono = PreciseDateTime.from_utc_string("13-FEB-2023 09:33:58.480826322795")
    azimuth_res_attitude = PreciseDateTime.from_utc_string("13-FEB-2023 09:34:00.629921460211")
    range_res = 0.0036229998783991087
    range_res_mono = 0.0036229998773038815
    range_res_attitude = 0.0036245410151209715
    init_guess_res = PreciseDateTime.from_utc_string("13-FEB-2023 09:33:58.500000000000")
    init_guess_res_mono = PreciseDateTime.from_utc_string("13-FEB-2023 09:33:58.000000000000")
    return {
        "trajectory": trajectory,
        "attitude": get_testing_attitude(),
        "wavelength": wavelength,
        "doppler_frequency": doppler_freq,
        "init_guess": init_guess,
        "ground_point": ground_point,
        "az_reps": az_reps,
        "rng_reps": rng_reps,
        "expected_azimuth": azimuth_res,
        "expected_azimuth_mono": azimuth_res_mono,
        "expected_azimuth_attitude": azimuth_res_attitude,
        "expected_range": range_res,
        "expected_range_mono": range_res_mono,
        "expected_range_attitude": range_res_attitude,
        "expected_init_guess": init_guess_res,
        "expected_init_guess_mono": init_guess_res_mono,
        "tolerance": {"atol": 1e-10, "rtol": 0},
        "low_tolerance": {"atol": 1e-12, "rtol": 0},
        "residual_tolerance": {"atol": 1e-10, "rtol": 0},
    }
