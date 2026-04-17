# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Test data fixtures for geometry utilities module tests."""

import numpy as np


def get_reference_frames_test_data():
    """Return fixture data for reference-frame tests.

    Returns
    -------
    dict[str, object]
        Sensor state, reference frames, and tolerance.
    """
    return {
        "sensor_position": np.asarray([26512.279931507, 1064819.379506800, -7083173.555337110]),
        "sensor_velocity": np.asarray([7529.609430015988, -342.978175622686, -23.376907795264]),
        "zerodoppler_frame_reference": np.asarray(
            [
                [0.998959378858231, 0.045461584707689, -0.003661107352025],
                [-0.045503192226166, 0.987972305556891, -0.147784244592682],
                [-0.003101433282541, 0.147797049254939, 0.989012847916106],
            ],
            dtype=float,
        ),
        "geocentric_frame_reference": np.asarray(
            [
                [0.998949483740530, 0.045675252972446, -0.003701378179995],
                [-0.045717707103110, 0.987831098247604, -0.148659384473924],
                [-0.003133718520000, 0.148672433896922, 0.988881533454540],
            ],
            dtype=float,
        ),
        "geodetic_frame_reference": np.asarray(
            [
                [0.998949449735728, 0.045676868615385, -0.003690602413993],
                [-0.045719074555789, 0.987896068814253, -0.148226594925155],
                [-0.003124595085362, 0.148239606363606, 0.988946538499789],
            ],
            dtype=float,
        ),
        "geodetic_point_reference": np.asarray([23539.167841732884, 945409.529474431, -6286488.197273431]),
        "tolerance": 1e-9,
    }


def get_rotation_test_data():
    """Return fixture data for Euler-angle rotation tests.

    Returns
    -------
    dict[str, object]
        Yaw/pitch/roll values and reference tolerance.
    """
    yaw = np.deg2rad(10)
    pitch = np.deg2rad(15)
    roll = np.deg2rad(20)

    return {
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
        "euler_angles": np.stack([yaw, pitch, roll], axis=-1),
        "tolerance": 1e-9,
    }


def get_ground_velocity_test_data():
    """Return fixture data for ground-velocity tests.

    Returns
    -------
    dict[str, object]
        Look-angle samples, expected velocities, and tolerance.
    """
    look_angles_deg = np.arange(15.0, 50.0, 5.0)
    look_angles_rad = np.deg2rad(look_angles_deg)

    return {
        "azimuth_time_offset": 2.3,
        "look_angles_deg": look_angles_deg,
        "look_angles_rad": look_angles_rad,
        "expected_velocities": np.array(
            [
                7073.866866931723,
                7068.0743880794025,
                7061.324649385491,
                7053.284329192286,
                7043.447653796795,
                7031.009082103516,
                7014.600593133645,
            ]
        ),
        "tolerance": 1e-6,
    }
