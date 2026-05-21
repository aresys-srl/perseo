# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Test data fixtures for geometry module tests."""

import numpy as np

from perseo_core.geometry.utilities.ellipsoid import WGS84
from tests.fixtures.models_data import get_testing_trajectory


def get_angles_test_data() -> dict[str, object]:
    """Return fixture data for look/incidence-angle tests.

    Returns
    -------
    dict[str, object]
        Sensor/ground-point arrays plus comparison tolerances.
    """
    return {
        "sensor_positions": np.array(
            [
                [5317606.94350283, 610603.985945038, 4577936.89859885],
                [5313024.53547427, 608285.563877273, 4583547.15708167],
                [5308435.7651548, 605967.120830312, 4589152.18047604],
                [5303840.63790599, 603648.660435838, 4594751.96221552],
                [5299239.15894225, 601330.18624638, 4600346.49592944],
                [5294631.33350784, 599011.701824865, 4605935.7752263],
                [5290017.16682646, 596693.210719223, 4611519.79375494],
            ]
        ),
        "points": 1.0e6
        * np.array(
            [
                [4.759710115562946, 0.723739860905043, 4.169511582485821],
                [4.740767822131609, 0.785940591895703, 4.179749794811581],
                [4.719765088454115, 0.852956178108469, 4.190295797874668],
                [4.695901593234693, 0.926811645389169, 4.201333096521494],
                [4.668001681937613, 1.010339507423213, 4.213072517214957],
                [4.634228769201612, 1.107766560515324, 4.225761582103892],
                [4.591484972384566, 1.225899798385040, 4.239684548934286],
            ]
        ),
        "surface_normals": np.array(
            [
                [0.234003747210404, 0.035581544956606, 0.206369066952520],
                [0.233072478806222, 0.038639547183139, 0.206875805039975],
                [0.232039912061006, 0.041934264280303, 0.207397777162885],
                [0.230866700422629, 0.045565253495203, 0.207944065853625],
                [0.229495044663803, 0.049671770959056, 0.208525105922023],
                [0.227834651920851, 0.054461620540163, 0.209153148422883],
                [0.225733219610407, 0.060269457500911, 0.209842262631383],
            ]
        ),
        "nadir_directions": np.array(
            [
                [-5.076811128664492, -0.582954164154652, -4.397261878693197],
                [-5.072502925338587, -0.580748363122939, -4.402708576084343],
                [-5.068188606556682, -0.578542492291151, -4.408150450773756],
                [-5.063868176538823, -0.576336554975789, -4.413587496083914],
                [-5.059541639367873, -0.574130554419002, -4.419019705524370],
                [-5.055208999156272, -0.571924493873826, -4.424447072581663],
                [-5.050870259998338, -0.569718376579988, -4.429869590778411],
            ]
        ),
        "tolerance": {"atol": 1e-8, "rtol": 0},
    }


def get_coords_conversions_test_data():
    """Return fixture data for ``xyz2llh`` and ``llh2xyz`` tests.

    Returns
    -------
    dict[str, object]
        Scalar/vector coordinate samples and tolerances.
    """
    xyz = [2.354828227500000e4, 9.457755947560000e05, 6.286558297154000e06]
    llh = [1.4224117467026256, 1.5459030877183293, 123.45599971152842]
    llh_deg = [np.rad2deg(llh[0]), np.rad2deg(llh[1]), llh[2]]

    xyz_vec = np.stack(
        [
            [5336078.7743305163, WGS84.a, 0.0, -WGS84.a, 0.0],
            [2346746.5942683504, 0.0, WGS84.a, -WGS84.a, 0.0],
            [4033846.0446414836, 0.0, 0.0, 0.0, WGS84.b],
        ],
        axis=-1,
    )
    llh_vec = np.stack(
        [
            [0.608159140099359, 0.0, 0.0, 0.0, np.pi / 2],
            [0.414329746479487, 0.0, np.pi / 2, -2.35619449019234, 0.0],
            [717733.676999941, 0.0, 0.0, 2641910.84807364, 0.0],
        ],
        axis=-1,
    )
    llh_vec_deg = llh_vec.copy()
    llh_vec_deg[:, :2] = np.rad2deg(llh_vec_deg[:, :2])

    return {
        "xyz": xyz,
        "llh": llh,
        "llh_deg": llh_deg,
        "xyz_vec": xyz_vec,
        "llh_vec": llh_vec,
        "llh_vec_deg": llh_vec_deg,
        "tolerance": {"atol": 1e-8, "rtol": 1e-8},
    }


def get_doppler_test_data() -> dict[str, object]:
    """Return fixture data for doppler functions tests."""
    trajectory = get_testing_trajectory()
    return {
        "N": 5,
        "trajectory": trajectory,
        "azimuth_time": trajectory.domain[0] + 2.0,
        "ground_point": np.array(
            [-2243618.48435212, -4728341.28615007, 3633267.229522297],
        ),
        "frequency_doppler_centroid": 0.0,
        "wavelength": 1.0,
        "carrier_frequency": 5405000454.33435,
        "az_steering_rate_rad_s": 0.027757171601738514,
        "doppler_rate": -2202.4494321715547,
        "pv_scalar": 3.954046405851841e-07,
        "distance": 2940.7387133883235,
        "sensor_velocity": np.array([-856.1384108174528, -329.7629775067583, 398.55830806407346]),
        "los": np.array([1416.1211861753836, -2320.122846400016, 1122.3078651148826]),
        "doppler_result": 2.6891518024707355e-10,
        "gradient_result": np.array([0.5822607815646932, 0.22427220480713764, -0.2710599933611937]),
        "doppler_rate_result": -3607.893770499216,
        "steering_doppler_frequency_result": [-1711.8272616173901, 1711.8262922977972],
        "doppler_centroid_result": 96.22056463133517,
        "residual_monostatic_result": 96.22056463133458,
        "tolerance": {"atol": 1e-8, "rtol": 0},
    }
