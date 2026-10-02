# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Test data fixtures for navigation and pointing module tests."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from perseo_core.geometry.navigation import CubicSplineTrajectory
from perseo_core.geometry.pointing import (
    Attitude,
    compute_antenna_attitude_from_euler_angles,
    compute_sensor_local_axis,
)
from perseo_core.timing import PreciseDateTime


@dataclass
class TestingStateVectors:
    """Testing state vectors dataclass"""

    sensor_positions: np.ndarray
    sensor_velocities: np.ndarray
    time_axis_relative: np.ndarray
    time_axis_origin: PreciseDateTime
    time_axis: np.ndarray
    time_step: float


def get_attitude_test_data() -> dict:
    """Return fixture data for ``Attitude`` tests.

    Returns
    -------
    dict
        Knot times, Euler angles, and reference rotations.
    """
    times = np.array([0.0, 4.0, 8.0])
    euler_angles = np.deg2rad(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 45.0], [0.0, 45.0, 90.0]]))
    antenna_reference_frames = Rotation.from_euler(seq="ZYX", angles=euler_angles, degrees=False).as_matrix()

    return {
        "times": times,
        "euler_angles": euler_angles,
        "antenna_reference_frames": antenna_reference_frames,
    }


def _get_testing_state_vectors() -> TestingStateVectors:
    """Getting testing state vectors"""
    positions = np.array(
        [
            [-2542286.449576481, -5094859.4894666, 3901083.7183820857],
            [-2542066.5079547316, -5092594.238796566, 3904175.2612526673],
            [-2541845.6347068036, -5090327.468904538, 3907265.6186505724],
            [-2541623.8297894364, -5088059.180544005, 3910354.7896390846],
            [-2541401.0931600337, -5085789.374468956, 3913442.7732819766],
            [-2541177.424776237, -5083518.051433686, 3916529.5686432645],
            [-2540952.8245959855, -5081245.212193193, 3919615.1747874315],
            [-2540727.2925774334, -5078970.857502782, 3922699.5907792132],
            [-2540500.828679304, -5076694.988118281, 3925782.8156838706],
            [-2540273.432860546, -5074417.604795827, 3928864.8485669065],
            [-2540045.1050804476, -5072138.708292266, 3931945.688494257],
            [-2539815.8452985254, -5069858.299364754, 3935025.334532101],
            [-2539585.653474864, -5067576.378770969, 3938103.785747145],
            [-2539354.5295697646, -5065292.947268894, 3941181.0412063445],
            [-2539122.47354382, -5063008.0056172125, 3944257.099977135],
            [-2538889.4853578685, -5060721.554574907, 3947331.9611271904],
            [-2538655.5649734233, -5058433.594901471, 3950405.6237246613],
            [-2538420.712352204, -5056144.127356704, 3953478.086837955],
            [-2538184.9274560884, -5053853.152701135, 3956549.3495360035],
            [-2537948.21024718, -5051560.671695597, 3959619.410887985],
        ]
    )
    velocities = np.array(
        [
            [439, 4529, 6184],
            [440, 4532, 6181],
            [442, 4535, 6179],
            [444, 4538, 6177],
            [446, 4544, 6174],
            [448, 4547, 6172],
            [450, 4550, 6170],
            [452, 4552, 6167],
            [454, 4554, 6165],
            [455, 4555, 6162],
            [457, 4557, 6160],
            [459, 4560, 6158],
            [461, 4565, 6155],
            [463, 4568, 6153],
            [465, 4572, 6150],
            [466, 4574, 6148],
            [468, 4578, 6146],
            [469, 4580, 6144],
            [472, 4583, 6141],
            [474, 4586, 6138],
        ]
    )
    time_step = 0.5
    time_axis_relative = np.asarray([time_step * k for k in range(positions.size // 3)])
    time_axis_origin = PreciseDateTime.from_utc_string("13-FEB-2023 09:33:56.000000")
    return TestingStateVectors(
        sensor_positions=positions,
        sensor_velocities=velocities,
        time_axis_relative=time_axis_relative,
        time_axis_origin=time_axis_origin,
        time_axis=time_axis_relative + time_axis_origin,
        time_step=time_step,
    )


def get_testing_trajectory() -> CubicSplineTrajectory:
    """Getting a testing trajectory object"""
    state_vectors = _get_testing_state_vectors()
    return CubicSplineTrajectory(
        times=state_vectors.time_axis,
        positions=state_vectors.sensor_positions,
        velocities=state_vectors.sensor_velocities,
    )


def get_testing_attitude() -> Attitude:
    """Getting a testing attitude object"""
    trajectory = get_testing_trajectory()
    angles_rad = np.zeros_like(trajectory.positions)
    angles_rad[:, 2] = np.deg2rad(-30.0)

    # this is the Zero Doppler reference frame expressed in ECEF coordinates
    zero_doppler_local_axis = compute_sensor_local_axis(
        sensor_positions=trajectory.position(trajectory.times),
        sensor_velocities=trajectory.velocity(trajectory.times),
        reference_frame="ZERODOPPLER",
    )

    # compute antenna attitude in ECEF from Euler angles
    return compute_antenna_attitude_from_euler_angles(
        ypr_rad=angles_rad, rotation_order="YPR", times=trajectory.times, sensor_local_axis=zero_doppler_local_axis
    )


def get_trajectory_test_data() -> dict:
    """Return fixture data for ``CubicSplineTrajectory`` tests.

    Returns
    -------
    dict
        State vectors, trajectory object, expected samples, and tolerance.
    """
    state_vectors = _get_testing_state_vectors()
    trajectory = get_testing_trajectory()

    tolerance = 1e-6
    test_times = np.array([0.67, 2.56, 3.23, 5.8, 7.3, 8.4, 9.28]) + trajectory.domain[0]

    expected_pos = np.array(
        [
            [-2541991.51667295, -5091823.70786736, 3905226.12193514],
            [-2541150.5219575, -5083245.3907668, 3916899.90420767],
            [-2540849.19560279, -5080199.19720824, 3921034.15400884],
            [-2539677.84205497, -5068489.32836387, 3936872.54869185],
            [-2538982.79233349, -5061636.31586544, 3946102.16084407],
            [-2538467.75565909, -5056602.13916507, 3952863.69473292],
            [-2538052.45005212, -5052569.50979786, 3958268.80885579],
        ]
    )
    expected_vel = np.array(
        [
            [441.4473803, 4533.05342227, 6181.09986721],
            [448.49214067, 4544.52639022, 6172.11662186],
            [450.98948273, 4548.58823598, 6168.92732246],
            [460.57005549, 4564.14349903, 6156.66343748],
            [466.16286972, 4573.20402512, 6149.48291368],
            [470.25419444, 4579.82631009, 6144.18013695],
            [473.58261429, 4585.18911946, 6140.06974106],
        ]
    )
    expected_acc = np.array(
        [
            [3.77763037, 6.095279, -5.02893538],
            [3.72759827, 6.06460755, -4.75951289],
            [3.72737768, 6.06038945, -4.76204088],
            [3.72838663, 6.04500478, -4.78121927],
            [3.72227767, 6.02750557, -4.80892479],
            [3.59289969, 5.85534651, -5.14276888],
            [2.60597199, 4.59067105, -7.62766475],
        ]
    )

    return {
        "state_vectors": state_vectors,
        "trajectory": trajectory,
        "tolerance": tolerance,
        "test_times": test_times,
        "expected_positions": expected_pos,
        "expected_velocities": expected_vel,
        "expected_accelerations": expected_acc,
    }


def get_orbit_test_data() -> dict:
    positions = np.array(
        [
            [-292078.981035, -2517031.275239, 6596473.696232],
            [-316039.760165, -2449298.315404, 6620768.594309],
            [-339866.153398, -2381256.354981, 6644316.964116],
            [-363555.049673, -2312913.213226, 6667116.165252],
            [-387103.364844, -2244276.747187, 6689163.641456],
            [-410508.042067, -2175354.850826, 6710456.920867],
            [-433766.052182, -2106155.454104, 6730993.616277],
            [-456874.394095, -2036686.522077, 6750771.425375],
            [-479830.095164, -1966956.053964, 6769788.131022],
            [-502630.211573, -1896972.082251, 6788041.601543],
            [-525271.828703, -1826742.671722, 6805529.790921],
            [-547752.061516, -1756275.918481, 6822250.739033],
            [-570068.054909, -1685579.948996, 6838202.571836],
            [-592216.984066, -1614662.919177, 6853383.501556],
            [-614196.054791, -1543533.013452, 6867791.826816],
            [-636002.50386, -1472198.443815, 6881425.932854],
            [-657633.599357, -1400667.448918, 6894284.291735],
            [-679086.64102, -1328948.293109, 6906365.46252],
            [-700358.96058, -1257049.265448, 6917668.091426],
            [-721447.922116, -1184978.67869, 6928190.911978],
            [-742350.92237, -1112744.86835, 6937932.745126],
            [-763065.391069, -1040356.191722, 6946892.499321],
            [-783588.791236, -967821.026922, 6955069.170629],
            [-803918.6195, -895147.771925, 6962461.842841],
            [-824052.406391, -822344.843618, 6969069.687581],
            [-843987.716644, -749420.676813, 6974891.964382],
            [-863722.149494, -676383.723255, 6979928.020778],
            [-883253.338965, -603242.450647, 6984177.292362],
            [-902578.954148, -530005.341676, 6987639.302849],
            [-921696.699478, -456680.893025, 6990313.66411],
        ]
    )
    velocities = np.array(
        [
            [-2402.692841, 6757.586242, 2466.726099],
            [-2389.410691, 6788.87598, 2392.208091],
            [-2375.816104, 6819.385716, 2317.421845],
            [-2361.91175, 6849.111622, 2242.375759],
            [-2347.700338, 6878.049961, 2167.078261],
            [-2333.184619, 6906.197086, 2091.537804],
            [-2318.367381, 6933.549442, 2015.762869],
            [-2303.25145, 6960.103568, 1939.761962],
            [-2287.839691, 6985.856095, 1863.543613],
            [-2272.135006, 7010.803747, 1787.116376],
            [-2256.140333, 7034.943343, 1710.488827],
            [-2239.858646, 7058.271796, 1633.669562],
            [-2223.292956, 7080.786111, 1556.667198],
            [-2206.446309, 7102.483391, 1479.490372],
            [-2189.321783, 7123.360829, 1402.147739],
            [-2171.922496, 7143.415718, 1324.647973],
            [-2154.251594, 7162.645443, 1246.999763],
            [-2136.312262, 7181.047485, 1169.211816],
            [-2118.107715, 7198.619423, 1091.292852],
            [-2099.641204, 7215.358932, 1013.251605],
            [-2080.91601, 7231.263782, 935.096825],
            [-2061.935448, 7246.331844, 856.837268],
            [-2042.702863, 7260.561082, 778.481705],
            [-2023.22163, 7273.949561, 700.038914],
            [-2003.495157, 7286.495441, 621.51768],
            [-1983.526878, 7298.196978, 542.926799],
            [-1963.320257, 7309.052528, 464.275072],
            [-1942.878788, 7319.060543, 385.571308],
            [-1922.20599, 7328.219572, 306.824321],
            [-1901.305411, 7336.528264, 228.04293],
        ]
    )
    initial_time = PreciseDateTime.from_numeric_datetime(2026, 1, 1, 22, 59, 42)
    time_axis = np.arange(0, 300, 10) + initial_time
    return {
        "positions": positions,
        "velocities": velocities,
        "time_axis": time_axis,
        "tolerance": 0.01,
    }
