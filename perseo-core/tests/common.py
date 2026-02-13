# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Common Unittest data and utilities"""

from __future__ import annotations

import numpy as np

from perseo_core.models.trajectories import CubicSplineTrajectory
from perseo_core.timing.precise_datetime import PreciseDateTime


def get_testing_state_vectors() -> dict[str, np.ndarray]:
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
    dT = 0.5
    time_axis_relative = np.asarray([dT * k for k in range(positions.size // 3)])
    time_axis_origin = PreciseDateTime.from_utc_string("13-FEB-2023 09:33:56.000000")
    return {
        "sensor_positions": positions,
        "sensor_velocities": velocities,
        "time_axis_relative": time_axis_relative,
        "time_axis_origin": time_axis_origin,
        "time_axis": time_axis_relative + time_axis_origin,
        "dT": dT,
    }


def get_testing_trajectory() -> CubicSplineTrajectory:
    """Getting a testing trajectory object"""
    state_vectors = get_testing_state_vectors()
    return CubicSplineTrajectory(
        times=state_vectors["time_axis"],
        positions=state_vectors["sensor_positions"],
        velocities=state_vectors["sensor_velocities"],
    )
