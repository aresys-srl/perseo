# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Geodynamics Earth Crustal Displacement: Plate Tectonics submodule
-----------------------------------------------------------------

Accounting for Plate Tectonics movement using drift velocities or ITRF2014 model.
"""

from __future__ import annotations

from enum import Enum

import numpy as np


# custom errors
class WrongTectonicPlateReferenceError(ValueError):
    """Wrong Tectonic Plate reference value, not listed in ITRF2014-PMM"""


# custom enum classes
class ITRF2014PlatesRotationPoles(Enum):
    """Absolute plate rotation poles (or angular velocities) in [milliarcsec/yr], ITRF2014-PMM"""

    ANTA = [-0.248, -0.324, 0.675]
    ARAB = [1.154, -0.136, 1.444]
    AUST = [1.510, 1.182, 1.215]
    EURA = [-0.085, -0.531, 0.770]
    INDI = [1.154, -0.005, 1.454]
    NAZC = [-0.333, -1.544, 1.623]
    NOAM = [0.024, -0.694, -0.063]
    NUBI = [0.099, -0.614, 0.733]
    PCFC = [-0.409, 1.047, -2.169]
    SOAM = [-0.270, -0.301, -0.140]
    SOMA = [-0.121, -0.794, 0.884]


# TODO: change this code accordingly to the bugfix that has been identified, related to ECEF ITRF 2014 reference system


def compute_displacement(
    xyz_coords: np.ndarray,
    time_delta: float,
    plate_ref: str | ITRF2014PlatesRotationPoles | None = None,
    drift_vel: np.ndarray = None,
) -> np.ndarray:
    """Compute point target coordinates displacement due to tectonic plate motion using ITRF2014 plate motion model.

    Source (Plate tectonics)
    Zuheir Altamimi et al., "ITRF2014 plate motion model", Geophysical Journal International, 2017
    'https://academic.oup.com/gji/article/209/3/1906/3095992'

    Parameters
    ----------
    xyz_coords : np.ndarray
        xyz coordinates, in the form (3,) or (N, 3)
    time_delta : float
        time difference between product/scene acquisition time and point target coordinates reference time in seconds
    plate_ref : str | ITRF2014PlatesRotationPoles | None, optional
        plate reference name id, by default None
    drift_vel : np.ndarray, optional
        drift velocities along x, y and z, same shape as xyz_coords, by default None

    Returns
    -------
    np.ndarray
        [x, y, z] displacement in meters due to plate tectonics for each input point

    Raises
    ------
    WrongTectonicPlateReference
        if plate reference name id is not listed in the ITRF2014-PMM
    """

    if plate_ref is None and drift_vel is None:
        raise RuntimeError("specify at least one input between plate ref and drift velocities")

    if isinstance(plate_ref, str):
        try:
            rotation_poles = np.array(ITRF2014PlatesRotationPoles[plate_ref.upper()].value)
        except KeyError as exc:
            raise WrongTectonicPlateReferenceError(f"Plate {plate_ref} is not defined") from exc

    elif isinstance(plate_ref, ITRF2014PlatesRotationPoles):
        rotation_poles = np.array(plate_ref.value)

    if drift_vel is not None:
        # using input drift velocities to compute the displacement
        return drift_vel * time_delta

    # converting rotation poles [milliarcsec/yr -> rad/s]
    rotation_poles *= (1 / 1000 * 1 / 3600 * np.pi / 180) / (3600 * 24 * 365.25)

    # compute displacement velocity
    displacement_velocity = np.cross(rotation_poles, xyz_coords)

    return displacement_velocity * time_delta
