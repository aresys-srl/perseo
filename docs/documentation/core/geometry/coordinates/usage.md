---
icon: lucide/book-open-check
title: "Tutorial"
tags:
    - coordinates
    - usage
    - tutorial
    - core
---

# Coordinates usage

The main functions available in ``perseo_core.geometry.coords_conversions`` are:

- [`xyz2llh`][perseo_core.geometry.coords_conversions.xyz2llh]: Convert XYZ ECEF cartesian coordinates to geodetic LLH (latitude, longitude, height)
- [`llh2xyz`][perseo_core.geometry.coords_conversions.llh2xyz]: Convert geodetic LLH coordinates to XYZ ECEF cartesian coordinates
- [`ecef2eci`][perseo_core.geometry.coords_conversions.ecef2eci]: Transform positions and velocities from ECEF (ITRS) to ECI (GCRS) frame
- [`eci2ecef`][perseo_core.geometry.coords_conversions.eci2ecef]: Transform positions and velocities from ECI (GCRS) to ECEF (ITRS) frame

All functions support both scalar inputs and batch operations on arrays of coordinates.

## ECEF <-> LLH conversion

Convert Earth-Centered Earth-Fixed (ECEF) cartesian coordinates to geodetic Latitude, Longitude, Height (LLH) and
vice versa.

```python title="Batch coordinate conversion"
import numpy as np
from perseo_core.geometry.coords_conversions import xyz2llh, llh2xyz

# multiple points (N, 3) array, lat [deg]/lon[deg]/height[m]
llh_coords_deg = np.array([
    [-26.83470987, 151.1656039, 409.4544],  # Point 1
    [-26.94651144, 151.1438779, 390.5168],  # Point 2
    [-27.08563567, 150.2596331, 371.1494],  # Point 3
    [-27.30887139, 151.2719591, 385.242],   # Point 4
])

# convert all points at once
xyz_array_from_deg = llh2xyz(llh_coords_deg, radians=False)  # Returns (N, 3) array

# this is equivalent to:
llh_coords_rad = llh_coords_deg.copy()
llh_coords_rad[:, :2] = np.deg2rad(llh_coords_deg[:, :2])

xyz_array_from_rad = llh2xyz(llh_coords_rad)  # Returns (N, 3) array

# assessing equivalence
np.testing.assert_allclose(xyz_array_from_rad, xyz_array_from_deg, atol=1e-9)

# convert back
llh_coords_rad_back = xyz2llh(xyz_array_from_rad)

# assessing equivalence
np.testing.assert_allclose(llh_coords_rad_back, llh_coords_rad, atol=1e-9)
```

## ECEF <-> ECI conversion

Convert Earth-Centered Earth-Fixed (ECEF/ITRS) coordinates to Earth-Centered Inertial (ECI/GCRS) coordinates.
This transformation accounts for Earth rotation at specific UTC times.

```python title="Batch ECEF to ECI"
import numpy as np
from perseo_core.geometry.coords_conversions import ecef2eci
from perseo_core.timing.precise_datetime import PreciseDateTime

# Multiple positions at different times
ecef_positions = np.array(
    [
        [3.724745197526000e06, -6.015622745767000e06, 1.321425413040000e05],
        [3.684766194699000e06, -6.041272641007000e06, -9.076226593700000e04],
        [3.640972112690000e06, -6.060645974807000e06, -3.135749264620000e05],
        [3.593435105405000e06, -6.073707420235000e06, -5.360690852350000e05],
    ]
)

ecef_velocities = np.array(
    [
        [-1.35408519e03, -8.2027485e02, -7.4304872e03],
        [-1.48086852e03, -6.1083496e02, -7.4261573e03],
        [-1.56405031e03, -4.7056104e02, -7.4190789e03],
        [-1.64611421e03, -3.2984706e02, -7.4086513e03],
    ]
)

# Times can be scalar (same time for all) or array (one time per position)
times_str = [
    "2024-07-07T09:55:08.417634",
    "2024-07-07T09:55:18.417635",
    "2024-07-07T09:56:08.417635",
    "2024-07-07T09:56:28.417634",
]
times = np.array([PreciseDateTime.from_utc_string(t) for t in times_str])

# Transform all at once
eci_positions, eci_velocities = ecef2eci(
    positions=ecef_positions,
    velocities=ecef_velocities,
    times=times
)

# convert back
ecef_positions_back, ecef_velocities_back = eci2ecef(
    positions=eci_positions,
    velocities=eci_velocities,
    times=times
)

# assessing equivalence
np.testing.assert_allclose(ecef_positions, ecef_positions_back, atol=1e-9)
np.testing.assert_allclose(ecef_velocities, ecef_velocities_back, atol=1e-9)
```
