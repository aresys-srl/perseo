---
icon: lucide/rotate-3d
tags:
    - geometry
    - attitude
    - reference frames
    - pointing
    - quaternions
    - euler angles
    - core
---

# Attitude { #attitude data-toc-label="Attitude" }

The `attitude` module provides Spherical Linear Interpolation (SLERP) capabilities for sensor and antenna attitude representation within the PERSEO framework. The [`Attitude`][perseo_core.geometry.pointing.attitude.Attitude] class wraps SciPy's [`SLERP`][scipy.spatial.transform.Slerp] interpolator to enable smooth, numerically stable rotation interpolation on the SO(3) manifold, ensuring geometrically correct attitude transitions between discrete time samples.

The main object is the [`Attitude`][perseo_core.geometry.pointing.attitude.Attitude] class which allows to:

- Construct an attitude timeline from reference frames, quaternions or Euler angles (yaw, pitch, roll)
- Interpolate reference frames at arbitrary query times using Spherical Linear Interpolation (SLERP)

All attitude instances maintain reference frame consistency, meaning that the interpolation preserves the coordinate system of the input data (e.g., ECEF or local sensor frames) throughout the interpolation domain.
