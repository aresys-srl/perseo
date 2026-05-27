---
icon: lucide/shapes
tags:
    - geometry
    - attitude
    - trajectory
    - direct geocoding
    - inverse geocoding
    - coordinates
    - angles
    - core
---

# Geometry { #geom data-toc-label="Geometry" }

The `geometry` package provides SAR-specific geometric computations for the PERSEO framework.
It is built around the WGS84 ellipsoid and operates on vectorized NumPy arrays of ECEF coordinates.
Core functionalities range from coordinate conversions between terrestrial and celestial frames
to angular computations (squint, incidence, look), Doppler analysis, direct and inverse geocoding
(monostatic and bistatic), antenna pointing and attitude, and ellipsoidal geometry utilities.

<div class="result" markdown>

:lucide-axis-3d:{ .lg .middle } [__Coordinates__](coordinates/index.md#coords){ data-preview } : XYZ ↔ LLH, UTM ↔ LLH and ECEF ↔ ECI conversions

:lucide-map:{ .lg .middle } [__Geocoding__](geocoding/index.md#geocoding){ data-preview } : direct and inverse geocoding, monostatic and bistatic

:lucide-bow-arrow:{ .lg .middle } [__Pointing__](pointing/index.md#pointing){ data-preview } : reference frames, antenna reference frame, and attitude

:lucide-orbit:{ .lg .middle } [__Trajectory__](trajectory/index.md#traj){ data-preview } : interpolated trajectory, velocity, and acceleration

</div>

!!! info "Geometry"

    This package is the **geometric engine** of the PERSEO framework. All modules rely on the WGS84 ellipsoid model and are designed for vectorized operations `(N, 3)` coordinate arrays.
