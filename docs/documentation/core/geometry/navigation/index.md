---
icon: lucide/orbit
tags:
    - geometry
    - trajectory
    - state vectors
    - orbit
    - core
---

# Trajectory { #traj data-toc-label="Trajectory" }

The `trajectory` module defines the interface and implementations for sensor trajectory modeling in the PERSEO framework.
It provides an abstract base class [`Trajectory`][perseo_core.geometry.navigation.trajectory.Trajectory] that specifies the contract for position, velocity,
and acceleration queries at arbitrary times, along with two concrete implementations:

- [`CubicSplineTrajectory`][perseo_core.geometry.navigation.cubic_spline_trajectory.CubicSplineTrajectory]
implementation using Scipy's [`CubicSpline`][scipy.interpolate.CubicSpline] interpolator.

- [`PropagatedCubicSplineOrbit`][perseo_core.geometry.navigation.propagated_cubic_spline_orbit.PropagatedCubicSplineOrbit]{ #propagated-cubic-spline-orbit }
extends orbit evaluation beyond the time range covered by the input state vectors using high-precision numerical
integration provided by [SatKit](https://satkit.dev/). Orbit propagation can be performed both forward and backward in time.
This implementation is intended exclusively for satellite orbit propagation.


All trajectories support vectorized evaluation.

The `CubicSplineTrajectory` implementation ensures is the preferred trajectory for all PERSEO modules that require interpolated trajectory state vectors.

!!! tip "Custom Trajectory"

    The trajectory interface is designed to be easily adapted to support custom implementations. This means that any interpolator can be used to define a trajectory, as long as it provides the necessary methods and attributes.
