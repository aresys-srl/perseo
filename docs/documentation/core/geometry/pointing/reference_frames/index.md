---
icon: lucide/move-3d
tags:
    - geometry
    - reference frames
    - pointing
    - sensor local axis
    - antenna reference frame
    - rotations
    - core
---

# Reference Frames { #ref-frames data-toc-label="Reference Frames" }

In attitude and navigation applications, rotations are always defined between reference frames. It is therefore essential to clearly specify which frames are involved and how they are related.

This section defines three commonly used reference frames:

- A **Global Reference Frame** (e.g., ECEF)
- A **Local Reference Frame** (Sensor Body-Centered) (e.g., Zero Doppler)
- An **Antenna Reference Frame (ARF)**, centered to the antenna mounted on the sensor

All rotations must explicitly state the source and target frames.

The [`reference_frames`][perseo_core.geometry.pointing.reference_frames] module provides utilities to compute rotations needed to transform between reference frames.

## Global Reference Frame

The Global Reference Frame provides an Earth-fixed or inertial reference for navigation and positioning, e.g. *ECEF*

**Example: ECEF (Earth-Centered, Earth-Fixed)**

- **Origin**: Earth's center of mass
- **Axis X**: Intersection of equator and Greenwich meridian
- **Axis Y**: 90° east along equator
- **Axis Z**: Earth's rotation axis (North pole)
- Right-handed orthonormal frame

## Local Reference Frame

The Local Reference Frame is rigidly attached to the sensor body platform, e.g. *Zero Doppler, Geodetic, Geocentric*

**Example: Zero Doppler**

- **Origin**: Defined by the sensor manufacturer (often the mechanical center or IMU origin)
- **Axis X**: oriented as sensor velocity
- **Axis Y**: cross product between x and sensor position corrected with Earth eccentricity
- **Axis Z**: unit vector completing the reference frame
- Right-handed orthonormal frame

## Antenna Reference Frame (ARF)

The Antenna Reference Frame is rigidly attached to the antenna mounted on the sensor.

- **Origin**: Antenna mounting point
- **Axis X**: normal to the boresight plane
- **Axis Y**: unit vector completing the reference frame
- **Axis Z**: boresight unit vector
- Right-handed orthonormal frame

## Changing the Reference System

**PERSEO Core** provides utilities to express an existing rotation in a new reference system. This means that the user can define rotations in a given reference frame, may it be a Local Reference Frame (i.e. Zero Doppler) or a Global Reference Frame (i.e. ECEF), and then re-express the interpolated rotations in a different reference system.

To do so, the user must provide a time-dependent sequence of reference frame transformations as ``numpy`` (N, 3) array objects,
computed for example using the [`compute_sensor_local_axis`][perseo_core.geometry.pointing.reference_frames.compute_sensor_local_axis] function, where each rotation matrix represents a proper rotation matrix defining the transformation between the original and the
new reference system for each time frame.

These transformations can be applied to rotations expressed in the original reference frame to **re-express** them in the
new reference system.

Conceptually, if:

- $R_{A}$ is the original attitude expressed in frame A,
- $R_{B \leftarrow A}$ is the rotation from frame A to frame B,

then the re-expressed attitude becomes:

$$
R_{B} = R_{B \leftarrow A} \, R_{A}
$$
