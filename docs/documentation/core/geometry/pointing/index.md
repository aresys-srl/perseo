---
icon: lucide/bow-arrow
tags:
    - geometry
    - attitude
    - reference frames
    - pointing
    - sensor local axis
    - antenna reference frame
    - rotations
    - core
---

# Pointing { #pointing data-toc-label="Pointing" }

The `pointing` module manages sensor attitude and antenna pointing within the PERSEO geometry framework.
It defines three fundamental sensor local reference frames (ZeroDoppler, Geocentric, Geodetic) and provides tools to build Antenna Reference Frames (ARF) from Euler angles, interpolate attitudes via SLERP, and compute pointing directions from antenna steering angles. All operations work with `scipy.spatial.transform.Rotation` objects under the hood, and vectorized NumPy arrays for efficient batch processing.

<div class="result" markdown>

:lucide-move-3d:{ .lg .middle } [__Reference Frames__](reference_frames/index.md#ref-frames){ data-preview } : sensor local axes (ZeroDoppler, Geocentric, Geodetic), antenna reference frame, rotations

:lucide-rotate-3d:{ .lg .middle } [__Attitude__](attitude/index.md#attitude){ data-preview } : SLERP-based attitude interpolation from quaternions or Euler angles

</div>

!!! tip "Pointing"

    This module is essential for SAR sensor modeling, providing the mathematical machinery to transform between body-fixed sensor frames, antenna frames, and Earth-centered reference systems.
