---
icon: lucide/alarm-clock
tags:
    - timing
    - precise_datetime
    - core
---

# Timing { #time data-toc-label="Timing" }

The `timing` module provides high-precision time handling for SAR operations within the PERSEO framework. It implements the `PreciseDateTime` class for picoseconds timing accuracy required by synthetic aperture radar processing, where precise orbit state interpolation, geocoding and Doppler calculations depend on exact time tagging.

The `PreciseDateTime` class provides arithmetic operations for time differences while maintaining precision. It is the main time representation used in the PERSEO framework.
