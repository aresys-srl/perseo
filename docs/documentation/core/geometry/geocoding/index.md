---
icon: lucide/map
tags:
    - geometry
    - direct geocoding
    - inverse geocoding
    - core
---

# Geocoding { #geocoding data-toc-label="Geocoding" }

The `geocoding` module provides the core SAR geocoding capabilities for the PERSEO framework, implementing both direct and inverse geocoding algorithms.

The module supports both monostatic (single sensor as both transmitter and receiver) and bistatic (separate transmitter and receiver) SAR configurations.
All algorithms use Newton iteration for solving the system of equations and operate on vectorized NumPy arrays for efficient batch processing.
The WGS84 ellipsoid serves as the Earth surface model with configurable altitude offsets.

- **Direct geocoding**: computes ground point coordinates from sensor state vectors (position, velocity) at given azimuth times.
- **Inverse geocoding**: computes the sensor times (azimuth, range) at which given ground points are observed.
