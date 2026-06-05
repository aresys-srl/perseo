---
icon: lucide/axis-3d
tags:
    - geometry
    - coordinates conversion
    - ecef
    - eci
    - latitude
    - longitude
    - core
---

# Coordinates { #coords data-toc-label="Coordinates" }

The `perseo_core.geometry.coords_conversions` module provides vectorized coordinate transformations involving terrestrial and celestial reference frames used in SAR geometry computations. It leverages `pyproj` [`Transformer`][pyproj.transformer.Transformer] for geodetic conversions (WGS84-based) and [`astropy.coordinates`][astropy.coordinates]  for precise Earth rotation transformations between Earth-fixed and inertial frames.

The module handles conversions between ECEF cartesian coordinates in `EPSG:4978`, geodetic LLH (Latitude, Longitude, Height) in `EPSG:4326`, UTM coordinates (Easting, Northing, Height) in `EPSG:326xx` or `EPSG:327xx`, and transformations to/from ECI inertial coordinates (`GCRS`) accounting for Earth rotation at specific UTC times.

!!! tip "Coordinates"

    This package provides the **coordinate foundation** for PERSEO geometry operations. All conversions support vectorized operations on `(N, 3)` arrays and properly handle the WGS84 ellipsoid for geodetic calculations
    and Earth rotation for inertial frame transformations.
