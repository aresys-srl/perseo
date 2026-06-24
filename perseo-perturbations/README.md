# perseo-perturbations

[![PyPI version](https://img.shields.io/pypi/v/perseo-perturbations)](https://pypi.org/project/perseo-perturbations/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)

**P**ython **E**cosystem for **R**emote **S**ensing & **E**arth **O**bservation (PERSEO) PERTURBATIONS package for computing
**geodynamics displacements** (Plate Tectonics, Solid Earth Tides) and **atmospheric delays** (Ionospheric,
Tropospheric) needed for SAR products analysis, together with a **Geomagnetic Field** model (IGRF-14).

## Features

### Atmospheric Delays

Correction of SAR signal propagation delays through the Earth's atmosphere.

- **Tropospheric Delay**: Vienna Mapping Function 3 (VMF3) model with precomputed Legendre coefficients
  - Hydrostatic (dry) and wet components
  - Station grid point coordinates (1&deg;&times;1&deg; and 5&deg;&times;5&deg; grids)
- **Ionospheric Delay**: Total Electron Content (TEC) based dispersive medium model
  - Frequency-dependent phase advance

### Geodynamics Displacements

Accounting for solid Earth deformations affecting SAR geolocation and phase interpretation.

- **Earth Solid Tides**: periodic crustal deformations from Moon and Sun gravitational attraction
  - Python wrapper of the IERS Conventions (2003) `solid.for` code by Dennis Milbert[^1]
  - Vertical displacements up to ~40 cm, horizontal up to ~10 cm
- **Plate Tectonics (Secular Motion)**: ITRF2014 plate motion model
  - Linear displacement over time, mm/year to cm/year velocities

### Geomagnetic Field

- **IGRF-14**: 14th iteration of the International Geomagnetic Reference Field (released October 2024)
  - Spherical harmonic model up to degree 13
  - Valid from 1900 to 2030
  - Geocentric and geodetic (WGS84) coordinate support
  - Output components: radial, southward, eastward (geocentric) / east, north, up (geodetic)
  - Derived magnetic inclination and declination

## Installation

```bash
pip install perseo-perturbations
```

> **Note**: This package uses a Fortran wrapper for solid tide computations and requires a working Fortran compiler for building from source. Pre-built wheels are available on PyPI.

## License

This project is licensed under the MIT License. See the [LICENSE.txt](LICENSE.txt) file for details.

Copyright &copy; 2026-present Aresys S.r.L. <info@aresys.it>

[^1]: Dennis Milbert, Ph.D., Chief Geodesist, National Geodetic Survey, NOAA, (retired) [https://geodesyworld.github.io/SOFTS/solid.htm](https://geodesyworld.github.io/SOFTS/solid.htm)
