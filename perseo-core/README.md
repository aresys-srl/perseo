# perseo-core

[![PyPI version](https://img.shields.io/pypi/v/perseo-core)](https://pypi.org/project/perseo-core/)
[![Conda](https://img.shields.io/conda/vn/conda-forge/perseo-core)](https://anaconda.org/conda-forge/perseo-core)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)

**P**ython **E**cosystem for **R**emote **S**ensing & **E**arth **O**bservation (PERSEO) CORE package for generic and broadly
used operations and structures, serving all other PERSEO modules and projects depending on this framework.

## Features

### Geometry

The geometric engine of the PERSEO framework, built around the WGS84 ellipsoid with vectorized NumPy arrays of ECEF coordinates.

- **Angles**: squint, incidence, look angle computation
- **Coordinates**: XYZ ↔ LLH, UTM ↔ LLH, ECEF ↔ ECI conversions
- **Geocoding**: direct and inverse geocoding, monostatic and bistatic
- **Pointing**: reference frames, antenna reference frame, attitude
- **Navigation**: `CubicSplineTrajectory` for interpolated trajectory, velocity and acceleration

### Timing

High-precision time handling for SAR operations.

- `PreciseDateTime`, a picosecond precision timing
- Arithmetic operations for time differences with full precision
- Time conversion utilities

### Logging

Centralized logging built on the standard `logging` module and the Rich library.

- Custom log levels: `TRACE`, `FAIL`, `SUCCESS`
- TTY-aware console output with Rich colors and markup on terminals, plain text otherwise
- Rich tracebacks on unhandled exceptions
- Silent by default (`NullHandler`) until `initialize()` is called

## Installation

```bash
pip install perseo-core
```

You can also install via conda (conda-forge channel):

```bash
conda install -c conda-forge perseo-core
```

See the [perseo-core conda-forge page](https://anaconda.org/conda-forge/perseo-core) for details. The feedstock is hosted at [github.com/conda-forge/perseo-core-feedstock](https://github.com/conda-forge/perseo-core-feedstock).

Optional dependencies for development:

```bash
pip install perseo-core[dev,type,test,docs]
```

## License

This project is licensed under the MIT License. See the [LICENSE.txt](LICENSE.txt) file for details.

Copyright &copy; 2026-present Aresys S.r.L. <info@aresys.it>
