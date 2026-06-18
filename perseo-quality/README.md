# perseo-quality

[![PyPI version](https://img.shields.io/pypi/v/perseo-quality)](https://pypi.org/project/perseo-quality/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)

**P**ython **E**cosystem for **R**emote **S**ensing & **E**arth **O**bservation (PERSEO) QUALITY package for SAR products
quality analysis and calibration assessment.

## Features

### Point Target Analysis

Analysis of point targets (passive corner reflectors or transponders) in SAR scenes.

- **Impulse Response Function (IRF)**: range and azimuth resolution, PSLR, ISLR, SSLR (1D and 2D)
- **Radar Cross-Section (RCS)**: RCS estimation, RCS errors, peak phase error, clutter, SCR
- **Localization Errors**: slant range and ground localization error, azimuth localization error

### Radiometric Analysis

Global quality assessment on homogeneous distributed targets.

- **Block-wise**: automatic scene partitioning into azimuth blocks (bursts for TopSAR/ScanSAR)
  - Noise Equivalent Sigma-Zero (NESZ) profiles
  - Average Elevation Profiles
  - Scalloping Profiles
  - KPI estimation
  - Configurable block size, range margin, outlier removal and smoothing
- **Point-wise**: profiles extracted around a selected location

### Interferometric Analysis

Coherence analysis from interferometric SAR products.

- 2D coherence intensity histograms along range and azimuth
- Burst-by-burst processing with configurable partitioning
- Graphical representation

### Spectral Analysis

Spectral content investigation in the frequency domain.

- **Point Target**: absolute and phase spectra at each target location
- **Distributed Target**: spectral amplitude on bursts or azimuth blocks
- Range and azimuth profiles at each third of the data portion

### Elevation Notch Analysis

Antenna pointing calibration from dedicated Elevation Notch (EN) products.

- Mis-pointing angle estimation by matching measured range profiles with theoretical EAP
- Robust estimation using EN patterns with central low-power "hole"

### Target Ambiguity Ratio (PTAR / DTAR)

Signal-to-ambiguity ratio computation.

- **PTAR**: Point Target Ambiguity Ratio
- **DTAR**: Distributed Target Ambiguity Ratio
- Left and right ambiguity localization

### Data Model

All analyses are designed around a generic Python protocol, making them format-agnostic and independent of the
input product type. The protocol and its utilities are available in the `perseo_quality.io` module.

## Installation

```bash
pip install perseo-quality[graphs]
```

The `[graphs]` extra enables graphical output (`matplotlib`). For development:

```bash
pip install perseo-quality[dev,test,docs]
```

## License

This project is licensed under the MIT License. See the [LICENSE.txt](LICENSE.txt) file for details.

Copyright &copy; 2026-present Aresys S.r.L. <info@aresys.it>
