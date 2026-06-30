# PERSEO

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://python.org)
[![Core CI](https://github.com/aresys-srl/perseo/actions/workflows/perseo_core.yml/badge.svg)](https://github.com/aresys-srl/perseo/actions/workflows/perseo_core.yml)
[![Quality CI](https://github.com/aresys-srl/perseo/actions/workflows/perseo_quality.yml/badge.svg)](https://github.com/aresys-srl/perseo/actions/workflows/perseo_quality.yml)
[![Perturbations CI](https://github.com/aresys-srl/perseo/actions/workflows/perseo_perturbations.yml/badge.svg)](https://github.com/aresys-srl/perseo/actions/workflows/perseo_perturbations.yml)

**P**ython **E**cosystem for **R**emote **S**ensing & **E**arth **O**bservation is the new Aresys Python project
dedicated to SAR products analysis and core functionalities for performing fundamental operations on data.
 
This project is a monorepo consisting of several standalone packages that can be installed separately to perform
thematic operations as indicated by the package name.

## Available Packages

| Package | PyPI | Conda | Description |
|---------|------|-------|-------------|
| **perseo-core** | [![PyPI version](https://img.shields.io/pypi/v/perseo-core)](https://pypi.org/project/perseo-core/) | [![Conda](https://img.shields.io/conda/vn/conda-forge/perseo-core)](https://anaconda.org/conda-forge/perseo-core) | Generic and broadly used operations: geometry, timing, logging |
| **perseo-perturbations** | [![PyPI version](https://img.shields.io/pypi/v/perseo-perturbations)](https://pypi.org/project/perseo-perturbations/) | — | Atmospheric delays, geodynamics displacements, geomagnetic field |
| **perseo-quality** | [![PyPI version](https://img.shields.io/pypi/v/perseo-quality)](https://pypi.org/project/perseo-quality/) | — | SAR products quality analysis and calibration assessment |

## Installation

Install all packages via pip:

```bash
pip install perseo-core
pip install perseo-perturbations
pip install perseo-quality[graphs]
```

`perseo-core` is also available on conda-forge:

```bash
conda install -c conda-forge perseo-core
```

See the [perseo-core conda-forge page](https://anaconda.org/conda-forge/perseo-core) for details. The feedstock is hosted at [github.com/conda-forge/perseo-core-feedstock](https://github.com/conda-forge/perseo-core-feedstock).

## Documentation

Full documentation is available at [https://opensource.aresys.it/perseo](https://opensource.aresys.it/perseo).

## Contributing

Contributions are welcome! If you encounter a bug, have a feature request, or want to contribute code:

- **Report bugs & request features**: open an issue on [GitHub](https://github.com/aresys-srl/perseo/issues). Include a clear description, steps to reproduce, and your environment details.
- **Submit changes**: fork the repository, create a feature branch, and open a pull request. Ensure your code passes the existing linting and test suite.
- **Questions**: use GitHub Discussions for general questions and discussions.

## License

This project is licensed under the MIT License.

Copyright &copy; 2026-present Aresys S.r.L. <info@aresys.it>
