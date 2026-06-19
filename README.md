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

| Package | PyPI | Description | Installation |
|---------|------|-------------|-------------|
| **perseo-core** | [![PyPI version](https://img.shields.io/pypi/v/perseo-core)](https://pypi.org/project/perseo-core/) | Generic and broadly used operations: geometry, timing, logging | `pip install perseo-core` |
| **perseo-perturbations** | [![PyPI version](https://img.shields.io/pypi/v/perseo-perturbations)](https://pypi.org/project/perseo-perturbations/) | Atmospheric delays, geodynamics displacements, geomagnetic field | `pip install perseo-perturbations` |
| **perseo-quality** | [![PyPI version](https://img.shields.io/pypi/v/perseo-quality)](https://pypi.org/project/perseo-quality/) | SAR products quality analysis and calibration assessment | `pip install perseo-quality[graphs]` |

## Documentation

Full documentation is available at [https://aresys-srl.github.io/perseo](https://aresys-srl.github.io/perseo).

## Contributing

Contributions are welcome! If you encounter a bug, have a feature request, or want to contribute code:

- **Report bugs & request features**: open an issue on [GitHub](https://github.com/aresys-srl/perseo/issues). Include a clear description, steps to reproduce, and your environment details.
- **Submit changes**: fork the repository, create a feature branch, and open a pull request. Ensure your code passes the existing linting and test suite.
- **Questions**: use GitHub Discussions for general questions and discussions.

## License

This project is licensed under the MIT License.

Copyright &copy; 2026-present Aresys S.r.L. <info@aresys.it>
