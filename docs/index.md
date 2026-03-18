---
icon: lucide/sparkles
title: "PERSEO"
tags:
    - perseo
---

# Python Ecosystem for Remote Sensing & Earth Observation

**P**ython **E**cosystem for **R**emote **S**ensing & **E**arth **O**bservation (**PERSEO**) is the Aresys is a modular
Python framework designed to simplify and standardize the handling, processing, and analysis of Synthetic Aperture Radar
(SAR) products and their auxiliary data.

It is distributed as a collection of interoperable packages available on [PyPI](https://pypi.org/user/aresys/):

<div class="result" markdown>

:lucide-atom:{ .lg .middle } [__PERSEO Core__](documentation/core/index.md#core){ data-preview }

:lucide-badge-check:{ .lg .middle } [__PERSEO Quality__](documentation/quality/index.md#quality){ data-preview }

:lucide-cloud-sun-rain:{ .lg .middle } [__PERSEO Perturbations__](documentation/perturbations/index.md#perturb){ data-preview }

</div>

Together, these packages provide a cohesive ecosystem for working with SAR data, enabling users to build scalable,
maintainable, and reproducible processing pipelines.

## Design Principles

PERSEO is built around a few key principles:

- **Modularity** – Each package focuses on a specific domain, allowing users to adopt only what they need.
- **Interoperability** – Packages are designed to work seamlessly together.
- **Extensibility** – The architecture supports the addition of new processing components and data models.
- **Reproducibility** – Standardized workflows ensure consistent and traceable results.
- **Openness** – The framework is open-source and community-driven, fostering collaboration and innovation.
- **State-of-the-art** – PERSEO is built on the latest technologies and best practices, ensuring optimal performance and
reliability.

## What Perseo Enables

By combining its modules, Perseo allows users to:

- Manage SAR products rasters and metadata exploiting abstractions and common data structures
- Integrate support for reading and using auxiliary datasets such as atmospheric maps  
- Apply corrections and perturbation models consistently  
- Perform quality assessment and validation

## Target Users

Perseo is designed for:

- Remote sensing scientists  
- Geospatial data engineers  
- SAR processing specialists  
- Researchers working with Earth observation data  

## Getting Started

PERSEO python packages can be installed using ``pip`` in any python environment satisfying the requirements.

> :lucide-circle-chevron-right: Refer to the [installation documentation](install.md) for further information.
