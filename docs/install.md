---
icon: lucide/arrow-big-down-dash
title: "Install"
tags:
    - python
    - pip
    - install
---

# Installation

!!! note "Requirements"

    PERSEO projects requires a Python version **equal or higher than 3.11**.

Each python package of the PERSEO framework can be installed using ``pip``:

!!! danger "Name Collision Warning"

    There are other Python packages available on PyPI with names similar to **PERSEO** that are **not related to this project**.  
    To avoid confusion or dependency issues, make sure you install and use only the official packages that are part of this framework.  
    Any other packages with similar names are not maintained as part of the PERSEO ecosystem and may be incompatible.


## Core

```bash title="install with pip"
pip install perseo-core
```

## Quality

```bash title="install with pip"
pip install perseo-quality
```

Optional dependencies can be installed to generate graphical output:

```bash title="install with pip"
pip install perseo-quality[graphs]
```

## Perturbations

```bash title="install with pip"
pip install perseo-perturbations
```

!!! tip "Virtual Environments"

    We recommend using a dedicated virtual environment to install these packages.  
    This will ensure that the software is installed in a separate environment and avoids conflicts with other packages
    or dependencies.
