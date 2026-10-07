---
icon: lucide/history
title: "Changelog"
tags:
    - changelog
    - release notes
---

# Changelog

## v1.1.1

**Other Changes**

- River masking: adding a stopping criterion for mask growth
- Restrict perseo-core dependency to version below the next major release to prevent incompatible changes
- Setting Spectral Analysis variables dtypes in NetCDF output equal to `np.float64`

**Bug Fixes**

- Fixing bug in Point Target Analysis IRF graphical output when plotted objects exceed roi boundaries

## v1.1.0

**Additional Features**

- Adding cross masking algorithm for RCS computation in point target analysis

## v1.0.0

First official release.