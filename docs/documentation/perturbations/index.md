---
icon: lucide/cloud-sun-rain
title: "Core"
tags:
    - perturbations
    - atmosphere
    - ionosphere
    - troposphere
    - geodynamics
    - solid tides
    - plate tectonics
---

# PERTURBATIONS { #perturb data-toc-label="PERTURBATIONS" }

``Perturbations`` is part of the `PERSEO` Aresys python project, and it's aimed at computing **geophysics**
displacements (Plate Tectonics, Solid Earth Tides) and **atmospheric** (Ionospheric, Tropospheric) signal delays.

<div class="result" markdown>

:lucide-cloud-download:{ .lg .middle } [__Atmospheric Delays__](atmospheric/index.md#atm){ data-preview } : Ionospheric and Tropospheric delays

:lucide-earth:{ .lg .middle } [__Geodynamics Displacements__](geodynamics/index.md#geo){ data-preview } : Solid Earth Tides and Plate Tectonics

</div>

## Atmospheric Perturbations

This module relies on resources files that have been downloaded and attached to this project in order to properly perform
atmospheric computations. In particular, **tropospheric VMF3** model **legendre coefficients** and **stations grid points coordinates**
(both for 1x1 and 5x5 grids) have been added as separate files in the resources module and actively use in the code.

These files can be found here:

==**VMF3 Legendre coefficients**==: [https://vmf.geo.tuwien.ac.at/codes/vmf3.m](https://vmf.geo.tuwien.ac.at/codes/vmf3.m)

==**Station coordinates grid 1x1**==: [https://vmf.geo.tuwien.ac.at/station_coord_files/gridpoint_coord_1x1.txt](https://vmf.geo.tuwien.ac.at/station_coord_files/gridpoint_coord_1x1.txt)

==**Station coordinates grid 5x5**==: [https://vmf.geo.tuwien.ac.at/station_coord_files/gridpoint_coord_5x5.txt](https://vmf.geo.tuwien.ac.at/station_coord_files/gridpoint_coord_5x5.txt)

## Geodynamics Perturbations

To estimate **Earth Solid Tides**, this package relies on a python wrapper implementation of the *solid.for* Fortran code by
**Dennis Milbert**[^1].

Program Solid is based on an edited version of the **dehanttideinelMJD.f** source code provided by Professor V. Dehant.
This code is an implementation of the Solid Earth Tide computation found in section 7.1.2 of the IERS Conventions (2003)[^2],
IERS Technical Note No. 32.

**Plate Tectonics** displacement correction is based on the ITRF2014 plate motion model[^3].

[^1]: Dennis Milbert, [https://geodesyworld.github.io/SOFTS/solid.htm](https://geodesyworld.github.io/SOFTS/solid.htm)
[^2]: [IERS Technical Note No. 32, IERS Conventions (2003)](https://www.iers.org/IERS/EN/Publications/TechnicalNotes/tn32.html).
[^3]: Zuheir Altamimi et al., "ITRF2014 plate motion model", Geophysical Journal International, 2017, [https://doi.org/10.1093/gji/ggx136](https://doi.org/10.1093/gji/ggx136)
