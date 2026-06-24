---
icon: lucide/earth
tags:
    - geodynamics perturbations
    - solid tides
    - plate tectonics
    - localization
---

# Geodynamics Displacements { #geo data-toc-label="Geodynamics Displacements" }

Precise geolocation and phase interpretation in SAR and InSAR applications require accounting not only for atmospheric effects
but also for **solid Earth deformations**. These deformations arise from gravitational interactions and long-term tectonic motion,
introducing displacements that can reach centimeters to decimeters—well within the sensitivity of modern SAR systems.

!!! tip "Corner Reflectors coordinates update"

    These motions causes **Corner Reflectors** to shift over time relative to their originally surveyed positions. As a result,
    their coordinates at the time of a SAR acquisition no longer match those measured during the survey. The displacement can
    be accurately accounted for by applying the **ITRF2014 plate motion model** and **Solid Tides**, updating the reflector
    position to the SAR product acquisition time.

---

## Earth Solid Tides

**Earth Solid Tides** are periodic deformations of the Earth’s crust caused by the gravitational attraction of the **Moon**
and the **Sun**. Unlike ocean tides, these affect the solid Earth itself, producing elastic displacements of the surface.

Key characteristics:

- **Vertical displacements** up to ~30–40 cm
- **Horizontal displacements** up to ~10 cm
- Dominant periodicities:
    - Semi-diurnal (~12 hours)
    - Diurnal (~24 hours)

This package computes these effects using a Python wrapper of the original *solid.for* code by Dennis Milbert[^1], itself
based on an implementation derived from the IERS Conventions (2003). These conventions provide internationally standardized
models for Earth orientation and deformation, ensuring consistency with geodetic and geophysical reference frames.

---

## Plate Tectonics (Secular Motion)

In addition to periodic tidal effects, the Earth’s crust undergoes long-term motion due to plate tectonics. This motion is
generally linear over time and can be described using plate motion models such as **ITRF2014**.

Key characteristics:

- Typical velocities: mm/year to cm/year
- Direction and magnitude depend on tectonic plate location
- Accumulates over time, leading to measurable displacement between acquisitions

The displacement is computed as:

$$
d(t) = v \cdot (t - t_0)
$$

where $v$ is the plate velocity vector from the ITRF2014 model and $t_0$ is the location survey time.

[^1]: Dennis Milbert, Ph.D., Chief Geodesist, National Geodetic Survey, NOAA, (retired) [https://geodesyworld.github.io/SOFTS/solid.htm](https://geodesyworld.github.io/SOFTS/solid.htm)
