---
icon: lucide/crosshair
tags:
    - point target analysis
    - analysis
    - quality
    - IRF
    - RCS
    - localization
---

# Point Target Analysis { #pta data-toc-label="Point Target Analysis" }

This analysis consist in computing an measuring quality parameters from SAR products containing information about point
targets (mainly passive corner reflectors) in the recorded scene.

**Impulse Response Function** (IRF), **Radar Cross-Section** (RCS) estimation and **Localization Errors** can be assessed from the input
product providing the location of point target in the scene with an external file.

## Computed and Estimated quantities

A full list of the output quantities that can be computed and estimated during this process can be found here, grouped by
the parent quality analysis they belong to. All these groups can be enabled or disabled during the analysis if needed
(by default they are all enabled).

<div class="grid cards" markdown>

-   :lucide-radio:{ .lg .middle } **Impulse Response Function (IRF)**

    ---

    ==**Range**==: resolution, PSLR [^1], ISLR [^2], SSLR [^3]  
    ==**Azimuth**==: resolution, PSLR, ISLR, SSLR  
    ==**2D**==: PSLR, ISLR, SSLR

-   :lucide-radar:{ .lg .middle } **Radar Cross-Section (RCS)**

    ---

    ==**2D**==: RCS, RCS Errors, Peak Phase Error, Clutter, SCR [^4]

-   :lucide-map-pin:{ .lg .middle } **Localization Errors**

    ---

    ==**Range**==: slant localization error, ground localization error  
    ==**Azimuth**==: localization error

</div>

[^1]: Peak-to-Side-Lobe-Ratio

[^2]: Integral-Side-Lobe-Ratio

[^3]: Secondary-Side-Lobe-Ratio

[^4]: Signal-to-Clutter-Ratio
