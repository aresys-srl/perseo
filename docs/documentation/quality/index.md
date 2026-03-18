---
icon: lucide/badge-check
title: "Quality"
tags:
    - quality
---

# QUALITY { #quality data-toc-label="QUALITY" }

``Quality`` is part of the `PERSEO` Aresys python project, and it's aimed at providing fundamental functionalities
for analyzing SAR data and quantitatively and qualitatively assess their quality and calibration features.

<figure markdown="span">
    ![Supporting Projects](../../assets/images/q/quality_intro.png){ width="850" }
    <figcaption>Example of quality graphical output obtained from Point Target Analysis.</figcaption>
</figure>

The following analyses have been implemented:

<div class="result" markdown>

:lucide-crosshair:{ .lg .middle } [__Point Target Analysis__](analyses/point-target/index.md#pta){ data-preview } : Impulse Response Function (IRF), Radar Cross Section (RCS) and Localization Errors

:lucide-signal:{ .lg .middle } [__Radiometric Analysis__](analyses/radiometry/index.md#rad){ data-preview } : Noise Equivalent Sigma-Zero (NESZ), Average Elevation Profiles, Scalloping Profiles and custom radiometric profiles

:lucide-git-branch:{ .lg .middle } [__Interferometric Analysis__](analyses/interferometry/index.md#interf){ data-preview } : interferometric coherence analysis and graphical representation

:lucide-ghost:{ .lg .middle } [__Spectral Analysis__](analyses/spectra/index.md#spectra){ data-preview } : absolute and phase spectral analysis for Point Targets and Distributed Targets

:lucide-satellite-dish:{ .lg .middle } [__Elevation Notch Analysis__](analyses/notch/index.md#notch){ data-preview } : antenna pointing estimation from dedicated elevation notch products

:lucide-ratio:{ .lg .middle } [__Target Ambiguity Ratio (PTAR/DTAR)__](analyses/ambiguity-ratio/index.md#tar){ data-preview } : ambiguity ratio analysis both for Point Targets and Distributed Targets

:lucide-eye:{ .lg .middle } [__Equivalent Number of Looks (ENL)__](analyses/enl/index.md#enl){ data-preview } : equivalent number of looks analysis

</div>
