---
icon: lucide/git-branch
tags:
    - interferometric analysis
    - analysis
    - quality
    - coherence
---

# Interferometric Analysis { #interf data-toc-label="Interferometric Analysis" }

This analysis consist in computing 2D coherence intensity histograms along range and azimuth directions from SAR products
containing interferogram information or pre-computed coherence values.

## Computed and Estimated quantities

Coherence is computed as the ratio between an interferogram's complex data values and their magnitude after applying a Boxcar
filter with a specific kernel to the whole image. Image is processed burst by burst so to keep the information as relevant as
possible.

The results of this computation are two 2D coherence intensity histograms along both SAR dimensions computed after partitioning
each burst data for a configurable number of times along each direction.
