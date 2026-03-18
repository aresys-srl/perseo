---
icon: lucide/signal
tags:
    - radiometric analysis
    - analysis
    - nesz
    - rain forest
    - radiometric profiles
    - scalloping
---

# Radiometric Analysis { #rad data-toc-label="Radiometric Analysis" }

Radiometric analysis can be used to assess and verify product quality globally on images with homogeneous distributed targets.

It's used to extract **Radiometric Profiles** from the input SAR image corresponding to azimuth blocks obtained by
partitioning the whole scene.

## Block-wise Radiometric Analysis

Using this algorithm the whole scene is automatically analyzed after being subdivided in azimuth blocks spanning a given
amount of azimuth lines in a direction and the whole slant range axis in the other (except near and far range values removed using a margin).
In case of TopSAR/ScanSAR acquisitions, **bursts** are used as default azimuth blocks.

!!! tip "Parameters Configuration"

    Blocks dimension can be changed using the configuration file:

    ==**azimuth_block_size**==: size of block in terms of azimuth lines (pixels) for acquisitions without bursts  
    ==**range_pixel_margin**==: near/far range margin removed from axis

Once the image is partitioned, each block is analyzed and a radiometric profiles is extracted. Pre-implemented outlier removal
algorithms and smoothening filters can be applied to the block values if needed using the `radiometric_profiles.profile_parameters`
configuration sections.

Image can be provided in any radiometric quantity provided that it's specified in the configuration file if different from
the default beta-nought. Output quantity can also be selected as an argument of the ``radiometric_profiles`` function.

Output data, both numeric and graphical, can be used to verify and check data properties such as: expected profiles trends
with respect to incidence angle, noise magnitude and azimuth trends.

For profiles extracted along the *RANGE* direction, the incidence angle axis is computed using direct geocoding, while for
*AZIMUTH* profiles a relative time axis in seconds is returned.

Few pre-configured radiometric profiles have already been developed and implemented in this module, namely:

- **Noise Equivalent Sigma-Zero (NESZ) Profiles**
- **Average Elevation Profiles**
- **Scalloping Profiles**

These are all wrapper on the same generic radiometric profile core function with arguments and key parameters pre-set to perform
that specific task.

## Point-wise Radiometric Analysis

Radiometric analysis can be used to assess and verify product quality globally on images with homogeneous distributed targets.

It's used to extract **Radiometric Profiles** from the input SAR image corresponding to an interval of lines/samples around
the selected location.
