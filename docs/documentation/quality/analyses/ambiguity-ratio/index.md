---
icon: lucide/ratio
tags:
    - equivalent number of looks analysis
    - analysis
    - enl
---

# Target Ambiguity Ratio Analysis { #tar data-toc-label="Target Ambiguity Ratio Analysis" }

Target Ambiguity Ratio (TAR) analyses can be performed on Point Targets (**PTAR**) or Distributed Targets (**DTAR**) to
compute the ratio between the signal and its ambiguities, if captured by the SAR image.

## Ambiguities Location

Left and Right ambiguities for a given target are located at well defined azimuth and range time deltas.
The azimuth distance is given by the absolute ratio between the sensor PRF and the Doppler Rate at the target location.
The range distance is instead derived from the Line of Sight variation between the target and the ambiguity location.
