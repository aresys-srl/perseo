---
icon: lucide/cooking-pot
title: "Algorithm"
tags:
    - equivalent number of looks analysis
    - analysis
    - enl
---

# Algorithm description

The algorithm used to determine the equivalent number of looks is pretty simple and it should be used only on isolated
homogeneous salt and pepper like portions of the raster data:

$$
ENL = \frac{I_{mean}^2}{I_{std}^2}
$$
