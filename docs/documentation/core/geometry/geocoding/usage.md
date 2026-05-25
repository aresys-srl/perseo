---
icon: lucide/book-open-check
title: "Tutorial"
tags:
    - geocoding
    - usage
    - tutorial
    - core
---

# Geocoding usage

The ``perseo_core.geometry.geocoding`` module provides several direct and inverse geocoding capabilities for SAR
geometry computations.

## Direct Geocoding

The main implemented functions are:

- [`direct_geocoding_monostatic`][perseo_core.geometry.geocoding.direct_geocoding.direct_geocoding_monostatic]
- [`direct_geocoding_bistatic`][perseo_core.geometry.geocoding.direct_geocoding.direct_geocoding_bistatic]
- [`direct_geocoding_with_look_angles`][perseo_core.geometry.geocoding.direct_geocoding.direct_geocoding_with_look_angles]
- [`direct_geocoding_with_looking_direction`][perseo_core.geometry.geocoding.direct_geocoding.direct_geocoding_with_looking_direction]
- [`direct_geocoding_with_pointing`][perseo_core.geometry.geocoding.direct_geocoding.direct_geocoding_with_pointing]

All functions support vectorized operations on `(N, 3)` coordinate arrays.

Here are some examples on how to use the direct geocoding functions:

### Direct geocoding monostatic

```python title=""
import numpy as np
from perseo_core.geometry.geocoding.direct_geocoding import direct_geocoding_monostatic

sensor_position = np.array(
    [4387348.749948771, 762123.3489877012, 4553067.931912004],
)
sensor_velocity = np.array(
    [-856.1384108174528, -329.7629775067583, 398.55830806407346],
)

ground_points = direct_geocoding_monostatic(
    sensor_positions=sensor_position,
    sensor_velocities=sensor_velocity,
    range_times=2.05624579e-05,
    doppler_frequencies=0.0,
    altitude=0.0,
    look_direction="RIGHT",
    wavelength=1.0,
)
```

### Direct geocoding bistatic

```python title=""
import numpy as np
from perseo_core.geometry.geocoding.direct_geocoding import direct_geocoding_bistatic

sensor_position = np.array(
    [4387348.749948771, 762123.3489877012, 4553067.931912004],
)
sensor_velocity = np.array(
    [-856.1384108174528, -329.7629775067583, 398.55830806407346],
)

ground_points = direct_geocoding_bistatic(
    sensor_positions_rx=sensor_position,
    sensor_positions_tx=sensor_position,
    sensor_velocities_rx=sensor_velocity,
    sensor_velocities_tx=sensor_velocity,
    range_times=2.05624579e-05,
    doppler_frequencies=0.0,
    altitude=0.0,
    look_direction="RIGHT",
    wavelength=1.0,
)
```

### Direct geocoding with looking direction

```python title=""
import numpy as np
from perseo_core.geometry.geocoding.direct_geocoding import direct_geocoding_with_looking_direction

sensor_positions = np.array(
    [
        [5317606.94350283, 610603.985945038, 4577936.89859885],
        [5313024.53547427, 608285.563877273, 4583547.15708167],
        [5308435.7651548, 605967.120830312, 4589152.18047604],
    ]
)
looking_direction = np.array([5317606.0, 610603.0, 4577936.0])

ground_points = direct_geocoding_with_looking_direction(
    sensor_positions=sensor_positions,
    looking_directions=looking_direction,
    altitude=0.0,
)
```

## Inverse Geocoding

The main implemented functions are:

- [`inverse_geocoding_monostatic`][perseo_core.geometry.geocoding.inverse_geocoding.inverse_geocoding_monostatic]
- [`inverse_geocoding_bistatic`][perseo_core.geometry.geocoding.inverse_geocoding.inverse_geocoding_bistatic]

All functions support vectorized operations on `(N, 3)` coordinate arrays.

Here are some examples on how to use the inverse geocoding functions:

### Inverse geocoding monostatic

```python title=""
import numpy as np
from perseo_core.geometry.geocoding.inverse_geocoding import inverse_geocoding_monostatic

trajectory = ... # (1)!

ground_point = np.array(
    [-2243618.48435212, -4728341.28615007, 3633267.229522297],
)

azimuth_times, range_times = inverse_geocoding_monostatic(
    trajectory=trajectory,
    ground_points=ground_point,
    frequencies_doppler_centroid=0.0,
    wavelength=1.0,
    az_initial_time_guesses=trajectory.domain[0],
)
```

1. Refer to the [trajectory tutorial](../trajectory/usage.md) for further information on how to create a trajectory object.

### Inverse geocoding bistatic

```python title=""
import numpy as np
from perseo_core.geometry.geocoding.inverse_geocoding import inverse_geocoding_bistatic

trajectory_rx = ... # (1)!
trajectory_tx = ...

ground_point = np.array(
    [-2243618.48435212, -4728341.28615007, 3633267.229522297],
)

azimuth_times, range_times = inverse_geocoding_bistatic(
    trajectory_rx=trajectory_rx,
    trajectory_tx=trajectory_tx,
    ground_points=ground_point,
    frequencies_doppler_centroid=0.0,
    wavelength=1.0,
    az_initial_time_guesses=trajectory.domain[0] + 0.5,
)
```

1. Refer to the [trajectory tutorial](../trajectory/usage.md) for further information on how to create a trajectory object.
