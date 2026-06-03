---
icon: lucide/book-open-check
title: "Tutorial"
tags:
    - trajectory
    - usage
    - tutorial
    - core
---

# Trajectory usage

The examples in this section demonstrate how to use the trajectory interface to create and evaluate trajectories. These examples use the `CubicSplineTrajectory` implementation, that is the suggested default implementation.

## Basic construction from state vectors

Creating a trajectory object from state vectors (time tagged sensor positions and velocities).

```python title="Trajectory from state vectors"
import numpy as np

from perseo_core.models.cubic_spline_trajectory import CubicSplineTrajectory
from perseo_core.timing.precise_datetime import PreciseDateTime

# discrete state vectors 
time_axis_origin = PreciseDateTime.from_utc_string("13-FEB-2023 09:33:56.000000")
time_axis = np.arange(8) * 0.5 + time_axis_origin

positions = np.array(
    [
        [-2542286.449576481, -5094859.4894666, 3901083.7183820857],
        [-2542066.5079547316, -5092594.238796566, 3904175.2612526673],
        [-2541845.6347068036, -5090327.468904538, 3907265.6186505724],
        [-2541623.8297894364, -5088059.180544005, 3910354.7896390846],
        [-2541401.0931600337, -5085789.374468956, 3913442.7732819766],
        [-2541177.424776237, -5083518.051433686, 3916529.5686432645],
        [-2540952.8245959855, -5081245.212193193, 3919615.1747874315],
        [-2540727.2925774334, -5078970.857502782, 3922699.5907792132],
    ]
)

velocities = np.array(
    [
        [439, 4529, 6184],
        [440, 4532, 6181],
        [442, 4535, 6179],
        [444, 4538, 6177],
        [446, 4544, 6174],
        [448, 4547, 6172],
        [450, 4550, 6170],
        [452, 4552, 6167],
    ]
)

# create the trajectory
trajectory = CubicSplineTrajectory(
    times=time_axis,
    positions=positions,
    velocities=velocities
)
```

## Evaluation

Once constructed, the trajectory can be evaluated at arbitrary times within its domain to retrieve position, velocity, and acceleration.

```python title="Evaluating trajectory components"
# query at a single time point
query_time = 0.75 + time_axis_origin

# these evaluations return array with shape (3,)
position = trajectory.position(query_time)
velocity = trajectory.velocity(query_time)
acceleration = trajectory.acceleration(query_time)

# query at multiple time points
query_times = np.array([0.25, 0.5, 0.75, 1.0, 1.5]) + time_axis_origin

# these evaluations return array with shape (query_times.size, 3)
positions = trajectory.position(query_times)
velocities = trajectory.velocity(query_times)
accelerations = trajectory.acceleration(query_times)
```

## Domain and Time Validation

Each trajectory has a defined time domain accessible via the `domain` property:

```python title="Checking trajectory domain"
# Get the valid time range
domain_start, domain_end = trajectory.domain
print(f"Trajectory valid from {domain_start} to {domain_end}")
```

!!! danger "Extrapolation is forbidden"

    Attempting to evaluate the trajectory outside its defined domain will raise a `RuntimeError`:
    
    ```python
    # This will raise RuntimeError - outside domain
    trajectory.position(-1.0)  # Error: before start time
    trajectory.position(3.0)   # Error: after end time
    ```

## Use Cases

The trajectory object evaluated as shown in the previous section can be used to perform a variety of tasks:

- it can be used to evaluate sensor positions and velocities at specific azimuth times for geocoding computations
- it can be directly provided to functions to perform inverse geocoding operations, ground velocity computations, doppler computations, etc.

Here are some examples:

```python title="Indirect trajectory usage"
from perseo_core.geometry.geocoding.direct_geocoding import direct_geocoding_monostatic

query_times = np.array([0.5, 1.2, 2.8]) + time_axis_origin

ground_points = direct_geocoding_monostatic(
    sensor_positions=trajectory.position(query_times),
    sensor_velocities=trajectory.velocity(query_times),
    range_times=0.0036229998783991087,
    doppler_frequencies=0.0,
    wavelength=1.0,
    look_direction="RIGHT",
    altitude=0.0,
)
```

```python title="Direct trajectory usage"
from perseo_core.geometry.geocoding.inverse_geocoding import inverse_geocoding_monostatic

ground_points = np.array(
    [
        [-2244336.5269435 , -4736678.06390127,  3622022.95230392],
        [-2244084.37158288, -4733735.65581756,  3625996.93151137],
        [-2243501.43978246, -4726996.21905426,  3635077.07787725]
    ]
)

azimuth_times, range_times = inverse_geocoding_monostatic(
    trajectory=trajectory,
    ground_points=ground_points,
    doppler_frequencies=0,
    wavelength=1,
    az_initial_time_guesses=time_axis_origin,
)

```
