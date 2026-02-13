.. _core_attitude:

Attitude and Sensor Rotations
=============================

Overview
--------

The ``perseo_core.models.attitude`` module provides utilities to represent and interpolate the **attitude** (orientation)
of a sensor over time.

It is built on top of :class:`scipy.spatial.transform.Rotation` and uses SciPy's implementation of
**Spherical Linear Interpolation (SLERP)** :class:`scipy.spatial.transform.Slerp` to generate smooth rotational trajectories.
This ensures numerically stable and geometrically correct interpolation on the rotation manifold (SO(3)).

The main entry point is the :class:`Attitude <perseo_core.models.attitude.Attitude>` class, which allows you to:

- Construct an attitude timeline from quaternions or Euler angles (yaw, pitch, roll)
- Interpolate rotations at arbitrary query times using Spherical Linear Interpolation (SLERP)
- Evaluate first derivatives of the interpolated rotations

The implementation relies internally on SciPy rotations, so all conventions (quaternion ordering (scalar-last),
Euler sequences, etc.) follow SciPy's standards.

Rotations and Reference Systems
-------------------------------

Definition Frame
~~~~~~~~~~~~~~~~

An :class:`Attitude <perseo_core.models.attitude.Attitude>` instance is always defined with respect to a **reference system**.

The reference system is implicitly the one in which the input quaternions or Euler angles are expressed. For example:

- If Euler angles describe orientation with respect to a local navigation frame, that navigation frame becomes the
  reference system of the attitude.
- If Quaternions are provided in a Global Reference frame (i.e. ECEF), then this is the reference frame of the attitude.

.. note::
    **SLERP interpolation does not change the reference system**. The rotations are interpolated in the same frame in
    which they were defined. Only the rotation values evolve in time — the underlying coordinate frame remains
    unchanged.

Attitude Representation
-----------------------

An :class:`Attitude <perseo_core.models.attitude.Attitude>` object represents a time-parameterized sequence of sensor rotations.

You typically:

1. Define known rotation samples at discrete times.
2. Create an :class:`Attitude <perseo_core.models.attitude.Attitude>` instance from those samples.
3. Evaluate the attitude at arbitrary query times using SLERP.

Construction Methods
--------------------

from_quaternions
~~~~~~~~~~~~~~~~

An ``Attitude.from_quaternions()`` method is provided to construct an :class:`Attitude <perseo_core.models.attitude.Attitude>`
object from time-tagged quaternions.

Example:

.. code-block:: python

    import numpy as np
    from perseo_core.models.attitude import Attitude

    times = np.array([0.0, 1.0, 2.0])

    # Identity, 90° about Z, 180° about Z
    quaternions = np.array([
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.70710678, 0.70710678],
        [0.0, 0.0, 1.0, 0.0],
    ])

    attitude = Attitude.from_quaternions(quaternions=quaternions, times=times)

from_euler_angles
~~~~~~~~~~~~~~~~~

An ``Attitude.from_euler_angles()`` method is provided to construct an :class:`Attitude <perseo_core.models.attitude.Attitude>`
object from time-tagged quaternions.

Example:

.. code-block:: python

    import numpy as np
    from perseo_core.models.attitude import Attitude

    times = np.array([0.0, 2.0, 4.0])

    # Yaw, Pitch, Roll rotations in radians
    angles_rad = np.deg2rad(np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 45.0],
        [0.0, 45.0 90.0],
    ]))

    attitude = Attitude.from_euler_angles(euler_angles_rad=angles_rad, rotation_order="YPR", times=times)

Interpolation with SLERP
------------------------

Once constructed, the :class:`Attitude <perseo_core.models.attitude.Attitude>` object internally creates a SciPy ``Slerp`` interpolator.
You can evaluate the rotation at arbitrary query times within the defined time interval. This returns a :class:`scipy.spatial.transform.Rotation`
object corresponding to the requested interpolated rotations.

The first derivative of the SLERP interpolation provides the instantaneous rotational rate associated with the
interpolated attitude.

Example:

.. code-block:: python

    query_times = np.linspace(0.0, 2.0, 5)

    interpolated_rotations = attitude.evaluate(query_times)
    rotations_derivative = attitude.evaluate_first_derivatives(query_times)

    # Convert to Euler angles for inspection (yaw, pitch, roll)
    euler = interpolated_rotations.as_euler("ZYX", degrees=True)

Changing the Reference System
-----------------------------

**PERSEO Core** provides utilities to express an existing rotation in a new reference system. This means that the user can
define rotations in a given reference frame, may it be a Local Reference Frame (i.e. Zero Doppler) or a Global Reference Frame (i.e. ECEF),
and then re-express the interpolated rotations in a different reference system.

Functionalities for this are provided by the module :mod:`perseo_core.geometry.utilities.rotations`.
Please refer to :ref:`the dedicated documentation on Reference Frames <core_reference_frames_ch>` for further details.

Use Cases
---------

ECEF Quaternions
~~~~~~~~~~~~~~~~

An example of common use case is the following:

- Attitude information is provided in a Global Reference Frame (i.e. ECEF) via quaternions
- The user wants to compute Euler angles in a Local Reference Frame (i.e. Zero Doppler)

.. code-block:: python

    import numpy as np
    from perseo_core.models.attitude import Attitude
    from perseo_core.models.orbit import Orbit
    from perseo_core.geometry.utilities.reference_frames import compute_sensor_local_axis
    from perseo_core.geometry.utilities.rotations import rotation_to_euler_angles

    # state vectors in a Global Reference System (i.e. ECEF)
    times = np.array([0.0, 1.0, 2.0])
    positions = np.array([
        [4454010.19620684, 773703.063195028, 4486349.18475909],
        [4453313.69416755, 773582.074302417, 4487056.65289135],
        [4452617.07996439, 773461.065925896, 4487764.01275803],
    ])
    velocities = np.array([
        [-139.30040786,  -24.19777852,  141.49362645],
        [-140.40959374,  -24.39045444,  140.39797284],
        [-141.49296109,  -24.57864544,  139.27619603],
    ])
    orbit = Orbit(times=times, positions=positions, velocities=velocities)

    # quaternions in a Global Reference System (i.e. ECEF)
    quaternions = np.array([
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.70710678, 0.70710678],
        [0.0, 0.0, 1.0, 0.0],
    ])
    # attitude is therefore expressed in the Global Reference System (i.e. ECEF)
    attitude = Attitude.from_quaternions(quaternions=quaternions, times=times)

    # interpolating times
    query_times = np.array([0.5, 0.85, 1.2, 1.8])
    # get interpolated rotations in the same reference system as the attitude's
    interpolated_rotations = attitude.evaluate(query_times)

    # this is the Zero Doppler reference frame expressed in the Global Reference System (i.e. ECEF)
    zero_doppler_local_axis = compute_sensor_local_axis(
        sensor_positions=orbit.evaluate(times),
        sensor_velocities=orbit.evaluate_first_derivatives(times),
        reference_frame="ZERODOPPLER"
    )

    # convert rotations to Euler angles in a Local Reference Frame (i.e. Zero Doppler)
    interpolated_rotations_local = zero_doppler_local_axis * interpolated_rotations
    # NOTE: euler angles columns order is the same as the rotation order requested
    euler_angles_local = rotation_to_euler_angles(
        rotation=interpolated_rotations_local,
        rotation_order="YPR",
    )


Sensor Euler Angles
~~~~~~~~~~~~~~~~~~~

Another example of common use case is the following:

- Attitude information is provided in a Local Reference Frame (i.e. Zero Doppler) via Euler angles (Yaw, Pitch, Roll)
- The user wants to compute the Antenna Reference Frame (ARF) in Global Reference Frame (i.e. ECEF) from the attitude information

.. code-block:: python

    import numpy as np
    from perseo_core.models.attitude import Attitude
    from perseo_core.models.orbit import Orbit
    from perseo_core.geometry.utilities.reference_frames import compute_sensor_local_axis

    # state vectors in a Local Reference System (i.e. Zero Doppler)
    times = np.array([0.0, 1.0, 2.0])
    positions = np.array([
        [4454010.19620684, 773703.063195028, 4486349.18475909],
        [4453313.69416755, 773582.074302417, 4487056.65289135],
        [4452617.07996439, 773461.065925896, 4487764.01275803],
    ])
    velocities = np.array([
        [-139.30040786,  -24.19777852,  141.49362645],
        [-140.40959374,  -24.39045444,  140.39797284],
        [-141.49296109,  -24.57864544,  139.27619603],
    ])
    orbit = Orbit(times=times, positions=positions, velocities=velocities)

    # Euler Angles in a Local Reference System (i.e. Zero Doppler)
    angles_rad = np.deg2rad(np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 45.0],
        [0.0, 45.0, 90.0],
    ]))
    # attitude is therefore expressed in the Local Reference System (i.e. Zero Doppler)
    attitude = Attitude.from_euler_angles(euler_angles_rad=angles_rad, rotation_order="YPR", times=times)

    # interpolating times
    query_times = np.array([0.5, 0.85, 1.2, 1.8])
    # get interpolated rotations in the same reference system as the attitude's
    interpolated_rotations = attitude.evaluate(query_times)

    # this is the Zero Doppler reference frame expressed in the Global Reference System (i.e. ECEF)
    zero_doppler_local_axis = compute_sensor_local_axis(
        sensor_positions=orbit.evaluate(times),
        sensor_velocities=orbit.evaluate_first_derivatives(times),
        reference_frame="ZERODOPPLER"
    )
    # compute the ARF in the Global Reference System (i.e. ECEF)
    antenna_reference_frame = zero_doppler_local_axis * interpolated_rotations

Another example starting from Euler angles:

.. code-block:: python

    import numpy as np

    # Euler Angles in a Local Reference System (i.e. Zero Doppler)
    angles_rad = np.deg2rad(np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 45.0],
        [0.0, 45.0, 90.0],
    ]))

    # this is the Zero Doppler reference frame expressed in the Global Reference System (i.e. ECEF)
    zero_doppler_local_axis = compute_sensor_local_axis(
        sensor_positions=orbit.evaluate(times),
        sensor_velocities=orbit.evaluate_first_derivatives(times),
        reference_frame="ZERODOPPLER"
    )
    antenna_reference_frame = zero_doppler_local_axis * euler_angles_to_rotation(angles_rad, rotation_order="YPR")

    # attitude is therefore expressed in the Local Reference System (i.e. Zero Doppler)
    attitude = Attitude(rotations=antenna_reference_frame, times=times)
