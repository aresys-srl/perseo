.. _core_attitude:

Attitude
========

Overview
--------

The ``perseo_core.geometry.pointing.attitude`` module provides utilities to handle the different reference frames
involved in the definition of the attitude of a sensor, and to perform interpolation of these frames over time.

It is built on top of :class:`scipy.spatial.transform.Rotation` and uses SciPy's implementation of
**Spherical Linear Interpolation (SLERP)** :class:`scipy.spatial.transform.Slerp` to generate smooth rotational trajectories.
This ensures numerically stable and geometrically correct interpolation on the rotation manifold (SO(3)).

The main object is the :class:`Attitude <perseo_core.geometry.pointing.attitude.Attitude>` class which allows you to:

- Construct an attitude timeline from reference frames, quaternions or Euler angles (yaw, pitch, roll)
- Interpolate reference frames at arbitrary query times using Spherical Linear Interpolation (SLERP)

Rotations and Reference Frames
------------------------------

Definition Frame
~~~~~~~~~~~~~~~~

An :class:`Attitude <perseo_core.geometry.pointing.attitude.Attitude>` instance is always defined with respect to a **reference system**.

The reference system is implicitly the one in which the input reference frames, quaternions, or Euler angles are expressed. For example:

- If reference frames matrices are provided in a global reference frame (i.e. ECEF), then the attitude is defined in that global reference frame.
- If Euler angles describe orientation with respect to a local sensor frame (e.g. the zero doppler reference frame), then 
  the attitude is defined in that local reference frame.
- If quaternions are provided in a global reference frame (i.e. ECEF), then the attitude is defined in that global reference frame.

.. note::
    **SLERP interpolation does not change the reference system**. The rotations are interpolated in the same frame in
    which they were defined.

.. note::
    A typical scenario is that the attitude is defined via yaw, pitch, roll angles (Euler angles) with respect to
    a local reference frame (e.g. zero doppler), but then the user needs to express the attitude in a global reference frame (i.e. ECEF).
    For this use case, the :func:`compute_antenna_attitude_from_euler_angles
    <perseo_core.geometry.pointing.attitude.compute_antenna_attitude_from_euler_angles>` function is provided to directly compute the 
    overall antenna attitude in a global reference frame (i.e. ECEF) starting from Euler angles defined in a local reference frame (e.g. zero doppler).

Attitude Representation
-----------------------

An :class:`Attitude <perseo_core.geometry.pointing.attitude.Attitude>` object represents a time-parameterized sequence of reference frames.

You typically:

1. Define directly the reference frames of the system at discrete time samples, or provide quaternions or Euler angles at discrete time samples.
2. Create an :class:`Attitude <perseo_core.geometry.pointing.attitude.Attitude>` instance from those samples.
3. Evaluate the attitude at arbitrary query times using SLERP.

Construction Methods
--------------------

from reference frames
~~~~~~~~~~~~~~~~~~~~~~

The most direct way to construct an :class:`Attitude <perseo_core.geometry.pointing.attitude.Attitude>` object is to provide the reference frames of the system at discrete time samples.

Example:

.. code-block:: python

    import numpy as np
    from perseo_core.geometry.pointing.attitude import Attitude

    times = np.array([0.0, 1.0, 2.0])

    reference_frames = np.array([
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], # First frame is identity
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], # Second frame is 90° about Z 
        [[-1.0,   0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], # Third frame is 180° about Z
    ])

    attitude = Attitude(reference_frames=reference_frames, times=times)

from quaternions
~~~~~~~~~~~~~~~~

An ``Attitude.from_quaternions()`` method is provided to construct an :class:`Attitude <perseo_core.geometry.pointing.attitude.Attitude>`
object from time-tagged quaternions.

Example:

.. code-block:: python

    import numpy as np
    from perseo_core.geometry.pointing.attitude import Attitude

    times = np.array([0.0, 1.0, 2.0])

    quaternions = np.array([
        [0.0, 0.0, 0.0, 1.0], # First quaternion is identity
        [0.0, 0.0, 0.70710678, 0.70710678], # Second quaternion is 90° about Z
        [0.0, 0.0, 1.0, 0.0], # Third quaternion is 180° about Z
    ])

    attitude = Attitude.from_quaternions(quaternions=quaternions, times=times)

from Euler angles
~~~~~~~~~~~~~~~~~

An ``Attitude.from_euler_angles()`` method is provided to construct an :class:`Attitude <perseo_core.geometry.pointing.attitude.Attitude>`
object from time-tagged Euler angles.

Example:

.. code-block:: python

    import numpy as np
    from perseo_core.geometry.pointing.attitude import Attitude

    times = np.array([0.0, 2.0, 4.0])

    angles_rad = np.deg2rad(np.array([
        [0.0, 0.0, 0.0], # First set of angles is identity
        [0.0, 0.0, 45.0], # Second set of angles is 45° roll (about X)
        [0.0, 45.0, 90.0], # Third set of angles is 45° pitch (about Y) and 90° roll (about X)
    ]))

    attitude = Attitude.from_euler_angles(euler_angles_rad=angles_rad, rotation_order="YPR", times=times)

Interpolation with SLERP
------------------------

Once constructed, the :class:`Attitude <perseo_core.geometry.pointing.attitude.Attitude>` object internally creates a SciPy ``Slerp`` interpolator.
You can evaluate the rotation at arbitrary query times within the defined time interval. This returns a numpy array with shape (N, 3, 3) containing
the reference frame matrices corresponding to the query times.

Example:

.. code-block:: python

    query_times = np.linspace(0.0, 2.0, 5)

    reference_frames = attitude.evaluate(query_times)

    # Convert to Euler angles if needed
    from scipy.spatial.transform import Rotation
    from perseo_core.geometry.pointing.rotations import rotation_to_euler_angles

    euler_angles = rotation_to_euler_angles(Rotation.from_matrix(reference_frames), order="YPR")

Changing the Reference System
-----------------------------

**PERSEO Core** provides utilities to express an existing rotation in a new reference system. This means that the user can
define rotations in a given reference frame, may it be a local reference frame (i.e. Zero Doppler) or a global reference frame (i.e. ECEF),
and then re-express the interpolated rotations in a different reference system.

Functionalities for this are provided by the module :mod:`perseo_core.geometry.pointing.rotations`.
Please refer to :ref:`the dedicated documentation on Reference Frames <core_reference_frames_ch>` for further details.

Use Cases
---------

ECEF Quaternions
~~~~~~~~~~~~~~~~

An example of common use case is the following:

- Attitude information is provided in ECEF via quaternions
- The user wants to compute Euler angles in a sensor local reference frame (e.g. Zero Doppler)

.. code-block:: python

    import numpy as np
    from scipy.spatial.transform import Rotation
    from perseo_core.models.cubic_spline_trajectory import CubicSplineTrajectory
    from perseo_core.geometry.pointing.antenna_reference_frame import compute_euler_angles_from_antenna_reference_frame
    from perseo_core.geometry.pointing.attitude import Attitude
    from perseo_core.geometry.pointing.reference_frames import compute_sensor_local_axis
    from perseo_core.geometry.pointing.rotations import rotation_to_euler_angles

    # Init ECEF sensor trajectory
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
    trajectory = CubicSplineTrajectory(times=times, positions=positions, velocities=velocities)

    # Init ECEF antenna attitude
    quaternions = np.array([
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.70710678, 0.70710678],
        [0.0, 0.0, 1.0, 0.0],
    ])
    attitude = Attitude.from_quaternions(quaternions=quaternions, times=times)

    # Evaluate antenna reference frames at given times
    query_times = np.array([0.5, 0.85, 1.2, 1.8])
    antenna_reference_frames = attitude.evaluate(query_times)

    # Compute the sensor local reference frame (i.e. Zero Doppler) at the same query times
    zero_doppler_local_axis = compute_sensor_local_axis(
        sensor_positions=trajectory.position(query_times),
        sensor_velocities=trajectory.velocity(query_times),
        reference_frame="ZERODOPPLER"
    )

    # compute Euler angles in the Zero Doppler reference frame
    ypr_rad = compute_euler_angles_from_antenna_reference_frame(
        antenna_reference_frame=antenna_reference_frames,
        initial_reference_frame=zero_doppler_local_axis,
        rotation_order="YPR"
    )

Sensor Euler Angles
~~~~~~~~~~~~~~~~~~~

Another example of common use case is the following:

- Attitude information is provided in a sensor local reference frame (i.e. Zero Doppler) via Euler angles (Yaw, Pitch, Roll)
- The user wants to compute the antenna reference frame (ARF) in ECEF from the attitude information

.. code-block:: python

    import numpy as np
    from perseo_core.geometry.pointing.attitude import compute_antenna_attitude_from_euler_angles
    from perseo_core.geometry.pointing.reference_frames import compute_sensor_local_axis
    from perseo_core.geometry.pointing.rotations import euler_angles_to_rotation
    from perseo_core.models.cubic_spline_trajectory import CubicSplineTrajectory

    # Init ECEF sensor trajectory
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
    trajectory = CubicSplineTrajectory(times=times, positions=positions, velocities=velocities)

    # Euler Angles in a local reference frame (i.e. Zero Doppler)
    angles_rad = np.deg2rad(np.array([
        [0.0, 0.0, 30.0],
        [-0.1, 0.5, 31.0],
        [-0.2, 1.0, 32.0],
    ]))

    # this is the Zero Doppler reference frame expressed in ECEF coordinates
    zero_doppler_local_axis = compute_sensor_local_axis(
        sensor_positions=trajectory.position(times),
        sensor_velocities=trajectory.velocity(times),
        reference_frame="ZERODOPPLER"
    )

    # Directly compute antenna attitude in ECEF from Euler angles using helper function
    antenna_attitude = compute_antenna_attitude_from_euler_angles(
        ypr_rad=angles_rad,
        rotation_order="YPR",
        times=times,
        sensor_local_axis=zero_doppler_local_axis
    )
