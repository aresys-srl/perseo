.. _quality_notch:

Elevation Pointing Notch Analysis
=================================

Elevation pointing calibration estimates and corrects any bias between the nominal and actual antenna elevation pointing,
which is critical for accurate SAR data calibration. An incorrect pointing leads to improper compensation of the
**Elevation Antenna Pattern (EAP)**, causing radiometric artifacts such as gain trends in the SAR imagery.

.. figure:: ../_static/images/elevation_notch_schema.png
   :align: center
   :width: 1000

   *Figure 1: Satellite pointing schematics*

The actual antenna pointing is estimated directly from SAR data, accounting for the fact that the beam is generally steered
toward a swath-dependent look angle rather than the antenna boresight. The calibration is performed by finding the
mis-pointing angle that best matches the measured SAR range profiles with the theoretical EAP.

To improve robustness, dedicated Elevation Notch (EN) patterns, characterized by a central low-power "hole" are used,
as shown below by an EN acquisition over rainforest areas.

.. figure:: ../_static/images/elevation_notch_s1.png
   :align: center
   :width: 1000

   *Figure 2: Sentinel-1 C Elevation Notch Acquisition*

Analysis Algorithm
^^^^^^^^^^^^^^^^^^

The procedure for the estimation of the elevation pointing offset is based on the Least Square fitting of the elevation
profile of the focused data with a three parameters model:

   .. math::

      d(\theta_{off};k,\theta_{mis},n) = k \cdot p(\theta_{off} - \theta_{mis}) + n \cdot f(\theta_{off})

where :math:`p(\theta_{off})` is the Elevation Notch antenna pattern as a function of the off-boresight angle and
:math:`f(\theta_{off})` is the noise floor after compensation of the spread losses and the conversion of the data from
Beta Nought to Gamma Nought.

The model parameters are estimated by minimizing the sum of the squared differences between the measured and theoretical
profiles as shown in the following equation:

   .. math::

      argmin_{k,\theta_{mis},n}||\hat{d}(\theta_{i})-d(\theta_{i})||^{2}

where :math:`\hat{d}(\theta_{i})` is the vector containing the values of the range profile measured on the SAR data at
certain off-boresight angles, and :math:`d(\theta_{i})` is the model in evaluated at the same angles.

The three parameters to be estimated are:

- :math:`k` the calibration factor for the antenna pattern
- :math:`\theta_{mis}` the elevation mis-pointing angle
- :math:`n` the calibration factor for the thermal noise

The resulting :math:`\theta_{mis}` angle is the estimated pointing bias.

The previously described estimation procedure is carried out **only when the Antenna Pattern is provided**. This input
must be provided as a nested dictionary of XArray Datasets containing the gain of and the elevation angles axis as shown
below.

.. code-block:: python

   import xarray as xr

   antenna_pattern_datasets = {
      "swath": {
         "polarization": xr.Dataset(
            {
               "gain": (
                  ["azimuth_angles", "elevation_angles"],
                  gain_data,  # in dB
               ),
               ...
            },
            coords={
               "elevation_angles": elevation_angles_axis,  # in deg
               "azimuth_angles": azimuth_angles_axis,  # in deg
               ...
            },
         )
      }
   }


In addition to this estimation procedure based on the Antenna Pattern, another estimation method is carried out using a
**Parabolic Fit** of the data profile minimum. This is always performed and does not require additional inputs other than
the product itself.

Analysis Output
^^^^^^^^^^^^^^^

Elevation Notch analysis output consists in a NetCDF file containing estimated profiles for each channel analyzed.
Graphical output can also be generated using the ``graphical_output.plot_elevation_notch_analysis`` function to obtain
the plots.

.. figure:: ../_static/images/elevation_notch_estimate.png
   :align: center
   :width: 1000

   *Figure 3: Elevation Notch graphical output*

.. note::

    Graphical output functionalities are available only if the package has been installed with the [graphs] optional
    dependencies. Refer to the :ref:`installation documentation<pkg_install>` for more information.