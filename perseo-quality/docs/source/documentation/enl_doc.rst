.. _quality_enl:

Equivalent Number of Looks (ENL) Analysis
=========================================

This analysis is usually used to check the validity of the SAR raster data in exam with respect to its focusing level.
For example, Single Look Complex (SLC) data should have an ENL value close to 1 when selecting an homogeneous noise data
portion.

The algorithm used to determine the equivalent number of looks is pretty simple and it should be used only on isolated
homogeneous salt and pepper like portions of the raster data:

    .. math::

        ENL = \frac{I_{mean}^2}{I_{std}^2}
