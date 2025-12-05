.. toctree::
   :maxdepth: 2
   :hidden:

   point_target_analysis_doc
   radiometric_analysis_doc
   interferometric_analysis_doc
   target_ambiguity_ratio_doc
   spectral_analysis_doc
   enl_doc
   elevation_notch_doc

Documentation
=============

`PERSEO Quality` is the python module needed to perform quality analyses on SAR products.
It is a fully developed python library that can be used integrating its functionalities in custom scripts.

The main analyses that can be performed using this tool are:

- **Point Target Analysis**
- **Radiometric Analysis**
- **Interferometric Analysis**
- **Point & Distributed Target Ambiguity Ratio Analysis**
- **Point & Distributed Target Spectral Analysis**
- **Equivalent Number of Looks Analysis**
- **Elevation Notch Analysis**

These functionalities are located in different submodules of this framework that can be accessed as:

- `perseo_quality.point_target_analysis`
- `perseo_quality.radiometric_analysis`
- `perseo_quality.interferometric_analysis`
- `perseo_quality.target_ambiguity_ratio_analysis`
- `perseo_quality.spectral_analysis`
- `perseo_quality.enl_analysis`
- `perseo_quality.elevation_notch_analysis`

These operations can be fully customized down to low level parameters in order to tweak and tune the algorithms behavior
to the users' needs.
