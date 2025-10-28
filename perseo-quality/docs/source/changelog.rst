Changelog
=========

v1.0.7
------

**Bug Fixes**

- Fixing Spectral Analysis graphs generation, wrong code block indentation

**Other Changes**

- Adding acquisition_start_time to ``RadiometricOutputProductGeneralInfo`` dataclass
- Adding acquisition_start_time to Point Target Analysis ``GenericInfoOutput`` dataclass

v1.0.6
------

**Bug Fixes**

- Pinning numba version >=0.61
- Fixing bug in Point Target Analysis graphical output generation, missing f-string in title generation

v1.0.5
------

**New Features**

- Spectral Analysis: added a new block-wise spectral analysis

**Other Changes**

- Radiometric Analysis Block-Wise: silenced a All NaN slice encountered warning in graphical_output.py
- Improved Radiometric Analysis NESZ performance using numba
- Radiometry: zeroes in raster data are now treated as NaNs
- Silencing all NaN slice warning
- Spectral Analysis: refactoring of Point Target Spectral Analysis

**Bug Fixes**

- Average Radiometric profiles: fixed a bug for `variability_index` KPI computation

v1.0.4
------

**Other Changes**

- Removing ``pulse_rate`` property from quality protocol definition
- Radiometric output KPI `product_name` field changed to `product`
- Adding the complete list of dataframe columns for Point Target Analysis output as ``PTA_OUTPUT_COLUMNS_DF_UM``
- Adding the complete list of dataframe columns for Block-Wise Radiometric Analysis KPIs outputs

v1.0.3
------

**Bug Fixing**

- Radiometric profiles missing ``kpi_estimator`` argument

v1.0.2
------

**New Features**

- Added KPI estimation for Block-Wise Radiometric Analysis
- Added RCS Geometrical computation for trihedral corner reflectors

**Other Changes**

- Fixed graphs name and title for Point Target Analysis and Radiometric Analysis Block-Wise
- Added ``sensor_name`` property to quality input channel protocol
- Added ``sensor``, ``product`` and ``acquisition_mode`` info to Point Target Analysis output .csv
- Removed ``pulse_latch_time`` and ``swst_changes`` from protocol definition

**Incompatible Changes**

- Changing ENL core function behavior ``compute_equivalent_number_of_looks``, moving intensity data conversion to the wrapping routine

v1.0.1
------

**Other Changes**

- Added a centralized dedicated logger for the whole package
- Changed few Log ERROR statements to WARNING

v1.0.0
------

First released version.
