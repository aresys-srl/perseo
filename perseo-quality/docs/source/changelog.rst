Changelog
=========

v1.0.4
------

**Other Changes**

- Removing ``pulse_rate`` property from quality protocol definition
- Radiometric output KPI `product_name` field changed to `product`

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
