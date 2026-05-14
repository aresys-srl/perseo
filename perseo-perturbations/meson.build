project(
  'perturbations',
  'c',
  'cpp',
  version: '1.0.2',
  license: 'BSD-3',
  meson_version: '>= 1.1.0',
  default_options: [
    'warning_level=2',
    'c_std=c99',
    'fortran_std=legacy'
  ],
)

py3_mod = import('python')
py3 = py3_mod.find_installation(pure: false)
py3_dep = py3.dependency()


py3.install_sources(
    ['./src/perseo_perturbations/__init__.py'],
    subdir: 'perseo_perturbations'
)
py3.install_sources(
    ['./src/perseo_perturbations/resources/__init__.py'],
    subdir: 'perseo_perturbations/resources'
)
py3.install_sources(
    [
      './src/perseo_perturbations/resources/troposphere_support/gridpoint_coord_1x1.txt',
      './src/perseo_perturbations/resources/troposphere_support/gridpoint_coord_5x5.txt',
    ],
    subdir: 'perseo_perturbations/resources/troposphere_support'
)
py3.install_sources(
    [
      './src/perseo_perturbations/resources/troposphere_support/tropospheric_legendre_coefficients/anm_bh.txt',
      './src/perseo_perturbations/resources/troposphere_support/tropospheric_legendre_coefficients/anm_bw.txt',
      './src/perseo_perturbations/resources/troposphere_support/tropospheric_legendre_coefficients/anm_ch.txt',
      './src/perseo_perturbations/resources/troposphere_support/tropospheric_legendre_coefficients/anm_cw.txt',
      './src/perseo_perturbations/resources/troposphere_support/tropospheric_legendre_coefficients/bnm_bh.txt',
      './src/perseo_perturbations/resources/troposphere_support/tropospheric_legendre_coefficients/bnm_bw.txt',
      './src/perseo_perturbations/resources/troposphere_support/tropospheric_legendre_coefficients/bnm_ch.txt',
      './src/perseo_perturbations/resources/troposphere_support/tropospheric_legendre_coefficients/bnm_cw.txt',
    ],
    subdir: 'perseo_perturbations/resources/troposphere_support/tropospheric_legendre_coefficients'
)
py3.install_sources(
    [
      './tests/test_solid_earth_tides.py',
      './tests/test_plate_tectonics.py',
      './tests/test_atmospheric_map_names_generation.py',
      './tests/__init__.py'
    ],
    subdir: 'perseo_perturbations/tests'
)
py3.install_sources(
    [
      './src/perseo_perturbations/atmospheric/__init__.py',
      './src/perseo_perturbations/atmospheric/ionosphere.py',
      './src/perseo_perturbations/atmospheric/troposphere.py'
    ],
    subdir: 'perseo_perturbations/atmospheric'
)

add_languages('fortran', native: false)

subdir('src/perseo_perturbations/geodynamics')