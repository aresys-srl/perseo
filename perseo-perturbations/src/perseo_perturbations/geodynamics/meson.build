incdir_numpy = run_command(py3, [
    '-c',
    '''import os, numpy as np
try:
    print(os.path.relpath(np.get_include()))
except:
    print(np.get_include())
    ''',
], check: true).stdout().strip()
incdir_f2py = incdir_numpy / '..' / '..' / 'f2py' / 'src'
fortranobject_c = incdir_f2py / 'fortranobject.c'
inc_np = include_directories(incdir_numpy, incdir_f2py)

# Don't use the deprecated NumPy C API
numpy_nodepr_api = '-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION'

# Dependencies for Fortran
fortranobject_lib = static_library(
    '_fortranobject',
    fortranobject_c,
    c_args: numpy_nodepr_api,
    dependencies: py3_dep,
    include_directories: inc_np,
)
fortranobject_dep = declare_dependency(
    link_with: fortranobject_lib,
    include_directories: inc_np,
)

solid_source = custom_target(
    'solid',
    input : 'solid_src/solid.for',
    output : ['solidmodule.c', 'solid-f2pywrappers.f'],
    # command: [py3, '-m', 'numpy.f2py', '-c', '-m', 'solid', '@INPUT@', '--build-dir', '@OUTDIR@']
    command: [py3, '-m', 'numpy.f2py', '@INPUT@', '-m', 'solid', '--build-dir', '@OUTDIR@']
)

py3.extension_module(
    'solid',
    [solid_source, 'solid_src/solid.for'],
    include_directories: inc_np,
    c_args: numpy_nodepr_api,
    dependencies: fortranobject_dep,
    install : true,
    link_language: 'fortran',
    subdir: 'perseo_perturbations/geodynamics',
)

py3.install_sources(
    ['__init__.py', 'plate_tectonics.py', 'solid_tides.py'],
    subdir: 'perseo_perturbations/geodynamics'
)
