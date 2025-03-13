project( 'quagmire',
    'fortran', 'c', 
    version:'0.9.7b0',
    license:'GPLv3',
    meson_version: '>=0.64.0',
    default_options : ['warning_level=2', 'python.install_env=auto'])

# make some environment variables
proj_name = meson.project_name()
py_mod    = import('python')
py        = py_mod.find_installation(pure: false)
py_dep    = py.dependency()

# gather numpy and fortran include dirs
incdir_numpy = run_command(py,
  ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

incdir_f2py = run_command(py,
    ['-c', 'import os; os.chdir(".."); import numpy.f2py; print(numpy.f2py.get_include())'],
    check : true
).stdout().strip()

inc_np = include_directories(incdir_numpy, incdir_f2py)

# Dictionary of sources for each extension.
# Here we index each extension by name.
# use pre existing pyf files as module headers
qextensions = {
    '_fortran': { 
        'src':'src/trimesh.f90', 
        'fsig':'src/quagmire.pyf', 
        'extra_files': '' 
    },
}

# # Create extension modules
foreach mod : qextensions.keys()
    mdict   = qextensions.get(mod)
    sources = mdict.get('src')
    target  = mdict.get('fsig')
    largs   = mdict.get('extra_files')

    #message('Building extension module: ' + mod)
    #message(sources)
    ext_source = custom_target(mod,
          input : target,  # .f so no F90 wrappers
          output : [mod+'module.c', mod+'-f2pywrappers.f'],
          command : [py, '-m', 'numpy.f2py', '@INPUT@']
        
        # this causes mysterious errors - not sure why
        #command : [py, '-m', 'numpy.f2py', '@INPUT@', '-m', mod, '--lower']
    )
   
    # compile wrapper to shared object and install to python site-packages
    py.extension_module(mod,
      [sources, ext_source],
      incdir_f2py / 'fortranobject.c',
      link_args: largs, ### DOUBLE CHECK THIS
      include_directories: inc_np,
      dependencies : py_dep,
      subdir: proj_name,
      install : true
    )

    message('Extension module: ' + mod + ' to be built and installed')
endforeach

install_subdir('./quagmire/', install_dir: py.get_install_dir())

## Install other data or resources (like LICENSE or README)
#py.install_sources('../LICENSE', subdir: proj_name)
#py.install_sources('./badlands/__init__.py', subdir: proj_name)
#py.install_sources('./badlands/model.py', subdir: proj_name)
#install_subdir('./badlands/', install_dir: py.get_install_dir())
