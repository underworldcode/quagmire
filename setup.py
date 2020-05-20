from setuptools import setup, find_packages
from numpy.distutils.core import setup, Extension

try: 
    from distutils.command import bdist_conda
except ImportError:
    pass


import os
import io
import subprocess
import platform 

# in development set version to none and ...
PYPI_VERSION = "0.9.1b"

def git_version():
    
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout = subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', '--short', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION


if PYPI_VERSION is None:
    PYPI_VERSION = git_version()



ext = Extension(name    = 'quagmire._fortran',
                sources = ['src/quagmire.pyf','src/trimesh.f90'])


this_directory = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


if __name__ == "__main__":
    setup(name = 'quagmire',
          author            = "Ben Mather",
          author_email      = "ben.mather@sydney.edu.au",
          url               = "https://github.com/underworldcode/quagmire",
          version           = PYPI_VERSION,
          description       = "Python surface process framework on highly scalable unstructured meshes",
          long_description  = long_description,
          long_description_content_type='text/markdown',
          ext_modules       = [ext],
          packages          = ['quagmire',
                               'quagmire.tools',
                               'quagmire.equation_systems',
                               'quagmire.function',
                               'quagmire.scaling',
                               'quagmire.mesh',
                               'quagmire.topomesh'],
          install_requires  = ['numpy>=1.16.0', 'scipy>=1.0.0', 'stripy>=1.2', 'petsc4py', 'mpi4py', 'h5py', 'pint'],
          classifiers       = ['Programming Language :: Python :: 2',
                               'Programming Language :: Python :: 2.7',
                               'Programming Language :: Python :: 3',
                               'Programming Language :: Python :: 3.5',
                               'Programming Language :: Python :: 3.6',
                               'Programming Language :: Python :: 3.7',]
          )
