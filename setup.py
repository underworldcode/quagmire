from setuptools import setup, find_packages
from numpy.distutils.core import setup, Extension

import os
import io
import subprocess
import platform 

PYPI_VERSION = "0.9.5"

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
                               'quagmire.tools.cloud',
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
