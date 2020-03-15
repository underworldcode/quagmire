from setuptools import setup, find_packages
from numpy.distutils.core import setup, Extension

try: 
    from distutils.command import bdist_conda
except ImportError:
    pass



from os import path
import io

ext = Extension(name    = 'quagmire._fortran',
                sources = ['src/quagmire.pyf','src/trimesh.f90'])


this_directory = path.abspath(path.dirname(__file__))
with io.open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


if __name__ == "__main__":
    setup(name = 'quagmire',
          author            = "Ben Mather",
          author_email      = "ben.mather@sydney.edu.au",
          url               = "https://github.com/underworldcode/quagmire",
          version           = "0.7.0",
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
                               'quagmire.topomesh',
                               'quagmire.surfmesh'],
          install_requires  = ['numpy>=1.16.0', 'scipy>=1.0.0', 'stripy>=1.2', 'petsc4py', 'mpi4py', 'h5py', 'pint'],
          package_data      = {'quagmire': ['Examples/Notebooks/data',
                                            'Examples/Notebooks/*.ipynb',
                                            'Examples/Notebooks/WorkedExamples/*.ipynb',  ## Leave out Unsupported
                                            'Examples/Notebooks/Tutorial/*.ipynb',
                                            'Examples/Scripts/Meshes/*.py',
                                            'Examples/Scripts/LandscapePreprocessing/*.py']},
          classifiers       = ['Programming Language :: Python :: 2',
                               'Programming Language :: Python :: 2.7',
                               'Programming Language :: Python :: 3',
                               'Programming Language :: Python :: 3.5',
                               'Programming Language :: Python :: 3.6',
                               'Programming Language :: Python :: 3.7',]
          )
