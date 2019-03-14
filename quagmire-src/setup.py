from setuptools import setup, find_packages
from numpy.distutils.core import setup, Extension
from os import path
import io

ext = Extension(name    = 'quagmire._fortran',
                        sources = ['fortran/quagmire.pyf','fortran/trimesh.f90'])


this_directory = path.abspath(path.dirname(__file__))
with io.open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()


if __name__ == "__main__":
    setup(name = 'quagmire',
          author            = "Louis Moresi",
          author_email      = "louis.moresi@unimelb.edu.au",
          url               = "https://github.com/University-of-Melbourne-Geodynamics/quagmire",
          version           = "0.5.0",
          description       = "Python surface process framework on highly scalable unstructured meshes",
          long_description  = long_description,
          long_description_content_type='text/markdown',
          ext_modules       = [ext],
          packages          = ['quagmire', 'quagmire.tools', 'quagmire.function', 'quagmire.mesh', 'quagmire.topomesh', 'quagmire.surfmesh'],
          install_requires = ['numpy', 'stripy'],
          package_data      = {'quagmire': ['Examples/Notebooks/data',
                                            # 'Examples/Notebooks/IdealisedExamples/*.ipynb',
                                            # 'Examples/Notebooks/LandscapeEvolution/*.ipynb',
                                            'Examples/Notebooks/WorkedExamples/*.ipynb',  ## Leave out Unsupported
                                            'Examples/Notebooks/Tutorial/*.ipynb',
                                            'Examples/Scripts/IdealisedExamples',
                                            'Examples/Scripts/LandscapeEvolution',
                                            'Examples/Scripts/LandscapePreprocessing',    ## Leave out Unsupported
                                            'Examples/Scripts/Scripts/*.py']},
          classifiers       = ['Programming Language :: Python :: 2',
                               'Programming Language :: Python :: 2.6',
                               'Programming Language :: Python :: 2.7',
                               'Programming Language :: Python :: 3',
                               'Programming Language :: Python :: 3.3',
                               'Programming Language :: Python :: 3.4',
                               'Programming Language :: Python :: 3.5',
                               'Programming Language :: Python :: 3.6']
          )
