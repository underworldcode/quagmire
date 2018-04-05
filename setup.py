from numpy.distutils.core import setup, Extension

ext = Extension(name    = 'quagmire._fortran',
                sources = ['fortran/quagmire.pyf','fortran/trimesh.f90'])

if __name__ == "__main__":
    setup(name = 'quagmire',
          author            = "Louis Moresi",
          author_email      = "louis.moresi@unimelb.edu.au",
          url               = "https://github.com/University-of-Melbourne-Geodynamics/quagmire",
          version           = "0.1",
          description       = "Python surface process framework on highly scalable unstructured meshes",
          ext_modules       = [ext],
          packages          = ['quagmire', 'quagmire.tools', 'quagmire.mesh', 'quagmire.topomesh', 'quagmire.surfmesh'],
          package_data      = ['quagmire': ['Examples/Notebooks/data',
                                            'Examples/Notebooks/IdealisedExamples',
                                            'Examples/Notebooks/LandscapeEvolution',
                                            'Examples/Notebooks/LandscapePreprocessing',  ## Leave out Unsupported
                                            'Examples/Notebooks/Meshes'],
                                            'Examples/Scripts/IdealisedExamples',
                                            'Examples/Scripts/LandscapeEvolution',
                                            'Examples/Scripts/LandscapePreprocessing',    ## Leave out Unsupported
                                            'Examples/Scripts/Meshes']],
          classifiers       = ['Programming Language :: Python :: 2',
                               'Programming Language :: Python :: 2.6',
                               'Programming Language :: Python :: 2.7',
                               'Programming Language :: Python :: 3',
                               'Programming Language :: Python :: 3.3',
                               'Programming Language :: Python :: 3.4',
                               'Programming Language :: Python :: 3.5',
                               'Programming Language :: Python :: 3.6']
          )
