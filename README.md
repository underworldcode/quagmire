# Quagmire

![build/test](https://github.com/underworldcode/quagmire/workflows/build/test/badge.svg)

Quagmire is a Python surface process framework for building erosion and deposition models on highly parallel, decomposed structured and unstructured meshes.

![Quagmire Surface Process Framework](https://raw.githubusercontent.com/underworldcode/quagmire/dev/docs/images/AusFlow.png)



## Documentation

The documentation is in the form of jupyter notebooks that are online in the form of a jupyter-book

### "Stable" code (developer release, 2020)

  - Documentation / Notebooks [https://underworldcode.github.io/quagmire/0.9.5](https://underworldcode.github.io/quagmire/0.9.5)
  - API documentation [https://underworldcode.github.io/stripy/quagmire/0.9.5](https://underworldcode.github.io/stripy/quagmire/0.9.5_api)


### Bleeding edge code 

  - Documentation / Notebooks [https://underworldcode.github.io/quagmire/0.9.6b1](https://underworldcode.github.io/quagmire/0.9.6b1)
  - API documentation [https://underworldcode.github.io/stripy/quagmire/0.9.6b1_api](https://underworldcode.github.io/stripy/quagmire/0.9.6b1_api)




## Demonstration

[![Launch Demo](https://img.shields.io/badge/Launch-Quagmire_Demo-Blue)](https://demon.underworldcloud.org/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2Funderworld-community%2Fquagmire-examples-and-workflows&urlpath=lab%2Ftree%2Fquagmire-examples-and-workflows%2F0-StartHere.ipynb)


## Installation

The __native installation__ of Quagmire requires a number of [dependencies](#Dependencies) and a fortran compiler, preferably [gfortran](https://gcc.gnu.org/wiki/GFortran). Install Quagmire using __setuptools__:
```sh
python setup.py build
python setup.py install
```

Or using __pip__:
```sh
pip3 install quagmire
```

- If you change the fortran compiler, you may have to add the flags `config_fc --fcompiler=<compiler name>` when setup.py is run (see docs for [numpy.distutils](http://docs.scipy.org/doc/numpy-dev/f2py/distutils.html)).

If you are using Python 3.7+ and Linux or MacOS, then you may wish to install Quagmire using __conda__:
```sh
conda install -c underworldcode quagmire
```

To run the demonstration examples:
```sh
conda install -c conda-forge gdal cartopy
conda install -c underworldcode quagmire
```

However, we are aware that the dependency list is quite large and restrictive and this can make it tough for Anaconda to install other complicated packages. You may need to do this in a separate conda environment. 

Some parts of the examples include references to the [lavavu](https://github.com/lavavu/LavaVu) package which has its own installation requirements and it might be best to read [their documentation](https://lavavu.github.io/Documentation/)

Alternatively


## Dependencies

Installing these dependencies is not required if you follow the Conda or Docker installation method. If you choose to install Quagmire natively, then the following packages are required:

- Python 3.7.x and above
- Numpy 1.9 and above
- Scipy 0.15 and above
- [mpi4py](http://pythonhosted.org/mpi4py/usrman/index.html)
- [petsc4py](https://pythonhosted.org/petsc4py/usrman/install.html)
- [stripy](https://github.com/University-of-Melbourne-Geodynamics/stripy)
- [h5py](http://docs.h5py.org/en/latest/mpi.html#building-against-parallel-hdf5)

#### Optional dependencies

These dependencies are required to run the Jupyter Notebook examples:

- Matplotlib
- [Cartopy](https://scitools.org.uk/cartopy/docs/latest/)
- [lavavu](https://github.com/lavavu/LavaVu)

#### PETSc installation

PETSc is used extensively via the Python frontend, petsc4py. It is required that PETSc be configured and installed on your local machine prior to using Quagmire. You can use pip to install petsc4py and its dependencies.

```sh
[sudo] pip install numpy mpi4py
[sudo] pip install petsc petsc4py
```

If that fails you must compile these manually.

#### HDF5 installation

If you are compiling HDF5 from [source](https://support.hdfgroup.org/downloads/index.html) it should be configured with the `--enable-parallel` flag:

```sh
CC=/usr/local/mpi/bin/mpicc ./configure --enable-parallel --enable-shared --prefix=INSTALL-DIRECTORY
make	# build the library
make check	# verify the correctness
make install
```

You can then point to this installation directory when you install [h5py](http://docs.h5py.org/en/latest/mpi.html#building-against-parallel-hdf5).

## Usage

Quagmire is highly scalable. All of the python scripts in the *tests* subdirectory can be run in parallel, e.g.

```
mpirun -np 4 python stream_power.py
```

where the number after the `-np` flag specifies the number of processors.

## Tutorials and worked examples

Tutorials with worked examples can be found in the [Quagmire Community repository](https://github.com/underworld-community/quagmire-examples-and-workflows). The topics covered in the Notebooks include:

**Meshing**

- Square mesh
- Elliptical mesh
- Mesh refinement (e.g. Lloyd's mesh improvement)
- Poisson disc sampling
- Mesh Variables
- quagmire function interface (requires a base mesh)

**Flow algorithms**

- Single and multiple downhill pathways
- Accumulating flow

**Erosion and deposition**

- Long-range stream flow models
- Short-range diffusive evolution

**Landscape evolution**

- Explicit timestepping and numerical stability
- Landscape equilibrium metrics
- Basement uplift

## Credits

The primary authors of the code are [Ben Mather](https://github.com/brmather), [Louis Moresi](https://github.com/lmoresi) and [Romain Beucher](https://github.com/rbeucher). We take collective responsibility for creating and maintaining the code. Here and there in the source code we mention who originated the code or modified it in order to help direct questions.
