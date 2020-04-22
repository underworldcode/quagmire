# Quagmire



Quagmire is a Python surface process framework for building erosion and deposition models on highly parallel, decomposed structured and unstructured meshes.

Quagmire is structured into three major classes that inherit methods and attributes from lower tiers.

![Quagmire hierarchy](https://github.com/Underworldcode/quagmire/blob/master/Notebooks/Images/hierarchy_chart.png)

The Surface Processes class inherits from the Topography class, which in turn inherits from TriMesh or PixMesh depending on the type of mesh.

## Demonstration

[![https://img.shields.io/badge/<LABEL>-<MESSAGE>-<COLOR>](https://img.shields.io/badge/Launch-Quagmire_Demo-blue)](https://demon.underworldcloud.org/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2Funderworldcode%2Fquagmire&urlpath=lab%2Ftree%2Fquagmire%2Fquagmire%2FExamples%2FNotebooks)


## Installation

Numpy and a fortran compiler, preferably [gfortran](https://gcc.gnu.org/wiki/GFortran), are required to install Quagmire.

- ``python setup.py build``
   - If you change the fortran compiler, you may have to add the
flags `config_fc --fcompiler=<compiler name>` when setup.py is run
(see docs for [numpy.distutils](http://docs.scipy.org/doc/numpy-dev/f2py/distutils.html)).
- ``python setup.py install``'

If you are using python 3.7+ and linux or Macos, then you may wish to try

- `conda install -c underworldcode quagmire` 

To run the demonstration examples

- `conda install -c conda-forge gdal cartopy`
- `conda install -c underworld code quagmire`

However, we are aware that the dependency list is quite large and restrictive and this can make it tough for anaconda to install other complicated packages. You make need to do this in a separate conda environment. 

Some parts of the examples include references to the [lavavu](https://github.com/lavavu/LavaVu) package which has its own installation requirements and it might be best to read [their documentation](https://lavavu.github.io/Documentation/)


## Dependencies

Running this code requires the following packages to be installed:

- Python 3.7.x and above
- Numpy 1.9 and above
- Scipy 0.15 and above
- [mpi4py](http://pythonhosted.org/mpi4py/usrman/index.html)
- [petsc4py](https://pythonhosted.org/petsc4py/usrman/install.html)
- [stripy](https://github.com/University-of-Melbourne-Geodynamics/stripy)
- [h5py](http://docs.h5py.org/en/latest/mpi.html#building-against-parallel-hdf5) (optional - for saving parallel data)
- Matplotlib (optional - for visualisation)
- lavavu (optional - for visualisation)

### PETSc installation

PETSc is used extensively via the Python frontend, petsc4py. It is required that PETSc be configured and installed on your local machine prior to using Quagmire. You can use pip to install petsc4py and its dependencies.

```
[sudo] pip install numpy mpi4py
[sudo] pip install petsc petsc4py
```

If that fails you must compile these manually.

### HDF5 installation

This is an optional installation, but it is very useful for saving data that is distributed across multiple processes. If you are compiling HDF5 from [source](https://support.hdfgroup.org/downloads/index.html) it should be configured with the `--enable-parallel` flag:

```
CC=/usr/local/mpi/bin/mpicc ./configure --enable-parallel --enable-shared --prefix=<install-directory>
make	# build the library
make check	# verify the correctness
make install
```

You can then point to this install directory when you install [h5py](http://docs.h5py.org/en/latest/mpi.html#building-against-parallel-hdf5).

## Usage

Quagmire is highly scalable. All of the python scripts in the *tests* subdirectory can be run in parallel, e.g.

```
mpirun -np 4 python stream_power.py
```

where the number after the `-np` flag specifies the number of processors.

## Tutorials

Tutorials with worked examples can be found in the *Notebooks* subdirectory. These are Jupyter Notebooks that can be run locally. We recommend installing [FFmpeg](https://ffmpeg.org/) to create videos in some of the notebooks.

The topics covered in the Notebooks include:

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

The primary authors of the code are Ben Mather, Louis Moresi and Romain Beucher. We take collective responsibility for creating and maintaining the code. Here and there in the source code we mention who originated the code or modified it in order to help direct questions.


## Release Notes v0.6.0b

Adding some functionality to help match underworld and quagmire in their usage patterns:

  - Equation systems - diffusion added as a template
  - Add the scaling module
  -




## Release Notes v0.5.0b

This is the first formal 'release' of the code which

Summary of changes

 - Introducing quagmire.function which is a collection of lazy-evaluation objects similar to underworld functions
 - Introducing MeshVariables which wrap PETSc data vectors and provide interoperability with quagmire functions
 - Providing context manager support for changes to topography that automatically update matrices appropriately
 - Making all mesh variable data arrays view only except for assignment from a suitably sized numpy array (this is to ensure correct synchronisation of information in parallel).
 - various @property definitions to handle cases where changes require rebuilding of data structures
 - making many mesh methods private and exposing them via functions
   - upstream integration is a function on the mesh
   - upstream / downstream smoothing is via a mesh function
   - rbf smoothing builds a manager that provides a function interface
