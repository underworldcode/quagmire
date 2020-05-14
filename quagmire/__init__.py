# Copyright 2016-2020 Louis Moresi, Ben Mather, Romain Beucher
# 
# This file is part of Quagmire.
#
# Quagmire is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or any later version.
# 
# Quagmire is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with Quagmire.  If not, see <http://www.gnu.org/licenses/>.

"""
<img src="https://raw.githubusercontent.com/underworldcode/quagmire/dev/docs/images/AusFlow.png" style="display: block; margin: 0 auto">

**Quagmire is a Python surface process framework for building erosion and deposition models on highly parallel, decomposed structured and unstructured meshes.**

Quagmire is structured into three major tiers that inherit methods and attributes from different classes:

<img src="https://raw.githubusercontent.com/underworldcode/quagmire/dev/docs/images/quagmire-flowchart.png" style="width: 321px; float:right">

1. `SurfaceProcessMesh`
    - Calculate erosion-deposition rates
    - Landscape analysis
    - Long range flow models
2. `TopoMesh`
    - Assemble downhill connectivity matrix
    - Calculate upstream area
    - Compute slope
    - Identify flat spots, low points, high points.
3. `FlatMesh`
    - calculating spatial derivatives
    - identifying node neighbour relationships
    - interpolation / extrapolation
    - smoothing operators
    - importing and saving mesh information

The `quagmire.surfmesh.surfmesh.SurfMesh` class (1) inherits from
the `quagmire.topomesh.topomesh.TopoMesh` class (2), which in turn inherits from
the `quagmire.mesh.pixmesh.PixMesh`, `quagmire.mesh.trimesh.TriMesh`, or
`quagmire.mesh.strimesh.sTriMesh` class (3) depending on the type of mesh.


## Installation

Numpy and a fortran compiler, preferably [gfortran](https://gcc.gnu.org/wiki/GFortran), are required to install Quagmire.

- ``python setup.py build``
   - If you change the fortran compiler, you may have to add the
flags `config_fc --fcompiler=<compiler name>` when setup.py is run
(see docs for [numpy.distutils](http://docs.scipy.org/doc/numpy-dev/f2py/distutils.html)).
- ``python setup.py install``

## Dependencies

Running this code requires the following packages to be installed. The visualisation options are required for the notebooks and are included in the docker image.

- Python 2.7.x and above
- Numpy 1.9 and above
- Scipy 0.15 and above
- [mpi4py](http://pythonhosted.org/mpi4py/usrman/index.html)
- [petsc4py](https://pythonhosted.org/petsc4py/usrman/install.html)
- [stripy](https://github.com/University-of-Melbourne-Geodynamics/stripy)
- [h5py](http://docs.h5py.org/en/latest/mpi.html#building-against-parallel-hdf5) (optional - for saving parallel data)
- Matplotlib (optional - for visualisation)
- lavavu (optional - for visualisation)

### PETSc installation

PETSc is used extensively via the Python frontend, petsc4py. It is required that PETSc be configured and installed on your local machine prior to using Quagmire. You can use pip or petsc to install petsc4py and its dependencies with consistent versions.

```sh
[sudo] pip install numpy mpi4py
[sudo] pip install petsc petsc4py
```

If that fails you must compile these manually.

### HDF5 installation

This is an optional installation, but it is very useful for saving data that is distributed across multiple processes. If you are compiling HDF5 from [source](https://support.hdfgroup.org/downloads/index.html) it should be configured with the `--enable-parallel` flag:

```sh
CC=/usr/local/mpi/bin/mpicc ./configure --enable-parallel --enable-shared --prefix=<install-directory>
make    # build the library
make check  # verify the correctness
make install
```

You can then point to this install directory when you install [h5py](http://docs.h5py.org/en/latest/mpi.html#building-against-parallel-hdf5).

## Usage

Quagmire is scalable in parallel. All of the python scripts in the *tests* subdirectory can be run in parallel, e.g.

```sh
mpirun -np 4 python stream_power.py
```

where the number after the `-np` flag specifies the number of processors.

## Tutorials

Tutorials with worked examples can be found in the *Notebooks* subdirectory. These are Jupyter Notebooks that can be run locally.
We recommend installing [FFmpeg](https://ffmpeg.org/) to create videos in some of the notebooks.

The topics covered in the Notebooks include:

**Meshing**

Triangulations and regular 2d arrays are included and have the same API.

- Square mesh
- Elliptical mesh
- Mesh refinement (e.g. Lloyd's mesh improvement)

**Flow algorithms**

- Single and multiple downhill pathways
- Accumulating flow

**Erosion and deposition**

Operators for

- Long-range stream flow models
- Short-range diffusive evolution

**Landscape evolution**

_Work in progress_

- Explicit timestepping and numerical stability
- Landscape equilibrium metrics
- Basement uplift

**Benchmarking**

_Work in progress_
"""

from .mesh import PixMesh as _PixMesh
from .mesh import TriMesh as _TriMesh
from .mesh import sTriMesh as _sTriMesh
from petsc4py import PETSc as _PETSc
from .topomesh import TopoMesh as _TopoMeshClass

from . import documentation
from . import tools
from . import function
from . import scaling
from . import equation_systems

try:
    import lavavu
except:
    pass

_display = None

from mpi4py import MPI as _MPI
mpi_rank = _MPI.COMM_WORLD.rank
mpi_size = _MPI.COMM_WORLD.size


class _xvfb_runner(object):
    """
    This class will initialise the X virtual framebuffer (Xvfb).
    Xvfb is useful on headless systems. Note that xvfb will need to be
    installed, as will pyvirtualdisplay.
    This class also manages the lifetime of the virtual display driver. When
    the object is garbage collected, the driver is stopped.
    """
    def __init__(self):
        from pyvirtualdisplay import Display
        self._xvfb = Display(visible=0, size=(1600, 1200))
        self._xvfb.start()

    def __del__(self):
        if not self._xvfb is None :
            self._xvfb.stop()

import os as _os

# disable collection of data if requested
if "GLUCIFER_USE_XVFB" in _os.environ:
    from mpi4py import MPI as _MPI
    _comm = _MPI.COMM_WORLD
    if _comm.rank == 0:
        _display = _xvfb_runner()


known_basemesh_classes = {"PixMesh"  : _PixMesh, \
                          "TriMesh"  : _TriMesh, \
                          "sTriMesh" : _sTriMesh}


def _get_label(DM):
    """
    Retrieves all points in the DM that is marked with a specific label.
    e.g. "boundary", "coarse"
    """

    n = DM.getNumLabels()
    success = False

    for i in range(n):
        label = DM.getLabelName(i)
        if label in known_basemesh_classes:
            success = True
            break

    if not success:
        raise NameError("Cannot identify mesh type. DM is not valid.")

    return label



def QuagMesh(DM, *args, **kwargs):
    """
    Instantiates a mesh with a height and rainfall field.
    QuagMesh identifies the type of DM and builds the necessary
    data structures for landscape processing and analysis.


    Parameters
    ----------
    DM : PETSc DM object
        Either a DMDA or DMPlex object created using the meshing
        functions within `tools.meshtools`

    Returns
    -------
    QuagMesh : object
        Inherits methods and attributes from:

        - `mesh.commonmesh.CommonMesh`
        - `mesh.pixmesh.PixMesh` (if `DM` is a regularly-spaced Cartesian grid)
        - `mesh.trimesh.TriMesh` (if `DM` is an unstructred Cartesian mesh)
        - `mesh.strimesh.sTriMesh` (if `DM` is an unstructured spherical mesh)
        - `topomesh.topomesh.TopoMesh`
    """

    BaseMeshType = _get_label(DM)

    if BaseMeshType in known_basemesh_classes:
        class QuagMeshClass(known_basemesh_classes[BaseMeshType], _TopoMeshClass):
            def __init__(self, dm, *args, **kwargs):
                known_basemesh_classes[BaseMeshType].__init__(self, dm, *args, **kwargs)
                _TopoMeshClass.__init__(self, *args, **kwargs)
                # super(QuagMeshClass, self).__init__(dm, *args, **kwargs)

        return QuagMeshClass(DM, *args, **kwargs)

    else:
        raise TypeError("Mesh type {:s} unknown\n\
            Known mesh types: {}".format(BaseMeshType, list(known_basemesh_classes.keys())))

    return
