"""
Copyright 2016-2017 Louis Moresi, Ben Mather, Romain Beucher

Quagmire is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or any later version.

Quagmire is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with Quagmire.  If not, see <http://www.gnu.org/licenses/>.
"""

from .mesh import PixMesh as _PixMesh
from .mesh import TriMesh as _TriMesh
from .mesh import sTriMesh as _sTriMesh
from petsc4py import PETSc as _PETSc
from .topomesh import TopoMesh as _TopoMeshClass
from .surfmesh import SurfMesh as _SurfaceProcessMeshClass

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


def FlatMesh(DM, *args, **kwargs):
    """
    Instantiates a mesh using TriMesh, sTriMesh or PixMesh objects.

    This object contains methods for the following operations:
     - calculating derivatives
     - interpolation (nearest-neighbour, linear, cubic)
     - local smoothing operations
     - identifying node neighbours

    Parameters
    ----------
     DM : PETSc DM object
        Either a DMDA or DMPlex object created using the meshing
        functions within the tools subdirectory

    Returns
    -------
     FlatMesh : object
    """

    # get DM label name
    BaseMeshType = _get_label(DM)

    if BaseMeshType in known_basemesh_classes:

        class FlatMeshClass(known_basemesh_classes[BaseMeshType]):

            __count = 0

            @classmethod
            def _count(cls):
                FlatMeshClass.__count += 1
                return FlatMeshClass.__count

            @property
            def id(self):
                return self.__id

            def __init__(self, dm, *args, **kwargs):

                # I can't see how to do this automatically
                # but it does seem useful to have the ID reflect the
                # mesh type (perhaps)

                if isinstance(self, _TriMesh):
                    self.__id = "trimesh_{}".format(self._count())
                elif isinstance(self, _PixMesh):
                    self.__id = "pixmesh_{}".format(self._count())
                else:
                    self.__id = "flatmesh_{}".format(self._count())

                known_basemesh_classes[BaseMeshType].__init__(self, dm, *args, **kwargs)

                # super(FlatMeshClass, self).__init__(dm, *args, **kwargs)

        return FlatMeshClass(DM, *args, **kwargs)

    else:
      raise TypeError("Mesh type {:s} unknown\n\
        Known mesh types: {}".format(BaseMeshType, list(known_basemesh_classes.keys())))

    return

def TopoMesh(DM, *args, **kwargs):
    """
    Instantiates a mesh with a height field.
    TopoMesh inherits from FlatMesh.

    This object contains methods for the following operations:
     - calculating the slope from height field
     - constructing downhill matrices
     - cumulative downstream flow
     - handling flat spots and local minima

    Call update_height to initialise these data structures.

    Parameters
    ----------
     DM : PETSc DM object
        Either a DMDA or DMPlex object created using the meshing
        functions within the tools subdirectory

    Returns
    -------
     TopoMesh : object
    """

    BaseMeshType = _get_label(DM)

    if BaseMeshType in known_basemesh_classes:
        class TopoMeshClass(known_basemesh_classes[BaseMeshType], _TopoMeshClass):
            def __init__(self, dm, *args, **kwargs):
                known_basemesh_classes[BaseMeshType].__init__(self, dm, *args, **kwargs)
                _TopoMeshClass.__init__(self, *args, **kwargs)
                # super(TopoMeshClass, self).__init__(dm, *args, **kwargs)

        return TopoMeshClass(DM, *args, **kwargs)

    else:
      raise TypeError("Mesh type {:s} unknown\n\
        Known mesh types: {}".format(BaseMeshType, list(known_basemesh_classes.keys())))

    return



def SurfaceProcessMesh(DM, *args, **kwargs):
    """
    Instantiates a mesh with a height and rainfall field.
    SurfaceProcessMesh inherits from FlatMesh and TopoMesh.

    This object contains methods for the following operations:
     - long-range flow models
     - calculate erosion and deposition rates
     - landscape equilibrium metrics
     - stream-wise smoothing

    Call update_height and update_surface_processes to initialise
    these data structures.

    Parameters
    ----------
     DM : PETSc DM object
        Either a DMDA or DMPlex object created using the meshing
        functions within the tools subdirectory

    Returns
    -------
     SurfaceProcessMesh : object
    """

    BaseMeshType = _get_label(DM)

    if BaseMeshType in known_basemesh_classes:
        class SurfaceProcessMeshClass(known_basemesh_classes[BaseMeshType], _SurfaceProcessMeshClass):
            def __init__(self, dm, *args, **kwargs):
                known_basemesh_classes[BaseMeshType].__init__(self, dm, *args, **kwargs)
                _TopoMeshClass.__init__(self, *args, **kwargs)
                _SurfaceProcessMeshClass.__init__(self, *args, **kwargs)
                # super(SurfaceProcessMeshClass, self).__init__(dm, *args, **kwargs)

        return SurfaceProcessMeshClass(DM, *args, **kwargs)

    else:
      raise TypeError("Mesh type {:s} unknown\n\
        Known mesh types: {}".format(BaseMeshType, list(known_basemesh_classes.keys())))

    return
