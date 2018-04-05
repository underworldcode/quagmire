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
from petsc4py import PETSc as _PETSc
from .topomesh import TopoMesh as _TopoMeshClass
from .surfmesh import SurfMesh as _SurfaceProcessMeshClass
from . import documentation

import tools

known_basemesh_classes = {type(_PETSc.DMDA())   : _PixMesh,\
                          type(_PETSc.DMPlex()) : _TriMesh}


def FlatMesh(DM, *args, **kwargs):
    """
    Instantiates a 2-D mesh using TriMesh or PixMesh objects.

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
    BaseMeshType = type(DM)
    if BaseMeshType in known_basemesh_classes.keys():

        class FlatMeshClass(known_basemesh_classes[BaseMeshType]):
            def __init__(self, dm, *args, **kwargs):
                known_basemesh_classes[BaseMeshType].__init__(self, dm, *args, **kwargs)
                # super(FlatMeshClass, self).__init__(dm, *args, **kwargs)

        return FlatMeshClass(DM, *args, **kwargs)

    else:
      raise TypeError("Mesh type {:s} unknown\n\
        Known mesh types: {}".format(BaseMeshType, known_basemesh_classes.keys()))

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
    BaseMeshType = type(DM)
    if BaseMeshType in known_basemesh_classes.keys():
        class TopoMeshClass(known_basemesh_classes[BaseMeshType], _TopoMeshClass):
            def __init__(self, dm, *args, **kwargs):
                known_basemesh_classes[BaseMeshType].__init__(self, dm, *args, **kwargs)
                _TopoMeshClass.__init__(self, *args, **kwargs)
                # super(TopoMeshClass, self).__init__(dm, *args, **kwargs)

        return TopoMeshClass(DM, *args, **kwargs)

    else:
      raise TypeError("Mesh type {:s} unknown\n\
        Known mesh types: {}".format(BaseMeshType, known_basemesh_classes.keys()))

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
    BaseMeshType = type(DM)
    if BaseMeshType in known_basemesh_classes.keys():
        class SurfaceProcessMeshClass(known_basemesh_classes[BaseMeshType], _TopoMeshClass, _SurfaceProcessMeshClass):
            def __init__(self, dm, *args, **kwargs):
                known_basemesh_classes[BaseMeshType].__init__(self, dm, *args, **kwargs)
                _TopoMeshClass.__init__(self, *args, **kwargs)
                _SurfaceProcessMeshClass.__init__(self, *args, **kwargs)
                # super(SurfaceProcessMeshClass, self).__init__(dm, *args, **kwargs)

        return SurfaceProcessMeshClass(DM, *args, **kwargs)

    else:
      raise TypeError("Mesh type {:s} unknown\n\
        Known mesh types: {}".format(BaseMeshType, known_basemesh_classes.keys()))

    return
