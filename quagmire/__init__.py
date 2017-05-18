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

import tools

known_basemesh_classes = {type(_PETSc.DMDA())   : _PixMesh,
                          type(_PETSc.DMPlex()) : _TriMesh}


def FlatMesh(DM):
    BaseMeshType = type(DM)
    if BaseMeshType in known_basemesh_classes.keys():
        class FlatMeshClass(known_basemesh_classes[BaseMeshType]):
            def __init__(self, dm):
                known_basemesh_classes[BaseMeshType].__init__(self, dm)
                # super(FlatMeshClass, self).__init__(dm)

        return FlatMeshClass(DM)

    else:
        print "Warning !! Mesh type {:s} unknown".format(BaseMeshType)
        print "Known mesh types: {}".format(known_mesh_classes.keys())

    return

def TopoMesh(DM):
    BaseMeshType = type(DM)
    if BaseMeshType in known_basemesh_classes.keys():
        class TopoMeshClass(known_basemesh_classes[BaseMeshType], _TopoMeshClass):
            def __init__(self, dm):
                known_basemesh_classes[BaseMeshType].__init__(self, dm)
                _TopoMeshClass.__init__(self)
                # super(TopoMeshClass, self).__init__(dm)

        return TopoMeshClass(DM)

    else:
        print "Warning !! Mesh type {:s} unknown".format(BaseMeshType)
        print "Known mesh types: {}".format(known_mesh_classes.keys())


    return

def SurfaceProcessMesh(DM):
    BaseMeshType = type(DM)
    if BaseMeshType in known_basemesh_classes.keys():
        class SurfaceProcessMeshClass(known_basemesh_classes[BaseMeshType], _TopoMeshClass, _SurfaceProcessMeshClass):
            def __init__(self, dm):
                known_basemesh_classes[BaseMeshType].__init__(self, dm)
                _TopoMeshClass.__init__(self)
                _SurfaceProcessMeshClass.__init__(self)
                # super(SurfaceProcessMeshClass, self).__init__(dm)

        return SurfaceProcessMeshClass(DM)

    else:
        print "Warning !! Mesh type {:s} unknown".format(BaseMeshType)
        print "Known mesh types: {}".format(known_mesh_classes.keys())

    return
