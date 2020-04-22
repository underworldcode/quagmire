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
The mesh module provides 3 fundamental spatial data structures

- `pixmesh.PixMesh`: for structured data on a regular grid.
- `trimesh.TriMesh`: for unstructured data in Cartesian coordinates.
- `strimesh.sTriMesh`: for unstructured data on the sphere.

Each of these data structures are built on top of a `PETSc DM` object
(created from `quagmire.tools.meshtools`) and implement the general functionality of:

- calculating spatial derivatives
- identifying node neighbour relationships
- interpolation / extrapolation
- smoothing operators
- importing and saving mesh information

"""

from .trimesh import TriMesh
from .pixmesh import PixMesh
from .strimesh import sTriMesh
from .basemesh import MeshVariable
from .basemesh import VectorMeshVariable


def check_object_is_a_q_mesh(mesh_object):
    """
    Is this object a `quagmire.mesh` of some kind?

    Parameters
    ----------
    mesh_object : object
        Checks if one of `trimesh.TriMesh`, `pixmesh.PixMesh`, `strimesh.sTriMesh`

    Returns
    -------
    statement : bool
        True or False
    """

    return isinstance(mesh_object, (TriMesh, PixMesh, sTriMesh))

def check_object_is_a_q_mesh_and_raise(mesh_object):
    """
    If this object is not a `quagmire.mesh` then raises a RuntimeError

    Parameters
    ----------
    mesh_object : object
        Checks if one of `trimesh.TriMesh`, `pixmesh.PixMesh`, `strimesh.sTriMesh`

    Returns
    -------
    statement : bool
        True or False
    """

    if not check_object_is_a_q_mesh(mesh_object):
        raise RuntimeError("Expecting a quagmire.mesh object")

    return True
