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

from .meshtools import create_DMPlex_from_hdf5 as _create_DMPlex_from_hdf5
from mpi4py import MPI as _MPI
_comm = _MPI.COMM_WORLD

def load_quagmire_project(filename):

    from quagmire import FlatMesh as _FlatMesh
    from quagmire import TopoMesh as _TopoMesh
    from quagmire import SurfaceProcessMesh as _SurfaceProcessMesh
    import h5py

    filename = str(filename)

    DM = _create_DMPlex_from_hdf5(filename)


    known_mesh_classes = {'FlatMesh': _FlatMesh, \
                          'TopoMesh': _TopoMesh, \
                          'SurfaceProcessMesh' : _SurfaceProcessMesh}


    with h5py.File(filename, mode='r', driver='mpio', comm=_comm) as h5:
        quag = h5['quagmire']
        verbose         = quag.attrs['verbose']
        mesh_type       = quag.attrs['mesh_type']
        mesh_id         = quag.attrs['id']
        radius          = quag.attrs['radius']
        down_neighbours = quag.attrs['downhill_neighbours']

        # are there any other fields in here?
        field_variable_list = []
        if 'fields' in h5:
            for field in h5['fields']:
                field_variable_list.append(field)


    BaseMeshClass = known_mesh_classes[mesh_type]

    mesh = BaseMeshClass(DM, verbose=verbose)
    mesh.__id = mesh_id
    if not radius:
        radius = mesh.add_variable("radius")
        radius.load(filename)
    mesh.radius = radius

    if mesh_type in ['TopoMesh', 'SurfaceProcessMesh']:
        mesh.topography.unlock()
        mesh.topography.load(filename)
        mesh.topography.lock()
        # this should trigger a rebuild of downhill matrices
        mesh.downhill_neighbours = down_neighbours


    print("Quagmire {} is successfully rebuilt!".format(mesh_type))
    if field_variable_list:
        print("Additional mesh variables are available to load:")
        for field in field_variable_list:
            print(" - {}".format(field))
        print("\nAdd a MeshVariable with the same name and load from this file.")
        print("Or load a tuple of MeshVariables using load_saved_MeshVariables.")

    return mesh


def load_saved_MeshVariables(mesh, filename):

    import h5py

    with h5py.File(filename, mode='r', driver='mpio', comm=_comm) as h5:
        if 'fields' in h5:
            field_variable_list = []
            for field in h5['fields']:
                field_variable_list.append(field)


    MeshVariable_list = []
    for field in field_variable_list:
        mvar = mesh.add_variable(field)
        mvar.load(filename)
        MeshVariable_list.append(mvar)

    return MeshVariable_list
