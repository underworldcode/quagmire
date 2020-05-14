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
from .meshtools import create_DMDA as _create_DMDA
from mpi4py import MPI as _MPI
_comm = _MPI.COMM_WORLD

def load_quagmire_project(filename):
    """
    Load a Quagmire project from a HDF5 file.

    Detects which mesh object was saved, i.e.

    - `quagmire.FlatMesh`
    - `quagmire.TopoMesh`
    - `quagmire.SurfaceProcessMesh`

    and rebuilds all data structures onto the mesh object.

    Parameters
    ----------
    filename : str
        path of the HDF5 from which to load the Quagmire project.

    Returns
    -------
    mesh : object
        Quagmire mesh object. One of:

        - `quagmire.FlatMesh`
        - `quagmire.TopoMesh`
        - `quagmire.SurfaceProcessMesh`
    """

    from quagmire import QuagMesh
    import h5py

    filename = str(filename)
    if not filename.endswith('.h5'):
        filename += '.h5'

    with h5py.File(filename, mode='r', driver='mpio', comm=_comm) as h5:
        quag = h5['quagmire']
        verbose         = quag.attrs['verbose']
        mesh_id         = quag.attrs['id']
        radius          = quag.attrs['radius']
        down_neighbours = quag.attrs['downhill_neighbours']
        topo_modified   = quag.attrs['topography_modified']

        if mesh_id.startswith('pixmesh'):
            geom = h5['geometry']
            minX, maxX = geom.attrs['minX'], geom.attrs['maxX']
            minY, maxY = geom.attrs['minY'], geom.attrs['maxY']
            resX, resY = geom.attrs['resX'], geom.attrs['resY']


    # create DM object
    if mesh_id.startswith("pixmesh"):
        DM = _create_DMDA(minX, maxX, minY, maxY, resX, resY)
    else:
        DM = _create_DMPlex_from_hdf5(filename)


    mesh = QuagMesh(DM, downhill_neighbours=down_neighbours, verbose=verbose)
    mesh.__id = mesh_id
    mesh._topography_modified_count = topo_modified
    if mesh.id.startswith('strimesh'):    
        if not radius:
            radius_meshVariable = mesh.add_variable("radius")
            radius_meshVariable.load(filename)
            radius = radius_meshVariable.data
        mesh.radius = radius

    mesh.topography.unlock()
    mesh.topography.load(filename)
    mesh.topography.lock()

    if mesh._topography_modified_count > 0:
        # this should trigger a rebuild of downhill matrices
        mesh.downhill_neighbours = down_neighbours


    print("Quagmire project is successfully rebuilt on {}".format(mesh.id))

    return mesh


def load_saved_MeshVariables(mesh, filename, mesh_variable_list):
    """
    Loads all mesh variables saved onto the HDF5 file.

    Parameters
    ----------
    mesh : object
        Quagmire mesh object
    filename : str
        path of the HDF5 from which to load the mesh variables.
    mesh_variable_list : list
        list of mesh variables to load from the HDF5 file
        each entry should be a string.
    """

    MeshVariable_list = []
    for field_name in mesh_variable_list:
        mvar = mesh.add_variable(field_name)
        mvar.load(filename)
        MeshVariable_list.append(mvar)

    return MeshVariable_list
