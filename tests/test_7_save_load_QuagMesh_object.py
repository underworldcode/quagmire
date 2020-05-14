
import pytest
import numpy as np
import numpy.testing as npt
import h5py
import quagmire
import petsc4py
from quagmire.tools import io
from mpi4py import MPI
comm = MPI.COMM_WORLD

from conftest import load_triangulated_mesh_DM
from conftest import load_triangulated_spherical_mesh_DM
from conftest import load_pixelated_mesh_DM


def test_save_QuagMesh_trimesh_object(load_triangulated_mesh_DM):
    mesh = quagmire.QuagMesh(load_triangulated_mesh_DM)
    mesh.topography.unlock()
    mesh.topography.data = 1
    mesh.topography.lock()

    try:
        mesh.save_quagmire_project("my_trimesh_project.h5")

    except petsc4py.PETSc.Error:
        print("This error means that PETSc was not installed with hdf5")

    else:
        with h5py.File('my_trimesh_project.h5', 'r') as h5:
            keys = list(h5.keys())

        assert 'quagmire' in keys, "quagmire mesh info not found in HDF5 file"
        assert 'fields' in keys, "no fields are found in the HDF5 file"


        # load mesh object in again
        mesh = io.load_quagmire_project("my_trimesh_project.h5")
        assert mesh.id.startswith("trimesh"), "could not load trimesh object"
        assert int(round(mesh.topography.sum())) == mesh.npoints, "could not load topography MeshVariable"


def test_save_QuagMesh_strimesh_object(load_triangulated_spherical_mesh_DM):
    mesh = quagmire.QuagMesh(load_triangulated_spherical_mesh_DM)
    mesh.topography.unlock()
    mesh.topography.data = 1
    mesh.topography.lock()

    try:
        mesh.save_quagmire_project("my_strimesh_project.h5")

    except petsc4py.PETSc.Error:
        print("This error means that PETSc was not installed with hdf5")

    else:
        with h5py.File('my_strimesh_project.h5', 'r') as h5:
            keys = list(h5.keys())

        assert 'quagmire' in keys, "quagmire mesh info not found in HDF5 file"
        assert 'fields' in keys, "no fields are found in the HDF5 file"


        # load mesh object in again
        mesh = io.load_quagmire_project("my_strimesh_project.h5")
        assert mesh.id.startswith("strimesh"), "could not load trimesh object"
        assert int(round(mesh.topography.sum())) == mesh.npoints, "could not load topography MeshVariable"


def test_save_QuagMesh_pixmesh_object(load_pixelated_mesh_DM):
    mesh = quagmire.QuagMesh(load_pixelated_mesh_DM)
    mesh.topography.unlock()
    mesh.topography.data = 1
    mesh.topography.lock()

    try:
        mesh.save_quagmire_project("my_pixmesh_project.h5")

    except petsc4py.PETSc.Error:
        print("This error means that PETSc was not installed with hdf5")

    else:
        with h5py.File('my_pixmesh_project.h5', 'r') as h5:
            keys = list(h5.keys())

        assert 'quagmire' in keys, "quagmire mesh info not found in HDF5 file"
        assert 'h(x,y)' in keys, "no fields are found in the HDF5 file"


        # load mesh object in again
        mesh = io.load_quagmire_project("my_pixmesh_project.h5")
        assert mesh.id.startswith("pixmesh"), "could not load trimesh object"
        assert int(round(mesh.topography.sum())) == mesh.npoints, "could not load topography MeshVariable"    
