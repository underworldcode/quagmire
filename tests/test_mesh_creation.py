
import pytest
import numpy as np
import numpy.testing as npt
import quagmire
from quagmire.tools import meshtools
from mpi4py import MPI
comm = MPI.COMM_WORLD

from conftest import load_triangulated_mesh


def test_DMPlex_creation(load_triangulated_mesh):
    x = load_triangulated_mesh['x']
    y = load_triangulated_mesh['y']
    simplices = load_triangulated_mesh['simplices']
    DM = meshtools.create_DMPlex(x, y, simplices)
    coords = DM.getCoordinatesLocal().array.reshape(-1,2)

    if comm.size == 1:
        assert coords[:,0] == x, "DMPlex x coordinates are not identical to input x coordinates"
        assert coords[:,1] == y, "DMPlex y coordinates are not identical to input y coordinates"


def test_DMPlex_creation_from_points(load_triangulated_mesh):
    x = load_triangulated_mesh['x']
    y = load_triangulated_mesh['y']
    simplices = load_triangulated_mesh['simplices']
    DM = meshtools.create_DMPlex_from_points(x, y)

    if comm.size == 1:
        assert coords[:,0] == x, "DMPlex x coordinates are not identical to input x coordinates"
        assert coords[:,1] == y, "DMPlex y coordinates are not identical to input y coordinates"


def test_DMPlex_refinement(load_triangulated_mesh):
    x = load_triangulated_mesh['x']
    y = load_triangulated_mesh['y']
    simplices = load_triangulated_mesh['simplices']
    DM = meshtools.create_DMPlex(x, y, simplices)

    DM_r1 = meshtools.refine_DM(DM, refinement_levels=1)
    DM_r2 = meshtools.refine_DM(DM, refinement_levels=2)
    DM_r3 = meshtools.refine_DM(DM, refinement_levels=3)

    v1 = DM_r1.DM.createLocalVector().getSize()
    v2 = DM_r2.DM.createLocalVector().getSize()
    v3 = DM_r3.DM.createLocalVector().getSize()

    assert v1 < v2 < v3, "Mesh refinement is not working"


def test_DMPlex_creation_from_box():
    minX, maxX = -5., 5.
    minY, maxY = -5., 5.
    resX, resY = 0.1, 0.1
    DM = create_DMPlex_from_box(minX, maxX, minY, maxY, resX, resY, refinement=None)
    coords = DM.getCoordinatesLocal().array.reshape(-1,2)

    if comm.size == 1:
        assert coords.size >= 4, "Mesh creation from bounding box failed"


def test_DMDA_creation():
    minX, maxX = -5., 5.
    minY, maxY = -5., 5.
    resX, resY = 0.1, 0.1
    DM = create_DMDA(minX, maxX, minY, maxY, resX, resY)
    coords = DM.getCoordinatesLocal().array.reshape(-1,2)

    # create expected grid coordinates
    xcoords = np.arange(minX, maxX+resX, resX)
    ycoords = np.arange(minY, maxY+resY, resY)
    xq, yq = np.meshgrid(xcoords, ycoords)
    coords0 = np.column_stack([xq.ravel(), yq.ravel()])

    if comm.size == 1:
        err_msg = "Coordinates are not a regularly-spaced grid"
        npt.assert_allclose(coords, coords0, err_msg=err_msg)


def test_mesh_improvement(load_triangulated_mesh):
    from quagmire import FlatMesh

    x = load_triangulated_mesh['x']
    y = load_triangulated_mesh['y']
    simplices = load_triangulated_mesh['simplices']

    DM = meshtools.create_DMPlex(x, y, simplices)
    mesh = FlatMesh(DM)

    bmask = mesh.bmask.copy()

    # perturb with Lloyd's mesh improvement algorithm
    x1, y1 = lloyd_mesh_improvement(x, y, bmask, 3)
    DM1 = meshtools.create_DMPlex_from_points(x1, y1)
    mesh1 = FlatMesh(DM)

    mesh_equant = mesh.neighbour_cloud_distances.mean(axis=1) / ( np.sqrt(mesh.area))
    mesh1_equant = mesh1.neighbour_cloud_distances.mean(axis=1) / ( np.sqrt(mesh1.area))

    assert np.std(mesh_equant1) < np.std(mesh_equant), "Mesh points are not more evenly spaced than previously"


def test_mesh_save_to_hdf5(load_triangulated_mesh):
    x = load_triangulated_mesh['x']
    y = load_triangulated_mesh['y']
    simplices = load_triangulated_mesh['simplices']

    DM = meshtools.create_DMPlex(x, y, simplices)

    # save to hdf5 file
    save_DM_to_hdf5(DM, "test_mesh.h5")


def test_mesh_load_from_hdf5():
    from quagmire import FlatMesh

    try:
        DM = create_DMPlex_from_hdf5("test_mesh.h5")
    except:
        DM = create_DMPlex_from_hdf5("tests/test_mesh.h5")

    mesh = FlatMesh(DM)
    assert mesh.npoints > 0, "mesh could not be successfully loaded"
