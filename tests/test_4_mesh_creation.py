
import pytest
import numpy as np
import numpy.testing as npt
import quagmire
from quagmire.tools import meshtools
from mpi4py import MPI
from petsc4py import PETSc
comm = MPI.COMM_WORLD

from conftest import load_triangulated_mesh
from conftest import load_triangulated_spherical_mesh


def test_DMPlex_creation(load_triangulated_mesh):
    x = load_triangulated_mesh['x']
    y = load_triangulated_mesh['y']
    simplices = load_triangulated_mesh['simplices']
    DM = meshtools.create_DMPlex(x, y, simplices)
    coords = DM.getCoordinatesLocal().array.reshape(-1,2)

    if comm.size == 1:
        npt.assert_allclose(coords[:,0], x, err_msg="DMPlex x coordinates are not identical to input x coordinates")
        npt.assert_allclose(coords[:,1], y, err_msg="DMPlex y coordinates are not identical to input y coordinates")


def test_DMPlex_creation_from_points(load_triangulated_mesh):
    x = load_triangulated_mesh['x']
    y = load_triangulated_mesh['y']
    simplices = load_triangulated_mesh['simplices']
    DM = meshtools.create_DMPlex_from_points(x, y)
    coords = DM.getCoordinatesLocal().array.reshape(-1,2)

    if comm.size == 1:
        npt.assert_allclose(coords[:,0], x, err_msg="DMPlex x coordinates are not identical to input x coordinates")
        npt.assert_allclose(coords[:,1], y, err_msg="DMPlex y coordinates are not identical to input y coordinates")


def test_DMPlex_refinement(load_triangulated_mesh):
    x = load_triangulated_mesh['x']
    y = load_triangulated_mesh['y']
    simplices = load_triangulated_mesh['simplices']
    DM = meshtools.create_DMPlex(x, y, simplices)

    DM_r1 = meshtools.refine_DM(DM, refinement_levels=1)
    DM_r2 = meshtools.refine_DM(DM, refinement_levels=2)
    DM_r3 = meshtools.refine_DM(DM, refinement_levels=3)

    v1 = DM_r1.createLocalVector().getSize()
    v2 = DM_r2.createLocalVector().getSize()
    v3 = DM_r3.createLocalVector().getSize()

    assert v1 < v2 < v3, "Mesh refinement is not working"


def test_DMPlex_spherical_creation(load_triangulated_spherical_mesh):
    lons = load_triangulated_spherical_mesh['lons']
    lats = load_triangulated_spherical_mesh['lats']
    simplices = load_triangulated_spherical_mesh['simplices']
    DM = meshtools.create_spherical_DMPlex(lons, lats, simplices)


def test_DMPlex_creation_from_spherical_points(load_triangulated_spherical_mesh):
    lons = load_triangulated_spherical_mesh['lons']
    lats = load_triangulated_spherical_mesh['lats']
    simplices = load_triangulated_spherical_mesh['simplices']
    DM = meshtools.create_DMPlex_from_spherical_points(lons, lats)


# def test_DMPlex_creation_from_box():
#     minX, maxX = -5., 5.
#     minY, maxY = -5., 5.
#     resX, resY = 0.1, 0.1
#     DM = meshtools.create_DMPlex_from_box(minX, maxX, minY, maxY, resX, resY, refinement=None)
#     coords = DM.getCoordinatesLocal().array.reshape(-1,2)

#     if comm.size == 1:
#         assert coords.size >= 4, "Mesh creation from bounding box failed"


def test_DMDA_creation():
    minX, maxX = -5., 5.
    minY, maxY = -5., 5.
    Nx, Ny = 10, 10
    DM = meshtools.create_DMDA(minX, maxX, minY, maxY, Nx, Ny)
    coords = DM.getCoordinatesLocal().array.reshape(-1,2)

    # create expected grid coordinates
    xcoords = np.linspace(minX, maxX, Nx)
    ycoords = np.linspace(minY, maxY, Ny)
    xq, yq = np.meshgrid(xcoords, ycoords)
    coords0 = np.column_stack([xq.ravel(), yq.ravel()])

    if comm.size == 1:
        err_msg = "Coordinates are not a regularly-spaced grid"
        npt.assert_allclose(coords, coords0, err_msg=err_msg)


def test_mesh_improvement(load_triangulated_mesh):
    from quagmire import QuagMesh
    from quagmire.tools import lloyd_mesh_improvement

    x = np.array([0., 0., 1., 1.])
    y = np.array([0., 1., 0., 1.])

    x = np.hstack([x, 0.5*np.random.random(size=100)])
    y = np.hstack([y, 0.5*np.random.random(size=100)])


    DM = meshtools.create_DMPlex_from_points(x, y)
    mesh = QuagMesh(DM)

    bmask = mesh.bmask.copy()

    # perturb with Lloyd's mesh improvement algorithm
    x1, y1 = lloyd_mesh_improvement(x, y, bmask, 3)
    DM1 = meshtools.create_DMPlex_from_points(x1, y1)
    mesh1 = QuagMesh(DM1)

    mesh_equant = mesh.neighbour_cloud_distances.mean(axis=1) / ( np.sqrt(mesh.area))
    mesh1_equant = mesh1.neighbour_cloud_distances.mean(axis=1) / ( np.sqrt(mesh1.area))

    mask_bbox = np.ones(mesh.npoints, dtype=bool)
    mask_bbox[x1 < x.min()] = False
    mask_bbox[x1 > x.max()] = False
    mask_bbox[y1 < y.min()] = False
    mask_bbox[y1 > y.max()] = False

    mesh_equant = mesh_equant[mask_bbox]
    mesh1_equant = mesh1_equant[mask_bbox]

    assert np.std(mesh1_equant) < np.std(mesh_equant), "Mesh points are not more evenly spaced than previously"


# This fails in conda (there is no PETSc build with hdf5)

def test_petsc_save_hdf5():
    dm = PETSc.DMDA().create(2, sizes=(10, 10), stencil_width=1)
    ViewHDF5 = PETSc.Viewer()
    ViewHDF5.createHDF5('test.h5', mode='w')
    ViewHDF5.view(obj=dm)
    ViewHDF5 = None

def test_dmplex_save_hdf5():
    DIM = 2
    CELLS = [[0, 1, 3], [1, 3, 4], [1, 2, 4], [2, 4, 5],
             [3, 4, 6], [4, 6, 7], [4, 5, 7], [5, 7, 8]]
    COORDS = [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
              [0.0, 0.5], [0.5, 0.5], [1.0, 0.5],
              [0.0, 1.0], [0.5, 1.0], [1.0, 1.0]]
    DOFS = [1, 0, 0]


    dm = PETSc.DMPlex().createFromCellList(DIM, CELLS, COORDS)

    ViewHDF5 = PETSc.Viewer()
    ViewHDF5.createHDF5('test.h5', mode='w')
    ViewHDF5.view(obj=dm)
    ViewHDF5 = None

def test_mesh_save_to_hdf5(load_triangulated_mesh):
    import petsc4py

    x = load_triangulated_mesh['x']
    y = load_triangulated_mesh['y']
    simplices = load_triangulated_mesh['simplices']

    dm = meshtools.create_DMPlex(x, y, simplices)

    # save to hdf5 file
    #meshtools.save_DM_to_hdf5(DM, "test_mesh.h5")
    ViewHDF5 = PETSc.Viewer()
    ViewHDF5.createHDF5('test.h5', mode='w')
    ViewHDF5.view(obj=dm)
    ViewHDF5 = None


# This fails in conda (we need our own PETSc build with hdf5)

def test_mesh_load_from_hdf5():
    from quagmire import QuagMesh
    import petsc4py

    DM = meshtools.create_DMPlex_from_hdf5("test_mesh.h5")

    mesh = QuagMesh(DM)
    assert mesh.npoints > 0, "mesh could not be successfully loaded"

 