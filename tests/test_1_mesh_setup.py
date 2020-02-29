# -*- coding: utf-8 -*-

import pytest

# ==========================

DM = None


## First of all test the "pixmesh"

def test_DMDA():
    from quagmire.tools import meshtools

    global DM

    minX, maxX = -5.0, 5.0
    minY, maxY = -5.0, 5.0

    resX = 75
    resY = 75

    DM = meshtools.create_DMDA(minX, maxX, minY, maxY, resX, resY)

    assert DM.sizes == (75, 75)

    return


def test_flatmesh_from_DMDA():

    from quagmire import FlatMesh

    global DM

    mesh = FlatMesh(DM)
    assert mesh.sizes == ((5625, 5625), (5625, 5625))

    return

## Now test the triangulated mesh


def test_DMPlex():

    from quagmire.tools import meshtools

    global DM

    minX, maxX = -5.0, 5.0
    minY, maxY = -5.0, 5.0

    spacingX = 0.1
    spacingY = 0.1

    x, y, simplices = meshtools.elliptical_mesh(minX, maxX, minY, maxY, spacingX, spacingY)
    DM = meshtools.create_DMPlex(x, y, simplices)
    assert (DM.getCoordinates().array.shape) == (4880,)



def test_flatmesh_from_DMPlex():

    from quagmire import FlatMesh

    global DM

    mesh = FlatMesh(DM)
    assert mesh.sizes == ((2440, 2440), (2440, 2440))

    return

# We should test the Lloyd mesh improvement at this point
# TODO: test mesh improvement


## Test refinement of triangulated meshes (use existing mesh from previous test)

def test_DM_refinement():

    from quagmire.tools import meshtools

    minX, maxX = -5.0, 5.0
    minY, maxY = -5.0, 5.0

    spacingX = 0.5
    spacingY = 0.5

    x, y, simplices = meshtools.elliptical_mesh(minX, maxX, minY, maxY, spacingX, spacingY)
    DM = meshtools.create_DMPlex(x, y, simplices)

    # ToDo: Figure out this recent change
    # The following now fails with
    # E   petsc4py.PETSc.Error: error code 62
    # E   [0] DMRefine() line 1885 in .../interface/dm.c
    # E   [0] DMRefine_Plex() line 10496 in .../src/dm/impls/plex/plexrefine.c
    # E   [0] DMPlexRefineUniform_Internal() line 10204 in .../src/dm/impls/plex/plexrefine.c
    # E   [0] Invalid argument
    # E   [0] Mesh must be interpolated for regular refinement

    # DM_r1 = meshtools._refine_DM(DM, refinement_levels=1)
    # DM_r2 = meshtools._refine_DM(DM, refinement_levels=2)

    DM_r1 = meshtools.create_DMPlex(x, y, simplices, refinement_levels=1)
    DM_r2 = meshtools.create_DMPlex(x, y, simplices, refinement_levels=2)

    assert (DM.getCoordinates().array.shape)    == (182,)
    assert (DM_r1.getCoordinates().array.shape) == (686,)
    assert (DM_r2.getCoordinates().array.shape) == (2666,)

    return DM







