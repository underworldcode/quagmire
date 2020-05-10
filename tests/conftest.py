import pytest
import numpy as np
import quagmire
from quagmire.tools import meshtools

@pytest.fixture(scope="module")
def load_triangulated_mesh():
    minX, maxX = -5.0, 5.0
    minY, maxY = -5.0, 5.0
    spacingX = 0.1
    spacingY = 0.1

    # construct an elliptical mesh
    x, y, simplices = meshtools.elliptical_mesh(minX, maxX, minY, maxY, spacingX, spacingY)

    # return as a dictionary
    mesh_dict = {'x': x, 'y': y, 'simplices': simplices}
    return mesh_dict


@pytest.fixture(scope="module")
def load_triangulated_mesh_DM():
    minX, maxX = -5.0, 5.0
    minY, maxY = -5.0, 5.0
    spacingX = 0.1
    spacingY = 0.1

    # construct an elliptical mesh
    x, y, simplices = meshtools.elliptical_mesh(minX, maxX, minY, maxY, spacingX, spacingY)

    return meshtools.create_DMPlex(x, y, simplices)


@pytest.fixture(scope="module")
def load_triangulated_spherical_mesh():
    import stripy

    # construct a spherical stripy mesh
    sm = stripy.spherical_meshes.icosahedral_mesh(refinement_levels=3)

    # return as a dictionary
    mesh_dict = {'lons': sm.lons, 'lats': sm.lats, 'simplices': sm.simplices}
    return mesh_dict


@pytest.fixture(scope="module")
def load_triangulated_spherical_mesh_DM():
    import stripy

    # construct a spherical stripy mesh
    sm = stripy.spherical_meshes.icosahedral_mesh(refinement_levels=3)

    return meshtools.create_spherical_DMPlex(sm.lons, sm.lats, sm.simplices)


@pytest.fixture(scope="module")
def load_pixelated_mesh_DM():
    minX, maxX = 0., 1.
    minY, maxY = 0., 1.
    resX, resY = 50, 50
    return meshtools.create_DMDA(minX, maxX, minY, maxY, resX, resY)