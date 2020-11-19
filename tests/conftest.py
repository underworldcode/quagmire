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

    lons = np.degrees(sm.lons)
    lats = np.degrees(sm.lats)

    # return as a dictionary
    mesh_dict = {'lons': lons, 'lats': lats, 'simplices': sm.simplices}
    return mesh_dict


@pytest.fixture(scope="module")
def load_triangulated_spherical_mesh_DM():
    import stripy

    minX, maxX = -5.0, 5.0
    minY, maxY = -5.0, 5.0
    spacingX = 0.1
    spacingY = 0.1

    # construct an elliptical mesh
    lons, lats, bmask = meshtools.generate_elliptical_points(minX, maxX, minY, maxY, spacingX, spacingY, 1500, 200)

    return meshtools.create_DMPlex_from_spherical_points(lons, lats, bmask)


@pytest.fixture(scope="module")
def load_pixelated_mesh_DM():
    minX, maxX = -5.0, 5.0
    minY, maxY = -5.0, 5.0
    resX, resY = 50, 50
    return meshtools.create_DMDA(minX, maxX, minY, maxY, resX, resY)


@pytest.fixture(scope="module", params=["PixMesh", "TriMesh", "sTriMesh"])
def load_multi_mesh_DM(request, load_pixelated_mesh_DM, load_triangulated_mesh_DM, load_triangulated_spherical_mesh_DM):
    
    DM_dict = {"PixMesh": load_pixelated_mesh_DM, \
               "TriMesh": load_triangulated_mesh_DM, \
              "sTriMesh": load_triangulated_spherical_mesh_DM }
    
    DM_type = request.param
    
    return DM_dict[DM_type]