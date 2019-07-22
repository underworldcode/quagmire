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