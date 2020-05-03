# -*- coding: utf-8 -*-

import pytest

# ==========================

from quagmire.tools import meshtools
from quagmire import FlatMesh
from quagmire.mesh import MeshVariable
import numpy as np

minX, maxX = -5.0, 5.0
minY, maxY = -5.0, 5.0,
dx, dy = 0.02, 0.02

x,y, bound = meshtools.generate_elliptical_points(minX, maxX, minY, maxY, dx, dy, 60000, 300)
DM = meshtools.create_DMPlex_from_points(x, y, bmask=bound)
mesh = FlatMesh(DM, downhill_neighbours=1)

def test_mesh_variable_instance():

    global mesh

    phi = mesh.add_variable(name="PHI(X,Y)")
    psi = mesh.add_variable(name="PSI(X,Y)")

    # check the variable data is available

    phi.data = np.cos(mesh.coords[:,0])**2.0 + np.sin(mesh.coords[:,0])**2.0
    psi.data = np.cos(mesh.coords[:,1])**2.0 + np.sin(mesh.coords[:,1])**2.0

    # check that evaluation / interpolation is possible

    assert np.isclose(phi.interpolate(0.01,1.0), 1.0)
    assert np.isclose(psi.interpolate(0.01,1.0), 1.0)

    assert np.isclose(phi.evaluate(0.01,1.0),    1.0)
    assert np.isclose(psi.evaluate(0.01,1.0),    1.0)

    ## This is the alternate interface to access the same code

    phi1 = MeshVariable(name="PHI(X,Y)", mesh=mesh)

    return

def test_mesh_variable_properties():

    global mesh


    ## Don't specify a name

    phi = mesh.add_variable()

    ## a locked array

    phi = mesh.add_variable(name="phi", locked=True)
    assert "RO" in phi.__repr__()


    try:
        phi.data[:] = 0.0
    except ValueError:
        ## This is the expected behaviour !
        pass


    phi.unlock()
    assert "RW" in phi.__repr__()

    phi.data = 1.0
    assert phi.data.mean() == 1.0


    return



def test_mesh_variable_derivative():

    # Functions we can differentiate easily

    phi = mesh.add_variable(name="PHI(X,Y)")
    psi = mesh.add_variable(name="PSI(X,Y)")


    phi.data = np.sin(mesh.coords[:,0])
    psi.data = np.cos(mesh.coords[:,0])

    assert(np.isclose(phi.fn_gradient[0].evaluate(0.0,0.0), psi.evaluate(0.0,0.0), rtol=1.0e-3))


    return


