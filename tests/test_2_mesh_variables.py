# -*- coding: utf-8 -*-

import pytest

# ==========================

from quagmire.tools import meshtools
from quagmire import QuagMesh
from quagmire.mesh import MeshVariable
import numpy as np

from conftest import load_multi_mesh_DM as DM


def test_mesh_variable_instance(DM):

    mesh = QuagMesh(DM, downhill_neighbours=1)

    phi = mesh.add_variable(name="PHI(X,Y)")
    psi = mesh.add_variable(name="PSI(X,Y)")

    # check the variable data is available

    phi.data = np.cos(mesh.coords[:,0])**2.0 + np.sin(mesh.coords[:,0])**2.0
    psi.data = np.cos(mesh.coords[:,1])**2.0 + np.sin(mesh.coords[:,1])**2.0

    # check that evaluation / interpolation is possible

    assert np.isclose(phi.interpolate(0.01,1.0), 1.0)
    assert np.isclose(psi.interpolate(0.01,1.0), 1.0)

    assert np.isclose(phi.evaluate([0.01,1.0]),    1.0)
    assert np.isclose(psi.evaluate([0.01,1.0]),    1.0)

    ## This is the alternate interface to access the same code

    phi1 = MeshVariable(name="PHI(X,Y)", mesh=mesh)

    return


def test_mesh_variable_evaluation_1(DM):
    mesh = QuagMesh(DM, downhill_neighbours=1)
    meshVar = mesh.add_variable(name="meshVar")
    results = meshVar.evaluate()
    assert(results.size == mesh.data[:, 0].size)

def test_mesh_variable_evaluation_2(DM):
    mesh = QuagMesh(DM, downhill_neighbours=1)
    meshVar = mesh.add_variable(name="meshVar")
    results = meshVar.evaluate([0.,0.])
    assert(results.size == 1)

def test_mesh_variable_evaluation_3(DM):
    mesh = QuagMesh(DM, downhill_neighbours=1)
    meshVar = mesh.add_variable(name="meshVar")
    results = meshVar.evaluate(mesh)
    assert(results.size == mesh.data[:, 0].size)

def test_mesh_variable_evaluation_4(DM):
    mesh = QuagMesh(DM, downhill_neighbours=1)
    mesh2 = QuagMesh(DM, downhill_neighbours=1)
    meshVar = mesh.add_variable(name="meshVar")
    results = meshVar.evaluate(mesh2)
    assert(results.size == mesh.data[:, 0].size)

def test_mesh_variable_properties(DM):

    mesh = QuagMesh(DM, downhill_neighbours=1)

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



def test_mesh_variable_derivative(DM):

    mesh = QuagMesh(DM, downhill_neighbours=1)

    # Functions we can differentiate easily

    phi = mesh.add_variable(name="PHI(X,Y)")
    psi = mesh.add_variable(name="PSI(X,Y)")


    phi.data = np.sin(mesh.coords[:,0])
    psi.data = np.cos(mesh.coords[:,0])

    assert(np.isclose(phi.fn_gradient[0].evaluate([0.0,0.0]), psi.evaluate([0.0,0.0]), rtol=0.01))


    return


