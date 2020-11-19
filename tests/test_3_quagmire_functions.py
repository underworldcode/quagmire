# -*- coding: utf-8 -*-

import pytest

# ==========================

# This has already been tested by now

from quagmire.tools import meshtools
from quagmire import QuagMesh
from quagmire.mesh import MeshVariable
from quagmire import function as fn
import numpy as np

from conftest import load_multi_mesh_DM as DM

def test_parameters():

    ## Assignment

    A = fn.parameter(10.0)


    ## And this is how to update A

    A.value = 10.0
    assert fn.math.exp(A).evaluate([0.0,0.0]) == np.exp(10.0)

    ## This works too ... and note the floating point conversion

    A(11)
    assert fn.math.exp(A).evaluate([0.0,0.0]) == np.exp(11.0)

    ## More complicated examples
    assert (fn.math.sin(A)**2.0 + fn.math.cos(A)**2.0).evaluate([0.0,0.0]) == 1.0


def test_fn_description():


    A = fn.parameter(10.0)
    B = fn.parameter(3)


    ## These will flush out any changes in the interface

    assert A.description == "10"
    assert B.description == "3"

    assert (fn.math.sin(A)+fn.math.cos(B)).description == "sin(10) + cos(3)"
    assert (fn.math.sin(A)**2.0 + fn.math.cos(A)**2.0).description == "sin(10)^2 + cos(10)^2"


def test_fn_description_derivatives(DM):

    mesh = QuagMesh(DM, down_neighbours=1)

    PHI = mesh.add_variable(name="PHI")

    description = PHI.derivative(0).description
    assert description == 'grad(PHI)|x0'

    description = PHI.derivative(1).description
    assert description == 'grad(PHI)|x1'



def test_fn_mesh_variables(DM):

    mesh = QuagMesh(DM, down_neighbours=2)

    height = mesh.add_variable(name="h(X,Y)")
    height.data = np.ones(mesh.npoints)
    height.lock()

    h_scale  = fn.parameter(2.0)
    h_offset = fn.parameter(1.0)

    scaled_height = height * h_scale + h_offset

    assert scaled_height.evaluate(mesh).max() == 3.0

    h_offset.value = 10.0
    height.unlock()

    assert scaled_height.evaluate(mesh).max() == 12.0


def test_fn_first_derivative(DM):

    mesh = QuagMesh(DM, down_neighbours=2)

    fn_xcoord = fn.misc.coord(dirn=0)
    phi = fn.math.sin(fn_xcoord)
    psi = fn.math.cos(fn_xcoord)

    assert(np.isclose(phi.derivative(0).evaluate([0.0,0.0]), psi.evaluate([0.0,0.0]), rtol=0.01))

def test_fn_second_derivative(DM):

    mesh = QuagMesh(DM, down_neighbours=2)

    fn_xcoord = fn.misc.coord(dirn=0)
    phi = fn.math.sin(fn_xcoord)
    psi = fn.math.cos(fn_xcoord)

    assert(np.isclose(phi.derivative(0).derivative(0).evaluate([0.0,0.0]), psi.derivative(0).evaluate([0.0,0.0]), rtol=0.05))


def test_fn_math(DM):

    mesh = QuagMesh(DM, down_neighbours=2)

    psi = mesh.add_variable(name="PSI")
    psi.data = mesh.coords[:,0]
    dx, dy = psi.grad()

    # fn.math.div(dx, dy).evaluate(mesh)
    # fn.math.curl(dx, dy).evaluate(mesh)
    fn.math.hypot(dx, dy).evaluate(mesh)
    fn.math.arctan2(dx, dy).evaluate(mesh)


def test_function_mul(DM):
    mesh = QuagMesh(DM, downhill_neighbours=1)
    func1 = fn.LazyEvaluation().convert(4.0)
    func2 = fn.LazyEvaluation().convert(1.0)
    func = func1 * func2
    assert(np.all(func.evaluate(mesh) == 4.0))

def test_function_rmul(DM):
    mesh = QuagMesh(DM, downhill_neighbours=1)
    func1 = fn.LazyEvaluation().convert(4.0)
    func2 = fn.LazyEvaluation().convert(1.0)
    func = func2 * func1
    assert(np.all(func.evaluate(mesh) == 4.0))

def test_function_add(DM):
    mesh = QuagMesh(DM, downhill_neighbours=1)
    func1 = fn.LazyEvaluation().convert(4.0)
    func2 = fn.LazyEvaluation().convert(1.0)
    func = func2 + func1
    assert(np.all(func.evaluate(mesh) == 5.0))

def test_function_radd(DM):
    mesh = QuagMesh(DM, downhill_neighbours=1)
    func1 = fn.LazyEvaluation().convert(4.0)
    func2 = fn.LazyEvaluation().convert(1.0)
    func = func1 + func2
    assert(np.all(func.evaluate(mesh) == 5.0))

def test_function_truediv(DM):
    mesh = QuagMesh(DM, downhill_neighbours=1)
    func1 = fn.LazyEvaluation().convert(4.0)
    func2 = fn.LazyEvaluation().convert(1.0)
    func = func2 / func1
    assert(np.all(func.evaluate(mesh) == 0.25))

def test_function_sub(DM):
    mesh = QuagMesh(DM, downhill_neighbours=1)
    func1 = fn.LazyEvaluation().convert(4.0)
    func2 = fn.LazyEvaluation().convert(1.0)
    func = func1 - func2
    assert(np.all(func.evaluate(mesh) == 3.0))

def test_function_rsub(DM):
    mesh = QuagMesh(DM, downhill_neighbours=1)
    func1 = fn.LazyEvaluation().convert(4.0)
    func2 = fn.LazyEvaluation().convert(1.0)
    func = func2 - func1
    assert(np.all(func.evaluate(mesh) == -3.0))

def test_function_neg(DM):
    mesh = QuagMesh(DM, downhill_neighbours=1)
    func1 = fn.LazyEvaluation().convert(4.0)
    func2 = fn.LazyEvaluation().convert(1.0)
    func = -func1
    assert(np.all(func.evaluate(mesh) == -4.0))

def test_function_pow(DM):
    mesh = QuagMesh(DM, downhill_neighbours=1)
    func1 = fn.LazyEvaluation().convert(4.0)
    func2 = fn.LazyEvaluation().convert(1.0)
    func = func1**2
    assert(np.all(func.evaluate(mesh) == 16.0))

def test_function_lt(DM):
    mesh = QuagMesh(DM, downhill_neighbours=1)
    func1 = fn.LazyEvaluation().convert(4.0)
    func2 = fn.LazyEvaluation().convert(1.0)
    func = func1 < func2
    assert(np.all(func.evaluate(mesh) == False))

def test_function_le(DM):
    mesh = QuagMesh(DM, downhill_neighbours=1)
    func1 = fn.LazyEvaluation().convert(4.0)
    func2 = fn.LazyEvaluation().convert(1.0)
    func = func1 <= 4.0
    assert(np.all(func.evaluate(mesh) == True))
    func = func2 <= 0.0
    assert(np.all(func.evaluate(mesh) == False))

def test_function_eq(DM):
    mesh = QuagMesh(DM, downhill_neighbours=1)
    func1 = fn.LazyEvaluation().convert(4.0)
    func2 = fn.LazyEvaluation().convert(1.0)
    func = func1 == func2
    assert(np.all(func.evaluate(mesh)) == False)
    func = func1 == 4.0
    assert(np.all(func.evaluate(mesh)) == True)

def test_function_ne(DM):
    mesh = QuagMesh(DM, downhill_neighbours=1)
    func1 = fn.LazyEvaluation().convert(4.0)
    func2 = fn.LazyEvaluation().convert(1.0)
    func = func1 != func2
    assert(np.all(func.evaluate(mesh)) == True)
    func = func1 != 1.0
    assert(np.all(func.evaluate(mesh)) == True)

def test_function_ge(DM):
    mesh = QuagMesh(DM, downhill_neighbours=1)
    func1 = fn.LazyEvaluation().convert(4.0)
    func2 = fn.LazyEvaluation().convert(1.0)
    func = func1 >= func2
    assert(np.all(func.evaluate(mesh)) == True)
    func = func1 >= 4.0
    assert(np.all(func.evaluate(mesh)) == True)
    func = func2 >= func1
    assert(np.all(func.evaluate(mesh)) == False)

def test_function_gt(DM):
    mesh = QuagMesh(DM, downhill_neighbours=1)
    func1 = fn.LazyEvaluation().convert(4.0)
    func2 = fn.LazyEvaluation().convert(1.0)
    func = func1 > func2
    assert(np.all(func.evaluate(mesh)) == True)
    func = func1 > 4.0
    assert(np.all(func.evaluate(mesh)) == False)
    func = func2 > func1
    assert(np.all(func.evaluate(mesh)) == False)