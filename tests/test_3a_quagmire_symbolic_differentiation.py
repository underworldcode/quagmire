# -*- coding: utf-8 -*-

import pytest

# ==========================

# This has already been tested by now

from quagmire.tools import meshtools
from quagmire import QuagMesh
from quagmire.mesh import MeshVariable
from quagmire import function as fn
import numpy as np
from numpy.random import random

from conftest import load_multi_mesh_DM as DM

def test_parameters():

    ## Assignment

    A = fn.parameter(10.0)
    X = fn.misc.coord(0)

    ## And this is how to update A

    assert(A.sderivative(0)).evaluate(random(), random()) == 0


def test_fn_description():

    X = fn.misc.coord(0)
    A = fn.parameter(10.0)
    B = fn.math.sin(A * X)

    ## These will flush out any changes in the interface

    assert B.description == 'sin((10.0)*(X))'
    assert B.sderivative(0).description == '(cos((10.0)*(X)))*(((0.0)*(X))+((10.0)*(1.0)))'
    assert B.sderivative(1).description == '(cos((10.0)*(X)))*(((0.0)*(X))+((10.0)*(0.0)))'

def test_fn_sderivative_values():

    X = fn.misc.coord(0)
    A = fn.parameter(10.0)
    B = fn.math.sin(A * X)

    ## Derivatives of quantities dependent on coordinate variables

    assert B.sderivative(1).evaluate(random(), random()) == 0.0
    assert B.sderivative(0).evaluate(random(), random()) != 0.0
    assert np.fabs(B.sderivative(0).evaluate(1.0,0.0) + 8.390715290764524) < 1.0e-10

    return 


def test_fn_sderivatives_defined():

    A  = fn.parameter(1.0)
    B  = fn.parameter(2.0)
    S  = fn.math.sin(A)
    T  = fn.math.sin(S)
    X  = fn.misc.coord(0)
    Y  = fn.misc.coord(1)
    SX = fn.math.sin(X)
    CX = fn.math.cos(X)
    X2 = X**2
    XY = X*Y
    SX2 = fn.math.sin(X**2)
    CX2 = fn.math.cos(X**2)

    assert np.fabs((SX**2 + CX**2).evaluate(random(),random()) - 1.0)  < 1.0e-8
    assert SX.sderivative(0).evaluate(0.0,0.0) == 1.0 

    assert CX2.sderivative(0).description == '(-(sin((X)**(2.0))))*(((2.0)*(1.0))*((X)**((2.0)-(1.0))))'
    assert CX2.sderivative(1).description == '(-(sin((X)**(2.0))))*(((2.0)*(0.0))*((X)**((2.0)-(1.0))))'

    assert fn.math.cos(X).sderivative(1)
    assert fn.math.cos(Y).sderivative(1)
    assert fn.math.sin(Y).sderivative(1)
    assert fn.math.tan(Y).sderivative(1)
    assert fn.math.sinh(Y).sderivative(1)
    assert fn.math.cosh(Y).sderivative(1)
    assert fn.math.arccosh(Y).sderivative(1)
    assert fn.math.arcsinh(Y).sderivative(1)
    assert fn.math.arctanh(Y).sderivative(1)

    ### Continue to exhaustion here 

    return

def test_fn_sderivative_meshvar(DM):

    mesh = QuagMesh(DM, down_neighbours=1)
    PHI = mesh.add_variable(name="PHI")

    assert PHI.sderivative(0)
    assert PHI.sderivative(1)

    A  = fn.parameter(1.0)
    B  = fn.parameter(2.0)
    S  = fn.math.sin(A)
    T  = fn.math.sin(S)
    X  = fn.misc.coord(0)
    Y  = fn.misc.coord(1)
    P = PHI * X**2 

    assert fn.math.sin(P).sderivative(0).description == '(cos((PHI)*((X)**(2.0))))*(((d(PHI)/dX)*((X)**(2.0)))+((PHI)*(((2.0)*(1.0))*((X)**((2.0)-(1.0))))))'

    return


