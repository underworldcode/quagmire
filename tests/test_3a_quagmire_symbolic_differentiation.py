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

## We want to be able to evaluate at arbitrary points in different formats
pointsa   = np.array([[0.0,0.0], [0.0,1.0], [0.0,2.0]])
pointst  = (0.1,0.1)
pointsl  = [0.1,0.1]
pointsl2 = [[0.0,0.0], [0.0,1.0], [0.0,2.0]]

def test_parameters():

    ## Assignment

    A = fn.parameter(10.0)
    X = fn.misc.coord(0)

    ## And this is how to update A

    assert(A.derivative(0)).evaluate(random(), random()) == 0

def test_arg_type_conversion():

    X = fn.misc.coord(0)
    A = fn.parameter(10.0)

    # Float / int should convert to parameter so these are equivalent
    B = fn.math.sin(A * X)
    C = fn.math.sin(10.0 * X)
    D = fn.math.sin(10 * X)

    assert B.description == C.description
    assert B.description == D.description

    # Should be true for +, - , /, **, etc 

    D1 = fn.math.cos(10+X)
    D2 = fn.math.cos(10-X)
    D3 = fn.math.cos(1.0/X)
    D4 = fn.math.cos(X/2.0)
    D5 = fn.math.cos(X**2.0)
    D6 = fn.math.cos(-X)

    # And in both directions 

    E1 = fn.math.cos(X+10)
    E2 = fn.math.cos(X-10)

    # This checks various other functions (these are templated so we are not really testing individual ones)
    F1 = fn.math.tan(X * 10)
    F2 = fn.math.arccos(X * 10)
    F3 = fn.math.arctan(X * 10)
    F4 = fn.math.sinh(X * 10)
    F5 = fn.math.cosh(X * 10)
    F6 = fn.math.tanh(X * 10)


def test_special_cases():

    A = fn.parameter(10.0)
    B = fn.parameter(0.0)
    C = fn.parameter(1.0)

    assert A * C is A 
    assert A * B is B # really this does return the B object although it could be any zero parameter object

    X = fn.misc.coord(0)

    return

def test_fn_description():

    X = fn.misc.coord(0)
    A = fn.parameter(10.0)
    B = fn.math.sin(A * X)

    ## These will flush out any changes in the interface

    assert B.description == 'sin(10*X)'
    assert B.derivative(0).description == 'cos(10*X)*10'
    assert B.derivative(1).description == '0'


def test_fn_derivative_values():

    X = fn.misc.coord(0)
    A = fn.parameter(10.0)
    B = fn.math.sin(A * X)

    ## Derivatives of quantities dependent on coordinate variables

    assert B.derivative(1).evaluate(random(), random()) == 0.0
    assert B.derivative(0).evaluate(random(), random()) != 0.0
    assert np.fabs(B.derivative(0).evaluate(1.0,0.0) + 8.390715290764524) < 1.0e-10

    return 


def test_fn_derivatives_defined():

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
    assert SX.derivative(0).evaluate(0.0,0.0) == 1.0 

    assert CX2.derivative(0).description == '-sin(X^2)*2*X'
    assert CX2.derivative(1).description == '0'

    ## Some random collection of things that should not throw errors

    assert fn.math.cos(X).derivative(1)
    assert fn.math.cos(Y).derivative(1)
    assert fn.math.sin(Y).derivative(1)
    assert fn.math.tan(Y).derivative(1)
    assert fn.math.sinh(Y).derivative(1)
    assert fn.math.cosh(Y).derivative(1)
    assert fn.math.arccosh(Y).derivative(1)
    assert fn.math.arcsinh(Y).derivative(1)
    assert fn.math.arctanh(Y).derivative(1)

    ### Continue to exhaustion here 

    return

def test_fn_derivative_meshvar(DM):

    mesh = QuagMesh(DM, down_neighbours=1)
    PHI = mesh.add_variable(name="PHI")

    assert PHI.derivative(0)
    assert PHI.derivative(1)

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

    P = PHI * X**2 

    PHI.data = (S * T * X * Y).evaluate(mesh)

    assert fn.math.sin(P).derivative(0).description == 'cos(PHI*X^2)*(d(PHI)/dX*X^2 + PHI*2*X)'

    ## Should be able to evaluate at mesh, points in the form of tuple, list or array, points as 2 args
    ## Check the raw variable and the derivative

    assert fn.math.sin(P).evaluate((0.0,0.0)).shape == (1,)
    assert fn.math.sin(P).evaluate(pointst).shape   == (1,)
    assert fn.math.sin(P).evaluate(pointsl).shape   == (1,)
    assert fn.math.sin(P).evaluate(pointsl2).shape  == (3,)
    assert fn.math.sin(P).evaluate(pointsa).shape   == (3,)
    assert fn.math.sin(PHI).evaluate(0.0,0.0).shape == (1,)

    ## Check values 

    assert np.any(np.isclose(fn.math.cos(PHI).evaluate(0.0,0.0),   np.array([1.0]), rtol=1.0e-4, atol=1.0e-5  ))
    assert np.any(np.isclose(fn.math.cos(PHI).evaluate((0.0,0.0)), np.array([1.0]), rtol=1.0e-4, atol=1.0e-5 ))

    ## Check derivative evaluations are possible / exist for meshVariable


    assert fn.math.sin(P).derivative(0).evaluate((0.0,0.0)).shape == (1,)
    assert fn.math.sin(P).derivative(1).evaluate((0.0,0.0)).shape == (1,)
    assert fn.math.sin(P).derivative(0).evaluate(pointst).shape   == (1,)
    assert fn.math.sin(P).derivative(0).evaluate(pointsl).shape   == (1,)
    assert fn.math.sin(P).derivative(0).evaluate(pointsl2).shape  == (3,)
    assert fn.math.sin(P).derivative(0).evaluate(pointsa).shape   == (3,)

    ## Check values 

    assert np.any(np.isclose(fn.math.cos(P).derivative(0).evaluate((0.0,0.0)),  np.array([0.0]), rtol=1.0e-4, atol=1.0e-5 ))

    ## Check symbolic representation is reasonable

    assert P.derivative(0).description == 'd(PHI)/dX*X^2 + PHI*2*X'
    assert P.derivative(1).description == 'd(PHI)/dY*X^2'

    F = SX2 + CX2 + SX**2 + CX**2

    assert isinstance(F.derivative(1), fn.parameter)
    assert F.derivative(1).value == 0.0

    return


## Test nested derivatives ... the chain rule should be OK but object recursion may be tricky for some cases

def test_higher_derivatives(DM):

    mesh2 = QuagMesh(DM, down_neighbours=1)
    PHI = mesh2.add_variable(name="PHI")

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

    F = SX2 + CX2 + SX**2 + CX**2

    # check these don't fail (symbolic)

    dF2dXdY = F.derivative(0).derivative(1)
    dF2dX2  = F.derivative(0).derivative(0)
    dF2dY2  = F.derivative(1).derivative(1)  # This one should be the derivative of a constant 

    assert dF2dXdY.description == '0'
    assert dF2dX2.description == '-sin(X^2)*2*X*2*X + cos(X^2)*2 + -1*cos(X^2)*2*X*2*X + -sin(X^2)*2 + 2*-sin(X)*sin(X) + 2*cos(X)*cos(X) + 2*-1*cos(X)*cos(X) + 2*-sin(X)*-sin(X)'

    # check these don't fail (numeric)

    dP2dX2  = PHI.derivative(0).derivative(0)    
    dP2dY2  = PHI.derivative(1).derivative(1)    
    dP2dXdY = PHI.derivative(0).derivative(1)

    ## Triple nesting

    assert PHI.derivative(0).derivative(0).derivative(1).description == 'd(d(d(PHI)/dX)/dX)/dY'

    # Mixtures of symbolic plus numeric

    FP = PHI * SX2 + PHI**2 * CX2 + SX**2 + CX**2 
    assert FP.derivative(1).derivative(0).description == 'd(d(PHI)/dY)/dX*sin(X^2) + d(PHI)/dY*cos(X^2)*2*X + (2*d(d(PHI)/dY)/dX*PHI + 2*d(PHI)/dY*d(PHI)/dX)*cos(X^2) + 2*d(PHI)/dY*PHI*-sin(X^2)*2*X'

    FP.derivative(1).derivative(0).evaluate(0.0,0.0)    
    FP.derivative(1).derivative(0).evaluate(mesh2)