# -*- coding: utf-8 -*-

import pytest

# ==========================

# This has already been tested by now

from quagmire.tools import meshtools
from quagmire import QuagMesh
from quagmire.mesh import MeshVariable
from quagmire import function as fn
import numpy as np

from conftest import load_triangulated_mesh_DM

def test_parameters():

    ## Assignment

    A = fn.parameter(10.0)


    ## And this is how to update A

    A.value = 100.0
    assert fn.math.exp(A).evaluate(0.0,0.0) == np.exp(100.0)

    ## This works too ... and note the floating point conversion
    A(101)
    assert fn.math.exp(A).evaluate(0.0,0.0) == np.exp(101.0)

    ## More complicated examples
    assert (fn.math.sin(A)**2.0 + fn.math.cos(A)**2.0).evaluate(0.0,0.0) == 1.0



def test_fn_description():


    A = fn.parameter(10.0)
    B = fn.parameter(3)


    ## These will flush out any changes in the interface

    assert A.description == "10.0"
    assert B.description == "3.0"

    assert (fn.math.sin(A)+fn.math.cos(B)).description == "(sin(10.0))+(cos(3.0))"
    assert (fn.math.sin(A)**2.0 + fn.math.cos(A)**2.0).description == "((sin(10.0))**(2.0))+((cos(10.0))**(2.0))"


    return

def test_fn_mesh_variables(load_triangulated_mesh_DM):

    mesh = QuagMesh(load_triangulated_mesh_DM, down_neighbours=2)

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


