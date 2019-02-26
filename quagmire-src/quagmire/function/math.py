"""
Copyright 2016-2019 Louis Moresi, Ben Mather, Romain Beucher

This file is part of Quagmire.

Quagmire is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or any later version.

Quagmire is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with Quagmire.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
from .function_classes import LazyEvaluation as _LazyEvaluation

## Functions of a single variable

def _make_npmath_op(op, name, lazyFn):
    newLazyFn = _LazyEvaluation(mesh=lazyFn._mesh)
    newLazyFn.evaluate = lambda *args, **kwargs : op(lazyFn.evaluate(*args, **kwargs))
    newLazyFn.gradient = lambda *args, **kwargs : op(lazyFn.gradient(*args, **kwargs))
    newLazyFn.description = "{}({})".format(name,lazyFn.description)
    return newLazyFn

## Trig

def sin(lazyFn):
    return _make_npmath_op(np.sin, "SIN", lazyFn)

def arcsin(lazyFn):
    return _make_npmath_op(np.arcsin, "ARCSIN", lazyFn)

def cos(lazyFn):
    return _make_npmath_op(np.cos, "COS", lazyFn)

def arccos(lazyFn):
    return _make_npmath_op(np.arccos, "ARCCOS", lazyFn)

def tan(lazyFn):
    return _make_npmath_op(np.tan, "TAN", lazyFn)

def arctan(lazyFn):
    return _make_npmath_op(np.arctan, "ARCTAN", lazyFn)

## Hyperbolic

def sinh(lazyFn):
    return _make_npmath_op(np.sinh, "SINH", lazyFn)

def arcsinh(lazyFn):
    return _make_npmath_op(np.arcsinh, "ARCSINH", lazyFn)

def cosh(lazyFn):
    return _make_npmath_op(np.cosh, "COSH", lazyFn)

def arccosh(lazyFn):
    return _make_npmath_op(np.arccosh, "ARCCOSH", lazyFn)

def tanh(lazyFn):
    return _make_npmath_op(np.tanh, "TANH", lazyFn)

def arctanh(lazyFn):
    return _make_npmath_op(np.arctanh, "ARCTANH", lazyFn)


## exponentiation

def exp(lazyFn):
    return _make_npmath_op(np.exp, "EXP", lazyFn)

def log(lazyFn):
    return _make_npmath_op(np.log, "LOG", lazyFn)

def log10(lazyFn):
    return _make_npmath_op(np.log10, "LOG10", lazyFn)

# misc

def fabs(lazyFn):
    return _make_npmath_op(np.fabs, "FABS", lazyFn)

def sqrt(lazyFn):
    return _make_npmath_op(np.sqrt, "SQRT", lazyFn)



# grad

def grad(lazyFn):
    """Lazy evaluation of 2D gradient operator on a scalar field"""

    return lazyFn.fn_gradient(0), lazyFn.fn_gradient(1)

def div(lazyFn_x, lazyFn_y):
    """Lazy evaluation of divergence operator on a 2D vector field"""
    newLazyFn = _LazyEvaluation(mesh=lazyFn_x._mesh) # should check the meshes match etc
    fn_dx = lazyFn_x.fn_gradient(0)
    fn_dy = lazyFn_y.fn_gradient(1)
    newLazyFn.evaluate = lambda *args, **kwargs : fn_dx.evaluate(*args, **kwargs) + fn_dy.evaluate(*args, **kwargs)
    newLazyFn.description = "d({})/dX + d({})/dY".format(lazyFn_x.description, lazyFn_y.description)
    return newLazyFn

def curl(lazyFn_x, lazyFn_y):
    """Lazy evaluation of curl operator on a 2D vector field"""
    newLazyFn = _LazyEvaluation(mesh=lazyFn_x._mesh)  # should check the meshes match etc
    fn_dvydx = lazyFn_y.fn_gradient(0)
    fn_dvxdy = lazyFn_x.fn_gradient(1)
    newLazyFn.evaluate = lambda *args, **kwargs : fn_dvydx.evaluate(*args, **kwargs) - fn_dvxdy.evaluate(*args, **kwargs)
    newLazyFn.description = "d({})/dX - d({})/dY".format(lazyFn_y.description, lazyFn_x.description)
    return newLazyFn


## These are not defined yet

# hypot(x1, x2, /[, out, where, casting, …])	Given the “legs” of a right triangle, return its hypotenuse.
# arctan2(x1, x2, /[, out, where, casting, …])	Element-wise arc tangent of x1/x2 choosing the quadrant correctly.
# degrees(x, /[, out, where, casting, order, …])	Convert angles from radians to degrees.
# radians(x, /[, out, where, casting, order, …])	Convert angles from degrees to radians.
# unwrap(p[, discont, axis])	Unwrap by changing deltas between values to 2*pi complement.
# deg2rad(x, /[, out, where, casting, order, …])	Convert angles from degrees to radians.
# rad2deg(x, /[, out, where, casting, order, …])	Convert angles from radians to degrees.
