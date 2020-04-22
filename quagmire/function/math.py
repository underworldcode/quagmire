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

import numpy as _np
from .function_classes import LazyEvaluation as _LazyEvaluation

## Functions of a single variable

def _make_npmath_op(op, name, lazyFn):
    newLazyFn = _LazyEvaluation(mesh=lazyFn._mesh)
    newLazyFn.evaluate = lambda *args, **kwargs : op(lazyFn.evaluate(*args, **kwargs))
    newLazyFn.gradient = lambda *args, **kwargs : op(lazyFn.gradient(*args, **kwargs))
    newLazyFn.description = "{}({})".format(name,lazyFn.description)
    newLazyFn.dependency_list = lazyFn.dependency_list
    return newLazyFn

## Trig

def sin(lazyFn):
    return _make_npmath_op(_np.sin, "sin", lazyFn)

def asin(lazyFn):
    return _make_npmath_op(_np.arcsin, "asin", lazyFn)

def arcsin(lazyFn):
    return _make_npmath_op(_np.arcsin, "arcsin", lazyFn)

def cos(lazyFn):
    return _make_npmath_op(_np.cos, "cos", lazyFn)

def acos(lazyFn):
    return _make_npmath_op(_np.arccos, "acos", lazyFn)

def arccos(lazyFn):
    return _make_npmath_op(_np.arccos, "arccos", lazyFn)

def tan(lazyFn):
    return _make_npmath_op(_np.tan, "tan", lazyFn)

def atan(lazyFn):
    return _make_npmath_op(_np.arctan, "atan", lazyFn)

def arctan(lazyFn):
    return _make_npmath_op(_np.arctan, "arctan", lazyFn)


## Hyperbolic

def sinh(lazyFn):
    return _make_npmath_op(_np.sinh, "sinh", lazyFn)

def asinh(lazyFn):
    return _make_npmath_op(_np.arcsinh, "asinh", lazyFn)

def arcsinh(lazyFn):
    return _make_npmath_op(_np.arcsinh, "arcsinh", lazyFn)

def cosh(lazyFn):
    return _make_npmath_op(_np.cosh, "cosh", lazyFn)

def acosh(lazyFn):
    return _make_npmath_op(_np.arccosh, "acosh", lazyFn)

def arccosh(lazyFn):
    return _make_npmath_op(_np.arccosh, "arccosh", lazyFn)

def tanh(lazyFn):
    return _make_npmath_op(_np.tanh, "tanh", lazyFn)

def atanh(lazyFn):
    return _make_npmath_op(_np.arctanh, "atanh", lazyFn)

def arctanh(lazyFn):
    return _make_npmath_op(_np.arctanh, "arctanh", lazyFn)


## exponentiation

def exp(lazyFn):
    return _make_npmath_op(_np.exp, "exp", lazyFn)

def log(lazyFn):
    return _make_npmath_op(_np.log, "log", lazyFn)

def log10(lazyFn):
    return _make_npmath_op(_np.log10, "log10", lazyFn)

# misc

def fabs(lazyFn):
    return _make_npmath_op(_np.fabs, "fabs", lazyFn)

def sqrt(lazyFn):
    return _make_npmath_op(_np.sqrt, "sqrt", lazyFn)



# grad

def grad(lazyFn):
    """Lazy evaluation of 2D gradient operator on a scalar field"""

    return lazyFn.fn_gradient(0), lazyFn.fn_gradient(1)

def div(lazyFn_x, lazyFn_y):
    """Lazy evaluation of divergence operator on a 2D vector field"""
    if lazyFn_x._mesh.id != lazyFn_y._mesh.id:
        raise ValueError("Both meshes must be identical")
    newLazyFn = _LazyEvaluation(mesh=lazyFn_x._mesh)
    fn_dx = lazyFn_x.fn_gradient(0)
    fn_dy = lazyFn_y.fn_gradient(1)
    newLazyFn.evaluate = lambda *args, **kwargs : fn_dx.evaluate(*args, **kwargs) + fn_dy.evaluate(*args, **kwargs)
    newLazyFn.description = "diff({},X) + diff({},Y)".format(lazyFn_x.description, lazyFn_y.description)
    newLazyFn.dependency_list = lazyFn_x.dependency_list | lazyFn_y.dependency_list

    return newLazyFn

def curl(lazyFn_x, lazyFn_y):
    """Lazy evaluation of curl operator on a 2D vector field"""
    if lazyFn_x._mesh.id != lazyFn_y._mesh.id:
        raise ValueError("Both meshes must be identical")
    newLazyFn = _LazyEvaluation(mesh=lazyFn_x._mesh)
    fn_dvydx = lazyFn_y.fn_gradient(0)
    fn_dvxdy = lazyFn_x.fn_gradient(1)
    newLazyFn.evaluate = lambda *args, **kwargs : fn_dvydx.evaluate(*args, **kwargs) - fn_dvxdy.evaluate(*args, **kwargs)
    newLazyFn.description = "diff({},X) - diff({},Y)".format(lazyFn_y.description, lazyFn_x.description)
    newLazyFn.dependency_list = lazyFn_x.dependency_list | lazyFn_y.dependency_list

    return newLazyFn


## These are not defined yet (LM)

# hypot(x1, x2, /[, out, where, casting, …])	Given the “legs” of a right triangle, return its hypotenuse.
# arctan2(x1, x2, /[, out, where, casting, …])	Element-wise arc tangent of x1/x2 choosing the quadrant correctly.
# degrees(x, /[, out, where, casting, order, …])	Convert angles from radians to degrees.
# radians(x, /[, out, where, casting, order, …])	Convert angles from degrees to radians.
# unwrap(p[, discont, axis])	Unwrap by changing deltas between values to 2*pi complement.
# deg2rad(x, /[, out, where, casting, order, …])	Convert angles from degrees to radians.
# rad2deg(x, /[, out, where, casting, order, …])	Convert angles from radians to degrees.
