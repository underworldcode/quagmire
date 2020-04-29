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

def degrees(lazyFn):
    return _make_npmath_op(_np.degrees, "180/pi*", lazyFn)

def radians(lazyFn):
    return _make_npmath_op(_np.radians, "pi/180*", lazyFn)

def rad2deg(lazyFn):
    return degrees(lazyFn)

def deg2rad(lazyFn):
    return radians(lazyFn)


# grad

def grad(lazyFn):
    """Lazy evaluation of gradient operator on a scalar field"""
    return lazyFn.fn_gradient()

def div(*args):
    """Lazy evaluation of divergence operator on a N-D vector field"""
    def _div(lazyFn_list, *args, **kwargs):
        lazy_ev = 0.0
        for lazyFn in lazyFn_list:
            lazy_ev += lazyFn.evaluate(*args, **kwargs)
        return lazy_ev

    dims = ['X', 'Y', 'Z']
    lazyFn_id = set()
    lazyFn_list = []
    lazyFn_description = ""
    lazyFn_dependency = set()
    for f, lazyFn in enumerate(args):
        lazyFn_id.add(lazyFn._mesh.id)
        lazyFn_list.append(lazyFn.fn_gradient(f))
        lazyFn_description += "diff({},{}) + ".format(lazyFn.description, dims[f])
        lazyFn_dependency.union(lazyFn.dependency_list)
    lazyFn_description = lazyFn_description[:-3]
    if len(lazyFn_id) > 1:
        raise ValueError("Meshes must be identical")

    newLazyFn = _LazyEvaluation(mesh=lazyFn._mesh)
    newLazyFn.evaluate = lambda *args, **kwargs : _div(lazyFn_list, *args, **kwargs)
    newLazyFn.description = lazyFn_description
    newLazyFn.dependency_list = lazyFn_dependency

    return newLazyFn

def curl(*args):
    """Lazy evaluation of curl operator on a 2D vector field"""

    if len(args) == 2:
        lazyFn_x = args[0]
        lazyFn_y = args[1]
        if lazyFn_x._mesh.id != lazyFn_y._mesh.id:
            raise ValueError("Both meshes must be identical")
        
        newLazyFn = _LazyEvaluation(mesh=lazyFn_x._mesh)
        fn_dvydx = lazyFn_y.fn_gradient(0)
        fn_dvxdy = lazyFn_x.fn_gradient(1)
        newLazyFn.evaluate = lambda *args, **kwargs : fn_dvydx.evaluate(*args, **kwargs) - fn_dvxdy.evaluate(*args, **kwargs)
        newLazyFn.description = "diff({},X) - diff({},Y)".format(lazyFn_y.description, lazyFn_x.description)
        newLazyFn.dependency_list = lazyFn_x.dependency_list | lazyFn_y.dependency_list

    elif len(args) == 3:
        lazyFn_x = args[0]
        lazyFn_y = args[1]
        lazyFn_z = args[2]
        if lazyFn_x._mesh.id != lazyFn_y._mesh.id != lazyFn_z._mesh.id :
            raise ValueError("All meshes must be identical")
        
        newLazyFn = _LazyEvaluation(mesh=lazyFn_x._mesh)
        fn_dvxdx, fn_dvxdy, fn_dvxdz = lazyFn_x.fn_gradient()
        fn_dvydx, fn_dvydy, fn_dvydz = lazyFn_y.fn_gradient()
        fn_dvzdx, fn_dvzdy, fn_dvzdz = lazyFn_z.fn_gradient()
        fn_curl = (fn_dvzdy-fn_dvydz) + (fn_dvxdz-fn_dvzdx) + (fn_dvydx-fn_dvxdy)
        newLazyFn.evaluate = lambda *args, **kwargs : fn_curl.evaluate(*args, **kwargs)
        desc = "(diff({},dy) - diff({},dz)) + (diff({},dz) - diff({},dx)) + (diff({},dx) - diff({},dy))"
        newLazyFn.description = desc.format(\
            lazyFn_z.description, lazyFn_y.description, \
            lazyFn_x.description, lazyFn_z.description, \
            lazyFn_y.description, lazyFn_x.description)
        newLazyFn.dependency_list = lazyFn_x.dependency_list | lazyFn_y.dependency_list | lazyFn_z.dependency_list

    else:
        raise ValueError("Enter a valid number of arguments")

    return newLazyFn

def hypot(*args):
    """Lazy evaluation of hypot operator on N fields"""
    def _hyp(lazyFn_list, *args, **kwargs):
        lazy_ev = []
        for lazyFn in lazyFn_list:
            lazy_ev.append( lazyFn.evaluate(*args, **kwargs) )
        return _np.hypot(*lazy_ev)
    lazyFn_list = []
    lazyFn_description = ""
    lazyFn_dependency = set()
    for lazyFn in args:
        lazyFn_list.append(lazyFn)
        lazyFn_description += "{}^2 + ".format(lazyFn.description)
        lazyFn_dependency.union(lazyFn.dependency_list)
    lazyFn_description = lazyFn_description[:-3]

    newLazyFn = _LazyEvaluation(mesh=lazyFn._mesh)
    newLazyFn.evaluate = lambda *args, **kwargs : _hyp(lazyFn_list, *args, **kwargs)
    newLazyFn.description = "sqrt({})".format(lazyFn_description)
    newLazyFn.dependency_list = lazyFn_dependency

    return newLazyFn


def arctan2(*args):
    """Lazy evaluation of arctan2 operator on N fields"""
    def _arctan2(lazyFn_list, *args, **kwargs):
        lazy_ev = []
        for lazyFn in lazyFn_list:
            lazy_ev.append( lazyFn.evaluate(*args, **kwargs) )
        return _np.arctan2(*lazy_ev)
    lazyFn_list = []
    lazyFn_description = ""
    lazyFn_dependency = set()
    for lazyFn in args:
        lazyFn_list.append(lazyFn)
        lazyFn_description += "{},".format(lazyFn.description)
        lazyFn_dependency.union(lazyFn.dependency_list)
    lazyFn_description = lazyFn_description[:-1]

    newLazyFn = _LazyEvaluation(mesh=lazyFn._mesh)
    newLazyFn.evaluate = lambda *args, **kwargs : _arctan2(lazyFn_list, *args, **kwargs)
    newLazyFn.description = "arctan2({})".format(lazyFn_description)
    newLazyFn.dependency_list = lazyFn_dependency

    return newLazyFn


def slope(lazyFn):
    """Lazy evaluation of the slope of a scalar field"""
    
    # need to take a few lines from `function_classes` gradient method

    if lazyFn._mesh is None:
        raise RuntimeError("fn_gradient is a numerical differentiation routine based on " + \
            "derivatives of a fitted spline function on a mesh. " + \
            "The function {} has no associated mesh. ".format(lazyFn.__repr__())) + \
            "To obtain *numerical* derivatives of this function, " + \
            "you can provide a mesh to the gradient function. " + \
            "The usual reason for this error is that your function is not based upon " + \
            "mesh variables and can, perhaps, be differentiated without resort to interpolating splines. "

    elif lazyFn._mesh is not None:
        diff_mesh = lazyFn._mesh
    else:
        import quagmire
        quagmire.mesh.check_object_is_a_q_mesh_and_raise(mesh)
        diff_mesh = mesh

    def new_fn_slope(*args, **kwargs):
        local_array = lazyFn.evaluate(diff_mesh)
        df_tuple = diff_mesh._derivative_grad_cartesian(local_array, nit=10, tol=1e-8)

        grad_f = _np.hypot(*df_tuple)/diff_mesh._radius

        if len(args) == 1 and args[0] == diff_mesh:
            return grad_f
        elif len(args) == 1 and quagmire.mesh.check_object_is_a_q_mesh_and_raise(args[0]):
            mesh = args[0]
            return diff_mesh.interpolate(mesh.coords[:,0], mesh.coords[:,1], zdata=grad_f, **kwargs)
        elif len(args) > 1:
            xi = np.atleast_1d(args[0])  # .resize(-1,1)
            yi = np.atleast_1d(args[1])  # .resize(-1,1)
            i, e = diff_mesh.interpolate(xi, yi, zdata=grad_f, **kwargs)
            return i
        else:
            err_msg = "Invalid number of arguments\n"
            err_msg += "Input a valid mesh or coordinates in x,y directions"
            raise ValueError(err_msg)

    newLazyFn = _LazyEvaluation(mesh=diff_mesh)
    newLazyFn.evaluate = new_fn_slope
    newLazyFn.description = "sqrt(d({0})/dX^2 + d({0})/dY^2)".format(lazyFn.description)
    newLazyFn.dependency_list = lazyFn.dependency_list
    return newLazyFn

## These are not defined yet (LM)

# unwrap(p[, discont, axis])	Unwrap by changing deltas between values to 2*pi complement.