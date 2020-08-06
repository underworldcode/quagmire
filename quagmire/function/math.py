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
from .function_classes import parameter as _parameter 

## Functions of a single variable

def _make_npmath_op(op, name, lazyFn):
    newLazyFn = _LazyEvaluation()
    newLazyFn.evaluate = lambda *args, **kwargs : op(lazyFn.evaluate(*args, **kwargs))
    newLazyFn.description = "{}({})".format(name,lazyFn.description)
    newLazyFn.dependency_list = lazyFn.dependency_list
    return newLazyFn

## Trig

def sin(lazyFn):
    newFn = _make_npmath_op(_np.sin, "sin", lazyFn)
    newFn.derivative = lambda dirn : cos( lazyFn ) * lazyFn.derivative(dirn)
    return newFn

def asin(lazyFn):
    newFn = _make_npmath_op(_np.arcsin, "asin", lazyFn)
    newFn.derivative = lambda dirn : lazyFn.derivative(dirn) / sqrt(_parameter(1.0) - lazyFn**2) 
    return newFn

def arcsin(lazyFn):
    newFn = _make_npmath_op(_np.arcsin, "arcsin", lazyFn)
    newFn.derivative = lambda dirn : lazyFn.derivative(dirn) / sqrt(_parameter(1.0) - lazyFn**2) 
    return newFn

def cos(lazyFn):
    newFn = _make_npmath_op(_np.cos, "cos", lazyFn)
    newFn.derivative = lambda dirn : -sin( lazyFn ) * lazyFn.derivative(dirn)
    return newFn

def acos(lazyFn):
    newFn = _make_npmath_op(_np.arcsin, "acos", lazyFn)
    newFn.derivative = lambda dirn : - lazyFn.derivative(dirn) / sqrt(_parameter(1.0) - lazyFn**2) 
    return newFn

def arccos(lazyFn):
    newFn = _make_npmath_op(_np.arcsin, "arccos", lazyFn)
    newFn.derivative = lambda dirn : - lazyFn.derivative(dirn) / sqrt(_parameter(1.0) - lazyFn**2) 
    return newFn

def tan(lazyFn):
    newFn = _make_npmath_op(_np.tan, "tan", lazyFn)
    newFn.derivative = lambda dirn : (_parameter(1.0) + tan(lazyFn)**2 ) * lazyFn.derivative(dirn)
    return newFn

def atan(lazyFn):
    newFn = _make_npmath_op(_np.arcsin, "atan", lazyFn)
    newFn.derivative = lambda dirn : lazyFn.derivative(dirn) / (_parameter(1.0) + lazyFn**2) 
    return newFn

def arctan(lazyFn):
    newFn = _make_npmath_op(_np.arcsin, "arctan", lazyFn)
    newFn.derivative = lambda dirn : lazyFn.derivative(dirn) / (_parameter(1.0) + lazyFn**2) 
    return newFn

## Hyperbolic

def sinh(lazyFn):
    newFn = _make_npmath_op(_np.sinh, "sinh", lazyFn)
    newFn.derivative = lambda dirn : cosh( lazyFn ) * lazyFn.derivative(dirn)
    return newFn

def asinh(lazyFn):
    newFn = _make_npmath_op(_np.arcsinh, "asinh", lazyFn)
    newFn.derivative = lambda dirn : lazyFn.derivative(dirn) / sqrt(_parameter(1.0) + lazyFn**2) 
    return newFn

def arcsinh(lazyFn):
    newFn = _make_npmath_op(_np.arcsinh, "arcsinh", lazyFn)
    newFn.derivative = lambda dirn : lazyFn.derivative(dirn) / sqrt(_parameter(1.0) + lazyFn**2) 
    return newFn

def cosh(lazyFn):
    newFn = _make_npmath_op(_np.cosh, "cosh", lazyFn)
    newFn.derivative = lambda dirn : sinh( lazyFn ) * lazyFn.derivative(dirn)
    return newFn

def acosh(lazyFn):
    newFn = _make_npmath_op(_np.arccosh, "acosh", lazyFn)
    newFn.derivative = lambda dirn : lazyFn.derivative(dirn) / sqrt(_parameter(-1.0) + lazyFn**2) 
    return newFn

def arccosh(lazyFn):
    newFn = _make_npmath_op(_np.arccosh, "acosh", lazyFn)
    newFn.derivative = lambda dirn : lazyFn.derivative(dirn) / sqrt(_parameter(-1.0) + lazyFn**2) 
    return newFn

def tanh(lazyFn):
    newFn = _make_npmath_op(_np.tanh, "tanh", lazyFn)
    newFn.derivative = lambda dirn : (_parameter(1.0) - lazyFn.derivative(dirn)**2) * lazyFn.derivative(dirn)
    return newFn

def atanh(lazyFn):
    newFn = _make_npmath_op(_np.arctanh, "atanh", lazyFn)
    newFn.derivative = lambda dirn : lazyFn.derivative(dirn) / (_parameter(1.0) - lazyFn**2) 
    return newFn

def arctanh(lazyFn):
    newFn = _make_npmath_op(_np.arctanh, "arctanh", lazyFn)
    newFn.derivative = lambda dirn : lazyFn.derivative(dirn) / (_parameter(1.0) - lazyFn**2) 
    return newFn    


## exponentiation

def exp(lazyFn):
    newFn =  _make_npmath_op(_np.exp, "exp", lazyFn)
    newFn.derivative = lambda dirn : exp(lazyFn) * lazyFn.derivative(dirn)
    return newFn

def log(lazyFn):
    newFn =  _make_npmath_op(_np.log, "log", lazyFn)
    newFn.derivative = lambda dirn :  lazyFn.derivative(dirn) / lazyFn
    return newFn

def log10(lazyFn):
    newFn =  _make_npmath_op(_np.log10, "log10", lazyFn)
    ## See definition of d/dx(log10(x))
    newFn.derivative = lambda dirn :  _parameter(0.434294481903) * lazyFn.derivative(dirn) / lazyFn
    return newFn

def sqrt(lazyFn):
    newFn = _make_npmath_op(_np.sqrt, "sqrt", lazyFn)
    newFn.derivative = lambda dirn : (_parameter(1.0) / lazyFn**2) * lazyFn.derivative(dirn)
    return newFn


# misc

## Not sure about the absolute value and symbolic derivative

def fabs(lazyFn):
    return _make_npmath_op(_np.fabs, "fabs", lazyFn)

def degrees(lazyFn):
    newFn = _make_npmath_op(_np.degrees, "degrees", lazyFn)
    newFn.derivative = lambda dirn : _parameter(180.0/_np.pi) * lazyFn.derivative(dirn)
    return newFn

def radians(lazyFn):
    newFn = _make_npmath_op(_np.radians, "radians", lazyFn)
    newFn.derivative = lambda dirn : _parameter(_np.pi/180.0) * lazyFn.derivative(dirn)
    return newFn

def rad2deg(lazyFn):
    return degrees(lazyFn)

def deg2rad(lazyFn):
    return radians(lazyFn)


## ToDo: No derivatives for these yet

# grad

def grad(lazyFn):
    """Lazy evaluation of gradient operator on a scalar field"""
    return lazyFn.fn_gradient[0], lazyFn.fn_gradient[1]

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
        lazyFn_list.append(lazyFn.fn_gradient[f])
        lazyFn_description += "diff({},{}) + ".format(lazyFn.description, dims[f])
        lazyFn_dependency.union(lazyFn.dependency_list)
    lazyFn_description = lazyFn_description[:-3]

    newLazyFn = _LazyEvaluation()
    newLazyFn.evaluate = lambda *args, **kwargs : _div(lazyFn_list, *args, **kwargs)
    newLazyFn.description = lazyFn_description
    newLazyFn.dependency_list = lazyFn_dependency

    return newLazyFn

def curl(*args):
    """Lazy evaluation of curl operator on a 2D vector field"""

    if len(args) == 2:
        lazyFn_x = args[0]
        lazyFn_y = args[1]
        
        newLazyFn = _LazyEvaluation()
        fn_dvydx = lazyFn_y.fn_gradient[0]
        fn_dvxdy = lazyFn_x.fn_gradient[1]
        newLazyFn.evaluate = lambda *args, **kwargs : fn_dvydx.evaluate(*args, **kwargs) - fn_dvxdy.evaluate(*args, **kwargs)
        newLazyFn.description = "diff({},X) - diff({},Y)".format(lazyFn_y.description, lazyFn_x.description)
        newLazyFn.dependency_list = lazyFn_x.dependency_list | lazyFn_y.dependency_list

    elif len(args) == 3:
        raise NotImplementedError
        lazyFn_x = args[0]
        lazyFn_y = args[1]
        lazyFn_z = args[2]
        
        newLazyFn = _LazyEvaluation()
        fn_dvxdx, fn_dvxdy, fn_dvxdz = lazyFn_x.fn_gradient
        fn_dvydx, fn_dvydy, fn_dvydz = lazyFn_y.fn_gradient
        fn_dvzdx, fn_dvzdy, fn_dvzdz = lazyFn_z.fn_gradient
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

    newLazyFn = _LazyEvaluation()
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

    newLazyFn = _LazyEvaluation()
    newLazyFn.evaluate = lambda *args, **kwargs : _arctan2(lazyFn_list, *args, **kwargs)
    newLazyFn.description = "arctan2({})".format(lazyFn_description)
    newLazyFn.dependency_list = lazyFn_dependency

    return newLazyFn


def slope(meshVar):
    """Lazy evaluation of the slope of a scalar field"""
    
    # need to take a few lines from `function_classes` gradient method

    if meshVar._mesh is None:
        raise RuntimeError("fn_gradient is a numerical differentiation routine based on " + \
            "derivatives of a fitted spline function on a mesh. " + \
            "The function {} has no associated mesh. ".format(meshVar.__repr__())) + \
            "To obtain *numerical* derivatives of this function, " + \
            "you can provide a mesh to the gradient function. " + \
            "The usual reason for this error is that your function is not based upon " + \
            "mesh variables and can, perhaps, be differentiated without resort to interpolating splines. "

    elif meshVar._mesh is not None:
        diff_mesh = meshVar._mesh
    else:
        import quagmire
        quagmire.mesh.check_object_is_a_q_mesh_and_raise(mesh)
        diff_mesh = mesh

    def new_fn_slope(*args, **kwargs):
        local_array = meshVar.evaluate(diff_mesh)
        df_tuple = diff_mesh._derivative_grad_cartesian(local_array, nit=10, tol=1e-8)

        # need to input tuple because spherical grads are nx3
        grad_f = _np.hypot(*list(df_tuple.T))/diff_mesh._radius

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

    newLazyFn = _LazyEvaluation()
    newLazyFn.evaluate = new_fn_slope
    newLazyFn.description = "sqrt(d({0})/dX^2 + d({0})/dY^2)".format(meshVar.description)
    newLazyFn.dependency_list = meshVar.dependency_list
    return newLazyFn


## These are not defined yet (LM)

# unwrap(p[, discont, axis])	Unwrap by changing deltas between values to 2*pi complement.