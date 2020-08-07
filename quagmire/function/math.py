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

def _make_npmath_op(op, name, lazyFn, lformat):
    newLazyFn = _LazyEvaluation()
    newLazyFn.evaluate = lambda *args, **kwargs : op(lazyFn.evaluate(*args, **kwargs))
    newLazyFn.description = "{}({})".format(name,lazyFn.description)
    newLazyFn.latex = lformat.format(lazyFn.latex)
    newLazyFn.dependency_list = lazyFn.dependency_list
    return newLazyFn

## Trig

def sin(lazyFn):
    newFn = _make_npmath_op(_np.sin, "sin", lazyFn, r"\sin\left({}\right)")
    newFn.derivative = lambda dirn : cos( lazyFn ) * lazyFn.derivative(dirn)
    return newFn

def asin(lazyFn):
    newFn = _make_npmath_op(_np.arcsin, "asin", lazyFn, r"\sin^{{-1}}\left({}\right)")
    newFn.derivative = lambda dirn : lazyFn.derivative(dirn) / sqrt(_parameter(1.0) - lazyFn**2) 
    return newFn

def arcsin(lazyFn):
    newFn = _make_npmath_op(_np.arcsin, "arcsin", lazyFn, r"\sin^{{-1}}\left({}\right)")
    newFn.derivative = lambda dirn : lazyFn.derivative(dirn) / sqrt(_parameter(1.0) - lazyFn**2) 
    return newFn

def cos(lazyFn):
    newFn = _make_npmath_op(_np.cos, "cos", lazyFn, r"\cos\left({}\right)")
    newFn.derivative = lambda dirn : -sin( lazyFn ) * lazyFn.derivative(dirn)
    return newFn

def acos(lazyFn):
    newFn = _make_npmath_op(_np.arcsin, "acos", lazyFn, r"\cos^{{-1}}\left({}\right)")
    newFn.derivative = lambda dirn : - lazyFn.derivative(dirn) / sqrt(_parameter(1.0) - lazyFn**2) 
    return newFn

def arccos(lazyFn):
    newFn = _make_npmath_op(_np.arcsin, "arccos", lazyFn, r"\cos^{{-1}}\left({}\right)")
    newFn.derivative = lambda dirn : - lazyFn.derivative(dirn) / sqrt(_parameter(1.0) - lazyFn**2) 
    return newFn

def tan(lazyFn):
    newFn = _make_npmath_op(_np.tan, "tan", lazyFn, r"\tan\left({}\right)")
    newFn.derivative = lambda dirn : (_parameter(1.0) + tan(lazyFn)**2 ) * lazyFn.derivative(dirn)
    return newFn

def atan(lazyFn):
    newFn = _make_npmath_op(_np.arcsin, "atan", lazyFn,  r"\tan^{{-1}}\left({}\right)")
    newFn.derivative = lambda dirn : lazyFn.derivative(dirn) / (_parameter(1.0) + lazyFn**2) 
    return newFn

def arctan(lazyFn):
    newFn = _make_npmath_op(_np.arcsin, "arctan", lazyFn, r"\tan^{{-1}}\left({}\right)")
    newFn.derivative = lambda dirn : lazyFn.derivative(dirn) / (_parameter(1.0) + lazyFn**2) 
    return newFn

## Hyperbolic

def sinh(lazyFn):
    newFn = _make_npmath_op(_np.sinh, "sinh", lazyFn, r"\sinh\left({}\right)")
    newFn.derivative = lambda dirn : cosh( lazyFn ) * lazyFn.derivative(dirn)
    return newFn

def asinh(lazyFn):
    newFn = _make_npmath_op(_np.arcsinh, "asinh", lazyFn,  r"\sinh^{{-1}}\left({}\right)")
    newFn.derivative = lambda dirn : lazyFn.derivative(dirn) / sqrt(_parameter(1.0) + lazyFn**2) 
    return newFn

def arcsinh(lazyFn):
    newFn = _make_npmath_op(_np.arcsinh, "arcsinh",  lazyFn, r"\sinh^{{-1}}\left({}\right)")
    newFn.derivative = lambda dirn : lazyFn.derivative(dirn) / sqrt(_parameter(1.0) + lazyFn**2) 
    return newFn

def cosh(lazyFn):
    newFn = _make_npmath_op(_np.cosh, "cosh", lazyFn, r"\cosh\left({}\right)")
    newFn.derivative = lambda dirn : sinh( lazyFn ) * lazyFn.derivative(dirn)
    return newFn

def acosh(lazyFn):
    newFn = _make_npmath_op(_np.arccosh, "acosh", lazyFn, r"\cosh^{{-1}}\left({}\right)")
    newFn.derivative = lambda dirn : lazyFn.derivative(dirn) / sqrt(_parameter(-1.0) + lazyFn**2) 
    return newFn

def arccosh(lazyFn):
    newFn = _make_npmath_op(_np.arccosh, "arccosh", lazyFn, r"\cosh^{{-1}}\left({}\right)")
    newFn.derivative = lambda dirn : lazyFn.derivative(dirn) / sqrt(_parameter(-1.0) + lazyFn**2) 
    return newFn

def tanh(lazyFn):
    newFn = _make_npmath_op(_np.tanh, "tanh", lazyFn, r"\tanh\left({}\right)")
    newFn.derivative = lambda dirn : (_parameter(1.0) - lazyFn.derivative(dirn)**2) * lazyFn.derivative(dirn)
    return newFn

def atanh(lazyFn):
    newFn = _make_npmath_op(_np.arctanh, "atanh", lazyFn,  r"\tanh^{{-1}}\left({}\right)")
    newFn.derivative = lambda dirn : lazyFn.derivative(dirn) / (_parameter(1.0) - lazyFn**2) 
    return newFn

def arctanh(lazyFn):
    newFn = _make_npmath_op(_np.arctanh,"arctanh", lazyFn,  r"\tanh^{{-1}}\left({}\right)")
    newFn.derivative = lambda dirn : lazyFn.derivative(dirn) / (_parameter(1.0) - lazyFn**2) 
    return newFn    


## exponentiation

def exp(lazyFn):
    newFn =  _make_npmath_op(_np.exp, "exp", lazyFn, r"\exp\left({}\right)")
    newFn.derivative = lambda dirn : exp(lazyFn) * lazyFn.derivative(dirn)
    return newFn

def log(lazyFn):
    newFn =  _make_npmath_op(_np.log, "log", lazyFn, r"\log_e\left({}\right)")
    newFn.derivative = lambda dirn :  lazyFn.derivative(dirn) / lazyFn
    return newFn

def log10(lazyFn):
    newFn =  _make_npmath_op(_np.log10, "log10",lazyFn, r"\log_{{10}}\left({}\right)")
    ## See definition of d/dx(log10(x))
    newFn.derivative = lambda dirn :  _parameter(0.434294481903) * lazyFn.derivative(dirn) / lazyFn
    return newFn

def sqrt(lazyFn):
    newFn = _make_npmath_op(_np.sqrt, "sqrt", lazyFn, r"\sqrt{{ {} }}")
    newFn.derivative = lambda dirn :  lazyFn.derivative(dirn) / sqrt(lazyFn) 
    return newFn


# misc

## Not sure about the absolute value and symbolic derivative

def fabs(lazyFn):
    return _make_npmath_op(_np.fabs, "fabs", lazyFn, r"\left|{}\right|")

def degrees(lazyFn):
    newFn = _make_npmath_op(_np.degrees, "degrees", lazyFn, r"\frac{{180}}{{\pi}}{}")
    newFn.derivative = lambda dirn : _parameter(180.0/_np.pi) * lazyFn.derivative(dirn)
    return newFn

def radians(lazyFn):
    newFn = _make_npmath_op(_np.radians, "radians", lazyFn, r"\frac{{\pi}}{{180}}{}")
    newFn.derivative = lambda dirn : _parameter(_np.pi/180.0) * lazyFn.derivative(dirn)
    return newFn

def rad2deg(lazyFn):
    return degrees(lazyFn)

def deg2rad(lazyFn):
    return radians(lazyFn)


## Vector Operators  (Cartesian assumed here - beware !!)

def grad(lazyFn):
    """Lazy evaluation of gradient operator on a scalar field"""

    if isinstance(lazyFn, (float, int, _parameter)):
        return _parameter(0.0)


    # check if this function has a fn_grad method, otherwise brute force

    try:
        return lazyFn.fn_grad()
    except:
        return lazyFn.derivative(0), lazyFn.derivative(1)

def div(lazyFn_0, lazyFn_1):
    """Lazy evaluation of divergence operator on a vector field"""

    if isinstance(lazyFn_0, (float, int, _parameter)):
        lazyFn_0x = _parameter(0.0)
    else:
        lazyFn_0x = lazyFn_0.derivative(0)

    if isinstance(lazyFn_1, (float, int, _parameter)):
        lazyFn_1x = _parameter(0.0)
    else:
        lazyFn_1x = lazyFn_1.derivative(0)

    return lazyFn_0x + lazyFn_1x
 

def laplacian(lazyFn, lazyFn_coeff=None):
    """Lazy evaluation of Laplacian with variable coefficient"""

    if isinstance(lazyFn, (float, int, _parameter)):
        return _parameter(0.0)

    if lazyFn_coeff is None:
        lazyFn_coeff = _parameter(1.0)

    try:
        return lazyFn.fn_laplacian(lazyFn_coeff)
    except:
        
        f1, f2 = grad(lazyFn)
        fl = div(lazyFn_coeff*f1, lazyFn_coeff*f1)
        fl.description = "div.grad({})".format(lazyFn.description)
        fl.latex = r"\nabla^2\left( {} \right)".format(lazyFn.latex)
        fl.exposed_operator = "S"
        return fl

def slope(lazyFn):
    """Lazy evaluation of gradient operator on a scalar field"""

    if isinstance(lazyFn, (float, int, _parameter)):
        return _parameter(0.0)

    # check if this function has a fn_slope method, otherwise brute force

    try:
        return lazyFn.fn_slope()
    except:
        newlazyFn = sqrt(lazyFn.derivative(0)**2 + lazyFn.derivative(1)**2)
        newlazyFn.description = "slope({})".format(lazyFn.description)
        newlazyFn.latex = r"\left| \nabla {} \right|".format(lazyFn.description)
        newlazyFn.exposed_operator = "S"
        return newlazyFn

  


# def div(*args):
#     """Lazy evaluation of divergence operator on a N-D vector field"""
#     def _div(lazyFn_list, *args, **kwargs):
#         lazy_ev = 0.0
#         for lazyFn in lazyFn_list:
#             lazy_ev += lazyFn.evaluate(*args, **kwargs)
#         return lazy_ev

#     dims = ['X', 'Y', 'Z']
#     lazyFn_id = set()
#     lazyFn_list = []
#     lazyFn_description = ""
#     lazyFn_dependency = set()
#     for f, lazyFn in enumerate(args):
#         lazyFn_list.append(lazyFn.fn_gradient[f])
#         lazyFn_description += "diff({},{}) + ".format(lazyFn.description, dims[f])
#         lazyFn_dependency.union(lazyFn.dependency_list)
#     lazyFn_description = lazyFn_description[:-3]

#     newLazyFn = _LazyEvaluation()
#     newLazyFn.evaluate = lambda *args, **kwargs : _div(lazyFn_list, *args, **kwargs)
#     newLazyFn.description = lazyFn_description
#     newLazyFn.dependency_list = lazyFn_dependency

#     return newLazyFn

def curl(*args):
    """Lazy evaluation of curl operator on a 2D vector field"""

    if len(args) == 2:
        lazyFn_x = args[0]
        lazyFn_y = args[1]
        
        newLazyFn = _LazyEvaluation()
        fn_dvydx = lazyFn_y.derivative(0)
        fn_dvxdy = lazyFn_x.derivative(1)
        newLazyFn.evaluate = lambda *args, **kwargs : fn_dvydx.evaluate(*args, **kwargs) - fn_dvxdy.evaluate(*args, **kwargs)
        newLazyFn.description = "diff({},X) - diff({},Y)".format(lazyFn_y.description, lazyFn_x.description)
        newLazyFn.latex = r"\partial \left( {} \right) / \partial x_0 - \partial \left( {} \right) / \partial x_1".format(lazyFn_x.latex, lazyFn_y.latex)
        newLazyFn.dependency_list = lazyFn_x.dependency_list | lazyFn_y.dependency_list

    # elif len(args) == 3:
    #     raise NotImplementedError
    #     lazyFn_x = args[0]
    #     lazyFn_y = args[1]
    #     lazyFn_z = args[2]
        
    #     newLazyFn = _LazyEvaluation()
    #     fn_dvxdx, fn_dvxdy, fn_dvxdz = lazyFn_x.fn_gradient
    #     fn_dvydx, fn_dvydy, fn_dvydz = lazyFn_y.fn_gradient
    #     fn_dvzdx, fn_dvzdy, fn_dvzdz = lazyFn_z.fn_gradient
    #     fn_curl = (fn_dvzdy-fn_dvydz) + (fn_dvxdz-fn_dvzdx) + (fn_dvydx-fn_dvxdy)
    #     newLazyFn.evaluate = lambda *args, **kwargs : fn_curl.evaluate(*args, **kwargs)
    #     desc = "(diff({},dy) - diff({},dz)) + (diff({},dz) - diff({},dx)) + (diff({},dx) - diff({},dy))"
    #     newLazyFn.description = desc.format(\
    #         lazyFn_z.description, lazyFn_y.description, \
    #         lazyFn_x.description, lazyFn_z.description, \
    #         lazyFn_y.description, lazyFn_x.description)
    #     newLazyFn.dependency_list = lazyFn_x.dependency_list | lazyFn_y.dependency_list | lazyFn_z.dependency_list

    # else:
    #     raise ValueError("Enter a valid number of arguments")

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

