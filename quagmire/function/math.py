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
from .function_classes import convert as _convert


## Functions of a single variable

def _make_npmath_op(op, name, lazyFn, lformat):
    lazyFn = _convert(lazyFn)
    newLazyFn = _LazyEvaluation()
    newLazyFn.coordinate_system = lazyFn.coordinate_system
    newLazyFn.evaluate = lambda *args, **kwargs : op(lazyFn.evaluate(*args, **kwargs))
    newLazyFn.description = "{}({})".format(name,lazyFn.description)
    newLazyFn.latex = lformat.format(lazyFn.latex)
    newLazyFn.math  = lambda : lformat.format(lazyFn.math())

    newLazyFn.dependency_list = lazyFn.dependency_list
    return newLazyFn

## Trig

def sin(lazyFn):
    newFn = _make_npmath_op(_np.sin, "sin", lazyFn, r"\sin\left({}\right)")
    newFn.derivative = lambda dirn : lazyFn.derivative(dirn) * cos( lazyFn ) 
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
    newFn.derivative = lambda dirn : -lazyFn.derivative(dirn) * sin( lazyFn ) 
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
    newFn.derivative = lambda dirn : lazyFn.derivative(dirn) * (_parameter(1.0) + tan(lazyFn)**2 ) 
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
    newFn.derivative = lambda dirn : lazyFn.derivative(dirn) * cosh( lazyFn ) 
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
    newFn.derivative = lambda dirn : lazyFn.derivative(dirn) * sinh( lazyFn ) 
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
    newFn.derivative = lambda dirn : lazyFn.derivative(dirn) * (_parameter(1.0) - lazyFn.derivative(dirn)**2) 
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
    newFn.derivative = lambda dirn : lazyFn.derivative(dirn) * exp(lazyFn) 
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


##  ?? Are these geometry safe ? Probably not !


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

