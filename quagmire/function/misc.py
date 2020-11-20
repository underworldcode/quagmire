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
import quagmire
from .function_classes import LazyEvaluation as _LazyEvaluation
from .function_classes import parameter as _parameter 


def coord(dirn):

    def extract_xs(*args, **kwargs):
        """ If a mesh, return the coords at the nodes of that mesh
        In all other cases pass through the coordinates given.
        Valid format for coordinates is anything that can be coerced into 
        a numpy array with trailing dimension 2 OR a pair of arguments
        that are interpreted as a single coordinate"""

        if len(args) == 1:
            if  quagmire.mesh.check_object_is_a_q_mesh(args[0]):
                mesh = args[0]
                return mesh.coords[:,0]
            else:
                # coerce to np.array 
                arr = _np.array(args[0]).reshape(-1,2)
                return arr[:,0].reshape(-1)
        else:
            return args[0]

    def extract_ys(*args, **kwargs):
        """ If no arguments or the argument is the mesh, return the
            coords at the nodes. In all other cases pass through the
            coordinates given """

        if len(args) == 1:
            if  quagmire.mesh.check_object_is_a_q_mesh(args[0]):
                mesh = args[0]
                return mesh.coords[:,1]
            else:
                # coerce to np.array 
                arr = _np.array(args[0]).reshape(-1,2)
                return arr[:,1].reshape(-1)
        else:
            return args[1]

    newLazyFn_xs = _LazyEvaluation()
    newLazyFn_xs.evaluate = extract_xs
    newLazyFn_xs.description = "X"
    newLazyFn_xs.latex = r"\xi_0"
    newLazyFn_xs.math = lambda : newLazyFn_xs.latex
    newLazyFn_xs.derivative = lambda ddirn : _parameter(1.0) if str(ddirn) in '0' else _parameter(0.0)

    newLazyFn_ys = _LazyEvaluation()
    newLazyFn_ys.evaluate = extract_ys
    newLazyFn_ys.description = "Y"
    newLazyFn_ys.latex = r"\xi_1"
    newLazyFn_ys.math = lambda : newLazyFn_ys.latex
    newLazyFn_ys.derivative = lambda ddirn : _parameter(1.0) if str(ddirn) in '1' else _parameter(0.0)

    if dirn == 0:
        return newLazyFn_xs
    else:
        return newLazyFn_ys




# def levelset(fn2mask, lazyFn, alpha=0.5, invert=False):

#     assert isinstance(lazyFn, _LazyEvaluation), """
#         lazyFn argument is not of type a function"""

#     newLazyFn = _LazyEvaluation()

#     def threshold(*args, **kwargs):
#         if not invert:
#             values = fn2mask.evaluate(*args, **kwargs) * (lazyFn.evaluate(*args, **kwargs) > alpha).astype(float)
#         else:
#             values = fn2mask.evaluate(*args, **kwargs) * (lazyFn.evaluate(*args, **kwargs) < alpha).astype(float)

#         return values

#     newLazyFn.evaluate = threshold
#     newLazyFn.dependency_list = lazyFn.dependency_list

#     if not invert:
#         newLazyFn.description = "({} if {} > {} else 0".format(fn2mask.description, lazyFn.description, alpha)
#         newLazyFn.latex = r"\left\{{{} \textrm{{ if}} \;\; {} \gt {}; \;\;\textrm{{ else 0}} \right\}}".format(fn2mask.latex, lazyFn.latex, alpha)
#     else:
#         newLazyFn.description = "({} if {} < {} else 0".format(fn2mask.description, lazyFn.description, alpha)
#         newLazyFn.latex = r"\left\{{{} \textrm{{ if}} \;\; {} \lt {}; \;\; \textrm{{ else 0}} \right\}}".format(fn2mask.latex, lazyFn.latex, alpha)


#     newLazyFn.derivative = lambda dirn : levelset(fn2mask.derivative(dirn), lazyFn, alpha, invert)

#     return newLazyFn


def maskNaN(lazyFn, invert=False):
    
    assert isinstance(lazyFn, _LazyEvaluation), """
        lazyFn argument is not of type a function"""

    newLazyFn = _LazyEvaluation()

    def threshold(*args, **kwargs):
        if invert:
            values = _np.isnan(lazyFn.evaluate(*args, **kwargs)).astype(float)
        else:
            values = _np.logical_not(_np.isnan(lazyFn.evaluate(*args, **kwargs))).astype(float)

        return values

    newLazyFn.evaluate = threshold
    newLazyFn.dependency_list = lazyFn.dependency_list

    if invert:
        newLazyFn.description = "isNaN({})".format(lazyFn.description)
    else:
        newLazyFn.description = "notNan({})".format(lazyFn.description)

    return newLazyFn


def replaceNan(lazyFn1, lazyFn2):
    
    lazyFn1 = _LazyEvaluation.convert(lazyFn1)
    lazyFn2 = _LazyEvaluation.convert(lazyFn2)

    def replaceNan_nodebynode(*args, **kwargs):

        values1     = lazyFn1.evaluate(*args, **kwargs)
        values2     = lazyFn2.evaluate(*args, **kwargs)  # Note where we evaluate !
        mask_values = _np.isnan(*args, **kwargs)
        replaced_values  = _np.where(mask_values, values2, values1)
        return replaced_values

    newLazyFn = _LazyEvaluation()
    newLazyFn.evaluate = replaceNan_nodebynode
    newLazyFn.description = "Nan([{}]<-[{}])".format(lazyFn1.description, lazyFn2.description)
    newLazyFn.dependency_list = lazyFn1.dependency_list | lazyFn2.dependency_list

    return newLazyFn


# ToDo - add derivative 
def where(maskFn, lazyFn1, lazyFn2):

    maskFn = _LazyEvaluation.convert(maskFn)
    lazyFn1 = _LazyEvaluation.convert(lazyFn1)
    lazyFn2 = _LazyEvaluation.convert(lazyFn2)

    def mask_nodebynode(*args, **kwargs):
        values1     = lazyFn1.evaluate(*args, **kwargs)
        values2     = lazyFn2.evaluate(*args, **kwargs)
        mask_values = maskFn.evaluate(*args, **kwargs)

        replaced_values  = _np.where(mask_values > 0.0, values1, values2)
        return replaced_values

    newLazyFn = _LazyEvaluation()
    newLazyFn.evaluate = mask_nodebynode
    newLazyFn.description = "where({}: [{}]<-[{}])".format(maskFn.description, lazyFn1.description, lazyFn2.description)
    newLazyFn.latex = r"\left\{{{} \textrm{{ if}} \; {} \textrm{{> 0; else}} \;\; {}\right\}}".format(lazyFn1.latex, maskFn.latex, lazyFn2.latex)

    newLazyFn.dependency_list = maskFn.dependency_list | lazyFn1.dependency_list | lazyFn2.dependency_list

    newLazyFn.derivative = lambda dirn: where(maskFn, lazyFn1.derivative(dirn), lazyFn2.derivative(dirn))

    return newLazyFn


def conditional(clauses):

    _clauses = []
    _description = "("
    _dependency_list = set()
    for clause in clauses:
        if not isinstance(clause, (list, tuple)):
            raise TypeError("Clauses within the clause list must be of python type 'list' or 'tuple")
        if len(clause) !=2:
            raise ValueError("Clauses tuples must be of length 2.")
        conditionFn = _LazyEvaluation.convert(clause[0])
        resultantFn = _LazyEvaluation.convert(clause[1])
        _clauses.append((conditionFn, resultantFn))
        _description += "if ({0}) then ({1}), ".format(conditionFn, resultantFn)
        _dependency_list |= conditionFn.dependency_list | resultantFn.dependency_list

    def evaluate(*args, **kwargs):
        mask = _LazyEvaluation().convert(1).evaluate(*args, **kwargs)
        values = None
        for clause in _clauses:
            values = _np.where(clause[0].evaluate(*args, **kwargs) * mask,
                               clause[1].evaluate(*args, **kwargs),
                               values)
            mask = _np.where(clause[0].evaluate(*args, **kwargs) * mask,
                             0, mask) 
        if _np.any(mask):
            raise ValueError("")
        return values
                
    newLazyFn = _LazyEvaluation()
    newLazyFn.evaluate = evaluate
    newLazyFn.description = _description[:-2] + " )" 

    return newLazyFn
