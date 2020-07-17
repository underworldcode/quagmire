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


def _make_npmath_op(op, name, lazyFn):
    newLazyFn = _LazyEvaluation(mesh=lazyFn._mesh)
    newLazyFn.evaluate = lambda *args, **kwargs : op(lazyFn.evaluate(*args, **kwargs))
    newLazyFn.gradient = lambda *args, **kwargs : op(lazyFn.gradient(*args, **kwargs))
    newLazyFn.description = "{}({})".format(name, lazyFn.description)
    newLazyFn.dependency_list = lazyFn.dependency_list

    return newLazyFn

## Trig

def coord(dirn):

    def extract_xs(*args, **kwargs):
        """ If no arguments or the argument is the mesh, return the
            coords at the nodes. In all other cases pass through the
            coordinates given """

        if len(args) == 1 and quagmire.mesh.check_object_is_a_q_mesh(args[0]):
            mesh = args[0]
            return mesh.coords[:,0]
        else:
            return args[0]

    def extract_ys(*args, **kwargs):
        """ If no arguments or the argument is the mesh, return the
            coords at the nodes. In all other cases pass through the
            coordinates given """

        if len(args) == 1 and quagmire.mesh.check_object_is_a_q_mesh(args[0]):
            mesh = args[0]
            return mesh.coords[:,1]
        else:
            return args[1]


    newLazyFn_xs = _LazyEvaluation(mesh=None)
    newLazyFn_xs.evaluate = extract_xs
    newLazyFn_xs.description = "X"

    newLazyFn_ys = _LazyEvaluation(mesh=None)
    newLazyFn_ys.evaluate = extract_ys
    newLazyFn_ys.description = "Y"

    if dirn == 0:
        return newLazyFn_xs
    else:
        return newLazyFn_ys


def levelset(lazyFn, alpha=0.5, invert=False):

    assert isinstance(lazyFn, _LazyEvaluation), """
        lazyFn argument is not of type a function"""

    newLazyFn = _LazyEvaluation(mesh=lazyFn._mesh)

    def threshold(*args, **kwargs):
        if not invert:
            values = (lazyFn.evaluate(*args, **kwargs) > alpha).astype(float)
        else:
            values = (lazyFn.evaluate(*args, **kwargs) < alpha).astype(float)

        return values

    newLazyFn.evaluate = threshold
    newLazyFn.dependency_list = lazyFn.dependency_list
    if not invert:
        newLazyFn.description = "(level({}) > {}".format(lazyFn.description, alpha)
    else:
        newLazyFn.description = "(level({}) < {}".format(lazyFn.description, alpha)

    return newLazyFn


def maskNaN(lazyFn, invert=False):
    
    assert isinstance(lazyFn, _LazyEvaluation), """
        lazyFn argument is not of type a function"""

    newLazyFn = _LazyEvaluation(mesh=lazyFn._mesh)

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

    newLazyFn = _LazyEvaluation(mesh=lazyFn1._mesh)
    newLazyFn.evaluate = replaceNan_nodebynode
    newLazyFn.description = "Nan([{}]<-[{}])".format(lazyFn1.description, lazyFn2.description)
    newLazyFn.dependency_list = lazyFn1.dependency_list | lazyFn2.dependency_list

    return newLazyFn


def where(maskFn, lazyFn1, lazyFn2):

    maskFn = _LazyEvaluation.convert(maskFn)
    lazyFn1 = _LazyEvaluation.convert(lazyFn1)
    lazyFn2 = _LazyEvaluation.convert(lazyFn2)

    def mask_nodebynode(*args, **kwargs):
        values1     = lazyFn1.evaluate(*args, **kwargs)
        values2     = lazyFn2.evaluate(*args, **kwargs)
        mask_values = maskFn.evaluate(*args, **kwargs)

        replaced_values  = _np.where(mask_values < 0.5, values1, values2)
        return replaced_values

    newLazyFn = _LazyEvaluation(mesh=lazyFn1._mesh)
    newLazyFn.evaluate = mask_nodebynode
    newLazyFn.description = "where({}: [{}]<-[{}])".format(maskFn.description, lazyFn1.description, lazyFn2.description)
    newLazyFn.dependency_list = maskFn.dependency_list |     lazyFn1.dependency_list | lazyFn2.dependency_list


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
