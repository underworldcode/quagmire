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
import quagmire
from .function_classes import LazyEvaluation as _LazyEvaluation


def _make_npmath_op(op, name, lazyFn):
    newLazyFn = _LazyEvaluation(mesh=lazyFn._mesh)
    newLazyFn.evaluate = lambda *args, **kwargs : op(lazyFn.evaluate(*args, **kwargs))
    newLazyFn.gradient = lambda *args, **kwargs : op(lazyFn.gradient(*args, **kwargs))
    newLazyFn.description = "{}({})".format(name,lazyFn.description)
    return newLazyFn

## Trig

def coord(dirn):

    def extract_xs(*args, **kwargs):
        """ If no arguments or the argument is the mesh, return the
            coords at the nodes. In all other cases pass through the
            coordinates given """

        if len(args) == 1 and isinstance(args[0], (quagmire.mesh.trimesh.TriMesh, quagmire.mesh.pixmesh.PixMesh) ):
            mesh = args[0]
            return mesh.coords[:,0]
        else:
            return args[0]

    def extract_ys(*args, **kwargs):
        """ If no arguments or the argument is the mesh, return the
            coords at the nodes. In all other cases pass through the
            coordinates given """

        if len(args) == 1 and isinstance(args[0], (quagmire.mesh.trimesh.TriMesh, quagmire.mesh.pixmesh.PixMesh) ):
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

    newLazyFn = _LazyEvaluation(mesh=lazyFn._mesh)

    def threshold(*args, **kwargs):
        if not invert:
            values = (lazyFn.evaluate(*args, **kwargs) > alpha).astype(float)
        else:
            values = (lazyFn.evaluate(*args, **kwargs) < alpha).astype(float)

        return values

    newLazyFn.evaluate = threshold
    if not invert:
        newLazyFn.description = "(level({}) > {}".format(lazyFn.description, alpha)
    else:
        newLazyFn.description = "(level({}) < {}".format(lazyFn.description, alpha)

    return newLazyFn
