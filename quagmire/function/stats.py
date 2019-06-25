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

import quagmire
import numpy as _np
from .function_classes import LazyEvaluation as _LazyEvaluation
from mpi4py import MPI as _MPI
_comm = _MPI.COMM_WORLD

supported_operations = {"MIN" : (_np.min, _MPI.MIN), \
                        "MAX" : (_np.max, _MPI.MAX), \
                        "SUM" : (_np.sum, _MPI.SUM)}

def _get_array_size(lazyFn):
    return lazyFn.evaluate().size

def _all_reduce(array, name):

    np_reduce, mpi_reduce = supported_operations[name]

    # We should check if it is a mesh variable first
    # and use those in-built variables

    larray = _np.array(np_reduce(array))
    garray = larray.copy()
    _comm.Allreduce([larray, _MPI.DOUBLE], [garray, _MPI.DOUBLE], op=mpi_reduce)
    return garray

def _make_reduce_op(name, lazyFn):
    newLazyFn = _LazyEvaluation(mesh=lazyFn._mesh)
    newLazyFn.evaluate = lambda *args, **kwargs : _all_reduce(lazyFn.evaluate(*args, **kwargs), name)
    newLazyFn.gradient = lambda *args, **kwargs : _all_reduce(lazyFn.gradient(*args, **kwargs), name)
    newLazyFn.description = "{}({})".format(name, lazyFn.description)
    newLazyFn.dependency_list = lazyFn.dependency_list
    return newLazyFn


## global-safe reduction operations

def min(lazyFn, axis=None):
    return _make_reduce_op("MIN", lazyFn)

def max(lazyFn, axis=None):
    return _make_reduce_op("MAX", lazyFn)

def sum(lazyFn, axis=None):
    return _make_reduce_op("SUM", lazyFn)

def mean(lazyFn, axis=None):
    from quagmire import function as fn
    size = _get_array_size(lazyFn)
    return sum(lazyFn) / fn.parameter(size)

def std(lazyFn, axis=None):
    from quagmire import function as fn
    size = _get_array_size(lazyFn)
    return fn.math.sqrt(sum((lazyFn - mean(lazyFn))**2)) / fn.parameter(size)

