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


class LazyEvaluation(object):

    __count = 0

    @classmethod
    def _count(cls):
        LazyEvaluation.__count += 1
        return LazyEvaluation.__count

    @property
    def id(self):
        return self.__id


    def __init__(self, mesh=None):
        """Lazy evaluation of mesh variables / parameters
           If no mesh is provided then no gradient function can be implemented"""

        self.__id = "q_fn_{}".format(self._count())

        self.description = ""
        self._mesh = mesh
        self.mesh_data = False
        self.dependency_list = set([self.id])

        return

    def __repr__(self):
        return("quagmire.fn: {}".format(self.description))

    def evaluate(self, *args, **kwargs):
        raise(NotImplementedError)

    def fn_gradient(self, dirn, mesh=None):
        """
        The generic mechanism for obtaining the gradient of a lazy variable is
        to evaluate the values on the mesh at the time in question and use the mesh gradient
        operators to compute the new values.

        Sub classes may have more efficient approaches. MeshVariables have
        stored data and don't need to evaluate values. Parameters have Gradients
        that are identically zero ... etc
        """

        import quagmire

        if self._mesh is None and mesh is None:
            raise RuntimeError("fn_gradient is a numerical differentiation routine based on derivatives of a fitted spline function on a mesh. The function {} has no associated mesh. To obtain *numerical* derivatives of this function, you can provide a mesh to the gradient function. The usual reason for this error is that your function is not based upon mesh variables and can, perhaps, be differentiated without resort to interpolating splines. ".format(self.__repr__()))


        elif self._mesh is not None:
            diff_mesh = self._mesh
        else:
            quagmire.mesh.check_object_is_a_q_mesh_and_raise(mesh)
            diff_mesh = mesh

        def new_fn_x(*args, **kwargs):
            local_array = self.evaluate(diff_mesh)
            dx, dy = diff_mesh.derivative_grad(local_array, nit=10, tol=1e-8)

            if len(args) == 1 and args[0] == diff_mesh:
                return dx

            elif len(args) == 1 and quagmire.mesh.check_object_is_a_q_mesh_and_raise(args[0]):
                mesh = args[0]
                return diff_mesh.interpolate(mesh.coords[:,0], mesh.coords[:,1], zdata=dx, **kwargs)
            else:
                xi = np.atleast_1d(args[0])
                yi = np.atleast_1d(args[1])
                i, e = diff_mesh.interpolate(xi, yi, zdata=dx, **kwargs)
                return i

        def new_fn_y(*args, **kwargs):
            local_array = self.evaluate(diff_mesh)
            dx, dy = diff_mesh.derivative_grad(local_array, nit=10, tol=1e-8)

            if len(args) == 1 and args[0] == diff_mesh:
                return dy
            elif len(args) == 1 and quagmire.mesh.check_object_is_a_q_mesh_and_raise(args[0]):
                mesh = args[0]
                return diff_mesh.interpolate(mesh.coords[:,0], mesh.coords[:,1], zdata=dy, **kwargs)
            else:
                xi = np.atleast_1d(args[0])  # .resize(-1,1)
                yi = np.atleast_1d(args[1])  # .resize(-1,1)
                i, e = diff_mesh.interpolate(xi, yi, zdata=dy, **kwargs)
                return i

        newLazyFn_dx = LazyEvaluation(mesh=diff_mesh)
        newLazyFn_dx.evaluate = new_fn_x
        newLazyFn_dx.description = "d({})/dX".format(self.description)
        newLazyFn_dy = LazyEvaluation(mesh=diff_mesh)
        newLazyFn_dy.evaluate = new_fn_y
        newLazyFn_dy.description = "d({})/dY".format(self.description)

        if dirn == 0:
            return newLazyFn_dx
        else:
            return newLazyFn_dy


    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, value):
        self._description = "{}".format(value)

## Arithmetic operations

    def __mul__(self, other):
        mesh = self._mesh
        if mesh == None:
            mesh = other._mesh
        newLazyFn = LazyEvaluation(mesh=mesh)
        newLazyFn.evaluate = lambda *args, **kwargs : self.evaluate(*args, **kwargs) * other.evaluate(*args, **kwargs)
        newLazyFn.description = "({})*({})".format(self.description, other.description)
        newLazyFn.dependency_list |= self.dependency_list | other.dependency_list
        return newLazyFn

    def __add__(self, other):
        mesh = self._mesh
        if mesh == None:
            mesh = other._mesh
        newLazyFn = LazyEvaluation(mesh=mesh)
        newLazyFn.evaluate = lambda *args, **kwargs : self.evaluate(*args, **kwargs) + other.evaluate(*args, **kwargs)
        newLazyFn.description = "({})+({})".format(self.description, other.description)
        newLazyFn.dependency_list |= self.dependency_list | other.dependency_list

        return newLazyFn

    def __truediv__(self, other):
        mesh = self._mesh
        if mesh == None:
            mesh = other._mesh
        newLazyFn = LazyEvaluation(mesh=mesh)
        newLazyFn.evaluate = lambda *args, **kwargs : self.evaluate(*args, **kwargs) / other.evaluate(*args, **kwargs)
        newLazyFn.description = "({})/({})".format(self.description, other.description)
        newLazyFn.dependency_list |= self.dependency_list | other.dependency_list

        return newLazyFn

    def __sub__(self, other):
        mesh = self._mesh
        if mesh == None:
            mesh = other._mesh
        newLazyFn = LazyEvaluation(mesh=mesh)
        newLazyFn.evaluate = lambda *args, **kwargs : self.evaluate(*args, **kwargs) - other.evaluate(*args, **kwargs)
        newLazyFn.description = "({})-({})".format(self.description, other.description)
        newLazyFn.dependency_list |= self.dependency_list | other.dependency_list

        return newLazyFn

    def __neg__(self):
        newLazyFn = LazyEvaluation(mesh=self._mesh)
        newLazyFn.evaluate = lambda *args, **kwargs : -1.0 * self.evaluate(*args, **kwargs)
        newLazyFn.description = "-({})".format(self.description)
        newLazyFn.dependency_list |= self.dependency_list

        return newLazyFn

    def __pow__(self, exponent):
        if isinstance(exponent, (float, int)):
            exponent = parameter(float(exponent))
        newLazyFn = LazyEvaluation(mesh=self._mesh)
        newLazyFn.evaluate = lambda *args, **kwargs : np.power(self.evaluate(*args, **kwargs), exponent.evaluate(*args, **kwargs))
        newLazyFn.description = "({})**({})".format(self.description, exponent.description)
        newLazyFn.dependency_list |= self.dependency_list | exponent.dependency_list

        return newLazyFn


## need a fn.coord to extract (x or y) ??

# class variable(LazyEvaluation):
#     """Lazy evaluation of Mesh Variables"""
#     def __init__(self, meshVariable):
#         super(variable, self).__init__()
#         self._mesh_variable = meshVariable
#         self.description = meshVariable._name
#         return
#
#     def evaluate(self, *args, coords=None):
#         return self._mesh_variable.evaluate(*args)

class parameter(LazyEvaluation):
    """Floating point parameter / coefficient for lazy evaluation of functions"""

    def __init__(self, value):
        super(parameter, self).__init__()
        self.value = value
        return

    def fn_gradient(self, dirn):
        """Gradients information is not provided by default for lazy evaluation objects:
           it is necessary to implement the gradient method"""

        px = parameter(0.0)
        px.description = "d({})/dX===0.0".format(self.description)
        py = parameter(0.0)
        py.description = "d({})/dY===0.0".format(self.description)

        if dirn == 0:
            return px
        else:
            return py

    def __call__(self, value=None):
        """Set value (X) of this parameter (equivalent to Parameter.value=X)"""
        if value is not None:
            self.value = value
        return

    def __repr__(self):
        return("quagmire lazy evaluation parameter: {}".format(self._value))

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = float(value)
        self.description = "{}".format(self._value)

    def evaluate(self, *args, **kwargs):

        if len(args) == 1 and quagmire.mesh.check_object_is_a_q_mesh_and_raise(args[0]):
            mesh = args[0]
            return self.value * np.ones(mesh.npoints)

        elif any(args):
            xi = np.atleast_1d(args[0])
            yi = np.atleast_1d(args[1])

            return self.value * np.ones_like(xi)

        else:
            return self.value
