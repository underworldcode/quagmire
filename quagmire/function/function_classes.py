# Copyright 2016-2020 Louis Moresi, Ben Mather, Romain Beucher
# 
# This file is part of Quagmire.
# 
# Quagmire is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or any later version.
# 
# Quagmire is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with Quagmire.  If not, see <http://www.gnu.org/licenses/>.

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

    def __init__(self):
        self.__id = "q_fn_{}".format(self._count())
        self.description = ""
        self.dependency_list = set([self.id])

        return

    def __repr__(self):
        return("quagmire.fn: {}".format(self.description))

    @staticmethod
    def convert(obj):
        """
        This method will attempt to convert the provided input into an
        equivalent quagmire function. If the provided input is already
        of LazyEvaluation type, it is immediately returned. Likewise if
        the input is of None type, it is also returned.

        Parameters
        ----------

        obj: The object to be converted

        Returns
        -------

        LazyEvaluation function or None.
        """

        if isinstance(obj, (LazyEvaluation, type(None))):
            return obj
        else:
            try:
                return parameter(obj)
            except Exception as e:
                raise e

    def evaluate(self, *args, **kwargs):
        raise(NotImplementedError)

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, value):
        self._description = "{}".format(value)

    @property
    def fn_gradient(self):
        return self.derivative

    @property
    def derivative(self):
        """
        Compute values of the derivatives of PHI in the x, y directions at the nodal points.
        This routine uses SRFPACK to compute derivatives on a C-1 bivariate function.

        Parameters
        ----------
        PHI : ndarray of floats, shape (n,)
            compute the derivative of this array
        nit : int optional (default: 10)
            number of iterations to reach convergence
        tol : float optional (default: 1e-8)
            convergence is reached when this tolerance is met

        Returns
        -------
        PHIx : ndarray of floats, shape (n,)
            first partial derivative of PHI in x direction
        PHIy : ndarray of floats, shape (n,)
            first partial derivative of PHI in y direction
        """
        newLazyGradient = LazyGradientEvaluation()
        newLazyGradient.evaluate = self._fn_gradient
        newLazyGradient.description = "d({0})/dX,d({0})/dY".format(self.description)
        newLazyGradient.dependency_list = self.dependency_list
        return newLazyGradient

    def sderivative(self, dirn):
        """
        Compute values of the derivatives of PHI in the x, y directions at the nodal points.
        This routine uses SRFPACK to compute derivatives on a C-1 bivariate function.

        Parameters
        ----------
        dirn : ['0', 'X'] or ['1', 'Y']

        """

        if str(dirn) in "1Y":
            dirn = 1
        else:
            dirn = 0

        raise NotImplementedError
     


    def _fn_gradient(self, *args, **kwargs):

        import quagmire

        if len(args) == 1 and quagmire.mesh.check_object_is_a_q_mesh(args[0]):
            mesh = args[0]
            local_array = self.evaluate(mesh)
            return mesh.derivative_grad(local_array)

        elif len(args) == 1 and hasattr(self, "_mesh"):
            mesh = self._mesh # we are a mesh variable
            local_array = self.evaluate()
            grads = mesh.derivative_grad(local_array)

            coords = np.atleast_2d(args[0])
            
            i0, ierr = mesh.interpolate(coords[:,0], coords[:,1], grads[:,0])
            i1, ierr = mesh.interpolate(coords[:,0], coords[:,1], grads[:,1])
            return np.c_[i0,i1]

        elif hasattr(self, "_mesh"):
            mesh = self._mesh
            local_array = self.evaluate()
            return mesh.derivative_grad(local_array)


        else:
            raise NotImplementedError("why don't you don't supply a mesh...?")

        # if len(args) == 1 and args[0] == diff_mesh:
        #     return df_tuple
        # elif len(args) == 1 and quagmire.mesh.check_object_is_a_q_mesh_and_raise(args[0]):
        #     mesh = args[0]
        #     df_interp = []
        #     for df in df_tuple:
        #         result = diff_mesh.interpolate(mesh.coords[:,0], mesh.coords[:,1], zdata=df, **kwargs)
        #         df_interp.append(result)
        #     return df_interp
        # elif len(args) > 1:
        #     xi = np.atleast_1d(args[0])  # .resize(-1,1)
        #     yi = np.atleast_1d(args[1])  # .resize(-1,1)
        #     df_interp = []
        #     for df in df_tuple:
        #         i, e = diff_mesh.interpolate(xi, yi, zdata=df, **kwargs)
        #         df_interp.append(i)
        #     return df_interp
        # else:
        #     err_msg = "Invalid number of arguments\n"
        #     err_msg += "Input a valid mesh or coordinates in x,y directions"
        #     raise ValueError(err_msg)

## Arithmetic operations

    def __mul__(self, other):
        other = self.convert(other)
        newLazyFn = LazyEvaluation()
        newLazyFn.evaluate = lambda *args, **kwargs : self.evaluate(*args, **kwargs) * other.evaluate(*args, **kwargs)
        newLazyFn.description = "({})*({})".format(self.description, other.description)
        newLazyFn.dependency_list |= self.dependency_list | other.dependency_list

        newLazyFn.sderivative = lambda dirn : self.sderivative (dirn) * other +  \
                                              self * other.sderivative(dirn) 
                     
        return newLazyFn
    
    def __rmul__(self, other):
        other = self.convert(other)
        newLazyFn = LazyEvaluation()
        newLazyFn.evaluate = lambda *args, **kwargs : self.evaluate(*args, **kwargs) * other.evaluate(*args, **kwargs)
        newLazyFn.description = "({})*({})".format(other.description, self.description)
        newLazyFn.dependency_list |= other.dependency_list | self.dependency_list

        newLazyFn.sderivative = lambda dirn : other.sderivative (dirn) * self +  \
                                              other * self.sderivative(dirn) 
  
        return newLazyFn

    def __add__(self, other):
        other = self.convert(other)

        newLazyFn = LazyEvaluation()
        newLazyFn.evaluate = lambda *args, **kwargs : self.evaluate(*args, **kwargs) + other.evaluate(*args, **kwargs)
        newLazyFn.description = "({})+({})".format(self.description, other.description)
        newLazyFn.dependency_list |= self.dependency_list | other.dependency_list

        newLazyFn.sderivative = lambda dirn : self.sderivative(dirn) + other.sderivative(dirn)

        return newLazyFn
    
    def __radd__(self, other):
        other = self.convert(other)
        newLazyFn = LazyEvaluation()
        newLazyFn.evaluate = lambda *args, **kwargs : self.evaluate(*args, **kwargs) + other.evaluate(*args, **kwargs)
        newLazyFn.description = "({})+({})".format(other.description, self.description)
        newLazyFn.dependency_list |= other.dependency_list | self.dependency_list

        newLazyFn.sderivative = lambda dirn : self.sderivative(dirn) + other.sderivative(dirn)

        return newLazyFn

    def __truediv__(self, other):
        other = self.convert(other)
        newLazyFn = LazyEvaluation()
        newLazyFn.evaluate = lambda *args, **kwargs : self.evaluate(*args, **kwargs) / other.evaluate(*args, **kwargs)
        newLazyFn.description = "({})/({})".format(self.description, other.description)
        newLazyFn.dependency_list |= self.dependency_list | other.dependency_list

        newLazyFn.sderivative = lambda dirn : -1.0 *  self * other.sderivative(dirn) / (other * other) + self.sderivative(dirn) / other

        return newLazyFn

    def __sub__(self, other):
        other = self.convert(other)
        newLazyFn = LazyEvaluation()
        newLazyFn.evaluate = lambda *args, **kwargs : self.evaluate(*args, **kwargs) - other.evaluate(*args, **kwargs)
        newLazyFn.description = "({})-({})".format(self.description, other.description)
        newLazyFn.dependency_list |= self.dependency_list | other.dependency_list

        newLazyFn.sderivative = lambda dirn : self.sderivative(dirn) - other.sderivative(dirn)

        return newLazyFn
    
    def __rsub__(self, other):
        other = self.convert(other)
        newLazyFn = LazyEvaluation()
        newLazyFn.evaluate = lambda *args, **kwargs : self.evaluate(*args, **kwargs) - other.evaluate(*args, **kwargs)
        newLazyFn.description = "({})-({})".format(other.description, self.description)
        newLazyFn.dependency_list |= other.dependency_list | self.dependency_list

        newLazyFn.sderivative = lambda dirn : other.sderivative(dirn) - self.sderivative(dirn) 

        return newLazyFn

    def __neg__(self):
        newLazyFn = LazyEvaluation()
        newLazyFn.evaluate = lambda *args, **kwargs : -1.0 * self.evaluate(*args, **kwargs)
        newLazyFn.description = "-({})".format(self.description)
        newLazyFn.dependency_list |= self.dependency_list

        newLazyFn.sderivative = lambda dirn : -1.0 * self.sderivative(dirn)

        return newLazyFn

    def __pow__(self, exponent):
        if isinstance(exponent, (float, int)):
            exponent = parameter(exponent)
        newLazyFn = LazyEvaluation()
        newLazyFn.evaluate = lambda *args, **kwargs : np.power(self.evaluate(*args, **kwargs), exponent.evaluate(*args, **kwargs))
        newLazyFn.description = "({})**({})".format(self.description, exponent.description)
        newLazyFn.dependency_list |= self.dependency_list | exponent.dependency_list
                
        newLazyFn.sderivative = lambda dirn : exponent * self.sderivative(dirn) * (self) ** (exponent-1.0)
        
        return newLazyFn


## Logical operations

## I am not sure how to differentiate these yet !

    def __lt__(self, other):
        other = self.convert(other)
        newLazyFn = LazyEvaluation()
        newLazyFn.evaluate = lambda *args, **kwargs : self.evaluate(*args, **kwargs) < other.evaluate(*args, **kwargs)
        newLazyFn.description = "({})<({})".format(self.description, other.description)
        newLazyFn.dependency_list |= self.dependency_list | other.dependency_list
        return newLazyFn

    def __le__(self, other):
        other = self.convert(other)
        newLazyFn = LazyEvaluation()
        newLazyFn.evaluate = lambda *args, **kwargs : self.evaluate(*args, **kwargs) <= other.evaluate(*args, **kwargs)
        newLazyFn.description = "({})<=({})".format(self.description, other.description)
        newLazyFn.dependency_list |= self.dependency_list | other.dependency_list
        return newLazyFn

    def __eq__(self, other):
        other = self.convert(other)
        newLazyFn = LazyEvaluation()
        newLazyFn.evaluate = lambda *args, **kwargs : self.evaluate(*args, **kwargs) == other.evaluate(*args, **kwargs)
        newLazyFn.description = "({})==({})".format(self.description, other.description)
        newLazyFn.dependency_list |= self.dependency_list | other.dependency_list
        return newLazyFn
    
    def __ne__(self, other):
        other = self.convert(other)
        newLazyFn = LazyEvaluation()
        newLazyFn.evaluate = lambda *args, **kwargs : self.evaluate(*args, **kwargs) != other.evaluate(*args, **kwargs)
        newLazyFn.description = "({})!=({})".format(self.description, other.description)
        newLazyFn.dependency_list |= self.dependency_list | other.dependency_list
        return newLazyFn

    def __ge__(self, other):
        other = self.convert(other)
        newLazyFn = LazyEvaluation()
        newLazyFn.evaluate = lambda *args, **kwargs : self.evaluate(*args, **kwargs) >= other.evaluate(*args, **kwargs)
        newLazyFn.description = "({})>=({})".format(self.description, other.description)
        newLazyFn.dependency_list |= self.dependency_list | other.dependency_list
        return newLazyFn
    
    def __gt__(self, other):
        other = self.convert(other)
        newLazyFn = LazyEvaluation()
        newLazyFn.evaluate = lambda *args, **kwargs : self.evaluate(*args, **kwargs) > other.evaluate(*args, **kwargs)
        newLazyFn.description = "({})>({})".format(self.description, other.description)
        newLazyFn.dependency_list |= self.dependency_list | other.dependency_list
        return newLazyFn
        

class parameter(LazyEvaluation):
    """Floating point parameter / coefficient for lazy evaluation of functions"""

    def __init__(self, value, *args, **kwargs):
        super(parameter, self).__init__(*args, **kwargs)
        self.value = value
        return

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

        if len(args) == 1 and quagmire.mesh.check_object_is_a_q_mesh(args[0]):
            mesh = args[0]
            return self.value * np.ones(mesh.npoints)
        else:
            return self.value

    def sderivative(self, *args, **kwargs):
        return parameter(0.0)


class LazyGradientEvaluation(LazyEvaluation):

    def __init__(self):
        super(LazyGradientEvaluation, self).__init__()


    def __getitem__(self, dirn):
        return self._fn_gradient(dirn=dirn)


    @property
    def fn_gradient(self):
        raise NotImplementedError("gradient operations on the 'LazyGradientEvaluation' sub-class are not supported")

    def _fn_gradient(self, dirn=None):
        """
        The generic mechanism for obtaining the gradient of a lazy variable is
        to evaluate the values on the mesh at the time in question and use the mesh gradient
        operators to compute the new values.
        Sub classes may have more efficient approaches. MeshVariables have
        stored data and don't need to evaluate values. Parameters have Gradients
        that are identically zero ... etc
        """

        def new_fn_dir(*args, **kwargs):
            if len(args) == 1:
                grads = self.evaluate(args[0], **kwargs)
                return grads[:,dirn]
            else:
                grads = self.evaluate()
                return grads[:,dirn]

        d1 = len(self.description)//2
        d2 = d1 + 1
        new_description = [self.description[:d1], self.description[d2:]]

        newLazyFn = LazyEvaluation()
        newLazyFn.evaluate = new_fn_dir
        newLazyFn.description = new_description[dirn]
        return newLazyFn