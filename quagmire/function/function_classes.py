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

## Arithmetic operations

    def __mul__(self, other):
        other = self.convert(other)
        newLazyFn = LazyEvaluation()
        newLazyFn.evaluate = lambda *args, **kwargs : self.evaluate(*args, **kwargs) * other.evaluate(*args, **kwargs)
        newLazyFn.description = "({})*({})".format(self.description, other.description)
        newLazyFn.dependency_list |= self.dependency_list | other.dependency_list
        return newLazyFn
    
    def __rmul__(self, other):
        other = self.convert(other)
        newLazyFn = LazyEvaluation()
        newLazyFn.evaluate = lambda *args, **kwargs : self.evaluate(*args, **kwargs) * other.evaluate(*args, **kwargs)
        newLazyFn.description = "({})*({})".format(other.description, self.description)
        newLazyFn.dependency_list |= other.dependency_list | self.dependency_list
        return newLazyFn

    def __add__(self, other):
        other = self.convert(other)
        newLazyFn = LazyEvaluation()
        newLazyFn.evaluate = lambda *args, **kwargs : self.evaluate(*args, **kwargs) + other.evaluate(*args, **kwargs)
        newLazyFn.description = "({})+({})".format(self.description, other.description)
        newLazyFn.dependency_list |= self.dependency_list | other.dependency_list
        return newLazyFn
    
    def __radd__(self, other):
        other = self.convert(other)
        newLazyFn = LazyEvaluation()
        newLazyFn.evaluate = lambda *args, **kwargs : self.evaluate(*args, **kwargs) + other.evaluate(*args, **kwargs)
        newLazyFn.description = "({})+({})".format(other.description, self.description)
        newLazyFn.dependency_list |= other.dependency_list | self.dependency_list
        return newLazyFn

    def __truediv__(self, other):
        other = self.convert(other)
        newLazyFn = LazyEvaluation()
        newLazyFn.evaluate = lambda *args, **kwargs : self.evaluate(*args, **kwargs) / other.evaluate(*args, **kwargs)
        newLazyFn.description = "({})/({})".format(self.description, other.description)
        newLazyFn.dependency_list |= self.dependency_list | other.dependency_list

        return newLazyFn

    def __sub__(self, other):
        other = self.convert(other)
        newLazyFn = LazyEvaluation()
        newLazyFn.evaluate = lambda *args, **kwargs : self.evaluate(*args, **kwargs) - other.evaluate(*args, **kwargs)
        newLazyFn.description = "({})-({})".format(self.description, other.description)
        newLazyFn.dependency_list |= self.dependency_list | other.dependency_list
        return newLazyFn
    
    def __rsub__(self, other):
        other = self.convert(other)
        newLazyFn = LazyEvaluation()
        newLazyFn.evaluate = lambda *args, **kwargs : self.evaluate(*args, **kwargs) - other.evaluate(*args, **kwargs)
        newLazyFn.description = "({})-({})".format(other.description, self.description)
        newLazyFn.dependency_list |= other.dependency_list | self.dependency_list
        return newLazyFn

    def __neg__(self):
        newLazyFn = LazyEvaluation()
        newLazyFn.evaluate = lambda *args, **kwargs : -1.0 * self.evaluate(*args, **kwargs)
        newLazyFn.description = "-({})".format(self.description)
        newLazyFn.dependency_list |= self.dependency_list

        return newLazyFn

    def __pow__(self, exponent):
        if isinstance(exponent, (float, int)):
            exponent = parameter(float(exponent))
        newLazyFn = LazyEvaluation()
        newLazyFn.evaluate = lambda *args, **kwargs : np.power(self.evaluate(*args, **kwargs), exponent.evaluate(*args, **kwargs))
        newLazyFn.description = "({})**({})".format(self.description, exponent.description)
        newLazyFn.dependency_list |= self.dependency_list | exponent.dependency_list
        return newLazyFn

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

        if len(args) == 1 and quagmire.mesh.check_object_is_a_q_mesh_and_raise(args[0]):
            mesh = args[0]
            return self.value * np.ones(mesh.npoints)
        else:
            return self.value
