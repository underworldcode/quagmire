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
        self.latex = ""
        self.dependency_list = set([self.id])
        self._exposed_operator = "S"  # Singleton unless over-ridden with operator
        self.math = lambda : self.latex
        self.coordinate_system = None

        return

    def __repr__(self):
        return("quagmire.fn: {}".format(self.description))

    def _ipython_display_(self):
        from IPython.display import display, Math
        display(Math(self.math()))

    def display(self):
        try:
            self._ipython_display_()
        except:
            print(self.__repr__)

    def validate_coordinate_system(self, obj):

        if self.coordinate_system is not None:
            if obj.coordinate_system is not None:
                assert self.coordinate_system == obj.coordinate_system

    def compatible_coordinate_system(self, obj):

        self.validate_coordinate_system(obj)
        if obj.coordinate_system is not None:
            return obj.coordinate_system
        else:
            return self.coordinate_system


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

        return convert(obj)

    def evaluate(self, *args, **kwargs):
        raise(NotImplementedError)

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, value):
        self._description = "{}".format(value)

    def fn_gradient(self, dirn):
        try:
            return self.coordinate_system.grad(self)[dirn]
        except:
            return None

    @property
    def exposed_operator(self):
        return self._exposed_operator

    @exposed_operator.setter
    def exposed_operator(self, value):
        self._exposed_operator = value
        
    def derivative(self, dirn):
        """
        Compute values of the derivatives of PHI in the x, y directions at the nodal points.
        This routine uses SRFPACK to compute derivatives on a C-1 bivariate function.

        Parameters
        ----------
        dirn : '0' or '1', 0 or 1

        """

        raise NotImplementedError
     

## Arithmetic operations

    def __mul__(self, other):

        other = self.convert(other)

        # Some special cases: 

        if isinstance(other, parameter):
            if other.value == 0.0:
                return other
            if other.value == 1.0:
                return self

            if isinstance(self, parameter):
                return parameter(self.value * other.value)
                
        if isinstance(self, parameter):
            if self.value == 0.0:
                return self
            if self.value == 1.0:
                return other


        ## The exposed operator will also need to be lazy

        fstring  = "({})*" if self.exposed_operator in "+-" else "{}*"
        fstring += "({})" if other.exposed_operator in "+-" else "{}"

        lstring  = "({}) \;" if self.exposed_operator in "+-" else "{} \;"
        lstring += "({})" if other.exposed_operator in "+-" else "{}"

        # Will handle things like float / int combined with lazy operation (or define rmul, rsub etc )
 
        newLazyFn = LazyEvaluation()
        newLazyFn.evaluate = lambda *args, **kwargs : self.evaluate(*args, **kwargs) * other.evaluate(*args, **kwargs)
        newLazyFn.description = fstring.format(self.description, other.description)
        newLazyFn.latex       = lstring.format(self.latex, other.latex)
        newLazyFn.math        = lambda : lstring.format(self.math(), other.math() )
        
        newLazyFn.dependency_list |= self.dependency_list | other.dependency_list
        newLazyFn.exposed_operator = "*"

        newLazyFn.derivative = lambda dirn : self.derivative (dirn) * other +  \
                                              self * other.derivative(dirn) 
                     
        newLazyFn.coordinate_system = self.compatible_coordinate_system(other)

        return newLazyFn

    def __rmul__(self, other):

        other = self.convert(other)

        # Some special cases: 

        if isinstance(other, parameter):
            if other.value == 0.0:
                return parameter(0.0)
            if other.value == 1.0:
                return self
            if isinstance(self, parameter):
                return parameter(self.value * other.value)

        if isinstance(self, parameter):
            if self.value == 0.0:
                return parameter(0.0)
            if self.value == 1.0:
                return other

        newLazyFn = other.__mul__(self)

        return newLazyFn

    def __add__(self, other):

        other = self.convert(other)

        # Some special cases:

        if isinstance(self, parameter) and isinstance(other, parameter):
            return parameter(self.value + other.value)

        if isinstance(other, parameter):
            if other.value == 0.0:
                return self
        if isinstance(self, parameter):
            if self.value == 0.0:
                return other 

        # Otherwise    

        newLazyFn = LazyEvaluation()
        newLazyFn.coordinate_system = self.compatible_coordinate_system(other)

        newLazyFn.evaluate = lambda *args, **kwargs : self.evaluate(*args, **kwargs) + other.evaluate(*args, **kwargs)
        newLazyFn.description = "{} + {}".format(self.description, other.description)
        newLazyFn.latex       = "{} + {}".format(self.latex, other.latex)
        newLazyFn.math        = lambda : r"{} \, + \, {}".format(self.math(), other.math() )

        newLazyFn.dependency_list |= self.dependency_list | other.dependency_list
        newLazyFn.exposed_operator = "+"

        newLazyFn.derivative = lambda dirn : self.derivative(dirn) + other.derivative(dirn)


        return newLazyFn

    def __radd__(self, other):

        other = self.convert(other)

        # Some special cases:

        if isinstance(self, parameter) and isinstance(other, parameter):
            return parameter(self.value + other.value)
 
        if isinstance(other, parameter):
            if other.value == 0.0:
                return self
                
        if isinstance(self, parameter):
            if self.value == 0.0:
                return other 

        # Otherwise    

        newLazyFn = other.__add__(self)
        return newLazyFn

    def __truediv__(self, other):

        other = self.convert(other)

        # Special case:
        if isinstance(self, parameter) and self.value == 0.0:
            return self

        if isinstance(other, parameter) and other.value == 1.0:
            return self 

        fstring  = "({})/" if not self.exposed_operator in "S^" else "{}/"
        fstring += "({})" if not other.exposed_operator in "S^" else "{}"

        newLazyFn = LazyEvaluation()
        newLazyFn.coordinate_system = self.compatible_coordinate_system(other)

        newLazyFn.evaluate = lambda *args, **kwargs : self.evaluate(*args, **kwargs) / other.evaluate(*args, **kwargs)
        newLazyFn.description = "({})/({})".format(self.description, other.description)
        newLazyFn.latex = r"\frac{{ {} }}{{ {}  }}".format(self.latex, other.latex)
        newLazyFn.math  = lambda : r"\frac{{ {} }}{{ {}  }}".format(self.math(), other.math())

        newLazyFn.dependency_list |= self.dependency_list | other.dependency_list
        newLazyFn.exposed_operator = "/"

        newLazyFn.derivative = lambda dirn : -1.0 *  self * other.derivative(dirn) / (other * other) + self.derivative(dirn) / other

        return newLazyFn


    def __rtruediv__(self, other):

        other = self.convert(other)

        # Special case:
        if isinstance(other, parameter) and other.value == 0.0:
            return other

        newLazyFn = other.__truediv__(self)
        return newLazyFn


    def __sub__(self, other):

        other = self.convert(other)

        # Some special cases:

        if isinstance(self, parameter) and isinstance(other, parameter):
            return parameter(self.value - other.value)
 
        if isinstance(other, parameter):
            if other.value == 0.0:
                return self
                
        if isinstance(self, parameter):
            if self.value == 0.0:
                return -other   

        newLazyFn = LazyEvaluation()
        newLazyFn.coordinate_system = self.compatible_coordinate_system(other)

        newLazyFn.evaluate = lambda *args, **kwargs : self.evaluate(*args, **kwargs) - other.evaluate(*args, **kwargs)
        newLazyFn.description = "{} - {}".format(self.description, other.description)
        newLazyFn.latex       = "{} - {}".format(self.latex, other.latex)
        newLazyFn.math  = lambda : "{} - {}".format(self.math(), other.math())

        newLazyFn.dependency_list |= self.dependency_list | other.dependency_list
        newLazyFn.exposed_operator = "-"

        newLazyFn.derivative = lambda dirn : self.derivative(dirn) - other.derivative(dirn)

        return newLazyFn


    def __rsub__(self, other):

        other = self.convert(other)


        # Some special cases:

        if isinstance(self, parameter) and isinstance(other, parameter):
           return parameter(other.value - self.value)
 
        if isinstance(other, parameter):
            if other.value == 0.0:
                return -self

        if isinstance(self, parameter):
            if self.value == 0.0:
                return other   

        newLazyFn = other.__sub__(self)
        return newLazyFn
    
    def __neg__(self):

        if isinstance(self, parameter):
            return parameter(-self.value)

        fstring  = "-{}" if self.exposed_operator in "S^*/" else "-({})"

        newLazyFn = LazyEvaluation()
        newLazyFn.coordinate_system = self.coordinate_system

        newLazyFn.evaluate = lambda *args, **kwargs : -1.0 * self.evaluate(*args, **kwargs)
        newLazyFn.description = "-{}".format(self.description)
        newLazyFn.latex       = "-{}".format(self.latex)
        newLazyFn.math = lambda : "-{}".format(self.math())

        newLazyFn.dependency_list |= self.dependency_list
        newLazyFn.exposed_operator = "S"

        newLazyFn.derivative = lambda dirn : -1.0 * self.derivative(dirn)

        return newLazyFn

    def __pow__(self, exponent):

        if isinstance(exponent, (float, int)):
            exponent = parameter(exponent)

        # special cases:

        if exponent.value == 0.0:
            return parameter(1.0)
        if exponent.value == 1.0:
            return self 

        fstring  = "({})^{}" if not self.exposed_operator in "S" else "{}^{}"
        lstring  = r"\left({}\right)^{{{}}}" if not self.exposed_operator in "S" else r"{}^{{{}}}"

        newLazyFn = LazyEvaluation()
        newLazyFn.coordinate_system = self.coordinate_system

        newLazyFn.evaluate = lambda *args, **kwargs : np.power(self.evaluate(*args, **kwargs), exponent.evaluate(*args, **kwargs))
        newLazyFn.description = fstring.format(self.description, exponent.description)
        newLazyFn.latex       = lstring.format(self.latex, exponent.latex)
        newLazyFn.math = lambda : lstring.format(self.math(), exponent.math())
        newLazyFn.dependency_list |= self.dependency_list | exponent.dependency_list
        newLazyFn.exposed_operator = "^"

        newLazyFn.derivative = lambda dirn : exponent * self.derivative(dirn) * (self) ** (exponent-parameter(1.0))
        
        return newLazyFn


## Logical operations - return Boolean arrays. What happens when interpolated ... pass to level set ?

## I am not sure how to differentiate these yet - perhaps leave as NotImplemented
## Also not sure about LaTeX version of these operations

    def __lt__(self, other):
        other = self.convert(other)
        newLazyFn = LazyEvaluation()
        newLazyFn.coordinate_system = self.compatible_coordinate_system(other)
        newLazyFn.evaluate = lambda *args, **kwargs : self.evaluate(*args, **kwargs) < other.evaluate(*args, **kwargs)
        newLazyFn.description = "({})<({})".format(self.description, other.description)
        newLazyFn.dependency_list |= self.dependency_list | other.dependency_list
        return newLazyFn

    def __le__(self, other):
        other = self.convert(other)
        newLazyFn = LazyEvaluation()
        newLazyFn.coordinate_system = self.compatible_coordinate_system(other)
        newLazyFn.evaluate = lambda *args, **kwargs : self.evaluate(*args, **kwargs) <= other.evaluate(*args, **kwargs)
        newLazyFn.description = "({})<=({})".format(self.description, other.description)
        newLazyFn.dependency_list |= self.dependency_list | other.dependency_list
        return newLazyFn

    def __eq__(self, other):
        other = self.convert(other)
        newLazyFn = LazyEvaluation()
        newLazyFn.coordinate_system = self.compatible_coordinate_system(other)
        newLazyFn.evaluate = lambda *args, **kwargs : self.evaluate(*args, **kwargs) == other.evaluate(*args, **kwargs)
        newLazyFn.description = "({})==({})".format(self.description, other.description)
        newLazyFn.dependency_list |= self.dependency_list | other.dependency_list
        return newLazyFn
    
    def __ne__(self, other):
        other = self.convert(other)
        newLazyFn = LazyEvaluation()
        newLazyFn.coordinate_system = self.compatible_coordinate_system(other)
        newLazyFn.evaluate = lambda *args, **kwargs : self.evaluate(*args, **kwargs) != other.evaluate(*args, **kwargs)
        newLazyFn.description = "({})!=({})".format(self.description, other.description)
        newLazyFn.dependency_list |= self.dependency_list | other.dependency_list
        return newLazyFn

    def __ge__(self, other):
        other = self.convert(other)
        newLazyFn = LazyEvaluation()
        newLazyFn.coordinate_system = self.compatible_coordinate_system(other)
        newLazyFn.evaluate = lambda *args, **kwargs : self.evaluate(*args, **kwargs) >= other.evaluate(*args, **kwargs)
        newLazyFn.description = "({})>=({})".format(self.description, other.description)
        newLazyFn.dependency_list |= self.dependency_list | other.dependency_list
        return newLazyFn
    
    def __gt__(self, other):
        other = self.convert(other)
        newLazyFn = LazyEvaluation()
        newLazyFn.coordinate_system = self.compatible_coordinate_system(other)
        newLazyFn.evaluate = lambda *args, **kwargs : self.evaluate(*args, **kwargs) > other.evaluate(*args, **kwargs)
        newLazyFn.description = "({})>({})".format(self.description, other.description)
        newLazyFn.dependency_list |= self.dependency_list | other.dependency_list
        return newLazyFn
        

class symbol(LazyEvaluation):
    """A placeholder symbol"""

    def __init__(self, name=None, lname=None, *args, **kwargs):
        super(symbol, self).__init__(*args, **kwargs)
        self._lname = lname

        if name is not None:
            self._name = name
        else:
            self._name = "s_{}".format(self.id)
        
        self.description = self._name
        
        if self._lname is not None:
            self.latex = self._lname
        else:
            self.latex = self.description

        self.math = lambda : self.latex  # function returning a string

        return

    def evaluate(self, *args, **kwargs):
        raise RuntimeWarning('Cannot evaluate a symbol - consider substitution') from None
 

    def derivative(self, dirn, *args, **kwargs):

        def cant_evaluate(*args, **kwargs):
            print("Symbols cannot be evaluated", flush=True)
            raise NotImplementedError   

        newLazyFn_dx = symbol()
        newLazyFn_dx.evaluate = cant_evaluate
        newLazyFn_dx.description = "d({})/dX".format(self.description)
        newLazyFn_dx.latex = r"\frac{{ \partial }}{{\partial x}}{}".format(self.latex)
        newLazyFn_dx.math = lambda : newLazyFn_dx.latex
        newLazyFn_dx.exposed_operator = "d"

        newLazyFn_dy = symbol()
        newLazyFn_dy.evaluate = cant_evaluate
        newLazyFn_dy.description = "d({})/dY".format(self.description)
        newLazyFn_dy.latex = r"\frac{{\partial}}{{\partial y}}{}".format(self.latex)
        newLazyFn_dy.math = lambda : newLazyFn_dy.latex
        newLazyFn_dy.exposed_operator = "d"

        if dirn == 0:
            return newLazyFn_dx
        else:
            return newLazyFn_dy

    def substitute(self, lazyFn):

        self.evaluate    = lazyFn.evaluate
        self.derivative  = lazyFn.derivative 
        self.description = lazyFn.description

        before = self.math()
        self.math = lambda : r"\left\{{  {} \leftarrow {}\right\}}".format(before, lazyFn.math())
        self.latex = lazyFn.latex 

class parameter(LazyEvaluation):
    """Floating point parameter / coefficient for lazy evaluation of functions"""

    def __init__(self, value, *args, **kwargs):
        super(parameter, self).__init__(*args, **kwargs)
        self.value = value
        self.math = lambda : self.latex

        return

    def __call__(self, value=None):
        """Set value (X) of this parameter (equivalent to Parameter.value=X)"""
        if value is not None:
            self.value = value
        return

    def __repr__(self):
        return("quagmire lazy evaluation parameter: {}".format( self._value))

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = float(value)
        self.description = "{:.3g}".format(  int(self._value) if self._value.is_integer() else self._value)
        self.latex = self.description

    def evaluate(self, *args, **kwargs):

        if len(args) == 1:
            if quagmire.mesh.check_object_is_a_q_mesh(args[0]):
                mesh = args[0]
                return self.value * np.ones(mesh.npoints)
            else: # could be a tuple or a single np.array object
                coords = np.array(args[0]).reshape(-1, 2)
                return np.ones_like(coords[:,0]) * self.value
        else:  # see if args can be interpreted in the form of coordinate pairs
                coords = np.array(args).reshape(-1, 2)
                return np.ones_like(coords[:,0]) * self.value
 

    def derivative(self, *args, **kwargs):
        return parameter(0.0)

class vector_field(tuple, LazyEvaluation):

    def __new__ (cls, a, b, name=None, lname=None):

        a1 = convert(a)
        b1 = convert(b)

        return super(vector_field, cls).__new__(cls, (a1,b1))

    def __init__(self, a, b, name=None, lname=None):

        LazyEvaluation.__init__(self)

        self.coordinate_system = a.compatible_coordinate_system(b)

        if name is not None:
            self._name = name
        else:
            self._name = "v_{}".format(self.id)

        if name is not None:
            self.description = name 
        else:
            self.description = "({} , {})".format(self[0].description, self[1].description)

        if lname is not None:
            self.latex = lname
            self._lname = lname
        else:
            self.latex = "({} , {})".format(self[0].latex, self[1].latex)
            self._lname = None

        self.math = lambda : self.latex 

    def __repr__(self):
        return self.description

    
    def evaluate(self, *args, **kwargs):

        return ( self[0].evaluate(*args, **kwargs), self[1].evaluate(*args, **kwargs))

    def derivative(self, dirn, expand=False):

        new_vector_field = vector_field( self[0].derivative(dirn), self[1].derivative(dirn))

        if self._lname is None or expand == True:
            new_vector_field.latex = r"\left( {}, {} \right)".format(self[0].derivative(dirn).latex, self[1].derivative(dirn).latex  )
        else:
            new_vector_field.latex = r"\frac{{ \partial }} {{ \partial {} }} {}".format(self.coordinate_system.xi[dirn].latex, self.latex, )

        return new_vector_field

    def __mul__(self, other):

        if isinstance(other, vector_field):
            newLazyFn = self[0] * other[0] + self[1] * other[1]
            return newLazyFn

        else:
            other = convert(other)

            if isinstance(other, LazyEvaluation):
                newVectorField = vector_field( self[0] * other,  self[1] * other )
                return newVectorField

            else:
                raise NotImplementedError

    def __rmul__(self, other):

        other = convert(other)
        if isinstance(other, LazyEvaluation):
            return other.__mul__(self)
        else:
            raise NotImplementedError


    def __add__(self, other):

        if isinstance(other, vector_field):
            newVectorField = vector_field( self[0] + other[0],  self[1] + other[1] )
            return newVectorField

        else:
            other = convert(other)

            if isinstance(other, LazyEvaluation):
                newVectorField = vector_field( self[0] + other,  self[1] + other )
                return newVectorField

            else:
                raise NotImplementedError

    def __radd__(self, other):

        other = convert(other)
        if isinstance(other, LazyEvaluation):
            return other.__add__(self)
        else:
            raise NotImplementedError

    def __neg__(self):

        newVectorField = vector_field( -self[0] ,  -self[1] )
        return newVectorField

    def __sub__(self, other):

        if isinstance(other, vector_field):
            newVectorField = vector_field( self[0] - other[0],  self[1] - other[1] )
            return newVectorField

        else:
            other = convert(other)

            if isinstance(other, LazyEvaluation):
                newVectorField = vector_field( self[0] - other,  self[1] - other )
                return newVectorField

            else:
                raise NotImplementedError

    def __rsub__(self, other):

        other = convert(other)

        if isinstance(other, LazyEvaluation):
            newVectorField = vector_field( other - self[0], other - self[1] )
            return newVectorField

        else:
            raise NotImplementedError

    def __truediv__(self, other): 

        if isinstance(other, vector_field):
            
            raise ValueError("Cannot divide by vector field")
        else:
            newVectorField = vector_field( self[0] / other, self[1] / other )
            return newVectorField


    def __rtruediv__(self, other):

        raise ValueError("Cannot divide by vector field")

    def __pow__(self, exponent):

        raise NotImplementedError
    

    
    


def convert(lazyFnCandidate):
        """
        This method will attempt to convert the provided input into an
        equivalent quagmire function. If the provided input is already
        of LazyEvaluation type, it is immediately returned. Likewise if
        the input is of None type, it is also returned.

        Parameters
        ----------

        lazyFn: The object to be converted

        Returns
        -------

        LazyEvaluation function or None.
        """

        from . import LazyEvaluation, parameter

        if isinstance(lazyFnCandidate, (LazyEvaluation, type(None))):
            return lazyFnCandidate
        else:
            try:
                return parameter(lazyFnCandidate)
            except Exception as e:
                raise e
