import sympy
import numpy as np
from sympy.vector import CoordSys3D, CoordSysCartesian
from sympy.core.sympify import sympify
from sympy.core.function import FunctionClass, AppliedUndef
from sympy import Symbol


R = CoordSys3D('R')
R2 = CoordSys3D('R2')

v = 3*R.i + 4*R.j + 5*R.k


F = sympy.Function("F")(R.x, R.y, R.z)

F.data = np.array((10,3))


X, Y, t = sympy.symbols('xi eta t')


## this is based on the sympy UndefinedFunction class


class MeshVar(FunctionClass):
   """
   Mesh Variables
   """

   from sympy.core.function import UndefSageHelper

   _undef_sage_helper = UndefSageHelper()

   def __new__(mcl, name, bases=(AppliedUndef,), __dict__=None, **kwargs):
       from sympy.core.symbol import _filter_assumptions


       from sympy.core.function import UndefSageHelper

       _undef_sage_helper = UndefSageHelper()

       # Allow Function('f', real=True)
       # and/or Function(Symbol('f', real=True))
       assumptions, kwargs = _filter_assumptions(kwargs)
       if isinstance(name, Symbol):
           assumptions = name._merge(assumptions)
           name = name.name
       elif not isinstance(name, str):
           raise TypeError('expecting string or Symbol for name')
       else:
           commutative = assumptions.get('commutative', None)
           assumptions = Symbol(name, **assumptions).assumptions0
           if commutative is None:
               assumptions.pop('commutative')
       __dict__ = __dict__ or {}
       # put the `is_*` for into __dict__
       __dict__.update({'is_%s' % k: v for k, v in assumptions.items()})
       # You can add other attributes, although they do have to be hashable
       # (but seriously, if you want to add anything other than assumptions,
       # just subclass Function)
       __dict__.update(kwargs)
       # add back the sanitized assumptions without the is_ prefix
       kwargs.update(assumptions)
       # Save these for __eq__
       __dict__.update({'_kwargs': kwargs})
       # do this for pickling
       __dict__['__module__'] = None
       obj = super(MeshVar, mcl).__new__(mcl, name, bases, __dict__)
       obj.name = name
       obj._sage_ = _undef_sage_helper
       return obj

   def __instancecheck__(cls, instance):
       return cls in type(instance).__mro__

   _kwargs = {}  # type: tDict[str, Optional[bool]]

   def __hash__(self):
       return hash((self.class_key(), frozenset(self._kwargs.items())))

   def __eq__(self, other):
       return (isinstance(other, self.__class__) and
           self.class_key() == other.class_key() and
           self._kwargs == other._kwargs)

   def __ne__(self, other):
       return not self == other

   @property
   def _diff_wrt(self):
       return False



def qderiv(A,B):

    return np.float(1.0)

A = MeshVar("A")(X,Y)
B = sympy.diff(A,X)



dAdx = sympy.lambdify((X,Y), B , [{"Derivative" : qderiv}, 'math'] )
