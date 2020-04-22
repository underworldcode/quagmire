# -*- coding: utf-8 -*-

import sympy
import quagmire
import numpy as np
from quagmire import function as fn

sympy.init_printing()

X, Y, t = sympy.symbols('xi eta t')

Z = sympy.simplify((X+Y)*(X-Y))
ZZ = sympy.sin(Z)

# sympy.lambdify(args, expr)


F = sympy.lambdify((X,Y), ZZ, modules="numpy" )

print(F(4.0,3.0))



class qsymbol(sympy.Symbol):

    def evaluate(self):
        pass



# class qfunction(sympy.Function):

#     __new__


def qderiv(A,B):

    return np.float(1.0)



H = sympy.Function('height')(X,Y)
H2 = sympy.sin(X)**2 + 3


dhdx = sympy.lambdify((X,Y), H.diff(X), [{"Derivative" : qderiv}, 'scipy', 'numpy'] )
dh2dx = sympy.lambdify((X,Y), H2.diff(X), [{"Derivative" : qderiv}, 'scipy', 'numpy'] )



class meshVariable(sympy.Function):
    """mesh variable"""

    def __new__(cls, name, *args, **options):
        # instance = sympy.Function(name)(X,Y)
        instance = super(meshVariable, cls).__new__(cls, name, *args, **options)
        return instance


    def __init__(self, *args, **options):
        """ Who am i ?"""

        print("INIT")

        # self.dims = dims



A = meshVariable("A", (X,Y))





# from sympy.core.cache import cacheit

# class Dimension(sympy.Symbol):

#     def __new_stage2__(cls, name, minVal, maxVal):
#         obj = super(Dimension, cls).__xnew__(cls, name, real=True)
#         obj.params = (minVal, maxVal)
#         # obj.grid = numpy.linspace(start, stop, points, endpoint=False)
#         return obj

#     def __new__(cls, name, *args, **kwds):
#         obj = Dimension.__xnew_cached_(cls, name, *args, **kwds)
#         return obj

#     __xnew__ = staticmethod(__new_stage2__)
#     __xnew_cached_ = staticmethod(cacheit(__new_stage2__))

#     def _hashable_content(self):
#         return (Symbol._hashable_content(self), self.params)


# D = Dimension("delta", 0, 1.0)

# B = sympy.Function("\\varepsilon")(D)


# def Variation(path_, st_, en_, v_):

#     class Variation(sympy.Function):

#         nargs = 2

#         path = path_
#         st = st_
#         en = en_
#         ends = [st, en]
#         v = v_

#         @classmethod
#         def eval(cls, tt, ss):
#             if tt in cls.ends:
#                 return cls.path(tt)

#             if ss == 0:
#                 return cls.path(tt)

#             return cls.v(tt, ss)

#         def __init__(self):

#             print("INIT V")

#             return

#     return Variation

# s,t,a,b = sympy.symbols('s t a b')
# c = sympy.Function('c')

# Var = Variation(c, a, b, sympy.Function(r'\vartheta'))



# T = sympy.Function("T")(X,Y,t)

# eta = 10.0 * sympy.exp(-2.0*T)





