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
from .. import function as _fn
from .function_classes import LazyEvaluation as _LazyEvaluation
from .function_classes import parameter as _parameter 
from . import convert as _convert



## define a 3D / 2D coordinate system - 2 basis vectors plus grad div curl del2 etc


class CoordinateSystem2D(object):
    def __init__(self, *args, **kwargs):

        self.set_basis_vectors()
        self.set_grad_operator()
        self.set_div_operator()
        self.set_laplacian_operator()
                   
        return
    
    
    def set_basis_vectors(self):
        
        
        def extract_coord(dirn):
            
            def parse_args(*args, **kwargs):
            
                if len(args) == 1:
                    if  quagmire.mesh.check_object_is_a_q_mesh(args[0]):
                        mesh = args[0]
                        return mesh.coords[:,dirn]
                    else:
                        # coerce to np.array 
                        arr = _np.array(args[0]).reshape(-1,2)
                        return arr[:,1].reshape(-1)
                else:
                    return args[1]

            return parse_args

        
        self.xi0 = _LazyEvaluation()
        self.xi0.evaluate = extract_coord(0)
        self.xi0.description = "x0"
        self.xi0.latex = r"\xi_0"
        self.xi0.math = lambda : self.xi0.latex
        self.xi0.derivative = lambda ddirn : _parameter(1.0) if str(ddirn) in '0' else _parameter(0.0)
        
        self.xi1 = _LazyEvaluation()
        self.xi1.evaluate = extract_coord(1)
        self.xi1.description = "x1"
        self.xi1.latex = r"\xi_1"
        self.xi1.math = lambda : self.xi1.latex
        self.xi1.derivative = lambda ddirn : _parameter(1.0) if str(ddirn) in '1' else _parameter(0.0)
        
    def set_grad_operator(self):
        raise NotImplementedError
        
    def set_div_operator(self):
        raise NotImplementedError
    
    def set_laplacian_operator(self):
        raise NotImplementedError

    
class CartesianCoordinates2D(CoordinateSystem2D):

    def __init__(self, *args, **kwargs):
        
        super(CartesianCoordinates2D, self).__init__(*args, **kwargs)
                                  
        return
    
    def set_basis_vectors(self):
        
        super(CartesianCoordinates2D, self).set_basis_vectors()

        # Can over-ride the standard naming here, for example
          
        self.xi0.description = "x0"
        self.xi0.latex = r"x"
        
        self.xi1.description = "x1"
        self.xi1.latex = r"y"
        
    def set_grad_operator(self):

        def grad(lazyFn):
            lazyFn = _convert(lazyFn)
            if isinstance(lazyFn, _parameter):
                return (_parameter(0), _parameter(0))

            return (lazyFn.derivative(0), lazyFn.derivative(1)) 
        
        self.grad  = grad
        
        def slope(lazyFn):
            lazyFn = _convert(lazyFn)
            if isinstance(lazyFn, _parameter):
                return _parameter(0)

            dx0, dx1 = self.grad(lazyFn)
            return _fn.math.sqrt(dx0**2 + dx1**2)
            
        self.slope    = slope 
        
          
    def set_div_operator(self):
        
        def div(*args):
            if len(args)==1:
                lazyFn_xi0, lazyFn_xi1 = args[0]
            elif len(args)==2:
                lazyFn_xi0 = args[0]
                lazyFn_xi1 = args[1]       
            else:
                raise ValueError
                
            lazyFn_xi0 = _convert(lazyFn_xi0)
            lazyFn_xi1 = _convert(lazyFn_xi1)

            return  (lazyFn_xi0.derivative(0) + lazyFn_xi1.derivative(1))

        self.div = div
        
    def set_laplacian_operator(self):
        
        def lapl(lazyFn1, lazyFnCoeff):

            lazyFnCoeff = _convert(lazyFnCoeff)
            lazyFn1     = _convert(lazyFn1)

            ## Special cases:

            if isinstance(lazyFn1, _parameter):
                return _parameter(0)

            l0, l1 = self.grad(lazyFn1)
            if isinstance(l0, _parameter) and isinstance(l1, _parameter):
                return _parameter(0)


            if isinstance(lazyFnCoeff, _parameter):
                if lazyFnCoeff.value == 1:
                    laplacian = self.div( lazyFnCoeff * l0, lazyFnCoeff * l1)
                    laplacian.math = lambda : r" \nabla^2 \left( {} \right)".format( lazyFn1.math())
                elif lazyFnCoeff.value == 0:
                    laplacian = parameter(0)
                else:
                    laplacian = lazyFnCoeff * self.div( lazyFnCoeff * l0, lazyFnCoeff * l1)
                    laplacian.math = lambda : r"{} \cdot \nabla^2 \left( {} \right)".format(lazyFnCoeff.math(), lazyFn1.math())

            else:
                laplacian = self.div( lazyFnCoeff * l0, lazyFnCoeff * l1)
                laplacian.math = lambda : r"\nabla \left( {} \cdot \nabla \left(  {} \right)\right)".format(lazyFnCoeff.math(), lazyFn1.math())
            
            return laplacian

        
        self.laplacian = lapl
         
        
class SphericalSurfaceLonLat2D(CoordinateSystem2D):
    """Coordinates for navigating the surface of a sphere in 2D - note (Lon, Lat') tuples
       are assumed whereas most mathematical coordinate derivations use colatitude
       We use (\theta, \phi') as the symbols for the coordinate directions.
       """

    def __init__(self, R0=1.0, *args, **kwargs):
        
        super(SphericalSurfaceLonLat2D, self).__init__(*args, **kwargs)
        
        self.R = _parameter(R0)
                                  
        return
    
        
    def set_basis_vectors(self):
        
        super(SphericalSurfaceLonLat2D, self).set_basis_vectors()

        # Can over-ride the standard naming here, for example
                  
        self.xi0.description = "ln"
        self.xi0.latex = r"\theta"
        
        self.xi1.description = "lt"
        self.xi1.latex = r"\phi' "
        
    def set_grad_operator(self):
      
        def grad(lazyFn):
            lazyFn = _convert(lazyFn)

            if isinstance(lazyFn, _parameter):
                return (_parameter(0), _parameter(0))

            return (lazyFn.derivative(0) / (_fn.math.cos(self.xi0) * self.R), lazyFn.derivative(1) / self.R) 
        
        self.grad  = grad
        
        def slope(lazyFn):
            lazyFn = _convert(lazyFn)
            if isinstance(lazyFn, _parameter):
                return _parameter(0)

            dx0, dx1 = self.grad(lazyFn)
            return _fn.math.sqrt(dx0**2 + dx1**2)

        
    def set_div_operator(self):
        
        def div(*args):
            if len(args)==1:
                lazyFn_xi0, lazyFn_xi1 = args[0]
            elif len(args)==2:
                lazyFn_xi0 = args[0]
                lazyFn_xi1 = args[1]       
            else:
                raise ValueError

            lazyFn_xi0 = _convert(lazyFn_xi0)
            lazyFn_xi1 = _convert(lazyFn_xi1)
      
            C = _fn.math.cos(self.xi0)
            newLazyFn = lazyFn_xi0.derivative(0) / (C * self.R) + (lazyFn_xi1 * C).derivative(1) / (C * self.R)
            newLazyFn.math = lambda : r"\nabla \cdot \left( {} , {} \right)".format(lazyFn_xi0.math(), lazyFn_xi1.math())

            return  newLazyFn

        self.div = div
        
    def set_laplacian_operator(self):
        
        def lapl(lazyFn1, lazyFnCoeff):

            lazyFnCoeff = _convert(lazyFnCoeff)
            lazyFn1     = _convert(lazyFn1)
                        

            ## Special cases:

            if isinstance(lazyFn1, _parameter):
                return _parameter(0)

            l0, l1 = self.grad(lazyFn1)
            if isinstance(l0, _parameter) and isinstance(l1, _parameter):
                return _parameter(0)

            if isinstance(lazyFnCoeff, _parameter):
                if lazyFnCoeff.value == 1:
                    laplacian = self.div( lazyFnCoeff * l0, lazyFnCoeff * l1)
                    laplacian.math = lambda : r" \nabla^2 \left( {} \right)".format( lazyFn1.math())
                elif lazyFnCoeff.value == 0:
                    laplacian = parameter(0)
                else:
                    laplacian = self.div( lazyFnCoeff * l0, lazyFnCoeff * l1)
                    laplacian.math = lambda : r"{} \cdot \nabla^2 \left( {} \right)".format(lazyFnCoeff.math(), lazyFn1.math())

            else:
                laplacian = self.div( lazyFnCoeff * l0, lazyFnCoeff * l1)
                laplacian.math = lambda : r"\nabla \left( {} \cdot \nabla \left(  {} \right)\right)".format(lazyFnCoeff.math(), lazyFn1.math())

            
            return laplacian
        
        self.laplacian = lapl
        
    
    
        