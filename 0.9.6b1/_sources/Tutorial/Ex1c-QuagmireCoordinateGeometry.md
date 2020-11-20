---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.6.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

## Quagmire.function.geometry

Quagmire mesh variables are numerically differentiable and the various operators defined in the quagmire functions module also support limited (symbolic) differentiation

This is to allow the construction of more complicated operators that are equivalent on the flat plane and the surface of the spehre. 

This functionality is supported through the `quagmire.function.geometry` package. 

```{code-cell}
import numpy as np
from quagmire.function import display

from quagmire import QuagMesh 
from quagmire import tools as meshtools
from quagmire import function as fn
from quagmire.function import display
from mpi4py import MPI

import lavavu
import stripy
comm = MPI.COMM_WORLD

import matplotlib.pyplot as plt
from matplotlib import cm
%matplotlib inline
```

## Naked coordinate systems

Quagmire supports flat 2D meshes and logically 2D meshes on the surface of a sphere. When a mesh is defined, it attaches a geometry to itself which is the starting point for computing
any differential operators that include mesh variables. It is also possible to access the coordinate system definitions directly to see how the various operators are defined. Let us first
access a simple, 2D, Cartesian Coordinate Geometry

```{code-cell}
CartesianCoordinates = fn.geometry.CartesianCoordinates2D()
SphericalCoordinates = fn.geometry.SphericalSurfaceLonLat2D()
```

```{code-cell}
# These are the coordinate directions is symbolic form. They are also functions that 
# mask the appropriate member of a coordinate tuple

x = CartesianCoordinates.xi0
y = CartesianCoordinates.xi1

print(x.evaluate(1.0,2.0), y.evaluate(1.0,2.0))

# Another example:

points = np.zeros((20,2))
points[:,0] = np.linspace(0.0,2.0*np.pi,20)

print(x.evaluate(points))

S = fn.math.sin(x)

# These are equivalent

print(S.evaluate(points) - np.sin(points[:,0]))
```

### Vector operators

Partial differential equations are often the balancing of gradients of different quantities. The physical gradients may be very different when expressed in different coordinate systems and therefore we have to be careful to develop coordinate-independent expression for gradients that appear in PDEs. We do this by constructing operators such as div, grad, curl and the Laplacian that are expressed independently of the geometry but understand the underlying coordinate system. 

When we calculate a derivative, they should be symbolically equivalent but the gradients are not.

```{code-cell}
for CoordinateSystem in [CartesianCoordinates, SphericalCoordinates]:
    
    xi0 = CoordinateSystem.xi0
    xi1 = CoordinateSystem.xi1
    
    ## Derivatives:
    
    A = xi0 * fn.math.sin(xi0**2) +  xi1 * fn.math.sin(xi1**2) +  xi0 * xi1
    
    print("Derivatives")
    ddx0 = A.derivative(0)
    ddx1 = A.derivative(1)
    display(ddx0)
    display(ddx1)

    gradA = CoordinateSystem.grad(A)
    print("Grad 0")
    gradA[0].display()
    print("Grad 1")
    gradA[1].display()
    
    print("div.grad")
    CoordinateSystem.div(gradA, expand=True).display()
    
    print("Laplacian")
    CoordinateSystem.laplacian(A, 1, expand=True).display()  # Note this is written for variable coefficient problems

    CoordinateSystem.laplacian(A, xi0*xi1, expand=False).display()  ## Neater for display purposes only
```

## Symbols

These are placeholders for developing a function that you can then substitute later. This is most useful as an aid to debugging functions ... 

```{code-cell}
k0 = fn.symbol(name="k_0", lname="k_0")
k1 = fn.symbol(name="k_1", lname="k_1")
display((k0, k1))

H = x * fn.math.sin(k0*x + k0/(k1+x))
display(H)

try:
    H.evaluate((0.1,1.0))
except:
    print("Cannot evaluate H ... try making a subsitution")

k0.substitute(fn.parameter(4))
k1.substitute(fn.parameter(0.0001))


display(k1)
display(H)

print("H evaluated at (0.1,1.0): ",H.evaluate((0.1,1.0)))
```

## The same sort of thing but with some mesh variables

Mesh variables have an associated geometry that they obtain from their mesh and they have some numerical differentiation routines
that are used when a derivative or gradient needs to be evaluated.

```{code-cell}
## Add a mesh

st0 = stripy.spherical_meshes.icosahedral_mesh(refinement_levels=6, include_face_points=True)
dm = meshtools.create_spherical_DMPlex(np.degrees(st0.lons), np.degrees(st0.lats), st0.simplices)
mesh = QuagMesh(dm, downhill_neighbours=1, permute=True)

lon = mesh.geometry.xi0
lat = mesh.geometry.xi1
```

```{code-cell}
F = mesh.add_variable("F", lname="\cal{F}")

l0 = fn.parameter(5.0)
l1 = fn.parameter(2.0)

G = fn.math.cos(l0 * fn.math.radians(lon)) * fn.math.sin(l1 * fn.math.radians(lat))
F.data = G.evaluate(mesh)
```

```{code-cell}
F.display()
G.display()
```

```{code-cell}
display(G.fn_gradient(0))
display(G.derivative(0))
mesh.geometry.grad(G)[0]
```

```{code-cell}
## Something with symbols we can substitute

m0 = fn.symbol(name="m_0", lname="m_0")
m1 = fn.symbol(name="m_1", lname="m_1")

H = fn.math.cos(m0 * fn.math.radians(lon)) * fn.math.sin(m1 * fn.math.radians(lat))
H.display()

m0.substitute(F**2)
m1.substitute(F**4)

H.display()

m0.substitute(F+1)
m1.substitute(F-1)

H.display()

# G.fn_gradient(1)
```

```{code-cell}
H.evaluate(mesh)
```
