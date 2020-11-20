---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.7.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Example 2 - Meshes for Topography 

This notebook introduces the `QuagMesh` object, which builds the following data structures:

- hill slope
- downhill propagation matrices
- upstream area

in addition to the data structures inherited from `QuagMesh`. These form the necessary structures to propagate information from higher to lower elevations. Derivatives are computed on the mesh to calculate the height field, smoothing operators are available to reduce short wavelength features and artefacts.

In this notebook we setup a height field and calculate its derivatives on an unstructued mesh. We smooth the derivatives using the radial-basis function (RBF) smoothing kernel.

> Note: The API for the structured mesh is identical

```{code-cell} ipython3
from quagmire.tools import meshtools
from quagmire import QuagMesh, QuagMesh
from quagmire import function as fn
```

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
# from scipy.ndimage import imread
# from quagmire import tools as meshtools
# from quagmire import QuagMesh
%matplotlib inline
```

```{code-cell} ipython3
minX, maxX = -5.0, 5.0
minY, maxY = -5.0, 5.0,
dx, dy = 0.02, 0.02

x, y, simplices = meshtools.elliptical_mesh(minX, maxX, minY, maxY, dx, dy)

DM = meshtools.create_DMPlex_from_points(x, y, bmask=None)
```

```{code-cell} ipython3
mesh = QuagMesh(DM, downhill_neighbours=1)

print ("Triangulation has {} points".format(mesh.npoints))
```

## Height field

We generate a cylindrically symmetry domed surface and add multiple channels incised along the boundary. The height and slope fields reside as attributes on the `QuagMesh` instance.

```{code-cell} ipython3
radius  = np.sqrt((x**2 + y**2))
theta   = np.arctan2(y,x) + 0.1

height  = np.exp(-0.025*(x**2 + y**2)**2) + 0.25 * (0.2*radius)**4  * np.cos(5.0*theta)**2 ## Less so
height  += 0.5 * (1.0-0.2*radius)
height  += np.random.random(height.size) * 0.01 # random noise
```

```{code-cell} ipython3
# This fails because the topography variable is locked
mesh.topography.data = height

# This unlocks the variable and rebuilds the necessary downhill data structures
with mesh.deform_topography():
    print("Update topography data array (automatically rebuilds matrices)")
    mesh.topography.data = height
    print("Update topography data array (automatically rebuilds matrices ONCE ONLY)")
    mesh.topography.data = height + 0.01
```

```{code-cell} ipython3
mesh.topography.data
```

```{code-cell} ipython3
s = mesh.slope
s
```

## Derivatives and slopes

The slope of the topography is defined through a built in lazy-evaluate function `mesh.slope` (which was described in the Functions notebook). Other gradients are available through the usual quagmire mathematics functions. 

---

If you want more control of the underlying operations, derivatives can also be evaluated on the mesh using the inbuilt routine in the `stripy` object. It employs automatically selected tension factors to preserve shape properties of the data and avoid overshoot and undershoot associated with steep gradients. **Note:** In parallel it is wise to check if this tensioning introduces artefacts near the boundaries.

```python
dfdx, dfdy = mesh.derivative_grad(f, nit=10, tol=1e-8):
```
where `nit` and `tol` control the convergence criteria.

+++

## Smoothing

We have included the capacity to build (Gaussian) Radial Basis Function kernels on the mesh that can be used for smoothing operations. **Radial-basis function** (RBF) smoothing kernel works by setting up a series of gaussian functions based on the distance $d$ between neighbouring nodes and a scaling factor, $\Delta$:

$$
W_i = \frac{\exp \left( \frac{d_i}{\Delta} \right)^2}{\sum_{i} \left( \frac{d_i}{\Delta} \right)^2}
$$

`delta` is set to the mean distance between nodes by default, but it may be changed to increase or decrease the _smoothness_:

```python
rbf1  = mesh.build_rbf_smoother(1.0, 1)
rbf01 = mesh.build_rbf_smoother(0.1, 1)
rbf001 = mesh.build_rbf_smoother(0.01, 1)

print(rbf1.smooth_fn(rainfall, iterations=1).evaluate(0.0,0.0))
print(rbf1.smooth_fn(height, iterations=1).evaluate(0.0,0.0))
print(rbf01.smooth_fn(rainfall, iterations=1).evaluate(0.0,0.0))
```

```{code-cell} ipython3
rbf005 = mesh.build_rbf_smoother(0.05, 1)
rbf010 = mesh.build_rbf_smoother(0.10, 1)
rbf050 = mesh.build_rbf_smoother(0.50, 1)
```

```{code-cell} ipython3
rbf_slope005 = rbf005.smooth_fn(mesh.slope).evaluate(mesh)
rbf_slope010 = rbf010.smooth_fn(mesh.slope).evaluate(mesh)
rbf_slope050 = rbf050.smooth_fn(mesh.slope).evaluate(mesh)
```

**NOTE** - Building the RBF smoothing machinery is expensive and cannot be reused if the kernel properties are changed. We therefore have a two-stage implementation which builds and caches the smoothing matrices and defines a quagmire function that can be used in the usual way.

```{code-cell} ipython3
import lavavu

points = np.column_stack([mesh.tri.points, height])

lv = lavavu.Viewer(border=False, background="#FFFFFF", resolution=[600,600], near=-10.0)

tri1 = lv.triangles("triangles")
tri1.vertices(points)
tri1.indices(mesh.tri.simplices)
tri1.values(mesh.slope.evaluate(mesh), "slope")
tri1.values(rbf_slope005, "smooth_slope_a")
tri1.values(rbf_slope010, "smooth_slope_b")
tri1.values(rbf_slope050, "smooth_slope_c")

tri1.colourmap("#990000 #FFFFFF #000099")
tri1.colourbar()

lv.control.Panel()
lv.control.ObjectList()
tri1.control.List(options=["slope", "smooth_slope_a", "smooth_slope_b", "smooth_slope_c", ], property="colourby", value="slope", command="redraw")

lv.control.show()
```

```{code-cell} ipython3

```
