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

# quagmire.mesh MeshVariable

Like Underworld, quagmire provides the concept of a "variable" which is associated with a mesh. These are parallel data structures on distributed meshes that support various capabilities such as interpolation, gradients, save and load, as well as supporting a number of mathematical operators through the `quagmire.function` interface (examples in the next notebook).

```{code-cell} ipython3
from quagmire.tools import meshtools
from quagmire import QuagMesh
from quagmire.mesh import MeshVariable
import numpy as np  
```

## Working mesh

First we create a basic mesh so that we can define mesh variables and obtain gradients etc.

```{code-cell} ipython3
minX, maxX = -5.0, 5.0
minY, maxY = -5.0, 5.0,
dx, dy = 0.02, 0.02

x,y, bound = meshtools.generate_elliptical_points(minX, maxX, minY, maxY, dx, dy, 60000, 300)
DM = meshtools.create_DMPlex_from_points(x, y, bmask=bound)
mesh = QuagMesh(DM, downhill_neighbours=1)
```

## Basic usage

Mesh variables can be instantiated directly or by adding a new variable to an existing mesh. 
`print` will display and expanded description of the variable.

Note, for use in jupyter notebooks (etc), you can add a latex description (see below)and it
will be displayed "nicely". The coordinates are added automatically because many things depend
on the geometrical context (spherical v. flat). Note that the use of rstrings (`r"\LaTeX"`) to make sure the 
names are not corrupted due to unexpected special characters.

```{code-cell} ipython3
phi = mesh.add_variable(name="PHI(X,Y)", lname=r"\phi")
psi = mesh.add_variable(name="PSI(X,Y)", lname=r"\psi")

# is equivalent to

phi1 = MeshVariable(name="PSI(X,Y)", mesh=mesh)
psi1 = MeshVariable(name="PHI(X,Y)", mesh=mesh)

print(psi)
```

```{code-cell} ipython3

```

```{code-cell} ipython3
## latex / jupyter additions:

phi = mesh.add_variable(name="PHI(X,Y)", lname=r"\phi")
psi = mesh.add_variable(name="PSI(X,Y)", lname=r"\psi")

print("Printed version:")
print(phi)
print("\n")
print("Displayed version (latex)")
phi.display()
print("\n")

# Or just display the object using jupyter
print("Automatically displayed version (latex in notebook, printed in python)")
phi
```

Mesh variables store their data in a PETSc distributed vector with values on the local mesh accessible through a numpy interface (via to petsc4py). For consistency with `underworld`, the numpy array is accessed as the `data` property on the variable as follows

```{code-cell} ipython3
phi.data = np.sin(mesh.coords[:,0])**2.0 
psi.data = np.cos(mesh.coords[:,0])**2.0 * np.sin(mesh.coords[:,1])**2.0 
```

Note that the following is not allowed

```python
phi.data[0] = 1.0
```

and nor is any other change to a single value in the array. This is done so that we can be sure that
the values in the array are always synchronised across processors at the end of an assignment. It is also
done to control cases where there are dependencies on the variable that go beyond synchronisation (for example,
changing the topography variable rebuilds the flow pathways in a surface process context). 

You can work with a local copy of the vector and update all at once if you need to build incremental changes to values, work without synchronisation across processors or avoid rebuilding of dependent quantities.

+++

A MeshVariable object responds to a `print` statement by stating what it is and its name. To print the contents of the variable (locally), access the values through the data property:

```{code-cell} ipython3
print(phi, "|", psi)
print(phi.data)
```

Mesh variables can be read only (locked). The RO (read only) and RW (read / write) markers are shown when the variable is printed. 

```python
phi.lock()
print(phi)
phi.unlock()
print(phi)
```

Generally locking is done to prevent changes to a variable's data because additional updates depend on controlling when changes are made. Access to `lock` and `unlock` is

```{code-cell} ipython3
phi.lock()
print(phi)
phi.unlock()
print(phi)
```

## Parallel support

The `MeshVariable` class has a `sync` method that, when called, will replace shadow information with values from adjacent sections of the decomposition (or optionally, merge values in the shadow zone - an operation that should be used with caution for global reduction type operations). 

If you alter data in the shadow zone in a way that cannot be guaranteed to be the same on another processor, then some form of synchronisation is required when you are done. This is not fully automated as there may be several stages to your updates that you only want to synchronise at the end. But, still, be careful !

```{code-cell} ipython3
phi.sync()

phi.sync(mergeShadow=True) # this will add the values from each processor in parallel
```

These kinds of parallel operations must be called on every processor or they will block while waiting for everyone to finish. Be careful not to call sync inside a conditional which may be executed differently 

```python

import quagmire

# Don't do this (obviously)
if quagmire.rank == 0:
    phi.sync()   
   
# or something a little bit less obvious
if delta_phi > 0.0:
    phi.sync()
    
# This might be OK but it is not required
if quagmire.size > 1:
    phi.sync()

```

+++

## Evaluate method and fn_gradient

MeshVariables support the `evaluate` method (because they are `quagmire.functions`) which is useful as it generalises various interfaces that are available to access the data. If a mesh is supplied, then evaluate checks to see if this corresponds to the mesh associated with the mesh variable and returns the raw data if it does. Otherwise the mesh coordinates are used for interpolation. If two coordinate arrays are supplied then these are passed to the interpolator. 

NOTE: the interpolator will also extrapolate and may (is likely to) produce poor results for off-processor coordinates. If this is a problem, the `MeshVariable.interpolate` method can be accessed directly with the `err` optional argument.

```{code-cell} ipython3
## Raw nodal point data for the local domain

print(phi.data)
print(phi.evaluate(mesh))
print(phi.evaluate(phi._mesh)) 
print(phi.evaluate()) 


## interpolation at a point 

print(phi.interpolate(0.01,1.0))
print(phi.evaluate((0.01, 1.0)))
```

## Derivatives / gradients

Mesh based variables can be differentiated in (X,Y). There is a `_gradient` method inherited from the `stripy` package that supplies the coefficients of the derivative surface at the nodal points (these may then need to be interpolated). A more general interface is also provided in the form of functions which not only compute the gradient but also handle interpolation between meshes and are also evaluated on demand (i.e. can be composed into functions).

```{code-cell} ipython3
dpsi = psi._gradient()
dpsidx_nodes = dpsi[:,0]
dpsidy_nodes = dpsi[:,1]
print("dPSIdx - N: ",  dpsidx_nodes)
print("dPSIdy - N: ",  dpsidy_nodes)

dpsidx_fn = psi.fn_gradient(0) # (0) for X derivative, (1) for Y
print("dPSIdx - F: ",  dpsidx_fn.evaluate(mesh))
print("dPSIdx - point evaluation ",  dpsidx_fn.evaluate((0.01, 1.0)))
```

## Higher derivatives

The easiest way to form higher derivatives for a mesh variable is to use the nesting properties of the gradient functions. 
These form gradients and use the same mesh to find *their* gradients. However, the do so in a way that does not require
defining and handling intermediary mesh variables. 

Let's take a look to see how these methods work:

```{code-cell} ipython3
## This way uses the underlying mesh structure to extract gradients and store the result

dpsidx_var = mesh.add_variable("dpsidx")
dpsidx_var.data = dpsidx_nodes
dpsidy_var = mesh.add_variable("dpsidy")
dpsidy_var.data = dpsidy_nodes


print( dpsidx_var._gradient()[:,0] )


## And this way is function based (two equivalent interfaces)

d2psidx2_fn  = dpsidx_fn.fn_gradient(0)
d2psidx2_fn2 = dpsidx_fn.derivative(0)

print( d2psidx2_fn.evaluate(mesh))
print( d2psidx2_fn2.evaluate(mesh))
```

The function based method is more simple and has some advantages if the data in the original variable change - the gradient function 
handles those updates automatically. Note how the mesh variable version does not update whereas the function based version does. We will see
in the next example notebook how the function system works in detail. For now, though, simply note that the function is not evaluated
until it is needed and so can be defined in an abstract manner independent of the data in the variable.

```{code-cell} ipython3
## change the data in the PSI variable:

psi.data = 2.0 * np.cos(mesh.coords[:,0])**2.0 * np.sin(mesh.coords[:,1])**2.0 

print( dpsidx_var._gradient()[:,0] )
print(  0.5 * d2psidx2_fn.evaluate(mesh) )
print( (0.5 * d2psidx2_fn).evaluate(mesh))
```

This is how we can define a Laplacian of the variable (in Cartesian geometry) using the function interface. This approach allows
the operator to be a self-updating variable that can be passed around as though it was a simple mesh variable. Note that the
functions have some helpful associated descriptive text that explains what they are. 

Also note, this is not a generic operator as it is specific to this variable but more general operators can be constructed 
with very little overhead because the interface is very lightweight.

```{code-cell} ipython3
laplace_phi_xy = phi.derivative(0).derivative(0) + phi.derivative(1).derivative(1)
print(laplace_phi_xy)

print(laplace_phi_xy.evaluate(mesh))
```

## Visualisation

+++

The following should all evaluate to zero everywhere and so act as a test on the accuracy of the gradient operator

```{code-cell} ipython3
import lavavu

xyz = np.column_stack([mesh.tri.points, np.zeros_like(phi.data)])

lv = lavavu.Viewer(border=False, background="#FFFFFF", resolution=[1000,600], near=-10.0)

tris = lv.triangles("triangles",  wireframe=False, colour="#77ff88", opacity=1.0)
tris.vertices(xyz)
tris.indices(mesh.tri.simplices)
tris.values(phi.evaluate(mesh), label="phi")
tris.values(psi.evaluate(mesh), label="psi")
tris.values(dpsidx_nodes, label="dpsidx_nodes")



tris.colourmap("elevation")
cb = tris.colourbar()

lv.control.Panel()
lv.control.Range('specular', range=(0,1), step=0.1, value=0.4)
lv.control.Checkbox(property='axis')
lv.control.ObjectList()
tris.control.Checkbox(property="wireframe")
tris.control.List(options = ["phi", "psi", "dpsidx_nodes"], property="colourby", value="psi", command="redraw", label="Display:")
lv.control.show()
```

## Saving and loading mesh variables

There are 2 equivalent ways to save a mesh (PETSc DM) to a file. The first is to call `meshtools.save_DM_to_hdf5` and the second is to call the mesh object's own method `mesh.save_mesh_to_hdf5(filename)`. Mesh variables have a similar save method

```{code-cell} ipython3
mesh.save_mesh_to_hdf5("Ex1a-circular_mesh.h5")
psi.save("Ex1a-circular_mesh_psi.h5")
phi.save("Ex1a-circular_mesh_phi.h5")
```

We can then use these files to:
  - Build a new copy of the mesh
  - Add new mesh variables to that mesh
  - read the values back in

```{code-cell} ipython3
DM2 = meshtools.create_DMPlex_from_hdf5("Ex1a-circular_mesh.h5")
mesh2 = QuagMesh(DM2)

print(mesh.npoints, mesh2.npoints)

phi2 = mesh2.add_variable(name="PHI(X,Y)")
psi2 = mesh2.add_variable(name="PSI(X,Y)")

psi2.load("Ex1a-circular_mesh_psi.h5")
phi2.load("Ex1a-circular_mesh_phi.h5")
```

## Mesh variable save / load and names

The names that are stored in the mesh variable hdf5 file are needed to retrieve the information again. That means the mesh variable that is loaded needs to match the one that was saved *Exactly*. This will not work:

``` python
psi3 = mesh2.add_variable(name="PSI")
psi3.load("Ex1a-circular_mesh_psi.h5")
```

but, as long as you know the name of the original MeshVariable, you can do this:


``` python
psi3.load("Ex1a-circular_mesh_psi.h5", name="PSI(X,Y)")
```

```{code-cell} ipython3
psi3 = mesh2.add_variable(name="PSI")
psi3.load("Ex1a-circular_mesh_psi.h5", name="PSI(X,Y)")
psi3.data
```

```{code-cell} ipython3
print(phi.data[0], psi.data[0])
```
