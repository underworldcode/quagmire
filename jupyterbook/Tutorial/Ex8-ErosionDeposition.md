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

# Example 8 - Incision and Deposition

This notebook explores three laws to simulate erosion and deposition. All augment the so-called "stream power law", which is a flux term related the available energy of rivers. The stream power law forms the basis of landscape evolution models. Various authors propose different behaviours from detachment-limited to transport-limited sediment transport, some of which we will explore in this notebook.


### Contents

1. Local equilibrium model
2. Saltation length model
3. $\xi - q$ model

```{code-cell}
from quagmire import QuagMesh
from quagmire import tools as meshtools
from quagmire import function as fn
from quagmire import equation_systems as systems
import quagmire
import numpy as np
import matplotlib.pyplot as plt
from time import time

%matplotlib inline
```

```{code-cell}
minX, maxX = -5.0, 5.0
minY, maxY = -5.0, 5.0,

spacingX = 0.02
spacingY = 0.02

x, y, simplices = meshtools.elliptical_mesh(minX, maxX, minY, maxY, spacingX, spacingY, random_scale=1.0)
DM = meshtools.create_DMPlex(x, y, simplices)

mesh = QuagMesh(DM)

print( "\nNumber of points in the triangulation: {}".format(mesh.npoints))
print( "Downhill neighbour paths: {}".format(mesh.downhill_neighbours))
```

```{code-cell}
x = mesh.coords[:,0]
y = mesh.coords[:,1]
boundary_mask_fn = fn.misc.levelset(mesh.mask, 0.5)

radius  = np.sqrt((x**2 + y**2))
theta   = np.arctan2(y,x) + 0.1

height  = np.exp(-0.025*(x**2 + y**2)**2)
height -= height.min()

with mesh.deform_topography():
    mesh.downhill_neighbours = 2
    mesh.topography.data = height

rainfall_fn = mesh.topography ** 2.0
```

## Stream power law

The incision rate, written in the so-called stream power form, is,

$$
\dot{h}(\mathbf{x})_\textrm{incision} = K(\mathbf{x}) q_r(\mathbf{x})^m \left| \nabla h(\mathbf{x}) \right|^n
$$

where

- $q_r$ is the runoff flux
- $\left| \nabla h(\mathbf{x}) \right|$ is the slope
- $K$ is the erodability
- $m$ is the stream power exponent
- $n$ is the slope exponent

The runoff flux can be calculated from the upstream integral of runoff for all areas upstream of the point $\mathbf{x}$,

$$
q_r(\mathbf{x})  = \int_{\mathrm{upstream}} R(\xi) \mathrm{d}\xi
$$

We can compute this by assembling a function and evaluating it on the mesh

```{code-cell}
# vary these and visualise difference
m = fn.parameter(1.0)
n = fn.parameter(1.0)
K = fn.parameter(1.0)

# create stream power function
upstream_precipitation_integral_fn = mesh.upstream_integral_fn(rainfall_fn)
stream_power_fn = K*upstream_precipitation_integral_fn**m * mesh.slope**n * boundary_mask_fn

# evaluate on the mesh
stream_power = stream_power_fn.evaluate(mesh)
```

```{code-cell}
import lavavu

verts = np.reshape(mesh.tri.points, (-1,2))
verts = np.insert(verts, 2, values=mesh.topography.data, axis=1)

# setup viewer
lv = lavavu.Viewer(border=False, background="#FFFFFF", resolution=[1000,600], near=-10.0)

tri1 = lv.triangles("triangles", wireframe=False)
tri1.vertices(verts)
tri1.indices(mesh.tri.simplices)

tri1.values(stream_power, "stream_power")

tri1.colourmap("drywet")
tri1.colourbar()
lv.window()
```

## Erosion and deposition

The erosion rate is controlled by $m$ and $n$ which augment the incision done by runoff flux compared to bed slope. The deposition rate is related to the amount of material eroded and carried downstream. In the simplest case we assume the local deposition rate is the amount of material that can be eroded from upstream, but later we will see that eroded material may be suspended over a certain length scale before it is deposited downstream.

+++

### 1. Local equilibrium

The assumption of the stream power law is that sediment transport is in a state of local equilibrium in which the transport rate is (less than or) equal to the local carrying capacity. If we neglect suspended-load transport for a moment and assume only bed-load transport then the local deposition is the amount of material that can be eroded from upstream.

```{code-cell}
efficiency = fn.parameter(1.0)

erosion_rate_fn = efficiency*stream_power_fn
deposition_rate_fn = mesh.upstream_integral_fn(erosion_rate_fn)

# combined rate of change
dHdt_fn1 = deposition_rate_fn - erosion_rate_fn
```

### 2. Saltation length

This model relates the length of time it takes for a grain to settle to a material property, $L_s$.
From Beaumont et al. 1992, Kooi & Beaumont 1994, 1996 we see a linear dependency of deposition flux to stream capacity:

$$
\frac{dh}{dt} = \frac{dq_s}{dl} = \frac{D_c}{q_c} \left(q_c - q_s \right)
$$

where

$$
\frac{D_c}{q_c} = \frac{1}{L_s}
$$

$D_c$ is the detachment capacity, $q_c$ is the carrying capacity, $q_s$ is the stream capacity, and $L_s$ is the erosion length scale (a measure of the detachability of the substrate). When the flux equals capacity, $q_c = q_s$, no erosion is possible.

```{code-cell}
efficiency = fn.parameter(1.0)
length_scale = fn.parameter(10.0)

erosion_rate = efficiency*stream_power_fn
deposition_rate = mesh.upstream_integral_fn(erosion_rate_fn)

# combined rate of change
dHdt_fn2 = (deposition_rate - erosion_rate_fn)/length_scale
```

### 3. $\xi - q$ model

Davy and Lague (2009) propose a similar suspended-load model that encapsulates a range of behaviours between detachment and transport-limited end members. This model couples erodability as a function of stream power with a sedimentation term weighted by $\alpha$.

$$
\frac{dh}{dt} = -K q_r^m S^n + \frac{Q_s}{\alpha Q_w}
$$

where $Q_s$ and $Q_w$ are the sedimentary and water discharge, respectively.

```{code-cell}
efficiency = fn.parameter(1.0)
alpha = fn.parameter(0.5)
r = fn.parameter(1)

erosion_rate = efficiency*stream_power_fn
deposition_rate = mesh.upstream_integral_fn(erosion_rate_fn)/(alpha*stream_power_fn)

dHdt_fn3 = deposition_rate - erosion_rate
```

```{code-cell}
import lavavu

verts = np.reshape(mesh.tri.points, (-1,2))
verts = np.insert(verts, 2, values=mesh.topography.data, axis=1)

# setup viewer
lv = lavavu.Viewer(border=False, background="#FFFFFF", resolution=[1000,600], near=-10.0)

tri1 = lv.triangles("triangles", wireframe=False)
tri1.vertices(verts)
tri1.indices(mesh.tri.simplices)

tri1.values(dHdt_fn1.evaluate(mesh), "EroDep1")
tri1.values(dHdt_fn2.evaluate(mesh), "EroDep2")
tri1.values(dHdt_fn3.evaluate(mesh), "EroDep3")

#Create colour bar then load a colourmap into it
tri1.colourmap([(0, 'blue'), (0.2, 'white'), (1, 'orange')], reverse=True)
tri1.colourbar(size=[0.95,15])
tri1.control.List(options=
                 ["EroDep1", "EroDep2", "EroDep3"], 
                  property="colourby", value="kappa", command="redraw")
lv.window()
```

All of these erosion-deposition laws are built-in functions that can be accessed by instanting the `ErosionDepositionEquation` object:

```python
quagmire.equation_systems.ErosionDepositionEquation(
    mesh=None,
    rainfall_fn=None,
    m=1.0,
    n=1.0,
)
```

A timestepping routine is available to be used in conjunction with `DiffusionEquation` to form the necessary components of a landscape evolution model.

```{code-cell}

```
