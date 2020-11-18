# ---
# jupyter:
#   jupytext:
#     formats: Notebooks/IdealisedExamples//ipynb,Examples/IdealisedExamples//py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## Rosenbrock landscape
#
# The Rosenbrock function is a typical non-convex bivariate that is commonly used to benchmark optimisation routines. It parameratises two minimas where streams diverge around a saddle point. This showcases Quagmire's ability to direct flow to multiple downhill neighbours.
#
# Here we explore 1 and 2 downhill neighbour pathways on an unstructured mesh using this function.
#
# #### Contents
#
# - [Rosenbrock function](#Rosenbrock-function)
# - [Compare one and two downhill pathways](#Compare-one-and-two-downhill-pathways)
# - [Probability densities](#Probability-densities)
# - [Animation of stream flow](#Animation-of-stream-flow)

import numpy as np
import matplotlib.pyplot as plt
from quagmire import QuagMesh
from quagmire import tools as meshtools
import scipy as sp
# %matplotlib inline

# +
minX, maxX = -2.0, 2.0
minY, maxY = -2.0, 3.0

x, y, bmask = meshtools.generate_square_points(minX, maxX, minY, maxY, 0.05, 0.05, samples=10000, boundary_samples=1000)
x, y = meshtools.lloyd_mesh_improvement(x, y, bmask, 5)

DM = meshtools.create_DMPlex_from_points(x, y, bmask)
sp = QuagMesh(DM)

x = sp.tri.points[:,0]
y = sp.tri.points[:,1]
# -

# ## Rosenbrock function
#
# The height field is defined by the Rosenbrock function:
#
# $$
# h(x,y) = (1-x)^2 + 100(y-x^2)^2
# $$
#
# we introduce a small incline to ensure the streams terminate at the boundary.

# +
height = (1.0 - x)**2 + 100.0*(y - x**2)**2 # Rosenbrock function
height -= 100*y # make a small incline

with sp.deform_topography():
    sp.topography.data = height

gradient = sp.slope.evaluate(sp)

# +
import lavavu

lv = lavavu.Viewer(border=False, resolution=[666,666], background="#FFFFFF")
lv["axis"]=True
lv['specular'] = 0.5

verts = np.reshape(sp.tri.points, (-1,2))
verts = np.insert(verts, 2, values=sp.topography.data / 1000.0, axis=1)

tris  = lv.triangles("spmesh", wireframe=True,  logScale=False, colour="Red")
tris.vertices(verts)
tris.indices(sp.tri.simplices)
tris.values(sp.topography.data / 1000.0, label="height")
tris.values(gradient / 1000.0, label="slope")
tris.colourmap(["(-1.0)Blue (-0.5)Green (0.0)Yellow (1.0)Brown (5.0)White"])


nodes = lv.points("vertices", pointsize=1.0)
nodes.vertices(verts)
nodes.values(sp.bmask)

lv.control.Panel()
tris.control.List(options=["height", "slope"], property="colourby", value="height", command="redraw", label="Display:")
lv.control.show()
# -


tris["colourby"]="slope"
lv.commands("redraw")

# ## Compare one and two downhill pathways
#
# The Rosenbrock function encapsulates a Y-junction where a river splits in two.

# +
sp.downhill_neighbours = 2
down2 = sp.downhillMat.copy() # 2 downhill neighbours

sp.downhill_neighbours = 1
down1 = sp.downhillMat.copy() # 1 downhill neighbour


# compute upstream area for each downhill matrix


sp.downhillMat = down1
upstream_area1 = sp.cumulative_flow(sp.area)

sp.downhillMat = down2
upstream_area2 = sp.cumulative_flow(sp.area)

# +
lv = lavavu.Viewer(border=False, resolution=[666,666], background="#FFFFFF")
lv["axis"]=True
lv['specular'] = 0.5


tris  = lv.triangles("spmesh", wireframe=False,  logScale=False, colour="Red")
tris.vertices(verts)
tris.indices(sp.tri.simplices)
tris.values(upstream_area1, "upstream1")
tris.values(upstream_area2, "upstream2")
tris.colourmap("drywet")


nodes = lv.points("vertices", pointsize=1.0)
nodes.vertices(verts)
nodes.values(sp.bmask)

lv.control.Panel()
tris.control.List(options=["upstream1", "upstream2"], property="colourby", value="height", command="redraw", label="Display:")
lv.control.show()


# -

# ## Probability densities
#
# The downhill matrix is well suited to probability analysis. The parcel of water from a donor node is split across recipient nodes based on slope. The ratio of all recipient nodes must sum to 1, which is an identical characteristic of a probability tree. The cumulative flow routine sums the flow at each increment, the same as a cumulative probability density. A wider probability tree can be cast with more downhill neighbours:
#
# ```python
# sp.downhill_neighbours = 3
# sp.update_height(height)
# ```
#
# Here we track the cumulative probability of a particle appearing (i) upstream and (ii) downstream. This is useful to explore provenance relationships of water packets. In practise, we drop a scalar value of 1 at a selected vertex and use the `cumulative_flow` routine to propogate this across the mesh.

# +
def cumulative_probability_upstream(self, vertex):
    P = np.zeros(self.npoints)
    P[vertex] = 1.0
    Pall = self.cumulative_flow(P, uphill=True)
    return Pall

def cumulative_probability_downstream(self, vertex):
    P = np.zeros(self.npoints)
    P[vertex] = 1.0
    Pall = self.cumulative_flow(P, uphill=False)
    return Pall


# Choose a vertex to analyse
vertex = 3520

Pdownstream = cumulative_probability_downstream(sp, vertex)
Pupstream   = cumulative_probability_upstream(sp, vertex)


# +
lv = lavavu.Viewer(border=False, resolution=[666,666], background="#FFFFFF")
lv["axis"]=True
lv['specular'] = 0.5


tris  = lv.triangles("spmesh", wireframe=False,  logScale=False, colour="Red")
tris.vertices(verts)
tris.indices(sp.tri.simplices)
tris.values(Pupstream, "upstream")
tris.values(Pdownstream, "downstream")
tris.colourmap("drywet")


nodes = lv.points("vertices", pointsize=1.0)
nodes.vertices(verts)
nodes.values(sp.bmask)

vert = lv.points("vertices", pointsize=50.0, colour="Red")
vert.vertices(verts[vertex])

lv.control.Panel()
tris.control.List(options=["upstream", "downstream"], property="colourby", value="height", command="redraw", label="Display:")
lv.control.show()
# -

# ## Animation of stream flow
#
# A finite volume of water is propagated downstream where a single stream meets a saddle point and the river must diverge. Downstream flow is better represented using 2 downhill pathways in this example.

# +
rain = np.zeros_like(height)
rain[np.logical_and(x > -0.1, x < 0.1)] = 10.
rain[y > 0.] = 0.0

sp.downhill_neighbours = 2
smooth_rain = sp.local_area_smoothing(rain, its=10)

# +
# Create an animation
lv = lavavu.Viewer(border=False, resolution=[666,666], background="#FFFFFF")
lv["axis"]=True
lv['specular'] = 0.5


tris  = lv.triangles("spmesh", wireframe=False,  logScale=False, colour="Red")
tris.vertices(verts)
tris.indices(sp.tri.simplices)
tris.colourmap("drywet")


nodes = lv.points("vertices", pointsize=1.0)
nodes.vertices(verts)
nodes.values(sp.bmask)


DX0 = sp.gvec.duplicate()
DX1 = sp.gvec.duplicate()
DX0.set(0.0)
DX1.setArray(smooth_rain)

step = 0
values = []
while DX1.array.any():
#     values.append(DX1.array.copy())
    
    tris.values(DX1.array.copy(), "stream flow")
    lv.addstep()
    
    step += 1
    DX1 = sp.downhillMat*DX1
    DX0 += DX1


# +

lv.control.Panel()
lv.control.TimeStepper()
lv.control.Range("scalepoints", range=(0.1,5), step=0.1)
lv.control.ObjectList()
lv.control.show()


# +
lv.timestep(0)
total_steps = lv.timesteps()

print("number of timesteps: {}".format(len(total_steps)))
# -


