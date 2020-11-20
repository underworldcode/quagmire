# Example 6- Catchment analysis from the matrix transpose.

We start with "Swamp Mountain" from the previous notebooks. This is slightly modified so that there are no lakes / pits right at the boundary.

The catchments are identified by first finding all the outflow points of the mesh (local minima that correspond to the boundary mask) and then using the transpose of the downhill-propagation matrix $D^T$ to run information (the unique ID of each outflow points) up to the top of the catchment. 

The stopping condition is that no further change occurs. 

*Note* in the context of multiple pathways, this operation produces a fuzzy catchment. The first thing we do in this notebook is to specify `downhill_neighbours=1`


## Notebook contents

- [1-2-3 downhill neighbours](#1-2-3-downhill-neighbours)
- [Upstream propogation](#Set-neighbours-to-1-and-compute-"uphill"-connectivity)
- [Dense downhill matrices](#Dense-downhill-matrices)

import matplotlib.pyplot as plt
import numpy as np
from quagmire import tools as meshtools
%matplotlib inline

## Construct the swamp-mountain mesh

This time we take care to avoid lakes etc on the boundaries as this makes the catchment analysis more complicated. Visualise the mesh to make sure that this works.

from quagmire import QuagMesh 
from quagmire import QuagMesh # all routines we need are within this class
from quagmire import QuagMesh

minX, maxX = -5.0, 5.0
minY, maxY = -5.0, 5.0,

spacingX = 0.05
spacingY = 0.05

x, y, simplices = meshtools.elliptical_mesh(minX, maxX, minY, maxY, spacingX, spacingY, 1.)

DM = meshtools.create_DMPlex(x, y, simplices)
DM = meshtools.refine_DM(DM, refinement_levels=2)

mesh = QuagMesh(DM, downhill_neighbours=1)

# Note ... this is what refinement does 
x = mesh.coords[:,0]
y = mesh.coords[:,1]

print( "\nNumber of points in the triangulation: {}".format(mesh.npoints))

radius  = np.sqrt((x**2 + y**2))
theta   = np.arctan2(y,x)+0.1

height  = np.exp(-0.025*(x**2 + y**2)**2) + 0.25 * (0.2*radius)**4  * np.cos(5.0*theta)**2 ## Less so
height  += 0.5 * (1.0-0.2*radius)
height  -= height.min()

## Add smoothed random noise to make some "lakes" 

mesh._construct_rbf_weights(delta=mesh.delta*3.0)

randpts1 = np.where(np.random.random(height.shape)>0.995, -1.0, 0.0)
hrand1   = 20.0 * mesh.rbf_smoother(randpts1, iterations=10)
heightn = height + hrand1 * np.exp(-radius**2/15.0) 


with mesh.deform_topography():
    mesh.downhill_neighbours = 2
    mesh.topography.data = heightn


# let's use a rainfall proportional to height (any choice is ok)

rainfall_fn  = mesh.topography ** 2.0
flowrate_fn  = mesh.upstream_integral_fn(rainfall_fn)
str_power_fn = mesh.upstream_integral_fn(rainfall_fn)**2.0 * mesh.slope ** 2.0

## Process the surface to fill swamps and lakes

## Quagmire also has a swamp filling algorithm

mesh1s = QuagMesh(DM)
with mesh1s.deform_topography():
    mesh1s.topography.data = mesh.topography.data
    
mesh1s.low_points_local_patch_fill(its=5, smoothing_steps=1)

for i in range(0,50):
    mesh1s.low_points_swamp_fill(ref_height=-0.01)
    
    # In parallel, we can't break if ANY processor has work to do (barrier / sync issue)
    low_points2 = mesh1s.identify_global_low_points()
    
    print("{} : {}".format(i,low_points2[0]))
    if low_points2[0] == 0:
        break

rainfall_fn_1s  = mesh1s.topography ** 2.0
flowrate_fn_1s  = mesh1s.upstream_integral_fn(rainfall_fn_1s)
str_power_fn_1s = mesh1s.upstream_integral_fn(rainfall_fn_1s)**2.0 * mesh.slope ** 2.0

## Locating and viewing the outflow points

`quagmire` provides a mechanism to find the outflow points of a domain and return the node values. *Note:* in parallel these are the local node numbers and are not a unique ID. To do this, we can obtain the global ID from PETSc but it does also help to have the indices all be small numbers so we can map colours effectively.

outflows = mesh1s.identify_outflow_points()
print("Mesh has {} outflows".format(outflows.shape[0]))

import quagmire
print(quagmire.mesh.check_object_is_a_q_mesh_and_raise(mesh1s))

import lavavu

outpoints = np.column_stack([mesh.tri.points[outflows], heightn[outflows]])
points = np.column_stack([mesh.tri.points, heightn])

lv = lavavu.Viewer(border=False, background="#FFFFFF", resolution=[600,600], near=-10.0)

nodes = lv.points("nodes", pointsize=10.0, pointtype="shiny", colour="#FF0000" )
nodes.vertices(outpoints)

tri1 = lv.triangles("triangles", wireframe=False)
tri1.vertices(points)
tri1.indices(mesh.tri.simplices)

tri1.values(np.log(flowrate_fn_1s.evaluate(mesh1s)), "cum-rain-swamp")
tri1.values(np.log(flowrate_fn.evaluate(mesh)),     "cumulative rain")

tri1.colourmap("#BBEEBB #889988 #000099")
tri1.colourbar()

## Swamped

points = np.column_stack([mesh1s.tri.points, mesh1s.topography.data-0.01])

tri2 = lv.triangles("triangles2", wireframe=False)
tri2.vertices(points)
tri2.indices(mesh1s.tri.simplices)

tri2.values(mesh1s.topography.data-mesh.topography.data,   "swamps")
tri2.values(np.ones_like(mesh1s.topography.data), "blank")
tri2.values(np.log(flowrate_fn_1s.evaluate(mesh1s)), "cum-rain-swamp")

tri2.colourmap("#003366:0.5, #000099:0.8, #000099")
tri2.colourbar()


lv.control.Panel()
lv.control.ObjectList()
tri1.control.List(options=["cum-rain-swamp",
                   "cumulative rain", 
                   ], property="colourby", command="redraw")

tri2.control.List(options=["blank", "swamps", 
                   "cum-rain-swamp"], property="colourby", command="redraw")


lv.control.show()

## Stream power / slope where the lakes / swamps are located:

## Set neighbours to 1 and compute "uphill" connectivity

In serial, i.e. for this demonstration, we number the outflow points by their order in the local mesh numbering. We can then use the `mesh.uphill_propagation` routine to propagate this information from the outflow to the top of the catchment. 

This routine returns the mesh data (for this processor) of a globally synchronised map of the information propagated from the selected points. 

The routine is used in the computation of flood fills for the swamp algorithm and should be polished up a bit (sorry).

## Unique catchments requires the downhill matrix with downhill_neighbours=1

mesh1s.near_neighbours=1
# mesh1s.update_height(mesh1s.heightVariable.data)

## Need a unique ID that works in parallel ... global node number would work but 
## not that easy to map to colours in lavavu 

from petsc4py import PETSc
outflows
outflowID = mesh1s.lgmap_row.apply(outflows.astype(PETSc.IntType))

# But on 1 proc, this is easier / better:

outflowID = np.array(range(0, outflows.shape[0]))
ctmt = mesh1s.uphill_propagation(outflows,  outflowID, its=99999, fill=-999999).astype(np.int)

## Visualise the catchment information

import lavavu

outpoints = np.column_stack([mesh.tri.points[outflows], heightn[outflows]])
points = np.column_stack([mesh.tri.points, mesh1s.topography.data])

lv = lavavu.Viewer(border=False, background="#FFFFFF", resolution=[900,600], near=-10.0)

nodes = lv.points("nodes", pointsize=10.0, pointtype="shiny", colour="#FF0000" )
nodes.vertices(outpoints)

tri1 = lv.triangles("triangles", wireframe=False, opacity=1.0)
tri1.vertices(points)
tri1.indices(mesh.tri.simplices)
tri1.values(np.log(flowrate_fn_1s.evaluate(mesh1s)),"cum-rain-swamp")

tri1.colourmap("#BBEEBB:0.0, #889988:0.2, #889988:0.2, #0000FF:0.2, #0000FF")
tri1.colourbar(visible=False)

## Swamped

tri2 = lv.triangles("triangles2", wireframe=False)
tri2.vertices(points-(0.0,0.0,0.01))
tri2.indices(mesh1s.tri.simplices)

tri2.values(ctmt,   "catchments")

tri2.colourmap("spectral", discrete=True, range=[0, outflows.shape[0]-1])
tri2.colourbar(visible=False)

tri3 = lv.triangles("triangles3", wireframe=False)
tri3.vertices(points-(0.0,0.0,0.005))

tri3.indices(mesh1s.tri.simplices)
tri3.values(mesh1s.topography.data-mesh.topography.data,   "swamps")
tri3.colourmap("#003366:0.0, #000099,  #000099, #000099, #0000FF")
tri3.colourbar(visible=False)


lv.control.Panel()
lv.control.ObjectList()

lv.control.show()

