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

# ## Swamp filling of local minima
#
# This notebook illustrates the strategy that the swamp-filling algorithm takes and shows why multiple iterations are needed.
#
# We start with a simple, semi-circular channel (gutter) and introduce nested hemispherical depressions (local minima).
#
#
#
# <!-- #### Contents
#
# - [Rosenbrock function](#Rosenbrock-function)
# - [Compare one and two downhill pathways](#Compare-one-and-two-downhill-pathways)
# - [Probability densities](#Probability-densities)
# - [Animation of stream flow](#Animation-of-stream-flow) -->

# +
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from quagmire import QuagMesh
from quagmire import tools as meshtools
from quagmire import function as fn
import scipy

# %matplotlib inline

# +
minX, maxX = -1.0, 1.0
minY, maxY =  0.0, 3.0

x, y, bmask = meshtools.generate_square_points(minX, maxX, minY, maxY, 0.01, 0.01, samples=50000, boundary_samples=500)
x, y = meshtools.lloyd_mesh_improvement(x, y, bmask, 5)

DM = meshtools.create_DMPlex_from_points(x, y, bmask)
sp = QuagMesh(DM)
catchments0 = sp.add_variable(name="low_point_catchments0")

x = sp.tri.points[:,0]
y = sp.tri.points[:,1]
# -


# ## Semi-circular gutter with incline
#
# $$
# h(x,y) = (1-x)^2 
# $$
#
# we introduce a small incline to ensure the streams terminate at the boundary.

# +
height = 1.0 - np.sin(np.arccos((0.5*x)/abs(x).max()))  

with sp.deform_topography():
    sp.topography.data = height

gradient = sp.slope.evaluate(sp)

# +
## Add hemispherical blisters

blisters = [(0.0,2.4,0.3,0.5),(0.0,1.9, 0.4, 0.5), (0.0, 1.5, 0.3, 0.5), (0.2, 1.0, 0.2, 0.75), 
            (0.5, 0.95, 0.1, 0.5), (0.65, 0.9, 0.1, 1.0), (0.8, 0.8, 0.1, 1.5),]


height = sp.topography.data.copy()
dh = np.zeros_like(height)

for blis in blisters:
    x0  = blis[0]
    y0  = blis[1]
    r0  = blis[2]
    s0  = blis[3]
    
    points = sp.cKDTree.query_ball_point((x0,y0), r0)   
    r  = np.hypot(sp.data[points][:,0] - x0, sp.data[points][:,1] - y0)
    dh[points] = np.maximum(dh[points], s0 * np.sqrt(r0**2 - r**2))
    

height -= dh
# -

height += 0.05 * y # make a small incline
height -= height.min()

ones = fn.parameter(1.0, mesh=sp)

# +
with sp.deform_topography():
    sp.topography.data = height
    
# Compute a flow ... 

cumulative_flow_0 = np.log10(1.0e-6 + sp.upstream_integral_fn(ones).evaluate(sp))
lowpts0 = sp.identify_low_points(ref_height=-0.01)
height0 = sp.topography.data.copy()

print("Low points - {}".format(lowpts0))
# -

catchments0.data = sp.uphill_propagation(points = lowpts0, values=np.indices((lowpts0.shape)), fill=-1.0, its=1000)


outflows = sp.identify_outflow_points()

# +
sp.low_points_swamp_fill(its=1000, ref_height=-0.01, saddles=False)
cumulative_flow_1 = np.log10(1.0e-6 + sp.upstream_integral_fn(ones).evaluate(sp))
lowpts1 = sp.identify_low_points(ref_height=-0.01)
height1 = sp.topography.data.copy()

print("{}: Low points - {}".format(lowpts1.shape[0],lowpts1))
# +
sp.low_points_swamp_fill(its=1000, ref_height=0.0, saddles=False)

cumulative_flow_2 = np.log10(1.0e-6 + sp.upstream_integral_fn(ones).evaluate(sp))
lowpts2 = sp.identify_low_points()
height2 = sp.topography.data.copy()
print("{}: Low points - {}".format(lowpts2.shape[0],lowpts2))

# +
import lavavu

lv = lavavu.Viewer(border=False, resolution=[1000,600], background="#FFFFFF")
lv["axis"]=False
lv['specular'] = 0.0

verts = np.reshape(sp.tri.points, (-1,2))
verts = np.insert(verts, 2, values=height0 , axis=1)

verts_s1 = np.reshape(sp.tri.points, (-1,2))
verts_s1 = np.insert(verts_s1, 2, values=np.where(height1>height0, height1, height0-0.01), axis=1)

verts_s2 = np.reshape(sp.tri.points, (-1,2))
verts_s2 = np.insert(verts_s2, 2, values=np.where(height2>height0, height2, height0-0.01), axis=1)

tris  = lv.triangles("spmesh", wireframe=False,  logScale=False, opacity=0.9)
tris.vertices(verts)
tris.indices(sp.tri.simplices)
tris.values(cumulative_flow_0, label="flow0")
tris.values(cumulative_flow_1, label="flow1")
tris.values(cumulative_flow_2, label="flow2")
tris.values(ones.evaluate(sp), label="blank")
tris.colourmap("Grey #335599", range=[-2, -1])

trisW  = lv.triangles("spwire", wireframe=True,  colour="#335599")
trisW.vertices(verts+(0.0,0.0,0.001))
trisW.indices(sp.tri.simplices)

trisC = lv.triangles("spcatch", wireframe=False,  colour="#335599", opacity=0.95)
trisC.vertices(verts)
trisC.indices(sp.tri.simplices)
trisC.values(catchments0.evaluate(sp), label="catchment")
trisC.colourmap("#BBBBBB #AA5522 #8899FF #00FF66 #FFFF55 #999999")

tris_s1  = lv.triangles("spmesh_s1", wireframe=False,  logScale=False, colour="#88BBAA", opacity=0.95)
tris_s1.vertices(verts_s1)
tris_s1.indices(sp.tri.simplices)
# tris_s1.values(cumulative_flow_1, label="flow")
# tris_s1.values(ones.evaluate(sp), label="blank")
# tris_s1.colourmap("Grey Blue", range=[-2, -1])

tris_s2  = lv.triangles("spmesh_s2", wireframe=False,  logScale=False, colour="#88BBAA", opacity=0.95)
tris_s2.vertices(verts_s2)
tris_s2.indices(sp.tri.simplices)
# tris_s2.values(ones.evaluate(sp), label="blank")
# tris_s2.colourmap("Grey Blue", range=[-2, 0])


nodes = lv.points("vertices", pointsize=2.0, colour="#666677", opacity=0.5)
nodes.vertices(verts+(0.0,0.0,0.001))
nodes.values(sp.bmask)

lownodes = lv.points("lows1", pointsize=15.0, colour="Green")
lownodes.vertices(verts_s1[lowpts1]+(0.0,0.0,0.01))

lownodes0 = lv.points("lows", pointsize=15.0, colour="Red")
lownodes0.vertices(verts[lowpts0]+(0.0,0.0,0.01))

outflownodes0 = lv.points("outflows", pointsize=15.0, colour="Blue")
outflownodes0.vertices(verts[outflows]+(0.0,0.0,0.01))

lv.translation(0.348, 0.304, -3.403)
lv.rotation(-46.218, 0.0, 0.0)

lv.control.Panel()
lv.control.ObjectList()

tris.control.List(options=["blank", "flow0", "flow1", "flow2", "catch"],    value="blank", property="colourby", command="redraw", label="0")

lv.control.show()
# -


#
#

# +
## Figure: Mesh plus low points 

lv.translation(0.348, 0.304, -3.403)
lv.rotation(-46.218, 0.0, 0.0)

tris["visible"] = False
trisC["visible"] = False
tris_s1["visible"] = False
tris_s2["visible"] = False
lownodes["visible"] = False
lownodes0["visible"] = True

lv.redraw()
lv.image('NodesAndLows0.png', resolution=(3000,1500))

# +
## Figure: Initial flow and lows

lv.translation(0.348, 0.304, -3.403)
lv.rotation(-46.218, 0.0, 0.0)

tris["visible"] = True
trisW["visible"] = False
trisC["visible"] = False
tris_s1["visible"] = False
tris_s2["visible"] = False
lownodes["visible"] = False
lownodes0["visible"] = True
tris["colourby"] = "flow0"


lv.redraw()
lv.image('FlowsAndLows0.png', resolution=(3000,1500))

# +
## Figure: One Iteration flow and lows

lv.translation(0.348, 0.304, -3.403)
lv.rotation(-46.218, 0.0, 0.0)

tris["visible"] = True
trisC["visible"] = False
trisW["visible"] = False
tris_s1["visible"] = True
tris_s2["visible"] = False
lownodes["visible"] = True
lownodes0["visible"] = False
tris["colourby"] = "flow1"

lv.redraw()
lv.image('FlowsAndLows1.png', resolution=(3000,1500))

# +
## Figure: Two iterations flow and lows

lv.translation(0.348, 0.304, -3.403)
lv.rotation(-46.218, 0.0, 0.0)

tris["visible"] = True
tris_s1["visible"] = True
tris_s2["visible"] = True
lownodes["visible"] = False
lownodes0["visible"] = False
tris["colourby"] = "flow2"

lv.redraw()
lv.image('FlowsAndLows2.png', resolution=(3000,1500))
# -

# ! open .

lv.camera()

# +
## Close up view of the catchment boundary - perhaps the catchment information for the low nodes here too ... 

lv.translation(0.476, 0.414, -1.211)
lv.rotation(-2.026, -30.0, -90.587)
tris["visible"] = False
trisC["visible"] = True
trisW["visible"] = True
tris_s1["visible"] = False
tris_s2["visible"] = False
nodes["visible"] = True
lownodes["visible"] = False
lownodes0["visible"] = True
lv.redraw()
lv.image("CatchmentCloseUp.png", resolution=(3000,1500))
# -




