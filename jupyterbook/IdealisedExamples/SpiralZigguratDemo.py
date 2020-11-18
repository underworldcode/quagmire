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

import numpy as np
# import matplotlib.pyplot as plt
from quagmire import QuagMesh 
from quagmire import tools as meshtools
from mpi4py import MPI
comm = MPI.COMM_WORLD

# +
minX, maxX = -5., 5.
minY, maxY = -5., 5.
spacing = 0.01

ptsx, ptsy, simplices = meshtools.elliptical_mesh(minX, maxX, minY, maxY, spacing, spacing)
dm = meshtools.create_DMPlex_from_points(ptsx, ptsy, bmask=None, refinement_levels=0)

# +
mesh = QuagMesh(dm, downhill_neighbours=3)

#if comm.rank == 0:
print("Number of nodes in mesh - {}: {}".format(comm.rank, mesh.npoints))

# retrieve local mesh
x = mesh.tri.x
y = mesh.tri.y

# dm generated bmask

bmask = mesh.bmask

# +
import stripy

# create height field - make 2 spirals as strings of points and interpolate between them
# to make a smooth surface for the model. 

#  
theta = np.linspace(0.0000001, 66.6*np.pi, 10000)
s1 = 0.30 * theta 
s2 = 0.25 * theta 
x1 = s1 * np.cos(theta)
y1 = s1 * np.sin(theta)
x2 = s2 * np.cos(theta)
y2 = s2 * np.sin(theta)

rmean = (s1 + s2) / 2.0
z = np.exp(-rmean**2.0 / 20.0)

h2 = (1.0 - s1 / s1.max()) * z
h1 = (1.0 - s2 / s1.max()) * z + 0.05 

x0 = np.hstack( [x1,x2] )
y0 = np.hstack( [y1,y2] )
h0 = np.hstack( [h1,h2] )
shade = np.hstack( [np.zeros_like(h1), np.ones_like(h2)])

points = np.transpose(np.array( [x0,y0] ))
newpoints = np.transpose(np.array([x,y]))

interp = stripy.Triangulation(points[:,0], points[:,1])

height, ierr = interp.interpolate_linear(newpoints[:,0], newpoints[:,1], h0)
shade, ierr  = interp.interpolate_linear(newpoints[:,0], newpoints[:,1], shade)

height = height + (1.0 + 0.01 * np.random.random(size=height.shape))

# +
rank = np.ones_like(height)*comm.rank
shadow = np.zeros_like(height)

# get shadow zones
shadow_zones = mesh.lgmap_row.indices < 0
shadow[shadow_zones] = 1
shadow_vec = mesh.gvec.duplicate()

mesh.lvec.setArray(shadow)
mesh.dm.localToGlobal(mesh.lvec, shadow_vec, addv=True)


# +
import lavavu
import stripy

vertices = np.column_stack([x, y, 3 * height])
tri = mesh.tri

lv = lavavu.Viewer(border=False, background="#FFFFFF", resolution=[600,600], near=-10.0)

# sa = lv.points("subaerial", colour="red", pointsize=0.2, opacity=0.75)
# sa.vertices(vertices[subaerial])

tris = lv.triangles("mesh",  wireframe=False, colour="#77ff88", opacity=1.0)
tris.vertices(vertices)
tris.indices(tri.simplices)
tris.values(height, label="elevation")
#tris.values(shade, label="shade")
tris.colourmap('dem3')
cb = tris.colourbar()

# sm = lv.points("submarine", colour="blue", pointsize=0.5, opacity=0.75)
# sm.vertices(vertices[submarine])

lv.control.Panel()
lv.control.ObjectList()
# tris.control.Checkbox(property="wireframe")
lv.control.show()


# +
with mesh.deform_topography():
    mesh.topography.data = height
    
gradient = mesh.slope.evaluate(mesh)

# +
low_points1 = mesh.identify_global_low_points(ref_height=-0.001)
print("0 : {}".format(low_points1[0]))

for repeat in range(0,3):    
    mesh.low_points_local_patch_fill(its=5, smoothing_steps=1)
    low_points2 = mesh.identify_global_low_points(ref_height=-0.001)
    print("{} : {}".format(repeat,low_points2[0]), flush=True)

    if low_points2[0] == 0:
        break
    
    for i in range(0,10):
 
        mesh.low_points_swamp_fill(its=5000, ref_height=-0.001, ref_gradient=0.000001)

        # In parallel, we can't break if ANY processor has work to do (barrier / sync issue)
        low_points3 = mesh.identify_global_low_points(ref_height=0.0)

        print("{} : {}".format(i,low_points3[0]))
        if low_points3[0] == 0:
            break
# -


outflow_points = mesh.identify_outflow_points()
low_points     = mesh.identify_low_points()

# +
from quagmire import function as fn

ones = fn.parameter(1.0, mesh=mesh)
rain = fn.misc.levelset(mesh.topography, alpha=0.99)

cumulative_flow_0 = np.log10(1.0e-10 + mesh.upstream_integral_fn(ones).evaluate(mesh))


# +
## Smoothing is purely for the purpose of visualisation

rbf_smoother = mesh.build_rbf_smoother(0.015)
smoothed_flow = rbf_smoother.smooth_fn(mesh.upstream_integral_fn(ones))
cumulative_flow_1 = np.log10(1.0e-10 + smoothed_flow.evaluate(mesh))
# -

cumulative_flow_0.min(), cumulative_flow_1.min()
cumulative_flow_0.max(), cumulative_flow_1.max()

# +
import lavavu
import stripy

vertices = np.column_stack([x, y, 3 * height])
tri = mesh.tri

lv = lavavu.Viewer(border=False, background="#FFFFFF", resolution=[1200,600], near=-10.0)

outs = lv.points("outflows", colour="green", pointsize=5.0, opacity=1.0)
outs.vertices(vertices[outflow_points])

lows = lv.points("lows", colour="red", pointsize=5.0, opacity=0.75)
lows.vertices(vertices[low_points])

flowball = lv.points("flowballs", pointsize=4.0)
flowball.vertices(vertices+(0.0,0.0,0.001))
flowball.values(cumulative_flow_1, label="flow1")
flowball.colourmap("rgba(255,255,255,0.0) rgba(128,128,255,0.5) rgba(0,50,200,1.0)")

heightball = lv.points("heightballs", pointsize=3.0, opacity=0.9)
heightball.vertices(vertices)
heightball.values(height, label="height")
heightball.colourmap('dem3')

lv.translation(-1.012, 2.245, -13.352)
lv.rotation(53.217, 18.104, 161.927)

lv.control.Panel()
lv.control.ObjectList()
lv.control.show()


# -

lv.image(filename="SpiralZiggurat.png", resolution=(1500,750), quality=4)




