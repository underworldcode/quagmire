# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 2
#     language: python
#     name: python2
# ---

# %% [markdown]
# # Visualising with LavaVu
#
# [LavaVu](https://github.com/OKaluza/LavaVu) is a lightweight, automatable visualisation and analysis viewing utility.
#

# %%
import lavavu

import numpy as np
from quagmire import QuagMesh, QuagMesh
from quagmire import tools as meshtools

import petsc4py
from petsc4py import PETSc

from mpi4py import MPI
comm = MPI.COMM_WORLD

# %%
minX, maxX = -5., 5.
minY, maxY = -5., 5.
spacing = 0.033

ptsx, ptsy, bmask = meshtools.elliptical_mesh(minX, maxX, minY, maxY, spacing, spacing, 5000, 100)


ptsx, ptsy, bmask = meshtools.square_mesh(minX, maxX, minY, maxY, 0.05, 0.05, samples=10000, boundary_samples=1000)
ptsx, ptsy = meshtools.lloyd_mesh_improvement(ptsx, ptsy, bmask, 5)


# %%
dm = meshtools.create_DMPlex_from_points(ptsx, ptsy, bmask, refinement_levels=3)
mesh = QuagMesh(dm)


# %%
x, y, simplices, bmask = mesh.get_local_mesh()

print x.shape

# x = pts[:,0]
# y = pts[:,1]

# create height field
radius  = np.sqrt((x**2 + y**2))
theta   = np.arctan2(y,x)

height  = np.exp(-0.025*(x**2 + y**2)**2) + \
          0.25 * (0.2*radius)**4  * np.cos(10.0*theta)**2
height  += 0.5 * (1.0-0.2*radius)

mesh.update_height(height)

# %%
## Create flows

flowpaths = mesh.cumulative_flow(np.ones_like(mesh.height))

# %%
mesh.npoints

# %% [markdown]
# **Create a viewer**

# %%
lv = lavavu.Viewer(border=False, background="#FFFFFF", resolution=[1000,600], near=-10.0)

# %% [markdown]
# **Plot a triangle surface**
#
# Can be vertices only (3 per tri) or vertices (shared) with indices (3 per tri)

# %%
# Prepare the triangles

tris  = lv.triangles("surface",      wireframe=False,  logScale=False)
tris2 = lv.triangles("flow_surface", wireframe=False,  logScale=False)

verts = np.reshape(mesh.tri.points, (-1,2))
verts = np.insert(verts, 2, values=mesh.height, axis=1)

tris.vertices(verts)
tris.indices(mesh.tri.simplices)

verts[:,2] += 0.01

tris2.vertices(verts)
tris2.indices(mesh.tri.simplices)

# %% [markdown]
# **Add values, can be used to colour and filter the data**

# %%
#Use topography value to colour the surface

tris.values(mesh.height, 'topography')
tris2.values((flowpaths), 'flowpaths')

tris.colourmap(["(-1.0)Blue (-0.5)Green (0.0)Yellow (1.0)Brown (5.0)White"] , logscale=False, range=[-1.0,5.0])   # Apply a built in colourmap
tris2.colourmap(["#FFFFFF:0.0 #0055FF:0.6 #000033"], logscale=True)   # Apply a built in colourmap

cb = tris.colourbar(visible=True, label="Topography Colormap") # Add a colour bar


#Filter by min height value
tris["zmin"] = -0.1

# %% [markdown]
# **Apply an initial rotation and display an interactive viewer window**
#
# Along with viewer window, controls can be added to adjust properties dynamically

# %%
    # lv.rotate('x', -60)
lv.window()

tris.control.Checkbox('wireframe',  label="Topography wireframe")
tris2.control.Checkbox('wireframe', label="Flow wireframe")
tris2.control.Range('opacity', label="Flow field opacity", command="reload")

lv.control.Checkbox(property='axis')
#lv.control.Command()
lv.control.ObjectList()
lv.control.Range('specular', range=(0,1), step=0.1, value=0)

lv.control.show() #Show the control panel, including the viewer window


# %%
