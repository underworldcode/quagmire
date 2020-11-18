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

# # TriMeshes and the Octopants Landscape
#
#

# +
import matplotlib.pyplot as plt
import numpy as np

from quagmire import tools as meshtools
# %matplotlib inline
# -

# ## TriMesh
#
#

# +
from quagmire import QuagMesh 

minX, maxX = -5.0, 5.0
minY, maxY = -5.0, 5.0,

spacingX = 0.05
spacingY = 0.05

x, y, simplices = meshtools.elliptical_mesh(minX, maxX, minY, maxY, spacingX, spacingY, 1.)
DM = meshtools.create_DMPlex(x, y, simplices, refinement_levels=2)

mesh = QuagMesh(DM)

print( "\nNumber of points in the triangulation: {}".format(mesh.npoints))

# +
x = mesh.coords[:,0]
y = mesh.coords[:,1]
bmask = mesh.bmask

radius  = np.sqrt((x**2 + y**2))
theta   = np.arctan2(y,x)

height  = np.exp(-0.025*(x**2 + y**2)**2) + 0.25 * (0.2*radius)**4  * np.cos(10.0*theta)**2 ## Less so
height  += 0.5 * (1.0-0.2*radius)

rainfall = np.ones_like(height)
rainfall[np.where( radius > 5.0)] = 0.0 

with mesh.deform_topography():
    mesh.topography.data = height
# -

mo1 = mesh.identify_outflow_points()
i = np.argsort(theta[mo1])
outflows = mo1[i]

# +
mesh.downhill_neighbours = 2
flowpaths = mesh.cumulative_flow(rainfall*mesh.area)
logpaths = np.log10(1e-10 + flowpaths)
sqrtpaths = np.sqrt(flowpaths)

mesh.downhill_neighbours = 3
flowpaths3 = mesh.cumulative_flow(rainfall*mesh.area)
logpaths3 = np.log10(1e-10 + flowpaths3)
sqrtpaths3 = np.sqrt(flowpaths3)

mesh.downhill_neighbours = 1
flowpaths1 = mesh.cumulative_flow(rainfall*mesh.area)
logpaths1 = np.log10(1e-10 + flowpaths1)
sqrtpaths1 = np.sqrt(flowpaths1)
# -


# Choose a scale to plot all six flow results
fmax = 1.0

# +
from scipy import ndimage

fig = plt.figure(1, figsize=(10.0, 10.0))
ax = fig.add_subplot(111)
ax.axis('off')
sc = ax.scatter(x[bmask], y[bmask], s=1, c=mesh.topography.data[bmask], vmin=0.0, vmax=1.0)
sc = ax.scatter(x[~bmask], y[~bmask], s=5, c=mesh.topography.data[~bmask], vmin=0.0, vmax=1.0)

# fig.colorbar(sc, ax=ax, label='height')
plt.show()

# +
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))
for ax in [ax1, ax2]:
    ax.axis('equal')
    ax.axis('off')
    
    
im1 = ax1.tripcolor(x, y, mesh.tri.simplices, flowpaths1 ,     cmap='Blues')
im2 = ax2.tripcolor(x, y, mesh.tri.simplices, flowpaths,       cmap="Blues")

fig.colorbar(im1, ax=ax1)
fig.colorbar(im2, ax=ax2)
plt.show()
