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

# %%
from quagmire import QuagMesh
from quagmire import tools as meshtools
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

import petsc4py


# %%
import h5py

meshFile = h5py.File(name="../Scripts/Octopants.h5", mode="r")
points = meshFile["geometry"]["vertices"]
x1 = points.value[:,0]
y1 = points.value[:,1]
bmask = meshFile["fields"]["bmask"].value[:].astype(bool)
height = meshFile["fields"]["height"].value[:]
lakes = meshFile["fields"]["swamps"].value[:]

# %%
dm  = meshtools.create_DMPlex_from_points(x1,y1,bmask=bmask)
SPM = QuagMesh(dm)
x = SPM.coords[:,0]
y = SPM.coords[:,1]


# %%
from scipy.spatial import ckdtree
old_nodes = ckdtree.cKDTree( points.value )

# %%
distance, mapping = old_nodes.query(SPM.coords)
SPM.update_height(height[mapping])
lakefill = lakes[mapping]

# %%
its, flowpaths2 = SPM.cumulative_flow_verbose(SPM.area, maximum_its=2000, verbose=True)

# %%
low_points = SPM.identify_low_points()
print low_points

# %%
# Plot the stream power, erosion and deposition rates
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(50,15))
for ax in [ax1, ax2, ax3]:
    ax.axis('equal')
    ax.axis('off')

    
im1 = ax1.tripcolor(x, y, SPM.tri.simplices, SPM.height, cmap=plt.cm.RdBu)    
# ax1.tripcolor(x, y, SPM.tri.simplices, height, 10)

# im1 = ax1.tripcolor(x, y, sp.tri.simplices, sp.height, cmap=plt.cm.terrain)
im2 = ax2.tripcolor(x, y, SPM.tri.simplices, lakefill, cmap='Greens', vmax=None)

# ax3.tripcolor(x, y, SPM.tri.simplices, height, cmap=plt.cm.gray, zorder=1, vmin=-0.75, alpha=0.5)
im3 = ax3.tripcolor(x, y, SPM.tri.simplices, np.log(flowpaths2), cmap='Blues', zorder=0)
ax3.scatter(x[low_points], y[low_points], color="Red", s=25.0)

fig.colorbar(im1, ax=ax1)
fig.colorbar(im2, ax=ax2)
fig.colorbar(im3, ax=ax3)
plt.show()



# %%
from petsc4py import PETSc



# %%
vec = SPM.gvec.duplicate()
vec.setName("noname")

filename = "test0.h5"

SPM.save_mesh_to_hdf5(filename)

viewh5 = PETSc.Viewer()
viewh5.createHDF5(filename, mode='a', comm=dm.comm)
viewh5.view(obj=vec)
viewh5.destroy()



# %%
testFile = h5py.File(name=filename, mode="r")



# %%
testFile.keys()

# %%
# rm test0.h5


# %%
