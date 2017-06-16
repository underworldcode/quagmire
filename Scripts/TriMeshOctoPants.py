
# coding: utf-8

# # TriMeshes
#
#

# In[1]:

# import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import imread
from quagmire import tools as meshtools
# get_ipython().magic('matplotlib inline')


# ## TriMesh
#
#

# In[40]:

from quagmire import FlatMesh
from quagmire import TopoMesh # all routines we need are within this class
from quagmire import SurfaceProcessMesh

minX, maxX = -5.0, 5.0
minY, maxY = -5.0, 5.0,

x1, y1, bmask1 = meshtools.poisson_elliptical_mesh(minX, maxX, minY, maxY, 0.1, 500, r_grid=None)

# In[41]:

DM = meshtools.create_DMPlex_from_points(x1, y1, bmask1, refinement_steps=2)
mesh = SurfaceProcessMesh(DM)  ## cloud array etc can surely be done better ...


# In[42]:

x = mesh.coords[:,0]
y = mesh.coords[:,1]
bmask = mesh.bmask

radius  = np.sqrt((x**2 + y**2))
theta   = np.arctan2(y,x)

height  = np.exp(-0.025*(x**2 + y**2)**2) + 0.25 * (0.2*radius)**4  * np.cos(10.0*theta)**2 ## Less so
height  += 0.5 * (1.0-0.2*radius)

rainfall = np.ones_like(height)
rainfall[np.where( radius > 1.0)] = 0.0

# In[44]:

mesh.downhill_neighbours = 2
mesh.update_height(height)

flowpaths = mesh.cumulative_flow(rainfall*mesh.area)
sqrtpaths = np.sqrt(flowpaths)

mesh.downhill_neighbours = 3
mesh.update_height(height)


flowpaths3 = mesh.cumulative_flow(rainfall*mesh.area)
sqrtpaths3 = np.sqrt(flowpaths3)

mesh.downhill_neighbours = 1
mesh.update_height(height)

flowpaths1 = mesh.cumulative_flow(rainfall*mesh.area, iterations=500)
sqrtpaths1 = np.sqrt(flowpaths1)

# In[46]:

# Choose a scale to plot all flow results
fmax = 1.0


decomp = np.ones_like(mesh.height) * mesh.dm.comm.rank


filename="Octopants.h5"

mesh.save_mesh_to_hdf5(filename)
mesh.save_field_to_hdf5(filename, height=mesh.height,
                                  slope=mesh.slope,
                                  flow=np.sqrt(flowpaths1),
                                  flow2=np.sqrt(flowpaths),
                                  decomp=decomp)

# to view in Paraview
meshtools.generate_xdmf(filename)
