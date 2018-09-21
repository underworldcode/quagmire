
# coding: utf-8

# # PixMeshes
# 
# A pixmesh is an optimised regular mesh for handling high resolution DEM data of a small region. The issues of mesh regularity on the landscape analysis need to be considered (multiple descent pathways and smoothing, for example). 

# In[1]:

# import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import imread
from quagmire import tools as meshtools
# get_ipython().magic('matplotlib inline')


# ### Landscape
# 
# 

# In[2]:

dem = imread('data/port_macquarie.tif', mode='F')
#dem = np.fliplr(dem)

rows, columns = dem.shape
aspect_ratio = float(columns) / float(rows)

spacing = 5.0

#minX, maxX = 0.0, spacing*dem.shape[1]
#minY, maxY = 0.0, spacing*dem.shape[0]


#fig = plt.figure(1, figsize=(10*aspect_ratio,10))
#ax = fig.add_subplot(111)
#ax.axis('off')
#im = ax.imshow(dem, cmap='terrain_r', origin='upper')
#fig.colorbar(im, ax=ax, label='height')


# In[3]:

# dem.shape, dem.size


# In[ ]:




# ## PixMesh
# 
# We saw how to triangulate, but this is how to use the regular equivalent. 18000000 points is still high, so let's work at 1/4 that size. This is still over four million points, so this gives us a sense of the trade off between regular meshes and triangulations.
# 
# The creation of the distributed object, like the triangulation, reorders x,y points.

# In[4]:

from quagmire import FlatMesh 
from quagmire import TopoMesh # all routines we need are within this class
from quagmire import SurfaceProcessMesh

DM = meshtools.create_DMDA(0.0, 5124.0, 0.0, 3612.0, 5124/2, 3612/2 )


# In[9]:

mesh = SurfaceProcessMesh(DM, verbose=True, neighbour_cloud_size=25)  ## cloud array etc can surely be done better ... 
mesh.downhill_neighbours = 3


# In[ ]:




# In[10]:

# Creating the DM reorders points

print mesh.neighbour_array.shape
print mesh.coords.shape
print dem[::2,::2].shape


# In[11]:

mesh.update_height(dem[::2,::2].reshape(-1))


# In[22]:

mesh_heights_no_lows = mesh.handle_low_points(its=50)


# In[23]:

flowpaths = mesh.cumulative_flow(np.ones_like(mesh.height))
flowpathsS = mesh.rbf_smoother(flowpaths)
low_points = mesh.identify_low_points()

print low_points.shape[0]

# In[24]:

flats = np.where(mesh.identify_flat_spots())[0]
print flats.shape[0]


# In[36]:

manifold = np.reshape(mesh.coords, (-1,2))
manifold = np.insert(manifold, 2, values=mesh.height*0.2, axis=1)


# In[39]:

from LavaVu import lavavu

lv = lavavu.Viewer(border=False, background="#FFFFFF", resolution=[1000,600], near=-10.0)

topo = lv.points(pointsize=1.0, pointtype='flat')
topo.vertices(manifold)
topo.values(mesh.height, label='height')
# topo.values(np.sqrt(flowpaths), label='flow')


topo2 = lv.points(pointsize=1.0, pointtype='flat')
topo2.vertices(manifold+(0.0,0.0,1.0))
# topo.values(mesh.height, label='height')
topo2.values(np.sqrt(flowpathsS), label='flow')


topo.colourmap(["#004420", "#FFFFFF", "#444444"] , logscale=False, range=[-200.0, 400.0])   # Apply a built in colourmap
# topo.colourmap(["#FFFFFF:0.0", "#0033FF:0.3", "#000033"], logscale=False)   # Apply a built in colourmap
topo2.colourmap(["#FFFFFF:0.0", "#0033FF:0.1", "#000033"], logscale=True)   # Apply a built in colourmap




# In[40]:

## Viewer

lv.window()

# tris.control.Range(property='zmin', range=(-1,1), step=0.001)
# lv.control.Range(command='background', range=(0,1), step=0.1, value=1)
# lv.control.Range(property='near', range=[-10,10], step=2.0)
lv.control.Checkbox(property='axis')
lv.control.Command()
lv.control.ObjectList()
lv.control.show()


# Landscape analysis statistics

# In[ ]:

gradient_max = mesh.slope.max()
gradient_mean = mesh.slope.mean()
flat_spots = np.where(mesh.slope < gradient_mean*0.001)[0]
low_points = mesh.identify_low_points()

nodes = np.arange(0, mesh.npoints)
lows =  np.where(np.logical_and(mesh.down_neighbour[1] == nodes, mesh.height > 0.5))[0]

# print statistics
print("mean gradient {}\nnumber of flat spots {}\nnumber of low points {}".format(gradient_mean,
                                                                                  flat_spots.size,
                                                                                  low_points.shape[0]))


# In[ ]:

filename = 'port_macquarie_pixmesh.h5'

mesh.save_mesh_to_hdf5(filename)
mesh.save_field_to_hdf5(filename, height=mesh.height, slope=mesh.slope, flow=np.sqrt(flowpaths))

# to view in Paraview
# meshtools.generate_xdmf(filename)


# In[ ]:




# In[ ]:



