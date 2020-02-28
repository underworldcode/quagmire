
# coding: utf-8

# # Manipulation of a landscape with a triangular mesh
#
# Quality meshes are important for producing reliable solution in surface process modelling. For any given node in an unstructured mesh, its neighbours should be spaced more or less at an equal radius. For this we turn to Poisson disc sampling using an efficient $O(N)$ [algorithm](http://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf).
#
# The premise of this algorithm is to ensure that points are tightly packed together,
# but no closer than a specified minimum distance. This distance can be uniform across
# the entire domain, or alternatively a 2D numpy array of radius lengths can be used
# to bunch and relax the spacing of nodes.


# In[1]:

# import matplotlib.pyplot as plt
import numpy as np
# from matplotlib.pyplot import imread
import imageio
from quagmire import tools as meshtools

# get_ipython().magic('matplotlib inline')


# ### Landscape
#
# In this example we create higher resolution according to the laplacian of the topography which is related to the local-slope-diffusion term.

# In[2]:

!pwd

dem = imageio.imread('../../data/port_macquarie.tif')
dem = np.fliplr(dem)

dem.shape
rows, columns = dem.shape
aspect_ratio = float(columns) / float(rows)

spacing = 5.0

minX, maxX = 0.0, spacing*dem.shape[1]
minY, maxY = 0.0, spacing*dem.shape[0]


#fig = plt.figure(1, figsize=(10*aspect_ratio,10))
#ax = fig.add_subplot(111)
#ax.axis('off')
#im = ax.imshow(dem, cmap='terrain_r', origin='lower')
#fig.colorbar(im, ax=ax, label='height')


# In[3]:

gradX, gradY = np.gradient(dem, 5., 5.) # 5m resolution in each direction
gradXX, gradXY = np.gradient(gradX, 5., 5.)
gradYX, gradYY = np.gradient(gradY, 5., 5.)

laplacian = np.abs(gradYY + gradXX)
slope = np.hypot(gradX, gradY)

# In[4]:

height, width = slope.shape

radius_min = 50.0
radius_max = 150.0

radius = 1.0/(laplacian + 0.02)
radius = (radius - radius.min()) / (radius.max() - radius.min())
radius = radius * (radius_max-radius_min) + radius_min

# apply gaussian filter for better results
from scipy.ndimage import gaussian_filter
radius2 = gaussian_filter(radius, 5.)


#fig = plt.figure(1, figsize=(10*aspect_ratio, 10))
#ax = fig.add_subplot(111)
#ax.axis('off')
#im = ax.imshow((radius2), cmap='jet', origin='lower', aspect=aspect_ratio)
#fig.colorbar(im, ax=ax, label='radius2')

#plt.show()


# In[5]:

x1, y1, bmask1 = meshtools.poisson_square_mesh(minX, maxX, minY, maxY, spacing*20,
                                            boundary_samples=100)
print(("{} samples".format(x1.size)))

# randomize

x = np.zeros_like(x1)
y = np.zeros_like(y1)
bmask = bmask1.copy()

index = np.random.permutation(x.shape[0])

x[:] = x1[index]
y[:] = y1[index]
bmask[:] = bmask1[index]

# x += np.random.random(x.shape) * 0.001
# y += np.random.random(y.shape) * 0.001


# In[6]:

from scipy import ndimage

coords = np.stack((y, x)).T / spacing
meshheights = ndimage.map_coordinates(dem, coords.T, order=3, mode='nearest')

#fig = plt.figure(1, figsize=(10*aspect_ratio, 10))
#ax = fig.add_subplot(111)
#ax.axis('off')
#sc = ax.scatter(x[bmask], y[bmask], s=1, c=meshheights[bmask])
#sc = ax.scatter(x[~bmask], y[~bmask], s=5, c=meshheights[~bmask])

#fig.colorbar(sc, ax=ax, label='height')
#plt.show()


# ## TriMesh
#
# Now the points can be triangulated to become a quality unstructured mesh.
# Triangulation reorders x,y points - be careful!

# In[55]:

from quagmire import FlatMesh
from quagmire import TopoMesh # all routines we need are within this class
from quagmire import SurfaceProcessMesh

dm = meshtools.create_DMPlex_from_points(x, y, bmask, refinement_levels=2)

print("gLPoints: ", dm.getCoordinates().array.shape[0]/2)
print(" LPoints: ", dm.getCoordinatesLocal().array.shape[0]/2)


# In[68]:

mesh = SurfaceProcessMesh(dm, verbose=True, neighbour_cloud_size=33)
coords = np.stack((mesh.tri.points[:,1], mesh.tri.points[:,0])).T / spacing
meshheights = ndimage.map_coordinates(dem, coords.T, order=3, mode='nearest')


# In[70]:

mesh.update_height(meshheights)
meshheights = mesh.handle_low_points(its=500, smoothing_steps=3)
low_points = mesh.identify_low_points()
flats = np.where(mesh.identify_flat_spots())[0]


# In[71]:

flowpaths = mesh.cumulative_flow(np.ones_like(mesh.height))
flow2 = mesh.rbf_smoother(flowpaths)


# In[75]:

manifold = np.reshape(mesh.coords, (-1,2))
manifold = np.insert(manifold, 2, values=mesh.height*3.0, axis=1)
low_cloud = manifold[low_points]


# In[79]:

from LavaVu import lavavu

lv = lavavu.Viewer(border=False, background="#FFFFFF", resolution=[1000,600], near=-10.0)

topo  = lv.triangles("topography",  wireframe=False)
topo.vertices(manifold)
topo.indices(mesh.tri.simplices)

flowverlay = lv.triangles("flow_surface", wireframe=False)
flowverlay.vertices(manifold + (0.0,0.0, 5.0))
flowverlay.indices(mesh.tri.simplices)

flowverlay2 = lv.triangles("flow_surface_smooth", wireframe=False)
flowverlay2.vertices(manifold + (0.0,0.0, 5.0))
flowverlay2.indices(mesh.tri.simplices)



# Add properties to manifolds

topo.values(mesh.height, 'topography')
flowverlay.values(np.sqrt(flowpaths), 'flowpaths')
flowverlay2.values(np.sqrt(flow2), 'flowpaths')


cb = topo.colourbar("topocolourbar", visible=False) # Add a colour bar
cb2 = flowverlay.colourbar("flowcolourbar", visible=False) # Add a colour bar
cb3 = flowverlay2.colourbar("flowcolourbar2", visible=False) # Add a colour bar

cm = topo.colourmap(["#004420", "#FFFFFF", "#444444"] , logscale=False, range=[-200.0, 400.0])   # Apply a built in colourmap
cm2 = flowverlay.colourmap(["#FFFFFF:0.0", "#0033FF:0.3", "#000033"], logscale=True)   # Apply a built in colourmap
cm3 = flowverlay2.colourmap(["#FFFFFF:0.0", "#0033FF:0.3", "#000033"], logscale=True)   # Apply a built in colourmap



#Filter by min height value
topo["zmin"] = -10.0

lows = lv.points(pointsize=5.0, pointtype="shiny", opacity=0.75)
lows.vertices(low_cloud+(0.0,0.0,20.0))
lows.values(mesh.height[low_points])
lows.colourmap(lavavu.cubeHelix()) #, range=[0,0.1])
lows.colourbar(visible=True)


# In[80]:

## Viewer

lv.window()

# topo.control.Checkbox('wireframe',  label="Topography wireframe")
# flowverlay.control.Checkbox('wireframe', label="Flow wireframe")

# tris.control.Range(property='zmin', range=(-1,1), step=0.001)
# lv.control.Range(command='background', range=(0,1), step=0.1, value=1)
# lv.control.Range(property='near', range=[-10,10], step=2.0)

lv.control.Checkbox(property='axis')
lv.control.Command()
lv.control.ObjectList()
lv.control.show()


# Landscape analysis statistics

# In[20]:

gradient_max = mesh.slope.max()
gradient_mean = mesh.slope.mean()
flat_spots = np.where(mesh.slope < gradient_mean*0.001)[0]
low_points = mesh.identify_low_points()

# print statistics
print(("mean gradient {}\nnumber of flat spots {}\nnumber of low points {}".format(gradient_mean,
                                                                                  flat_spots.size,
                                                                                  low_points.shape[0])))


# In[21]:

filename = 'port_macquarie_mesh_lows500x3.h5'

mesh.save_mesh_to_hdf5(filename)
mesh.save_field_to_hdf5(filename, height=mesh.height, slope=mesh.slope, flow=np.sqrt(flow2))

# to view in Paraview
meshtools.generate_xdmf(filename)
