
# coding: utf-8

# # Meshing ETOPO1
#
# In this notebook we:
#
# 1. Find the land surface in a region by filtering ETOPO1
# 2. Optionally correct for the geoid (important in low-gradient / low-lying areas)
# 4. Create a DM object and refine a few times
# 5. Save the mesh to HDF5 file

# In[1]:

from osgeo import gdal

import numpy as np

import quagmire
from quagmire import tools as meshtools

from scipy.ndimage import imread
from scipy.ndimage.filters import gaussian_filter


## Define region of interest (here NZ)


filename = 'NZTopo.h5'
bounds = (165.0, -48.5, 179.9, -34.0)
minX, minY, maxX, maxY = bounds

xres = 300
yres = 300

xx = np.linspace(minX, maxX, xres)
yy = np.linspace(minY, maxY, yres)
x1, y1 = np.meshgrid(xx,yy)
x1 += np.random.random(x1.shape) * 0.2 * (maxX-minX) / xres
y1 += np.random.random(y1.shape) * 0.2 * (maxY-minY) / yres

x1 = x1.flatten()
y1 = y1.flatten()

pts = np.stack((x1, y1)).T


# In[35]:

gtiff = gdal.Open("../Notebooks/data/ETOPO1_Ice_c_geotiff.tif")

width = gtiff.RasterXSize
height = gtiff.RasterYSize
gt = gtiff.GetGeoTransform()
img = gtiff.GetRasterBand(1).ReadAsArray().T

img = np.fliplr(img)

sliceLeft   = int(180+minX) * 60
sliceRight  = int(180+maxX) * 60
sliceBottom = int(90+minY) * 60
sliceTop    = int(90+maxY) * 60

LandImg = img[ sliceLeft:sliceRight, sliceBottom:sliceTop].T
LandImg = np.flipud(LandImg)


# In[37]:

coords = np.stack((y1, x1)).T

im_coords = coords.copy()
im_coords[:,0] -= minY
im_coords[:,1] -= minX

im_coords[:,0] *= LandImg.shape[0] / (maxY-minY)
im_coords[:,1] *= LandImg.shape[1] / (maxX-minX)
im_coords[:,0] =  LandImg.shape[0] - im_coords[:,0]


# In[38]:

from scipy import ndimage

meshheights = ndimage.map_coordinates(LandImg, im_coords.T, order=3, mode='nearest').astype(np.float)

# Fake geoid for this particular region
# meshheights -= 40.0 * (y1 - minY) / (maxY - minY)


# In[39]:

## Filter out the points we don't want at all

points = meshheights > -50

m1s = meshheights[points]
x1s = x1[points]
y1s = y1[points]

submarine = m1s < 0.0
subaerial = m1s >= 0.0


# ### 3. Create the DM
#
# The points are now read into a DM and refined so that we can achieve very high resolutions. Refinement is achieved by adding midpoints along line segments connecting each point.

# In[75]:

DM = meshtools.create_DMPlex_from_points(x1s, y1s, submarine, refinement_steps=4)
mesh = quagmire.SurfaceProcessMesh(DM, verbose=True)


# In[77]:

x2r = mesh.tri.x
y2r = mesh.tri.y
simplices = mesh.tri.simplices
bmaskr = mesh.bmask


# In[78]:

## Now re-do the allocation of points to the surface.
## In parallel this will be done process by process for a sub-set of points

coords = np.stack((y2r, x2r)).T

im_coords = coords.copy()
im_coords[:,0] -= minY
im_coords[:,1] -= minX

im_coords[:,0] *= LandImg.shape[0] / (maxY-minY)
im_coords[:,1] *= LandImg.shape[1] / (maxX-minX)
im_coords[:,0] =  LandImg.shape[0] - im_coords[:,0]


# In[79]:

from scipy import ndimage

spacing = 1.0
coords = np.stack((y2r, x2r)).T / spacing

meshheights = ndimage.map_coordinates(LandImg, im_coords.T, order=3, mode='nearest')
meshheights = mesh.rbf_smoother(meshheights, iterations=2)

subaerial =  meshheights >= 0.0
submarine = ~subaerial
mesh.bmask = subaerial


mesh.update_height(meshheights*0.001)
<<<<<<< Updated upstream
mesh.handle_low_points(its=200)
=======
mesh.handle_low_points(its=500)
>>>>>>> Stashed changes


# In[84]:

nits, flowpaths = mesh.cumulative_flow_verbose(mesh.area*np.ones_like(mesh.height), verbose=True, maximum_its=2500)
flowpaths2 = mesh.rbf_smoother(flowpaths, iterations=1)


# ## 5. Save to HDF5
#
# Save the mesh to an HDF5 file so that it can be visualised in Paraview or read into Quagmire another time. There are two ways to do this:
#
# 1. Using the `save_DM_to_hdf5` function in meshtools, or
# 2. Directly from trimesh interface using `save_mesh_to_hdf5` method.
#
# Remember to execute `petsc_gen_xdmf.py austopo.h5` to create the XML file structure necessary to visualise the mesh in paraview.

# In[85]:

<<<<<<< Updated upstream
filename = 'NZTopo.h5'
=======
>>>>>>> Stashed changes

decomp = np.ones_like(mesh.height) * mesh.dm.comm.rank

mesh.save_mesh_to_hdf5(filename)
mesh.save_field_to_hdf5(filename, height=meshheights*0.001,
                                  slope=mesh.slope,
                                  flowLP=np.sqrt(flowpaths2),
                                  decomp=decomp)

# to view in Paraview
meshtools.generate_xdmf(filename)
