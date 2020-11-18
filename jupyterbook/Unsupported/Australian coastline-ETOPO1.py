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
# # Meshing Australia
#
# In this notebook we:
#
# 1. Import a coastline from an ESRI shapefile
# 2. Sample its interior using the poisson disc generator
# 3. Resample the interior using a DEM
# 4. Create a DM object and refine a few times
# 5. Save the mesh to HDF5 file

# %%
from osgeo import gdal

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
# %matplotlib inline


import quagmire
from quagmire import tools as meshtools
import shapefile
from scipy.ndimage import imread
from scipy.ndimage.filters import gaussian_filter
from matplotlib.colors import LightSource


# %% [markdown]
# ## 1. Import coastline shapefile
#
# This requires pyshp to be installed. We scale the points to match the dimensions of the DEM we'll use later.

# %%
def remove_duplicates(a):
    """
    find unique rows in numpy array 
    <http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array>
    """
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    dedup = np.unique(b).view(a.dtype).reshape(-1, a.shape[1])
    return dedup

coast_shape = shapefile.Reader("data/AustCoast/AustCoast2.shp")
shapeRecs = coast_shape.shapeRecords()
coords = []
for record in shapeRecs:
    coords.append(record.shape.points[:])
    
coords = np.vstack(coords)

# Remove duplicates
points = remove_duplicates(coords)

# %%
gtiff = gdal.Open("data/ETOPO1_Ice_c_geotiff.tiff")

# %%
width = gtiff.RasterXSize
height = gtiff.RasterYSize
gt = gtiff.GetGeoTransform()
minX = gt[0]
minY = gt[3] + width*gt[4] + height*gt[5] 
maxX = gt[0] + width*gt[1] + height*gt[2]
maxY = gt[3]

img = gtiff.GetRasterBand(1).ReadAsArray().T
# img = np.flipud(img).astype(float)

img = np.fliplr(img)

print minX, minY, maxX, maxY

# %%
ausBounds = [110, -45 , 155, -5]
minX, minY, maxX, maxY = ausBounds

sliceLeft   = int(180+minX) * 60
sliceRight  = int(180+maxX) * 60
sliceBottom = int(90+minY) * 60
sliceTop    = int(90+maxY) * 60

AusImg = img[ sliceLeft:sliceRight, sliceBottom:sliceTop].T
AusImg = np.flipud(AusImg)

print AusImg.shape

# %%
plt.imshow(AusImg, extent=ausBounds)

# %%
fig = plt.figure(1, figsize=(14,10))
ax = fig.add_subplot(111, xlim=(minX,maxX), ylim=(minY,maxY))
# ax.axis('off')
ls = LightSource(azdeg=315, altdeg=45)
rgb = ls.shade(AusImg, cmap=cm.terrain, blend_mode='soft', vert_exag=2., dx=50, dy=50)
im1 = ax.imshow(rgb, extent=[minX, maxX, minY, maxY], cmap=cm.terrain, origin='upper')
ax.scatter(points[:,0], points[:,1], s=0.1)
plt.show()

# %% [markdown]
# ## 2. Poisson disc sampler
#
# We need seed points to initiate the poisson disc generator within the coastline of mainland Australia and Tasmania.
# Also, we add extra points around the coastline to prevent the flood fill algorithm to sample any ocean.

# %%
points2 = points.copy()
points2[:,1] -= 0.1

points3 = points.copy()
points3[:,0] -= 0.1

cpts = np.vstack((points, points2, points3))
spts = np.array([[135., -25.], [147., -42.5]])

x1, y1, bmask = meshtools.poisson_disc_sampler(minX, maxX, minY, maxY, 0.2, cpts=cpts, spts=spts)

# %%
fig = plt.figure(1, figsize=(9,7))
ax = fig.add_subplot(111)
ax.axis('off')
ax.scatter(x1[bmask], y1[bmask], s=0.1)
ax.scatter(x1[~bmask], y1[~bmask], s=0.1)
plt.show()

# %% [markdown]
# ## 3. Refined poisson disc sampler
#
# We resample the points from a minimum radius field based on the topography of Australia.
#
# The minimum radius field is created by smoothing the DEM we imported earlier, then rescaling the array to an appropriate radius. The radius field should always be greater than zero and an appropriate resolution (keeping in mind we will refine the points further after we create the DM).

# %%
gradX, gradY = np.gradient(img)
smooth_derivative = gaussian_filter(np.hypot(gradX, gradY), 50.)

r_min = 0.02
r_scale = 0.25

rgrid = -smooth_derivative
rgrid -= rgrid.min()
rgrid = r_scale*rgrid/rgrid.max() + r_min

rgrid = np.flipud(rgrid).astype(float)

# %%
plt.imshow(rgrid, origin='lower')
plt.colorbar()

# %%
points2 = points.copy()
points2[:,1] += 0.1

points3 = points.copy()
points3[:,0] += 0.1

# we require thicker set of constraint points (coasts have very small radius)
cpts2 = np.vstack((cpts, points2, points3))

# %%
rgrid.mean()

# %%
x2, y2, bmask = meshtools.poisson_disc_sampler(minX, maxX, minY, maxY, 0.2,  cpts=cpts2, spts=spts)
print("number of points {}".format(x2.shape[0]))

# %%
fig = plt.figure(1, figsize=(9,7))
ax = fig.add_subplot(111)
ax.axis('off')
ax.scatter(x2[bmask], y2[bmask], s=0.1)
ax.scatter(x2[~bmask], y2[~bmask], s=0.1)
plt.show()

# %% [markdown]
# ## 4. Create the DM
#
# The points are now read into a DM and refined so that we can achieve very high resolutions. Refinement is achieved by adding midpoints along line segments connecting each point.

# %%
DM = meshtools.create_DMPlex_from_points(x2, y2, bmask, refinement_steps=4)

# %%
DM.getCoordinates().array.shape

# %%
mesh = quagmire.QuagMesh(DM)

# %%
x2r = mesh.tri.x
y2r = mesh.tri.y
simplices = mesh.tri.simplices
bmaskr = mesh.bmask

# %%
coords = np.stack((y2r, x2r)).T

im_coords = coords.copy()
im_coords[:,0] -= minY
im_coords[:,1] -= minX

im_coords[:,0] *= img.shape[0] / (maxY-minY) 
im_coords[:,1] *= img.shape[1] / (maxX-minX) 

im_coords[:,0] =  img.shape[0] - im_coords[:,0]

# %%

# %%
from scipy import ndimage

spacing = 1.0
coords = np.stack((y2r, x2r)).T / spacing

meshheights = ndimage.map_coordinates(img, im_coords.T, order=3, mode='nearest')
meshheights = np.maximum(-100.0, meshheights)

meshheights = mesh.rbf_smoother(meshheights)
meshheights = mesh.rbf_smoother(meshheights)
meshheights = mesh.rbf_smoother(meshheights)



# %%
## Fake geoid. This should perhaps be a struct on the surface mesh as it is not an actual height change !

meshheights += 40.0*(mesh.coords[:,0]-minX)/(maxX-minX) + 40.0*(mesh.coords[:,1]-minY)/(maxY-minY) 

# %%

# %%
# for i in range(0, 10):
#     meshheights = mesh.handle_low_points(0.0, 20)
#     mesh.update_height(meshheights)
#     low_points = mesh.identify_low_points()
#     print low_points.shape[0]
    
mesh.update_height(meshheights*0.001)

flowpaths = mesh.cumulative_flow(np.ones_like(mesh.height))




# %%
fig = plt.figure(1, figsize=(14, 10))
ax = fig.add_subplot(111)
ax.axis('off')

# sc = ax.scatter(im_coords[bmaskr,1], im_coords[bmaskr,0], s=1, c=meshheights[bmaskr])
sc = ax.scatter(im_coords[~bmaskr,1], im_coords[~bmaskr,0], s=1, c=meshheights[~bmaskr])
ax.imshow(img, alpha=0.5)

fig.colorbar(sc, ax=ax, label='height')
plt.show()


# %%
from LavaVu import lavavu

low_points = mesh.identify_low_points()

manifold = np.reshape(mesh.tri.points, (-1,2))
manifold = np.insert(manifold, 2, values=mesh.height, axis=1)

low_cloud = manifold[low_points]

lv = lavavu.Viewer(border=False, background="#FFFFFF", resolution=[1000,600], near=-10.0)

topo  = lv.triangles("topography",  wireframe=False, opacity=0.5)
topo.vertices(manifold)
topo.indices(mesh.tri.simplices)

flowverlay = lv.triangles("flow_surface", wireframe=False)
flowverlay.vertices(manifold + (0.0,0.0, 0.01))
flowverlay.indices(mesh.tri.simplices)

# Add properties to manifolds

topo.values(mesh.height, 'topography')
flowverlay.values((flowpaths), 'flowpaths')

cb = topo.colourbar(visible=False) # Add a colour bar

cm = topo.colourmap(["#004420", "#FFFFFF", "#444444"] , logscale=False, range=[-600.0, 1200.0])   # Apply a built in colourmap
cm = flowverlay.colourmap(["#FFFFFF:0.0", "#0033FF:0.3", "#000033"], logscale=True)   # Apply a built in colourmap

#Filter by min height value
topo["zmin"] = 0.015

lows = lv.points("lows", pointsize=0.5, pointtype="shiny", opacity=0.75)
lows.vertices(low_cloud+(0.0,0.0,0.2))
lows.values(flowpaths[low_points])
lows.colourmap(lavavu.cubeHelix()) #, range=[0,0.1])
lows.colourbar(visible=True)

# %%
manifold[bmaskr,2].min()

# %%
lv.window()

topo.control.Checkbox('wireframe',  label="Topography wireframe")
flowverlay.control.Checkbox('wireframe', label="Flow wireframe")

# tris.control.Range(property='zmin', range=(-1,1), step=0.001)
# lv.control.Range(command='background', range=(0,1), step=0.1, value=1)
# lv.control.Range(property='near', range=[-10,10], step=2.0)
lv.control.Checkbox(property='axis')
lv.control.Command()
lv.control.ObjectList()
lv.control.show()

# %%

# %%
lv.image(filename="AusFlow.png", resolution=(6000,4000))

# %%
# fig = plt.figure(1, figsize=(14, 10))
# ax = fig.add_subplot(111)
# ax.axis('off')

# sc = ax.scatter(x2r[bmaskr], y2r[bmaskr], s=1, c=mesh.height[bmaskr])
# ax.scatter(x2r[~bmaskr], y2r[~bmaskr], s=1, c=mesh.height[~bmaskr])

# fig.colorbar(sc, ax=ax, label='height')
# plt.show()

# %% [markdown]
# ## 5. Save to HDF5
#
# Save the mesh to an HDF5 file so that it can be visualised in Paraview or read into Quagmire another time. There are two ways to do this:
#
# 1. Using the `save_DM_to_hdf5` function in meshtools, or
# 2. Directly from trimesh interface using `save_mesh_to_hdf5` method.
#
# Remember to execute `petsc_gen_xdmf.py austopo.h5` to create the XML file structure necessary to visualise the mesh in paraview.

# %%
mesh.save_mesh_to_hdf5('austopo.h5')

# %%
