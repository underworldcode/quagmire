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

# %% [markdown] deletable=true editable=true
# # Poisson disc sampling
#
# Quality meshes are important for producing reliable solution in surface process modelling. For any given node in an unstructured mesh, its neighbours should be spaced more or less at an equal radius. For this we turn to Poisson disc sampling using an efficient $O(N)$ [algorithm](http://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf).
#
# The premise of this algorithm is to ensure that points are tightly packed together, but no closer than a specified minimum distance. This distance can be uniform across the entire domain, or alternatively a 2D numpy array of radius lengths can be used to bunch and relax the spacing of nodes.

# %% deletable=true editable=true
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import imread
from quagmire import tools as meshtools
# %matplotlib inline

# %% [markdown] deletable=true editable=true
# ## Uniform spacing

# %% deletable=true editable=true
minX, maxX = -1.0, 1.0
minY, maxY = 0.0, 1.0

x, y, bmask = meshtools.poisson_square_mesh(minX, maxX, minY, maxY, 0.02, 80)
print("{} points".format(x.size))

fig = plt.figure(1, figsize=(8,4))
ax = fig.add_subplot(111)
ax.axis('off')
ax.scatter(x[bmask], y[bmask], s=1)
ax.scatter(x[~bmask], y[~bmask], s=2)
plt.show()

# %% deletable=true editable=true
x, y, bmask = meshtools.poisson_elliptical_mesh(minX, maxX, minY, maxY, 0.02, 200)
print("{} points".format(x.size))

fig = plt.figure(1, figsize=(8,4))
ax = fig.add_subplot(111)
ax.axis('off')
ax.scatter(x[bmask], y[bmask], s=1)
ax.scatter(x[~bmask], y[~bmask], s=2)
plt.show()

# %% [markdown] deletable=true editable=true
# ## Variable spacing
#
# This is a *Poisson* disc sampler, so we sample fish...

# %% deletable=true editable=true
img = imread('data/fish.jpg', mode='F')
img = np.flipud(img)

height, width = img.shape

fig = plt.figure(1, figsize=(8,4))
ax = fig.add_subplot(111)
ax.axis('off')
ax.imshow(img, cmap='bone', origin='lower')
plt.show()

# %% [markdown] deletable=true editable=true
# We adjust the numpy array to create sensible radii

# %% deletable=true editable=true
radius = img - img.min()
radius /= img.max()
radius = 0.015*radius + 0.001

fig = plt.figure(1, figsize=(10,4))
ax = fig.add_subplot(111)
ax.axis('off')
im = ax.imshow(radius, cmap='bone', origin='lower')
fig.colorbar(im, ax=ax, label='radius')
plt.show()

# %% deletable=true editable=true
# weight.fill(10.)
x, y, bmask = meshtools.poisson_disc_sampler(minX, maxX, minY, maxY, None, r_grid=radius)
print("number of points is {}".format(x.size))

fig = plt.figure(1, figsize=(8,4))
ax = fig.add_subplot(111)
ax.axis('off')
ax.scatter(x, y, s=1, c='k')
plt.show()

# %% [markdown] deletable=true editable=true
# This is good, but what if we do not want to sample the area outside the fish?
#
# Poisson disc sampling is a flood-fill algorithm, thus we can feed the sampler an array of points that reside on the boundary between the fish and the ocean, and initiate a seed point within the fish shape.

# %% deletable=true editable=true
from scipy.ndimage.filters import gaussian_filter

silhouette = (radius > 0.015).astype(float)
silhouette = gaussian_filter(silhouette, sigma=1.)

# gradient will be high across the border
gradX, gradY = np.gradient(silhouette)
gradS = np.hypot(gradX, gradY)

# Plot border
fig = plt.figure(1, figsize=(8,4))
ax = fig.add_subplot(111)
# ax.axis('off')
im = ax.imshow(gradS, origin='lower', cmap='Greys')
fig.colorbar(im)
plt.show()


# %% deletable=true editable=true
def transform_to_coords(points, minX, maxX, minY, maxY, width, height):
    coords = np.empty_like(points)
    coords[:,0] = (maxX-minX)*(points[:,0]/width) + minX
    coords[:,1] = (maxY-minY)*(points[:,1]/height) + minY
    return coords

Xcoords = np.linspace(minX, maxX, gradS.shape[1])
Ycoords = np.linspace(minY, maxY, gradS.shape[0])
xq, yq = np.meshgrid(Xcoords, Ycoords)

j, i = np.where(gradS > 0.2)
border = np.column_stack([xq[j,i], yq[j,i]])
border = np.vstack([border,border+0.01]) # slightly thicker boundary

originX = 0.5*(maxX + minX)
originY = 0.5*(maxY + minY)

seed = np.array([originX, originY]) # centre

# %% deletable=true editable=true
x, y, bmask = meshtools.poisson_disc_sampler(minX, maxX, minY, maxY, None, r_grid=radius,
                                             cpts=border, spts=seed)

print("number of points is {}".format(x.size))

fig = plt.figure(1, figsize=(8,4))
ax = fig.add_subplot(111)
ax.axis('off')
ax.scatter(x, y, s=1, c='k')
plt.show()

# %% [markdown] deletable=true editable=true
# ### Landscape
#
# In this example we create higher resolution where the slope is steeper.

# %% deletable=true editable=true
dem = imread('data/port_macquarie.tif', mode='F')

rows, columns = dem.shape
aspect_ratio = float(columns) / float(rows)

spacing = 5.0

minX, maxX = 0.0, spacing*dem.shape[1]
minY, maxY = 0.0, spacing*dem.shape[0]


fig = plt.figure(1, figsize=(10*aspect_ratio,10))
ax = fig.add_subplot(111)
ax.axis('off')
im = ax.imshow(dem, cmap='terrain_r', origin='lower', aspect=aspect_ratio)
fig.colorbar(im, ax=ax, label='height')

# %% deletable=true editable=true
gradX, gradY = np.gradient(dem, 5., 5.) # 5m resolution in each direction
slope = np.hypot(gradX, gradY)

print("min/max slope {}".format((slope.min(), slope.max())))

# %% deletable=true editable=true
height, width = slope.shape

radius_min = 50.0
radius_max = 100.0

radius = 1.0/(slope + 0.02)
radius = (radius - radius.min()) / (radius.max() - radius.min()) 
radius = radius * (radius_max-radius_min) + radius_min

# apply gaussian filter for better results
from scipy.ndimage import gaussian_filter
radius2 = gaussian_filter(radius, 5.)

# radius -= slope.min()
# radius /= slope.max()/100
# radius += 1e-8

fig = plt.figure(1, figsize=(10*aspect_ratio, 10))
ax = fig.add_subplot(111)
ax.axis('off')
im = ax.imshow((radius2), cmap='jet', origin='lower', aspect=aspect_ratio)
fig.colorbar(im, ax=ax, label='radius2')

plt.show()

# %% deletable=true editable=true
x, y, bmask = meshtools.poisson_square_mesh(minX, maxX, minY, maxY, spacing, boundary_samples=1000, r_grid=radius2)
print("{} samples".format(x.size))

# %% deletable=true editable=true
from scipy import ndimage

coords = np.stack((y, x)).T / spacing
meshheights = ndimage.map_coordinates(dem, coords.T, order=3, mode='nearest')


fig = plt.figure(1, figsize=(10*aspect_ratio, 10))
ax = fig.add_subplot(111)
ax.axis('off')
sc = ax.scatter(x[bmask], y[bmask], s=1, c=meshheights[bmask])
sc = ax.scatter(x[~bmask], y[~bmask], s=5, c=meshheights[~bmask])

fig.colorbar(sc, ax=ax, label='height')
plt.show()

# %% [markdown] deletable=true editable=true
# ## TriMesh
#
# Now the points can be triangulated to become a quality unstructured mesh.
#
# Triangulation reorders x,y points - be careful!

# %% deletable=true editable=true
from quagmire import QuagMesh # all routines we need are within this class
from quagmire import QuagMesh

dm = meshtools.create_DMPlex_from_points(x, y, bmask, refinement_steps=0)
mesh = QuagMesh(dm)

# Triangulation reorders points
coords = np.stack((mesh.tri.points[:,1], mesh.tri.points[:,0])).T / spacing
meshheights = ndimage.map_coordinates(dem, coords.T, order=3, mode='nearest')

mesh.update_height(meshheights)

# %% deletable=true editable=true
fig = plt.figure(1, figsize=(10*aspect_ratio,10))
ax = fig.add_subplot(111)
ax.axis('off')
im1 = ax.tripcolor(mesh.tri.x, mesh.tri.y, mesh.tri.simplices, mesh.slope, linewidth=0.1, cmap='jet')
fig.colorbar(im1, ax=ax, label='slope')
plt.show()

# %% [markdown] deletable=true editable=true
# Landscape analysis statistics

# %% deletable=true editable=true
gradient_max = mesh.slope.max()
gradient_mean = mesh.slope.mean()
flat_spots = np.where(mesh.slope < gradient_mean*0.01)[0]
low_points = mesh.identify_low_points()

nodes = np.arange(0, mesh.npoints)
lows =  np.where(mesh.down_neighbour1 == nodes)[0]

# print statistics
print("mean gradient {}\nnumber of flat spots {}\nnumber of low points {}".format(gradient_mean,
                                                                                  flat_spots.size,
                                                                                  low_points.shape[0]))

# %% deletable=true editable=true
filename = 'port_macquarie_mesh.h5'

mesh.save_mesh_to_hdf5(filename)
mesh.save_field_to_hdf5(filename, height=mesh.height, slope=mesh.slope)

# to view in Paraview
meshtools.generate_xdmf(filename)
