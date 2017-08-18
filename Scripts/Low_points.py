
# coding: utf-8

# # Poisson disc sampling
# 
# Quality meshes are important for producing reliable solution in surface process modelling. For any given node in an unstructured mesh, its neighbours should be spaced more or less at an equal radius. For this we turn to Poisson disc sampling using an efficient $O(N)$ [algorithm](http://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf).
# 
# The premise of this algorithm is to ensure that points are tightly packed together, but no closer than a specified minimum distance. This distance can be uniform across the entire domain, or alternatively a 2D numpy array of radius lengths can be used to bunch and relax the spacing of nodes.

# In[1]:

# import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import imread
from quagmire import tools as meshtools
from petsc4py import PETSc
from mpi4py import MPI
from quagmire import SurfaceProcessMesh


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


# ## Uniform spacing

# ### Landscape
# 
# In this example we create higher resolution where the slope is steeper.

# In[2]:

dem = imread('../Notebooks/data/port_macquarie.tif', mode='F')

rows, columns = dem.shape
aspect_ratio = float(columns) / float(rows)

spacing = 5.0

minX, maxX = 0.0, spacing*dem.shape[1]
minY, maxY = 0.0, spacing*dem.shape[0]


# fig = plt.figure(1, figsize=(10*aspect_ratio,10))
# ax = fig.add_subplot(111)
# ax.axis('off')
# im = ax.imshow(dem, cmap='terrain_r', origin='lower', aspect=aspect_ratio)
# fig.colorbar(im, ax=ax, label='height')


# In[3]:

gradX, gradY = np.gradient(dem, 5., 5.) # 5m resolution in each direction
slope = np.hypot(gradX, gradY)

print("min/max slope {}".format((slope.min(), slope.max())))


# In[4]:

height, width = slope.shape

radius_min = 50.0
radius_max = 100.0

radius = 1.0/(slope + 0.02)
radius = (radius - radius.min()) / (radius.max() - radius.min()) 
radius = radius * (radius_max-radius_min) + radius_min

# apply gaussian filter for better results
from scipy.ndimage import gaussian_filter
radius2 = gaussian_filter(radius, 5.)

# In[5]:

x, y, bmask = meshtools.poisson_square_mesh(minX, maxX, minY, maxY, spacing*2.0, boundary_samples=500, r_grid=radius2*2.0)
print("{} samples".format(x.size))


# In[6]:

from scipy import ndimage

coords = np.stack((y, x)).T / spacing
meshheights = ndimage.map_coordinates(dem, coords.T, order=3, mode='nearest')



dm = meshtools.create_DMPlex_from_points(x, y, bmask, refinement_steps=1)
mesh = SurfaceProcessMesh(dm)

# Triangulation reorders points

print rank, " : ", "Map DEM to local triangles"

coords = np.stack((mesh.tri.points[:,1], mesh.tri.points[:,0])).T / spacing
meshheights = ndimage.map_coordinates(dem, coords.T, order=3, mode='nearest')

print rank, " : ", "Rebuild height information"


mesh.downhill_neighbours = 2
mesh.update_height(meshheights)
raw_heights = mesh.height.copy()

gradient_max = mesh.slope.max()
gradient_mean = mesh.slope.mean()
flat_spots = np.where(mesh.slope < gradient_mean*0.01)[0]
low_points = mesh.identify_low_points()

# print statistics
print("mean gradient {}\nnumber of flat spots {}\nnumber of low points {}".format(gradient_mean,
                                                                                  flat_spots.size,
                                                                                  low_points.shape[0]))


# In[18]:

for ii in range(0,5):

	new_heights=mesh.low_points_local_patch_fill(its=3, smoothing_steps=1)

	glows, glow_points = mesh.identify_global_low_points(global_array=False)
	if rank == 0:
		print "gLows: ",glows

	for iii in range(0, 5):
		new_heights = mesh.low_points_swamp_fill()
		mesh._update_height_partial(new_heights)
		glows, glow_points = mesh.identify_global_low_points(global_array=False)
		if rank == 0:
			print "gLows: ",glows

		if glows == 0:
			break

		new_heights=mesh.low_points_local_flood_fill(its=10, scale=1.0001)
		mesh._update_height_partial(new_heights)

	glows, glow_points = mesh.identify_global_low_points(global_array=False)
	if rank == 0:
		print "gLows: ",glows

	if glows == 0:
		break


mesh.downhill_neighbours = 2
mesh.update_height(mesh.height)

its, flowpaths2 = mesh.cumulative_flow_verbose(mesh.area, maximum_its=2000, verbose=True)
flowpaths2 = mesh.rbf_smoother(flowpaths2)

decomp = np.ones_like(mesh.height) * mesh.dm.comm.rank

low_points = mesh.identify_low_points()
glow_points = mesh.lgmap_row.apply(low_points.astype(PETSc.IntType))

ctmt = mesh.uphill_propagation(low_points,  glow_points, scale=1.0, its=250, fill=-1).astype(np.int)


filename = 'portmacca.h5'

mesh.save_mesh_to_hdf5(filename)

mesh.save_field_to_hdf5(filename, 
						height0=raw_heights)

						# height=mesh.height, 
						# slope=mesh.slope, 
						# flow=np.sqrt(flowpaths2), 
						# lakes=(mesh.height - raw_heights),
						# catchments=ctmt,
						# decomp=decomp,
						# bmask=mesh.bmask)

# to view in Paraview
meshtools.generate_xdmf(filename)





