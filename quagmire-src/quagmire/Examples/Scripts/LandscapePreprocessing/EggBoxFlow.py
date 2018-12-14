
# coding: utf-8

# # TriMeshes
#
#

# In[1]:

# import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import imread
from quagmire import tools as meshtools
from petsc4py import PETSc
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


# get_ipython().magic('matplotlib inline')


from quagmire import FlatMesh
from quagmire import TopoMesh # all routines we need are within this class
from quagmire import SurfaceProcessMesh

minX, maxX = 0.0, 10.0
minY, maxY = 0.0, 10.0,

x1, y1, bmask1 = meshtools.square_mesh(minX, maxX, minY, maxY, 0.1, 0.1, 50001, 200)

# In[41]:

DM = meshtools.create_DMPlex_from_points(x1, y1, bmask1, refinement_steps=2)
mesh = SurfaceProcessMesh(DM)  ## cloud array etc can surely be done better ...

print(mesh.dm.comm.rank, "Number of nodes - ", mesh.npoints)
# print mesh.dm.comm.rank, "Range - ", mesh.lgmap_row.apply([0, mesh.npoints-1])
# print mesh.dm.comm.rank, "Range2 - ", mesh.lgmap_col.apply([0, mesh.npoints-1])


# In[42]:

x = mesh.coords[:,0]
y = mesh.coords[:,1]
bmask = mesh.bmask

radius  = np.sqrt((x**2 + y**2))
theta   = np.arctan2(y,x)

height  = 1.0 - (np.sin(0.5*x*np.pi)**2 * np.sin(0.5*y*np.pi)**2)
height  += np.random.random(height.shape) * 0.001

## Steep eggbox - algorithm is quick
# height  += 1.0 * (x + y) 
## Flatter eggbox - takes longer to find catchment spills and iterate
height  += 0.25 * (x + 0.5*y) 


height0 = height.copy()
new_heights = height.copy()


rainfall = np.ones_like(height)
rainfall = height ** 2.0
# rainfall[np.where( radius > 1.0)] = 0.0

mesh.update_height(height0)	    

new_heights = mesh.low_points_swamp_fill(saddles=False)
mesh._update_height_partial(new_heights)


print(rank, " : ", "Swamp fill ... ")

for ii in range(0,3):

	new_heights=mesh.low_points_local_patch_fill(its=2, smoothing_steps=2)

	glows, glow_points = mesh.identify_global_low_points(global_array=False)
	if rank == 0:
		print("gLows: ",glows)

	for iii in range(0, 5):
		new_heights = mesh.low_points_swamp_fill(saddles=True)
		mesh._update_height_partial(new_heights)
		glows, glow_points = mesh.identify_global_low_points(global_array=False)
		if rank == 0:
			print("gLows: ",glows)

		if glows == 0:
			break

		new_heights = mesh.low_points_swamp_fill(saddles=False)
		mesh._update_height_partial(new_heights)


	# new_heights=mesh.low_points_local_flood_fill(its=1000, scale=1.0001)
	# mesh._update_height_partial(new_heights)


	# glows, glow_points = mesh.identify_global_low_points(global_array=False)
	# if rank == 0:
	# 	print "gLows: ",glows

	if glows == 0:
		break


print("Compute downhill flow ... ")

# In[44]:

mesh.downhill_neighbours = 2
mesh.update_height(new_heights)
low_points1= mesh.identify_low_points()
print(rank," : Lows ", low_points1.shape[0])

its, flowpaths = mesh.cumulative_flow_verbose(mesh.area)
flowpaths = mesh.rbf_smoother(flowpaths, iterations=1)
sqrtpaths = np.sqrt(flowpaths)

decomp = np.ones_like(mesh.height) * mesh.dm.comm.rank

low_points = mesh.identify_low_points(include_shadows=False)
glow_points = mesh.lgmap_row.apply(low_points.astype(PETSc.IntType))
ctmt = mesh.uphill_propagation(low_points,  glow_points, scale=1.000001, its=1000, fill=-1).astype(np.int)

list_of_lows = comm.gather(glow_points, root=0)

if rank == 0:
   for i in range(size):
       print("Proc ",i,":",list_of_lows[i], mesh.npoints)

   lows = np.hstack(list_of_lows)
   lows = np.unique(lows)
   print(lows)

else:
   pass



filename="Eggbox.h5"

mesh.save_mesh_to_hdf5(filename)
mesh.save_field_to_hdf5(filename, height0=height0,
								  height=mesh.height,
								  swamps=mesh.height-height0,
								  deltah=new_heights-height0,
                                  slope=mesh.slope,
                                  flow=np.sqrt(flowpaths),
                                  catchments=ctmt,
                                  bmask=mesh.bmask,
                                  decomp=decomp)

# to view in Paraview
meshtools.generate_xdmf(filename)
