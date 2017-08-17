
# coding: utf-8

# In[4]:

from quagmire import SurfaceProcessMesh
from quagmire import tools as meshtools
import numpy as np

import petsc4py
from mpi4py import MPI
from petsc4py import PETSc


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()



# In[5]:

import h5py

meshFile = h5py.File(name="../Scripts/Octopants.h5", mode="r")
points = meshFile["geometry"]["vertices"]
x1 = points.value[:,0]
y1 = points.value[:,1]
bmask = meshFile["fields"]["bmask"].value[:].astype(bool)
height = meshFile["fields"]["height"].value[:]
lakes = meshFile["fields"]["swamps"].value[:]


# In[6]:

dm  = meshtools.create_DMPlex_from_points(x1,y1,bmask=bmask)
SPM = SurfaceProcessMesh(dm)
x = SPM.coords[:,0]
y = SPM.coords[:,1]


# In[7]:

from scipy.spatial import ckdtree
old_nodes = ckdtree.cKDTree( points.value )


# In[8]:

distance, mapping = old_nodes.query(SPM.coords)
SPM.update_height(height[mapping])
lakefill = lakes[mapping]


# In[9]:

its, flowpaths2 = SPM.cumulative_flow_verbose(SPM.area, maximum_its=2000, verbose=True)


# In[10]:

low_points = SPM.identify_low_points()
low_points_plus = SPM.identify_low_points(include_shadows=True)
print dm.comm.rank,"Lows: ", low_points.shape[0], low_points_plus.shape[0]

glow_points = SPM.lgmap_row.apply(low_points.astype(PETSc.IntType))
list_of_lows = comm.gather(glow_points, root=0)


if rank == 0:
   for i in range(size):
       print "Proc ",i,":",list_of_lows[i], SPM.npoints

   lows = np.hstack(list_of_lows)
   lows = np.unique(lows)
   print lows

else:
   pass


# In[12]:

filename = 'portmacca-from-hdf.h5'

SPM.save_mesh_to_hdf5(filename)
SPM.save_field_to_hdf5(filename, 
						height=SPM.height,
						flow=np.sqrt(flowpaths2), 
						lakes=lakefill)

# to view in Paraview
meshtools.generate_xdmf(filename)



# In[ ]:



