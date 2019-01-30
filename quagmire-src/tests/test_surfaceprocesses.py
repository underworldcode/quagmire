"""
Parallel test
"""


import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD

from quagmire import SurfaceProcessMesh
from quagmire import tools as meshtools


# Setup distributed mesh

minX, maxX = -5., 5.
minY, maxY = -5., 5.

x, y, bmask = meshtools.elliptical_mesh(minX, maxX, minY, maxY, 0.01, 0.01, 100000, 500)
DM = meshtools.create_DMPlex_from_points(x, y, bmask, refinement_steps=1)

mesh = SurfaceProcessMesh(DM)
x, y, simplices, bmask = mesh.get_local_mesh()

height = np.exp(-0.025*(x**2 + y**2)**2) + 0.0001
rain   = height**2

mesh.update_height(height)
mesh.update_surface_processes(rain, np.zeros_like(rain))


erosion_rate, deposition_rate, stream_power = \
         mesh.stream_power_erosion_deposition_rate_old(efficiency=0.01, 
                                                      smooth_power=0, 
                                                      smooth_low_points=2, 
                                                      smooth_erosion_rate=0, 
                                                      smooth_deposition_rate=2, 
                                                      smooth_operator=mesh.downhill_smoothing,
                                                      centre_weight_u=0.75, centre_weight=0.5)


dhdt = erosion_rate - deposition_rate

# Save mesh to hdf5 file
file = "erosion-deposition.h5"

mesh.save_mesh_to_hdf5(file)
mesh.save_field_to_hdf5(file, height=height, rain=rain,\
                              dhdt=dhdt,\
                              stream_power=stream_power,\
                              rank = np.ones_like(x)*comm.rank)
meshtools.generate_xdmf(file)