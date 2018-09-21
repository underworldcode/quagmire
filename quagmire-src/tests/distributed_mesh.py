import numpy as np
# import matplotlib.pyplot as plt
from quagmire import FlatMesh
from quagmire import tools as meshtools
from mpi4py import MPI
comm = MPI.COMM_WORLD


minX, maxX = -5., 5.
minY, maxY = -5., 5.
spacing = 0.05

ptsx, ptsy, bmask = meshtools.poisson_elliptical_mesh(minX, maxX, minY, maxY, spacing, 500)
dm = meshtools.create_DMPlex_from_points(ptsx, ptsy, bmask, refinement_steps=1)

mesh = FlatMesh(dm)

#if comm.rank == 0:
print "Number of nodes in mesh - {}: {}".format(comm.rank, mesh.npoints)

# retrieve local mesh
x = mesh.tri.x
y = mesh.tri.y

# create height field
radius  = np.sqrt((x**2 + y**2))
theta   = np.arctan2(y,x)

height  = np.exp(-0.025*(x**2 + y**2)**2) + \
          0.25 * (0.2*radius)**4  * np.cos(10.0*theta)**2
height  += 0.5 * (1.0-0.2*radius)


rank = np.ones_like(height)*comm.rank
shadow = np.zeros_like(height)


# get shadow zones
shadow_zones = mesh.lgmap_row.indices < 0
shadow[shadow_zones] = 1
shadow_vec = mesh.gvec.duplicate()

mesh.lvec.setArray(shadow)
mesh.dm.localToGlobal(mesh.lvec, shadow_vec, addv=True)


# Save fields to file
file = "spiral.h5"
mesh.save_mesh_to_hdf5(file)
mesh.save_field_to_hdf5(file, height=height, rank=rank, shadow=shadow_vec)

# Generate XDMF file for visualising in Paraview
meshtools.generate_xdmf(file)
