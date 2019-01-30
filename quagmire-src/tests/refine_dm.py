"""
Distribute a DM from points then refine.
Modify the number of refinements with
 refine = N

Run script with
 mpirun -np <procs> python refine_dm.py
"""
refine = 0

import numpy as np
import matplotlib.pyplot as plt
from quagmire import FlatMesh
from quagmire import tools as meshtools

minX, maxX = -1., 1.
minY, maxY = -1., 1.
dx, dy = 0.01, 0.01

x, y, bmask = meshtools.elliptical_mesh(minX, maxX, minY, maxY, dx, dy, 100, 10)

inverse_bmask = ~bmask
x = np.hstack([x, 1.1*x[inverse_bmask]])
y = np.hstack([y, 1.1*y[inverse_bmask]])
bmask = np.hstack([bmask, bmask[inverse_bmask]])

dm = meshtools.create_DMPlex_from_points(x,y,bmask, refinement_steps=refine)

mesh = FlatMesh(dm)
mesh.save_mesh_to_hdf5('coarse_mesh.h5')

local_x = mesh.tri.x
local_y = mesh.tri.y
simplices = mesh.tri.simplices
bmask = mesh.bmask
coarse = mesh.get_boundary("coarse")

fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.triplot(local_x, local_y, simplices, c='b', zorder=1)
ax.scatter(local_x[~mesh.bmask], local_y[~mesh.bmask], c='r', s=100, zorder=2, label="boundary points")
# ax.scatter(local_x[~coarse], local_y[~coarse], c='g', s=100, zorder=3, label="coarse points")
plt.legend()
plt.show()