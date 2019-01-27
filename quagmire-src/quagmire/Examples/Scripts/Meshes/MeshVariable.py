from quagmire.tools import meshtools
from quagmire import FlatMesh, TopoMesh, SurfaceProcessMesh

import petsc4py
import mpi4py
import quagmire

import numpy as np

comm = mpi4py.MPI.COMM_WORLD

minX, maxX = -5.0, 5.0
minY, maxY = -5.0, 5.0,
dx, dy = 0.1, 0.1

x, y, simplices = meshtools.elliptical_mesh(minX, maxX, minY, maxY, dx, dy)

DM = meshtools.create_DMPlex_from_points(x, y, bmask=None)

mesh = TopoMesh(DM, downhill_neighbours=1, verbose=False)

print("{}: mesh size:{}".format(comm.rank, mesh.npoints))

from quagmire.mesh import MeshVariable

v = MeshVariable("stuff", mesh)
h = np.ones(mesh.npoints) * comm.rank

v.data = h
# v.data[1] = 2.0

gdata = DM.getGlobalVec()
v.getGlobalVector(gdata)

print("{}: vl min/max = {}/{}".format(comm.rank, v.data.min(), v.data.max()))
print("{}: vg min/max = {}/{}".format(comm.rank, gdata.array.min(), gdata.array.max()))

v.data = mesh.sync(v.data)


print("{}: vlSync min/max = {}/{}".format(comm.rank, v.data.min(), v.data.max()))
print("{}: vgSync min/max = {}/{}".format(comm.rank, gdata.array.min(), gdata.array.max()))

v.sync(mergeShadow=False)

print("{}: vlSync min/max = {}/{}".format(comm.rank, v.data.min(), v.data.max()))
print("{}: vgSync min/max = {}/{}".format(comm.rank, gdata.array.min(), gdata.array.max()))





h = np.cos(mesh.coords[:,0])
v.data = h
dx, dy = v.gradient()

print("INTERP")
print(v.interpolate([0.1, 10.0], [0.1, 10.0], err=False))
print(v.interpolate(0.1, 0.1))
print(v.evaluate(0.1, 0.1))