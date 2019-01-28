from quagmire.tools import meshtools
from quagmire import FlatMesh, TopoMesh, SurfaceProcessMesh
from quagmire.mesh import MeshVariable, VectorMeshVariable

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



v = MeshVariable("stuff", mesh)
h = np.ones(mesh.npoints) * comm.rank

v.data = h
# v.data[1] = 2.0

gdata = DM.getGlobalVec()
v.getGlobalVector(gdata)

print("MeshVariable\n-------")
print("{}: vl min/max = {}/{}".format(comm.rank, v.data.array.min(), v.data.array.max()))
print("{}: vg min/max = {}/{}".format(comm.rank, gdata.array.min(), gdata.array.max()))

v.sync()
v.sync()

print("{}: vlSync min/max = {}/{}".format(comm.rank, v.data.array.min(), v.data.array.max()))
print("{}: vgSync min/max = {}/{}".format(comm.rank, gdata.array.min(), gdata.array.max()))

v.save()
print(v.data)

dv = v.gradient()
print(type(dv))


v = VectorMeshVariable("more_stuff", mesh)
h = np.ones(mesh.npoints*2) * comm.rank

v.data = h

gdata = DM.getCoordinates()
v.getGlobalVector(gdata)

print("VectorMeshVariable\n-------")
print("{}: vl min/max = {}/{}".format(comm.rank, v.data.array.min(), v.data.array.max()))
print("{}: vg min/max = {}/{}".format(comm.rank, gdata.array.min(), gdata.array.max()))

v.sync()
v.sync()

print("{}: vlSync min/max = {}/{}".format(comm.rank, v.data.array.min(), v.data.array.max()))
print("{}: vgSync min/max = {}/{}".format(comm.rank, gdata.array.min(), gdata.array.max()))

v.save()


print(v.data)