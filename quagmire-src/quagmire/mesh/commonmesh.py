"""
Copyright 2016-2017 Louis Moresi, Ben Mather, Romain Beucher

This file is part of Quagmire.

Quagmire is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or any later version.

Quagmire is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with Quagmire.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
from mpi4py import MPI
import sys,petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
# comm = MPI.COMM_WORLD
from time import clock

try: range = xrange
except: pass


class CommonMesh(object):

    def __init__(self, dm, verbose=True,  *args, **kwargs):

        self.timings = dict() # store times

        self.log = PETSc.Log()
        self.log.begin()

        self.verbose = verbose

        self.dm = dm
        self.gvec = dm.createGlobalVector()
        self.lvec = dm.createLocalVector()
        self.sect = dm.getDefaultSection()
        self.sizes = self.gvec.getSizes(), self.gvec.getSizes()

        self.comm = self.dm.comm
        self.rank = self.dm.comm.rank

        lgmap_r = dm.getLGMap()
        l2g = lgmap_r.indices.copy()
        offproc = l2g < 0

        l2g[offproc] = -(l2g[offproc] + 1)
        lgmap_c = PETSc.LGMap().create(l2g, comm=self.dm.comm)

        self.lgmap_row = lgmap_r
        self.lgmap_col = lgmap_c


        return

    def add_variable(self, name=None):

        from .basemesh import MeshVariable

        variable = MeshVariable(name=name, mesh=self)

        return variable

    def get_label(self, label):
        """
        Retrieves all points in the DM that is marked with a specific label.
        e.g. "boundary", "coarse"
        """
        pStart, pEnd = self.dm.getDepthStratum(0)


        labels = []
        for i in range(self.dm.getNumLabels()):
            labels.append(self.dm.getLabelName(i))

        if label not in labels:
            raise ValueError("There is no {} label in the DM".format(label))


        stratSize = self.dm.getStratumSize(label, 1)
        if stratSize > 0:
            labelIS = self.dm.getStratumIS(label, 1)
            pt_range = np.logical_and(labelIS.indices >= pStart, labelIS.indices < pEnd)
            indices = labelIS.indices[pt_range] - pStart
        else:
            indices = np.zeros((0,), dtype=np.int)

        return indices



    def set_label(self, label, indices):
        """
        Marks local indices in the DM with a label
        """
        pStart, pEnd = self.dm.getDepthStratum(0)
        indices += pStart

        labels = []
        for i in range(self.dm.getNumLabels()):
            labels.append(self.dm.getLabelName(i))

        if label not in labels:
            self.dm.createLabel(label)
        for ind in indices:
            self.dm.setLabelValue(label, ind, 1)
        return


    def get_boundary(self, marker="boundary"):
        """
        Find the nodes on the boundary from the DM
        If marker does not exist then the convex hull is used.
        """

        pStart, pEnd = self.dm.getDepthStratum(0)
        bmask = np.ones(self.npoints, dtype=bool)


        try:
            boundary_indices = self.get_label(marker)

        except ValueError:
            self.dm.markBoundaryFaces(marker) # marks line segments
            boundary_indices = self.tri.convex_hull()
            for ind in boundary_indices:
                self.dm.setLabelValue(marker, ind + pStart, 1)


        bmask[boundary_indices] = False
        return bmask



    def save_mesh_to_hdf5(self, file):
        """
        Saves mesh information stored in the DM to HDF5 file
        If the file already exists, it is overwritten.
        """
        file = str(file)
        if not file.endswith('.h5'):
            file += '.h5'

        ViewHDF5 = PETSc.Viewer()
        ViewHDF5.createHDF5(file, mode='w')
        ViewHDF5.view(obj=self.dm)
        ViewHDF5.destroy()


    def save_field_to_hdf5(self, file, *args, **kwargs):
        """
        Saves data on the mesh to an HDF5 file
         e.g. height, rainfall, sea level, etc.

        Pass these as arguments or keyword arguments for
        their names to be saved to the hdf5 file
        """
        import os.path

        file = str(file)
        if not file.endswith('.h5'):
            file += '.h5'

        # write mesh if it doesn't exist
        # if not os.path.isfile(file):
        #     self.save_mesh_to_hdf5(file)

        kwdict = kwargs
        for i, arg in enumerate(args):
            key = "arr_{}".format(i)
            if key in list(kwdict.keys()):
                raise ValueError("Cannot use un-named variables\
                                  and keyword: {}".format(key))
            kwdict[key] = arg

        vec = self.gvec.duplicate()
        vec = self.dm.createGlobalVec()

        if os.path.isfile(file):
            mode = "a"
        else:
            mode = "w"


        for key in kwdict:
            val = kwdict[key]
            try:
                vec.setArray(val)
            except:
                self.lvec.setArray(val)
                self.dm.localToGlobal(self.lvec, vec)

            vec.setName(key)
            if self.rank == 0 and self.verbose:
                print("Saving {} to hdf5".format(key))

            ViewHDF5 = PETSc.Viewer()
            ViewHDF5.createHDF5(file, mode=mode)
            ViewHDF5.view(obj=vec)
            ViewHDF5.destroy()
            mode = "a"

        vec.destroy()


    def _gather_root(self):
        """
        MPI gather operation to root process
        """
        self.tozero, self.zvec = PETSc.Scatter.toZero(self.gvec)


        # Gather x,y points
        pts = self.tri.points
        self.lvec.setArray(pts[:,0])
        self.dm.localToGlobal(self.lvec, self.gvec)
        self.tozero.scatter(self.gvec, self.zvec)

        self.root_x = self.zvec.array.copy()

        self.lvec.setArray(pts[:,1])
        self.dm.localToGlobal(self.lvec, self.gvec)
        self.tozero.scatter(self.gvec, self.zvec)

        self.root_y = self.zvec.array.copy()

        self.root = True # yes we have gathered everything


    def gather_data(self, data):
        """
        Gather data on root process
        """

        # check if we already gathered pts on root
        if not self.root:
            self._gather_root()

        self.lvec.setArray(data)
        self.dm.localToGlobal(self.lvec, self.gvec)
        self.tozero.scatter(self.gvec, self.zvec)

        return self.zvec.array.copy()

    def scatter_data(self, data):
        """
        Scatter data to all processes
        """

        toAll, zvec = PETSc.Scatter.toAll(self.gvec)

        self.lvec.setArray(data)
        self.dm.localToGlobal(self.lvec, self.gvec)
        toAll.scatter(self.gvec, zvec)

        return zvec.array.copy()

    def sync(self, vector):
        """
        Synchronise the local domain with the global domain
        """

        if self.dm.comm.Get_size() == 1:
            return vector
        else:

            # Is this the same under 3.10 ?

            self.lvec.setArray(vector)
            # self.dm.localToLocal(self.lvec, self.gvec)
            self.dm.localToGlobal(self.lvec, self.gvec)
            self.dm.globalToLocal(self.gvec, self.lvec)

            return self.lvec.array.copy()

