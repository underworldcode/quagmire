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
comm = MPI.COMM_WORLD
from time import clock

try: range = xrange
except: pass


class PixMesh(object):
    """
    Creating a global vector from a distributed DM removes duplicate entries (shadow zones)
    """
    def __init__(self, dm, verbose=True, *args, **kwargs):
        from scipy.spatial import Delaunay
        from scipy.spatial import cKDTree as _cKDTree

        self.timings = dict() # store times

        self.log = PETSc.Log()
        self.log.begin()

        self.verbose = verbose

        self.dm = dm
        self.gvec = dm.createGlobalVector()
        self.lvec = dm.createLocalVector()
        self.sizes = self.gvec.getSizes(), self.gvec.getSizes()

        lgmap_r = dm.getLGMap()
        l2g = lgmap_r.indices.copy()
        offproc = l2g < 0

        l2g[offproc] = -(l2g[offproc] + 1)
        lgmap_c = PETSc.LGMap().create(l2g, comm=comm)

        self.lgmap_row = lgmap_r
        self.lgmap_col = lgmap_c


        (minX, maxX), (minY, maxY) = dm.getBoundingBox()
        Nx, Ny = dm.getSizes()

        dx = (maxX - minX)/(Nx - 1)
        dy = (maxY - minY)/(Ny - 1)
        # assert dx == dy, "Uh oh! Each cell should be square, not rectangular."

        self.area = np.array(dx*dy)
        self.adjacency_weight = 0.5

        self.bc = dict()
        self.bc["top"]    = (1,maxY)
        self.bc["bottom"] = (1,minY)
        self.bc["left"]   = (0,minX)
        self.bc["right"]  = (0,maxX)

        (minI, maxI), (minJ, maxJ) = dm.getGhostRanges()

        nx = maxI - minI
        ny = maxJ - minJ

        self.dx, self.dy = dx, dy
        self.nx, self.ny = nx, ny


        # Get local coordinates
        self.coords = dm.getCoordinatesLocal().array.reshape(-1,2)

        (minX, maxX), (minY, maxY) = dm.getLocalBoundingBox()

        self.minX, self.maxX = minX, maxX
        self.minY, self.maxY = minY, maxY

        self.npoints = nx*ny


        # Find neighbours
        t = clock()
        self.construct_neighbours()
        self.timings['construct neighbours'] = [clock()-t, self.log.getCPUTime(), self.log.getFlops()]
        if self.verbose:
            print(" - Construct neighbour array {}s".format(clock()-t))

        # cKDTree
        t = clock()
        self.cKDTree = _cKDTree(self.coords)
        self.timings['cKDTree'] = [clock()-t, self.log.getCPUTime(), self.log.getFlops()]
        if self.verbose:
            print(" - cKDTree {}s".format(clock()-t))


        # Find boundary points
        t = clock()
        self.bmask = self.get_boundary()
        self.timings['find boundaries'] = [clock()-t, self.log.getCPUTime(), self.log.getFlops()]
        if self.verbose:
            print(" - Find boundaries {}s".format(clock()-t))


        # Build smoothing operator
        t = clock()
        self._build_smoothing_matrix()
        self.timings['smoothing matrix'] = [clock()-t, self.log.getCPUTime(), self.log.getFlops()]
        if self.verbose:
            print(" - Build smoothing matrix {}s".format(clock()-t))



        # Find neighbours
        t = clock()
        self.construct_neighbour_cloud()
        self.timings['construct neighbour cloud'] = [clock()-t, self.log.getCPUTime(), self.log.getFlops()]
        if self.verbose:
            print(" - Construct neighbour cloud array {}s".format(clock()-t))


        # RBF smoothing operator
        t = clock()
        self._construct_rbf_weights()
        self.timings['construct rbf weights'] = [clock()-t, self.log.getCPUTime(), self.log.getFlops()]
        if self.verbose:
            print(" - Construct rbf weights {}s".format(clock()-t))



        self.root = False


    def derivative_grad(self, PHI):

        u = PHI.reshape(self.ny, self.nx)
        u_x, u_y = np.gradient(u, self.dx, self.dy)

        return u_x.ravel(), u_y.ravel()


    def derivative_div(self, PHIx, PHIy):

        u_xx, u_xy = self.derivative_grad(PHIx)
        u_yx, u_yy = self.derivative_grad(PHIy)

        return u_xx + u_yy

    def construct_neighbours(self):

        nx, ny = self.nx, self.ny
        n = self.npoints
        nodes = np.arange(0, n, dtype=PETSc.IntType)

        index = np.empty((ny+2, nx+2), dtype=PETSc.IntType)
        index.fill(-1)
        index[1:-1,1:-1] = nodes.reshape(ny,nx)

        closure = [(0,-2), (1,-1), (2,0), (1,-1), (1,-1)]
        # closure = [(0,-2),(0,-2),(2,0),(2,0),(0,-2),(1,-1),(2,0),(1,-1),(1,-1)]
        nc = len(closure)

        rows = np.empty((nc,n), dtype=PETSc.IntType)
        cols = np.empty((nc,n), dtype=PETSc.IntType)
        vals = np.empty((nc,n))

        for i in xrange(0,nc):
            rs, re = closure[i]
            cs, ce = closure[-1+i]

            rows[i] = nodes
            cols[i] = index[rs:ny+re+2,cs:nx+ce+2].ravel()

        row = rows.ravel()
        col = cols.ravel()

        # sort by row
        sort = row.argsort()
        row = row[sort]
        col = col[sort]


        mask = col > -1

        # Save 5 point neighbour stencil
        # self.neighbour_block = np.ma.array(col.reshape(-1,nc), mask=~mask)

        # mask off-grid entries
        row = row[mask]
        col = col[mask]


        nnz = np.bincount(row) # number of nonzeros
        indptr = np.insert(np.cumsum(nnz),0,0)

        self.vertex_neighbours = nnz.astype(PETSc.IntType)
        self.vertex_neighbour_vertices = indptr, col


        # We may not need this, but constuct anyway for now!
        closed_neighbours = [[]]*self.npoints

        for i in xrange(0,indptr.size-1):
            start, end = indptr[i], indptr[i+1]
            closed_neighbours[i] = np.array(col[start:end])

        self.neighbour_array = np.array(closed_neighbours)


    def sort_nodes_by_field2(self, field):
        """
        Generate an array of the two lowest nodes and a highest node

        Sorting on masked arrays always returns with masked points last.
        Each node has at least 3 closed neighbour nodes, so we can vectorise
        this to some extent.
        """

        mask = self.neighbour_block.mask
        n_offgrid = mask.sum(axis=1)

        nfield = np.ma.array(field[self.neighbour_block], mask=mask)
        order = nfield.argsort(axis=1)


        # We know there is going to be at least 3 entries
        # [lowest, 2nd lowest, highest]
        neighbour_array_l2h = np.empty((self.npoints,3), dtype=PETSc.IntType)

        for i in xrange(0,3):
            node_mask = n_offgrid == i

            n0 = order[node_mask,0]    # lowest
            n1 = order[node_mask,1]    # second lowest
            n2 = order[node_mask,-1-i] # highest

            neighbour_array_l2h[node_mask,0] = self.neighbour_block[node_mask,n0]
            neighbour_array_l2h[node_mask,1] = self.neighbour_block[node_mask,n1]
            neighbour_array_l2h[node_mask,2] = self.neighbour_block[node_mask,n2]

        self.neighbour_array_l2h = neighbour_array_l2h


    def construct_neighbour_cloud(self, size=25):
        """
        Find neighbours from distance cKDTree.

        """

        nndist, nncloud = self.cKDTree.query(self.coords, k=size)

        self.neighbour_cloud = nncloud
        self.neighbour_cloud_distances = nndist

        return



    def _build_smoothing_matrix(self):

        indptr, indices = self.vertex_neighbour_vertices
        weight  = np.ones(self.npoints)*(self.area*4)
        nweight = weight[indices]

        lgmask = self.lgmap_row.indices >= 0


        nnz = self.vertex_neighbours[lgmask] + 1


        smoothMat = PETSc.Mat().create(comm=comm)
        smoothMat.setType('aij')
        smoothMat.setSizes(self.sizes)
        smoothMat.setLGMap(self.lgmap_row, self.lgmap_col)
        smoothMat.setFromOptions()
        smoothMat.setPreallocationNNZ(nnz)

        # read in data
        smoothMat.setValuesLocalCSR(indptr.astype(PETSc.IntType), indices.astype(PETSc.IntType), nweight)
        # self.lvec.setArray(weight)
        # self.dm.localToGlobal(self.lvec, self.gvec)
        # smoothMat.setDiagonal(self.gvec)

        smoothMat.assemblyBegin()
        smoothMat.assemblyEnd()

        self.localSmoothMat = smoothMat


    def local_area_smoothing(self, data, its=1, centre_weight=0.75):

        self.lvec.setArray(data)
        self.dm.localToGlobal(self.lvec, self.gvec)
        smooth_data = self.gvec.copy()

        for i in xrange(0, its):
            self.localSmoothMat.mult(smooth_data, self.gvec)
            smooth_data = centre_weight*smooth_data + (1.0 - centre_weight)*self.gvec

        self.dm.globalToLocal(smooth_data, self.lvec)

        return self.lvec.array


    def get_boundary(self):

        bmask = np.ones(self.npoints, dtype=bool)

        for key in self.bc:
            i, wall = self.bc[key]
            mask = self.coords[:,i] == wall
            bmask[mask] = False

        return bmask


    def _gather_root(self):
        """
        MPI gather operation to root processor
        """
        self.tozero, self.zvec = PETSc.Scatter.toZero(self.gvec)
        self.nvec = self.dm.createNaturalVector()
        self.root = True # yes we have gathered everything


    def gather_data(self, data):
        """
        Gather data on root processor
        """

        # check if we already gathered pts on root
        if not self.root:
            self._gather_root()

        self.lvec.setArray(data)
        self.dm.localToGlobal(self.lvec, self.gvec)
        self.dm.globalToNatural(self.gvec, self.nvec)
        self.tozero.scatter(self.nvec, self.zvec)

        return self.zvec.array.copy()

    def sync(self, vector):
        """
        Synchronise the local domain with the global domain
        """
        self.lvec.setArray(vector)
        self.dm.localToGlobal(self.lvec, self.gvec)
        self.dm.globalToLocal(self.gvec, self.lvec)
        return self.lvec.array.copy()

    def _construct_rbf_weights(self, delta=None):

        self.delta  = delta

        if self.delta == None:
            self.delta = self.neighbour_cloud_distances[:,1].mean() # * 0.75

        # Initialise the interpolants

        gaussian_dist_w       = np.zeros_like(self.neighbour_cloud_distances)
        gaussian_dist_w[:,:]  = np.exp(-np.power(self.neighbour_cloud_distances[:,:]/self.delta, 2.0))
        gaussian_dist_w[:,:] /= gaussian_dist_w.sum(axis=1).reshape(-1,1)

        self.gaussian_dist_w = gaussian_dist_w

        return

    def rbf_smoother(self, field):

        # Should do some error checking here to ensure the field and point cloud are compatible

        smoothfield = (field[self.neighbour_cloud[:,:]] * self.gaussian_dist_w[:,:]).sum(axis=1)

        return smoothfield
