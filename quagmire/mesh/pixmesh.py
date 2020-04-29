# Copyright 2016-2020 Louis Moresi, Ben Mather, Romain Beucher
# 
# This file is part of Quagmire.
# 
# Quagmire is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or any later version.
# 
# Quagmire is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with Quagmire.  If not, see <http://www.gnu.org/licenses/>.

"""
For structured data on a regular grid.

<img src="https://raw.githubusercontent.com/underworldcode/quagmire/dev/docs/images/quagmire-flowchart-pixmesh.png" style="width: 321px; float:right">

`PixMesh` implements the following functionality:

- calculating spatial derivatives
- identifying node neighbour relationships
- interpolation / extrapolation
- smoothing operators
- importing and saving mesh information

Supply a `PETSc DM` object (created from `quagmire.tools.meshtools`) to initialise the object.
"""

import numpy as np
from mpi4py import MPI
import sys,petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
# comm = MPI.COMM_WORLD
from time import perf_counter
from .commonmesh import CommonMesh as _CommonMesh

try: range = xrange
except: pass


class PixMesh(_CommonMesh):
    """
    Build spatial data structures on an __structured Cartesian grid__.

    Use `PixMesh` for:

    - calculating spatial derivatives
    - identifying node neighbour relationships
    - interpolation / extrapolation
    - smoothing operators
    - importing and saving mesh information

    Each of these data structures are built on top of a `PETSc DM` object
    (created from `quagmire.tools.meshtools`).

    Parameters
    ----------
    DM : PETSc DMDA object
        Build this unstructured Cartesian mesh object using one of:

        - `quagmire.tools.meshtools.create_DMDA`
    verbose : bool
        Flag toggles verbose output
    *args : optional arguments
    **kwargs : optional keyword arguments

    Attributes
    ----------
    dx : float
        spacing in the x direction
    dy : float
        spacing in the y direction
    npoints : int
        Number of points (n) in the mesh
    pointwise_area : Quagmire MeshVariable
        `quagmire.mesh.basemesh.MeshVariable` of point-wise area
    mask : Quagmire MeshVariable
        `quagmire.mesh.basemesh.MeshVariable` to denote points on the boundary
    data : array of floats, shape (n,2)
        Cartesian mesh coordinates in x,y directions
    coords : array of floats, shape (n,2)
        Same as `data`
    neighbour_array : array of ints, shape(n,25)
        array of node neighbours
    timings : dict
        Timing information for each Quagmire routine
    """

    ## This is a count of all the instances of the class that are launched
    ## so that we have a way to name / identify them

    __count = 0

    @classmethod
    def _count(cls):
        PixMesh.__count += 1
        return PixMesh.__count

    @property
    def id(self):
        return self.__id


    def __init__(self, dm, verbose=True, *args, **kwargs):
        from scipy.spatial import cKDTree as _cKDTree

        # initialise base mesh class
        super(PixMesh, self).__init__(dm, verbose)

        self.__id = "pixmesh_{}".format(self._count())

        (minX, maxX), (minY, maxY) = dm.getBoundingBox()
        Nx, Ny = dm.getSizes()

        dx = (maxX - minX)/(Nx - 1)
        dy = (maxY - minY)/(Ny - 1)
        # assert dx == dy, "Uh oh! Each cell should be square, not rectangular."

        self.area = np.array(dx*dy)
        self.adjacency_weight = 0.5
        self.pointwise_area = self.add_variable(name="area")
        self.pointwise_area.data = self.area
        self.pointwise_area.lock()

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
        self.data = self.coords

        (minX, maxX), (minY, maxY) = dm.getLocalBoundingBox()

        self.minX, self.maxX = minX, maxX
        self.minY, self.maxY = minY, maxY

        self.npoints = nx*ny


        # cKDTree
        t = perf_counter()
        self.cKDTree = _cKDTree(self.coords)
        self.timings['cKDTree'] = [perf_counter()-t, self.log.getCPUTime(), self.log.getFlops()]
        if self.rank == 0 and self.verbose:
            print("{} - cKDTree {}s".format(self.dm.comm.rank, perf_counter()-t))


        # Find boundary points
        t = perf_counter()
        self.bmask = self.get_boundary()
        self.mask = self.add_variable(name="Mask")
        self.mask.data = self.bmask.astype(PETSc.ScalarType)
        self.mask.lock()
        self.timings['find boundaries'] = [perf_counter()-t, self.log.getCPUTime(), self.log.getFlops()]
        if self.rank == 0 and self.verbose:
            print("{} - Find boundaries {}s".format(self.dm.comm.rank, perf_counter()-t))


        # Find neighbours
        t = perf_counter()
        self.construct_neighbour_cloud()
        self.timings['construct neighbour cloud'] = [perf_counter()-t, self.log.getCPUTime(), self.log.getFlops()]
        if self.rank == 0 and self.verbose:
            print("{} - Construct neighbour cloud array {}s".format(self.dm.comm.rank, perf_counter()-t))


        # RBF smoothing operator
        t = perf_counter()
        self._construct_rbf_weights()
        self.timings['construct rbf weights'] = [perf_counter()-t, self.log.getCPUTime(), self.log.getFlops()]
        if self.rank == 0 and self.verbose:
            print("{} - Construct rbf weights {}s".format(self.dm.comm.rank, perf_counter()-t))


        self.root = False

        # functions / parameters that are required for compatibility among FlatMesh types
        self._radius = 1.0


    def derivative_grad(self, PHI):
        """
        Compute derivatives of PHI in the x, y directions.
        This routine uses SRFPACK to compute derivatives on a C-1 bivariate function.

        Arguments
        ---------
        PHI : ndarray of floats, shape (n,)
            compute the derivative of this array
        nit : int optional (default: 10)
            number of iterations to reach convergence
        tol : float optional (default: 1e-8)
            convergence is reached when this tolerance is met

        Returns
        -------
        PHIx : ndarray of floats, shape(n,)
            first partial derivative of PHI in x direction
        PHIy : ndarray of floats, shape(n,)
            first partial derivative of PHI in y direction
        """
        u = PHI.reshape(self.ny, self.nx)
        u_x, u_y = np.gradient(u, self.dx, self.dy)

        return u_x.ravel(), u_y.ravel()


    def derivative_div(self, PHIx, PHIy):
        """
        Compute second order derivative from flux fields PHIx, PHIy
        We evaluate the gradient on these fields using the derivative-grad method.

        Arguments
        ---------
        PHIx : ndarray of floats, shape (n,)
            array of first partial derivatives in x direction
        PHIy : ndarray of floats, shape (n,)
            array of first partial derivatives in y direction
        kwargs : optional keyword-argument specifiers
            keyword arguments to be passed onto derivative_grad
            e.g. nit=5, tol=1e-3

        Returns
        -------
        del2PHI : ndarray of floats, shape (n,)
            second derivative of PHI
        """
        u_xx, u_xy = self.derivative_grad(PHIx)
        u_yx, u_yy = self.derivative_grad(PHIy)

        return u_xx + u_yy

    def construct_neighbours(self):
        """
        Find neighbours from edges and store as CSR coordinates.

        This allows you to directly ask the neighbours for a given node a la Qhull,
        or efficiently construct a sparse matrix (PETSc/SciPy)

        Notes
        -----
        This method searches only for immediate note neighbours that are connected
        by a line segment.
        """

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

        for i in range(0,nc):
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

        for i in range(0,indptr.size-1):
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

        for i in range(0,3):
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

        # identify corner nodes
        corners = [0, self.nx-1, -self.nx, -1]

        # interior nodes have 3*3 neighbours (including self)
        neighbours = np.full(self.npoints, 9, dtype=np.int)
        neighbours[~self.bmask] = 6 # walls have 3*2 neighbours
        neighbours[corners] = 4 # corners have 4 neighbours

        self.near_neighbours = neighbours + 2
        self.extended_neighbours = np.full_like(neighbours, size)

        self.near_neighbour_mask = np.zeros_like(self.neighbour_cloud, dtype=np.bool)

        for node in range(0,self.npoints):
            self.near_neighbour_mask[node, 0:self.near_neighbours[node]] = True

        return

    def _build_smoothing_matrix(self):

        indptr, indices = self.vertex_neighbour_vertices
        weight  = np.ones(self.npoints)*(self.area*4)
        nweight = weight[indices]

        lgmask = self.lgmap_row.indices >= 0


        nnz = self.vertex_neighbours[lgmask] + 1


        smoothMat = PETSc.Mat().create(comm=self.dm.comm)
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
        """
        Local area smoothing using radial-basis function smoothing kernel

        Parameters
        ----------
        data : array of floats, shape (n,)
            field variable to be smoothed
        its : int
            number of iterations (default: 1)
        centre_weight : float
            weight to apply to centre nodes (default: 0.75)
            other nodes are weighted by (1 - `centre_weight`)

        Returns
        -------
        sm : array of floats, shape (n,)
            smoothed field variable
        """

        smooth_data = data.copy()
        smooth_data_old = data.copy()

        for i in range(0, its):
            smooth_data_old[:] = smooth_data
            smooth_data = centre_weight*smooth_data_old + \
                          (1.0 - centre_weight)*self.rbf_smoother(smooth_data)
            smooth_data[:] = self.sync(smooth_data)

        return smooth_data


    def get_boundary(self):
        """
        Get boundary information on the mesh

        Returns
        -------
        bmask : array of bools, shape (n,)
            mask out the boundary nodes. Interior nodes are True;
            Boundary nodes are False.
        """
        bmask = np.ones(self.npoints, dtype=bool)

        for key in self.bc:
            i, wall = self.bc[key]
            mask = self.coords[:,i] == wall
            bmask[mask] = False

        return bmask


    def _construct_rbf_weights(self, delta=None):

        if delta == None:
            delta = self.neighbour_cloud_distances[:, 1].mean()

        self.delta  = delta
        self.gaussian_dist_w = self._rbf_weights(delta)

        return

    def _rbf_weights(self, delta=None):

        neighbour_cloud_distances = self.neighbour_cloud_distances

        if delta == None:
            delta = self.neighbour_cloud_distances[:, 1].mean()

        # Initialise the interpolants

        gaussian_dist_w       = np.zeros_like(neighbour_cloud_distances)
        gaussian_dist_w[:,:]  = np.exp(-np.power(neighbour_cloud_distances[:,:]/delta, 2.0))
        gaussian_dist_w[:,:] /= gaussian_dist_w.sum(axis=1).reshape(-1,1)

        return gaussian_dist_w

    def rbf_smoother(self, vector, iterations=1, delta=None):
        """
        Smoothing using a radial-basis function smoothing kernel

        Arguments
        ---------
        vector : array of floats, shape (n,)
            field variable to be smoothed
        iterations : int
            number of iterations to smooth vector
        delta : float / array of floats shape (n,)
            distance weights to apply the Gaussian interpolants

        Returns
        -------
        smooth_vec : array of floats, shape (n,)
            smoothed version of input vector
        """


        if type(delta) != type(None):
            self._construct_rbf_weights(delta)

        vector = self.sync(vector)

        for i in range(0, iterations):
            # print self.dm.comm.rank, ": RBF ",vector.max(), vector.min()

            vector_smoothed = (vector[self.neighbour_cloud[:,:]] * self.gaussian_dist_w[:,:]).sum(axis=1)
            vector = self.sync(vector_smoothed)

        return vector