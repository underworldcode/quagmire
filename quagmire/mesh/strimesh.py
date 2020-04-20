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
For unstructured data on the sphere.
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

from quagmire import function as fn
from quagmire.function import LazyEvaluation as _LazyEvaluation


class sTriMesh(_CommonMesh):
    """
    Build spatial data structures on an __unstructured spherical mesh__.

    Use `sTriMesh` for:

    - calculating spatial derivatives
    - identifying node neighbour relationships
    - interpolation / extrapolation
    - smoothing operators
    - importing and saving mesh information

    Each of these data structures are built on top of a `PETSc DM` object
    (created from `quagmire.tools.meshtools`).

    Parameters
    ----------
    DM : PETSc DMPlex object
        Build this unstructured spherical mesh object using one of:

        - `quagmire.tools.meshtools.create_spherical_DMPlex`
        - `quagmire.tools.meshtools.create_DMPlex_from_spherical_points`
    verbose : bool
        Flag toggles verbose output
    *args : optional arguments
    **kwargs : optional keyword arguments
    """

    __count = 0

    @classmethod
    def _count(cls):
        sTriMesh.__count += 1
        return sTriMesh.__count

    @property
    def id(self):
        return self.__id

    def __init__(self, dm, verbose=True, *args, **kwargs):
        import stripy
        from scipy.spatial import cKDTree as _cKDTree

        # initialise base mesh class
        super(sTriMesh, self).__init__(dm, verbose)

        self.__id = "strimesh_{}".format(self._count())


        # Delaunay triangulation
        t = perf_counter()
        coords = dm.getCoordinatesLocal().array.reshape(-1,3)
        minX, minY, minZ = coords.min(axis=0)
        maxX, maxY, maxZ = coords.max(axis=0)
        length_scale = np.sqrt((maxX - minX)*(maxY - minY)/coords.shape[0])

        # coords += np.random.random(coords.shape) * 0.0001 * length_scale # This should be aware of the point spacing (small perturbation)

        # r = np.sqrt(coords[:,0]**2 + coords[:,1]**2 + coords[:,2]**2) # should just equal 1
        # r = 1.0
        # lons = np.arctan2(coords[:,1], coords[:,0])
        # lats = np.arcsin(coords[:,2]/r)
        lons, lats = stripy.spherical.xyz2lonlat(coords[:,0], coords[:,1], coords[:,2])

        self.tri = stripy.sTriangulation(lons, lats)
        self.npoints = self.tri.npoints
        self.timings['triangulation'] = [perf_counter()-t, self.log.getCPUTime(), self.log.getFlops()]
        if self.rank == 0 and self.verbose:
            print("{} - Delaunay triangulation {}s".format(self.dm.comm.rank, perf_counter()-t))


        # Calculate weigths and pointwise area
        t = perf_counter()
        self.area, self.weight = self.calculate_area_weights()
        self.pointwise_area = self.add_variable(name="area")
        self.pointwise_area.data = self.area
        self.pointwise_area.lock()

        self.timings['area weights'] = [perf_counter()-t, self.log.getCPUTime(), self.log.getFlops()]
        if self.rank == 0 and self.verbose:
            print("{} - Calculate node weights and area {}s".format(self.dm.comm.rank, perf_counter()-t))


        # Find boundary points
        t = perf_counter()
        self.bmask = self.get_boundary()
        self.mask = self.add_variable(name="Mask")
        self.mask.data = self.bmask.astype(PETSc.ScalarType)
        self.mask.lock()

        self.timings['find boundaries'] = [perf_counter()-t, self.log.getCPUTime(), self.log.getFlops()]
        if self.rank == 0 and self.verbose:
            print("{} - Find boundaries {}s".format(self.dm.comm.rank, perf_counter()-t))

        # cKDTree
        t = perf_counter()
        self.cKDTree = _cKDTree(self.tri.points, balanced_tree=False)
        self.timings['cKDTree'] = [perf_counter()-t, self.log.getCPUTime(), self.log.getFlops()]
        if self.rank == 0 and self.verbose:
            print("{} - cKDTree {}s".format(self.dm.comm.rank, perf_counter()-t))


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
        self.coords = self.tri.points
        self.data = self.coords
        self.interpolate = self.tri.interpolate


    def get_local_mesh(self):
        """
        Retrieves the local mesh information

        Returns
        -------
        x : array of floats, shape (n,)
            x coordinates
        y : array of floats, shape (n,)
            y coordinates
        simplices : array of ints, shape (ntri, 3)
            simplices of the triangulation
        bmask  : array of bools, shape (n,2)
        """
        return self.tri.lons, self.tri.lats, self.tri.simplices, self.bmask


    def calculate_area_weights(self, r1=6384.4e3, r2=6352.8e3):
        """
        Calculate pointwise weights and area

        Parameters
        ----------
        r1 : float
            radius of the sphere at the equator
        r2 : float
            radius of the sphere at the poles

        Returns
        -------
        area : array of floats, shape (n,)
            point-wise area
        weights : array of ints, shape(n,)
            weighting for each point

        Notes
        -----
        This calls a fortran 90 routine which computes the weight and area
        for each point in the mesh.

        The area is calculated using a Lambert azimuthal equal-area projection
        https://en.wikipedia.org/wiki/Lambert_azimuthal_equal-area_projection
        where the geocentric radius uses the radius of the sphere at the
        equator and the poles (defaults to Earth values: 6384.4km and 6352.8km)
        """

        from quagmire._fortran import ntriw_s

        area, weight = ntriw_s(self.tri.lons, self.tri.lats, self.tri.simplices.T+1, r1, r2)
        
        return area, weight


    def node_neighbours(self, point):
        """
        Returns a list of neighbour nodes for a given point in the delaunay triangulation
        """

        return self.vertex_neighbour_vertices[1][self.vertex_neighbour_vertices[0][point]:self.vertex_neighbour_vertices[0][point+1]]


    def derivative_grad(self, PHI, nit=10, tol=1e-8):
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
        PHIx, PHIy, PHIz = self.tri.gradient(PHI, nit, tol)
        return self.tri.transform_to_spherical(PHIx, PHIy, PHIz)


    def derivative_div(self, PHIx, PHIy, **kwargs):
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
        u_xx, u_xy = self.derivative_grad(PHIx, **kwargs)
        u_yx, u_yy = self.derivative_grad(PHIy, **kwargs)

        return u_xx + u_yy


    def get_edge_lengths(self):
        """
        Find all edges in a triangluation and their lengths
        """
        points = self.tri.points

        i1 = np.sort([self.tri.simplices[:,0], self.tri.simplices[:,1]], axis=0)
        i2 = np.sort([self.tri.simplices[:,0], self.tri.simplices[:,2]], axis=0)
        i3 = np.sort([self.tri.simplices[:,1], self.tri.simplices[:,2]], axis=0)

        a = np.hstack([i1, i2, i3]).T

        # find unique rows in numpy array
        # <http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array>
        b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
        edges = np.unique(b).view(a.dtype).reshape(-1, a.shape[1])

        edge_lengths = np.linalg.norm(points[edges[:,0]] - points[edges[:,1]], axis=1)

        self.edges = edges
        self.edge_lengths = edge_lengths


    def construct_neighbours(self):
        """
        Find neighbours from edges and store as CSR coordinates.

        This allows you to directly ask the neighbours for a given node a la Qhull,
        or efficiently construct a sparse matrix (PETSc/SciPy)
        """

        row = np.hstack([self.edges[:,0], self.edges[:,1]])
        col = np.hstack([self.edges[:,1], self.edges[:,0]])
        val = np.hstack([self.edge_lengths, self.edge_lengths])

        # sort by row
        sort = row.argsort()
        row = row[sort].astype(PETSc.IntType)
        col = col[sort].astype(PETSc.IntType)
        val = val[sort]

        nnz = np.bincount(row) # number of nonzeros
        indptr = np.insert(np.cumsum(nnz),0,0)

        self.vertex_neighbours = nnz.astype(PETSc.IntType)
        self.vertex_neighbour_vertices = indptr, col
        self.vertex_neighbour_distance = val

        # We may not need this, but constuct anyway for now!
        neighbours = [[]]*self.npoints
        closed_neighbours = [[]]*self.npoints

        for i in range(indptr.size-1):
            start, end = indptr[i], indptr[i+1]
            neighbours[i] = np.array(col[start:end])
            closed_neighbours[i] = np.hstack([i, neighbours[i]])

        self.neighbour_list = np.array(neighbours)
        self.neighbour_array = np.array(closed_neighbours)


    def construct_extended_neighbour_cloud(self):
        """
        Find extended node neighbours
        """
        from quagmire._fortran import ncloud

        # nnz_max = np.bincount(self.tri.simplices.ravel()).max()

        unique, neighbours = np.unique(self.tri.simplices.ravel(), return_counts=True)
        self.near_neighbours = neighbours


        nnz_max = self.near_neighbours.max()

        cloud = ncloud(self.tri.simplices.T + 1, self.npoints, nnz_max)
        cloud -= 1 # convert to C numbering
        cloud_mask = cloud==-1
        cloud_masked = np.ma.array(cloud, mask=cloud_mask)

        self.extended_neighbours = np.count_nonzero(~cloud_mask, axis=1)
        self.extended_neighbours_mask = cloud_mask

        dx = self.tri.points[cloud_masked,0] - self.tri.points[:,0].reshape(-1,1)
        dy = self.tri.points[cloud_masked,1] - self.tri.points[:,1].reshape(-1,1)
        dist = np.hypot(dx,dy)
        dist[cloud_mask] = 1.0e50

        ii =  np.argsort( dist, axis=1)

        t = perf_counter()

        ## Surely there is some np.argsort trick here to avoid the for loop ???

        neighbour_cloud = np.ones_like(cloud, dtype=np.int )
        neighbour_cloud_distances = np.empty_like(dist)

        for node in range(0, self.npoints):
            neighbour_cloud[node, :] = cloud[node, ii[node,:]]
            neighbour_cloud_distances[node, :] = dist[node, ii[node,:]]

        # The same mask should be applicable to the sorted array

        self.neighbour_cloud = np.ma.array( neighbour_cloud, mask = cloud_mask)
        self.neighbour_cloud_distances = np.ma.array( neighbour_cloud_distances, mask = cloud_mask)

        # Create a mask that can pick the natural neighbours only

        ind = np.indices(self.neighbour_cloud.shape)[1]
        mask = ind > self.near_neighbours.reshape(-1,1)
        self.near_neighbours_mask = mask


        print(" - Array sort {}s".format(perf_counter()-t))


    def construct_neighbour_cloud(self, size=25):
        """
        Find neighbours from distance cKDTree.

        """
        from quagmire import _fortran

        nndist, nncloud = self.cKDTree.query(self.tri.points, k=size)

        self.neighbour_cloud = nncloud
        self.neighbour_cloud_distances = nndist

        neighbours = np.bincount(self.tri.simplices.flat)
        self.near_neighbours = neighbours + 2
        self.extended_neighbours = np.full_like(neighbours, size)

        near_neighbour_mask = _fortran.fill_mask_to_idx(size, self.near_neighbours)
        self.near_neighbour_mask = near_neighbour_mask.astype(bool)

        return


    def _build_smoothing_matrix(self):

        indptr, indices = self.vertex_neighbour_vertices
        weight  = 1.0/self.weight
        nweight = weight[indices]

        lgmask = self.lgmap_row.indices >= 0


        nnz = self.vertex_neighbours[lgmask] + 1

        # smoothMat = self.dm.createMatrix()
        # smoothMat.setOption(smoothMat.Option.NEW_NONZERO_LOCATIONS, False)
        smoothMat = PETSc.Mat().create(comm=comm)
        smoothMat.setType('aij')
        smoothMat.setSizes(self.sizes)
        smoothMat.setLGMap(self.lgmap_row, self.lgmap_col)
        smoothMat.setFromOptions()
        smoothMat.setPreallocationNNZ(nnz)

        # read in data
        smoothMat.setValuesLocalCSR(indptr.astype(PETSc.IntType), indices.astype(PETSc.IntType), nweight)
        self.lvec.setArray(weight)
        self.dm.localToGlobal(self.lvec, self.gvec)
        smoothMat.setDiagonal(self.gvec)

        smoothMat.assemblyBegin()
        smoothMat.assemblyEnd()

        self.localSmoothMat = smoothMat


    def local_area_smoothing(self, data, its=1, centre_weight=0.75):

        smooth_data = data.copy()
        smooth_data_old = data.copy()

        for i in range(0, its):
            smooth_data_old[:] = smooth_data
            smooth_data = centre_weight*smooth_data_old + \
                          (1.0 - centre_weight)*self.rbf_smoother(smooth_data)
            smooth_data[:] = self.sync(smooth_data)

        return smooth_data


    def local_area_smoothing_old(self, data, its=1, centre_weight=0.75):

        self.lvec.setArray(data)
        self.dm.localToGlobal(self.lvec, self.gvec)
        smooth_data = self.gvec.copy()

        for i in range(0, its):
            self.localSmoothMat.mult(smooth_data, self.gvec)
            smooth_data = centre_weight*smooth_data + (1.0 - centre_weight)*self.gvec

        self.dm.globalToLocal(smooth_data, self.lvec)

        return self.lvec.array


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