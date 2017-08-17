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


class TriMesh(object):
    """
    Creating a global vector from a distributed DM removes duplicate entries (shadow zones)
    We recommend having 1) triangle or 2) scipy installed for Delaunay triangulations.
    """
    def __init__(self, dm, verbose=True,  *args, **kwargs):
        import stripy
        from scipy.spatial import cKDTree as _cKDTree

        self.timings = dict() # store times

        self.log = PETSc.Log()
        self.log.begin()

        self.verbose = verbose

        self.dm = dm
        self.gvec = dm.createGlobalVector()
        self.lvec = dm.createLocalVector()
        self.sect = dm.getDefaultSection()
        self.sizes = self.gvec.getSizes(), self.gvec.getSizes()

        self.rank = self.dm.comm.rank

        lgmap_r = dm.getLGMap()
        l2g = lgmap_r.indices.copy()
        offproc = l2g < 0

        l2g[offproc] = -(l2g[offproc] + 1)
        lgmap_c = PETSc.LGMap().create(l2g, comm=comm)

        self.lgmap_row = lgmap_r
        self.lgmap_col = lgmap_c

        # Delaunay triangulation
        t = clock()
        coords = dm.getCoordinatesLocal().array.reshape(-1,2)

        minX, minY = coords.min(axis=0)
        maxX, maxY = coords.max(axis=0)
        length_scale = np.sqrt((maxX - minX)*(maxY - minY)/coords.shape[0])
        coords += np.random.random(coords.shape) * 0.0001 * length_scale # This should be aware of the point spacing (small perturbation)

        self.tri = stripy.Triangulation(coords[:,0], coords[:,1])
        self.npoints = self.tri.npoints
        self.timings['triangulation'] = [clock()-t, self.log.getCPUTime(), self.log.getFlops()]
        if self.verbose:
            print("{} - Delaunay triangulation {}s".format(self.dm.comm.rank, clock()-t))


        # Calculate weigths and pointwise area
        t = clock()
        self.calculate_area_weights()
        self.timings['area weights'] = [clock()-t, self.log.getCPUTime(), self.log.getFlops()]
        if self.verbose:
            print("{} - Calculate node weights and area {}s".format(self.dm.comm.rank, clock()-t))


        # Find boundary points
        t = clock()
        self.bmask = self.get_boundary()
        self.timings['find boundaries'] = [clock()-t, self.log.getCPUTime(), self.log.getFlops()]
        if self.verbose:
            print("{} - Find boundaries {}s".format(self.dm.comm.rank, clock()-t))

        # cKDTree
        t = clock()
        self.cKDTree = _cKDTree(self.tri.points, balanced_tree=False)
        self.timings['cKDTree'] = [clock()-t, self.log.getCPUTime(), self.log.getFlops()]
        if self.verbose:
            print("{} - cKDTree {}s".format(self.dm.comm.rank, clock()-t))


        # Find neighbours
        t = clock()
        self.construct_neighbour_cloud()
        self.timings['construct neighbour cloud'] = [clock()-t, self.log.getCPUTime(), self.log.getFlops()]
        if self.verbose:
            print("{} - Construct neighbour cloud array {}s".format(self.dm.comm.rank, clock()-t))


        # sync smoothing operator
        t = clock()
        self._construct_rbf_weights()
        self.timings['construct rbf weights'] = [clock()-t, self.log.getCPUTime(), self.log.getFlops()]
        if self.verbose:
            print("{} - Construct rbf weights {}s".format(self.dm.comm.rank, clock()-t))



        self.root = False
        self.coords = self.tri.points


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
        return self.tri.x, self.tri.y, self.tri.simplices, self.bmask


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

        labelIS = self.dm.getStratumIS(label, 1)
        pt_range = np.logical_and(labelIS.indices >= pStart, labelIS.indices < pEnd)
        indices = labelIS.indices[pt_range] - pStart
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


    def calculate_area_weights(self):
        """
        Calculate weigths and pointwise area
        """

        from quagmire._fortran import ntriw

        self.area, self.weight = ntriw(self.tri.x, self.tri.y, self.tri.simplices.T+1)


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
        return self.tri.gradient(PHI, nit, tol)


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

        t = clock()

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


        print(" - Array sort {}s".format(clock()-t))


    def construct_neighbour_cloud(self, size=25):
        """
        Find neighbours from distance cKDTree.

        """

        nndist, nncloud = self.cKDTree.query(self.tri.points, k=size)

        self.neighbour_cloud = nncloud
        self.neighbour_cloud_distances = nndist

        unique, neighbours = np.unique(self.tri.simplices.ravel(), return_counts=True)
        self.near_neighbours = neighbours + 2
        self.extended_neighbours = np.empty_like(neighbours).fill(size)

        self.near_neighbour_mask = np.zeros_like(self.neighbour_cloud, dtype=np.bool)

        for node in range(0,self.npoints):
            self.near_neighbour_mask[node, 0:self.near_neighbours[node]] = True

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

        return self.lvec.array.copy()


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
            print("Warning! No boundary information in DMPlex.\n\
                   Continuing with convex hull.")
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
            if key in kwdict.keys():
                raise ValueError("Cannot use un-named variables\
                                  and keyword: {}".format(key))
            kwdict[key] = arg

        vec = self.gvec.duplicate()

        for key in kwdict:
            val = kwdict[key]
            try:
                vec.setArray(val)
            except:
                self.lvec.setArray(val)
                self.dm.localToGlobal(self.lvec, vec)

            vec.setName(key)

            ViewHDF5 = PETSc.Viewer()
            ViewHDF5.createHDF5(file, mode='a')
            ViewHDF5.view(obj=vec)
            ViewHDF5.destroy()

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
        Replaces shadow values in the local domain (non additive)
        """

        self.lvec.setArray(vector)

        # self.dm.localToLocal(self.lvec, self.gvec)
        self.dm.localToGlobal(self.lvec, self.gvec)
        self.dm.globalToLocal(self.gvec, self.lvec)

        return self.lvec.array.copy()


    def _construct_rbf_weights(self, delta=None):

        self.delta  = delta

        # delta_x = self.tri.x[self.neighbour_cloud] - self.tri.x.reshape(-1,1)
        # delta_y = self.tri.y[self.neighbour_cloud] - self.tri.y.reshape(-1,1)
        #
        # neighbour_cloud_distances = np.hypot(delta_x, delta_y)

        neighbour_cloud_distances = self.neighbour_cloud_distances

        if self.delta == None:
            self.delta = self.neighbour_cloud_distances[:, 1].mean()

        # Initialise the interpolants

        gaussian_dist_w       = np.zeros_like(neighbour_cloud_distances)
        gaussian_dist_w[:,:]  = np.exp(-np.power(neighbour_cloud_distances[:,:]/self.delta, 2.0))
        gaussian_dist_w[:,:] /= gaussian_dist_w.sum(axis=1).reshape(-1,1)

        # gaussian_dist_w[self.extended_neighbours_mask] = 0.0

        self.gaussian_dist_w = gaussian_dist_w

        return



    def rbf_smoother(self, vector, iterations=1):

         # Should do some error checking here to ensure the field and point cloud are compatible

        #  lvec  = self.lvec.copy()
        #  gvec  = self.gvec.copy()

         vector = self.sync(vector)

         for i in range(0, iterations):
             # print self.dm.comm.rank, ": RBF ",vector.max(), vector.min()

             vector_smoothed = (vector[self.neighbour_cloud[:,:]] * self.gaussian_dist_w[:,:]).sum(axis=1)
             vector = self.sync(vector_smoothed)

         return vector
