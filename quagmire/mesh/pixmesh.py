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
        from scipy.interpolate import RegularGridInterpolator

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

        t1 = perf_counter()
        self.construct_natural_neighbour_cloud()
        self.timings['construct natural neighbour cloud'] = [perf_counter()-t1, self.log.getCPUTime(), self.log.getFlops()]
 
        if self.rank==0 and self.verbose:
            print("{} - Construct neighbour cloud arrays {}s, ({}s + {}s)".format(self.dm.comm.rank,  perf_counter()-t,
                                                                                self.timings['construct neighbour cloud'][0],
                                                                                self.timings['construct natural neighbour cloud'][0]  ))


        # RBF smoothing operator
        t = perf_counter()
        self._construct_rbf_weights()
        self.timings['construct rbf weights'] = [perf_counter()-t, self.log.getCPUTime(), self.log.getFlops()]
        if self.rank == 0 and self.verbose:
            print("{} - Construct rbf weights {}s".format(self.dm.comm.rank, perf_counter()-t))


        # construct interpolator object
        t = perf_counter()
        xcoords = np.linspace(minX, maxX, nx)
        ycoords = np.linspace(minY, maxY, ny)
        dvalues = np.empty((ny,nx))
        self._interpolator = RegularGridInterpolator((ycoords, xcoords), dvalues, bounds_error=False, fill_value=None)
        self.timings['construct interpolator object'] = [perf_counter()-t, self.log.getCPUTime(), self.log.getFlops()]
        if self.rank == 0 and self.verbose:
            print("{} - Construct interpolator object {}s".format(self.dm.comm.rank, perf_counter()-t))


        self.root = False

        # functions / parameters that are required for compatibility among FlatMesh types
        self._derivative_grad_cartesian = self.derivative_grad
        self._radius = 1.0


    def interpolate(self, xi, yi, zdata, order=1):

        order_dict = {1: "linear", 0: "nearest"}

        if order not in order_dict:
            raise ValueError("order must be set to 0 (nearest) or 1 (linear) interpolation")

        xi = np.array(xi)
        yi = np.array(yi)

        interpolator = self._interpolator
        interpolator.values = np.array(zdata).reshape(self.ny, self.nx)
        ival = interpolator((yi, xi), method=order_dict[order])

        # check if inside bounds
        inside_bounds = np.zeros_like(xi, dtype=np.bool)
        inside_bounds += xi < self.minX
        inside_bounds += xi > self.maxX
        inside_bounds += yi < self.minY
        inside_bounds += yi > self.maxY

        return ival, inside_bounds.astype(np.int)


    def derivative_grad(self, PHI, **kwargs):
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


    def construct_natural_neighbour_cloud(self):
        """
        Find the natural neighbours for each node in the triangulation and store in a
        numpy array for efficient lookup.
        """

        natural_neighbours = np.empty((self.npoints, 9), dtype=np.int)
        nodes = np.arange(0, self.npoints, dtype=np.int).reshape(self.ny,self.nx)
        index = np.pad(nodes, (1,1), constant_values=-1)


        natural_neighbours[:,0] = index[1:-1,1:-1].flat  # self
        natural_neighbours[:,1] = index[2:,  1:-1].flat  # right
        natural_neighbours[:,2] = index[1:-1, :-2].flat  # bottom
        natural_neighbours[:,3] = index[0:-2,1:-1].flat  # left
        natural_neighbours[:,4] = index[1:-1,2:  ].flat  # top
        natural_neighbours[:,5] = index[2:  ,2:  ].flat  # top right
        natural_neighbours[:,6] = index[2:  , :-2].flat  # bottom right
        natural_neighbours[:,7] = index[ :-2, :-2].flat  # bottom left
        natural_neighbours[:,8] = index[ :-2,2:  ].flat  # top left

        # shuffle -1 entries to the end
        natural_neighbour_mask = natural_neighbours != -1
        natural_neighbours_count = np.count_nonzero(natural_neighbour_mask, axis=1)

        for i in range(0, self.npoints):
            nnc = natural_neighbours_count[i]
            nnm = natural_neighbour_mask[i]

            natural_neighbours[i,:nnc], natural_neighbours[i,nnc:] = natural_neighbours[i,nnm], natural_neighbours[i,~nnm]


        self.natural_neighbours       = natural_neighbours
        self.natural_neighbours_count = natural_neighbours_count
        self.natural_neighbours_mask  = natural_neighbours != -1


    def construct_neighbour_cloud(self, size=25):
        """
        Find neighbours from distance cKDTree.

        Parameters
        ----------
        size : int
            Number of neighbours to search for

        Notes
        -----
        Use this method to search for neighbours that are not
        necessarily immediate node neighbours (i.e. neighbours
        connected by a line segment). Extended node neighbours
        should be captured by the search depending on how large
        `size` is set to.
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

        # self.near_neighbour_mask = np.zeros_like(self.neighbour_cloud, dtype=np.bool)

        # for node in range(0,self.npoints):
        #     self.near_neighbour_mask[node, 0:self.near_neighbours[node]] = True

        return


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


    def build_rbf_smoother(self, delta, iterations=1):

        pixmesh_self = self

        class _rbf_smoother_object(object):

            def __init__(inner_self, delta, iterations=1):

                if delta == None:
                    pixmesh_self.lvec.setArray(pixmesh_self.neighbour_cloud_distances[:, 1])
                    pixmesh_self.dm.localToGlobal(pixmesh_self.lvec, pixmesh_self.gvec)
                    delta = pixmesh_self.gvec.sum() / pixmesh_self.gvec.getSize()

                inner_self._mesh = pixmesh_self
                inner_self.delta = delta
                inner_self.iterations = iterations
                inner_self.gaussian_dist_w = inner_self._mesh._rbf_weights(delta)

                return


            def _apply_rbf_on_my_mesh(inner_self, lazyFn, iterations=1):
                import quagmire

                smooth_node_values = lazyFn.evaluate(inner_self._mesh)

                for i in range(0, iterations):
                    smooth_node_values = (smooth_node_values[inner_self._mesh.neighbour_cloud[:,:]] * inner_self.gaussian_dist_w[:,:]).sum(axis=1)
                    smooth_node_values = inner_self._mesh.sync(smooth_node_values)

                return smooth_node_values


            def smooth_fn(inner_self, lazyFn, iterations=None):

                if iterations is None:
                        iterations = inner_self.iterations

                def smoother_fn(*args, **kwargs):
                    import quagmire

                    smooth_node_values = inner_self._apply_rbf_on_my_mesh(lazyFn, iterations=iterations)

                    if len(args) == 1 and args[0] == lazyFn._mesh:
                        return smooth_node_values
                    elif len(args) == 1 and quagmire.mesh.check_object_is_a_q_mesh(args[0]):
                        mesh = args[0]
                        return inner_self._mesh.interpolate(lazyFn._mesh.coords[:,0], lazyFn._mesh.coords[:,1], zdata=smooth_node_values, **kwargs)
                    else:
                        xi = np.atleast_1d(args[0])
                        yi = np.atleast_1d(args[1])
                        i, e = inner_self._mesh.interpolate(xi, yi, zdata=smooth_node_values, **kwargs)
                        return i


                newLazyFn = _LazyEvaluation(mesh=lazyFn._mesh)
                newLazyFn.evaluate = smoother_fn
                newLazyFn.description = "RBFsmooth({}, d={}, i={})".format(lazyFn.description, inner_self.delta, iterations)

                return newLazyFn

        return _rbf_smoother_object(delta, iterations)