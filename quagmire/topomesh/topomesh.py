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

import numpy as np
import numpy.ma as ma
from mpi4py import MPI
import sys,petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
# comm = MPI.COMM_WORLD
from time import perf_counter

try: range = xrange
except: pass

from quagmire import function as fn
from quagmire.mesh import MeshVariable as _MeshVariable
from quagmire.function import LazyEvaluation as _LazyEvaluation

class TopoMesh(object):
    def __init__(self, downhill_neighbours=2, *args, **kwargs):

        # Initialise cumulative flow vectors
        self._DX0 = self.gvec.duplicate()
        self._DX1 = self.gvec.duplicate()
        self._dDX = self.gvec.duplicate()

        self.downhillMat = None

        # There is no topography yet set, so we need to avoid
        # triggering the matrix rebuilding in the setter of this property
        self._downhill_neighbours = downhill_neighbours
        self._topography_modified_count = 0

        # create a variable for the height (data) and fn_height
        # a context manager to ensure the matrices are updated when h(x,y) changes

        self.topography        = self.add_variable(name="h(x,y)", locked=True)
        self.upstream_area     = self.add_variable(name="A(x,y)", locked=True)
        self.deform_topography = self._height_update_context_manager_generator()

        ## TODO: Should be able to fix upstream area as a "partial" function of upstream area
        ## with a parameter value of 1.0 in the argument (or 1.0 above a ref height)

        # Slope (function)
        self.slope = fn.math.slope(self.topography)

        self._heightVariable = self.topography



    @property
    def downhill_neighbours(self):
        return self._downhill_neighbours

    @downhill_neighbours.setter
    def downhill_neighbours(self, value):
        self._downhill_neighbours = int(value)
        self.deform_topography = self._height_update_context_manager_generator()

        # if the topography is unlocked, don't rebuild anything
        # as it might break something

        if self.topography._locked == True:
            self._update_height()
            self._update_height_for_surface_flows()

        return


    def _update_height(self):
        """
        Update height field
        """

        t = perf_counter()
        self._build_downhill_matrix_iterate()
        self.timings['downhill matrices'] = [perf_counter()-t, self.log.getCPUTime(), self.log.getFlops()]

        if self.rank==0 and self.verbose:
            print(("{} - Build downhill matrices {}s".format(self.dm.comm.rank, perf_counter()-t)))
        return

    def _update_height_for_surface_flows(self):

        #self.rainfall_pattern = rainfall_pattern.copy()
        #self.sediment_distribution = sediment_distribution.copy()

        t = perf_counter()

        self.upstream_area.unlock()
        self.upstream_area.data = self.cumulative_flow(self.pointwise_area.data)  ## Should be upstream integral of 1.0
        self.upstream_area.lock()

        self.timings['Upstream area'] = [perf_counter()-t, self.log.getCPUTime(), self.log.getFlops()]

        if self.verbose:
            print(("{} - Build upstream areas {}s".format(self.dm.comm.rank, perf_counter()-t)))

        # Find low points
        self.low_points = self.identify_low_points()

        # Find high points
        self.outflow_points = self.identify_outflow_points()

    def _height_update_context_manager_generator(self):
        """Builds a context manager on the current mesh object to control when matrices are to be updated"""

        topomesh = self
        topographyVariable = self.topography
        downhill_neighbours = self.downhill_neighbours

        class Topomesh_Height_Update_Manager(object):
            """Manage when changes to the height information trigger a rebuild
            of the topomesh matrices and other internal data.
            """

            def __init__(inner_self, downhill_neighbours=downhill_neighbours):
                inner_self._topomesh = topomesh
                inner_self._topovar  = topographyVariable
                inner_self._downhill_neighbours = int(downhill_neighbours)
                return

            def __enter__(inner_self):
                # unlock
                inner_self._topovar.unlock()
                inner_self._topomesh._downhill_neighbours = inner_self._downhill_neighbours
                return

            def __exit__(inner_self, *args):
                inner_self._topomesh._update_height()
                inner_self._topomesh._update_height_for_surface_flows()
                inner_self._topovar.lock()
                inner_self._topomesh._topography_modified_count += 1
                return

        return Topomesh_Height_Update_Manager

        self._build_adjacency_matrix_iterate()
        if self.rank==0 and self.verbose:
            print((" - Partial rebuild of downhill matrices {}s".format(perf_counter()-t)))

        # revert to specified n-neighbours
        self.downhill_neighbours = neighbours

        return

    def _sort_nodes_by_field(self, height):

        # Sort neighbours by gradient
        indptr, indices = self.vertex_neighbour_vertices
        # gradH = height[indices]/self.vertex_neighbour_distance

        self.node_high_to_low = np.argsort(height)[::-1]

        neighbour_array_lo_hi = self.neighbour_array.copy()
        neighbour_array_2_low = np.empty((self.npoints, 2), dtype=PETSc.IntType)

        for i in range(indptr.size-1):
            # start, end = indptr[i], indptr[i+1]
            # neighbours = np.hstack([i, indices[start:end]])
            # order = height[neighbours].argsort()
            # neighbour_array_lo_hi[i] = neighbours[order]
            # neighbour_array_2_low[i] = neighbour_array_lo_hi[i][:2]

            neighbours = self.neighbour_array[i]
            order = height[neighbours].argsort()
            neighbour_array_lo_hi[i] = neighbours[order]
            neighbour_array_2_low[i] = neighbour_array_lo_hi[i][:2]

        self.neighbour_array_lo_hi = neighbour_array_lo_hi
        self.neighbour_array_2_low = neighbour_array_2_low



    def _adjacency_matrix_template(self, nnz=(1,1)):

        matrix = PETSc.Mat().create(comm=self.dm.comm)
        matrix.setType('aij')
        matrix.setSizes(self.sizes)
        matrix.setLGMap(self.lgmap_row, self.lgmap_col)
        matrix.setFromOptions()
        matrix.setPreallocationNNZ(nnz)

        return matrix


## This is the lowest near node / lowest extended neighbour

# hnear = np.ma.array(mesh.height[mesh.neighbour_cloud], mask=mesh.near_neighbours_mask)
# low_neighbours = np.argmin(hnear, axis=1)
#
# hnear = np.ma.array(mesh.height[mesh.neighbour_cloud], mask=mesh.extended_neighbours_mask)
# low_eneighbours = np.argmin(hnear, axis=1)
#
#

    def _build_down_neighbour_arrays(self, nearest=True):

        nodes = list(range(0,self.npoints))
        # nheight  = self.height[self.neighbour_cloud]
        nheight  = self._heightVariable.data[self.neighbour_cloud]

        nheightidx = np.argsort(nheight, axis=1)

        nheightn = nheight.copy()
        # nheightn[~self.near_neighbour_mask] += self.height.max()
        nheightn[~self.near_neighbour_mask] += self._heightVariable.data.max()
        nheightnidx = np.argsort(nheightn, axis=1)

        ## How many low neighbours are there in each ?

        idxrange  = np.where(nheightidx==0)[1]
        idxnrange = np.where(nheightnidx==0)[1]

        ## First the STD, 1-neighbour

        idx  = nheightidx[:,0]
        idxn = nheightnidx[:,0]

        # Pick either extended or standard ...
        use_extended = np.where(idxnrange == 0)

        index1 = self.neighbour_cloud[nodes, idxn[nodes]]

        if not nearest:
            index1[use_extended] = self.neighbour_cloud[use_extended, idx[use_extended]]

        # store in neighbour dictionary
        self.down_neighbour = dict()
        self.down_neighbour[1] = index1.astype(PETSc.IntType)


        ## Now all higher neighours

        for i in range(1, self.downhill_neighbours):
            n = i + 1

            idx  = nheightidx[:,i]
            idxn = nheightnidx[:,i]

            indexN = self.neighbour_cloud[nodes, idxn[nodes]]

            if not nearest:
                use_extended = np.where(idxnrange < n)
                indexN[use_extended] = self.neighbour_cloud[use_extended, idx[use_extended]]

                failed = np.where(idxrange < n)
                indexN[failed] = index1[failed]
            else:
                failed = np.where(idxnrange < n)
                indexN[failed] = index1[failed]

            # store in neighbour dictionary
            self.down_neighbour[n] = indexN.astype(PETSc.IntType)


    def _build_adjacency_matrix_iterate(self):

        self._build_down_neighbour_arrays(nearest=False)

        self.adjacency = dict()
        self.uphill = dict()
        data = np.ones(self.npoints)

        indptr = np.arange(0, self.npoints+1, dtype=PETSc.IntType)
        nodes = indptr[:-1]

        for i in range(1, self.downhill_neighbours+1):

            data[self.down_neighbour[i]==nodes] = 0.0

            adjacency = self._adjacency_matrix_template()
            adjacency.assemblyBegin()
            adjacency.setValuesLocalCSR(indptr, self.down_neighbour[i], data)
            adjacency.assemblyEnd()

            self.uphill[i] = adjacency.copy()
            self.adjacency[i] = adjacency.transpose()

            # self.down_neighbour[i] = down_neighbour.copy()

    def _build_downhill_matrix_iterate(self):

        self._build_adjacency_matrix_iterate()
        weights = np.empty((self.downhill_neighbours, self.npoints))

        height = self._heightVariable.data

        # Process weights
        for i in range(0, self.downhill_neighbours):
            down_N = self.down_neighbour[i+1]
            grad = np.abs(height - height[down_N]+1.0e-10) / (1.0e-10 + \
                   np.hypot(self.coords[:,0] - self.coords[down_N,0],
                            self.coords[:,1] - self.coords[down_N,1] ))

            weights[i,:] = np.sqrt(grad)

        weights /= weights.sum(axis=0)
        w = self.gvec.duplicate()


        # Store weighted downhill matrices
        downhill_matrices = [None]*self.downhill_neighbours
        for i in range(0, self.downhill_neighbours):
            N = i + 1
            self.lvec.setArray(weights[i])
            self.dm.localToGlobal(self.lvec, w)

            D = self.adjacency[N].copy()
            D.diagonalScale(R=w)
            downhill_matrices[i] = D


        # Sum downhill matrices
        self.downhillMat = downhill_matrices[0]
        for i in range(1, self.downhill_neighbours):
            self.downhillMat += downhill_matrices[i]
            downhill_matrices[i].destroy()

        return


    def build_cumulative_downhill_matrix(self):
        """
        Build non-sparse, single hit matrices to cumulative_flow information downhill
        (self.sweepDownToOutflowMat and self.downhillCumulativeMat)

        This may be expensive in terms of storage so this is only done if
        self.storeDense == True and the matrices are also out of date (which they
        will be if the height field is changed)

        downhillCumulativeMat = I + D + D**2 + D**3 + ... D**N where N is the length of the graph

        """

        comm = self.dm.comm

        downSweepMat    = self.accumulatorMat.copy()
        downHillaccuMat = self.downhillMat.copy()
        accuM           = self.downhillMat.copy()   # work matrix

        DX0 = self._DX0
        DX1 = self._DX1
        dDX = self._dDX
        DX0.set(1.0)

        err = np.array([True])
        err_proc = np.ones(comm.size, dtype=bool)

        while err_proc.any():
            downSweepMat    = downSweepMat*self.accumulatorMat  # N applications of the accumulator
            accuM           = accuM*self.downhillMat
            downHillaccuMat = downHillaccuMat + accuM
            DX0 = self.downhillMat*DX1

            err[0] = np.any(DX0)
            comm.Allgather([err, MPI.BOOL], [err_proc, MPI.BOOL])

        # add identity matrix
        I = np.arange(0, self.npoints+1, dtype=PETSc.IntType)
        J = np.arange(0, self.npoints, dtype=PETSc.IntType)
        V = np.ones(self.npoints)
        identityMat = self._adjacency_matrix_template()
        identityMat.assemblyBegin()
        identityMat.setValuesLocalCSR(I, J, V)
        identityMat.assemblyEnd()

        downHillaccuMat += identityMat

        self.downhillCumulativeMat = downHillaccuMat
        self.sweepDownToOutflowMat = downSweepMat


    def upstream_integral_fn(self, lazyFn):
        """Upsream integral implemented as an area-weighted upstream summation"""

        import quagmire

        def integral_fn(*args, **kwargs):
            node_values = lazyFn.evaluate(lazyFn._mesh) * lazyFn._mesh.area
            node_integral = self.cumulative_flow(node_values)

            if len(args) == 1 and args[0] == lazyFn._mesh:
                return node_integral
            elif len(args) == 1 and quagmire.mesh.check_object_is_a_q_mesh_and_raise(args[0]):
                mesh = args[0]
                return lazyFn._mesh.interpolate(mesh.coords[:,0], mesh.coords[:,1], zdata=node_integral, **kwargs)
            else:
                xi = np.atleast_1d(args[0])
                yi = np.atleast_1d(args[1])
                i, e = lazyFn._mesh.interpolate(xi, yi, zdata=node_integral, **kwargs)
                return i

        newLazyFn = _LazyEvaluation(mesh=lazyFn._mesh)
        newLazyFn.evaluate = integral_fn
        newLazyFn.description = "UpInt({})dA".format(lazyFn.description)
        newLazyFn.dependency_list |= lazyFn.dependency_list

        return newLazyFn


    def cumulative_flow(self, vector, *args, **kwargs):


        niter, cumulative_flow_vector = self._cumulative_flow_verbose(vector, **kwargs)
        return cumulative_flow_vector


    def _cumulative_flow_verbose(self, vector, verbose=False, maximum_its=None, uphill=False):

        if not maximum_its:
            maximum_its = 1000000000000

        # This is what happens if the mesh topography has never been set
        if not self.downhillMat:
            print("No downhill matrix exists ")
            temp_vec = self.lvec.array.copy()
            temp_vec = 0.0
            return 0, temp_vec


        # downhillMat2 = self.downhillMat * self.downhillMat
        # downhillMat4 = downhillMat2 * downhillMat2
        # downhillMat8 = downhillMat4 * downhillMat4
        if uphill:
            downhillMat = self.downhillMat.copy()
            downhillMat = downhillMat.transpose()
        else:
            downhillMat = self.downhillMat

        DX0 = self._DX0
        DX1 = self._DX1
        dDX = self._dDX

        self.lvec.setArray(vector)
        self.dm.localToGlobal(self.lvec, DX0, addv=PETSc.InsertMode.INSERT_VALUES)

        DX1.setArray(DX0)
        # DX0.assemble()
        # DX1.assemble()

        niter = 0
        equal = False

        tolerance = 1e-8 * DX1.max()[1]

        while not equal and niter < maximum_its:
            dDX.setArray(DX1)
            #dDX.assemble()

            downhillMat.mult(DX1, self.gvec)
            DX1.setArray(self.gvec)
            DX1.assemble()

            DX0 += DX1

            dDX.axpy(-1.0, DX1)
            dDX.abs()
            max_dDX = dDX.max()[1]

            equal = max_dDX < tolerance

            if self.dm.comm.rank==0 and verbose and niter%10 == 0:
                print("{}: Max Delta - {} ".format(niter, max_dDX))

            niter += 1

        if self.dm.comm.Get_size() == 1:
            return niter, DX0.array.copy()
        else:
            self.dm.globalToLocal(DX0, self.lvec)
            return niter, self.lvec.array.copy()


    def downhill_smoothing_fn(self, lazyFn, its=1, centre_weight=0.75):

        import quagmire

        def new_fn(*args, **kwargs):
            local_array = lazyFn.evaluate(self)
            smoothed = self._downhill_smoothing(local_array, its=its, centre_weight=centre_weight)
            smoothed = self.sync(smoothed)

            if len(args) == 1 and args[0] == lazyFn._mesh:
                return smoothed
            elif len(args) == 1 and quagmire.mesh.check_object_is_a_q_mesh_and_raise(args[0]):

                mesh = args[0]
                return self.interpolate(mesh.coords[:,0], mesh.coords[:,1], zdata=smoothed, **kwargs)
            else:
                xi = np.atleast_1d(args[0])  # .resize(-1,1)
                yi = np.atleast_1d(args[1])  # .resize(-1,1)
                i, e = self.interpolate(xi, yi, zdata=smoothed, **kwargs)
                return i

        newLazyFn = _LazyEvaluation(mesh=lazyFn._mesh)
        newLazyFn.evaluate = new_fn
        newLazyFn.description = "DnHSmooth({}), i={}, w={}".format(lazyFn.description,  its, centre_weight)
        newLazyFn.dependency_list |= lazyFn.dependency_list


        return newLazyFn


    def _downhill_smoothing(self, data, its, centre_weight=0.75):

        downhillMat = self.downhillMat

        norm = self.gvec.duplicate()
        smooth_data = self.gvec.duplicate()

        self.gvec.set(1.0)
        self.downhillMat.mult(self.gvec, norm)

        mask = norm.array == 0.0

        self.lvec.setArray(data)
        self.dm.localToGlobal(self.lvec, smooth_data)
        for i in range(0, its):
            self.downhillMat.mult(smooth_data, self.gvec)
            smooth_data.setArray((1.0 - centre_weight) * self.gvec.array + \
                                  smooth_data.array*np.where(mask, 1.0, centre_weight))

        if self.dm.comm.Get_size() == 1:
            return smooth_data.array.copy()
        else:
            self.dm.globalToLocal(smooth_data, self.lvec)
            return self.lvec.array.copy()


    def uphill_smoothing_fn(self, lazyFn, its=1, centre_weight=0.75):

        import quagmire

        def new_fn(*args, **kwargs):
            local_array = lazyFn.evaluate(self)
            smoothed = self._uphill_smoothing(local_array, its=its, centre_weight=centre_weight)
            smoothed = self.sync(smoothed)

            if len(args) == 1 and args[0] == lazyFn._mesh:
                return smoothed
            elif len(args) == 1 and quagmire.mesh.check_object_is_a_q_mesh_and_raise(args[0]):
                mesh = args[0]
                return self.interpolate(mesh.coords[:,0], mesh.coords[:,1], zdata=smoothed, **kwargs)
            else:
                xi = np.atleast_1d(args[0])  # .resize(-1,1)
                yi = np.atleast_1d(args[1])  # .resize(-1,1)
                i, e = self.interpolate(xi, yi, zdata=smoothed, **kwargs)
                return i

        newLazyFn = LazyEvaluation(mesh=lazyFn._mesh)
        newLazyFn.evaluate = new_fn
        newLazyFn.description = "UpHSmooth({}), i={}, w={}".format(lazyFn.description, )
        newLazyFn.dependency_list |= lazyFn.dependency_list


        return newLazyFn


    def _uphill_smoothing(self, data, its, centre_weight=0.75):

        downhillMat = self.downhillMat


        norm2 = self.gvec.duplicate()
        smooth_data = self.gvec.duplicate()

        self.gvec.set(1.0)
        self.downhillMat.multTranspose(self.gvec, norm2)

        mask = norm2.array == 0.0
        norm2.array[~mask] = 1.0/norm2.array[~mask]

        self.lvec.setArray(data)
        self.dm.localToGlobal(self.lvec, smooth_data)
        for i in range(0, its):
            self.downhillMat.multTranspose(smooth_data, self.gvec)
            smooth_data.setArray((1.0 - centre_weight) * self.gvec.array * norm2 + \
                                  smooth_data.array*np.where(mask, 1.0, centre_weight))


        if self.dm.comm.Get_size() == 1:
            smooth_data *= data.mean()/smooth_data.array.mean()
            return smooth_data.copy()
        else:

            self.dm.globalToLocal(smooth_data, self.lvec)
            self.lvec *= data.mean()/self.lvec.array.mean()
            return self.lvec.array.copy()



    def streamwise_smoothing_fn(self, lazyFn, its=1, centre_weight=0.75):

        import quagmire

        def new_fn(*args, **kwargs):
            local_array = lazyFn.evaluate(self)
            smoothed = self._streamwise_smoothing(local_array, its=its, centre_weight=centre_weight)
            smoothed = self.sync(smoothed)

            if len(args) == 1 and args[0] == lazyFn._mesh:
                return smoothed
            elif len(args) == 1 and quagmire.mesh.check_object_is_a_q_mesh_and_raise(args[0]):
                mesh = args[0]
                return self.interpolate(mesh.coords[:,0], mesh.coords[:,1], zdata=smoothed, **kwargs)
            else:
                xi = np.atleast_1d(args[0])  # .resize(-1,1)
                yi = np.atleast_1d(args[1])  # .resize(-1,1)
                i, e = self.interpolate(xi, yi, zdata=smoothed, **kwargs)
                return i

        newLazyFn = _LazyEvaluation(mesh=lazyFn._mesh)
        newLazyFn.evaluate = new_fn
        newLazyFn.description = "StmSmooth({}), i={}, w={}".format(lazyFn.description, its, centre_weight)
        newLazyFn.dependency_list |= lazyFn.dependency_list


        return newLazyFn




    def _streamwise_smoothing(self, data, its, centre_weight=0.75):
        """
        A smoothing operator that is limited to the uphill / downhill nodes for each point. It's hard to build
        a conservative smoothing operator this way since "boundaries" occur at irregular internal points associated
        with watersheds etc. Upstream and downstream smoothing operations bracket the original data (over and under,
        respectively) and we use this to find a smooth field with the same mean value as the original data. This is
        done for each application of the smoothing.
        """

        smooth_data_d = self._downhill_smoothing(data, its, centre_weight=centre_weight)
        smooth_data_u = self._uphill_smoothing(data, its, centre_weight=centre_weight)

        return 0.5*(smooth_data_d + smooth_data_u)



    def _node_lowest_neighbour(self, node):
        """
        Find the lowest node in the neighbour list of the given node
        """

        lowest = self.neighbour_array_lo_hi[node][0]

        if lowest != node:
            return lowest
        else:
            return -1



    def _node_highest_neighbour(self, node):
        """
        Find the highest node in the neighbour list of the given node
        """

        highest = self.neighbour_array_lo_hi[node][-1]

        if highest != node:
            return highest
        else:
            return -1


    def _node_walk_downhill(self, node):
        """
        Walks downhill terminating when the downhill node is already claimed
        """

        chain = -np.ones(self.npoints, dtype=np.int32)

        idx = 0
        max_idx = self.npoints
        chain[idx] = node
        low_neighbour = self._node_lowest_neighbour(node)
        junction = -1

        while low_neighbour != -1:
            idx += 1
            chain[idx] = low_neighbour
            if self.node_chain_lookup[low_neighbour] != -1:
                junction = self.node_chain_lookup[low_neighbour]
                break

            low_neighbour = self._node_lowest_neighbour(low_neighbour)

        return junction, chain[0:idx+1]


    def build_node_chains(self):
        """ NEEDS WORK
        Builds all the chains for the mesh which flow from high to low and terminate
        when they meet with an existing chain.

        The following data structures become available once this function has been called:

            self.node_chain_lookup - tells you the chain in which a given node number lies
            self.node_chain_list   - is a list of the chains of nodes (each of which is an list)

        The terminating node of a chain may be the junction with another (pre-exisiting) chain
        and will be a member of that chain. Backbone chains which run from the highest level
        to the base level or the boundary are those whose terminal node is also a member of the same chain.

        Nodes which are at a base level given by self.base, are collected separately
        into chain number 0.
        """

        self.node_chain_lookup = -np.ones(self.npoints, dtype=np.int32)
        self.node_chain_list = []


        node_chain_idx = 1

        self.node_chain_list.append([]) # placeholder for any isolated base-level nodes

        for node1 in self.node_high_to_low:
            if self.node_chain_lookup[node1] != -1:
                continue

            junction, this_chain = self._node_walk_downhill(node1)

            if len(this_chain) > 1:
                self.node_chain_list.append(this_chain)

                self.node_chain_lookup[this_chain[0:-1]] = node_chain_idx
                if self.node_chain_lookup[this_chain[-1]] == -1:
                    self.node_chain_lookup[this_chain[-1]] = node_chain_idx

                node_chain_idx += 1

            else:
                self.node_chain_list[0].append(this_chain[0])
                self.node_chain_lookup[this_chain[0]] = 0


### Methods copied over from obseleted SurfMesh class

    def low_points_local_flood_fill(self, its=99999, scale=1.0, smoothing_steps=2):
        """
        Fill low points with a local flooding algorithm.
          - its is the number of uphill propagation steps
          - scale
        """

        t = perf_counter()
        if self.rank==0 and self.verbose:
            print("Low point local flood fill")

        my_low_points = self.identify_low_points()

        self.topography.unlock()
        h = self.topography.data

        fill_height =  (h[self.neighbour_cloud[my_low_points,1:7]].mean(axis=1)-h[my_low_points])

        new_h = self.uphill_propagation(my_low_points,  fill_height, scale=scale,  its=its, fill=0.0)
        new_h = self.sync(new_h)

        smoothed_new_height = self.rbf_smoother(new_h, iterations=smoothing_steps)
        self.topography.data = np.maximum(0.0, smoothed_new_height) + h
        self.topography.sync()
        self.topography.lock()
        self._update_height()

        if self.rank==0 and self.verbose:
            print("Low point local flood fill ",  perf_counter()-t, " seconds")

        self._update_height_for_surface_flows()

        return

    def low_points_local_patch_fill(self, its=1, smoothing_steps=1):

        from petsc4py import PETSc
        t = perf_counter()
        if self.rank==0 and self.verbose:
            print("Low point local patch fill")

        for iteration in range(0,its):
            low_points = self.identify_low_points()

            self.topography.unlock()

            h = self.topography.data
            delta_height = np.zeros_like(h)

            ## Note, the smoother has a communication barrier so needs to be called even if it has no work to do on this process

            if len(low_points) != 0:
                delta_height[low_points] =  (h[self.neighbour_cloud[low_points,1:5]].mean(axis=1) -
                                                         h[low_points])
            ## Note, the smoother has a communication barrier so needs to be called even
            ## if len(low_points==0) and there is no work to do on this process
            smoothed_height = h + self.rbf_smoother(delta_height, iterations=smoothing_steps)

            self.topography.data = np.maximum(smoothed_height, h)
            self.topography.sync()
            self.topography.lock()
            self._update_height()

        if self.rank==0 and self.verbose:
            print("Low point local patch fill ",  perf_counter()-t, " seconds")

        ## and now we need to rebuild the surface process information
        self._update_height_for_surface_flows()

        return


    def low_points_swamp_fill(self, its=1000, saddles=True, ref_height=0.0, ref_gradient=0.001):

        import petsc4py
        from petsc4py import PETSc
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        t0 = perf_counter()

        # I'm not sure the two ref_height values really refer to the same quantity 
        # and perhaps should be separated out 

        my_low_points = self.identify_low_points(ref_height=ref_height)
        my_glow_points = self.lgmap_row.apply(my_low_points.astype(PETSc.IntType))

        t = perf_counter()
        ctmt = self.uphill_propagation(my_low_points,  my_glow_points, its=its, fill=-999999).astype(np.int)

        if self.rank==0 and self.verbose:
            print("Build low point catchments - ", perf_counter() - t, " seconds")

        if saddles:  # Find saddle points on the catchment edge - catchment detection is via down-neighbour 1,
                     # so neighbour 2 (or 3) is the earliest detection opportunity for a spill

            cedges = np.where(ctmt[self.down_neighbour[2]] != ctmt )[0] ## local numbering
            try:
                cedges3 = np.where(ctmt[self.down_neighbour[3]] != ctmt )[0] ## local numbering
                cedges = np.unique(np.hstack((cedges,cedges3)))
            except:
                pass

        else:        # Find all edge points (this does not seem to work in most cases ... why not ?)
            ctmt2 = ctmt[self.neighbour_cloud] - ctmt.reshape(-1,1)
            ctmt3 = ctmt2 * self.near_neighbour_mask
            cedges = np.where(ctmt3.any(axis=1))[0]

        outer_edges = np.where(~self.bmask)[0]
        edges = np.unique(np.hstack((cedges,outer_edges)))

        height = self.topography.data.copy()

        ## In parallel this is all the low points where this process may have a spill-point
        my_catchments = np.unique(ctmt)

        spills = np.empty((edges.shape[0]),
                         dtype=np.dtype([('c', int), ('h', float), ('x', float), ('y', float)]))

        ii = 0
        for l, this_low in enumerate(my_catchments):
            this_low_spills = edges[np.where(ctmt[edges] == this_low)]  ## local numbering

            for spill in this_low_spills:
                spills['c'][ii] = this_low
                spills['h'][ii] = height[spill]
                spills['x'][ii] = self.coords[spill,0]
                spills['y'][ii] = self.coords[spill,1]
                ii += 1

        t = perf_counter()

        spills.sort(axis=0)  # Sorts by catchment then height ...
        s, indices = np.unique(spills['c'], return_index=True)
        spill_points = spills[indices]

        if self.rank == 0 and self.verbose:
            print(self.rank, " Sort spills - ", perf_counter() - t)

        # Gather lists to process 0, stack and remove duplicates

        t = perf_counter()
        list_of_spills = comm.gather(spill_points,   root=0)

        if self.rank == 0 and self.verbose:
            print(self.rank, " Gather spill data - ", perf_counter() - t)

        if self.rank == 0:
            t = perf_counter()

            all_spills = np.hstack(list_of_spills)
            all_spills.sort(axis=0) # Sorts by catchment then height ...
            s, indices = np.unique(all_spills['c'], return_index=True)
            all_spill_points = all_spills[indices]

            if self.verbose:
                print(rank, " Sort all spills - ", perf_counter() - t)

        else:
            all_spill_points = None
            pass

        # Broadcast lists to everyone

        global_spill_points = comm.bcast(all_spill_points, root=0)

        height2 = np.zeros_like(height) + ref_height

        for i, spill in enumerate(global_spill_points):
            this_catchment = int(spill['c'])

            ## -ve values indicate that the point is connected
            ## to the outflow of the mesh and needs no modification
            if this_catchment < 0:
                continue

            catchment_nodes = np.where(ctmt == this_catchment)
            separation_x = (self.coords[catchment_nodes,0] - spill['x'])
            separation_y = (self.coords[catchment_nodes,1] - spill['y'])
            distance = np.hypot(separation_x, separation_y)

            ## Todo: this gradient needs to be relative to typical ones nearby and resolvable in a geotiff !
            height2[catchment_nodes] = spill['h'] + ref_gradient * distance  # A 'small' gradient (should be a user-parameter)

        height2 = self.sync(height2)

        new_height = np.maximum(height, height2)
        new_height = self.sync(new_height)


        # We only need to update the height not all
        # surface process information that is associated with it.
        self.topography.unlock()
        self.topography.data = new_height
        self._update_height()
        self.topography.lock()


        if self.rank==0 and self.verbose:
            print("Low point swamp fill ",  perf_counter()-t0, " seconds")

        ## but now we need to rebuild the surface process information
        self._update_height_for_surface_flows()
        return


    def backfill_points(self, fill_points, heights, its):
        """
        Handles *selected* low points by backfilling height array.
        This can be used to block a stream path, for example, or to locate lakes
        """

        if len(fill_points) == 0:
            return self.heightVariable.data

        new_height = self.lvec.duplicate()
        new_height.setArray(heights)
        height = np.maximum(self.height, new_height.array)

        # Now march the new height to all the uphill nodes of these nodes
        # height = np.maximum(self.height, delta_height.array)

        self.dm.localToGlobal(new_height, self.gvec)
        global_dH = self.gvec.copy()

        for p in range(0, its):
            self.adjacency[1].multTranspose(global_dH, self.gvec)
            global_dH.setArray(self.gvec)
            global_dH.scale(1.001)  # Maybe !
            self.dm.globalToLocal(global_dH, new_height)

            height = np.maximum(height, new_height.array)

        return height

    def uphill_propagation(self, points, values, scale=1.0, its=1000, fill=-1):

        t0 = perf_counter()

        local_ID = self.lvec.copy()
        global_ID = self.gvec.copy()

        local_ID.set(fill+1)
        global_ID.set(fill+1)

        identifier = np.empty_like(self.topography.data)
        identifier.fill(fill+1)

        if len(points):
            identifier[points] = values + 1

        local_ID.setArray(identifier)
        self.dm.localToGlobal(local_ID, global_ID)

        delta = global_ID.copy()
        delta.abs()
        rtolerance = delta.max()[1] * 1.0e-10

        for p in range(0, its):

            # self.adjacency[1].multTranspose(global_ID, self.gvec)
            gvec = self.uphill[1] * global_ID
            delta = global_ID - gvec
            delta.abs()
            max_delta = delta.max()[1]

            if max_delta < rtolerance:
                break

            self.gvec.scale(scale)

            if self.dm.comm.Get_size() == 1:
                local_ID.array[:] = gvec.array[:]
            else:
                self.dm.globalToLocal(gvec, local_ID)

            global_ID.array[:] = gvec.array[:]

            identifier = np.maximum(identifier, local_ID.array)
            identifier = self.sync(identifier)

        # Note, the -1 is used to identify out of bounds values

        if self.rank == 0 and self.verbose:
            print(p, " iterations, time = ", perf_counter() - t0)

        return identifier - 1



    def identify_low_points(self, include_shadows=False), ref_height=0:
        """
        Identify if the mesh has (internal) local minima and return an array of node indices
        """

        # from petsc4py import PETSc

        nodes = np.arange(0, self.npoints, dtype=np.int)
        gnodes = self.lgmap_row.apply(nodes.astype(PETSc.IntType))

        low_nodes = self.down_neighbour[1]
        mask = np.logical_and(nodes == low_nodes, self.bmask == True)
        mask = np.logical_and(mask, self.topography.data > ref_height)

        if not include_shadows:
            mask = np.logical_and(mask, gnodes >= 0)

        return nodes[mask]

    def identify_global_low_points(self, global_array=False):
        """
        Identify if the mesh as a whole has (internal) local minima and return an array of local lows in global
        index format.

        If global_array is True, then lows for the whole mesh are returned
        """

        import petsc4py
        from petsc4py import PETSc
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        # from petsc4py import PETSc

        nodes = np.arange(0, self.npoints, dtype=np.int)
        gnodes = self.lgmap_row.apply(nodes.astype(PETSc.IntType))

        low_nodes = self.down_neighbour[1]
        mask = np.logical_and(nodes == low_nodes, self.bmask == True)
        mask = np.logical_and(mask, gnodes >= 0)

        number_of_lows = np.count_nonzero(mask)
        low_gnodes = self.lgmap_row.apply(low_nodes.astype(PETSc.IntType))

        # gather/scatter numbers

        list_of_nlows  = comm.gather(number_of_lows,   root=0)
        if self.rank == 0:
            all_low_counts = np.hstack(list_of_nlows)
            no_global_lows0 = all_low_counts.sum()

        else:
            no_global_lows0 = None

        no_global_lows = comm.bcast(no_global_lows0, root=0)


        if global_array:

            list_of_lglows = comm.gather(low_gnodes,   root=0)

            if self.rank == 0:
                all_glows = np.hstack(list_of_lglows)
                global_lows0 = np.unique(all_glows)

            else:
                global_lows0 = None

            low_gnodes = comm.bcast(global_lows0, root=0)

        return no_global_lows, low_gnodes


    def identify_outflow_points(self):
        """
        Identify the (boundary) outflow points and return an array of (local) node indices
        """

        # nodes = np.arange(0, self.npoints, dtype=np.int)
        # low_nodes = self.down_neighbour[1]
        # mask = np.logical_and(nodes == low_nodes, self.bmask == False)
        #

        i = self.downhill_neighbours

        o = (np.logical_and(self.down_neighbour[i] == np.indices(self.down_neighbour[i].shape), self.bmask == False)).ravel()
        outflow_nodes = o.nonzero()[0]

        return outflow_nodes


    def identify_flat_spots(self):

        slope = self.slope.evaluate(self.slope._mesh)
        smooth_grad1 = self.local_area_smoothing(slope, its=1, centre_weight=0.5)

        # flat_spot_field = np.where(smooth_grad1 < smooth_grad1.max()/100, 0.0, 1.0)

        flat_spots = np.where(smooth_grad1 < smooth_grad1.max()/1000.0, True, False)

        return flat_spots