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
import numpy.ma as ma
from mpi4py import MPI
import sys,petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
# comm = MPI.COMM_WORLD
from time import clock

try: range = xrange
except: pass

from quagmire import function as fn
from quagmire.mesh import MeshVariable as _MeshVariable
from quagmire.function import LazyEvaluation as _LazyEvaluation

class TopoMesh(object):
    def __init__(self, downhill_neighbours=2, *args, **kwargs):


        # Initialise cumulative flow vectors
        self.DX0 = self.gvec.duplicate()
        self.DX1 = self.gvec.duplicate()
        self.dDX = self.gvec.duplicate()

        # Initialise mesh fields
        # self.height = self.gvec.duplicate()
        # self.slope = self.gvec.duplicate()

        # create a variable for the height (data) and fn_height
        # a context manager to ensure the matrices are updated when h(x,y) changes

        self.topography = self.add_variable(name="h(x,y)", locked=True)
        self.deform_topography = self._height_update_context_manager_generator()

        # There is no topography yet set, so we need to avoid
        # triggering the matrix rebuilding in the setter of this property
        self._downhill_neighbours = downhill_neighbours

        # Slope (function)
        self.slope = fn.math.sqrt(self.topography.fn_gradient(0)**2.0+self.topography.fn_gradient(1)**2.0)

        self._heightVariable = self.topography



    @property
    def downhill_neighbours(self):
        return self._downhill_neighbours

    @downhill_neighbours.setter
    def downhill_neighbours(self, value):
        self._downhill_neighbours = int(value)
        # if the topography is unlocked, don't rebuild anything
        # as it might break something

        if self.topography._locked == True:
            self._build_downhill_matrix_iterate()

        return


    def _update_height(self):
        """
        Update height field
        """

        t = clock()
        self._build_downhill_matrix_iterate()
        self.timings['downhill matrices'] = [clock()-t, self.log.getCPUTime(), self.log.getFlops()]

        if self.verbose:
            print(("{} - Build downhill matrices {}s".format(self.dm.comm.rank, clock()-t)))
        return

    def _height_update_context_manager_generator(self):
        """Builds a context manager on the current mesh object to control when matrices are to be updated"""

        topomesh = self
        topographyVariable = self.topography

        class Topomesh_Height_Update_Manager(object):
            """Manage when changes to the height information trigger a rebuild
            of the topomesh matrices and other internal data.
            """

            def __init__(inner_self):
                inner_self._topomesh = topomesh
                inner_self._topovar  = topographyVariable
                return

            def __enter__(inner_self):
                # unlock
                inner_self._topovar.unlock()
                return

            def __exit__(inner_self, *args):
                inner_self._topomesh._update_height()
                inner_self._topovar.lock()
                return

        return Topomesh_Height_Update_Manager


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

        DX1 = self.gvec.duplicate()
        DX0 = self.gvec.duplicate()
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
            elif len(args) == 1 and isinstance(args[0], (quagmire.mesh.trimesh.TriMesh, quagmire.mesh.pixmesh.PixMesh) ):
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

        return newLazyFn



    def cumulative_flow(self, vector, *args, **kwargs):

        niter, cumulative_flow_vector = self._cumulative_flow_verbose(vector, **kwargs)
        return cumulative_flow_vector


    def _cumulative_flow_verbose(self, vector, verbose=False, maximum_its=None, uphill=False):

        if not maximum_its:
            maximum_its = 1000000000000


        # downhillMat2 = self.downhillMat * self.downhillMat
        # downhillMat4 = downhillMat2 * downhillMat2
        # downhillMat8 = downhillMat4 * downhillMat4
        if uphill:
            downhillMat = self.downhillMat.copy()
            downhillMat = downhillMat.transpose()
        else:
            downhillMat = self.downhillMat

        DX0 = self.DX0
        DX1 = self.DX1
        dDX = self.dDX

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
            elif len(args) == 1 and isinstance(args[0], (quagmire.mesh.trimesh.TriMesh, quagmire.mesh.pixmesh.PixMesh) ):
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
            elif len(args) == 1 and isinstance(args[0], (quagmire.mesh.trimesh.TriMesh, quagmire.mesh.pixmesh.PixMesh) ):
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
            elif len(args) == 1 and isinstance(args[0], (quagmire.mesh.trimesh.TriMesh, quagmire.mesh.pixmesh.PixMesh) ):
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
